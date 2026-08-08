"""
Transcription worker — runs in the bundled Python runtime.
Communicates with the GUI via JSON lines on stdout.

Two engines:
  faster-whisper (ctranslate2)  - CPU, and NVIDIA CUDA when the optional
                                  GPU libraries are installed
  whisper.cpp (Vulkan)          - any modern GPU: AMD, Intel, NVIDIA;
                                  binaries ship in runtime/whispercpp

Audio/video decoding is PyAV in both cases — no system FFmpeg needed.

Usage:
    python transcribe_worker.py --input FILE --model MODEL --backend BACKEND
        [--output_dir DIR] [--timestamps] [--export_srt]
"""
import sys
import os
import re
import json
import time
import wave
import argparse
import tempfile
import itertools
import subprocess
import threading
import urllib.request

# Keep models inside AudioWhisper's data folder (GUI sets this; default for CLI use)
os.environ.setdefault("HF_HOME", os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "AudioWhisper", "models"))
# Force the classic HTTP download path so the cache directory grows
# linearly — that's how we measure model download progress
os.environ["HF_HUB_DISABLE_XET"] = "1"

# JSON messages must survive non-ASCII transcripts regardless of console codepage
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_emit_lock = threading.Lock()


def emit(msg_type, **kwargs):
    """Send a JSON message to the parent process.

    Single locked write — download watcher threads emit concurrently
    with the main thread.
    """
    kwargs["type"] = msg_type
    line = json.dumps(kwargs, ensure_ascii=False) + "\n"
    with _emit_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _download_progress(label, done, total):
    pct = min(done / total, 0.99) if total else 0
    mb_done = done / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    if mb_total >= 1024:
        emit("model_download", value=pct,
             msg=f"Downloading {label}: {mb_done / 1024:.1f} / {mb_total / 1024:.1f} GB")
    else:
        emit("model_download", value=pct,
             msg=f"Downloading {label}: {mb_done:.0f} / {mb_total:.0f} MB")


# ══════════════════════════════════════════════════════════
#  faster-whisper engine (CPU / NVIDIA CUDA)
# ══════════════════════════════════════════════════════════

MODEL_REPOS = {
    "tiny":     "Systran/faster-whisper-tiny",
    "base":     "Systran/faster-whisper-base",
    "small":    "Systran/faster-whisper-small",
    "medium":   "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Approximate repo download sizes in MB (dominated by model.bin) —
# used to compute progress while the cache directory fills up
MODEL_SIZES_MB = {
    "tiny": 76,
    "base": 145,
    "small": 484,
    "medium": 1528,
    "large-v3": 3087,
}


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _ensure_fw_model(model_name):
    """Download the faster-whisper model with progress, return cache path."""
    from huggingface_hub import snapshot_download, scan_cache_dir
    import huggingface_hub

    repo_id = MODEL_REPOS.get(model_name)
    if not repo_id:
        return model_name

    cache_dir = huggingface_hub.constants.HF_HUB_CACHE

    # Check if already cached — and actually complete. An interrupted
    # download leaves the small files (config.json etc.) in place, so
    # require model.bin before trusting the cache; otherwise fall through
    # to snapshot_download, which resumes partial downloads.
    try:
        cache_info = scan_cache_dir(cache_dir)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.size_on_disk > 0:
                for revision in repo.revisions:
                    snap = str(revision.snapshot_path)
                    if os.path.isfile(os.path.join(snap, "model.bin")):
                        return snap
    except Exception:
        pass

    emit("status", msg=f"Downloading {model_name} model...")

    # Report progress by watching the repo's cache folder grow. This works
    # regardless of how huggingface_hub downloads internally.
    expected = MODEL_SIZES_MB.get(model_name, 0) * 1024 * 1024
    repo_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    stop_watch = threading.Event()

    def _watch():
        last = -1
        while not stop_watch.wait(0.5):
            done = _dir_size(repo_dir)
            if done != last:
                last = done
                _download_progress(f"{model_name} model", done, expected)

    watcher = None
    if expected > 0:
        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    def _stop_watcher():
        stop_watch.set()
        if watcher:
            watcher.join(timeout=2)

    try:
        path = snapshot_download(repo_id)
        _stop_watcher()
        emit("model_download", value=1.0, msg=f"Model {model_name} ready")
        emit("status", msg=f"Model {model_name} ready")
        return path
    except Exception as e:
        emit("status", msg=f"Model download issue, trying fallback: {e}")
        return model_name
    finally:
        _stop_watcher()


def _check_vram(model_name, device):
    """Warn if the NVIDIA GPU likely lacks VRAM for the model."""
    if device != "cuda":
        return
    MODEL_VRAM_MB = {"tiny": 500, "base": 1000, "small": 2000,
                     "medium": 5000, "large-v3": 10000}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            free_mb = int(result.stdout.strip().split("\n")[0])
            required = MODEL_VRAM_MB.get(model_name, 0)
            if required > 0 and free_mb < required:
                emit("status",
                     msg=f"Warning: {model_name} needs ~{required // 1000}GB VRAM, "
                         f"but only {free_mb / 1000:.1f}GB free. Will use CPU if needed.")
    except Exception:
        pass


def run_faster_whisper(args, input_file, device):
    """Returns (duration_seconds, segment_iterator). Segments are (start, end, text)."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    _check_vram(args.model, device)
    model_path = _ensure_fw_model(args.model)

    BEAM_SIZES = {"tiny": 1, "base": 3, "small": 3, "medium": 5, "large-v3": 5}
    beam_size = args.beam_size if args.beam_size > 0 else BEAM_SIZES.get(args.model, 5)

    emit("status", msg="Loading model...")

    def _start(dev, compute):
        model = WhisperModel(model_path, device=dev, compute_type=compute)
        emit("status", msg="Transcribing...")
        segments, info = model.transcribe(input_file, beam_size=beam_size)
        seg_iter = iter(segments)
        # Pull the first segment now so CUDA failures surface here,
        # where we can still fall back to CPU
        first = next(seg_iter, None)
        return first, seg_iter, info

    # Fall back to CPU outside the except block so the failed CUDA
    # model (pinned by the in-flight exception's traceback) is freed
    # before the CPU model loads
    fallback = False
    try:
        first, seg_iter, info = _start(device, compute_type)
    except Exception:
        if device != "cuda":
            raise
        fallback = True
    if fallback:
        emit("status", msg="CUDA unavailable — switching to CPU...")
        first, seg_iter, info = _start("cpu", "int8")

    duration = getattr(info, "duration", 0) or 0
    chained = itertools.chain([first], seg_iter) if first is not None else seg_iter
    return duration, ((s.start, s.end, s.text.strip()) for s in chained)


# ══════════════════════════════════════════════════════════
#  whisper.cpp Vulkan engine (AMD / Intel / NVIDIA GPUs)
# ══════════════════════════════════════════════════════════

GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
# Exact byte sizes (huggingface.co/ggerganov/whisper.cpp) so a completed
# download is recognized and an interrupted one resumes
GGML_SIZES = {
    "tiny": 77691713,
    "base": 147951465,
    "small": 487601967,
    "medium": 1533763059,
    "large-v3": 3095033483,
}


def _whispercpp_dir(args):
    if args.whispercpp_dir:
        return args.whispercpp_dir
    return os.path.join(os.path.dirname(sys.executable), "whispercpp")


def _ensure_ggml_model(model_name):
    """Download the ggml model (single file) with progress and resume."""
    ggml_dir = os.path.join(os.environ["HF_HOME"], "ggml")
    os.makedirs(ggml_dir, exist_ok=True)
    dest = os.path.join(ggml_dir, f"ggml-{model_name}.bin")
    expected = GGML_SIZES.get(model_name, 0)

    if os.path.exists(dest) and expected and os.path.getsize(dest) >= expected:
        return dest

    url = GGML_URL.format(name=model_name)
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0

    emit("status", msg=f"Downloading {model_name} model...")
    headers = {"User-Agent": "AudioWhisper"}
    if done:
        headers["Range"] = f"bytes={done}-"

    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)
    if done and resp.status != 206:
        done = 0  # server ignored the range; start over
    total = done + int(resp.headers.get("Content-Length", 0)) or expected

    mode = "ab" if done else "wb"
    last_pct = -1
    with open(part, mode) as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(done * 100 / total) if total else 0
            if pct != last_pct:
                last_pct = pct
                _download_progress(f"{model_name} model", done, total)

    os.replace(part, dest)
    emit("model_download", value=1.0, msg=f"Model {model_name} ready")
    emit("status", msg=f"Model {model_name} ready")
    return dest


def _decode_to_wav16k(input_file):
    """Decode any audio/video to 16 kHz mono WAV via PyAV.
    Returns (wav_path, duration_seconds)."""
    import av
    from av.audio.resampler import AudioResampler

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="audiowhisper_")
    os.close(fd)

    frames_written = 0
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(input_file) as ic, wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        stream = next((s for s in ic.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("No audio track found in this file.")
        for frame in ic.decode(stream):
            for rf in resampler.resample(frame):
                data = rf.to_ndarray().tobytes()
                w.writeframes(data)
                frames_written += rf.samples
        for rf in resampler.resample(None):
            data = rf.to_ndarray().tobytes()
            w.writeframes(data)
            frames_written += rf.samples

    return wav_path, frames_written / 16000.0


# whisper-cli segment lines: [00:00:00.000 --> 00:00:02.340]   text
_CLI_SEGMENT_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s?(.*)$")


def _cli_ts(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def run_vulkan(args, input_file):
    """Returns (duration_seconds, segment_iterator) using whisper.cpp Vulkan."""
    cli = os.path.join(_whispercpp_dir(args), "whisper-cli.exe")
    if not os.path.isfile(cli):
        raise RuntimeError("whisper.cpp engine not found")

    model_path = _ensure_ggml_model(args.model)

    emit("status", msg="Preparing audio...")
    wav_path, duration = _decode_to_wav16k(input_file)

    emit("status", msg="Loading model...")
    threads = min(8, os.cpu_count() or 4)
    proc = subprocess.Popen(
        [cli, "-m", model_path, "-f", wav_path, "-l", "auto", "-t", str(threads)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # Drain stderr so a chatty engine can't deadlock the pipe
    stderr_tail = []

    def _drain():
        try:
            for raw in proc.stderr:
                stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
                del stderr_tail[:-20]
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    def _segments():
        emitted = False
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                m = _CLI_SEGMENT_RE.match(line)
                if not m:
                    continue
                g = m.groups()
                text = g[8].strip()
                if not text:
                    continue
                if not emitted:
                    emitted = True
                    emit("status", msg="Transcribing...")
                yield _cli_ts(*g[0:4]), _cli_ts(*g[4:8]), text
            proc.wait()
            if proc.returncode != 0 and not emitted:
                tail = " | ".join(l for l in stderr_tail[-3:] if l.strip())
                raise RuntimeError(f"whisper.cpp failed: {tail or proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                os.remove(wav_path)
            except OSError:
                pass

    return duration, _segments()


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--backend", default="",
                        choices=["", "cuda", "vulkan", "cpu"])
    parser.add_argument("--device", default="cpu")        # legacy
    parser.add_argument("--compute_type", default="int8")  # legacy
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--export_srt", action="store_true")
    parser.add_argument("--beam_size", type=int, default=0)
    parser.add_argument("--whispercpp_dir", default="")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir or os.path.dirname(input_file)
    backend = args.backend or ("cuda" if args.device == "cuda" else "cpu")

    try:
        emit("status", msg="Preparing model...")

        duration, segments = None, None
        if backend == "vulkan":
            try:
                duration, segments = run_vulkan(args, input_file)
            except Exception as e:
                emit("status", msg=f"GPU engine unavailable — using CPU... ({e})")
                backend = "cpu"

        if segments is None:
            duration, segments = run_faster_whisper(
                args, input_file, "cuda" if backend == "cuda" else "cpu")

        collected = []
        start_time = time.time()

        for seg_start, seg_end, text in segments:
            if duration and duration > 0:
                prog = min(seg_end / duration, 1.0)
                elapsed = time.time() - start_time
                eta = ""
                if prog > 0.01:
                    remaining = (elapsed / prog) - elapsed
                    mins, secs = divmod(int(remaining), 60)
                    eta = f"~{mins:02}:{secs:02} remaining"
                emit("progress", value=prog, eta=eta)

            ts = f"[{int(seg_start // 3600):02}:{int((seg_start % 3600) // 60):02}:{int(seg_start % 60):02}]"
            emit("segment", start=seg_start, end=seg_end, text=text, timestamp=ts)
            collected.append({"start": seg_start, "end": seg_end, "text": text})

        # Save files
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in collected:
                if args.timestamps:
                    f.write(f"[{seg['start']:.2f}s] {seg['text']}\n")
                else:
                    f.write(f"{seg['text']} ")

        if args.export_srt:
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(collected, start=1):
                    s = _format_srt_time(seg["start"])
                    e = _format_srt_time(seg["end"])
                    f.write(f"{i}\n{s} --> {e}\n{seg['text']}\n\n")

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        emit("done", msg=f"Done — saved to {output_dir}  ({mins}m {secs}s)", output_dir=output_dir)

    except Exception as e:
        emit("error", msg=str(e))


def _format_srt_time(seconds):
    total_sec = int(seconds)
    millis = int((seconds - total_sec) * 1000)
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


if __name__ == "__main__":
    main()
