"""
Transcription worker — runs in the bundled Python runtime.
Communicates with the GUI via JSON lines on stdout.

Audio/video decoding is handled by faster-whisper's bundled PyAV,
which ships its own FFmpeg libraries — no system FFmpeg needed.

Usage:
    python transcribe_worker.py --input FILE --model MODEL --device DEVICE
        --compute_type TYPE [--output_dir DIR] [--timestamps] [--export_srt]
"""
import sys
import os
import json
import time
import argparse
import itertools

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


def emit(msg_type, **kwargs):
    """Send a JSON message to the parent process.

    Single atomic write — the download watcher thread emits concurrently
    with the main thread, and print() would interleave its two writes.
    """
    kwargs["type"] = msg_type
    sys.stdout.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ── Model Download with Progress ───────────────────────

# Map faster-whisper model names to HuggingFace repo IDs
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


def _ensure_model_downloaded(model_name):
    """Download the model with progress reporting, return the cache path."""
    import threading
    from huggingface_hub import snapshot_download, scan_cache_dir
    import huggingface_hub

    repo_id = MODEL_REPOS.get(model_name)
    if not repo_id:
        # Unknown model — let faster-whisper handle it
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
        last_pct = -1.0
        while not stop_watch.wait(0.5):
            done = _dir_size(repo_dir)
            pct = min(done / expected, 0.99)
            if pct - last_pct >= 0.01:
                last_pct = pct
                mb_done = done / (1024 * 1024)
                mb_total = expected / (1024 * 1024)
                if mb_total >= 1024:
                    emit("model_download", value=pct,
                         msg=f"Downloading {model_name} model: {mb_done / 1024:.1f} / {mb_total / 1024:.1f} GB")
                else:
                    emit("model_download", value=pct,
                         msg=f"Downloading {model_name} model: {mb_done:.0f} / {mb_total:.0f} MB")

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


# ── VRAM Detection ─────────────────────────────────────

# Approximate VRAM requirements per model in MB
MODEL_VRAM_MB = {
    "tiny": 500,
    "base": 1000,
    "small": 2000,
    "medium": 5000,
    "large-v3": 10000,
}


def _check_vram(model_name, device):
    """Check if GPU has enough VRAM for the model. Emits a warning if not."""
    if device != "cuda":
        return
    try:
        import subprocess
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


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--export_srt", action="store_true")
    parser.add_argument("--beam_size", type=int, default=0)
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir or os.path.dirname(input_file)

    try:
        emit("status", msg="Preparing model...")

        from faster_whisper import WhisperModel

        # Check VRAM before downloading/loading
        _check_vram(args.model, args.device)

        # Download model with progress (if needed)
        model_path = _ensure_model_downloaded(args.model)

        # Adaptive beam size: smaller models use fewer beams for speed
        BEAM_SIZES = {"tiny": 1, "base": 3, "small": 3, "medium": 5, "large-v3": 5}
        beam_size = args.beam_size if args.beam_size > 0 else BEAM_SIZES.get(args.model, 5)

        emit("status", msg="Loading model...")

        def _start(device, compute_type):
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
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
            first, seg_iter, info = _start(args.device, args.compute_type)
        except Exception:
            if args.device != "cuda":
                raise
            fallback = True
        if fallback:
            emit("status", msg="GPU unavailable — switching to CPU...")
            first, seg_iter, info = _start("cpu", "int8")

        # PyAV decodes the full audio up front, so info.duration covers
        # both audio and video inputs — no FFmpeg or librosa needed
        total_duration = getattr(info, "duration", 0) or 0

        collected_segments = []
        start_time = time.time()

        segments_all = itertools.chain([first], seg_iter) if first is not None else seg_iter
        for segment in segments_all:
            # Progress & ETA
            if total_duration > 0:
                prog = min(segment.end / total_duration, 1.0)
                elapsed = time.time() - start_time
                eta = ""
                if prog > 0.01:
                    remaining = (elapsed / prog) - elapsed
                    mins, secs = divmod(int(remaining), 60)
                    eta = f"~{mins:02}:{secs:02} remaining"
                emit("progress", value=prog, eta=eta)

            ts = f"[{int(segment.start // 3600):02}:{int((segment.start % 3600) // 60):02}:{int(segment.start % 60):02}]"
            emit("segment", start=segment.start, end=segment.end, text=segment.text.strip(), timestamp=ts)
            collected_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

        # Save files
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in collected_segments:
                if args.timestamps:
                    f.write(f"[{seg['start']:.2f}s] {seg['text']}\n")
                else:
                    f.write(f"{seg['text']} ")

        if args.export_srt:
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(collected_segments, start=1):
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
