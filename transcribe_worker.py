"""
Transcription worker — runs in the embedded Python environment.
Communicates with the GUI via JSON lines on stdout.

Usage:
    python transcribe_worker.py --input FILE --model MODEL --device DEVICE
        --compute_type TYPE [--output_dir DIR] [--timestamps] [--export_srt]
"""
import sys
import os
import json
import time
import argparse


def emit(msg_type, **kwargs):
    """Send a JSON message to the parent process."""
    kwargs["type"] = msg_type
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


# ── Model Download with Progress ───────────────────────

# Map faster-whisper model names to HuggingFace repo IDs
MODEL_REPOS = {
    "tiny":     "Systran/faster-whisper-tiny",
    "base":     "Systran/faster-whisper-base",
    "small":    "Systran/faster-whisper-small",
    "medium":   "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def _ensure_model_downloaded(model_name):
    """Download the model with progress reporting, return the cache path."""
    from huggingface_hub import snapshot_download, scan_cache_dir
    import huggingface_hub

    repo_id = MODEL_REPOS.get(model_name)
    if not repo_id:
        # Unknown model — let faster-whisper handle it
        return model_name

    # Check if already cached
    try:
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
        cache_info = scan_cache_dir(cache_dir)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.size_on_disk > 0:
                # Already downloaded — find the snapshot path
                for revision in repo.revisions:
                    return str(revision.snapshot_path)
    except Exception:
        pass

    # Download with progress
    emit("status", msg=f"Downloading {model_name} model...")

    last_report = [0.0]

    def _progress_callback(current, total):
        if total and total > 0:
            pct = current / total
            # Only report every 2% to avoid flooding
            if pct - last_report[0] >= 0.02 or pct >= 1.0:
                last_report[0] = pct
                mb_done = current / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                if mb_total >= 1024:
                    emit("model_download", value=pct,
                         msg=f"Downloading {model_name} model: {mb_done / 1024:.1f} / {mb_total / 1024:.1f} GB")
                else:
                    emit("model_download", value=pct,
                         msg=f"Downloading {model_name} model: {mb_done:.0f} / {mb_total:.0f} MB")

    # Try using tqdm callback for progress
    try:
        from huggingface_hub.utils import tqdm as hf_tqdm
        import tqdm as tqdm_module

        original_init = tqdm_module.tqdm.__init__
        original_update = tqdm_module.tqdm.update
        _bars = {}

        class ProgressTracker:
            def __init__(self):
                self.total = 0
                self.current = 0

        tracker = ProgressTracker()

        def patched_init(self_tqdm, *args, **kwargs):
            original_init(self_tqdm, *args, **kwargs)
            if hasattr(self_tqdm, 'total') and self_tqdm.total and self_tqdm.total > 1024 * 1024:
                tracker.total = self_tqdm.total
                tracker.current = 0

        def patched_update(self_tqdm, n=1):
            original_update(self_tqdm, n)
            if tracker.total > 0:
                tracker.current += n
                _progress_callback(tracker.current, tracker.total)

        tqdm_module.tqdm.__init__ = patched_init
        tqdm_module.tqdm.update = patched_update
    except Exception:
        pass

    try:
        path = snapshot_download(repo_id)
        emit("status", msg=f"Model {model_name} ready")
        return path
    except Exception as e:
        emit("status", msg=f"Model download issue, trying fallback: {e}")
        return model_name
    finally:
        # Restore tqdm
        try:
            tqdm_module.tqdm.__init__ = original_init
            tqdm_module.tqdm.update = original_update
        except Exception:
            pass


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
                         f"but only {free_mb // 1000:.1f}GB free. May fall back to CPU.")
    except Exception:
        pass


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="float32")
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
        import librosa

        # Check VRAM before downloading/loading
        _check_vram(args.model, args.device)

        # Download model with progress (if needed)
        model_path = _ensure_model_downloaded(args.model)

        emit("status", msg="Loading model...")

        # Prepare audio
        total_duration = 0
        process_file = input_file
        temp_file = None

        is_video = input_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        if is_video:
            emit("status", msg="Extracting audio from video...")
            import ffmpeg
            temp_file = os.path.splitext(input_file)[0] + "_temp.wav"
            (
                ffmpeg.input(input_file)
                .output(temp_file, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True)
            )
            process_file = temp_file

        try:
            total_duration = librosa.get_duration(path=process_file)
        except Exception:
            total_duration = 0

        model = WhisperModel(model_path, device=args.device, compute_type=args.compute_type)

        # Adaptive beam size: smaller models use fewer beams for speed
        BEAM_SIZES = {"tiny": 1, "base": 3, "small": 3, "medium": 5, "large-v3": 5}
        beam_size = args.beam_size if args.beam_size > 0 else BEAM_SIZES.get(args.model, 5)

        emit("status", msg="Transcribing...")
        segments, info = model.transcribe(process_file, beam_size=beam_size)

        collected_segments = []
        start_time = time.time()

        for segment in segments:
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
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def _format_srt_time(seconds):
    total_sec = int(seconds)
    millis = int((seconds - total_sec) * 1000)
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


if __name__ == "__main__":
    main()
