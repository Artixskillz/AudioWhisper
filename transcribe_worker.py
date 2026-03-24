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
import datetime


def emit(msg_type, **kwargs):
    """Send a JSON message to the parent process."""
    kwargs["type"] = msg_type
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="float32")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--export_srt", action="store_true")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir or os.path.dirname(input_file)

    try:
        emit("status", msg="Loading model (may download on first use)...")

        from faster_whisper import WhisperModel
        import librosa

        # Get audio duration
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

        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

        emit("status", msg="Transcribing...")
        segments, info = model.transcribe(process_file, beam_size=5)

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
