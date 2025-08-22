import sys
import subprocess
import importlib
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext

# auto-install required packages
REQUIRED_PACKAGES = ["openai-whisper", "ffmpeg-python", "librosa", "soundfile"]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in REQUIRED_PACKAGES:
    try:
        importlib.import_module(package.split("-")[0])  # crude check
    except ImportError:
        print(f"Installing missing package: {package}")
        install(package)

import whisper
import ffmpeg
import librosa
import soundfile as sf
import math
import tempfile


def log_message(msg):
    """Append log message to GUI console and update status."""
    status_var.set(msg)
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)  # auto-scroll


def extract_audio(video_path, output_path):
    """Extract audio track from video using ffmpeg-python."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        raise RuntimeError(f"FFmpeg failed: {e}")


def get_unique_filename(base_path):
    """Ensure output file does not overwrite existing ones by adding (2), (3), etc."""
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    counter = 2
    new_path = f"{base} ({counter}){ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base} ({counter}){ext}"
    return new_path


def execute_whisper(input_path, output_dir, model_name, show_timestamps, language):
    """Run Whisper transcription in 5s chunks and stream results into the GUI log."""
    log_message(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    temp_audio_path = None
    start_time = time.time()
    try:
        # If input is a video, extract audio
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            log_message("Extracting audio from video...")
            temp_audio_path = os.path.splitext(input_path)[0] + "_temp.wav"
            extract_audio(input_path, temp_audio_path)
            input_path = temp_audio_path
            log_message("Audio extracted. Starting transcription...")

        # Load audio
        audio, sr = librosa.load(input_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)
        chunk_length = 10  # seconds
        num_chunks = math.ceil(duration / chunk_length)

        # Prepare output filename (with auto-increment)
        output_filename = "Extracted Audio.txt"
        output_path = os.path.join(output_dir, output_filename)
        output_path = get_unique_filename(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(num_chunks):
                start = i * chunk_length
                end = min((i+1) * chunk_length, duration)

                # Slice audio for this chunk
                chunk_audio = audio[int(start*sr):int(end*sr)]
                temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(temp_chunk.name, chunk_audio, sr)

                # Show progress %
                percent = int(((i+1) / num_chunks) * 100)
                log_message(f"🔊 Transcribing chunk {i+1}/{num_chunks} ({start:.1f}s–{end:.1f}s) ~{percent}% complete")

                # Transcribe chunk
                result = model.transcribe(temp_chunk.name, verbose=False, language=None if language=="auto" else language)

                # Stream results into GUI + save
                if show_timestamps:
                    for seg in result["segments"]:
                        seg_start = seg["start"] + start
                        timestamp = f"[{int(seg_start//3600):02}:{int((seg_start%3600)//60):02}:{int(seg_start%60):02}]"
                        line = f"{timestamp} {seg['text'].strip()}"
                        log_message(line)
                        f.write(line + "\n")
                else:
                    text = result["text"].strip()
                    log_message(text)
                    f.write(text + " ")

                # cleanup chunk file
                temp_chunk.close()
                os.remove(temp_chunk.name)

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        log_message(f"✅ Transcription saved: {output_path}")
        log_message(f"⏱ Finished in {mins}m {secs}s")

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            log_message("Temporary audio file removed.")


def start_transcription():
    input_path = input_entry.get()
    output_dir = output_entry.get()
    model_name = model_var.get()
    show_timestamps = timestamp_var.get()
    language = language_var.get()

    if not input_path or not os.path.exists(input_path):
        messagebox.showerror("Error", "Please select a valid input file.")
        return
    if not output_dir or not os.path.exists(output_dir):
        messagebox.showerror("Error", "Please select a valid output directory.")
        return

    log_box.delete("1.0", tk.END)  # clear previous log
    log_message("Starting transcription...")
    progress_bar.start()

    def worker():
        try:
            execute_whisper(input_path, output_dir, model_name, show_timestamps, language)
        except Exception as e:
            log_message(f"❌ Error: {e}")
        finally:
            progress_bar.stop()

    threading.Thread(target=worker, daemon=True).start()


# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("Whisper Transcription Tool (v1.7)")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Input file
ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
input_entry = ttk.Entry(input_frame, width=50)
input_entry.grid(row=0, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=lambda: input_entry.insert(0, filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")]))).grid(row=0, column=2)

# Drag & Drop support
def drop(event):
    file_path = event.data.strip("{}")  # clean path
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)
root.drop_target_register("DND_Files")
root.dnd_bind("<<Drop>>", drop)

# Output dir
ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
output_entry = ttk.Entry(input_frame, width=50)
output_entry.grid(row=1, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=lambda: output_entry.insert(0, filedialog.askdirectory())).grid(row=1, column=2)

# Model dropdown
ttk.Label(input_frame, text="Whisper Model:").grid(row=2, column=0, sticky=tk.W)
model_var = tk.StringVar(value="base")
model_dropdown = ttk.Combobox(input_frame, textvariable=model_var, values=["tiny", "base", "small", "medium", "large"])
model_dropdown.grid(row=2, column=1, padx=5, sticky=tk.W)

# Language dropdown
ttk.Label(input_frame, text="Language:").grid(row=3, column=0, sticky=tk.W)
language_var = tk.StringVar(value="english")
language_dropdown = ttk.Combobox(input_frame, textvariable=language_var, values=["auto","english","spanish","french","german","japanese","chinese"])
language_dropdown.grid(row=3, column=1, padx=5, sticky=tk.W)

# Timestamp toggle
timestamp_var = tk.BooleanVar(value=False)
ttk.Checkbutton(input_frame, text="Show timestamps", variable=timestamp_var).grid(row=4, column=1, sticky=tk.W, pady=5)

# Start button
ttk.Button(input_frame, text="Start Transcription", command=start_transcription).grid(row=5, column=1, pady=10)

# Progress bar
progress_bar = ttk.Progressbar(input_frame, length=300, mode='indeterminate')
progress_bar.grid(row=6, column=0, columnspan=3, pady=5)

# Status + log
status_var = tk.StringVar()
ttk.Label(input_frame, textvariable=status_var, wraplength=400).grid(row=7, column=0, columnspan=3, pady=5)

log_box = scrolledtext.ScrolledText(input_frame, width=60, height=15, wrap=tk.WORD)
log_box.grid(row=8, column=0, columnspan=3, pady=5)

root.mainloop()