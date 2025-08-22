import sys
import subprocess
import importlib
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

# ------------------- Package Check ------------------- #
REQUIRED_PACKAGES = ["openai-whisper", "ffmpeg-python", "tkinterdnd2"]

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
import numpy as np
import tempfile

# ------------------- Helpers ------------------- #
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

def extract_audio(video_path, output_path):
    """Extract audio track from video using ffmpeg-python, saves as WAV."""
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

def log_message(msg):
    """Append message to the GUI log box."""
    log_box.config(state="normal")
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    log_box.config(state="disabled")
    root.update_idletasks()

# ------------------- Transcription ------------------- #
def execute_whisper(input_path, output_dir, model_name, language, show_timestamps):
    model = whisper.load_model(model_name)
    temp_audio_path = None

    start_time = time.time()
    try:
        # Extract audio if video
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            log_message("Extracting audio from video...")
            temp_audio_path = os.path.splitext(input_path)[0] + "_temp.wav"
            extract_audio(input_path, temp_audio_path)
            input_path = temp_audio_path
            log_message("Audio extracted.")

        log_message("Starting transcription...")
        audio = whisper.load_audio(input_path)
        audio = whisper.pad_or_trim(audio)
        sr = whisper.audio.SAMPLE_RATE

        # Break into 5 second chunks
        chunk_size = sr * 5
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        log_message(f"Audio length: {len(audio)/sr:.1f}s, ~{num_chunks} chunks.")

        # Output file
        output_filename = "Extracted Audio.txt"
        output_path = os.path.join(output_dir, output_filename)
        output_path = get_unique_filename(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(audio))
                chunk = audio[start:end]

                mel = whisper.log_mel_spectrogram(chunk).to(model.device)
                options = {"language": language if language else None}
                result = model.decode(mel, whisper.DecodingOptions(**options))

                timestamp = f"[{int(start/sr//3600):02}:{int((start/sr%3600)//60):02}:{int((start/sr)%60):02}]"
                line = f"{timestamp} {result.text.strip()}" if show_timestamps else result.text.strip()

                if line:
                    f.write(line + "\n")
                    log_message(line)

                # Update progress %
                percent = int(((i + 1) / num_chunks) * 100)
                status_var.set(f"Progress: {percent}% ({i+1}/{num_chunks} chunks)")

        elapsed = time.time() - start_time
        log_message(f"✅ Transcription complete! Saved as: {output_path}")
        log_message(f"Elapsed Time: {elapsed:.2f}s")

    except Exception as e:
        log_message(f"❌ Error: {e}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            log_message("Temporary audio file removed.")

def start_transcription():
    input_path = input_entry.get()
    output_dir = output_entry.get()
    model_name = model_var.get()
    language = lang_var.get()
    show_timestamps = timestamps_var.get()

    if not input_path or not os.path.exists(input_path):
        messagebox.showerror("Error", "Please select a valid input file.")
        return
    if not output_dir or not os.path.exists(output_dir):
        messagebox.showerror("Error", "Please select a valid output directory.")
        return

    status_var.set("Starting transcription...")
    progress_bar.start()

    def worker():
        try:
            execute_whisper(input_path, output_dir, model_name, language, show_timestamps)
        finally:
            progress_bar.stop()

    threading.Thread(target=worker, daemon=True).start()

# ------------------- GUI ------------------- #
root = TkinterDnD.Tk()
root.title("Whisper Transcription Tool v1.7")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Input file
ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
input_entry = ttk.Entry(input_frame, width=50)
input_entry.grid(row=0, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=lambda: input_entry.insert(0, filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")]))).grid(row=0, column=2)

# Drag & drop
def drop(event):
    file_path = event.data.strip("{}")
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)
input_entry.drop_target_register(DND_FILES)
input_entry.dnd_bind("<<Drop>>", drop)

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
lang_var = tk.StringVar(value="en")
lang_dropdown = ttk.Combobox(input_frame, textvariable=lang_var, values=["en", "es", "fr", "de", "zh", "ja", "auto"])
lang_dropdown.grid(row=3, column=1, padx=5, sticky=tk.W)

# Timestamp toggle
timestamps_var = tk.BooleanVar(value=True)
ttk.Checkbutton(input_frame, text="Show Timestamps", variable=timestamps_var).grid(row=4, column=1, sticky=tk.W)

# Start button
ttk.Button(input_frame, text="Start Transcription", command=start_transcription).grid(row=5, column=1, pady=10)

# Progress bar + status
progress_bar = ttk.Progressbar(input_frame, length=300, mode='indeterminate')
progress_bar.grid(row=6, column=0, columnspan=3, pady=5)

status_var = tk.StringVar()
ttk.Label(input_frame, textvariable=status_var, wraplength=400).grid(row=7, column=0, columnspan=3)

# Log box
log_box = tk.Text(input_frame, height=15, width=60, state="disabled", wrap="word")
log_box.grid(row=8, column=0, columnspan=3, pady=10)

root.mainloop()
