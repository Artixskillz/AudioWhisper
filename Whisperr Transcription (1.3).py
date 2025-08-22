import sys
import subprocess
import importlib
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext

# auto-install required packages
REQUIRED_PACKAGES = ["openai-whisper", "ffmpeg-python"]

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


def execute_whisper(input_path, output_dir, model_name, show_timestamps):
    """Run Whisper transcription and save result."""
    log_message(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    temp_audio_path = None
    try:
        # If input is a video, extract audio
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            log_message("Extracting audio from video...")
            temp_audio_path = os.path.splitext(input_path)[0] + "_temp.wav"
            extract_audio(input_path, temp_audio_path)
            input_path = temp_audio_path
            log_message("Audio extracted. Starting transcription...")

        # Prepare output filename
        output_filename = "Extracted Audio.txt"
        output_path = os.path.join(output_dir, output_filename)
        output_path = get_unique_filename(output_path)

        # Transcribe
        log_message("Transcribing... this may take a while.")
        result = model.transcribe(input_path, verbose=False)

        with open(output_path, "w", encoding="utf-8") as f:
            if show_timestamps:
                for seg in result["segments"]:
                    start = seg["start"]
                    timestamp = f"[{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02}]"
                    line = f"{timestamp} {seg['text'].strip()}"
                    log_message(line)   # stream to GUI
                    f.write(line + "\n")
            else:
                for seg in result["segments"]:
                    line = seg["text"].strip()
                    log_message(line)   # stream to GUI
                    f.write(line + " ")

        log_message(f"✅ Transcription saved: {output_path}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            log_message("Temporary audio file removed.")


def start_transcription():
    input_path = input_entry.get()
    output_dir = output_entry.get()
    model_name = model_var.get()
    show_timestamps = timestamp_var.get()

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
            execute_whisper(input_path, output_dir, model_name, show_timestamps)
        except Exception as e:
            log_message(f"❌ Error: {e}")
        finally:
            progress_bar.stop()

    threading.Thread(target=worker, daemon=True).start()


# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("Whisper Transcription Tool (v1.5)")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Input file
ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
input_entry = ttk.Entry(input_frame, width=50)
input_entry.grid(row=0, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=lambda: input_entry.insert(0, filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")]))).grid(row=0, column=2)

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

# Timestamp toggle
timestamp_var = tk.BooleanVar(value=False)
ttk.Checkbutton(input_frame, text="Show timestamps", variable=timestamp_var).grid(row=3, column=1, sticky=tk.W, pady=5)

# Start button
ttk.Button(input_frame, text="Start Transcription", command=start_transcription).grid(row=4, column=1, pady=10)

# Progress bar
progress_bar = ttk.Progressbar(input_frame, length=300, mode='indeterminate')
progress_bar.grid(row=5, column=0, columnspan=3, pady=5)

# Status + log
status_var = tk.StringVar()
ttk.Label(input_frame, textvariable=status_var, wraplength=400).grid(row=6, column=0, columnspan=3, pady=5)

log_box = scrolledtext.ScrolledText(input_frame, width=60, height=15, wrap=tk.WORD)
log_box.grid(row=7, column=0, columnspan=3, pady=5)

root.mainloop()
