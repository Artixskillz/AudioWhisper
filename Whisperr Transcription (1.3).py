import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import whisper
import ffmpeg


def extract_audio(video_path, output_path):
    """
    Extract audio track from video using ffmpeg-python.
    Saves as WAV file.
    """
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


def execute_whisper(input_path, output_dir, model_name, status_var):
    """
    Run Whisper transcription on the given input file and save the result.
    """
    status_var.set(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    temp_audio_path = None
    try:
        # If input is a video, extract audio
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            status_var.set("Extracting audio from video...")
            temp_audio_path = os.path.splitext(input_path)[0] + "_temp.wav"
            extract_audio(input_path, temp_audio_path)
            input_path = temp_audio_path
            status_var.set("Audio extracted. Starting transcription...")

        # Transcribe
        status_var.set("Transcribing... this may take a while.")
        result = model.transcribe(input_path)

        # Save result
        output_filename = os.path.splitext(os.path.basename(input_path))[0].replace("_temp", "") + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        status_var.set(f"✅ Transcription saved: {output_path}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            status_var.set(status_var.get() + "\nTemporary audio file removed.")


def start_transcription():
    input_path = input_entry.get()
    output_dir = output_entry.get()
    model_name = model_var.get()

    if not input_path or not os.path.exists(input_path):
        messagebox.showerror("Error", "Please select a valid input file.")
        return
    if not output_dir or not os.path.exists(output_dir):
        messagebox.showerror("Error", "Please select a valid output directory.")
        return

    status_var.set("Starting transcription...")
    progress_bar.start()  # start spinning

    def worker():
        try:
            execute_whisper(input_path, output_dir, model_name, status_var)
        except Exception as e:
            status_var.set(f"❌ Error: {e}")
        finally:
            progress_bar.stop()

    threading.Thread(target=worker, daemon=True).start()


# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("Whisper Transcription Tool (Improved)")

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

# Start button
ttk.Button(input_frame, text="Start Transcription", command=start_transcription).grid(row=3, column=1, pady=10)

# Progress bar + status
progress_bar = ttk.Progressbar(input_frame, length=300, mode='indeterminate')
progress_bar.grid(row=4, column=0, columnspan=3, pady=5)

status_var = tk.StringVar()
ttk.Label(input_frame, textvariable=status_var, wraplength=400).grid(row=5, column=0, columnspan=3)

root.mainloop()
