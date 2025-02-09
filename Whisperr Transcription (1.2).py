import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import whisper
import moviepy.editor as mp

def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = os.path.splitext(video_path)[0] + "_temp.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def execute_whisper(input_path, output_dir, model_name, progress_var, status_var):
    status_var.set(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    status_var.set("Model loaded. Starting transcription...")
    
    temp_audio_path = None
    try:
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            status_var.set("Extracting audio from video...")
            temp_audio_path = extract_audio(input_path)
            input_path = temp_audio_path
            status_var.set("Audio extracted. Starting transcription...")
        
        result = model.transcribe(input_path)
        
        output_filename = os.path.splitext(os.path.basename(input_path))[0].replace("_temp", "") + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        status_var.set(f"Transcription saved to {output_path}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            status_var.set(status_var.get() + "\nTemporary audio file removed.")

def transcribe_thread(input_path, output_dir, model_name, progress_var, status_var):
    try:
        execute_whisper(input_path, output_dir, model_name, progress_var, status_var)
    except Exception as e:
        status_var.set(f"An error occurred: {str(e)}")

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")])
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)

def select_output_dir():
    dir_path = filedialog.askdirectory()
    output_entry.delete(0, tk.END)
    output_entry.insert(0, dir_path)

def start_transcription():
    input_path = input_entry.get()
    output_dir = output_entry.get()
    model_name = model_var.get()
    
    if not input_path or not os.path.exists(input_path):
        status_var.set("Please select a valid input file.")
        return
    
    if not output_dir or not os.path.exists(output_dir):
        status_var.set("Please select a valid output directory.")
        return
    
    progress_var.set(0)
    status_var.set("Starting transcription process...")
    threading.Thread(target=transcribe_thread, args=(input_path, output_dir, model_name, progress_var, status_var), daemon=True).start()
    update_progress()

def update_progress():
    if status_var.get().startswith("Starting"):
        progress_var.set(progress_var.get() + 1)
        root.after(1000, update_progress)  # Update every second

root = tk.Tk()
root.title("Whisperr Transcription (v1.2)")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
input_entry = ttk.Entry(input_frame, width=50)
input_entry.grid(row=0, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=select_input_file).grid(row=0, column=2)

ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
output_entry = ttk.Entry(input_frame, width=50)
output_entry.grid(row=1, column=1, padx=5)
ttk.Button(input_frame, text="Browse", command=select_output_dir).grid(row=1, column=2)

ttk.Label(input_frame, text="Whisper Model:").grid(row=2, column=0, sticky=tk.W)
model_var = tk.StringVar(value="base")
model_dropdown = ttk.Combobox(input_frame, textvariable=model_var, values=["tiny", "base", "small", "medium", "large"])
model_dropdown.grid(row=2, column=1, padx=5, sticky=tk.W)

ttk.Button(input_frame, text="Start Transcription", command=start_transcription).grid(row=3, column=1, pady=10)

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(input_frame, length=300, mode='indeterminate', variable=progress_var)
progress_bar.grid(row=4, column=0, columnspan=3, pady=5)

status_var = tk.StringVar()
status_label = ttk.Label(input_frame, textvariable=status_var, wraplength=400)
status_label.grid(row=5, column=0, columnspan=3)

root.mainloop()