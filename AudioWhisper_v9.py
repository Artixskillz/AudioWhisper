import sys
import subprocess
import importlib
import os
import threading
import time
import queue
import json
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# ---------------- Smart Installer Logic ---------------- #
def check_dependencies():
    """Check and install dependencies."""
    print("---------------------------------------------------")
    print("       AudioWhisper v9 Smart Installer")
    print("---------------------------------------------------")
    
    REQUIRED_PACKAGES = [
        "customtkinter",
        "faster-whisper",
        "ffmpeg-python",
        "librosa",
        "soundfile",
        "packaging",
        "tkinterdnd2",
        "numpy"
    ]

    for package in REQUIRED_PACKAGES:
        try:
            import_name = package
            if package == "faster-whisper": import_name = "faster_whisper"
            elif package == "ffmpeg-python": import_name = "ffmpeg"
            elif package == "customtkinter": import_name = "customtkinter"
            elif package == "tkinterdnd2": import_name = "tkinterdnd2"
            
            importlib.import_module(import_name)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
    try:
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

check_dependencies()

# ---------------- Imports ---------------- #
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from faster_whisper import WhisperModel
import ffmpeg
import librosa
import soundfile as sf
import torch

# ---------------- Configuration ---------------- #
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# ---------------- Custom Widgets ---------------- #

class WaveformVisualizer(ctk.CTkCanvas):
    def __init__(self, master, width=600, height=60, bg_color=None):
        super().__init__(master, width=width, height=height, highlightthickness=0)
        self.configure(bg=bg_color if bg_color else "gray20")
        self.bars = 100
        self.amplitudes = np.zeros(self.bars)
        self.width = width
        self.height = height
        self.bar_width = width / self.bars
        self.progress = 0.0 # 0.0 to 1.0

    def load_audio(self, file_path):
        """Load audio and compute simplified waveform."""
        try:
            # Load only 30 seconds max to be fast, or resample heavily
            y, sr = librosa.load(file_path, sr=8000, duration=None) # Load full for accurate shape
            
            # Resample to 'bars' number of points
            # We take the max amplitude in each chunk
            chunk_size = len(y) // self.bars
            if chunk_size < 1: chunk_size = 1
            
            new_amps = []
            for i in range(self.bars):
                start = i * chunk_size
                end = start + chunk_size
                chunk = y[start:end]
                if len(chunk) > 0:
                    new_amps.append(np.max(np.abs(chunk)))
                else:
                    new_amps.append(0)
            
            # Normalize
            max_val = max(new_amps) if max(new_amps) > 0 else 1
            self.amplitudes = np.array(new_amps) / max_val
            self.draw()
            
        except Exception as e:
            print(f"Waveform Error: {e}")

    def set_progress(self, progress):
        self.progress = progress
        self.draw()

    def draw(self):
        self.delete("all")
        
        for i, amp in enumerate(self.amplitudes):
            x1 = i * self.bar_width
            x2 = x1 + self.bar_width - 1 # 1px gap
            
            bar_h = amp * self.height
            y1 = (self.height - bar_h) / 2
            y2 = y1 + bar_h
            
            # Color based on progress
            # If this bar is "behind" the progress head, make it active color
            bar_pos_pct = i / self.bars
            
            if bar_pos_pct < self.progress:
                color = "#00E676" # Bright Green/Cyan
            else:
                color = "gray40"
                
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title="Advanced Settings"):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.animation = False
        self.is_expanded = False
        
        self.title_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.title_frame.grid(row=0, column=0, sticky="ew")
        self.title_frame.grid_columnconfigure(0, weight=1)

        self.toggle_btn = ctk.CTkButton(
            self.title_frame, 
            text=f"▶ {title}", 
            width=100, 
            anchor="w", 
            fg_color="transparent", 
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            command=self.toggle
        )
        self.toggle_btn.grid(row=0, column=0, sticky="ew")

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")

    def toggle(self):
        if self.is_expanded:
            self.content_frame.grid_forget()
            self.toggle_btn.configure(text=self.toggle_btn.cget("text").replace("▼", "▶"))
        else:
            self.content_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            self.toggle_btn.configure(text=self.toggle_btn.cget("text").replace("▶", "▼"))
        self.is_expanded = not self.is_expanded

class DropZone(ctk.CTkFrame):
    def __init__(self, master, command=None):
        super().__init__(master, fg_color=("gray85", "gray25"), corner_radius=15)
        self.command = command
        
        self.label = ctk.CTkLabel(
            self, 
            text="📂\n\nDrag & Drop Audio/Video Here\n\n- or click to browse -", 
            font=("Arial", 16, "bold"),
            text_color=("gray50", "gray70")
        )
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.bind("<Button-1>", self.on_click)
        self.label.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.command:
            self.command()

    def set_file(self, filename):
        self.label.configure(text=f"📄\n\n{os.path.basename(filename)}\n\n(Ready to Transcribe)")
        self.configure(fg_color=("green", "darkgreen"))

# ---------------- Main App ---------------- #

class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class AudioWhisperApp(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("AudioWhisper v9.0 (Pro)")
        self.geometry("800x900")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1) # Transcript expands

        # Settings File
        self.settings_file = "settings.json"

        # State Variables
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.model_name = ctk.StringVar(value="base")
        self.show_timestamps = ctk.BooleanVar(value=True)
        self.export_srt = ctk.BooleanVar(value=False)
        self.status_msg = ctk.StringVar(value="Ready")
        self.progress_val = ctk.DoubleVar(value=0.0)
        self.time_remaining_msg = ctk.StringVar(value="")
        
        self.is_transcribing = False
        self.is_paused = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.log_queue = queue.Queue()

        # Load Settings
        self.load_settings()

        # Device Detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"

        # UI Layout
        self.create_widgets()
        
        # Enable Drag & Drop (Global)
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop_file)

        # Start Log Polling
        self.check_log_queue()

    def create_widgets(self):
        # --- 1. Header ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        title = ctk.CTkLabel(self.header_frame, text="AudioWhisper 🎙️", font=("Arial", 24, "bold"))
        title.pack(side="left")

        self.theme_btn = ctk.CTkButton(self.header_frame, text="🌗", width=40, command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=10)

        device_color = "green" if self.device == "cuda" else "orange"
        device_text = f"{self.device.upper()}"
        ctk.CTkLabel(self.header_frame, text=device_text, text_color=device_color, font=("Arial", 12, "bold")).pack(side="right")

        # --- 2. Drop Zone ---
        self.drop_zone = DropZone(self, command=self.browse_input)
        self.drop_zone.grid(row=1, column=0, padx=20, pady=10, sticky="ew", ipady=30)

        # --- 3. Visualizer (NEW) ---
        self.viz_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.viz_frame.grid(row=2, column=0, padx=20, pady=0, sticky="ew")
        self.visualizer = WaveformVisualizer(self.viz_frame, width=760, height=60, bg_color="#2B2B2B") # Dark gray bg
        self.visualizer.pack(fill="x")

        # --- 4. Progress Section ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, variable=self.progress_val)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.progress_bar.set(0)
        
        # Status Row
        self.status_row = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        self.status_row.grid(row=1, column=0, sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.status_row, textvariable=self.status_msg, font=("Arial", 12), text_color="gray60")
        self.status_label.pack(side="left")
        
        self.time_label = ctk.CTkLabel(self.status_row, textvariable=self.time_remaining_msg, font=("Arial", 12, "bold"), text_color="gray60")
        self.time_label.pack(side="right")

        # --- 5. Controls & Settings ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # Buttons Row
        self.btn_row = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.btn_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_btn = ctk.CTkButton(self.btn_row, text="Start Transcription", fg_color="green", hover_color="darkgreen", height=40, font=("Arial", 14, "bold"), command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.pause_btn = ctk.CTkButton(self.btn_row, text="Pause", fg_color="#FFC107", text_color="black", hover_color="#FFD54F", height=40, state="disabled", command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=1, padx=5, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.btn_row, text="Stop", fg_color="#F44336", text_color="black", hover_color="#E57373", height=40, state="disabled", command=self.stop_transcription)
        self.stop_btn.grid(row=0, column=2, padx=5, sticky="ew")

        # Collapsible Settings
        self.settings_group = CollapsibleFrame(self.controls_frame, title="Advanced Settings")
        self.settings_group.grid(row=1, column=0, sticky="ew")
        self.settings_group.toggle()

        # Settings Content
        s_frame = self.settings_group.content_frame
        
        ctk.CTkLabel(s_frame, text="Output Folder:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(s_frame, textvariable=self.output_dir, width=300).grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkButton(s_frame, text="Browse", width=60, command=self.browse_output).grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkLabel(s_frame, text="Model Size:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(s_frame, values=["tiny", "base", "small", "medium", "large-v3"], variable=self.model_name).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkSwitch(s_frame, text="Timestamps", variable=self.show_timestamps).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkSwitch(s_frame, text="Export .SRT", variable=self.export_srt).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # --- 6. Live Transcript ---
        self.transcript_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.transcript_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.transcript_frame.grid_columnconfigure(0, weight=1)
        self.transcript_frame.grid_rowconfigure(0, weight=1)
        
        self.transcript_box = ctk.CTkTextbox(self.transcript_frame, font=("Consolas", 14))
        self.transcript_box.grid(row=0, column=0, sticky="nsew")
        self.transcript_box.insert("1.0", "Transcription will appear here...\n")
        self.transcript_box.configure(state="disabled")

        # Tools Row (Copy / Open Folder)
        self.tools_row = ctk.CTkFrame(self.transcript_frame, fg_color="transparent")
        self.tools_row.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        self.copy_btn = ctk.CTkButton(self.tools_row, text="📋 Copy Text", width=100, command=self.copy_text)
        self.copy_btn.pack(side="left")
        
        self.open_folder_btn = ctk.CTkButton(self.tools_row, text="📂 Open Folder", width=100, state="disabled", command=self.open_output_folder)
        self.open_folder_btn.pack(side="right")

        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Logic ---------------- #

    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
            self.visualizer.configure(bg="gray90") # Light bg for viz
        else:
            ctk.set_appearance_mode("Dark")
            self.visualizer.configure(bg="#2B2B2B") # Dark bg for viz

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                    self.model_name.set(data.get("model", "base"))
                    self.output_dir.set(data.get("output_dir", ""))
                    self.show_timestamps.set(data.get("timestamps", True))
                    self.export_srt.set(data.get("export_srt", False))
            except Exception:
                pass

    def save_settings(self):
        data = {
            "model": self.model_name.get(),
            "output_dir": self.output_dir.get(),
            "timestamps": self.show_timestamps.get(),
            "export_srt": self.export_srt.get()
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def on_close(self):
        self.save_settings()
        self.destroy()

    def drop_file(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        self.input_path.set(file_path)
        self.drop_zone.set_file(file_path)
        
        # Load waveform preview
        threading.Thread(target=self.visualizer.load_audio, args=(file_path,), daemon=True).start()

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")])
        if filename:
            self.input_path.set(filename)
            self.drop_zone.set_file(filename)
            threading.Thread(target=self.visualizer.load_audio, args=(filename,), daemon=True).start()

    def browse_output(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.output_dir.set(dirname)

    def log(self, message, is_transcript=False):
        self.log_queue.put((message, is_transcript))

    def check_log_queue(self):
        while not self.log_queue.empty():
            msg, is_transcript = self.log_queue.get()
            
            if is_transcript:
                self.transcript_box.configure(state="normal")
                self.transcript_box.insert("end", msg + "\n")
                self.transcript_box.see("end")
                self.transcript_box.configure(state="disabled")
            else:
                self.status_msg.set(msg)
        
        self.after(100, self.check_log_queue)

    def get_audio_duration(self, file_path):
        try:
            return librosa.get_duration(path=file_path)
        except:
            return 0

    def extract_audio(self, video_path):
        temp_audio = os.path.splitext(video_path)[0] + "_temp.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(temp_audio, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            return temp_audio
        except Exception as e:
            raise RuntimeError(f"FFmpeg failed: {e}")

    def run_transcription(self):
        input_file = self.input_path.get()
        output_folder = self.output_dir.get()
        model_type = self.model_name.get()
        
        temp_file = None
        start_time = time.time()
        
        try:
            self.log("Initializing model...", is_transcript=False)
            
            total_duration = self.get_audio_duration(input_file)
            
            try:
                model = WhisperModel(model_type, device=self.device, compute_type=self.compute_type)
            except Exception as e:
                self.log(f"Error loading model: {e}", is_transcript=False)
                return

            if self.stop_event.is_set(): return

            if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.log("Extracting audio...", is_transcript=False)
                temp_file = self.extract_audio(input_file)
                process_file = temp_file
                total_duration = self.get_audio_duration(process_file)
            else:
                process_file = input_file

            if self.stop_event.is_set(): return

            self.log("Transcribing...", is_transcript=False)
            self.transcript_box.configure(state="normal")
            self.transcript_box.delete("1.0", "end")
            self.transcript_box.configure(state="disabled")
            
            segments, info = model.transcribe(process_file, beam_size=5)
            
            collected_segments = []

            for segment in segments:
                if self.stop_event.is_set(): break
                while self.pause_event.is_set():
                    if self.stop_event.is_set(): break
                    time.sleep(0.1)

                # Update Progress & Time Remaining
                if total_duration > 0:
                    prog = segment.end / total_duration
                    self.progress_val.set(prog)
                    self.visualizer.set_progress(prog)
                    
                    # Estimate Time
                    elapsed = time.time() - start_time
                    if prog > 0.01:
                        total_estimated = elapsed / prog
                        remaining = total_estimated - elapsed
                        mins, secs = divmod(int(remaining), 60)
                        self.time_remaining_msg.set(f"Remaining: {mins:02}:{secs:02}")
                
                timestamp_str = f"[{int(segment.start//3600):02}:{int((segment.start%3600)//60):02}:{int(segment.start%60):02}]"
                self.log(f"{timestamp_str} {segment.text.strip()}", is_transcript=True)
                
                collected_segments.append(segment)

            if self.stop_event.is_set() and not collected_segments:
                self.log("Stopped.", is_transcript=False)
                return

            # Save Files
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            if not output_folder: output_folder = os.path.dirname(input_file)
            
            txt_path = os.path.join(output_folder, f"{base_name}_transcript.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in collected_segments:
                    if self.show_timestamps.get():
                        f.write(f"[{seg.start:.2f}s] {seg.text.strip()}\n")
                    else:
                        f.write(f"{seg.text.strip()} ")
            
            if self.export_srt.get():
                srt_path = os.path.join(output_folder, f"{base_name}.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(collected_segments, start=1):
                        start = self.format_srt_time(seg.start)
                        end = self.format_srt_time(seg.end)
                        f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")

            self.log("Done! Files saved.", is_transcript=False)
            self.progress_val.set(1.0)
            self.visualizer.set_progress(1.0)
            self.time_remaining_msg.set("Complete")
            self.open_folder_btn.configure(state="normal")

        except Exception as e:
            self.log(f"Error: {e}", is_transcript=False)
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            
            self.stop_event.clear()
            self.pause_event.clear()
            self.is_transcribing = False
            self.is_paused = False
            self.update_ui_state(transcribing=False)

    def format_srt_time(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        total_sec = int(seconds)
        millis = int((seconds - total_sec) * 1000)
        hours, remainder = divmod(total_sec, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def copy_text(self):
        text = self.transcript_box.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_msg.set("Copied to clipboard!")

    def open_output_folder(self):
        path = self.output_dir.get()
        if path and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showerror("Error", "Output folder not found.")

    def start_transcription(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select a file.")
            return
        
        self.is_transcribing = True
        self.is_paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        self.progress_val.set(0)
        self.visualizer.set_progress(0)
        self.time_remaining_msg.set("Calculating...")
        
        self.update_ui_state(transcribing=True)
        self.save_settings()
        
        threading.Thread(target=self.run_transcription, daemon=True).start()

    def stop_transcription(self):
        if self.is_transcribing:
            self.stop_event.set()
            if self.pause_event.is_set(): self.pause_event.clear()

    def toggle_pause(self):
        if not self.is_transcribing: return
        
        if self.is_paused:
            self.is_paused = False
            self.pause_event.clear()
            self.pause_btn.configure(text="Pause", fg_color="#FFC107")
            self.status_msg.set("Resumed...")
        else:
            self.is_paused = True
            self.pause_event.set()
            self.pause_btn.configure(text="Resume", fg_color="#00E676") # Green for resume
            self.status_msg.set("Paused")

    def update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            self.drop_zone.configure(fg_color="gray90")
            self.open_folder_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled")
            self.drop_zone.configure(fg_color=("gray85", "gray25"))

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
