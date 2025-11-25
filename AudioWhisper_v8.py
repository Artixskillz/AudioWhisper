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

# ---------------- Smart Installer Logic ---------------- #
def check_dependencies():
    """Check and install dependencies."""
    print("---------------------------------------------------")
    print("       AudioWhisper v8 Smart Installer")
    print("---------------------------------------------------")
    
    REQUIRED_PACKAGES = [
        "customtkinter",
        "faster-whisper",
        "ffmpeg-python",
        "librosa",
        "soundfile",
        "packaging",
        "tkinterdnd2"
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
        
        # Click to browse
        self.bind("<Button-1>", self.on_click)
        self.label.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.command:
            self.command()

    def set_file(self, filename):
        self.label.configure(text=f"📄\n\n{os.path.basename(filename)}\n\n(Ready to Transcribe)")
        self.configure(fg_color=("green", "darkgreen")) # Visual feedback

# ---------------- Main App ---------------- #

class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class AudioWhisperApp(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("AudioWhisper v8.0")
        self.geometry("800x850")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1) # Transcript expands

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

        # Theme Toggle
        self.theme_btn = ctk.CTkButton(self.header_frame, text="🌗", width=40, command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=10)

        # Device Badge
        device_color = "green" if self.device == "cuda" else "orange"
        device_text = f"{self.device.upper()}"
        ctk.CTkLabel(self.header_frame, text=device_text, text_color=device_color, font=("Arial", 12, "bold")).pack(side="right")

        # --- 2. Drop Zone ---
        self.drop_zone = DropZone(self, command=self.browse_input)
        self.drop_zone.grid(row=1, column=0, padx=20, pady=10, sticky="ew", ipady=40)

        # --- 3. Progress Section ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, variable=self.progress_val)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self.progress_frame, textvariable=self.status_msg, font=("Arial", 12), text_color="gray60")
        self.status_label.grid(row=1, column=0, sticky="w")

        # --- 4. Controls & Settings ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
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
        self.settings_group.toggle() # Open by default

        # Settings Content
        s_frame = self.settings_group.content_frame
        
        ctk.CTkLabel(s_frame, text="Output Folder:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(s_frame, textvariable=self.output_dir, width=300).grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkButton(s_frame, text="Browse", width=60, command=self.browse_output).grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkLabel(s_frame, text="Model Size:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(s_frame, values=["tiny", "base", "small", "medium", "large-v3"], variable=self.model_name).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkSwitch(s_frame, text="Timestamps", variable=self.show_timestamps).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkSwitch(s_frame, text="Export .SRT", variable=self.export_srt).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # --- 5. Live Transcript ---
        self.transcript_box = ctk.CTkTextbox(self, font=("Consolas", 14))
        self.transcript_box.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.transcript_box.insert("1.0", "Transcription will appear here...\n")
        self.transcript_box.configure(state="disabled")

        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Logic ---------------- #

    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")

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

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")])
        if filename:
            self.input_path.set(filename)
            self.drop_zone.set_file(filename)

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
                # System messages go to status label
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
        
        try:
            self.log("Initializing model...", is_transcript=False)
            
            # Get Duration for Progress Bar
            total_duration = self.get_audio_duration(input_file)
            
            # Load Model
            try:
                model = WhisperModel(model_type, device=self.device, compute_type=self.compute_type)
            except Exception as e:
                self.log(f"Error loading model: {e}", is_transcript=False)
                return

            if self.stop_event.is_set(): return

            # Handle Video
            if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.log("Extracting audio...", is_transcript=False)
                temp_file = self.extract_audio(input_file)
                process_file = temp_file
                # Update duration if we extracted audio (might be more accurate)
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

                # Update Progress
                if total_duration > 0:
                    prog = segment.end / total_duration
                    self.progress_val.set(prog)
                
                # Update Transcript
                timestamp_str = f"[{int(segment.start//3600):02}:{int((segment.start%3600)//60):02}:{int(segment.start%60):02}]"
                self.log(f"{timestamp_str} {segment.text.strip()}", is_transcript=True)
                
                collected_segments.append(segment)

            if self.stop_event.is_set() and not collected_segments:
                self.log("Stopped.", is_transcript=False)
                return

            # Save Files
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            if not output_folder: output_folder = os.path.dirname(input_file)
            
            # TXT
            txt_path = os.path.join(output_folder, f"{base_name}_transcript.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in collected_segments:
                    if self.show_timestamps.get():
                        f.write(f"[{seg.start:.2f}s] {seg.text.strip()}\n")
                    else:
                        f.write(f"{seg.text.strip()} ")
            
            # SRT
            if self.export_srt.get():
                srt_path = os.path.join(output_folder, f"{base_name}.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(collected_segments, start=1):
                        start = self.format_srt_time(seg.start)
                        end = self.format_srt_time(seg.end)
                        f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")

            self.log("Done! Files saved.", is_transcript=False)
            self.progress_val.set(1.0)

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

    def start_transcription(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select a file.")
            return
        
        self.is_transcribing = True
        self.is_paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        self.progress_val.set(0)
        
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
            self.pause_btn.configure(text="Pause", fg_color="orange")
            self.status_msg.set("Resumed...")
        else:
            self.is_paused = True
            self.pause_event.set()
            self.pause_btn.configure(text="Resume", fg_color="green")
            self.status_msg.set("Paused")

    def update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            self.drop_zone.configure(fg_color="gray90") # Visual disable
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled")
            self.drop_zone.configure(fg_color=("gray85", "gray25")) # Reset

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
