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
def get_gpu_info():
    """Detect NVIDIA GPU info using nvidia-smi or torch."""
    gpu_name = ""
    
    # Method 1: Try nvidia-smi (Most reliable for hardware ID)
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        if "GPU 0:" in output:
            gpu_name = output
    except Exception:
        pass

    # Method 2: Try torch (if installed)
    if not gpu_name:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    # Analyze GPU Name
    if "RTX 50" in gpu_name:
        return "blackwell" # RTX 50-series (Requires CUDA 12.8+)
    elif "NVIDIA" in gpu_name or "GeForce" in gpu_name or "Quadro" in gpu_name or "Tesla" in gpu_name:
        return "standard" # RTX 30/40, etc.
    
    return "none"

def check_dependencies():
    """Check and install dependencies, including faster-whisper."""
    print("---------------------------------------------------")
    print("       AudioWhisper v7 Smart Installer")
    print("---------------------------------------------------")
    
    REQUIRED_PACKAGES = [
        "customtkinter",
        "faster-whisper", # NEW: Replaces openai-whisper
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
            
    # Check Torch (still needed for some utils or if user wants to use it, 
    # though faster-whisper uses ctranslate2, we keep torch check for hardware detection reliability)
    try:
        import torch
    except ImportError:
        print("Installing PyTorch (helper)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

# Run the smart check
check_dependencies()

# ---------------- Imports ---------------- #
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from faster_whisper import WhisperModel
import ffmpeg
import librosa
import soundfile as sf
import torch # Used for device detection logic

# ---------------- Configuration ---------------- #
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Mixin for Drag & Drop support with CustomTkinter
class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class StdoutRedirector:
    """Redirects stdout to a callback function."""
    def __init__(self, callback):
        self.callback = callback

    def write(self, message):
        if message.strip():
            self.callback(message.strip())

    def flush(self):
        pass

class AudioWhisperApp(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("AudioWhisper v7.0 (Pause/Resume)")
        self.geometry("700x750") # Slightly taller for new buttons
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Settings File
        self.settings_file = "settings.json"

        # State Variables
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.model_name = ctk.StringVar(value="base")
        self.show_timestamps = ctk.BooleanVar(value=True)
        self.export_srt = ctk.BooleanVar(value=False)
        self.status_msg = ctk.StringVar(value="Ready")
        
        self.is_transcribing = False
        self.is_paused = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event() # Set when paused
        self.log_queue = queue.Queue()

        # Load Settings
        self.load_settings()

        # Device Detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_label_text = f"Device: {self.device.upper()} {'🟢' if self.device == 'cuda' else '🔴'}"
        self.device_color = "green" if self.device == "cuda" else "orange"

        # UI Layout
        self.create_widgets()
        
        # Enable Drag & Drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop_file)

        # Start Log Polling
        self.check_log_queue()

        # Log initial device status
        self.log(f"System initialized. Using device: {self.device.upper()}")
        if self.device == "cpu":
            self.log("⚠️ Warning: Running on CPU. Transcription may be slow.")

    def create_widgets(self):
        # --- Header / Device Status ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")
        
        self.device_label = ctk.CTkLabel(self.header_frame, text=self.device_label_text, text_color=self.device_color, font=("Arial", 14, "bold"))
        self.device_label.pack(side="right")

        # --- Input File ---
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.input_frame, text="Input File:").grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkEntry(self.input_frame, textvariable=self.input_path, placeholder_text="Drag & Drop file here...").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.input_frame, text="Browse", width=80, command=self.browse_input).grid(row=0, column=2, padx=10, pady=10)

        # --- Output Directory ---
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.output_frame, text="Output Dir:").grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkEntry(self.output_frame, textvariable=self.output_dir).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.output_frame, text="Browse", width=80, command=self.browse_output).grid(row=0, column=2, padx=10, pady=10)

        # --- Settings ---
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.settings_frame, text="Model:").grid(row=0, column=0, padx=10, pady=10)
        self.model_combo = ctk.CTkComboBox(self.settings_frame, values=["tiny", "base", "small", "medium", "large-v3"], variable=self.model_name)
        self.model_combo.grid(row=0, column=1, padx=10, pady=10)

        self.timestamp_switch = ctk.CTkSwitch(self.settings_frame, text="Timestamps (TXT)", variable=self.show_timestamps)
        self.timestamp_switch.grid(row=0, column=2, padx=20, pady=10)
        
        self.srt_switch = ctk.CTkSwitch(self.settings_frame, text="Export Subtitles (.SRT)", variable=self.export_srt)
        self.srt_switch.grid(row=0, column=3, padx=20, pady=10)

        # --- Controls ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_btn = ctk.CTkButton(self.controls_frame, text="Start Transcription", fg_color="green", hover_color="darkgreen", command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.pause_btn = ctk.CTkButton(self.controls_frame, text="Pause", fg_color="orange", hover_color="darkorange", state="disabled", command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.controls_frame, text="Stop", fg_color="red", hover_color="darkred", state="disabled", command=self.stop_transcription)
        self.stop_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        # --- Log Console ---
        self.log_box = ctk.CTkTextbox(self, width=600, height=200)
        self.log_box.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.log_box.configure(state="disabled")

        # --- Status Bar ---
        self.status_label = ctk.CTkLabel(self, textvariable=self.status_msg, anchor="w")
        self.status_label.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Logic ---------------- #

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
        self.log(f"File loaded via Drag & Drop: {os.path.basename(file_path)}")

    def log(self, message):
        self.log_queue.put(message)

    def check_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
            self.status_msg.set(msg)
        
        self.after(100, self.check_log_queue)

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mov *.mkv")])
        if filename:
            self.input_path.set(filename)

    def browse_output(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.output_dir.set(dirname)

    def get_unique_filename(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        counter = 2
        new_path = f"{base} ({counter}){ext}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base} ({counter}){ext}"
        return new_path

    def format_timestamp_srt(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        total_sec = int(seconds)
        millis = int((seconds - total_sec) * 1000)
        hours, remainder = divmod(total_sec, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

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
        use_timestamps = self.show_timestamps.get()
        do_export_srt = self.export_srt.get()

        temp_file = None
        start_time = time.time()

        try:
            self.log(f"Loading faster-whisper model: {model_type} on {self.device.upper()}...")
            
            # Compute type: float16 for GPU (if supported), float32 for CPU
            # To ensure ACCURACY, we avoid int8 quantization unless explicitly requested (which we aren't here)
            compute_type = "float16" if self.device == "cuda" else "float32"
            
            try:
                model = WhisperModel(model_type, device=self.device, compute_type=compute_type)
            except Exception as e:
                self.log(f"❌ Model Load Error: {e}")
                self.log("Falling back to CPU/float32...")
                self.device = "cpu"
                self.device_label.configure(text="Device: CPU (Fallback) 🔴", text_color="orange")
                model = WhisperModel(model_type, device="cpu", compute_type="float32")

            if self.stop_event.is_set(): return

            # Handle Video Input
            if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.log("Extracting audio from video...")
                temp_file = self.extract_audio(input_file)
                process_file = temp_file
            else:
                process_file = input_file

            if self.stop_event.is_set(): return

            self.log("Starting transcription...")
            
            # Transcribe (Generator)
            segments, info = model.transcribe(process_file, beam_size=5)
            
            self.log(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

            collected_segments = []

            # Iterate through segments (This allows us to Pause/Stop!)
            for segment in segments:
                # Check Stop
                if self.stop_event.is_set():
                    self.log("🛑 Transcription Stopped.")
                    break
                
                # Check Pause
                while self.pause_event.is_set():
                    if self.stop_event.is_set(): break
                    time.sleep(0.1) # Wait while paused
                
                # Process Segment
                start = segment.start
                end = segment.end
                text = segment.text.strip()
                
                timestamp_str = f"[{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02}]"
                self.log(f"{timestamp_str} {text}")
                
                collected_segments.append(segment)

            if self.stop_event.is_set() and not collected_segments:
                return # Stopped early with no data

            # Save Output (TXT)
            output_filename = "Transcription.txt"
            output_path = os.path.join(output_folder, output_filename)
            output_path = self.get_unique_filename(output_path)

            with open(output_path, "w", encoding="utf-8") as f:
                if use_timestamps:
                    for seg in collected_segments:
                        start = seg.start
                        timestamp = f"[{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02}]"
                        f.write(f"{timestamp} {seg.text.strip()}\n")
                else:
                    full_text = " ".join([seg.text.strip() for seg in collected_segments])
                    f.write(full_text)
            
            self.log(f"✅ Text saved to: {output_path}")

            # Save Output (SRT)
            if do_export_srt:
                srt_filename = os.path.splitext(output_filename)[0] + ".srt"
                srt_path = os.path.join(output_folder, srt_filename)
                srt_path = self.get_unique_filename(srt_path)
                
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(collected_segments, start=1):
                        start = self.format_timestamp_srt(seg.start)
                        end = self.format_timestamp_srt(seg.end)
                        text = seg.text.strip()
                        f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                
                self.log(f"✅ Subtitles saved to: {srt_path}")

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            self.log(f"⏱ Finished in {mins}m {secs}s")

        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                self.log("Temporary audio file removed.")
            
            self.stop_event.clear()
            self.pause_event.clear()
            self.is_transcribing = False
            self.is_paused = False
            self.update_ui_state(transcribing=False)

    def start_transcription(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file.")
            return
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        
        self.is_transcribing = True
        self.is_paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        
        self.update_ui_state(transcribing=True)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        
        self.save_settings()
        
        threading.Thread(target=self.run_transcription, daemon=True).start()

    def stop_transcription(self):
        if self.is_transcribing:
            self.log("🛑 Stopping...")
            self.stop_event.set()
            # If paused, we need to unpause so the loop can hit the stop check
            if self.pause_event.is_set():
                self.pause_event.clear()

    def toggle_pause(self):
        if not self.is_transcribing:
            return
            
        if self.is_paused:
            # Resume
            self.is_paused = False
            self.pause_event.clear()
            self.pause_btn.configure(text="Pause", fg_color="orange", hover_color="darkorange")
            self.log("▶️ Resumed.")
        else:
            # Pause
            self.is_paused = True
            self.pause_event.set()
            self.pause_btn.configure(text="Resume", fg_color="green", hover_color="darkgreen")
            self.log("⏸️ Paused.")

    def update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pause_btn.configure(state="normal", text="Pause", fg_color="orange")
            self.input_frame.configure(fg_color="gray90")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled", text="Pause", fg_color="orange")

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
