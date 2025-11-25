import sys
import subprocess
import importlib
import os
import threading
import time
import queue
import io
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

def check_torch_installation():
    """Check if torch is installed and if it matches the hardware."""
    print("Checking hardware compatibility...")
    gpu_type = get_gpu_info()
    print(f"Hardware Detection Result: {gpu_type.upper()} (based on GPU check)")

    try:
        import torch
        installed_version = torch.__version__
        print(f"Current PyTorch Version: {installed_version}")
        
        needs_reinstall = False
        
        if gpu_type == "blackwell":
            # RTX 50-series needs Nightly with CUDA 12.8+
            # Check for 'cu128' or newer in version string
            if "cu128" not in installed_version and "cu129" not in installed_version and "cu13" not in installed_version:
                print("⚠️ Incompatible PyTorch for RTX 50-series detected (Need CUDA 12.8+).")
                needs_reinstall = True
        elif gpu_type == "standard":
            # Standard GPU needs CUDA support
            if "+cu" not in installed_version:
                print("⚠️ CPU-only PyTorch detected on GPU machine.")
                needs_reinstall = True
        
        if needs_reinstall:
            print("🔄 Initiating Auto-Reinstall...")
            install_torch(gpu_type)
        else:
            print("✅ PyTorch version looks good.")
            
    except ImportError:
        print("⚠️ PyTorch not found.")
        install_torch(gpu_type)

def install_torch(gpu_type):
    """Install the correct PyTorch version based on GPU type."""
    print("---------------------------------------------------")
    print("       AudioWhisper Smart Installer")
    print("---------------------------------------------------")
    print("Uninstalling existing PyTorch...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    
    print(f"Installing optimized PyTorch for {gpu_type.upper()}...")
    
    if gpu_type == "blackwell":
        # Install Nightly with CUDA 12.8
        cmd = [sys.executable, "-m", "pip", "install", "--pre", "torch", "--index-url", "https://download.pytorch.org/whl/nightly/cu128"]
    elif gpu_type == "standard":
        # Install Stable with CUDA 12.4 (or 12.1 depending on availability, 12.4 is safe bet for 30/40 series)
        # Note: Python 3.13 might force us to Nightly even for standard GPUs, but let's try stable for standard if Python < 3.13
        # For now, to be safe and support Python 3.13 everywhere, we might default to Nightly for all CUDA users on 3.13
        if sys.version_info >= (3, 13):
             cmd = [sys.executable, "-m", "pip", "install", "--pre", "torch", "--index-url", "https://download.pytorch.org/whl/nightly/cu124"]
        else:
             cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"]
    else:
        # CPU Only
        cmd = [sys.executable, "-m", "pip", "install", "torch"]
        
    try:
        subprocess.check_call(cmd)
        print("✅ Installation Complete! Restarting application...")
        # Auto-restart the script
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation Failed: {e}")
        print("Falling back to CPU mode...")
        subprocess.call([sys.executable, "-m", "pip", "install", "torch"])

# Run the smart check BEFORE importing torch/whisper
check_torch_installation()

# ---------------- Other Dependencies ---------------- #
REQUIRED_PACKAGES = [
    "customtkinter",
    "openai-whisper",
    "ffmpeg-python",
    "librosa",
    "soundfile",
    "packaging"
]

for package in REQUIRED_PACKAGES:
    try:
        import_name = package
        if package == "openai-whisper": import_name = "whisper"
        elif package == "ffmpeg-python": import_name = "ffmpeg"
        elif package == "customtkinter": import_name = "customtkinter"
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ---------------- Imports ---------------- #
import customtkinter as ctk
import whisper
import ffmpeg
import librosa
import soundfile as sf
import torch

# ---------------- Configuration ---------------- #
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class StdoutRedirector:
    """Redirects stdout to a callback function."""
    def __init__(self, callback):
        self.callback = callback

    def write(self, message):
        if message.strip():
            self.callback(message.strip())

    def flush(self):
        pass

class AudioWhisperApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("AudioWhisper v5.0 (Smart Installer)")
        self.geometry("700x650")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # State Variables
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.model_name = ctk.StringVar(value="base")
        self.show_timestamps = ctk.BooleanVar(value=True)
        self.status_msg = ctk.StringVar(value="Ready")
        self.is_transcribing = False
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()

        # Device Detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_label_text = f"Device: {self.device.upper()} {'🟢' if self.device == 'cuda' else '🔴'}"
        self.device_color = "green" if self.device == "cuda" else "orange"

        # UI Layout
        self.create_widgets()
        
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
        ctk.CTkEntry(self.input_frame, textvariable=self.input_path).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
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
        self.model_combo = ctk.CTkComboBox(self.settings_frame, values=["tiny", "base", "small", "medium", "large"], variable=self.model_name)
        self.model_combo.grid(row=0, column=1, padx=10, pady=10)

        self.timestamp_switch = ctk.CTkSwitch(self.settings_frame, text="Save with Timestamps", variable=self.show_timestamps)
        self.timestamp_switch.grid(row=0, column=2, padx=20, pady=10)

        # --- Controls ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(self.controls_frame, text="Start Transcription", fg_color="green", hover_color="darkgreen", command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.controls_frame, text="Stop", fg_color="red", hover_color="darkred", state="disabled", command=self.stop_transcription)
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # --- Log Console ---
        self.log_box = ctk.CTkTextbox(self, width=600, height=200)
        self.log_box.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.log_box.configure(state="disabled")

        # --- Status Bar ---
        self.status_label = ctk.CTkLabel(self, textvariable=self.status_msg, anchor="w")
        self.status_label.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

    # ---------------- Logic ---------------- #

    def log(self, message):
        """Thread-safe logging."""
        self.log_queue.put(message)

    def check_log_queue(self):
        """Poll the log queue and update UI."""
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

    def extract_audio(self, video_path):
        """Extract audio from video using ffmpeg."""
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

        temp_file = None
        start_time = time.time()

        # Redirect stdout to capture Whisper's real-time output
        original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.log)

        try:
            self.log(f"Loading Whisper model: {model_type} on {self.device.upper()}...")
            
            # Load model with specific device
            try:
                model = whisper.load_model(model_type, device=self.device)
            except RuntimeError as e:
                if "CUDA error" in str(e) and self.device == "cuda":
                    self.log(f"⚠️ GPU Error detected: {e}")
                    self.log("⚠️ Falling back to CPU mode...")
                    self.device = "cpu"
                    self.device_label.configure(text="Device: CPU (Fallback) 🔴", text_color="orange")
                    model = whisper.load_model(model_type, device=self.device)
                else:
                    raise e

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
            
            # Transcribe
            # If CPU, force fp16=False to avoid warnings. If GPU, let it default (usually True for speed).
            fp16_option = False if self.device == "cpu" else True
            
            result = model.transcribe(process_file, verbose=True, fp16=fp16_option)

            if self.stop_event.is_set(): return

            # Save Output
            output_filename = "Transcription.txt"
            output_path = os.path.join(output_folder, output_filename)
            output_path = self.get_unique_filename(output_path)

            with open(output_path, "w", encoding="utf-8") as f:
                if use_timestamps:
                    for segment in result["segments"]:
                        start = segment["start"]
                        timestamp = f"[{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02}]"
                        text = segment["text"].strip()
                        line = f"{timestamp} {text}"
                        f.write(line + "\n")
                else:
                    f.write(result["text"].strip())

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            self.log(f"✅ Transcription saved to: {output_path}")
            self.log(f"⏱ Finished in {mins}m {secs}s")

        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
            # Cleanup
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                self.log("Temporary audio file removed.")
            
            self.stop_event.clear()
            self.is_transcribing = False
            self.update_ui_state(transcribing=False)

    def start_transcription(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file.")
            return
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        
        self.is_transcribing = True
        self.stop_event.clear()
        self.update_ui_state(transcribing=True)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        
        threading.Thread(target=self.run_transcription, daemon=True).start()

    def stop_transcription(self):
        if self.is_transcribing:
            self.log("🛑 Stopping transcription...")
            self.stop_event.set()

    def update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.input_frame.configure(fg_color="gray90")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
