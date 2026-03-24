import sys
import subprocess
import os
import threading
import time
import queue
import json
import shutil
import urllib.request
import zipfile
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import librosa

# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

APP_NAME = "AudioWhisper"
APP_VERSION = "1.0.0"

MODELS = {
    "tiny":     "Fastest — low accuracy, good for quick drafts",
    "base":     "Fast — decent accuracy for clear audio",
    "small":    "Balanced — good accuracy, moderate speed",
    "medium":   "Accurate — slower, great for most use cases",
    "large-v3": "Best — highest accuracy, requires more RAM/VRAM",
}

SUPPORTED_FORMATS = "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac *.mp4 *.avi *.mov *.mkv *.webm"

PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
PYTHON_EMBED_DIR = "python-3.11.9-embed"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def get_app_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~/.config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def check_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _resource_path(filename):
    """Get the path to a bundled resource file (works frozen or not)."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


# ──────────────────────────────────────────────────────────
#  Dependency Manager
# ──────────────────────────────────────────────────────────

class DependencyManager:
    """Manages the embedded Python environment and heavy dependencies."""

    def __init__(self, app_data_dir):
        self.app_data_dir = app_data_dir
        self.python_dir = os.path.join(app_data_dir, PYTHON_EMBED_DIR)
        self.python_exe = os.path.join(self.python_dir, "python.exe")
        self.worker_path = os.path.join(app_data_dir, "transcribe_worker.py")
        self.marker_file = os.path.join(app_data_dir, "deps_installed.json")

    def is_installed(self):
        if not os.path.exists(self.marker_file):
            return False
        try:
            with open(self.marker_file) as f:
                data = json.load(f)
            return data.get("version") == APP_VERSION
        except Exception:
            return False

    def has_gpu(self):
        """Check for NVIDIA GPU by looking for nvidia-smi."""
        return shutil.which("nvidia-smi") is not None

    def get_python_exe(self):
        return self.python_exe

    def get_worker_path(self):
        return self.worker_path

    def install(self, progress_callback=None, status_callback=None):
        """Download and install the embedded Python + heavy deps."""
        def status(msg):
            if status_callback:
                status_callback(msg)

        def progress(pct):
            if progress_callback:
                progress_callback(pct)

        try:
            # Step 1: Download embedded Python
            status("Downloading Python runtime...")
            progress(0.0)
            zip_path = os.path.join(self.app_data_dir, "python_embed.zip")

            if not os.path.exists(self.python_exe):
                self._download_file(PYTHON_EMBED_URL, zip_path, progress, 0.0, 0.1)

                # Extract
                status("Extracting Python runtime...")
                os.makedirs(self.python_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.python_dir)
                os.remove(zip_path)

                # Enable site-packages by editing ._pth file
                for f in os.listdir(self.python_dir):
                    if f.endswith("._pth"):
                        pth_path = os.path.join(self.python_dir, f)
                        with open(pth_path, "r") as fh:
                            content = fh.read()
                        content = content.replace("#import site", "import site")
                        with open(pth_path, "w") as fh:
                            fh.write(content)
                        break

            progress(0.1)

            # Step 2: Install pip
            status("Setting up pip...")
            pip_exe = os.path.join(self.python_dir, "Scripts", "pip.exe")
            if not os.path.exists(pip_exe):
                get_pip_path = os.path.join(self.app_data_dir, "get-pip.py")
                self._download_file(GET_PIP_URL, get_pip_path, progress, 0.1, 0.15)
                subprocess.run(
                    [self.python_exe, get_pip_path, "--no-warn-script-location"],
                    cwd=self.python_dir,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if os.path.exists(get_pip_path):
                    os.remove(get_pip_path)

            progress(0.15)

            # Step 3: Install torch
            use_gpu = self.has_gpu()
            if use_gpu:
                status("Installing PyTorch with GPU support (this may take a while)...")
                torch_cmd = [
                    self.python_exe, "-m", "pip", "install",
                    "torch", "--no-warn-script-location",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ]
            else:
                status("Installing PyTorch (CPU)...")
                torch_cmd = [
                    self.python_exe, "-m", "pip", "install",
                    "torch", "--no-warn-script-location",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                ]

            self._run_pip_with_progress(torch_cmd, progress, 0.15, 0.75, status)

            # Step 4: Install faster-whisper and audio deps
            status("Installing transcription engine...")
            whisper_cmd = [
                self.python_exe, "-m", "pip", "install",
                "faster-whisper", "ffmpeg-python", "librosa", "soundfile",
                "--no-warn-script-location",
            ]
            self._run_pip_with_progress(whisper_cmd, progress, 0.75, 0.95, status)

            progress(0.95)

            # Step 5: Deploy worker script
            status("Finalizing setup...")
            worker_src = _resource_path("transcribe_worker.py")
            shutil.copy2(worker_src, self.worker_path)

            # Mark as complete
            with open(self.marker_file, "w") as f:
                json.dump({"version": APP_VERSION, "gpu": use_gpu}, f)

            progress(1.0)
            status("Setup complete!")
            return True

        except Exception as e:
            status(f"Installation failed: {e}")
            # Clean up partial install so next attempt starts fresh
            if os.path.exists(self.marker_file):
                os.remove(self.marker_file)
            return False

    def _download_file(self, url, dest, progress_cb, start_pct, end_pct):
        """Download a file with progress reporting."""
        req = urllib.request.Request(url, headers={"User-Agent": "AudioWhisper/1.0"})
        response = urllib.request.urlopen(req)
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 65536

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = start_pct + (end_pct - start_pct) * (downloaded / total)
                    progress_cb(pct)

    def _run_pip_with_progress(self, cmd, progress_cb, start_pct, end_pct, status_cb):
        """Run a pip command with progress tracking and no visible console."""
        # Use --progress-bar off for cleaner output parsing
        full_cmd = cmd + ["--progress-bar", "off"]

        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=self.python_dir,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        lines = []
        step_count = 0
        # Estimate total steps based on typical pip output
        # (Collecting + Downloading + Installing = ~3 lines per package)
        estimated_steps = 15  # rough guess, adjusts as we go

        for line in proc.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            lines.append(stripped)

            # Track meaningful pip events for progress
            is_step = False

            if stripped.startswith("Collecting"):
                # "Collecting torch" → show package name
                pkg_name = stripped.split("Collecting")[-1].strip().split()[0]
                status_cb(f"Downloading {pkg_name}...")
                is_step = True

            elif stripped.startswith("Downloading"):
                # "Downloading torch-2.4.0-cp311-..whl (150.3 MB)"
                size_match = re.search(r"\(([0-9.]+\s*[kKmMgG][bB])\)", stripped)
                pkg_match = re.search(r"Downloading\s+(\S+)", stripped)
                pkg_name = ""
                if pkg_match:
                    # Extract just the package name from the URL/filename
                    raw = pkg_match.group(1).split("/")[-1]
                    pkg_name = raw.split("-")[0]
                size_str = size_match.group(1) if size_match else ""
                if size_str:
                    status_cb(f"Downloading {pkg_name} ({size_str})...")
                else:
                    status_cb(f"Downloading {pkg_name}...")
                is_step = True

            elif stripped.startswith("Installing collected"):
                status_cb("Installing packages...")
                is_step = True

            elif stripped.startswith("Successfully installed"):
                status_cb("Packages installed")
                is_step = True

            if is_step:
                step_count += 1
                estimated_steps = max(estimated_steps, step_count + 2)
                pct = start_pct + (end_pct - start_pct) * min(step_count / estimated_steps, 0.95)
                progress_cb(pct)

        proc.wait()
        if proc.returncode != 0:
            error_lines = "\n".join(lines[-15:])
            raise RuntimeError(f"pip install failed:\n{error_lines}")
        progress_cb(end_pct)


# ──────────────────────────────────────────────────────────
#  Custom Widgets
# ──────────────────────────────────────────────────────────

class WaveformVisualizer(ctk.CTkCanvas):
    def __init__(self, master, width=600, height=60, bg_color=None):
        super().__init__(master, width=width, height=height, highlightthickness=0)
        self.configure(bg=bg_color or "gray20")
        self.bars = 100
        self.amplitudes = np.zeros(self.bars)
        self._width = width
        self._height = height
        self.bar_width = width / self.bars
        self.progress = 0.0

    def load_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=8000, duration=None)
            chunk_size = max(len(y) // self.bars, 1)
            new_amps = []
            for i in range(self.bars):
                start = i * chunk_size
                chunk = y[start:start + chunk_size]
                new_amps.append(np.max(np.abs(chunk)) if len(chunk) > 0 else 0)
            max_val = max(new_amps) if max(new_amps) > 0 else 1
            self.amplitudes = np.array(new_amps) / max_val
            self.draw()
        except Exception:
            pass

    def set_progress(self, progress):
        self.progress = progress
        self.draw()

    def draw(self):
        self.delete("all")
        for i, amp in enumerate(self.amplitudes):
            x1 = i * self.bar_width
            x2 = x1 + self.bar_width - 1
            bar_h = amp * self._height
            y1 = (self._height - bar_h) / 2
            y2 = y1 + bar_h
            color = "#00E676" if (i / self.bars) < self.progress else "gray40"
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def reset(self):
        self.amplitudes = np.zeros(self.bars)
        self.progress = 0.0
        self.draw()


class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title="Advanced Settings"):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.is_expanded = False

        self.title_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.title_frame.grid(row=0, column=0, sticky="ew")
        self.title_frame.grid_columnconfigure(0, weight=1)

        self.toggle_btn = ctk.CTkButton(
            self.title_frame, text=f"▶ {title}", width=100, anchor="w",
            fg_color="transparent", text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"), command=self.toggle
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
            text="Drop Audio or Video File Here\n\nor click to browse",
            font=("Segoe UI", 15),
            text_color=("gray50", "gray70"),
        )
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        if self.command:
            self.command()

    def set_file(self, filename):
        self.label.configure(
            text=f"{os.path.basename(filename)}\n\nReady to transcribe"
        )
        self.configure(fg_color=("#c8e6c9", "#2e7d32"))

    def clear(self):
        self.label.configure(
            text="Drop Audio or Video File Here\n\nor click to browse"
        )
        self.configure(fg_color=("gray85", "gray25"))


# ──────────────────────────────────────────────────────────
#  Dependency Install Dialog
# ──────────────────────────────────────────────────────────

class InstallDialog(ctk.CTkToplevel):
    """Shows progress while installing dependencies."""

    def __init__(self, parent, dep_manager):
        super().__init__(parent)
        self.title(f"{APP_NAME} — Installing")
        self.geometry("500x280")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.dep_manager = dep_manager
        self.success = False

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = (self.winfo_screenheight() - 280) // 2
        self.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            self, text="Setting up AudioWhisper",
            font=("Segoe UI", 20, "bold"),
        ).pack(pady=(25, 5))

        ctk.CTkLabel(
            self,
            text="Downloading and installing the transcription engine.\nThis only happens once.",
            font=("Segoe UI", 12), text_color="gray60", justify="center",
        ).pack(pady=(0, 20))

        self.status_var = ctk.StringVar(value="Preparing...")
        ctk.CTkLabel(
            self, textvariable=self.status_var,
            font=("Segoe UI", 12),
        ).pack(pady=(0, 10))

        self.progress_var = ctk.DoubleVar(value=0.0)
        self.progress_bar = ctk.CTkProgressBar(self, variable=self.progress_var, width=400)
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar.set(0)

        self.pct_label = ctk.CTkLabel(
            self, text="0%", font=("Segoe UI", 11, "bold"), text_color="gray60",
        )
        self.pct_label.pack()

        self.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing during install

        self._msg_queue = queue.Queue()
        self.after(100, self._poll_queue)
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        def on_progress(pct):
            self._msg_queue.put(("progress", pct))

        def on_status(msg):
            self._msg_queue.put(("status", msg))

        result = self.dep_manager.install(
            progress_callback=on_progress,
            status_callback=on_status,
        )
        self._msg_queue.put(("done", result))

    def _poll_queue(self):
        while not self._msg_queue.empty():
            kind, value = self._msg_queue.get()
            if kind == "progress":
                self.progress_var.set(value)
                self.pct_label.configure(text=f"{int(value * 100)}%")
            elif kind == "status":
                self.status_var.set(value)
            elif kind == "done":
                self.success = value
                self.destroy()
                return
        self.after(100, self._poll_queue)


# ──────────────────────────────────────────────────────────
#  First-Run Setup Dialog
# ──────────────────────────────────────────────────────────

class SetupDialog(ctk.CTkToplevel):
    """Shown on first launch to welcome the user and pick defaults."""

    def __init__(self, parent, settings, has_gpu):
        super().__init__(parent)
        self.title(f"{APP_NAME} — Setup")
        self.geometry("520x480")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.settings = settings
        self.result = None

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 520) // 2
        y = (self.winfo_screenheight() - 480) // 2
        self.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            self, text=f"Welcome to {APP_NAME}",
            font=("Segoe UI", 22, "bold"),
        ).pack(pady=(30, 5))

        ctk.CTkLabel(
            self,
            text="Free, private, offline audio & video transcription.\nEverything runs on your machine — nothing is uploaded.",
            font=("Segoe UI", 13), text_color="gray60", justify="center",
        ).pack(pady=(0, 20))

        device = "NVIDIA GPU (CUDA)" if has_gpu else "CPU"
        device_color = "#4CAF50" if has_gpu else "#FF9800"
        ctk.CTkLabel(
            self, text=f"Detected hardware:  {device}",
            font=("Segoe UI", 13, "bold"), text_color=device_color,
        ).pack(pady=(0, 20))

        ctk.CTkLabel(
            self, text="Choose a default model size:",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=40)

        self.model_var = ctk.StringVar(value=settings.get("model", "base"))
        for name, desc in MODELS.items():
            ctk.CTkRadioButton(
                self, text=f"{name}  —  {desc}",
                variable=self.model_var, value=name, font=("Segoe UI", 12),
            ).pack(anchor="w", padx=60, pady=2)

        ctk.CTkLabel(
            self,
            text="The model will be downloaded automatically on first use.\nYou can change this later in Advanced Settings.",
            font=("Segoe UI", 11), text_color="gray50", justify="center",
        ).pack(pady=(15, 10))

        if not check_ffmpeg():
            ctk.CTkLabel(
                self,
                text="FFmpeg not found — video files won't work.\n"
                     "Install it:  winget install ffmpeg",
                font=("Segoe UI", 12, "bold"), text_color="#F44336", justify="center",
            ).pack(pady=(5, 5))

        ctk.CTkButton(
            self, text="Get Started", width=200, height=40,
            font=("Segoe UI", 14, "bold"),
            fg_color="#4CAF50", hover_color="#388E3C",
            command=self._finish,
        ).pack(pady=(10, 20))

        self.protocol("WM_DELETE_WINDOW", self._finish)

    def _finish(self):
        self.result = self.model_var.get()
        self.destroy()


# ──────────────────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────────────────

class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class AudioWhisperApp(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("800x900")
        self.minsize(700, 700)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Paths & managers
        self.app_data_dir = get_app_data_dir()
        self.settings_file = os.path.join(self.app_data_dir, "settings.json")
        self.dep_manager = DependencyManager(self.app_data_dir)

        # State
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.model_name = ctk.StringVar(value="base")
        self.show_timestamps = ctk.BooleanVar(value=True)
        self.export_srt = ctk.BooleanVar(value=False)
        self.status_msg = ctk.StringVar(value="Ready")
        self.progress_val = ctk.DoubleVar(value=0.0)
        self.time_remaining_msg = ctk.StringVar(value="")

        self.is_transcribing = False
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self._worker_proc = None

        # Load settings
        self.settings = self._load_settings()
        self.model_name.set(self.settings.get("model", "base"))
        self.output_dir.set(self.settings.get("output_dir", ""))
        self.show_timestamps.set(self.settings.get("timestamps", True))
        self.export_srt.set(self.settings.get("export_srt", False))

        # Device detection (lightweight — just check nvidia-smi)
        self.has_gpu = self.dep_manager.has_gpu()
        self.device = "cuda" if self.has_gpu else "cpu"
        self.compute_type = "float16" if self.has_gpu else "float32"

        # Build UI
        self._create_widgets()

        # Drag & drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._drop_file)

        # Log polling
        self._poll_log_queue()

        # First-run setup
        if not self.settings.get("setup_complete"):
            self.after(200, self._show_setup)

    # ── Settings ────────────────────────────────────────

    def _load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_settings(self):
        data = {
            "model": self.model_name.get(),
            "output_dir": self.output_dir.get(),
            "timestamps": self.show_timestamps.get(),
            "export_srt": self.export_srt.get(),
            "setup_complete": True,
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ── First-run ───────────────────────────────────────

    def _show_setup(self):
        dialog = SetupDialog(self, self.settings, self.has_gpu)
        self.wait_window(dialog)
        if dialog.result:
            self.model_name.set(dialog.result)
        self._save_settings()

        # Install dependencies if needed
        if not self.dep_manager.is_installed():
            self._install_deps()

    def _install_deps(self):
        dialog = InstallDialog(self, self.dep_manager)
        self.wait_window(dialog)
        if not dialog.success:
            messagebox.showerror(
                APP_NAME,
                "Dependency installation failed. Please check your internet connection and try again.",
            )

    # ── UI ──────────────────────────────────────────────

    def _create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        ctk.CTkLabel(
            header, text=APP_NAME, font=("Segoe UI", 24, "bold"),
        ).pack(side="left")

        self.theme_btn = ctk.CTkButton(
            header, text="🌗", width=40, command=self._toggle_theme,
        )
        self.theme_btn.pack(side="right", padx=10)

        device_color = "#4CAF50" if self.has_gpu else "#FF9800"
        device_text = "GPU" if self.has_gpu else "CPU"
        ctk.CTkLabel(
            header, text=device_text,
            text_color=device_color, font=("Segoe UI", 12, "bold"),
        ).pack(side="right")

        # Drop zone
        self.drop_zone = DropZone(self, command=self._browse_input)
        self.drop_zone.grid(row=1, column=0, padx=20, pady=10, sticky="ew", ipady=30)

        # Waveform visualizer
        viz_frame = ctk.CTkFrame(self, fg_color="transparent")
        viz_frame.grid(row=2, column=0, padx=20, pady=0, sticky="ew")
        self.visualizer = WaveformVisualizer(viz_frame, width=760, height=60, bg_color="#2B2B2B")
        self.visualizer.pack(fill="x")

        # Progress
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        prog_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(prog_frame, variable=self.progress_val)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.progress_bar.set(0)

        status_row = ctk.CTkFrame(prog_frame, fg_color="transparent")
        status_row.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(
            status_row, textvariable=self.status_msg,
            font=("Segoe UI", 12), text_color="gray60",
        ).pack(side="left")
        ctk.CTkLabel(
            status_row, textvariable=self.time_remaining_msg,
            font=("Segoe UI", 12, "bold"), text_color="gray60",
        ).pack(side="right")

        # Controls
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        controls.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(controls, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(
            btn_row, text="Start Transcription",
            fg_color="#4CAF50", hover_color="#388E3C",
            height=40, font=("Segoe UI", 14, "bold"),
            command=self._start_transcription,
        )
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop",
            fg_color="#F44336", hover_color="#E57373",
            height=40, state="disabled", command=self._stop_transcription,
        )
        self.stop_btn.grid(row=0, column=1, padx=5, sticky="ew")

        # Collapsible settings
        self.settings_group = CollapsibleFrame(controls, title="Advanced Settings")
        self.settings_group.grid(row=1, column=0, sticky="ew")

        sf = self.settings_group.content_frame
        ctk.CTkLabel(sf, text="Output Folder:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(sf, textvariable=self.output_dir, width=300).grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkButton(sf, text="Browse", width=60, command=self._browse_output).grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkLabel(sf, text="Model:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkComboBox(
            sf, values=list(MODELS.keys()), variable=self.model_name,
        ).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkSwitch(sf, text="Timestamps", variable=self.show_timestamps).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkSwitch(sf, text="Export .SRT", variable=self.export_srt).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Transcript
        tx_frame = ctk.CTkFrame(self, fg_color="transparent")
        tx_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="nsew")
        tx_frame.grid_columnconfigure(0, weight=1)
        tx_frame.grid_rowconfigure(0, weight=1)

        self.transcript_box = ctk.CTkTextbox(tx_frame, font=("Consolas", 14))
        self.transcript_box.grid(row=0, column=0, sticky="nsew")
        self.transcript_box.insert("1.0", "Your transcription will appear here.\n")
        self.transcript_box.configure(state="disabled")

        tools_row = ctk.CTkFrame(tx_frame, fg_color="transparent")
        tools_row.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ctk.CTkButton(
            tools_row, text="Copy Text", width=100, command=self._copy_text,
        ).pack(side="left")
        self.open_folder_btn = ctk.CTkButton(
            tools_row, text="Open Folder", width=100,
            state="disabled", command=self._open_output_folder,
        )
        self.open_folder_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Actions ─────────────────────────────────────────

    def _toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
            self.visualizer.configure(bg="gray90")
        else:
            ctk.set_appearance_mode("Dark")
            self.visualizer.configure(bg="#2B2B2B")

    def _drop_file(self, event):
        path = event.data
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        self.input_path.set(path)
        self.drop_zone.set_file(path)
        threading.Thread(target=self.visualizer.load_audio, args=(path,), daemon=True).start()

    def _browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[("Media Files", SUPPORTED_FORMATS)]
        )
        if path:
            self.input_path.set(path)
            self.drop_zone.set_file(path)
            threading.Thread(target=self.visualizer.load_audio, args=(path,), daemon=True).start()

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)

    def _copy_text(self):
        text = self.transcript_box.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_msg.set("Copied to clipboard!")

    def _open_output_folder(self):
        path = self.output_dir.get()
        if path and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showerror("Error", "Output folder not found.")

    def _on_close(self):
        self._save_settings()
        if self._worker_proc and self._worker_proc.poll() is None:
            self._worker_proc.terminate()
        self.destroy()

    # ── Logging ─────────────────────────────────────────

    def _log(self, message, is_transcript=False):
        self.log_queue.put((message, is_transcript))

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            msg, is_transcript = self.log_queue.get()
            if is_transcript:
                self.transcript_box.configure(state="normal")
                self.transcript_box.insert("end", msg + "\n")
                self.transcript_box.see("end")
                self.transcript_box.configure(state="disabled")
            else:
                self.status_msg.set(msg)
        self.after(100, self._poll_log_queue)

    # ── Transcription (subprocess) ──────────────────────

    def _start_transcription(self):
        if not self.input_path.get():
            messagebox.showwarning(APP_NAME, "Please select a file first.")
            return

        # Ensure deps are installed
        if not self.dep_manager.is_installed():
            self._install_deps()
            if not self.dep_manager.is_installed():
                return

        # Check FFmpeg for video files
        is_video = self.input_path.get().lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        if is_video and not check_ffmpeg():
            messagebox.showwarning(
                APP_NAME,
                "FFmpeg is required for video files.\nInstall it: winget install ffmpeg",
            )
            return

        self.is_transcribing = True
        self.stop_event.clear()
        self.progress_val.set(0)
        self.visualizer.set_progress(0)
        self.time_remaining_msg.set("Calculating...")
        self._update_ui_state(transcribing=True)
        self._save_settings()

        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.configure(state="disabled")

        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        """Launch the transcription worker as a subprocess."""
        python_exe = self.dep_manager.get_python_exe()
        worker_path = self.dep_manager.get_worker_path()

        cmd = [
            python_exe, worker_path,
            "--input", self.input_path.get(),
            "--model", self.model_name.get(),
            "--device", self.device,
            "--compute_type", self.compute_type,
        ]
        if self.output_dir.get():
            cmd.extend(["--output_dir", self.output_dir.get()])
        if self.show_timestamps.get():
            cmd.append("--timestamps")
        if self.export_srt.get():
            cmd.append("--export_srt")

        try:
            self._worker_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            for line in self._worker_proc.stdout:
                if self.stop_event.is_set():
                    self._worker_proc.terminate()
                    self._log("Stopped.")
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                if msg_type == "status":
                    self._log(msg["msg"])
                elif msg_type == "segment":
                    self._log(f"{msg['timestamp']} {msg['text']}", is_transcript=True)
                elif msg_type == "progress":
                    self.progress_val.set(msg["value"])
                    self.visualizer.set_progress(msg["value"])
                    if msg.get("eta"):
                        self.time_remaining_msg.set(msg["eta"])
                elif msg_type == "done":
                    self._log(msg["msg"])
                    self.progress_val.set(1.0)
                    self.visualizer.set_progress(1.0)
                    self.time_remaining_msg.set("Complete")
                    self.open_folder_btn.configure(state="normal")
                elif msg_type == "error":
                    self._log(f"Error: {msg['msg']}")

            self._worker_proc.wait()

        except Exception as e:
            self._log(f"Error: {e}")
        finally:
            self._worker_proc = None
            self.is_transcribing = False
            self._update_ui_state(transcribing=False)

    def _stop_transcription(self):
        if self.is_transcribing:
            self.stop_event.set()
            if self._worker_proc and self._worker_proc.poll() is None:
                self._worker_proc.terminate()

    def _update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.open_folder_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")


# ──────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
