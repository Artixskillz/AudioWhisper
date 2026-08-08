import sys
import subprocess
import os
import threading
import time
import queue
import json
import shutil
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

APP_NAME = "AudioWhisper"
APP_VERSION = "1.2.0"

MODELS = {
    "tiny":     "Fastest — low accuracy, good for quick drafts (75 MB)",
    "base":     "Fast — decent accuracy for clear audio (145 MB)",
    "small":    "Balanced — good accuracy, moderate speed (480 MB)",
    "medium":   "Accurate — slower, great for most use cases (1.5 GB)",
    "large-v3": "Best — highest accuracy, needs a powerful PC (3 GB)",
}

SUPPORTED_FORMATS = "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac *.mp4 *.avi *.mov *.mkv *.webm"

# Legacy v1.0 runtime location (Roaming AppData)
PYTHON_EMBED_DIR = "python-3.11.9-embed"

# Optional GPU acceleration packages (installed on demand into the runtime)
GPU_PACKAGES = ["nvidia-cublas-cu12==12.9.2.10", "nvidia-cudnn-cu12==9.24.0.43"]

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def get_app_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~/.config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)

    # One-time migration of settings from the v1.0 Roaming location
    if sys.platform == "win32":
        new_settings = os.path.join(path, "settings.json")
        old_settings = os.path.join(
            os.environ.get("APPDATA", ""), APP_NAME, "settings.json")
        if not os.path.exists(new_settings) and os.path.exists(old_settings):
            try:
                shutil.copy2(old_settings, new_settings)
            except Exception:
                pass
    return path


def _resource_path(filename):
    """Get the path to a bundled resource file (works frozen or not)."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


def _exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────
#  Runtime Manager
# ──────────────────────────────────────────────────────────

class RuntimeManager:
    """Locates the bundled Python runtime and manages the optional GPU add-on.

    The transcription engine ships inside the installer (in the `runtime`
    folder next to the EXE), so there is nothing to download on first run
    except the Whisper model itself.
    """

    def __init__(self, app_data_dir):
        self.app_data_dir = app_data_dir
        candidates = [
            os.path.join(_exe_dir(), "runtime"),
            # Legacy v1.0 install location
            os.path.join(os.environ.get("APPDATA", ""), APP_NAME, PYTHON_EMBED_DIR),
        ]
        self.python_dir = next(
            (c for c in candidates if os.path.exists(os.path.join(c, "python.exe"))),
            candidates[0],
        )
        self.python_exe = os.path.join(self.python_dir, "python.exe")
        self.is_bundled = self.python_dir == candidates[0]
        self.gpu_marker = os.path.join(app_data_dir, "gpu_installed.json")
        self.install_log = os.path.join(app_data_dir, "install.log")
        self.whispercpp_dir = os.path.join(self.python_dir, "whispercpp")
        self.gpu_auto_flag = os.path.join(_exe_dir(), "gpu_auto.flag")
        self._gpu_names = None  # filled by detect_gpus_async
        self._pip_proc = None

    # ── GPU detection ───────────────────────────────────

    def detect_gpus_async(self, on_done=None):
        """Enumerate display adapters in the background (WMI is slow)."""
        def _detect():
            names = []
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_VideoController | "
                     "Select-Object -ExpandProperty Name"],
                    capture_output=True, text=True, timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                skip = ("basic display", "basic render", "remote", "virtual", "parsec")
                names = [n.strip() for n in r.stdout.splitlines()
                         if n.strip() and not any(s in n.lower() for s in skip)]
            except Exception:
                pass
            self._gpu_names = names
            if on_done:
                on_done()

        threading.Thread(target=_detect, daemon=True).start()

    def vulkan_available(self):
        """whisper.cpp Vulkan engine works on any real GPU (AMD/Intel/NVIDIA)."""
        return (bool(self._gpu_names)
                and os.path.isfile(os.path.join(self.whispercpp_dir, "whisper-cli.exe")))

    def pick_backend(self):
        """Best available engine: CUDA > Vulkan > CPU."""
        if self.gpu_ready() and self.has_gpu():
            return "cuda"
        if self.vulkan_available():
            return "vulkan"
        return "cpu"

    def consume_auto_flag(self):
        """Installer task marker: auto-start the CUDA download on first launch."""
        if os.path.exists(self.gpu_auto_flag):
            try:
                os.remove(self.gpu_auto_flag)
            except OSError:
                pass
            return True
        return False

    def kill_pip(self):
        """Kill any in-flight pip install (called on app close)."""
        proc = self._pip_proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def runtime_ok(self):
        return os.path.exists(self.python_exe)

    def has_gpu(self):
        """Check for NVIDIA GPU by looking for nvidia-smi."""
        return shutil.which("nvidia-smi") is not None

    def _nvidia_dir(self):
        return os.path.join(self.python_dir, "Lib", "site-packages", "nvidia")

    def gpu_dll_dirs(self):
        base = self._nvidia_dir()
        return [os.path.join(base, "cublas", "bin"),
                os.path.join(base, "cudnn", "bin")]

    def gpu_ready(self):
        if not os.path.exists(self.gpu_marker):
            return False
        return all(os.path.isdir(d) for d in self.gpu_dll_dirs())

    def worker_env(self):
        """Environment for worker/probe subprocesses."""
        env = os.environ.copy()
        env["HF_HOME"] = os.path.join(self.app_data_dir, "models")
        env["PYTHONIOENCODING"] = "utf-8"
        if self.gpu_ready():
            env["PATH"] = os.pathsep.join(self.gpu_dll_dirs()) + os.pathsep + env.get("PATH", "")
        return env

    # ── GPU add-on installation ─────────────────────────

    def install_gpu(self, progress_cb, status_cb, cancel_event):
        """Install CUDA libraries into the runtime. Returns (ok, error_msg)."""
        log_lines = []
        try:
            free_gb = shutil.disk_usage(self.python_dir).free / (1024 ** 3)
            if free_gb < 4:
                raise RuntimeError(
                    f"Not enough disk space — need about 4 GB free, "
                    f"but only {free_gb:.1f} GB is available.")

            status_cb("Downloading GPU libraries...")
            cmd = [
                self.python_exe, "-m", "pip", "install",
                "--no-cache-dir", "--no-warn-script-location",
            ] + GPU_PACKAGES
            self._run_pip(cmd, progress_cb, status_cb, cancel_event, log_lines)

            if cancel_event.is_set():
                return False, "cancelled"

            with open(self.gpu_marker, "w") as f:
                json.dump({"packages": GPU_PACKAGES, "version": APP_VERSION}, f)
            progress_cb(1.0)
            status_cb("GPU acceleration ready!")
            return True, ""

        except Exception as e:
            if cancel_event.is_set():
                return False, "cancelled"
            self._write_log(log_lines, e)
            return False, str(e)

    def _run_pip(self, cmd, progress_cb, status_cb, cancel_event, log_lines):
        """Run a pip command with progress tracking and no visible console."""
        full_cmd = cmd + ["--progress-bar", "off"]

        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=self.python_dir,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._pip_proc = proc

        # pip is silent for minutes during large wheel downloads, so the
        # stdout loop can't see the cancel flag — a watchdog kills pip
        # the moment cancel is requested
        def _watchdog():
            while proc.poll() is None:
                if cancel_event.wait(0.5):
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return

        threading.Thread(target=_watchdog, daemon=True).start()

        step_count = 0
        estimated_steps = 8

        for line in proc.stdout:
            if cancel_event.is_set():
                proc.wait()
                self._pip_proc = None
                return
            stripped = line.strip()
            if not stripped:
                continue
            log_lines.append(stripped)

            is_step = False
            if stripped.startswith("Collecting"):
                pkg_name = stripped.split("Collecting")[-1].strip().split()[0]
                status_cb(f"Downloading {pkg_name}...")
                is_step = True
            elif stripped.startswith("Downloading"):
                size_match = re.search(r"\(([0-9.]+\s*[kKmMgG][bB])\)", stripped)
                pkg_match = re.search(r"Downloading\s+(\S+)", stripped)
                pkg_name = ""
                if pkg_match:
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
                progress_cb(min(step_count / estimated_steps, 0.95))

        proc.wait()
        self._pip_proc = None
        if proc.returncode != 0 and not cancel_event.is_set():
            error_lines = "\n".join(log_lines[-15:])
            raise RuntimeError(f"pip install failed:\n{error_lines}")

    def _write_log(self, log_lines, error):
        try:
            with open(self.install_log, "w", encoding="utf-8") as f:
                f.write(f"{APP_NAME} {APP_VERSION} — GPU install log\n")
                f.write(f"Error: {error}\n\n")
                f.write("\n".join(log_lines))
        except Exception:
            pass

    # ── Legacy v1.0 cleanup ─────────────────────────────

    def legacy_runtime_dir(self):
        """Return the old v1.0 runtime dir if it exists and isn't in use."""
        if not self.is_bundled:
            return None
        legacy = os.path.join(os.environ.get("APPDATA", ""), APP_NAME, PYTHON_EMBED_DIR)
        return legacy if os.path.isdir(legacy) else None

    def remove_legacy_runtime(self):
        legacy_root = os.path.join(os.environ.get("APPDATA", ""), APP_NAME)
        for name in (PYTHON_EMBED_DIR, "deps_installed.json", "transcribe_worker.py"):
            target = os.path.join(legacy_root, name)
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                elif os.path.exists(target):
                    os.remove(target)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────
#  Custom Widgets
# ──────────────────────────────────────────────────────────

class WaveformVisualizer(ctk.CTkCanvas):
    def __init__(self, master, width=600, height=60, bg_color=None):
        super().__init__(master, width=width, height=height, highlightthickness=0)
        self.configure(bg=bg_color or "gray20")
        self.bars = 100
        self.amplitudes = [0.0] * self.bars
        self._width = width
        self._height = height
        self.bar_width = width / self.bars
        self.progress = 0.0

    def set_amplitudes(self, peaks):
        """Set waveform peaks (list of 0..1 floats) and redraw."""
        amps = list(peaks)[:self.bars]
        amps += [0.0] * (self.bars - len(amps))
        self.amplitudes = amps
        self.draw()

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
        self.amplitudes = [0.0] * self.bars
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

    def set_file(self, filename, info=""):
        detail = info if info else "Ready to transcribe"
        self.label.configure(
            text=f"{os.path.basename(filename)}\n\n{detail}"
        )
        self.configure(fg_color=("#c8e6c9", "#2e7d32"))

    def clear(self):
        self.label.configure(
            text="Drop Audio or Video File Here\n\nor click to browse"
        )
        self.configure(fg_color=("gray85", "gray25"))


# ──────────────────────────────────────────────────────────
#  GPU Install Dialog
# ──────────────────────────────────────────────────────────

class GpuInstallDialog(ctk.CTkToplevel):
    """Shows progress while installing the optional GPU acceleration."""

    def __init__(self, parent, runtime_mgr):
        super().__init__(parent)
        self.title(f"{APP_NAME} — GPU Setup")
        self.geometry("500x300")
        self.resizable(False, False)
        try:
            self.after(200, lambda: self.iconbitmap(_resource_path("AudioWhisper.ico")))
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()
        self.runtime_mgr = runtime_mgr
        self.success = False
        self.error = ""
        self._cancel_event = threading.Event()

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 500) // 2
        y = (self.winfo_screenheight() - 300) // 2
        self.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            self, text="Setting up GPU Acceleration",
            font=("Segoe UI", 20, "bold"),
        ).pack(pady=(25, 5))

        ctk.CTkLabel(
            self,
            text="Downloading NVIDIA CUDA libraries (~1 GB).\nThis only happens once and unlocks the fastest engine.",
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

        self.cancel_btn = ctk.CTkButton(
            self, text="Cancel", width=100,
            fg_color="gray40", hover_color="gray30",
            command=self._cancel,
        )
        self.cancel_btn.pack(pady=(10, 0))

        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._msg_queue = queue.Queue()
        self.after(100, self._poll_queue)
        threading.Thread(target=self._run_install, daemon=True).start()

    def _cancel(self):
        self._cancel_event.set()
        self.status_var.set("Cancelling...")
        self.cancel_btn.configure(state="disabled")

    def _run_install(self):
        def on_progress(pct):
            self._msg_queue.put(("progress", pct))

        def on_status(msg):
            self._msg_queue.put(("status", msg))

        ok, err = self.runtime_mgr.install_gpu(
            on_progress, on_status, self._cancel_event)
        self._msg_queue.put(("done", (ok, err)))

    def _poll_queue(self):
        while not self._msg_queue.empty():
            kind, value = self._msg_queue.get()
            if kind == "progress":
                self.progress_var.set(value)
                self.pct_label.configure(text=f"{int(value * 100)}%")
            elif kind == "status":
                self.status_var.set(value)
            elif kind == "done":
                self.success, self.error = value
                self.destroy()
                return
        self.after(100, self._poll_queue)


# ──────────────────────────────────────────────────────────
#  First-Run Setup Dialog
# ──────────────────────────────────────────────────────────

class SetupDialog(ctk.CTkToplevel):
    """Shown on first launch to welcome the user and pick defaults."""

    def __init__(self, parent, settings, offer_gpu):
        super().__init__(parent)
        self.title(f"{APP_NAME} — Setup")
        self.geometry("540x520")
        self.resizable(False, False)
        try:
            self.after(200, lambda: self.iconbitmap(_resource_path("AudioWhisper.ico")))
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()
        self.settings = settings
        self.result = None

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 540) // 2
        y = (self.winfo_screenheight() - 520) // 2
        self.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            self, text=f"Welcome to {APP_NAME}",
            font=("Segoe UI", 22, "bold"),
        ).pack(pady=(30, 5))

        ctk.CTkLabel(
            self,
            text="Free, private audio & video transcription.\nEverything runs on your machine — nothing is uploaded.",
            font=("Segoe UI", 13), text_color="gray60", justify="center",
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
            text="The model downloads automatically the first time you transcribe.\nYou can change it later in Advanced Settings.",
            font=("Segoe UI", 11), text_color="gray50", justify="center",
        ).pack(pady=(15, 10))

        self.gpu_var = ctk.BooleanVar(value=offer_gpu)
        if offer_gpu:
            ctk.CTkCheckBox(
                self,
                text="Maximum GPU performance (NVIDIA, ~1 GB download)",
                variable=self.gpu_var, font=("Segoe UI", 12, "bold"),
            ).pack(pady=(5, 5))
            ctk.CTkLabel(
                self, text="NVIDIA GPU detected — recommended for the best speed",
                font=("Segoe UI", 11), text_color="#4CAF50",
            ).pack()

        ctk.CTkButton(
            self, text="Get Started", width=200, height=40,
            font=("Segoe UI", 14, "bold"),
            fg_color="#4CAF50", hover_color="#388E3C",
            command=self._finish,
        ).pack(pady=(15, 20))

        # Closing via the titlebar X keeps the model choice but must not
        # kick off the ~1 GB GPU download — that needs an explicit click
        self.protocol("WM_DELETE_WINDOW", self._dismiss)

    def _finish(self):
        self.result = {
            "model": self.model_var.get(),
            "install_gpu": self.gpu_var.get(),
        }
        self.destroy()

    def _dismiss(self):
        self.result = {
            "model": self.model_var.get(),
            "install_gpu": False,
        }
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
        try:
            self.iconbitmap(default=_resource_path("AudioWhisper.ico"))
        except Exception:
            pass
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Paths & managers
        self.app_data_dir = get_app_data_dir()
        self.settings_file = os.path.join(self.app_data_dir, "settings.json")
        self.runtime_mgr = RuntimeManager(self.app_data_dir)

        # State
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.model_name = ctk.StringVar(value="base")
        self.show_timestamps = ctk.BooleanVar(value=True)
        self.export_srt = ctk.BooleanVar(value=False)
        self.status_msg = ctk.StringVar(value="Ready")
        self.progress_val = ctk.DoubleVar(value=0.0)
        self.time_remaining_msg = ctk.StringVar(value="")
        self.device_label_var = ctk.StringVar(value="CPU")

        self.is_transcribing = False
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self._worker_proc = None

        # Load settings
        self.settings = self._load_settings()
        self.model_name.set(self.settings.get("model", "base"))
        default_output = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.isdir(default_output):
            default_output = os.path.expanduser("~")
        self.output_dir.set(self.settings.get("output_dir", "") or default_output)
        self.show_timestamps.set(self.settings.get("timestamps", True))
        self.export_srt.set(self.settings.get("export_srt", False))

        # Build UI
        self._create_widgets()
        self._refresh_device_label()
        # GPU enumeration is slow (WMI) — do it off-thread, then update the label
        self.runtime_mgr.detect_gpus_async(
            on_done=lambda: self.after(0, self._refresh_device_label))

        # Drag & drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._drop_file)

        # Log polling
        self._poll_log_queue()

        # First-run setup
        if not self.settings.get("setup_complete"):
            self.after(200, self._show_setup)
        else:
            self.after(600, self._check_gpu_auto_flag)
            self.after(1000, self._offer_legacy_cleanup)

        # Sanity check: the runtime ships with the installer
        if not self.runtime_mgr.runtime_ok():
            self.after(400, lambda: messagebox.showerror(
                APP_NAME,
                "The AudioWhisper runtime folder is missing.\n"
                "Please reinstall AudioWhisper to fix this.",
            ))

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
        self.settings.update({
            "model": self.model_name.get(),
            "output_dir": self.output_dir.get(),
            "timestamps": self.show_timestamps.get(),
            "export_srt": self.export_srt.get(),
            "setup_complete": True,
        })
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    # ── First-run ───────────────────────────────────────

    def _show_setup(self):
        offer_gpu = self.runtime_mgr.has_gpu() and not self.runtime_mgr.gpu_ready()
        dialog = SetupDialog(self, self.settings, offer_gpu)
        self.wait_window(dialog)
        if dialog.result:
            self.model_name.set(dialog.result["model"])
        self._save_settings()

        # Install the CUDA pack if the user asked in the dialog, or if the
        # installer's "maximum GPU performance" task was left checked
        auto = self.runtime_mgr.consume_auto_flag()
        if (dialog.result and dialog.result.get("install_gpu")) or \
                (auto and offer_gpu):
            self._install_gpu()

        self._offer_legacy_cleanup()

    def _check_gpu_auto_flag(self):
        """Upgrade installs skip the setup dialog but may carry the task flag."""
        auto = self.runtime_mgr.consume_auto_flag()
        if auto and self.runtime_mgr.has_gpu() and not self.runtime_mgr.gpu_ready():
            self._install_gpu()

    def _install_gpu(self):
        dialog = GpuInstallDialog(self, self.runtime_mgr)
        self.wait_window(dialog)
        if dialog.success:
            self._refresh_device_label()
            messagebox.showinfo(
                APP_NAME, "GPU acceleration is ready!\nTranscriptions will now use your NVIDIA GPU.")
        elif dialog.error and dialog.error != "cancelled":
            messagebox.showerror(
                APP_NAME,
                f"GPU setup didn't finish:\n{dialog.error[:400]}\n\n"
                f"Details saved to:\n{self.runtime_mgr.install_log}\n\n"
                "Don't worry — transcription still works on CPU.\n"
                "You can retry from Advanced Settings anytime.",
            )

    def _offer_legacy_cleanup(self):
        """Offer to remove the old v1.0 downloaded runtime (~5 GB)."""
        if self.settings.get("legacy_cleanup_done"):
            return
        legacy = self.runtime_mgr.legacy_runtime_dir()
        if not legacy:
            return
        if not messagebox.askyesno(
            APP_NAME,
            "AudioWhisper found a large leftover folder from an old version\n"
            "(about 5 GB) that is no longer needed.\n\nRemove it now to free up disk space?",
        ):
            # Respect the "no" — don't ask again
            self.settings["legacy_cleanup_done"] = True
            self._save_settings()
            return

        def _cleanup():
            self.runtime_mgr.remove_legacy_runtime()
            # Only mark done if it actually worked, so a locked file
            # (AV scan, open Explorer window) gets another chance later
            if not self.runtime_mgr.legacy_runtime_dir():
                def mark():
                    self.settings["legacy_cleanup_done"] = True
                    self._save_settings()
                    self.status_msg.set("Old version cleaned up — 5 GB freed.")
                self.after(0, mark)

        threading.Thread(target=_cleanup, daemon=True).start()

    def _refresh_device_label(self):
        # Same predicate _run_worker uses to pick the engine
        backend = self.runtime_mgr.pick_backend()
        if backend != "cpu":
            self.device_label_var.set("GPU")
            self.device_label.configure(text_color="#4CAF50")
        else:
            self.device_label_var.set("CPU")
            self.device_label.configure(text_color="#FF9800")
        if backend == "cuda" and hasattr(self, "gpu_btn"):
            self.gpu_btn.grid_remove()

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

        self.device_label = ctk.CTkLabel(
            header, textvariable=self.device_label_var,
            text_color="#FF9800", font=("Segoe UI", 12, "bold"),
        )
        self.device_label.pack(side="right")

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

        if self.runtime_mgr.has_gpu() and not self.runtime_mgr.gpu_ready():
            self.gpu_btn = ctk.CTkButton(
                sf, text="Maximum GPU Performance (NVIDIA, ~1 GB)",
                fg_color="#4CAF50", hover_color="#388E3C",
                command=self._install_gpu,
            )
            self.gpu_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=8, sticky="w")

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

    @staticmethod
    def _format_size(size_bytes):
        if size_bytes >= 1024 ** 3:
            return f"{size_bytes / (1024**3):.1f} GB"
        if size_bytes >= 1024 ** 2:
            return f"{size_bytes / (1024**2):.1f} MB"
        return f"{size_bytes / 1024:.0f} KB"

    @staticmethod
    def _format_duration(dur):
        if dur >= 3600:
            return f"{int(dur // 3600)}h {int((dur % 3600) // 60)}m"
        if dur >= 60:
            return f"{int(dur // 60)}m {int(dur % 60)}s"
        return f"{int(dur)}s"

    def _load_file(self, path):
        """Load a file into the UI, resetting previous state."""
        if self.is_transcribing:
            self.status_msg.set("Stop the current transcription before loading a new file.")
            return
        self.input_path.set(path)

        try:
            size_str = self._format_size(os.path.getsize(path))
        except Exception:
            size_str = ""
        self.drop_zone.set_file(path, info=size_str)

        # Reset progress from previous transcription
        self.progress_val.set(0)
        self.progress_bar.set(0)
        self.visualizer.reset()
        self.status_msg.set("Ready")
        self.time_remaining_msg.set("")
        self.open_folder_btn.configure(state="disabled")
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", "Your transcription will appear here.\n")
        self.transcript_box.configure(state="disabled")

        # Probe duration + waveform in the background (via the runtime)
        threading.Thread(
            target=self._probe_file, args=(path, size_str), daemon=True).start()

    def _probe_file(self, path, size_str):
        """Get duration + waveform peaks using the bundled runtime's PyAV."""
        if not self.runtime_mgr.runtime_ok():
            return
        try:
            result = subprocess.run(
                [self.runtime_mgr.python_exe, _resource_path("media_probe.py"),
                 path, str(self.visualizer.bars)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, env=self.runtime_mgr.worker_env(),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            data = json.loads(result.stdout)
        except Exception:
            return

        duration = data.get("duration") or 0
        peaks = data.get("peaks") or []

        def apply():
            if self.input_path.get() != path:
                return  # user loaded a different file meanwhile
            if duration:
                info = f"{self._format_duration(duration)}  |  {size_str}" if size_str \
                    else self._format_duration(duration)
                self.drop_zone.set_file(path, info=info)
            if peaks:
                self.visualizer.set_amplitudes(peaks)

        self.after(0, apply)

    def _drop_file(self, event):
        # tkinterdnd2 delivers multi-file drops as a Tcl list — take the first
        try:
            paths = self.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        path = paths[0]
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        if len(paths) > 1:
            self.status_msg.set("One file at a time — loaded the first one.")
        self._load_file(path)

    def _browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[("Media Files", SUPPORTED_FORMATS)]
        )
        if path:
            self._load_file(path)

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
        proc = self._worker_proc
        if proc and proc.poll() is None:
            proc.terminate()
        self.runtime_mgr.kill_pip()
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

        if not self.runtime_mgr.runtime_ok():
            messagebox.showerror(
                APP_NAME,
                "The AudioWhisper runtime folder is missing.\n"
                "Please reinstall AudioWhisper to fix this.",
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
        backend = self.runtime_mgr.pick_backend()

        cmd = [
            self.runtime_mgr.python_exe, _resource_path("transcribe_worker.py"),
            "--input", self.input_path.get(),
            "--model", self.model_name.get(),
            "--backend", backend,
            "--whispercpp_dir", self.runtime_mgr.whispercpp_dir,
        ]
        if self.output_dir.get():
            cmd.extend(["--output_dir", self.output_dir.get()])
        if self.show_timestamps.get():
            cmd.append("--timestamps")
        if self.export_srt.get():
            cmd.append("--export_srt")

        stderr_lines = []
        try:
            self._worker_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=self.runtime_mgr.worker_env(),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            # Drain stderr on a separate thread so a chatty worker can't
            # fill the pipe and deadlock both processes
            def _drain(pipe):
                try:
                    for l in pipe:
                        stderr_lines.append(l.rstrip())
                except Exception:
                    pass

            threading.Thread(
                target=_drain, args=(self._worker_proc.stderr,), daemon=True).start()

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

                # All UI mutations happen on the main thread
                self.after(0, lambda m=msg: self._handle_worker_msg(m))

            self._worker_proc.wait()

            # If the worker crashed, show its last stderr lines
            if self._worker_proc.returncode and not self.stop_event.is_set():
                tail = [l for l in stderr_lines if l.strip()][-3:]
                if tail:
                    self._log(f"Error: {' | '.join(tail)}")

        except Exception as e:
            self._log(f"Error: {e}")
        finally:
            self._worker_proc = None
            self.is_transcribing = False
            self.after(0, lambda: self._update_ui_state(transcribing=False))

    def _handle_worker_msg(self, msg):
        """Process one worker JSON message (main thread only)."""
        msg_type = msg.get("type")
        if msg_type == "status":
            self.status_msg.set(msg["msg"])
        elif msg_type == "model_download":
            self.status_msg.set(msg.get("msg", "Downloading model..."))
            self.time_remaining_msg.set(f"{int(msg.get('value', 0) * 100)}%")
        elif msg_type == "segment":
            self._log(f"{msg['timestamp']} {msg['text']}", is_transcript=True)
        elif msg_type == "progress":
            self.progress_val.set(msg["value"])
            self.visualizer.set_progress(msg["value"])
            if msg.get("eta"):
                self.time_remaining_msg.set(msg["eta"])
        elif msg_type == "done":
            self.status_msg.set(msg["msg"])
            self.progress_val.set(1.0)
            self.visualizer.set_progress(1.0)
            self.time_remaining_msg.set("Complete")
            self.open_folder_btn.configure(state="normal")
        elif msg_type == "error":
            self.status_msg.set(f"Error: {msg['msg']}")

    def _stop_transcription(self):
        if self.is_transcribing:
            if not messagebox.askyesno(
                APP_NAME, "Stop the current transcription?\nAny progress will be lost."
            ):
                return
            self.stop_event.set()
            proc = self._worker_proc
            if proc and proc.poll() is None:
                proc.terminate()

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
