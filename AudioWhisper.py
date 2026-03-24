import sys
import subprocess
import os
import threading
import time
import queue
import json
import datetime
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from faster_whisper import WhisperModel
import ffmpeg
import librosa
import soundfile as sf
import torch

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

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def get_app_data_dir():
    """Return the app's persistent data directory in AppData."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~/.config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def check_ffmpeg():
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


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
#  First-Run Setup Dialog
# ──────────────────────────────────────────────────────────

class SetupDialog(ctk.CTkToplevel):
    """Shown on first launch to welcome the user and pick defaults."""

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.title(f"{APP_NAME} — Setup")
        self.geometry("520x480")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.settings = settings
        self.result = None

        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 520) // 2
        y = (self.winfo_screenheight() - 480) // 2
        self.geometry(f"+{x}+{y}")

        # Content
        ctk.CTkLabel(
            self, text=f"Welcome to {APP_NAME}",
            font=("Segoe UI", 22, "bold"),
        ).pack(pady=(30, 5))

        ctk.CTkLabel(
            self,
            text="Free, private, offline audio & video transcription.\nEverything runs on your machine — nothing is uploaded.",
            font=("Segoe UI", 13),
            text_color="gray60",
            justify="center",
        ).pack(pady=(0, 20))

        # Device info
        device = "NVIDIA GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        device_color = "#4CAF50" if torch.cuda.is_available() else "#FF9800"
        ctk.CTkLabel(
            self, text=f"Detected hardware:  {device}",
            font=("Segoe UI", 13, "bold"), text_color=device_color,
        ).pack(pady=(0, 20))

        # Model picker
        ctk.CTkLabel(
            self, text="Choose a default model size:",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=40)

        self.model_var = ctk.StringVar(value=settings.get("model", "base"))
        for name, desc in MODELS.items():
            ctk.CTkRadioButton(
                self, text=f"{name}  —  {desc}",
                variable=self.model_var, value=name,
                font=("Segoe UI", 12),
            ).pack(anchor="w", padx=60, pady=2)

        ctk.CTkLabel(
            self,
            text="The model will be downloaded automatically on first use.\nYou can change this later in Advanced Settings.",
            font=("Segoe UI", 11), text_color="gray50", justify="center",
        ).pack(pady=(15, 10))

        # FFmpeg warning
        if not check_ffmpeg():
            ctk.CTkLabel(
                self,
                text="FFmpeg not found — video files won't work.\n"
                     "Install it:  winget install ffmpeg",
                font=("Segoe UI", 12, "bold"), text_color="#F44336",
                justify="center",
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

        # Paths
        self.app_data_dir = get_app_data_dir()
        self.settings_file = os.path.join(self.app_data_dir, "settings.json")

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
        self.is_paused = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.log_queue = queue.Queue()

        # Load settings
        self.settings = self._load_settings()
        self.model_name.set(self.settings.get("model", "base"))
        self.output_dir.set(self.settings.get("output_dir", ""))
        self.show_timestamps.set(self.settings.get("timestamps", True))
        self.export_srt.set(self.settings.get("export_srt", False))

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"

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
        dialog = SetupDialog(self, self.settings)
        self.wait_window(dialog)
        if dialog.result:
            self.model_name.set(dialog.result)
        self._save_settings()

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

        device_color = "#4CAF50" if self.device == "cuda" else "#FF9800"
        device_text = "GPU" if self.device == "cuda" else "CPU"
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
        btn_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_btn = ctk.CTkButton(
            btn_row, text="Start Transcription",
            fg_color="#4CAF50", hover_color="#388E3C",
            height=40, font=("Segoe UI", 14, "bold"),
            command=self._start_transcription,
        )
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.pause_btn = ctk.CTkButton(
            btn_row, text="Pause",
            fg_color="#FFC107", text_color="black", hover_color="#FFD54F",
            height=40, state="disabled", command=self._toggle_pause,
        )
        self.pause_btn.grid(row=0, column=1, padx=5, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop",
            fg_color="#F44336", hover_color="#E57373",
            height=40, state="disabled", command=self._stop_transcription,
        )
        self.stop_btn.grid(row=0, column=2, padx=5, sticky="ew")

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

    # ── Transcription ───────────────────────────────────

    def _get_audio_duration(self, file_path):
        try:
            return librosa.get_duration(path=file_path)
        except Exception:
            return 0

    def _extract_audio(self, video_path):
        temp_audio = os.path.splitext(video_path)[0] + "_temp.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(temp_audio, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True)
            )
            return temp_audio
        except Exception as e:
            raise RuntimeError(f"FFmpeg error: {e}")

    def _run_transcription(self):
        input_file = self.input_path.get()
        output_folder = self.output_dir.get()
        model_type = self.model_name.get()
        temp_file = None
        start_time = time.time()

        try:
            # Check FFmpeg for video files
            is_video = input_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
            if is_video and not check_ffmpeg():
                self._log("FFmpeg is required for video files. Install it: winget install ffmpeg")
                return

            self._log("Loading model (may download on first use)...")

            total_duration = self._get_audio_duration(input_file)

            try:
                model = WhisperModel(model_type, device=self.device, compute_type=self.compute_type)
            except Exception as e:
                self._log(f"Model error: {e}")
                return

            if self.stop_event.is_set():
                return

            if is_video:
                self._log("Extracting audio from video...")
                temp_file = self._extract_audio(input_file)
                process_file = temp_file
                total_duration = self._get_audio_duration(process_file)
            else:
                process_file = input_file

            if self.stop_event.is_set():
                return

            self._log("Transcribing...")
            self.transcript_box.configure(state="normal")
            self.transcript_box.delete("1.0", "end")
            self.transcript_box.configure(state="disabled")

            segments, info = model.transcribe(process_file, beam_size=5)
            collected_segments = []

            for segment in segments:
                if self.stop_event.is_set():
                    break
                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)

                # Progress & ETA
                if total_duration > 0:
                    prog = min(segment.end / total_duration, 1.0)
                    self.progress_val.set(prog)
                    self.visualizer.set_progress(prog)
                    elapsed = time.time() - start_time
                    if prog > 0.01:
                        remaining = (elapsed / prog) - elapsed
                        mins, secs = divmod(int(remaining), 60)
                        self.time_remaining_msg.set(f"~{mins:02}:{secs:02} remaining")

                ts = f"[{int(segment.start // 3600):02}:{int((segment.start % 3600) // 60):02}:{int(segment.start % 60):02}]"
                self._log(f"{ts} {segment.text.strip()}", is_transcript=True)
                collected_segments.append(segment)

            if self.stop_event.is_set() and not collected_segments:
                self._log("Stopped.")
                return

            # Save
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            if not output_folder:
                output_folder = os.path.dirname(input_file)

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
                        s = self._format_srt_time(seg.start)
                        e = self._format_srt_time(seg.end)
                        f.write(f"{i}\n{s} --> {e}\n{seg.text.strip()}\n\n")

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            self._log(f"Done — saved to {output_folder}  ({mins}m {secs}s)")
            self.progress_val.set(1.0)
            self.visualizer.set_progress(1.0)
            self.time_remaining_msg.set("Complete")
            self.open_folder_btn.configure(state="normal")

        except Exception as e:
            self._log(f"Error: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            self.stop_event.clear()
            self.pause_event.clear()
            self.is_transcribing = False
            self.is_paused = False
            self._update_ui_state(transcribing=False)

    def _format_srt_time(self, seconds):
        total_sec = int(seconds)
        millis = int((seconds - total_sec) * 1000)
        hours, remainder = divmod(total_sec, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _start_transcription(self):
        if not self.input_path.get():
            messagebox.showwarning(APP_NAME, "Please select a file first.")
            return

        self.is_transcribing = True
        self.is_paused = False
        self.stop_event.clear()
        self.pause_event.clear()
        self.progress_val.set(0)
        self.visualizer.set_progress(0)
        self.time_remaining_msg.set("Calculating...")
        self._update_ui_state(transcribing=True)
        self._save_settings()
        threading.Thread(target=self._run_transcription, daemon=True).start()

    def _stop_transcription(self):
        if self.is_transcribing:
            self.stop_event.set()
            if self.pause_event.is_set():
                self.pause_event.clear()

    def _toggle_pause(self):
        if not self.is_transcribing:
            return
        if self.is_paused:
            self.is_paused = False
            self.pause_event.clear()
            self.pause_btn.configure(text="Pause", fg_color="#FFC107")
            self.status_msg.set("Resumed...")
        else:
            self.is_paused = True
            self.pause_event.set()
            self.pause_btn.configure(text="Resume", fg_color="#00E676")
            self.status_msg.set("Paused")

    def _update_ui_state(self, transcribing):
        if transcribing:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            self.open_folder_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled", text="Pause", fg_color="#FFC107")


# ──────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AudioWhisperApp()
    app.mainloop()
