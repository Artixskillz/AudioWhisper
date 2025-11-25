# AudioWhisper 🎙️

**AudioWhisper** is a modern, user-friendly desktop application for transcribing audio and video files using OpenAI's state-of-the-art **Whisper** model.

It features a **Smart Installer** that automatically detects your hardware (NVIDIA GPU or CPU) and sets up the optimized environment for you—making it completely portable and hassle-free.

## 🚀 Key Features

*   **🤖 Smart Installation**: Automatically detects your GPU (including the latest RTX 50-series) and installs the correct PyTorch/CUDA drivers. No manual setup required.
*   **⚡ GPU Acceleration**: Fully utilizes NVIDIA GPUs for blazing fast transcription. Falls back to CPU automatically if needed.
*   **🖱️ Drag & Drop**: Simply drag your audio or video files onto the app to load them.
*   **📝 Subtitle Export**: Generate `.SRT` subtitles for YouTube, VLC, or Premiere Pro with a single click.
*   **💾 Persistent Settings**: Remembers your favorite model, output folder, and preferences.
*   **🎨 Modern UI**: Built with `CustomTkinter` for a sleek, dark-mode compatible interface.
*   **📹 Video Support**: Extracts audio automatically from MP4, MKV, AVI, and MOV files.

## 📦 Installation

1.  **Install Python** (3.10 or newer recommended).
2.  **Install FFmpeg**:
    *   *Windows*: `winget install ffmpeg` (or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH).
3.  **Download** `AudioWhisper_v6.py`.
4.  **Run it**:
    ```bash
    python AudioWhisper_v6.py
    ```
    *The app will automatically install all necessary Python libraries (Torch, Whisper, etc.) on the first run.*

## 🛠️ Usage

1.  **Select Input**: Drag & drop a file or click "Browse".
2.  **Select Output**: Choose where to save the text files.
3.  **Choose Model**:
    *   `tiny` / `base`: Fast, good for clear audio.
    *   `small` / `medium`: Balanced speed and accuracy.
    *   `large`: Maximum accuracy (requires more VRAM).
4.  **Options**:
    *   *Timestamps*: Saves a text file with `[00:00:10]` markers.
    *   *Export Subtitles*: Creates a standard `.srt` file.
5.  **Start**: Click "Start Transcription" and watch the real-time logs!

## 🔧 Requirements

*   Python 3.8+
*   FFmpeg (must be in system PATH)
*   **Optional**: NVIDIA GPU (RTX 30xx/40xx/50xx) for acceleration.

## 📄 License

MIT License. Free to use and modify.

---
*Powered by [OpenAI Whisper](https://github.com/openai/whisper) and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).*
