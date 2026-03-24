# AudioWhisper

**Free, private, offline transcription for audio and video files.**

AudioWhisper turns your spoken words into text — entirely on your own computer. Nothing is uploaded, no account is needed, and it works with or without an internet connection after setup.

Powered by OpenAI's [Whisper](https://github.com/openai/whisper) model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for maximum speed.

---

## Download

**[Download the latest installer from Releases](https://github.com/Artixskillz/AudioWhisper/releases/latest)**

Run the installer, launch the app, and start transcribing. That's it.

---

## Features

- **Drag & drop** any audio or video file to transcribe it
- **GPU accelerated** — automatically uses your NVIDIA GPU if available, falls back to CPU
- **Multiple models** — from fast/rough (`tiny`) to slow/accurate (`large-v3`)
- **Subtitle export** — generate `.srt` files for YouTube, Premiere, VLC, etc.
- **Timestamps** — optional time markers in the transcript
- **Pause & resume** — pause transcription and pick up where you left off
- **Waveform preview** — visual progress through your audio
- **Dark & light mode** — switch with one click
- **Private** — everything runs locally, nothing leaves your machine

## Supported Formats

**Audio:** MP3, WAV, M4A, FLAC, OGG, WMA, AAC
**Video:** MP4, AVI, MOV, MKV, WebM (requires [FFmpeg](https://ffmpeg.org))

## Requirements

- **Windows 10/11** (64-bit)
- **FFmpeg** for video files — install with `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Optional:** NVIDIA GPU (RTX 20/30/40/50 series) for faster transcription

## How It Works

1. **Install** — run the installer from the Releases page
2. **First launch** — the app will download and set up the transcription engine (~150 MB for CPU, ~2.5 GB for GPU). This only happens once and requires an internet connection.
3. **Drop a file** — drag an audio or video file into the app (or click to browse)
4. **Transcribe** — hit Start and watch the live transcript appear
5. **Save** — output is saved as a `.txt` file (and `.srt` if enabled)

The first time you use a model size, it will also be downloaded automatically (~75 MB for `base`, up to ~3 GB for `large-v3`).

## Building from Source

```bash
pip install pyinstaller customtkinter tkinterdnd2 librosa soundfile numpy
python build.py
```

The EXE will be in the `dist/` folder. To build the installer, install [Inno Setup](https://jrsoftware.org/isinfo.php) and compile `installer.iss`.

## License

MIT
