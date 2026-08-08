# AudioWhisper

**Free, private, offline transcription for audio and video files.**

AudioWhisper turns your spoken words into text — entirely on your own computer. Nothing is uploaded, no account is needed, and no extra software is required.

Powered by OpenAI's [Whisper](https://github.com/openai/whisper) model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for maximum speed.

---

## Download

**[Download the latest installer from Releases](https://github.com/Artixskillz/AudioWhisper/releases/latest)**

Run the installer, launch the app, and start transcribing. That's it — everything the app needs is included. No terminal, no Python, no FFmpeg to install.

> **Note:** Windows SmartScreen may warn about an unrecognized app (the installer isn't code-signed yet). Click **More info → Run anyway** to continue.

---

## Features

- **Drag & drop** any audio or video file to transcribe it
- **Fully self-contained** — video and audio decoding is built in, no FFmpeg needed
- **Optional GPU acceleration** — one click enables your NVIDIA GPU for 5-10x faster transcription
- **Multiple models** — from fast/rough (`tiny`) to slow/accurate (`large-v3`)
- **Subtitle export** — generate `.srt` files for YouTube, Premiere, VLC, etc.
- **Timestamps** — optional time markers in the transcript
- **Live transcript & waveform** — watch the text appear with visual progress
- **Dark & light mode** — switch with one click
- **Private** — everything runs locally, nothing leaves your machine

## Supported Formats

**Audio:** MP3, WAV, M4A, FLAC, OGG, WMA, AAC
**Video:** MP4, AVI, MOV, MKV, WebM

## Requirements

- **Windows 10/11** (64-bit)
- ~1 GB of disk space (plus the model size you choose)
- **Optional:** NVIDIA GPU (RTX 20/30/40/50 series) for faster transcription

## How It Works

1. **Install** — run the installer from the Releases page. The complete transcription engine ships inside it.
2. **Drop a file** — drag an audio or video file into the app (or click to browse)
3. **Transcribe** — hit Start and watch the live transcript appear
4. **Save** — output is saved as a `.txt` file (and `.srt` if enabled)

The first time you transcribe, the app downloads your chosen Whisper model with a progress bar (~145 MB for `base`, up to ~3 GB for `large-v3`). After that, transcription works completely offline.

If you have an NVIDIA GPU, the app offers one-click GPU acceleration (a one-time ~1 GB download). CPU transcription works out of the box either way.

## Building from Source

```bash
pip install pyinstaller customtkinter tkinterdnd2
python build.py
```

`build.py` builds the GUI, assembles the bundled Python runtime with the transcription engine, and compiles the installer (requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)). The finished setup lands in `installer_output/`.

## License

MIT
