# AudioWhisper

Free offline transcription for Windows. Drop in an audio or video file and get a text transcript. Everything runs on your own machine, so nothing gets uploaded anywhere and there's no account or subscription.

Built on OpenAI's [Whisper](https://github.com/openai/whisper) model, running through [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Download

Grab the installer from the [Releases page](https://github.com/Artixskillz/AudioWhisper/releases/latest). Install it, open the app, drop a file in. There's nothing else to set up.

Windows SmartScreen might warn you the first time because the installer isn't code-signed yet. Click "More info" and then "Run anyway".

## What it does

- Transcribes audio and video files. Video works out of the box, no FFmpeg or anything else to install.
- Shows the transcript live while it works, with a waveform and time remaining.
- Five model sizes, from tiny (fast but rough) to large-v3 (slow but very accurate).
- Saves .txt transcripts and optional .srt subtitle files, with or without timestamps.
- Dark and light mode.
- GPU acceleration on AMD, Intel, and NVIDIA. Any modern GPU gets picked up automatically through the built-in Vulkan engine, nothing to download. NVIDIA owners can additionally grab a ~1 GB CUDA pack (offered during install) for the fastest speeds.

## Formats

MP3, WAV, M4A, FLAC, OGG, WMA, AAC, MP4, AVI, MOV, MKV, WebM

## Requirements

Windows 10 or 11, 64-bit. About 1 GB of disk space plus whatever model size you pick. That's it.

## How it works

The installer ships with its own Python runtime and two transcription engines: faster-whisper for CPU and NVIDIA CUDA, and whisper.cpp with Vulkan for every other GPU. Nothing downloads during install. The first time you transcribe, the app fetches the Whisper model you picked (145 MB for base, up to 3 GB for large-v3) and shows a progress bar while it does. After that everything works fully offline.

The app picks the best engine on its own: CUDA if you've installed the NVIDIA pack, then Vulkan if you have any usable GPU, then CPU. If a GPU fails for any reason it quietly falls back to CPU instead of erroring out.

## Building from source

```bash
pip install pyinstaller customtkinter tkinterdnd2
python build.py
```

That builds the GUI, assembles the bundled runtime, and compiles the installer. You'll need [Inno Setup 6](https://jrsoftware.org/isinfo.php) for the last step, plus the `gh` CLI so the build can pull the whisper.cpp Vulkan binaries from this repo's `build-whispercpp` workflow (or run that workflow yourself first). The finished setup lands in `installer_output/`.

## License

MIT
