"""
Build script for AudioWhisper.
Produces a lightweight EXE (no torch/whisper — those install on first run).

Usage:
    python build.py
"""
import PyInstaller.__main__
import os
import customtkinter
import tkinterdnd2

ctk_path = os.path.dirname(customtkinter.__file__)
dnd_path = os.path.dirname(tkinterdnd2.__file__)

base_dir = os.path.dirname(__file__)
icon_path = os.path.join(base_dir, "AudioWhisper.ico")
version_path = os.path.join(base_dir, "version_info.txt")

cmd = [
    "AudioWhisper.py",
    "--name=AudioWhisper",
    "--noconsole",
    "--onefile",
    "--clean",
    # Bundle the worker script and icon
    "--add-data=transcribe_worker.py;.",
    "--add-data=AudioWhisper.ico;.",
    # Hidden imports for the GUI
    "--hidden-import=babel.numbers",
    "--hidden-import=tkinterdnd2",
    "--hidden-import=customtkinter",
    # Exclude heavy ML deps (installed at runtime)
    "--exclude-module=torch",
    "--exclude-module=faster_whisper",
    "--exclude-module=ctranslate2",
    "--exclude-module=huggingface_hub",
    "--exclude-module=tokenizers",
    "--exclude-module=safetensors",
    # Data files
    f"--add-data={ctk_path};customtkinter",
    f"--add-data={dnd_path};tkinterdnd2",
]

if os.path.exists(icon_path):
    cmd.append(f"--icon={icon_path}")
if os.path.exists(version_path):
    cmd.append(f"--version-file={version_path}")

print("Building AudioWhisper (lightweight — no torch/whisper)...")
PyInstaller.__main__.run(cmd)
print("Done! Check the 'dist' folder for AudioWhisper.exe")
