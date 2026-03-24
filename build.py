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

icon_path = os.path.join(os.path.dirname(__file__), "AudioWhisper.ico")
icon_arg = f"--icon={icon_path}" if os.path.exists(icon_path) else ""

cmd = [
    "AudioWhisper.py",
    "--name=AudioWhisper",
    "--noconsole",
    "--onefile",
    "--clean",
    # Bundle the worker script
    "--add-data=transcribe_worker.py;.",
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

if icon_arg:
    cmd.append(icon_arg)

print("Building AudioWhisper (lightweight — no torch/whisper)...")
PyInstaller.__main__.run(cmd)
print("Done! Check the 'dist' folder for AudioWhisper.exe")
