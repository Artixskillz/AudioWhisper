"""
Build script for AudioWhisper.
Produces a single-file EXE with PyInstaller.

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
    # Hidden imports
    "--hidden-import=babel.numbers",
    "--hidden-import=tkinterdnd2",
    "--hidden-import=customtkinter",
    "--hidden-import=faster_whisper",
    "--hidden-import=sklearn.utils._typedefs",
    "--hidden-import=sklearn.neighbors._partition_nodes",
    # Data files
    f"--add-data={ctk_path};customtkinter",
    f"--add-data={dnd_path};tkinterdnd2",
]

if icon_arg:
    cmd.append(icon_arg)

print("Building AudioWhisper...")
PyInstaller.__main__.run(cmd)
print("Done! Check the 'dist' folder for AudioWhisper.exe")
