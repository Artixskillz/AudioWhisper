"""
Build script for AudioWhisper.

Produces a self-contained app folder in dist/AudioWhisper:
  AudioWhisper.exe   - the GUI (PyInstaller onedir)
  runtime/           - embedded Python with faster-whisper preinstalled
then compiles the Inno Setup installer from it.

The installer ships everything - users never download dependencies.
Only the Whisper model itself downloads on first use (with progress UI).

Usage:
    python build.py [--skip-exe] [--skip-runtime] [--skip-installer]
"""
import os
import re
import sys
import shutil
import zipfile
import subprocess
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_APP = os.path.join(BASE_DIR, "dist", "AudioWhisper")
RUNTIME_DIR = os.path.join(DIST_APP, "runtime")
CACHE_DIR = os.path.join(BASE_DIR, "build_cache")

PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# Engine dependencies baked into the runtime (PyAV bundles FFmpeg,
# so video decoding needs no system dependencies)
RUNTIME_PACKAGES = [
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.1",
]

ISCC_CANDIDATES = [
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
]


def get_app_version():
    """Single source of truth: APP_VERSION in AudioWhisper.py."""
    with open(os.path.join(BASE_DIR, "AudioWhisper.py"), encoding="utf-8") as f:
        m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', f.read())
    if not m:
        raise RuntimeError("APP_VERSION not found in AudioWhisper.py")
    return m.group(1)


def write_version_info(version):
    """Generate version_info.txt for the EXE's Properties metadata."""
    parts = (version.split(".") + ["0", "0", "0"])[:3]
    ver_tuple = f"({parts[0]}, {parts[1]}, {parts[2]}, 0)"
    content = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={ver_tuple},
    prodvers={ver_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [StringStruct('CompanyName', 'Artixskillz'),
           StringStruct('FileDescription', 'AudioWhisper - Free offline transcription'),
           StringStruct('FileVersion', '{version}'),
           StringStruct('InternalName', 'AudioWhisper'),
           StringStruct('LegalCopyright', 'Copyright (c) 2026 Artixskillz. MIT License.'),
           StringStruct('OriginalFilename', 'AudioWhisper.exe'),
           StringStruct('ProductName', 'AudioWhisper'),
           StringStruct('ProductVersion', '{version}')])
      ]
    ),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    path = os.path.join(BASE_DIR, "version_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def download(url, dest):
    if os.path.exists(dest):
        print(f"  cached: {os.path.basename(dest)}")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "AudioWhisper-build/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def build_exe(version):
    print("\n[1/3] Building GUI EXE (PyInstaller onedir)...")
    # PyInstaller wipes dist/AudioWhisper, so park an existing runtime
    # (from a previous build) and restore it afterwards
    runtime_keep = os.path.join(CACHE_DIR, "runtime_keep")
    if os.path.exists(RUNTIME_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(runtime_keep):
            shutil.rmtree(runtime_keep)
        shutil.move(RUNTIME_DIR, runtime_keep)
    try:
        _run_pyinstaller(version)
    finally:
        if os.path.exists(runtime_keep) and not os.path.exists(RUNTIME_DIR):
            os.makedirs(DIST_APP, exist_ok=True)
            shutil.move(runtime_keep, RUNTIME_DIR)


def _run_pyinstaller(version):
    import PyInstaller.__main__
    import customtkinter
    import tkinterdnd2

    ctk_path = os.path.dirname(customtkinter.__file__)
    dnd_path = os.path.dirname(tkinterdnd2.__file__)
    version_path = write_version_info(version)
    icon_path = os.path.join(BASE_DIR, "AudioWhisper.ico")

    cmd = [
        "AudioWhisper.py",
        "--name=AudioWhisper",
        "--noconsole",
        "--onedir",
        "--clean",
        "--noconfirm",
        # Bundle the worker/probe scripts and icon
        "--add-data=transcribe_worker.py;.",
        "--add-data=media_probe.py;.",
        "--add-data=AudioWhisper.ico;.",
        # Hidden imports for the GUI
        "--hidden-import=tkinterdnd2",
        "--hidden-import=customtkinter",
        # The GUI needs none of these - keep the EXE small
        "--exclude-module=numpy",
        "--exclude-module=librosa",
        "--exclude-module=scipy",
        "--exclude-module=numba",
        "--exclude-module=torch",
        "--exclude-module=faster_whisper",
        "--exclude-module=ctranslate2",
        "--exclude-module=huggingface_hub",
        "--exclude-module=tokenizers",
        "--exclude-module=safetensors",
        "--exclude-module=soundfile",
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        f"--add-data={ctk_path};customtkinter",
        f"--add-data={dnd_path};tkinterdnd2",
        f"--icon={icon_path}",
        f"--version-file={version_path}",
    ]
    PyInstaller.__main__.run(cmd)


def build_runtime():
    print("\n[2/3] Building bundled Python runtime...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(RUNTIME_DIR):
        shutil.rmtree(RUNTIME_DIR)
    os.makedirs(RUNTIME_DIR)

    # Embedded Python
    embed_zip = os.path.join(CACHE_DIR, os.path.basename(PYTHON_EMBED_URL))
    download(PYTHON_EMBED_URL, embed_zip)
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(RUNTIME_DIR)

    # Enable site-packages in the embedded distribution
    for name in os.listdir(RUNTIME_DIR):
        if name.endswith("._pth"):
            pth = os.path.join(RUNTIME_DIR, name)
            with open(pth) as f:
                content = f.read()
            with open(pth, "w") as f:
                f.write(content.replace("#import site", "import site"))
            break

    python_exe = os.path.join(RUNTIME_DIR, "python.exe")

    # pip
    get_pip = os.path.join(CACHE_DIR, "get-pip.py")
    download(GET_PIP_URL, get_pip)
    print("  installing pip...")
    r = subprocess.run(
        [python_exe, get_pip, "--no-warn-script-location"],
        cwd=RUNTIME_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"get-pip failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    # Engine packages
    print(f"  installing {', '.join(RUNTIME_PACKAGES)} ...")
    r = subprocess.run(
        [python_exe, "-m", "pip", "install", "--no-cache-dir",
         "--no-warn-script-location"] + RUNTIME_PACKAGES,
        cwd=RUNTIME_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    # Prune caches
    pruned = 0
    for root, dirs, _files in os.walk(RUNTIME_DIR):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
                pruned += 1
    print(f"  pruned {pruned} __pycache__ dirs")

    # Smoke test: the engine imports inside the runtime
    r = subprocess.run(
        [python_exe, "-c", "import faster_whisper, av, ctranslate2; print('engine-ok')"],
        capture_output=True, text=True)
    if "engine-ok" not in r.stdout:
        raise RuntimeError(f"Runtime smoke test failed:\n{r.stdout}\n{r.stderr}")
    print("  runtime smoke test passed")

    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _dn, fn in os.walk(RUNTIME_DIR) for f in fn)
    print(f"  runtime size: {size / 1024**2:.0f} MB")


def build_installer(version):
    print("\n[3/3] Compiling installer (Inno Setup)...")
    # Never ship a half-assembled app folder
    for required in (
        os.path.join(DIST_APP, "AudioWhisper.exe"),
        os.path.join(RUNTIME_DIR, "python.exe"),
    ):
        if not os.path.exists(required):
            raise RuntimeError(
                f"Missing {required} - run the exe/runtime stages first")
    iscc = next((c for c in ISCC_CANDIDATES if os.path.exists(c)), None)
    if not iscc:
        raise RuntimeError(
            "ISCC.exe not found - install Inno Setup 6: winget install JRSoftware.InnoSetup")
    r = subprocess.run(
        [iscc, f"/DMyAppVersion={version}", "installer.iss"],
        cwd=BASE_DIR, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        raise RuntimeError(f"ISCC failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")


def main():
    version = get_app_version()
    print(f"Building AudioWhisper v{version} (self-contained installer)")

    args = sys.argv[1:]
    if "--skip-exe" not in args:
        build_exe(version)
    if "--skip-runtime" not in args:
        build_runtime()
    if "--skip-installer" not in args:
        build_installer(version)

    out = os.path.join(BASE_DIR, "installer_output", f"AudioWhisper_Setup_{version}.exe")
    if os.path.exists(out):
        print(f"\nDone: {out} ({os.path.getsize(out) / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
