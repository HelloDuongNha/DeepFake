"""PyInstaller onedir desktop package. Run from the repository root."""

import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH).parent
data = [(str(root / "locales"), "locales"), (str(root / "LICENSE"), ".")]

binaries = []
if sys.platform == "darwin":
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if not path:
            raise SystemExit(f"{name} is required for the macOS build")
        binaries.append((os.path.realpath(path), "bin"))
elif sys.platform == "win32":
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        path = os.environ.get(name[:-4].upper() + "_EXE") or str(root / "packaging" / "vendor" / "windows" / name)
        if not os.path.isfile(path):
            raise SystemExit(f"Set {name[:-4].upper()}_EXE to the path of {name}")
        binaries.append((path, "bin"))

hidden = collect_submodules("modules.processors.frame") + [
    "run", "insightface.model_zoo.inswapper", "cv2_enumerate_cameras",
]

a = Analysis(
    [str(root / "desktop_entry.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=data,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="DeepLiveCam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "packaging" / ("DeepLiveCam.icns" if sys.platform == "darwin" else "DeepLiveCam.ico")),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DeepLiveCam")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DeepLiveCam.app",
        icon=str(root / "packaging" / "DeepLiveCam.icns"),
        bundle_identifier="com.deeplivecam.desktop",
        info_plist={
            "CFBundleDisplayName": "Deep Live Cam",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1",
            "NSCameraUsageDescription": "Deep Live Cam uses the camera for its live preview.",
            "NSMicrophoneUsageDescription": "Deep Live Cam may use audio with video processing.",
            "NSHighResolutionCapable": True,
        },
    )
