"""Resource and writable paths shared by source and installed builds."""

import os
import sys

RESOURCE_DIR = getattr(
    sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
IS_PACKAGED = bool(getattr(sys, "frozen", False))


def _user_data_dir() -> str:
    override = os.environ.get("DLC_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "DeepLiveCam")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "DeepLiveCam")
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "DeepLiveCam")


DATA_DIR = _user_data_dir() if IS_PACKAGED or os.environ.get("DLC_DATA_DIR") else RESOURCE_DIR
ROOT_DIR = DATA_DIR  # InsightFace expects ROOT_DIR/models/buffalo_l.
MODELS_DIR = os.path.join(DATA_DIR, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SETTINGS_PATH = os.path.join(DATA_DIR, "switch_states.json")
