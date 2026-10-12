"""Prepare writable data and bundled command-line tools for a desktop build."""

import os

from modules.paths import CACHE_DIR, DATA_DIR, IS_PACKAGED, MODELS_DIR, RESOURCE_DIR


CORE_MODELS = (
    "inswapper_128.onnx", "GPEN-BFR-256.onnx",
    "buffalo_l/2d106det.onnx", "buffalo_l/det_10g.onnx",
    "buffalo_l/w600k_r50.onnx",
)


def prepare_desktop() -> None:
    """Create user-owned directories before importing the application runtime."""
    if not IS_PACKAGED:
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    cache_locations = {
        "XDG_CACHE_HOME": CACHE_DIR,
        "MPLCONFIGDIR": os.path.join(CACHE_DIR, "matplotlib"),
        "NUMBA_CACHE_DIR": os.path.join(CACHE_DIR, "numba"),
        "HF_HOME": os.path.join(CACHE_DIR, "huggingface"),
        "TORCH_HOME": os.path.join(CACHE_DIR, "torch"),
    }
    for name, path in cache_locations.items():
        os.environ.setdefault(name, path)

    binary_dirs = (RESOURCE_DIR, os.path.join(RESOURCE_DIR, "bin"))
    os.environ["PATH"] = os.pathsep.join(binary_dirs) + os.pathsep + os.environ.get("PATH", "")


def missing_core_models() -> list[str]:
    from modules.model_downloader import MODEL_SIZES

    missing = []
    for name in CORE_MODELS:
        target = os.path.join(MODELS_DIR, *name.split("/"))
        expected = MODEL_SIZES.get("buffalo_l/" + name if name.startswith("buffalo_l/") else name)
        if not os.path.isfile(target) or (expected and os.path.getsize(target) != expected):
            missing.append(name)
    return missing


def download_core_model(name: str) -> bool:
    from modules.model_downloader import ensure_model

    if name.startswith("buffalo_l/"):
        member = "buffalo_l/" + name
        return ensure_model(member, dest_dir=os.path.join(MODELS_DIR, "buffalo_l")) is not None
    return ensure_model(name) is not None
