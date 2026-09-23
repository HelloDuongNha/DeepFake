# --- START OF FILE globals.py ---

import os
import math
from typing import List, Dict, Any


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return max(minimum, min(maximum, value)) if math.isfinite(value) else default
    except ValueError:
        return default

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = os.environ.get("DLC_BLEND_MODE", "alpha").lower() == "poisson"
poisson_max_lab_distance: float = _env_float("DLC_POISSON_MAX_LAB_DISTANCE", 32.0, 5.0, 100.0)
color_match: bool = os.environ.get("DLC_COLOR_MATCH", "1").lower() not in {
    "0", "false", "off", "no"
}
# A narrow feather plus a small inward erosion avoids the grey halo that the
# old 3 px/3 px defaults left around the jaw and forehead.  Both values stay
# configurable so a difficult camera or a different crop can be tuned without
# changing code.
mask_blur: float = _env_float("DLC_MASK_BLUR", 1.5, 0.0, 16.0)
mask_erosion: int = _env_int("DLC_MASK_EROSION", 4, 0, 16)
enhancer_interval: int = _env_int("DLC_ENHANCER_INTERVAL", 1, 1, 6)
# Amount of high-frequency texture copied from the camera crop after GPEN or
# GFPGAN.  0 keeps the restoration output untouched; 1 fully restores the
# camera's fine detail.  A moderate default keeps eyebrows and wrinkles while
# avoiding visible camera noise.
detail_strength: float = _env_float(
    "DLC_DETAIL_STRENGTH",
    _env_float("DLC_ENHANCER_FIDELITY", 0.35, 0.0, 1.0),
    0.0,
    1.0,
)
film_grain_strength: float = _env_float("DLC_FILM_GRAIN", 0.35, 0.0, 1.0)
# Fraction of the aligned crop above the forehead that is kept out of the
# generic paste-back mask.  The live detector normally exposes only five
# landmarks, so this lightweight guard protects real hair without running a
# second segmentation model on every frame.
hairline_guard: float = _env_float("DLC_HAIRLINE_GUARD", 0.16, 0.0, 0.35)
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
