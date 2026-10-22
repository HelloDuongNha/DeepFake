"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from modules.processors.frame.face_masking import (
    TemporalLandmarkSmoother, is_plausible_face_geometry,
    track_face_landmarks,
)
from PySide6.QtCore import (
    QByteArray,
    QSize,
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap, QIcon
from modules.swap_models import SWAP_MODELS
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    detect_one_face_near,
    ensure_landmarks,
    get_one_face,
    get_source_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 700
ROOT_WIDTH = 640

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360
# Keep the original compact preview window. Capture resolution is configurable
# independently through DLC_CAPTURE_WIDTH/HEIGHT, so resizing the window does
# not force a full-screen-sized widget.
LIVE_CAPTURE_WIDTH = int(os.environ.get("DLC_CAPTURE_WIDTH", "1920"))
LIVE_CAPTURE_HEIGHT = int(os.environ.get("DLC_CAPTURE_HEIGHT", "1080"))
LIVE_CAPTURE_FPS = int(os.environ.get("DLC_CAPTURE_FPS", "60"))

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QListWidget { background-color: #262626; border: 1px solid #444; color: #e6e6e6; }
QListWidget::item:selected { background-color: #245799; border-radius: 6px; }
QListWidget::item:disabled { color: #888; }
QMainWindow, QDialog { background-color: #1e1e1e; color: #e6e6e6; }
QScrollArea, QWidget#mainContent { background-color: #1e1e1e; border: none; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; }

QGroupBox {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 4px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #3a3a3a;
}
QPushButton#secondary:hover { background-color: #4a4a4a; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 36px; height: 18px;
    border-radius: 9px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #2a2a2a;
    border: 2px dashed #444;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #262626;
    border-radius: 10px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = os.environ.get("DLC_SOURCE_DIR")
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    if new_size == (w, h):
        return image
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return gpu_resize(image, dsize=new_size, interpolation=interpolation)


def fill_image_to_size(image, width: int, height: int):
    """Center-crop a BGR frame to fill the preview without stretching."""
    if width <= 0 or height <= 0:
        return image
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return image
    if source_width * height > source_height * width:
        crop_width = max(1, round(source_height * width / height))
        left = (source_width - crop_width) // 2
        image = image[:, left:left + crop_width]
    else:
        crop_height = max(1, round(source_width * height / width))
        top = (source_height - crop_height) // 2
        image = image[top:top + crop_height, :]
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_CUBIC
    return gpu_resize(image, dsize=(width, height), interpolation=interpolation)


def fit_live_preview(image, width: int, height: int):
    """Portrait fills the window; landscape keeps the complete camera frame."""
    fitter = fill_image_to_size if modules.globals.live_portrait else fit_image_to_size
    return fitter(image, width, height)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    with Image.open(image_path) as opened:
        image=ImageOps.exif_transpose(opened)
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    from modules.user_settings import save
    return save(modules.globals)


def load_switch_states():
    from modules.user_settings import read, apply
    state=read()
    apply(state,modules.globals)
    modules.globals.map_faces=False
    modules.globals.eye_detail=False
    selected=os.environ.get("DLC_SWAP_MODEL",state.get("swapper_model",modules.globals.swapper_model))
    if selected in SWAP_MODELS:
        modules.globals.swapper_model=selected
    elif modules.globals.swapper_model=="alphaface_256":
        modules.globals.swapper_model="hyperswap_256"
    modules.globals.mouth_mask_size=0.
    modules.globals.mouth_mask=False
    modules.globals.show_mouth_mask_box=False


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration ─────────────────────────────────────────


def selected_camera_index(combo: QComboBox) -> Optional[int]:
    """Return only the integer device id stored behind a camera label."""
    value = combo.currentData(Qt.ItemDataRole.UserRole)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version}"
        )
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        root.setObjectName("mainContent")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())
        layout.addStretch(1)

        # Action buttons
        # Camera controls remain outside the scrollable options.

        # Camera selection
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll, 1)
        outer_layout.addWidget(self._build_camera_card())
        self.setCentralWidget(outer)

        # Keep status delivery compatible without occupying the camera UI.
        self._status_label = QLabel(self)
        self._status_label.hide()
        if modules.globals.window_geometry:
            self.restoreGeometry(QByteArray.fromBase64(modules.globals.window_geometry.encode()))
        if modules.globals.source_path:
            if not self._set_source(modules.globals.source_path, remember=False):
                modules.globals.source_path=None
        self._refresh_history_button()

    def _on_mask_toggled(self, enabled: bool) -> None:
        preview = _WEBCAM_PREVIEW
        if preview is None or not hasattr(preview, "_processed_queue"):
            return
        # Discard queued images from the previous mode. Show the last camera
        # frame immediately while an in-flight model inference finishes.
        while True:
            try:
                preview._processed_queue.get_nowait()
            except queue.Empty:
                break
        if not enabled and preview._processing_worker is not None:
            raw = preview._processing_worker._last_camera_frame
            if _is_valid_camera_frame(raw):
                preview._show_raw_frame(raw)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (128, 128))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        src_row.addWidget(self.btn_select_source)
        self.btn_history=QPushButton(_("Recent faces"))
        self.btn_history.clicked.connect(self._on_history)
        src_row.addWidget(self.btn_history)
        src_col.addLayout(src_row)

        self.target_label = QLabel()  # Legacy file-preview callbacks; no target UI.
        row.addStretch(1)
        row.addLayout(src_col)
        row.addStretch(1)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(12, 8, 12, 10)
        grid.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Seamless / Poisson Blend",
                               "Use seamlessClone for color-adaptive face blending; slower than alpha blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")
        self.sw_live_mirror = make(
            "live_mirror", "Mirror camera",
            "Flip the live camera horizontally so left and right follow your movement",
        )
        self.sw_mask = make(
            "mask_enabled", "Mask",
            "On: apply the selected face. Off: show the original live camera",
        )
        self.sw_mask.toggled.connect(self._on_mask_toggled)
        self.sw_inner_mouth = make(
            "preserve_inner_mouth", "Natural teeth",
            "Keep real teeth and the inner mouth when opening wide; leave lips and surrounding skin swapped",
        )

        self.sw_natural_eyes = make("natural_eyes", "Natural eyes",
            "Keep camera eyeballs inside the eyelids; leave brows and surrounding skin swapped")

        self.sw_natural_nose = make("natural_nose", "Natural nose",
            "Keep camera nose tip and nostrils; changes the lower nose to your own anatomy")

        self.sw_occlusion = make("occlusion_protection", "Hand protection",
            "Keep camera foreground over the mask; adds an XSeg inference per frame")

        self.sw_eyelid_repair = make("eyelid_repair", "Eyelid repair",
            "With Natural eyes: include the current eyelid rim to avoid doubled eyelid edges; preserves your eyelid shape")
        self.sw_nose_edge_repair = make("nose_edge_repair", "Nose edge repair",
            "Suppress narrow generated nose ridges before enhancement; block camera nose texture and keep the repaired model contours")

        self.sw_forehead_guard = make("forehead_guard", "Forehead guard",
            "At turns, keep your current forehead, eyebrows and skin between brows to avoid distorted patches. Frontal mask stays unchanged.")
        self.sw_eyelid_repair.setEnabled(modules.globals.natural_eyes)
        self.sw_natural_eyes.toggled.connect(self.sw_eyelid_repair.setEnabled)
        self.sw_nose_edge_repair.setEnabled(not modules.globals.natural_nose)
        self.sw_natural_nose.toggled.connect(lambda checked:self.sw_nose_edge_repair.setEnabled(not checked))
        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
            self.sw_live_mirror,
            self.sw_mask,
            self.sw_inner_mouth,
            self.sw_natural_eyes, self.sw_eyelid_repair,
            self.sw_natural_nose, self.sw_nose_edge_repair,
            self.sw_occlusion, self.sw_forehead_guard,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown
        enhancer_row = (len(items) + 1) // 2
        grid.setRowMinimumHeight(enhancer_row, 38)
        grid.setRowMinimumHeight(enhancer_row + 1, 38)
        enhancer_label = QLabel(_("Face Enhancer:"))
        grid.addWidget(enhancer_label, enhancer_row, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.setFixedHeight(38)
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-256", "RestoreFormer++ (quality)"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-256"
            modules.globals.fp_ui["face_enhancer_gpen512"] = False
            modules.globals.fp_ui["face_enhancer_gpen256"] = True
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        if modules.globals.fp_ui.get("face_enhancer_restoreformer", False):
            initial = "RestoreFormer++ (quality)"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, enhancer_row, 1)

        grid.addWidget(QLabel(_("Face model:")), enhancer_row + 1, 0)
        self.cb_swapper = QComboBox()
        self.cb_swapper.setFixedHeight(38)
        self.cb_swapper.addItems([item[0] for item in SWAP_MODELS.values()])
        self.cb_swapper.setCurrentText(
            SWAP_MODELS[modules.globals.swapper_model][0]
        )
        self.cb_swapper.setToolTip(_("Choose the face model; stop Live before changing it"))
        self.cb_swapper.currentTextChanged.connect(self._on_swapper_change)
        grid.addWidget(self.cb_swapper, enhancer_row + 1, 1)


        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setContentsMargins(12, 8, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, modules.globals.opacity, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(
            0.0, 5.0, modules.globals.sharpness, 100,
            self._on_sharpness_change,
        )
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 10)

        layout.addWidget(QLabel(_("Select Camera:")))
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            for camera_index, camera_name in zip(
                self._camera_indices, self._camera_names
            ):
                self.cb_camera.addItem(camera_name, int(camera_index))
            if platform.system() == "Darwin" and self.cb_camera.count() > 1:
                self.cb_camera.setCurrentIndex(1)
            cam_ok = True
        preferred=modules.globals.preferred_camera_index
        position=self.cb_camera.findData(preferred) if preferred is not None else -1
        if position>=0:self.cb_camera.setCurrentIndex(position)
        self.cb_camera.currentIndexChanged.connect(self._on_camera_change)
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        layout.addWidget(self.cb_camera, 1)

        self.btn_orientation = QPushButton("Dọc" if modules.globals.live_portrait else "Ngang")
        self.btn_orientation.setToolTip("Dọc: lấp đầy cửa sổ. Ngang: toàn bộ khung camera, không cắt ảnh.")
        self.btn_orientation.clicked.connect(self._on_orientation_change)
        layout.addWidget(self.btn_orientation)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        layout.addWidget(self.btn_live)

        return card

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path:self._set_source(path)

    def _set_source(self, path, remember=True):
        global _RECENT_SOURCE_DIR
        try:
            if not os.path.isfile(path) or not is_image(path):return False
            preview=render_image_preview(path,(128,128))
        except (OSError,ValueError):
            update_status("Cannot open source image.")
            return False
        source_changed=modules.globals.source_path!=os.path.abspath(path)
        modules.globals.source_path=os.path.abspath(path)
        if source_changed:self._on_mask_toggled(False)
        _RECENT_SOURCE_DIR=os.path.dirname(path)
        self.source_label.setPixmap(preview)
        self.source_label.setText("")
        if remember:
            from modules.user_settings import history
            modules.globals.source_history=history([modules.globals.source_path]+modules.globals.source_history)
            save_switch_states()
        self._refresh_history_button()
        return True

    def _refresh_history_button(self):
        self.btn_history.setText(_("Recent faces")+f" ({len(modules.globals.source_history)})")
        self.btn_history.setEnabled(bool(modules.globals.source_history))

    def _clear_source_history(self):
        modules.globals.source_history=[]
        modules.globals.source_path=None
        self._on_mask_toggled(False)
        self.source_label.clear()
        self.source_label.setText(_("Source face"))
        save_switch_states()
        self._refresh_history_button()

    def _on_history(self):
        dialog=QDialog(self)
        dialog.setWindowTitle(_("Recent faces"))
        dialog.resize(520,400)
        layout=QVBoxLayout(dialog)
        label=QLabel(_("Double-click a face to use it. Missing files are unavailable."))
        label.setWordWrap(True);layout.addWidget(label)
        album=QListWidget()
        album.setViewMode(QListWidget.ViewMode.IconMode)
        album.setIconSize(QSize(96,96));album.setGridSize(QSize(120,132))
        album.setResizeMode(QListWidget.ResizeMode.Adjust)
        album.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for path in modules.globals.source_history:
            item=QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole,path);item.setToolTip(path)
            try:item.setIcon(QIcon(render_image_preview(path,(96,96))))
            except (OSError,ValueError):item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            album.addItem(item)
        layout.addWidget(album,1)
        buttons=QHBoxLayout()
        use=QPushButton(_("Use selected"));use.setEnabled(False)
        clear=QPushButton(_("Clear history"))
        clear.setToolTip(_("Forget recent faces and the last selected face. Original image files are not deleted."))
        close=QPushButton(_("Close"))
        def choose(item=None):
            item=item or album.currentItem()
            if item and self._set_source(item.data(Qt.ItemDataRole.UserRole)):dialog.accept()
        def clear_history():
            self._clear_source_history();album.clear();use.setEnabled(False)
        album.currentItemChanged.connect(lambda current,previous:use.setEnabled(current is not None))
        album.itemDoubleClicked.connect(choose)
        use.clicked.connect(lambda:choose())
        clear.clicked.connect(clear_history);close.clicked.connect(dialog.reject)
        for button in (clear,use,close):buttons.addWidget(button)
        layout.addLayout(buttons);dialog.exec()

    def _on_camera_change(self, index):
        modules.globals.preferred_camera_index=selected_camera_index(self.cb_camera)
        save_switch_states()

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            self._set_source(temp_path)
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (128, 128)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
            "RestoreFormer++ (quality)": "face_enhancer_restoreformer",
        }
        selected = key_map.get(choice)
        states = dict(modules.globals.fp_ui)
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512", "face_enhancer_restoreformer"):
            states[key] = key == selected
        # Publish one selection, without transient all-off states between
        # toggles. The processing worker snapshots this at frame boundaries.
        modules.globals.fp_ui = states
        save_switch_states()

    def _on_swapper_change(self, choice: str) -> None:
        if modules.globals.webcam_preview_running:
            self.cb_swapper.blockSignals(True)
            self.cb_swapper.setCurrentText(
                SWAP_MODELS[modules.globals.swapper_model][0]
            )
            self.cb_swapper.blockSignals(False)
            update_status("Stop Live before changing the face model.")
            return
        from modules.processors.frame.face_swapper import select_swapper_model
        select_swapper_model(next(key for key, item in SWAP_MODELS.items() if item[0] == choice))
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        save_switch_states()
        pct = int(value * 100)
        if pct == 0:
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        save_switch_states()
        update_status(f"Sharpness set to {value:.1f}")

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_orientation_change(self) -> None:
        modules.globals.live_portrait = not modules.globals.live_portrait
        self.btn_orientation.setText("Dọc" if modules.globals.live_portrait else "Ngang")
        save_switch_states()
        if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
            _WEBCAM_PREVIEW.apply_orientation()

    def _on_live(self) -> None:
        camera_index = selected_camera_index(self.cb_camera)
        if camera_index is None:
            update_status("No camera available")
            return
        print(
            f"[webcam] Selected device index={camera_index} "
            f"name={self.cb_camera.currentText()!r}",
            flush=True,
        )
        modules.globals.map_faces = False
        modules.globals.preferred_camera_index=camera_index
        save_switch_states()
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        from modules.face_analyser import get_face_analyser
        from modules.processors.frame.face_swapper import get_face_swapper
        get_face_analyser()
        get_face_swapper()
        _open_webcam_preview(camera_index)

    def closeEvent(self, event):
        modules.globals.window_geometry=bytes(self.saveGeometry().toBase64()).decode()
        selected=selected_camera_index(self.cb_camera)
        if selected is not None:modules.globals.preferred_camera_index=selected
        save_switch_states()
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_source_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


def _is_valid_camera_frame(frame) -> bool:
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] >= 3
        and frame.size > 0
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


def face_swapper_metrics() -> tuple[str, str]:
    from modules.processors.frame.face_swapper import (
        get_last_swap_coverage, get_last_model_delta,
    )
    coverage = get_last_swap_coverage()
    model_delta = get_last_model_delta()
    return (
        f"{coverage:.2f}" if coverage is not None else "n/a",
        f"{model_delta:.2f}" if model_delta is not None else "n/a",
    )


def _frame_or_camera_fallback(candidate, camera_frame, stage: str):
    if _is_valid_camera_frame(candidate):
        return candidate
    print(
        f"[webcam] {stage} returned an empty frame; showing raw camera frame",
        flush=True,
    )
    return camera_frame


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    rawFrame = Signal(object)

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        consecutive_failures = 0
        sent_initial_preview = False
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or not _is_valid_camera_frame(frame):
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 30 == 0:
                    print(
                        f"[webcam] Camera read returned an empty frame "
                        f"(consecutive={consecutive_failures})",
                        flush=True,
                    )
                if consecutive_failures >= 150:
                    print("[webcam] Camera stopped after repeated empty frames", flush=True)
                    self._stop.set()
                    break
                self.msleep(20)
                continue
            consecutive_failures = 0
            # The raw preview is needed only while the first processed frame
            # loads. Queuing every full-resolution frame into Qt's UI thread
            # wastes bandwidth and can delay processed preview updates.
            if not sent_initial_preview:
                self.rawFrame.emit(frame)
                sent_initial_preview = True
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps
        self._last_camera_frame = None

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_loop()
            except Exception as error:
                # A model failure must not take down the camera thread. Raw
                # frames continue to be displayed while the pipeline retries.
                print(
                    f"[webcam] Processing error; keeping raw preview: {error}",
                    flush=True,
                )
                fallback = self._last_camera_frame
                if _is_valid_camera_frame(fallback):
                    try:
                        self._pq.put_nowait(fallback)
                    except queue.Full:
                        try:
                            self._pq.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._pq.put_nowait(fallback)
                        except queue.Full:
                            pass
                self.msleep(10)

    def _run_loop(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        # Detect on every frame that actually reaches the processing worker.
        # The capture queue intentionally drops frames under load, so two
        # processed frames are not guaranteed to be consecutive camera
        # frames. Relying on optical flow across that gap made tracking fail
        # after the capture resolution was raised to 720p.
        det_interval = 1
        landmark_smoother = TemporalLandmarkSmoother()
        previous_gray = None
        tracking_lost = False
        detector_miss_streak = 0
        last_stable_output = None
        last_tracking_status = None
        last_swap_status = None
        last_stage_report = 0.0
        last_processor_sample = time.monotonic()
        processed_since_sample = 0
        measured_processing_fps = 0.0

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
            enhancer_flags = dict(modules.globals.fp_ui)
            swap_applied = False
            swap_delta = None
            self._last_camera_frame = frame
            # Process a copy because face_swapper writes its paste-back in
            # place. Keep ``frame`` pristine for the raw-camera fallback and
            # the immediate preview signal.
            temp_frame = frame.copy()
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            frame_source_path=modules.globals.source_path
            mask_enabled = modules.globals.mask_enabled and bool(frame_source_path)
            if not mask_enabled:
                if not frame_source_path:
                    source_image=None
                    last_source_path=None
                # Bypass every detector, swapper, restorer and anatomy pass.
                cached_target_face = None
                cached_many_faces = None
                previous_gray = None
                last_stable_output = None
                tracking_lost = False
                detector_miss_streak = 0
                landmark_smoother.reset()
            elif not modules.globals.map_faces:
                if frame_source_path != last_source_path:
                    last_source_path=frame_source_path
                    source_image=get_source_face(imread_unicode(frame_source_path))
                    last_stable_output=None

                det_count += 1
                current_gray = None
                if not modules.globals.many_faces:
                    current_gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
                fresh_detection = False
                detector_ran = False
                active_det_interval = det_interval
                if cached_target_face is not None:
                    bbox = getattr(cached_target_face, "bbox", None)
                    if bbox is not None:
                        bbox = np.asarray(bbox, dtype=np.float32)
                        if bbox.shape == (4,) and min(bbox[2:] - bbox[:2]) < 96.0:
                            # Small/distant faces have fewer reliable optical-flow
                            # pixels, so refresh them from the detector every frame.
                            active_det_interval = 1
                if det_count % active_det_interval == 0:
                    detector_ran = True
                    if modules.globals.many_faces:
                        landmark_smoother.reset()
                        previous_gray = None
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        detected = detect_one_face_fast(
                            temp_frame, cached_target_face
                        )
                        if (detected is not None
                                and not is_plausible_face_geometry(detected)):
                            detected = None
                        # Full-frame RetinaFace can miss a heavily yawed or
                        # partly covered face even though it remains close to
                        # the last tracked position. Retry on that enlarged
                        # local crop before treating the face as absent.
                        if detected is None and cached_target_face is not None:
                            detected = detect_one_face_near(
                                temp_frame, cached_target_face
                            )
                            if (detected is not None
                                    and not is_plausible_face_geometry(detected)):
                                detected = None
                        cached_many_faces = None
                        if detected is not None:
                            # The final cheek mask uses the stable 106-point
                            # outline. Computing it on detector frames also
                            # keeps mouth opening and head turns represented
                            # directly instead of freezing an old contour.
                            ensure_landmarks(temp_frame, [detected])
                            # After LK has failed, accept the new detector
                            # geometry as a clean reacquisition. Feeding a
                            # stale pre-motion face back into the smoother can
                            # otherwise keep the mask detached indefinitely.
                            if tracking_lost:
                                landmark_smoother.reset()
                                reference_face = None
                            else:
                                reference_face = cached_target_face
                            cached_target_face = landmark_smoother.update(
                                detected, time.monotonic(), reference_face=reference_face
                            )
                            tracking_lost = False
                            detector_miss_streak = 0
                            fresh_detection = True
                        else:
                            detector_miss_streak += 1
                        # On a temporary miss, keep the last face and let
                        # optical flow carry it. Never drop the swap merely
                        # because yaw, a yawn or distance hid landmarks for a
                        # detector cycle.

                if (cached_target_face is not None and not fresh_detection
                        and current_gray is not None and previous_gray is not None):
                    tracked = track_face_landmarks(
                        previous_gray, current_gray, cached_target_face
                    )
                    if not tracked:
                        tracking_lost = True
                        # Pitch or a fast turn can temporarily invalidate LK.
                        # Reacquire in this frame instead of displaying a stale
                        # transform or waiting for the next detector cadence.
                        # A crop around the tracked face makes a strong yaw
                        # much larger for RetinaFace and preserves identity
                        # continuity. Only repeat a full-frame pass when this
                        # frame has not already had one.
                        emergency_face = detect_one_face_near(
                            temp_frame, cached_target_face
                        )
                        if emergency_face is None and not detector_ran:
                            emergency_face = detect_one_face_fast(temp_frame)
                        if (emergency_face is not None
                                and not is_plausible_face_geometry(emergency_face)):
                            emergency_face = None
                        if emergency_face is not None:
                            ensure_landmarks(temp_frame, [emergency_face])
                            landmark_smoother.reset()
                            cached_target_face = landmark_smoother.update(
                                emergency_face, time.monotonic(),
                                reference_face=None,
                            )
                            tracking_lost = False
                            detector_miss_streak = 0
                previous_gray = current_gray

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None and not tracking_lost:
                    cached_faces = [cached_target_face]

                # Fast detection skips dense landmarks; attach them when
                # any camera anatomy preservation feature needs its contour.
                # The helper no-ops if the current detection already has them.
                if (cached_faces and (
                    modules.globals.mouth_mask
                    or modules.globals.preserve_inner_mouth
                    or modules.globals.natural_eyes
                    or modules.globals.natural_nose
                    or modules.globals.nose_edge_repair
                    or modules.globals.forehead_guard
                )):
                    ensure_landmarks(temp_frame, cached_faces)

                tracking_status = (
                    "source_missing" if source_image is None else
                    "tracking_lost" if tracking_lost else
                    "no_face" if not cached_faces else "tracking"
                )
                if tracking_status != last_tracking_status:
                    print(f"[webcam] Face state: {tracking_status}", flush=True)
                    last_tracking_status = tracking_status

                hold_last_safe = tracking_lost and last_stable_output is not None
                if hold_last_safe:
                    temp_frame = last_stable_output.copy()
                # Swap must precede restoration, even after UI toggles change
                # the shared processor list.
                live_processors = sorted(
                    frame_processors, key=lambda p: p.NAME != "DLC.FACE-SWAPPER"
                )
                active_enhancers = [
                    p for p in live_processors
                    if p.NAME.startswith("DLC.FACE-ENHANCER")
                    and enhancer_flags.get(p.__name__.split(".")[-1], False)
                ]
                # A stale saved state may contain several enabled enhancers.
                # The UI offers a single choice, so never chain restorers.
                chosen_enhancer = active_enhancers[0] if active_enhancers else None
                nose_reference=None
                for fp in ([] if hold_last_safe else live_processors):
                    if fp.NAME.startswith("DLC.FACE-ENHANCER") and fp is not chosen_enhancer:
                        continue
                    if fp.NAME.startswith("DLC.FACE-ENHANCER") and not swap_applied:
                        continue
                    if (fp is chosen_enhancer and modules.globals.nose_edge_repair
                            and not modules.globals.natural_nose and not modules.globals.many_faces):
                        nose_reference=temp_frame.copy()
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if enhancer_flags.get("face_enhancer", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if enhancer_flags.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if enhancer_flags.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-RESTOREFORMER":
                        temp_frame = fp.process_frame(
                            None, temp_frame, detected_faces=cached_faces
                        )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                applied = fp.get_last_swap_status() == "applied"
                                swap_applied = swap_applied or applied
                                if applied and hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None and not tracking_lost:
                            diagnostic_due = time.monotonic() - last_stage_report >= 3.0
                            if diagnostic_due:
                                diagnostic_raw = temp_frame.copy()
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            swap_applied = fp.get_last_swap_status() == "applied"
                            if diagnostic_due and swap_applied:
                                box = np.rint(cached_target_face.bbox).astype(int)
                                h, w = temp_frame.shape[:2]
                                x1, y1 = max(0, box[0]), max(0, box[1])
                                x2, y2 = min(w, box[2]), min(h, box[3])
                                if x2 > x1 and y2 > y1:
                                    diagnostic_roi = (slice(y1, y2), slice(x1, x2))
                                    swap_delta = float(np.mean(cv2.absdiff(
                                        temp_frame[diagnostic_roi],
                                        diagnostic_raw[diagnostic_roi],
                                    )))
                            if (
                                swap_applied and hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        swap_status = (
                            fp.get_last_swap_status() if cached_faces else "no_target"
                        )
                        if swap_status != last_swap_status:
                            print(f"[webcam] Swap state: {swap_status}", flush=True)
                            last_swap_status = swap_status
                        if swapped_bboxes and chosen_enhancer is None:
                            temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
                if (swap_applied and not hold_last_safe
                        and cached_target_face is not None
                        and modules.globals.natural_eyes
                        and not modules.globals.many_faces):
                    from modules.processors.frame.face_detail import preserve_eye_apertures
                    camera_eye_frame = gpu_flip(frame, 1) if modules.globals.live_mirror else frame
                    temp_frame = preserve_eye_apertures(temp_frame, camera_eye_frame, cached_target_face,
                                                        repair_eyelids=modules.globals.eyelid_repair)
                if (swap_applied and not hold_last_safe and cached_target_face is not None
                        and modules.globals.nose_edge_repair and not modules.globals.natural_nose
                        and nose_reference is not None and not modules.globals.many_faces):
                    from modules.processors.frame.face_detail import repair_nose_edges
                    temp_frame=repair_nose_edges(temp_frame,nose_reference,cached_target_face)
                if (swap_applied and not hold_last_safe
                        and cached_target_face is not None
                        and modules.globals.natural_nose
                        and not modules.globals.many_faces):
                    from modules.processors.frame.face_detail import preserve_natural_nose
                    camera_nose_frame = gpu_flip(frame, 1) if modules.globals.live_mirror else frame
                    temp_frame = preserve_natural_nose(temp_frame, camera_nose_frame, cached_target_face)
                if (
                    swap_applied and cached_target_face is not None
                    and not modules.globals.many_faces
                    and modules.globals.preserve_inner_mouth
                    and not modules.globals.mouth_mask
                ):
                    from modules.processors.frame.face_masking import (
                        preserve_open_mouth_interior,
                    )
                    camera_mouth_frame = (
                        gpu_flip(frame, 1) if modules.globals.live_mirror else frame
                    )
                    temp_frame = preserve_open_mouth_interior(
                        temp_frame, camera_mouth_frame, cached_target_face
                    )
                if (swap_applied and not hold_last_safe and cached_target_face is not None
                        and modules.globals.forehead_guard and not modules.globals.many_faces):
                    from modules.processors.frame.upper_face import preserve_upper_face
                    current_camera=gpu_flip(frame,1) if modules.globals.live_mirror else frame
                    temp_frame=preserve_upper_face(temp_frame,current_camera,cached_target_face)
                if (swap_applied and not hold_last_safe
                        and cached_target_face is not None
                        and modules.globals.occlusion_protection
                        and not modules.globals.many_faces):
                    from modules.processors.frame.face_occlusion import protect_foreground
                    camera_foreground = gpu_flip(frame, 1) if modules.globals.live_mirror else frame
                    temp_frame = protect_foreground(temp_frame, camera_foreground, cached_target_face)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if enhancer_flags.get("face_enhancer", False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if enhancer_flags.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            if swap_delta is not None:
                final_delta = float(np.mean(cv2.absdiff(
                    temp_frame[diagnostic_roi], diagnostic_raw[diagnostic_roi]
                )))
                name = chosen_enhancer.NAME if chosen_enhancer else "None"
                coverage, model_delta = face_swapper_metrics()
                print(
                    f"[webcam] Face pixels: swap_delta={swap_delta:.2f} "
                    f"final_delta={final_delta:.2f} enhancer={name} "
                    f"opacity={modules.globals.opacity:.2f} "
                    f"mask_coverage={coverage} "
                    f"processing_fps={measured_processing_fps:.1f} "
                    f"model_delta={model_delta} "
                    f"detector_misses={detector_miss_streak} "
                    f"face_size={x2 - x1}x{y2 - y1}",
                    flush=True,
                )
                last_stage_report = time.monotonic()

            if (not modules.globals.mask_enabled
                    or modules.globals.source_path!=frame_source_path):
                # Mask/source can change while CoreML is processing this frame.
                temp_frame = gpu_flip(frame, 1) if modules.globals.live_mirror else frame.copy()
                swap_applied = False
                last_stable_output = None

            temp_frame = _frame_or_camera_fallback(
                temp_frame, frame, "processing pipeline"
            )

            # Cache only frames produced from current, verified face geometry.
            # This protects the preview from stale-mask distortion while the
            # face is covered or outside the detector's valid pose range.
            if (not modules.globals.map_faces and swap_applied and not tracking_lost
                    and detector_miss_streak == 0):
                last_stable_output = temp_frame.copy()

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass

            processed_since_sample += 1
            sample_time = time.monotonic()
            sample_duration = sample_time - last_processor_sample
            if sample_duration >= 2.0:
                measured_processing_fps = processed_since_sample / sample_duration
                processed_since_sample = 0
                last_processor_sample = sample_time


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(1, 1)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self._image_label, 1)

        self._has_processed_frame = False
        self._stop_event = threading.Event()
        self._capture_worker = None
        self._processing_worker = None
        self._timer = None

        self._cap = VideoCapturer(int(camera_index))
        if not self._cap.start(LIVE_CAPTURE_WIDTH, LIVE_CAPTURE_HEIGHT, LIVE_CAPTURE_FPS):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        self.apply_orientation()
        if modules.globals.live_window_geometry:
            self.restoreGeometry(QByteArray.fromBase64(modules.globals.live_window_geometry.encode()))
        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._capture_worker.rawFrame.connect(self._show_raw_frame)
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def apply_orientation(self) -> None:
        portrait = modules.globals.live_portrait
        self.setWindowTitle("Live Preview • portrait 9:16" if portrait
                            else "Live Preview • landscape • full camera")
        screen = self.screen() or QApplication.primaryScreen()
        area = screen.availableGeometry() if screen else None
        max_width = max(1, area.width() - 80) if area else 1200
        max_height = max(1, area.height() - 80) if area else 800
        ratio = 9 / 16 if portrait else self._cap.actual_width / max(1, self._cap.actual_height)
        height = min(960 if portrait else 720, max_height, max_width / ratio)
        self._image_label.clear()
        self.resize(max(1, round(height * ratio)), max(1, round(height)))

    def _show_raw_frame(self, bgr_frame) -> None:
        """Display the camera before the first AI-processed frame is ready."""
        if (self._has_processed_frame and modules.globals.mask_enabled) or not _is_valid_camera_frame(bgr_frame):
            return
        if modules.globals.live_mirror:
            bgr_frame = gpu_flip(bgr_frame, 1)
        fitted = fit_live_preview(bgr_frame, self._image_label.width(), self._image_label.height())
        if _is_valid_camera_frame(fitted):
            self._image_label.setPixmap(_bgr_to_qpixmap(fitted))

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        if not _is_valid_camera_frame(bgr_frame):
            print("[webcam] Ignoring empty processed preview frame", flush=True)
            return
        bgr_frame = fit_live_preview(bgr_frame, self._image_label.width(), self._image_label.height())
        if not _is_valid_camera_frame(bgr_frame):
            return
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))
        self._has_processed_frame = True

    def closeEvent(self, event) -> None:
        modules.globals.live_window_geometry=bytes(self.saveGeometry().toBase64()).decode()
        save_switch_states()
        self._stop_event.set()
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        for worker in (self._capture_worker, self._processing_worker):
            if worker is None:
                continue
            try:
                # A CoreML inference/model load can exceed two seconds.
                # Dropping this window then destroys a live QThread and aborts
                # the process; reopening can also race the shared model state.
                worker.wait()
            except Exception:
                pass
        from modules.processors.frame.eye_detail import stop_live_eye_detail
        stop_live_eye_detail()
        try:
            self._cap.release()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()



def shutdown_webcam() -> None:
    """Join camera workers before the application releases Qt objects."""
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    blend_mode = os.environ.get("DLC_BLEND_MODE", "").lower()
    if blend_mode in {"alpha", "poisson"}:
        _MAIN.sw_poisson.setChecked(blend_mode == "poisson")
        modules.globals.poisson_blend = blend_mode == "poisson"
    if os.environ.get("DLC_DISABLE_POISSON", "0").lower() in {
        "1", "true", "on", "yes"
    }:
        _MAIN.sw_poisson.setChecked(False)
        _MAIN.sw_poisson.setEnabled(False)
        _MAIN.sw_poisson.setToolTip(
            _("Disabled in the current profile to prevent live color flicker")
        )
        modules.globals.poisson_blend = False

    from modules.user_settings import read as read_preferences
    enhancer=os.environ.get("DLC_ENHANCER")
    if enhancer is None and "fp_ui" not in read_preferences():
        enhancer="GPEN-256" if platform.system()=="Darwin" and platform.machine()=="arm64" else None
    if enhancer == "GPEN-512":
        enhancer = "GPEN-256"
    if enhancer in {"None", "GFPGAN", "GPEN-256", "RestoreFormer++ (quality)"}:
        _MAIN.cb_enhancer.setCurrentText(enhancer)

    source_image = os.environ.get("DLC_SOURCE_IMAGE")
    if source_image and os.path.isfile(source_image) and is_image(source_image):
        _MAIN._set_source(source_image)

    camera_index = os.environ.get("DLC_CAMERA_INDEX")
    if camera_index is not None:
        try:
            camera_position = _MAIN._camera_indices.index(int(camera_index))
            _MAIN.cb_camera.setCurrentIndex(camera_position)
        except (ValueError, IndexError):
            pass

    if os.environ.get("DLC_AUTO_LIVE") == "1" and modules.globals.source_path:
        QTimer.singleShot(0, _MAIN._on_live)

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
