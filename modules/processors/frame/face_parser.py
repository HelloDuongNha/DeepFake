"""Optional CelebAMask-HQ face parsing guard for the live swap mask.

Uses yakhyo/face-parsing ResNet18 ONNX (19 labels). The model is optional:
the inexpensive landmark/hairline guard remains the fallback on Mac.
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

import modules.globals

_MODEL_NAME = "face_parsing_resnet18.onnx"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "models", _MODEL_NAME)
_SESSION = None
_SESSION_PATH = None
_LOCK = threading.Lock()
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_HAIR_LABEL = np.uint8(17)


def model_path() -> str:
    return os.environ.get("DLC_HAIR_PARSER_MODEL", _MODEL_PATH)


def _get_session():
    global _SESSION, _SESSION_PATH
    path = model_path()
    mode = os.environ.get("DLC_HAIR_PARSING", "off").lower()
    if mode in {"0", "off", "false", "no"} or not os.path.isfile(path):
        return None
    with _LOCK:
        if _SESSION is not None and _SESSION_PATH == path:
            return _SESSION
        import onnxruntime as ort
        available = ort.get_available_providers()
        requested = modules.globals.execution_providers or []
        providers = [provider for provider in requested if provider in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        _SESSION = ort.InferenceSession(path, providers=providers)
        _SESSION_PATH = path
        return _SESSION


def _face_roi(frame: np.ndarray, face) -> tuple[int, int, int, int] | None:
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    width, height = x2 - x1, y2 - y1
    if width < 16 or height < 16:
        return None
    h, w = frame.shape[:2]
    left = max(0, int(np.floor(x1 - 0.20 * width)))
    right = min(w, int(np.ceil(x2 + 0.20 * width)))
    top = max(0, int(np.floor(y1 - 0.45 * height)))
    bottom = min(h, int(np.ceil(y2 + 0.12 * height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def parse_face_skin(frame: np.ndarray, face):
    """Return (ROI, uint8 skin mask), or None when parsing is unavailable.

    The result is a subtractive hair guard. The landmark mask still defines
    the face region; parsing only removes pixels confidently classified as
    hair. This avoids carving holes in forehead, eyebrows or cheeks when a
    small/profile face receives an imperfect semantic label.
    """
    roi = _face_roi(frame, face)
    if roi is None:
        return None
    if getattr(face, "_dlc_parse_frame", None) is frame:
        return getattr(face, "_dlc_parse_result", None)
    try:
        session = _get_session()
        if session is None:
            return None
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(cv2.resize(crop, (512, 512)), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 19:
            return None
        labels = np.argmax(output[0], axis=0).astype(np.uint8)
        labels = cv2.resize(labels, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        hair = (labels == _HAIR_LABEL).astype(np.uint8)
        # Expand only the detected hair by one pixel. Non-hair labels stay
        # untouched, so parser uncertainty cannot punch holes in the face.
        hair = cv2.dilate(hair, np.ones((3, 3), dtype=np.uint8))
        allowed = np.where(hair > 0, 0, 255).astype(np.uint8)
        result = (roi, allowed)
        face._dlc_parse_frame = frame
        face._dlc_parse_result = result
        return result
    except Exception:
        # Bad or unsupported optional model must not stop the live stream.
        return None


def skin_guard_for_crop(parsed, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Place a parsed face ROI into a paste-back crop without full-frame work."""
    # Parsing is a subtractive hair guard. Pixels outside the parser ROI are
    # unknown and therefore must pass through (255); treating them as 0 turns
    # the ROI rectangle itself into a destructive crop mask.
    guard = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    (px1, py1, px2, py2), skin = parsed
    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)
    if ix2 > ix1 and iy2 > iy1:
        guard[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = (
            skin[iy1 - py1:iy2 - py1, ix1 - px1:ix2 - px1]
        )
    return guard
