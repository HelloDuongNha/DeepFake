"""Restore eye detail locally without replacing the current swapped identity.

GPEN-256 remains the full-face enhancer. GFPGAN is applied only within a soft
eye/eyebrow mask for large, near-frontal faces. Its changes elsewhere are
discarded, avoiding the face-wide identity changes of a second restorer.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np

from modules.processors.frame.face_enhancer import enhance_face
from modules.paths import MODELS_DIR


_MODEL_PATH = os.path.join(MODELS_DIR, "gfpgan-1024.onnx")


def _detail_weight(face, frame_shape: tuple[int, ...]) -> float:
    """Fade in at close range and fade out with yaw to avoid visible popping."""
    try:
        kps = np.asarray(face.kps, dtype=np.float32)
        bbox = np.asarray(face.bbox, dtype=np.float32)
        if kps.shape != (5, 2) or not np.isfinite(kps).all() or not np.isfinite(bbox).all():
            return 0.0
        face_width = float(bbox[2] - bbox[0])
        eye_span = float(abs(kps[1, 0] - kps[0, 0]))
        if face_width < 190.0 or eye_span < 35.0:
            return 0.0
        eye_mid = float((kps[0, 0] + kps[1, 0]) * .5)
        yaw = abs(float((kps[2, 0] - eye_mid) / eye_span))
        if yaw >= .28:
            return 0.0
        frame_height, frame_width = frame_shape[:2]
        if not (np.all(kps[:, 0] >= 0) and np.all(kps[:, 0] < frame_width)
                and np.all(kps[:, 1] >= 0) and np.all(kps[:, 1] < frame_height)):
            return 0.0
        size_weight = float(np.clip((face_width - 190.0) / 70.0, 0.0, 1.0))
        yaw_weight = float(np.clip((.28 - yaw) / .10, 0.0, 1.0))
        return size_weight * yaw_weight
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.0


def _eye_mask(shape: tuple[int, ...], face) -> np.ndarray:
    bbox = np.asarray(face.bbox, dtype=np.float32)
    face_width = float(bbox[2] - bbox[0])
    face_height = float(bbox[3] - bbox[1])
    mask = np.zeros(shape[:2], np.uint8)
    for point in np.asarray(face.kps, dtype=np.float32)[:2]:
        x, y = np.rint(point).astype(int)
        cv2.ellipse(mask, (x, y),
                    (round(face_width * .16), round(face_height * .075)),
                    0, 0, 360, 255, -1)
        cv2.ellipse(mask, (x, y - round(face_height * .09)),
                    (round(face_width * .17), round(face_height * .065)),
                    0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(3.0, face_width * .018))


def apply_eye_detail(target_face, processed_frame: np.ndarray) -> np.ndarray:
    if target_face is None or not os.path.isfile(_MODEL_PATH):
        return processed_frame
    weight = _detail_weight(target_face, processed_frame.shape)
    if weight <= 0.0:
        return processed_frame
    try:
        # Always infer from this frame. The ordinary live restorer can cache
        # previous eyes, which makes expressions appear frozen or jumpy.
        detailed = enhance_face(
            processed_frame.copy(), detected_faces=[target_face],
            use_temporal_cache=False,
        )
        if detailed.shape != processed_frame.shape:
            return processed_frame
        alpha = _eye_mask(processed_frame.shape, target_face).astype(np.float32)
        alpha *= (.65 * weight / 255.0)
        alpha = alpha[:, :, None]
        return np.clip(processed_frame.astype(np.float32) * (1.0 - alpha)
                       + detailed.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    except Exception as exc:
        print(f"[eye-detail] Keeping GPEN frame after detail error: {exc}", flush=True)
        return processed_frame


@dataclass
class _LiveDetail:
    """Only the changed eye pixels from one completed restoration."""

    source_key: int
    captured_at: float
    frame_shape: tuple[int, ...]
    kps: np.ndarray
    rect: tuple[int, int, int, int]
    original: np.ndarray
    delta: np.ndarray
    mask: np.ndarray


_live_executor: ThreadPoolExecutor | None = None
_live_future: Future | None = None
_live_result: _LiveDetail | None = None
_live_last_submit = 0.0
_LIVE_SAMPLE_INTERVAL = .35


def _source_key(source_face) -> int:
    embedding = getattr(source_face, "embedding", None)
    if embedding is None:
        return id(source_face)
    return hash(np.asarray(embedding, dtype=np.float32).tobytes())


def _restore_eye_sample(frame: np.ndarray, kps: np.ndarray, bbox: np.ndarray,
                        source_key: int, captured_at: float) -> _LiveDetail | None:
    face = SimpleNamespace(kps=kps, bbox=bbox)
    detailed = enhance_face(
        frame.copy(), detected_faces=[face], use_temporal_cache=False,
    )
    if detailed.shape != frame.shape:
        return None
    mask = _eye_mask(frame.shape, face)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    original = frame[y:y + height, x:x + width].copy()
    delta = (detailed[y:y + height, x:x + width].astype(np.float32)
             - original.astype(np.float32))
    return _LiveDetail(
        source_key, captured_at, frame.shape, kps, (x, y, width, height),
        original, delta, mask[y:y + height, x:x + width].copy(),
    )


def _apply_live_result(frame: np.ndarray, face, result: _LiveDetail,
                       weight: float, now: float) -> np.ndarray:
    age = now - result.captured_at
    if frame.shape != result.frame_shape or age < 0.0 or age > .75:
        return frame
    current_kps = np.asarray(face.kps, dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(
        result.kps, current_kps, method=cv2.LMEDS,
    )
    if transform is None or not np.isfinite(transform).all():
        return frame
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    rotation = float(abs(np.arctan2(transform[1, 0], transform[0, 0])))
    fitted = result.kps @ transform[:, :2].T + transform[:, 2]
    error = float(np.mean(np.linalg.norm(fitted - current_kps, axis=1)))
    face_width = float(face.bbox[2] - face.bbox[0])
    if not (.75 <= scale <= 1.33 and rotation < .35
            and error < max(5.0, face_width * .035)):
        return frame

    current_mask = _eye_mask(frame.shape, face)
    points = cv2.findNonZero(current_mask)
    if points is None:
        return frame
    x, y, width, height = cv2.boundingRect(points)
    sx, sy, _, _ = result.rect
    local = transform.copy()
    local[:, 2] += transform[:, :2] @ np.array([sx, sy], np.float64)
    local[:, 2] -= np.array([x, y], np.float64)
    size = (width, height)
    previous = cv2.warpAffine(result.original, local, size,
                              flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    old_mask = cv2.warpAffine(result.mask, local, size,
                              flags=cv2.INTER_LINEAR, borderValue=0)
    mask = np.minimum(old_mask, current_mask[y:y + height, x:x + width])
    if not np.any(mask):
        return frame
    current = frame[y:y + height, x:x + width]
    alpha = mask.astype(np.float32) / 255.0
    difference = np.abs(previous.astype(np.float32) - current.astype(np.float32))
    mae = float(np.sum(difference * alpha[:, :, None])
                / max(1.0, float(np.sum(alpha)) * 3.0))
    # A blink or changed expression must use the current GPEN pixels rather
    # than restoring eye detail from an older, incompatible expression.
    expression_weight = float(np.clip((18.0 - mae) / 10.0, 0.0, 1.0))
    age_weight = float(np.clip((.75 - age) / .20, 0.0, 1.0))
    gain = .65 * weight * expression_weight * age_weight
    if gain <= .01:
        return frame
    delta = cv2.warpAffine(result.delta, local, size,
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    current[:] = np.clip(current.astype(np.float32)
                         + delta * (alpha[:, :, None] * gain), 0, 255).astype(np.uint8)
    return frame


def apply_eye_detail_live(target_face, processed_frame: np.ndarray,
                          source_face) -> np.ndarray:
    """Use completed detail without waiting for GFPGAN on the camera thread."""
    global _live_executor, _live_future, _live_result, _live_last_submit
    if target_face is None or source_face is None or not os.path.isfile(_MODEL_PATH):
        return processed_frame
    weight = _detail_weight(target_face, processed_frame.shape)
    if weight <= 0.0:
        return processed_frame
    now = time.monotonic()
    key = _source_key(source_face)
    if _live_future is not None and _live_future.done():
        try:
            _live_result = _live_future.result()
        except Exception as exc:
            _live_result = None
            print(f"[eye-detail] Async restoration skipped: {exc}", flush=True)
        _live_future = None
    if _live_future is None and now - _live_last_submit >= _LIVE_SAMPLE_INTERVAL:
        if _live_executor is None:
            _live_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eye-detail")
        _live_future = _live_executor.submit(
            _restore_eye_sample, processed_frame.copy(),
            np.asarray(target_face.kps, dtype=np.float32).copy(),
            np.asarray(target_face.bbox, dtype=np.float32).copy(), key, now,
        )
        _live_last_submit = now
    if _live_result is None or _live_result.source_key != key:
        return processed_frame
    return _apply_live_result(processed_frame, target_face, _live_result, weight, now)


def stop_live_eye_detail() -> None:
    """Release pending camera work when its preview closes."""
    global _live_executor, _live_future, _live_result, _live_last_submit
    if _live_executor is not None:
        _live_executor.shutdown(wait=True, cancel_futures=True)
    _live_executor = None
    _live_future = None
    _live_result = None
    _live_last_submit = 0.0


def warmup_live_eye_detail() -> None:
    """Load and warm GFPGAN before camera workers start processing frames."""
    if not os.path.isfile(_MODEL_PATH):
        return
    try:
        from modules.processors.frame.face_enhancer import get_face_enhancer
        from modules.processors.frame._onnx_enhancer import warmup_session
        warmup_session(get_face_enhancer())
    except Exception as exc:
        print(f"[eye-detail] Warmup skipped: {exc}", flush=True)
