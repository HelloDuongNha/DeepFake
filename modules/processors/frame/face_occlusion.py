"""Current-frame XSeg hand/foreground protection after all face processing.

XSeg 2 weights: FaceFusion models-3.1.0, DeepFaceLab/GPL-3.0.
The model takes BGR NHWC [0,1] and returns visible-face probability.
"""
import hashlib
import threading

import cv2
import numpy as np
from insightface.utils import face_align

MODEL_FILE = 'xseg_2.onnx'
MODEL_SHA256 = 'cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042'
_SESSION = None
_LOCK = threading.Lock()
_REPORTED_ERROR = False
_LOAD_ERROR = None


def get_session():
    global _SESSION, _LOAD_ERROR
    with _LOCK:
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        if _SESSION is None:
            from modules.model_downloader import ensure_model
            from modules.processors.frame._onnx_enhancer import create_onnx_session
            try:
                path = ensure_model(MODEL_FILE)
                if path is None:
                    raise FileNotFoundError(MODEL_FILE)
                with open(path, 'rb') as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != MODEL_SHA256:
                        raise ValueError('XSeg checksum mismatch')
                _SESSION = create_onnx_session(path)
            except Exception as error:
                # Do not repeat downloads/session creation on every camera
                # frame after a failure; retry when the app is restarted.
                _LOAD_ERROR = str(error)
                raise
    return _SESSION


def restore_occluded_pixels(frame, camera, affine, visible):
    """Restore camera foreground only; use no cached frame or color adjustment."""
    if frame.shape != camera.shape or visible.shape != (256, 256):
        return frame
    if not np.isfinite(visible).all() or not np.isfinite(affine).all():
        return frame
    visible = cv2.GaussianBlur(np.clip(visible, 0., 1.).astype(np.float32), (0, 0), 1.)
    foreground = np.clip((.85 - visible) / .35, 0., 1.)
    # Expand the foreground boundary slightly so no generated edge survives
    # on the finger. Exact camera pixels are retained in the covered interior.
    foreground = cv2.dilate(foreground, np.ones((3, 3), np.uint8))
    if not np.any(foreground > .01):
        return frame
    inverse = cv2.invertAffineTransform(affine)
    corners = cv2.transform(np.array([[[0, 0], [256, 0], [256, 256], [0, 256]]],
                                    np.float32), inverse)[0]
    h, w = frame.shape[:2]
    x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int), [0, 0])
    x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int), [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame
    local = inverse.copy()
    local[:, 2] -= [x1, y1]
    alpha = cv2.warpAffine(foreground, local, (int(x2-x1), int(y2-y1)),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[:, :, None]
    roi = frame[y1:y2, x1:x2]
    roi[:] = np.rint(roi * (1. - alpha) + camera[y1:y2, x1:x2] * alpha).astype(np.uint8)
    return frame


def protect_foreground(frame, camera, face):
    global _REPORTED_ERROR
    kps = getattr(face, 'kps', None)
    if kps is None or frame.shape != camera.shape:
        return frame
    kps = np.asarray(kps, np.float32)
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        return frame
    try:
        affine = face_align.estimate_norm(kps, 128) * 2.
        crop = cv2.warpAffine(camera, affine, (256, 256), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        session = get_session()
        blob = crop[None].astype(np.float32) / 255.
        visible = session.run(None, {session.get_inputs()[0].name: blob})[0].squeeze()
        return restore_occluded_pixels(frame, camera, affine, visible)
    except Exception as error:
        if not _REPORTED_ERROR:
            print(f'[occlusion] Protection unavailable: {error}', flush=True)
            _REPORTED_ERROR = True
        return frame
