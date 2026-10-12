"""Small CPU passes for skin texture transfer and final eye/brow clarity.

Camera texture must not reintroduce the subject's brows or isolated skin marks.
Final clarity uses only the current processed frame, without another restorer
or a previous expression.
"""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def _aligned_skin_gate(height: int, width: int) -> np.ndarray:
    gate = np.full((height, width), 255, np.uint8)
    # INSwapper's ArcFace alignment puts the eyes at y ~= .46. Keep the
    # camera's eye/brow contrast out of the transferred skin residual.
    for x in (.342, .657):
        cv2.ellipse(
            gate, (round(width * x), round(height * .40)),
            (max(1, round(width * .16)), max(1, round(height * .15))),
            0, 0, 360, 0, -1,
        )
    gate = cv2.GaussianBlur(gate, (0, 0), max(.7, min(height, width) * .009))
    return gate.astype(np.float32) / 255.0


def skin_microtexture_gate(camera_crop: np.ndarray) -> np.ndarray:
    """Keep wrinkles/pores, exclude compact dark marks from camera transfer.

    This gates the additive camera residual; it never inpaints or smooths the
    generated face. Long wrinkles and broad illumination are left alone.
    """
    height, width = camera_crop.shape[:2]
    gate = _aligned_skin_gate(height, width).copy()
    gray = cv2.cvtColor(camera_crop, cv2.COLOR_BGR2GRAY)
    diameter = max(5, min(15, round(min(height, width) * .065) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    skin_only = np.full((height, width), 255, np.uint8)
    # Compact nostril shadows and tooth/lip edges are anatomy, not blemishes.
    # Keep their existing camera residual unchanged.
    for cy, ax, ay in ((.57, .13, .10), (.74, .20, .08)):
        cv2.ellipse(skin_only, (round(width * .5), round(height * cy)),
                    (max(1, round(width * ax)), max(1, round(height * ay))),
                    0, 0, 360, 0, -1)
    candidate = ((dark >= 12) & (gate > .85) & (skin_only > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    blocked = np.zeros((height, width), np.uint8)
    max_area = max(4, height * width * .0025)
    max_side = max(4, min(height, width) * .085)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        aspect = max(w, h) / max(1, min(w, h))
        if area <= max_area and max(w, h) <= max_side and aspect <= 2.2:
            roi = blocked[y:y + h, x:x + w]
            roi[labels[y:y + h, x:x + w] == label] = 255
    if np.any(blocked):
        radius = max(1, round(min(height, width) * .018))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        blocked = cv2.dilate(blocked, kernel)
        blocked = cv2.GaussianBlur(blocked, (0, 0), max(.6, radius * .4))
        blocked = np.minimum(blocked, skin_only)
        gate *= 1.0 - blocked.astype(np.float32) / 255.0
    return gate


def enhance_eye_clarity(frame: np.ndarray, face, strength: float) -> np.ndarray:
    """Sharpen current generated eye/brow edges after restoration, in place.

    The slider's 0..5 range controls this local pass. Pixels outside the two
    soft feature masks remain identical, including nose, mouth and face rim.
    """
    if face is None or strength <= 0.0 or frame.size == 0:
        return frame
    try:
        kps = np.asarray(face.kps, np.float32)
        bbox = np.asarray(face.bbox, np.float32)
        if (kps.shape != (5, 2) or bbox.shape != (4,)
                or not np.isfinite(kps).all() or not np.isfinite(bbox).all()):
            return frame
        face_width, face_height = bbox[2:] - bbox[:2]
        eye_span = float(np.linalg.norm(kps[1] - kps[0]))
        if face_width < 48 or face_height < 48 or eye_span < 16:
            return frame
        yaw = abs(float((kps[2, 0] - np.mean(kps[:2, 0])) / eye_span))
        pose_weight = float(np.clip((.38 - yaw) / .16, 0.0, 1.0))
        if pose_weight <= 0:
            return frame
        eye_line = kps[1] - kps[0]
        angle = float(np.degrees(np.arctan2(eye_line[1], eye_line[0])))
        up = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
        brows = kps[:2] + up * (face_height * .09)
        centers = np.concatenate((kps[:2], brows))
        padding = max(face_width * .18, face_height * .12) + 6
        x1 = max(0, int(np.floor(centers[:, 0].min() - padding)))
        y1 = max(0, int(np.floor(centers[:, 1].min() - padding)))
        x2 = min(frame.shape[1], int(np.ceil(centers[:, 0].max() + padding)))
        y2 = min(frame.shape[0], int(np.ceil(centers[:, 1].max() + padding)))
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        mask = np.zeros(roi.shape[:2], np.uint8)
        for index, point in enumerate(centers):
            axes = (round(face_width * (.135 if index < 2 else .15)),
                    round(face_height * (.043 if index < 2 else .045)))
            cv2.ellipse(mask, tuple(np.rint(point - [x1, y1]).astype(int)),
                        tuple(max(1, axis) for axis in axes), angle,
                        0, 360, 255, -1, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(mask, (0, 0), max(1., face_width * .007))
        alpha = alpha.astype(np.float32) / 255.
        alpha[alpha < .015] = 0.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        guide = cv2.bilateralFilter(gray, 3, 8., 1.)
        sigma = float(np.clip(face_width / 280., .8, 1.8))
        fine = guide - cv2.GaussianBlur(guide, (0, 0), sigma)
        medium = guide - cv2.GaussianBlur(guide, (0, 0), sigma * 2.)
        residual = fine * .65 + medium * .45
        residual = np.sign(residual) * np.maximum(np.abs(residual) - 2., 0.)
        gain = float(np.sqrt(np.clip(strength / 5., 0., 1.))) * pose_weight * .5
        correction = np.clip(residual * gain, -4., 4.)
        # Bound overshoot near high-contrast sclera/eyelids. There is no frame
        # cache and no spatial resampling, so blinks use this frame's geometry.
        kernel = np.ones((5, 5), np.uint8)
        proposed = np.clip(gray + correction,
                           cv2.erode(gray, kernel) - 3.,
                           cv2.dilate(gray, kernel) + 3.)
        correction = (proposed - gray) * alpha
        roi[:] = np.rint(np.clip(
            roi.astype(np.float32) + correction[:, :, None], 0, 255
        )).astype(np.uint8)
        return frame
    except (AttributeError, TypeError, ValueError, cv2.error):
        return frame
