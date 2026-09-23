import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize


_FOREHEAD_HEIGHT_EMA: float | None = None
_FOREHEAD_FACE_WIDTH_EMA: float | None = None
_FOREHEAD_AXIS_RATIO = 0.68
_FOREHEAD_EMA_ALPHA = 0.25


def is_plausible_face_geometry(face: Face) -> bool:
    """Reject detector/LK layouts that cannot represent one human face."""
    if face is None:
        return False
    kps = getattr(face, "kps", None)
    bbox = getattr(face, "bbox", None)
    if kps is None or bbox is None:
        return False
    kps = np.asarray(kps, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if (kps.shape != (5, 2) or bbox.size < 4
            or not np.all(np.isfinite(kps)) or not np.all(np.isfinite(bbox[:4]))):
        return False
    x1, y1, x2, y2 = bbox[:4]
    width, height = float(x2 - x1), float(y2 - y1)
    if width <= 12.0 or height <= 12.0:
        return False
    margin_x, margin_y = width * 0.30, height * 0.30
    if (np.any(kps[:, 0] < x1 - margin_x)
            or np.any(kps[:, 0] > x2 + margin_x)
            or np.any(kps[:, 1] < y1 - margin_y)
            or np.any(kps[:, 1] > y2 + margin_y)):
        return False
    eye_width = float(np.linalg.norm(kps[1] - kps[0]))
    mouth_width = float(np.linalg.norm(kps[4] - kps[3]))
    eye_min_x = float(np.min(kps[:2, 0]))
    eye_max_x = float(np.max(kps[:2, 0]))
    nose_x = float(kps[2, 0])
    profile_layout = (
        nose_x < eye_min_x - 0.02 * width
        or nose_x > eye_max_x + 0.02 * width
    )
    # In a true side profile the projected eye and mouth spans collapse. Allow
    # that only when the nose is clearly outside the eye interval; a frontal
    # detector collapse is still rejected by the original stricter limits.
    min_eye_ratio = 0.055 if profile_layout else 0.12
    min_mouth_ratio = 0.025 if profile_layout else 0.04
    if not (min_eye_ratio * width <= eye_width <= 0.95 * width):
        return False
    if not (min_mouth_ratio * width <= mouth_width <= 0.80 * width):
        return False
    eye_y = float(np.mean(kps[:2, 1]))
    mouth_y = float(np.mean(kps[3:5, 1]))
    nose_y = float(kps[2, 1])
    if mouth_y <= eye_y + 0.07 * height:
        return False
    if nose_y < eye_y - 0.08 * height or nose_y > mouth_y + 0.18 * height:
        return False
    return True


def reset_forehead_height_ema() -> None:
    """Reset live forehead geometry when tracking starts a new face/session."""
    global _FOREHEAD_HEIGHT_EMA, _FOREHEAD_FACE_WIDTH_EMA
    _FOREHEAD_HEIGHT_EMA = None
    _FOREHEAD_FACE_WIDTH_EMA = None


def _smooth_forehead_height(raw_height: float, face_width: float) -> float:
    """EMA-filter the scale-relative forehead height without pose ratios."""
    global _FOREHEAD_HEIGHT_EMA, _FOREHEAD_FACE_WIDTH_EMA
    raw_height = max(2.0, float(raw_height))
    face_width = max(1.0, float(face_width))
    if (_FOREHEAD_HEIGHT_EMA is None or _FOREHEAD_FACE_WIDTH_EMA is None
            or face_width > _FOREHEAD_FACE_WIDTH_EMA * 2.0
            or face_width < _FOREHEAD_FACE_WIDTH_EMA * 0.5):
        _FOREHEAD_HEIGHT_EMA = raw_height
        _FOREHEAD_FACE_WIDTH_EMA = face_width
    else:
        alpha = _FOREHEAD_EMA_ALPHA
        _FOREHEAD_HEIGHT_EMA += alpha * (raw_height - _FOREHEAD_HEIGHT_EMA)
        _FOREHEAD_FACE_WIDTH_EMA += alpha * (face_width - _FOREHEAD_FACE_WIDTH_EMA)
    return float(_FOREHEAD_HEIGHT_EMA)


def _estimate_pose_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Estimate yaw/pitch capable affine, falling back to a similarity model."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    transform = None
    if len(source) >= 3:
        transform, _ = cv2.estimateAffine2D(source, target, method=cv2.LMEDS)
        if transform is not None and np.all(np.isfinite(transform)):
            singular = np.linalg.svd(transform[:, :2], compute_uv=False)
            # Permit perspective-like foreshortening from yaw/pitch while
            # rejecting a detector matrix capable of exploding the face.
            if (singular[-1] < 0.55 or singular[0] > 1.65
                    or singular[0] / max(singular[-1], 1e-6) > 1.85
                    or np.linalg.det(transform[:, :2]) <= 0):
                transform = None
    if transform is None and len(source) >= 2:
        transform, _ = cv2.estimateAffinePartial2D(
            source, target, method=cv2.LMEDS
        )
    if transform is None or not np.all(np.isfinite(transform)):
        return None
    return transform.astype(np.float32)


def _track_face_region_translation(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    face: Face,
) -> bool:
    """Fallback translation from many texture corners when facial KPS fail."""
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return False
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return False
    frame_h, frame_w = previous_gray.shape[:2]
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(frame_w, int(np.ceil(bbox[2])))
    y2 = min(frame_h, int(np.ceil(bbox[3])))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return False
    feature_mask = np.zeros_like(previous_gray, dtype=np.uint8)
    cv2.ellipse(
        feature_mask,
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (max(3, int((x2 - x1) * 0.42)), max(3, int((y2 - y1) * 0.44))),
        0, 0, 360, 255, -1,
    )
    features = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=48, qualityLevel=0.015,
        minDistance=5, mask=feature_mask, blockSize=5,
    )
    if features is None or len(features) < 4:
        return False
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, features, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if tracked is None or status is None:
        return False
    valid = status.reshape(-1).astype(bool)
    if np.count_nonzero(valid) < 4:
        return False
    shifts = tracked[:, 0] - features[:, 0]
    median_shift = np.median(shifts[valid], axis=0)
    residual = np.linalg.norm(shifts - median_shift, axis=1)
    limit = max(2.0, 0.05 * max(x2 - x1, y2 - y1))
    inliers = valid & (residual <= limit)
    if np.count_nonzero(inliers) < 4:
        return False
    displacement = np.median(shifts[inliers], axis=0).astype(np.float32)
    if np.linalg.norm(displacement) > 0.45 * max(x2 - x1, y2 - y1):
        return False
    face.kps = np.asarray(face.kps, dtype=np.float32) + displacement
    face.bbox = bbox[:4] + np.tile(displacement, 2)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None:
        face.landmark_2d_106 = (
            np.asarray(landmarks, dtype=np.float32) + displacement
        )
    return is_plausible_face_geometry(face)


def track_face_landmarks(previous_gray: np.ndarray, current_gray: np.ndarray, face: Face) -> bool:
    """Follow the five landmarks between detector hits using optical flow."""
    if previous_gray is None or current_gray is None or face is None:
        return False
    kps = getattr(face, "kps", None)
    if kps is None or np.asarray(kps).shape != (5, 2):
        return False
    previous = np.asarray(kps, dtype=np.float32).reshape(-1, 1, 2)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous, None,
        winSize=(35, 35), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if next_points is None or status is None:
        return _track_face_region_translation(
            previous_gray, current_gray, face
        )
    valid = status.reshape(-1).astype(bool)
    if np.count_nonzero(valid) < 2 or not np.all(np.isfinite(next_points[valid])):
        return _track_face_region_translation(
            previous_gray, current_gray, face
        )
    previous_points = previous[:, 0]
    tracked_points = next_points[:, 0]
    # Optical flow is incremental. A full affine here can accumulate a small
    # shear on every detector gap until the aligned face becomes a trapezoid
    # across the background. Keep inter-frame tracking to a similarity
    # transform; fresh detector frames still carry real yaw/pitch geometry.
    transform, _ = cv2.estimateAffinePartial2D(
        previous_points[valid], tracked_points[valid], method=cv2.LMEDS
    )
    if transform is not None:
        scale = float(np.sqrt(max(0.0, np.linalg.det(transform[:, :2]))))
        if (not np.all(np.isfinite(transform)) or scale < 0.72 or scale > 1.38):
            transform = None
    if transform is not None and np.all(np.isfinite(transform)):
        predicted = cv2.transform(previous_points[None], transform)[0]
        eye_width = max(1.0, float(np.linalg.norm(
            previous_points[1] - previous_points[0]
        )))
        residual = np.linalg.norm(tracked_points - predicted, axis=1)
        inliers = valid & (residual <= max(3.0, eye_width * 0.22))
        updated = predicted
        updated[inliers] = tracked_points[inliers]
    else:
        shifts = tracked_points - previous_points
        displacement = np.median(shifts[valid], axis=0)
        residual = np.linalg.norm(shifts - displacement, axis=1)
        eye_width = max(1.0, float(np.linalg.norm(
            previous_points[1] - previous_points[0]
        )))
        inliers = valid & (residual <= max(3.0, eye_width * 0.22))
        updated = previous_points + displacement
        updated[inliers] = tracked_points[inliers]
    old_kps = np.asarray(face.kps, dtype=np.float32).copy()
    old_bbox = None
    old_landmarks = None
    if getattr(face, "bbox", None) is not None:
        old_bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    if getattr(face, "landmark_2d_106", None) is not None:
        old_landmarks = np.asarray(face.landmark_2d_106, dtype=np.float32).copy()
    face.kps = updated
    bbox = getattr(face, "bbox", None)
    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32)
        if transform is not None and np.all(np.isfinite(transform)):
            corners = np.array([
                [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                [bbox[2], bbox[3]], [bbox[0], bbox[3]],
            ], dtype=np.float32)
            warped = cv2.transform(corners[None], transform)[0]
            face.bbox = np.array([
                warped[:, 0].min(), warped[:, 1].min(),
                warped[:, 0].max(), warped[:, 1].max(),
            ], dtype=np.float32)
        else:
            face.bbox = bbox + np.tile(displacement, 2)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None:
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if transform is not None and np.all(np.isfinite(transform)):
            face.landmark_2d_106 = cv2.transform(landmarks[None], transform)[0]
        else:
            face.landmark_2d_106 = landmarks + displacement
    if not is_plausible_face_geometry(face):
        face.kps = old_kps
        if old_bbox is not None:
            face.bbox = old_bbox
        if old_landmarks is not None:
            face.landmark_2d_106 = old_landmarks
        return _track_face_region_translation(
            previous_gray, current_gray, face
        )
    return True


class TemporalLandmarkSmoother:
    """Apply light EMA without rejecting valid detector geometry."""

    def __init__(self):
        self.reset()

    def reset(self):
        reset_forehead_height_ema()
        self._kps = None
        self._landmarks = None
        self._bbox = None
        self._time = None

    def update(self, face: Face, now: float, reference_face: Face | None = None) -> Face:
        points = getattr(face, "kps", None)
        if points is None:
            self.reset()
            return face
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            self.reset()
            return face
        raw_bbox = getattr(face, "bbox", None)
        raw_bbox = np.asarray(raw_bbox, dtype=np.float32) if raw_bbox is not None else None
        if raw_bbox is not None and (raw_bbox.shape != (4,) or not np.all(np.isfinite(raw_bbox))):
            raw_bbox = None
        prior = getattr(reference_face, "kps", None)
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float32)
            if prior.shape != (5, 2) or not np.all(np.isfinite(prior)):
                prior = None
        prior_bbox = getattr(reference_face, "bbox", None)
        if prior_bbox is not None:
            prior_bbox = np.asarray(prior_bbox, dtype=np.float32)
            if (prior_bbox.shape != (4,)
                    or not np.all(np.isfinite(prior_bbox))):
                prior_bbox = None
        first = self._kps is None
        previous_smoothed = None if first else self._kps.copy()
        geometry_transform = None
        if first:
            smoothed = points.copy()
            stable_bbox = raw_bbox.copy() if raw_bbox is not None else None
            alpha = 1.0
        else:
            reference = prior if prior is not None else self._kps
            geometry_transform = _estimate_pose_affine(reference, points)
            cleaned = points.copy()
            if (geometry_transform is not None
                    and np.all(np.isfinite(geometry_transform))):
                predicted = cv2.transform(reference[None], geometry_transform)[0]
                reference_eye_width = max(
                    1.0, float(np.linalg.norm(reference[1] - reference[0]))
                )
                residual = np.linalg.norm(points - predicted, axis=1)
                median_residual = float(np.median(residual))
                mad = float(np.median(np.abs(residual - median_residual)))
                outlier_limit = max(
                    5.0, reference_eye_width * 0.45,
                    median_residual + 3.0 * max(1.0, mad),
                )
                outliers = residual > outlier_limit
                cleaned[outliers] = predicted[outliers]
            motion = float(np.mean(np.linalg.norm(cleaned - reference, axis=1)))
            eye_width = max(1.0, float(np.linalg.norm(points[1] - points[0])))
            alpha = float(np.clip(0.62 + motion / eye_width, 0.62, 0.92))
            smoothed = reference + alpha * (cleaned - reference)
            # Optical flow updates ``reference_face`` on every camera frame,
            # while ``self._bbox`` is refreshed only on detector frames. Use
            # that current tracked box as the smoothing anchor so a fast turn
            # cannot pull the mask back toward a stale temple position for one
            # frame when detection refreshes.
            bbox_anchor = prior_bbox if prior_bbox is not None else self._bbox
            if raw_bbox is not None and bbox_anchor is not None:
                stable_bbox = bbox_anchor + alpha * (raw_bbox - bbox_anchor)
            else:
                stable_bbox = raw_bbox.copy() if raw_bbox is not None else bbox_anchor
        face.kps = smoothed
        if stable_bbox is not None:
            face.bbox = stable_bbox
        self._kps = smoothed.copy()
        self._bbox = stable_bbox.copy() if stable_bbox is not None else None
        self._time = now
        landmarks = getattr(face, "landmark_2d_106", None)
        if landmarks is not None:
            landmarks = np.asarray(landmarks, dtype=np.float32)
            if (not first and self._landmarks is not None
                    and self._landmarks.shape == landmarks.shape):
                dense_transform = _estimate_pose_affine(
                    previous_smoothed, smoothed
                )
                if dense_transform is not None and np.all(np.isfinite(dense_transform)):
                    predicted = cv2.transform(
                        self._landmarks[None], dense_transform
                    )[0]
                    dense_residual = np.linalg.norm(landmarks - predicted, axis=1)
                    median_dense = float(np.median(dense_residual))
                    dense_mad = float(np.median(np.abs(
                        dense_residual - median_dense
                    )))
                    dense_threshold = max(
                        6.0,
                        float(np.linalg.norm(smoothed[1] - smoothed[0])) * 0.55,
                        median_dense + 3.5 * max(1.0, dense_mad),
                    )
                    landmarks = landmarks.copy()
                    landmarks[dense_residual > dense_threshold] = predicted[
                        dense_residual > dense_threshold
                    ]
                    landmarks = predicted + alpha * (landmarks - predicted)
                else:
                    landmarks = self._landmarks + alpha * (landmarks - self._landmarks)
            face.landmark_2d_106 = landmarks
            self._landmarks = landmarks.copy()
        else:
            self._landmarks = None
        return face

def apply_color_transfer(source, target):
    """Transfer the target's LAB tone to ``source`` (unmasked compatibility API)."""
    return match_color_lab(source, target)


def match_color_lab(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match colour statistics in LAB using only the supplied face mask.

    The masked path avoids sampling background, hair or clothing when a swap
    is blended.  All channel operations are vectorized NumPy expressions; the
    tiny face ROI is the only region converted back to BGR.
    """
    if source is None or target is None or source.size == 0 or target.size == 0:
        return source
    if source.shape[:2] != target.shape[:2]:
        source = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    src = np.asarray(source, dtype=np.uint8)
    ref = np.asarray(target, dtype=np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None:
        valid = np.ones(src_lab.shape[:2], dtype=bool)
    else:
        valid = np.asarray(mask) > 16
        if valid.shape != src_lab.shape[:2] or int(np.count_nonzero(valid)) < 16:
            return src
    if reference_mask is None:
        reference_valid = valid
    else:
        reference_valid = np.asarray(reference_mask) > 16
        if reference_valid.shape != valid.shape or int(np.count_nonzero(reference_valid)) < 16:
            reference_valid = valid
    src_values = src_lab[valid]
    ref_values = ref_lab[reference_valid]
    src_mean = src_values.mean(axis=0, dtype=np.float32)
    ref_mean = ref_values.mean(axis=0, dtype=np.float32)
    src_std = np.maximum(src_values.std(axis=0), 1.0)
    ref_std = ref_values.std(axis=0)
    # Keep luminance transfer conservative so Poisson still handles local
    # gradients instead of receiving a globally over-contrasted crop.
    ref_std = np.maximum(ref_std, 1.0)
    gain = np.clip(ref_std / src_std, 0.85, 1.15)
    offset = np.clip(ref_mean - src_mean, -12.0, 12.0)
    corrected = (src_lab - src_mean) * gain + src_mean + offset
    return cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_skin_tone_lab(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Conservatively match live face colour from corresponding cheek skin.

    Median LAB offsets avoid the highlight/background sensitivity of global
    mean/std transfer. Geometry and contrast are untouched; only a bounded
    colour offset is applied to the generated face.
    """
    if source is None or target is None or source.size == 0 or target.size == 0:
        return source
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return source
    if source.shape[:2] != target.shape[:2]:
        source = cv2.resize(
            source, (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    src = np.asarray(source, dtype=np.uint8)
    ref = np.asarray(target, dtype=np.uint8)
    height, width = ref.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx + 0.5) / max(1.0, float(width))
    ny = (yy + 0.5) / max(1.0, float(height))
    # Both cheeks between the lower eye line and upper mouth line. This avoids
    # hair, eyebrows, lips, nostrils, neck and clothing.
    cheek_roi = (
        (ny >= 0.54) & (ny <= 0.76)
        & (((nx >= 0.30) & (nx <= 0.45))
           | ((nx >= 0.55) & (nx <= 0.70)))
    )
    if mask is not None:
        supplied = np.asarray(mask)
        if supplied.shape == (height, width):
            cheek_roi &= supplied > 96
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Ignore deep shadows and clipped highlights in either image.
    valid = (
        cheek_roi & (src_lab[:, :, 0] > 25) & (src_lab[:, :, 0] < 235)
        & (ref_lab[:, :, 0] > 25) & (ref_lab[:, :, 0] < 235)
    )
    if int(np.count_nonzero(valid)) < 24:
        return src
    def bounded_delta(selection: np.ndarray) -> np.ndarray:
        value = (
            np.median(ref_lab[selection], axis=0)
            - np.median(src_lab[selection], axis=0)
        )
        value[0] = np.clip(value[0], -36.0, 36.0)
        value[1:] = np.clip(value[1:], -18.0, 18.0)
        return value

    global_delta = bounded_delta(valid)
    left_valid = valid & (nx < 0.50)
    right_valid = valid & (nx >= 0.50)
    left_delta = (
        bounded_delta(left_valid)
        if int(np.count_nonzero(left_valid)) >= 12 else global_delta
    )
    right_delta = (
        bounded_delta(right_valid)
        if int(np.count_nonzero(right_valid)) >= 12 else global_delta
    )
    # Independently anchor both cheeks, then interpolate across the nose. This
    # follows side lighting during yaw without introducing a seam at x=0.5.
    side_mix = np.clip((nx - 0.34) / 0.32, 0.0, 1.0)
    side_mix = side_mix * side_mix * (3.0 - 2.0 * side_mix)
    delta_map = (
        left_delta.reshape(1, 1, 3) * (1.0 - side_mix[:, :, None])
        + right_delta.reshape(1, 1, 3) * side_mix[:, :, None]
    )
    corrected = src_lab + delta_map * (0.90 * strength)

    # A single LAB offset cannot follow uneven webcam illumination (for
    # example, one cheek next to a bright window). Transfer only the smooth,
    # low-frequency part of the remaining colour difference. Facial texture,
    # wrinkles and eyebrows are far above this spatial frequency and remain
    # sourced from the swap/enhancer rather than being copied as a ghost.
    support = (
        (nx >= 0.16) & (nx <= 0.84) & (ny >= 0.18) & (ny <= 0.86)
        & (src_lab[:, :, 0] > 25) & (src_lab[:, :, 0] < 235)
        & (ref_lab[:, :, 0] > 25) & (ref_lab[:, :, 0] < 235)
    )
    if mask is not None and np.asarray(mask).shape == (height, width):
        support &= np.asarray(mask) > 128
    weights = support.astype(np.float32)
    if int(np.count_nonzero(support)) >= 48:
        sigma = max(4.0, min(height, width) / 11.0)
        normalizer = cv2.GaussianBlur(
            weights, (0, 0), sigmaX=sigma, sigmaY=sigma
        )
        residual = ref_lab - corrected
        residual[:, :, 0] = np.clip(residual[:, :, 0], -24.0, 24.0)
        residual[:, :, 1:] = np.clip(residual[:, :, 1:], -14.0, 14.0)
        local = np.zeros_like(residual)
        for channel in range(3):
            numerator = cv2.GaussianBlur(
                residual[:, :, channel] * weights,
                (0, 0), sigmaX=sigma, sigmaY=sigma,
            )
            local[:, :, channel] = numerator / np.maximum(normalizer, 1e-3)
        local[:, :, 0] = np.clip(local[:, :, 0], -14.0, 14.0)
        local[:, :, 1:] = np.clip(local[:, :, 1:], -8.0, 8.0)
        corrected += local * (0.72 * strength)
    return cv2.cvtColor(
        np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )


def harmonize_mask_boundary(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    preserve_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match low-frequency colour only along the inside of an alpha edge.

    A swapped crop can contain a faint dark source-hair tint at its outermost
    pixels. Alpha feathering alone makes that tint translucent but still
    visible. Adding a smooth local colour delta makes the edge meet the camera
    skin while preserving source high-frequency detail such as eyebrow hairs,
    pores, and wrinkles.
    """
    if (source is None or target is None or mask is None
            or source.size == 0 or target.size == 0 or mask.size == 0
            or source.shape != target.shape
            or mask.shape != source.shape[:2]):
        return source
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    # Measure inward from the visual midpoint of the feather. Starting from
    # the first non-zero alpha would spend the whole correction width in the
    # nearly invisible outer tail and leave the actual seam untouched.
    interior = (alpha > 0.50).astype(np.uint8)
    if not np.any(interior):
        return source
    distance = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
    band_width = max(3.0, min(source.shape[:2]) * 0.04)
    edge_weight = np.clip((band_width - distance) / band_width, 0.0, 1.0)
    # Do not alter pixels outside the mask. Ramp up through the translucent
    # fringe so an almost-zero alpha pixel cannot sample unrelated background.
    edge_weight *= np.clip(alpha / 0.35, 0.0, 1.0)
    if float(edge_weight.max()) <= 0.0:
        return source
    sigma = max(2.0, min(source.shape[:2]) / 48.0)
    source_f = source.astype(np.float32)
    target_f = target.astype(np.float32)
    source_low = cv2.GaussianBlur(source_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    target_low = cv2.GaussianBlur(target_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Keep the conservative low-frequency correction used for normal lighting.
    # An unlimited correction made the whole edge follow bright highlights and
    # produced the uneven skin reported under direct sunlight.
    delta = np.clip(target_low - source_low, -42.0, 42.0)
    corrected = source_f + delta * (edge_weight * 0.92)[:, :, None]

    # A narrow black/white source-hair streak is high-frequency and survives
    # the smooth correction above. Replace only strongly mismatched pixels at
    # the outer edge with their camera pixels. Normal pores stay below the
    # threshold, and the caller can explicitly protect real eyebrow hairs.
    mismatch = np.max(np.abs(source_f - target_f), axis=2)
    mismatch_weight = np.clip((mismatch - 22.0) / 54.0, 0.0, 1.0)
    # Inspect a slightly deeper but capped band for narrow 2-4 px source-hair
    # curves. This wider reach only affects large per-pixel mismatches; normal
    # skin and texture still receive zero detail replacement.
    detail_band = max(5.0, min(12.0, min(source.shape[:2]) * 0.065))
    detail_edge = np.clip((detail_band - distance) / detail_band, 0.0, 1.0)
    detail_edge *= np.clip(alpha / 0.35, 0.0, 1.0)
    detail_weight = np.clip(detail_edge * 1.20, 0.0, 1.0) * mismatch_weight

    # The lower oval crosses from generated chin into the real chin/neck. Even
    # a modest colour difference can show as a horizontal arc there, so make
    # the outer few pixels camera-exact independent of the artifact threshold.
    # The vertical ramp confines this to the bottom boundary and avoids
    # changing cheeks, mouth, or the upper-face work above.
    occupied_rows = np.flatnonzero(np.any(interior, axis=1))
    if occupied_rows.size:
        top = float(occupied_rows[0])
        bottom = float(occupied_rows[-1])
        span = max(1.0, bottom - top)
        yy = np.arange(source.shape[0], dtype=np.float32)[:, None]
        lower_ramp = np.clip(
            (yy - (top + span * 0.68)) / (span * 0.22), 0.0, 1.0
        )
        lower_edge_weight = np.clip(detail_edge * 1.35, 0.0, 1.0)
        detail_weight = np.maximum(
            detail_weight, lower_edge_weight * lower_ramp
        )

        # Seal the matching left/right cheek arcs as well. Select only the
        # lateral quarters of the oval and fade the effect in vertically, so
        # forehead highlights and the center of the face are untouched.
        occupied_cols = np.flatnonzero(np.any(interior, axis=0))
        if occupied_cols.size:
            left = float(occupied_cols[0])
            right = float(occupied_cols[-1])
            width = max(1.0, right - left)
            xx = np.arange(source.shape[1], dtype=np.float32)[None, :]
            left_side = np.clip(
                ((left + width * 0.28) - xx) / (width * 0.14), 0.0, 1.0
            )
            right_side = np.clip(
                (xx - (right - width * 0.28)) / (width * 0.14), 0.0, 1.0
            )
            side_gate = np.maximum(left_side, right_side)
            # Begin above the outer brow line so the same seam sealing reaches
            # the temples. Eyebrow hairs are protected later by preserve_mask,
            # and the face parser still prevents this edge from entering hair.
            upper_gate = np.clip(
                (yy - (top + span * 0.08)) / (span * 0.14), 0.0, 1.0
            )
            lower_gate = np.clip(
                ((top + span * 0.88) - yy) / (span * 0.14), 0.0, 1.0
            )
            cheek_gate = side_gate * upper_gate * lower_gate
            side_edge_weight = np.clip(detail_edge * 1.35, 0.0, 1.0)
            detail_weight = np.maximum(
                detail_weight, side_edge_weight * cheek_gate
            )
    if (preserve_mask is not None
            and np.asarray(preserve_mask).shape == mask.shape):
        detail_weight *= 1.0 - (
            np.asarray(preserve_mask, dtype=np.float32) / 255.0
        )
    corrected = (
        corrected * (1.0 - detail_weight[:, :, None])
        + target_f * detail_weight[:, :, None]
    )
    return np.clip(corrected, 0, 255).astype(np.uint8)

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        face_outline = landmarks[0:33]
        hull = cv2.convexHull(face_outline)
        # Vectorized hull padding — expand each point outward from center
        center = np.mean(face_outline, axis=0, dtype=np.float32)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        directions = hull_pts - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)  # avoid division by zero
        directions /= norms
        hull_padded = (hull_pts + directions * padding).astype(np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges (GPU-accelerated when available)
        mask = gpu_gaussian_blur(mask, (5, 5), 3)

    return mask


def create_hairline_safe_mask(
    face: Face,
    frame: Frame,
    blur_sigma: float | None = None,
    erosion_px: int | None = None,
) -> np.ndarray:
    """Build a full-face hull with a small landmark-relative forehead extension."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        landmarks = getattr(face, "landmark_2d_68", None)
    if landmarks is None:
        return mask
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if (landmarks.ndim != 2 or landmarks.shape[0] < 3
            or landmarks.shape[1] != 2 or not np.all(np.isfinite(landmarks))):
        return mask

    points = landmarks.copy()
    point_count = points.shape[0]
    if point_count >= 106:
        jaw_indices = (0, 32)
        brow_points = np.vstack((points[43:52], points[97:106]))
    elif point_count >= 68:
        jaw_indices = (0, 16)
        brow_points = points[17:27]
    else:
        return mask

    # Eyebrow tails bound the upper face. They are used to pull only the two
    # upper jaw/temple anchors inward; the lower cheek and chin landmarks keep
    # their natural contour, avoiding a rectangular vertical cut through face.
    left_brow_x = float(np.min(brow_points[:, 0]))
    right_brow_x = float(np.max(brow_points[:, 0]))

    left_index, right_index = jaw_indices
    if points[left_index, 0] > points[right_index, 0]:
        left_index, right_index = right_index, left_index
    face_width = max(
        1.0, float(points[right_index, 0] - points[left_index, 0])
    )
    temple_inset = face_width * 0.09
    points[left_index, 0] += temple_inset
    points[right_index, 0] -= temple_inset

    # Neither dense layout has a true hairline. Generate five anchors directly
    # above the eyebrow curve. Their height comes only from the eye-to-chin
    # facial axis and is filtered over time; nose pitch no longer changes it.
    brow_order = np.argsort(brow_points[:, 0])
    ordered_brows = brow_points[brow_order]
    forehead_x = np.linspace(
        float(ordered_brows[0, 0]), float(ordered_brows[-1, 0]), 5,
        dtype=np.float32,
    )
    brow_y = np.interp(
        forehead_x, ordered_brows[:, 0], ordered_brows[:, 1]
    ).astype(np.float32)
    kps = getattr(face, "kps", None)
    eye_y = None
    if kps is not None:
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape == (5, 2) and np.all(np.isfinite(kps)):
            eye_y = float(np.mean(kps[:2, 1]))
    if eye_y is None:
        if point_count >= 106:
            eye_points = np.vstack((points[33:43], points[87:97]))
        else:
            eye_points = points[36:48]
        eye_y = float(np.mean(eye_points[:, 1]))
    chin_y = float(points[(jaw_indices[0] + jaw_indices[1]) // 2, 1])
    raw_forehead_height = abs(chin_y - eye_y) * _FOREHEAD_AXIS_RATIO
    forehead_height = _smooth_forehead_height(raw_forehead_height, face_width)
    forehead_y = brow_y - forehead_height
    forehead_points = np.column_stack((forehead_x, forehead_y))

    hull_points = np.vstack((points, forehead_points))

    # A single bad dense landmark must not stretch the hull to a frame edge.
    # This bbox envelope is generous enough to leave valid face geometry
    # unchanged, but clips impossible coordinates before rasterisation.
    bbox = getattr(face, "bbox", None)
    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            bx1, by1, bx2, by2 = (float(value) for value in bbox[:4])
            box_w = bx2 - bx1
            box_h = by2 - by1
            if box_w > 4.0 and box_h > 4.0:
                hull_points[:, 0] = np.clip(
                    hull_points[:, 0], bx1 - 0.12 * box_w, bx2 + 0.12 * box_w
                )
                hull_points[:, 1] = np.clip(
                    hull_points[:, 1], by1 - 0.28 * box_h, by2 + 0.12 * box_h
                )
    points = np.rint(hull_points).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, frame.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, frame.shape[0] - 1)
    hull = cv2.convexHull(points)
    if hull is None or len(hull) < 3:
        return mask
    cv2.fillConvexPoly(mask, hull, 255)

    erosion = max(0, int(
        modules.globals.mask_erosion if erosion_px is None else erosion_px
    ))
    if erosion:
        # One pass with a 3x3 or 5x5 ellipse contracts the whole hull evenly.
        kernel_size = 3 if erosion <= 2 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        mask = cv2.erode(mask, kernel)

    sigma = min(1.5, max(0.0, float(
        modules.globals.mask_blur if blur_sigma is None else blur_sigma
    )))
    if sigma > 0:
        # Sigma 1.5 with a 7x7 kernel gives a 2-3 px transition without the
        # broad semi-transparent band that destabilizes seamlessClone.
        kernel_size = max(3, int(np.ceil(sigma * 2)) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

    return mask


def _yaw_rear_side(face: Face) -> tuple[int, float]:
    """Return the rear image side (-1 left, +1 right) and strong-yaw amount."""
    kps = getattr(face, "kps", None)
    if kps is None:
        return 0, 0.0
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2) or not np.all(np.isfinite(kps)):
        return 0, 0.0
    eye_span = max(1.0, abs(float(kps[1, 0] - kps[0, 0])))
    eye_mid = float((kps[0, 0] + kps[1, 0]) * 0.5)
    yaw = float((kps[2, 0] - eye_mid) / eye_span)
    amount = float(np.clip((abs(yaw) - 0.10) / 0.42, 0.0, 1.0))
    if amount <= 0.0:
        return 0, 0.0
    # When the nose moves left, the image-right temple is the rear side.
    return (1 if yaw < 0.0 else -1), amount


def create_profile_nostril_restore_mask(
    face: Face, frame: Frame
) -> np.ndarray:
    """Return a tiny soft mask for retaining the real nostril at strong yaw."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    rear_side, yaw_amount = _yaw_rear_side(face)
    if rear_side == 0 or yaw_amount < 0.35:
        return result
    bbox = getattr(face, "bbox", None)
    kps = getattr(face, "kps", None)
    if bbox is None or kps is None:
        return result
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    kps = np.asarray(kps, dtype=np.float32)
    if (bbox.size < 4 or kps.shape != (5, 2)
            or not np.all(np.isfinite(bbox[:4]))
            or not np.all(np.isfinite(kps))):
        return result
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    if width <= 8.0 or height <= 8.0:
        return result
    nose_x, nose_y = (float(value) for value in kps[2])
    center = (
        int(round(nose_x + rear_side * width * 0.025)),
        int(round(nose_y + height * 0.030)),
    )
    axes = (
        max(2, int(round(width * 0.042))),
        max(2, int(round(height * 0.028))),
    )
    cv2.ellipse(result, center, axes, 0, 0, 360, 255, -1)
    sigma = max(1.0, min(3.0, min(width, height) * 0.010))
    result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
    strength = float(np.clip((yaw_amount - 0.35) / 0.35, 0.0, 1.0))
    return np.clip(
        result.astype(np.float32) * strength, 0, 255
    ).astype(np.uint8)


def create_profile_temple_exclusion_mask(
    face: Face, frame: Frame
) -> np.ndarray:
    """Return a soft camera-only patch over the hidden temple at strong yaw."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    rear_side, yaw_amount = _yaw_rear_side(face)
    if rear_side == 0 or yaw_amount < 0.35:
        return result
    bbox = getattr(face, "bbox", None)
    kps = getattr(face, "kps", None)
    if bbox is None or kps is None:
        return result
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    kps = np.asarray(kps, dtype=np.float32)
    if (bbox.size < 4 or kps.shape != (5, 2)
            or not np.all(np.isfinite(bbox[:4]))
            or not np.all(np.isfinite(kps))):
        return result
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    width, height = x2 - x1, y2 - y1
    if width <= 8.0 or height <= 8.0:
        return result
    rear_eye_index = (
        int(np.argmax(kps[:2, 0])) if rear_side > 0
        else int(np.argmin(kps[:2, 0]))
    )
    eye_x, eye_y = (float(value) for value in kps[rear_eye_index])
    center = (
        int(round(eye_x + rear_side * width * 0.20)),
        int(round(eye_y - height * 0.015)),
    )
    axes = (
        max(3, int(round(width * 0.115))),
        max(4, int(round(height * 0.24))),
    )
    hard = np.zeros_like(result)
    cv2.ellipse(hard, center, axes, 0, 0, 360, 255, -1)
    # Start outside the rear eyebrow tail and stop above the lower cheek. This
    # removes the generated sideburn while retaining swapped identity features.
    inner_x = eye_x + rear_side * width * 0.085
    if rear_side > 0:
        hard[:, :max(0, int(round(inner_x)))] = 0
    else:
        hard[:, min(hard.shape[1], int(round(inner_x)) + 1):] = 0
    top = max(0, int(round(y1 + height * 0.015)))
    bottom = min(hard.shape[0], int(round(eye_y + height * 0.30)))
    hard[:top] = 0
    hard[bottom:] = 0

    # Never punch through actual eyebrow hairs at the inner edge of the patch.
    brow_guard = create_eyebrow_protection_mask(face, frame)
    if brow_guard.shape == hard.shape:
        hard = cv2.multiply(hard, 255 - brow_guard, scale=1.0 / 255.0)
    sigma = max(1.5, min(5.0, min(width, height) * 0.018))
    soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
    strength = float(np.clip((yaw_amount - 0.35) / 0.35, 0.0, 1.0))
    return np.clip(soft.astype(np.float32) * strength, 0, 255).astype(np.uint8)


def create_bbox_safety_mask(face: Face, frame: Frame) -> np.ndarray:
    """Return a soft containment envelope derived only from detector bbox."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return result
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return result
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    width = x2 - x1
    height = y2 - y1
    if width <= 4.0 or height <= 4.0:
        return result
    # Keep the safety envelope inside the detected facial area. In particular,
    # its top is below bbox top and its sides stop before the ears; this makes
    # it impossible for a fail-open landmark mask to expose the aligned crop
    # as a square over real hair or ears.
    center = (
        int(round((x1 + x2) * 0.5)),
        int(round(y1 + 0.55 * height)),
    )
    axes = (
        # Keep both cheeks and temples farther inside the detector box so the
        # swap never reaches sideburns or hair beside the ear.
        max(2, int(round(width * 0.39))),
        max(2, int(round(height * 0.43))),
    )
    cv2.ellipse(result, center, axes, 0, 0, 360, 255, -1)
    # Add one bbox-locked upper envelope for both eyebrow tails. The former
    # per-brow landmark lobes changed shape every frame and made the outer brow
    # shimmer even when the user was still. This ellipse has identical geometry
    # for identical tracked boxes and remains below the detector's hair area.
    cv2.ellipse(
        result,
        (center[0], int(round(y1 + 0.30 * height))),
        (
            max(2, int(round(width * 0.37))),
            max(2, int(round(height * 0.18))),
        ),
        0, 0, 360, 255, -1,
    )
    # The narrower side envelope stays away from hair. Restore only the two
    # eyebrow areas with fixed-size ellipses anchored to the already-smoothed
    # five-point eyes, so yaw cannot cut an occluded brow in half and dense
    # landmark noise cannot change their shape.
    kps = getattr(face, "kps", None)
    if kps is not None:
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape == (5, 2) and np.all(np.isfinite(kps)):
            for eye in kps[:2]:
                cv2.ellipse(
                    result,
                    (
                        int(round(eye[0])),
                        int(round(eye[1] - height * 0.085)),
                    ),
                    (
                        # Keep the full eyebrow tail at frontal/oblique poses;
                        # the strong-yaw rear cutoff below removes only the
                        # hidden-side extension that can reach source hair.
                        max(3, int(round(width * 0.18))),
                        max(3, int(round(height * 0.075))),
                    ),
                    0, 0, 360, 255, -1,
                )

            rear_side, yaw_amount = _yaw_rear_side(face)
            if rear_side and yaw_amount >= 0.35:
                # At a profile, stop the rear side shortly after the hidden
                # eye. Lock the cutoff to the already-smoothed bbox rather
                # than a per-frame eye coordinate, which can jump on detector
                # refreshes and make the edge visibly twitch.
                limit_x = center[0] + rear_side * width * 0.31
                fade_width = max(2.0, width * 0.025)
                roi_x1 = max(0, int(np.floor(x1)))
                roi_x2 = min(result.shape[1], int(np.ceil(x2)) + 1)
                roi_y1 = max(0, int(np.floor(y1)))
                roi_y2 = min(result.shape[0], int(np.ceil(y2)) + 1)
                yy = np.arange(roi_y1, roi_y2, dtype=np.float32)[:, None]
                xx = np.arange(roi_x1, roi_x2, dtype=np.float32)[None, :]
                signed_distance = rear_side * (xx - limit_x)
                horizontal_keep = np.clip(
                    1.0 - signed_distance / fade_width, 0.0, 1.0
                )
                top_gate = np.clip(
                    (yy - (y1 + height * 0.08)) / (height * 0.08),
                    0.0, 1.0,
                )
                bottom_gate = np.clip(
                    ((y1 + height * 0.88) - yy) / (height * 0.10),
                    0.0, 1.0,
                )
                vertical_gate = top_gate * bottom_gate
                strength = float(np.clip(
                    (yaw_amount - 0.35) / 0.35, 0.0, 1.0
                ))
                keep = 1.0 - strength * vertical_gate * (
                    1.0 - horizontal_keep
                )
                roi = result[roi_y1:roi_y2, roi_x1:roi_x2]
                result[roi_y1:roi_y2, roi_x1:roi_x2] = np.clip(
                    roi.astype(np.float32) * keep, 0, 255
                ).astype(np.uint8)
    sigma = max(1.5, min(3.0, min(width, height) * 0.014))
    result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return result

def _feature_landmarks(face: Face):
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        landmarks = getattr(face, "landmark_2d_68", None)
    if landmarks is None:
        return None, None
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        return None, None
    if points.shape[0] >= 106:
        # InsightFace 106 convention. The 68-point fallback below uses the
        # conventional 48:68 mouth and 36:48 eye ranges.
        return points[52:64], (points[33:43], points[87:97])
    if points.shape[0] >= 68:
        return points[48:68], (points[36:42], points[42:48])
    return None, None


def _eyebrow_landmarks(face: Face):
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        landmarks = getattr(face, "landmark_2d_68", None)
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 2
            or not np.all(np.isfinite(points))):
        return None
    if points.shape[0] >= 106:
        return points[43:52], points[97:106]
    if points.shape[0] >= 68:
        return points[17:22], points[22:27]
    return None


def create_eyebrow_protection_mask(face: Face, frame: Frame) -> np.ndarray:
    """Return a compact frame-space guard for real eyebrow hairs."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    guard = np.zeros(frame.shape[:2], dtype=np.uint8)
    bbox = getattr(face, "bbox", None)
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1) if bbox is not None else None
    kps = getattr(face, "kps", None)
    kps = np.asarray(kps, dtype=np.float32) if kps is not None else None
    if (kps is not None and kps.shape == (5, 2)
            and np.all(np.isfinite(kps))):
        if bbox is not None and bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            face_width = max(8.0, float(bbox[2] - bbox[0]))
            face_height = max(8.0, float(bbox[3] - bbox[1]))
        else:
            face_width = max(8.0, float(np.linalg.norm(kps[1] - kps[0])) * 2.2)
            face_height = face_width * 1.2
        # Stable eye-anchored ellipses protect the complete brow including its
        # tail without inheriting dense-landmark jitter.
        for eye in kps[:2]:
            cv2.ellipse(
                guard,
                (int(round(eye[0])), int(round(eye[1] - face_height * 0.085))),
                (
                    # Keep the guard on eyebrow hairs. A wider guard also
                    # covered the fake dark temple stroke and prevented edge
                    # decontamination from removing it.
                    max(3, int(round(face_width * 0.135))),
                    max(3, int(round(face_height * 0.065))),
                ),
                0, 0, 360, 255, -1,
            )
    else:
        brows = _eyebrow_landmarks(face)
        if brows:
            for brow in brows:
                hull = cv2.convexHull(np.rint(brow).astype(np.int32))
                cv2.fillConvexPoly(guard, hull, 255)
    if np.any(guard):
        if bbox is not None and bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            radius = max(2, int(round((bbox[2] - bbox[0]) * 0.018)))
        else:
            radius = 3
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        guard = cv2.dilate(guard, kernel)
        guard = cv2.GaussianBlur(guard, (0, 0), 1.0)
    return guard


def aligned_brow_eye_guard(face: Face, affine: np.ndarray, size: int) -> np.ndarray:
    """Protect expressive eyebrows/eyes from source-hair cleanup."""
    guard = np.zeros((size, size), dtype=np.uint8)
    _, eyes = _feature_landmarks(face)
    brows = _eyebrow_landmarks(face)
    feature_sets = []
    if eyes:
        feature_sets.extend(eyes)
    if brows:
        feature_sets.extend(brows)
    for feature in feature_sets:
        aligned = cv2.transform(np.asarray(feature, dtype=np.float32)[None], affine)[0]
        hull = cv2.convexHull(np.rint(aligned).astype(np.int32))
        cv2.fillConvexPoly(guard, hull, 255)
    if not feature_sets:
        kps = getattr(face, "kps", None)
        if kps is not None and np.asarray(kps).shape == (5, 2):
            centers = cv2.transform(
                np.asarray(kps[:2], dtype=np.float32)[None], affine
            )[0]
            for x, y in centers:
                cv2.ellipse(
                    guard,
                    (int(round(x)), int(round(y - size * 0.045))),
                    (max(3, int(size * 0.13)), max(3, int(size * 0.12))),
                    0, 0, 360, 255, -1,
                )
    if np.any(guard):
        radius = max(1, int(round(size * 0.018)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        guard = cv2.dilate(guard, kernel)
    return guard


def aligned_feature_guard(face: Face, affine: np.ndarray, size: int) -> np.ndarray | None:
    """255 means preserve the pre-enhancer eye/mouth pixels at crop size."""
    mouth, eyes = _feature_landmarks(face)
    guard = np.zeros((size, size), dtype=np.uint8)
    if eyes:
        for points in eyes:
            aligned = cv2.transform(points[None], affine)[0]
            hull = cv2.convexHull(np.rint(aligned).astype(np.int32))
            cv2.fillConvexPoly(guard, hull, 150)
    else:
        kps = getattr(face, "kps", None)
        if kps is not None and np.asarray(kps).shape == (5, 2):
            eye_centers = cv2.transform(np.asarray(kps[:2], dtype=np.float32)[None], affine)[0]
            for center in eye_centers:
                cv2.ellipse(guard, tuple(np.rint(center).astype(int)),
                            (max(2, size // 35), max(2, size // 65)), 0, 0, 360, 90, -1)
    kps = getattr(face, "kps", None)
    if kps is not None and np.asarray(kps).shape == (5, 2):
        eye_centers = cv2.transform(
            np.asarray(kps[:2], dtype=np.float32)[None], affine
        )[0]
        brow_centers = []
        for center in eye_centers:
            brow_center = (
                int(round(center[0])), int(round(center[1] - size * 0.065))
            )
            brow_centers.append(brow_center)
            cv2.ellipse(
                guard,
                brow_center,
                (max(3, int(size * 0.14)), max(3, int(size * 0.105))),
                0, 0, 360, 180, -1,
            )
        # Join both stable brow guards. With yaw, the far eyebrow can otherwise
        # cross the narrow gap between two masks and appear split in half.
        cv2.line(
            guard, brow_centers[0], brow_centers[1], 180,
            max(3, int(size * 0.12)), lineType=cv2.LINE_AA,
        )
        # GPEN can turn a profile nostril shadow into a black triangular mark.
        # Preserve the stable face-swap nose at strong yaw instead of cutting a
        # moving camera patch into it, which produced a visible seam and jitter.
        _, yaw_amount = _yaw_rear_side(face)
        if yaw_amount >= 0.35:
            nose = cv2.transform(
                np.asarray(kps[2:3], dtype=np.float32)[None], affine
            )[0, 0]
            cv2.ellipse(
                guard,
                tuple(np.rint(nose).astype(int)),
                (max(3, int(size * 0.055)),
                 max(2, int(size * 0.045))),
                0, 0, 360, 220, -1,
            )
    else:
        brows = _eyebrow_landmarks(face)
        if brows:
            for points in brows:
                aligned = cv2.transform(points[None], affine)[0]
                hull = cv2.convexHull(np.rint(aligned).astype(np.int32))
                cv2.fillConvexPoly(guard, hull, 180)
    if mouth is not None and getattr(modules.globals, "mouth_mask", False):
        aligned = cv2.transform(mouth[None], affine)[0]
        hull = cv2.convexHull(np.rint(aligned).astype(np.int32))
        cv2.fillConvexPoly(guard, hull, 255)
    if not np.any(guard):
        return None
    return cv2.GaussianBlur(guard, (5, 5), 1.0)


def _mask_blur(mask: np.ndarray) -> np.ndarray:
    sigma = float(getattr(modules.globals, "mask_blur", 1.5))
    if sigma <= 0:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """Create a mouth-only mask whose slider cannot reach the eye region."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    mouth_polygon = None
    mouth_box = (0, 0, 0, 0)
    mouth_points, eye_sets = _feature_landmarks(face)
    if mouth_points is None:
        return mask, mouth_cutout, mouth_box, mouth_polygon

    slider = float(np.clip(getattr(modules.globals, "mouth_mask_size", 0.0) / 100.0, 0.0, 1.0))
    center = mouth_points.mean(axis=0)
    offsets = mouth_points - center
    x_scale = 1.0 + 0.65 * slider
    y_scale = np.where(offsets[:, 1] >= 0.0, 1.0 + 1.8 * slider, 1.0 + 0.2 * slider)
    expanded = center + offsets * np.column_stack((np.full(len(offsets), x_scale), y_scale))

    # Hard separation from both eyes, including at the maximum slider value.
    if eye_sets:
        eye_bottom = max(float(np.max(eye[:, 1])) for eye in eye_sets if len(eye))
        mouth_height = max(1.0, float(np.ptp(mouth_points[:, 1])))
        safe_top = eye_bottom + max(2.0, 0.12 * mouth_height)
        expanded[:, 1] = np.maximum(expanded[:, 1], safe_top)

    expanded[:, 0] = np.clip(expanded[:, 0], 0, frame.shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, frame.shape[0] - 1)
    expanded = np.rint(expanded).astype(np.int32)
    # The 68-point layout includes inner lip points; its index order is not a
    # single simple outline. A hull makes the paint area stable for both layouts.
    expanded = cv2.convexHull(expanded).reshape(-1, 2)
    span = np.ptp(expanded, axis=0)
    pad_x = max(1, int(round(span[0] * 0.10)))
    pad_y = max(1, int(round(span[1] * 0.10)))
    min_x = max(0, int(np.min(expanded[:, 0])) - pad_x)
    max_x = min(frame.shape[1], int(np.max(expanded[:, 0])) + pad_x + 1)
    min_y = max(0, int(np.min(expanded[:, 1])) - pad_y)
    if eye_sets:
        min_y = max(min_y, int(np.ceil(safe_top)))
    max_y = min(frame.shape[0], int(np.max(expanded[:, 1])) + pad_y + 1)
    if max_x <= min_x or max_y <= min_y:
        return mask, mouth_cutout, mouth_box, mouth_polygon

    roi_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [expanded - np.array([min_x, min_y])], 255)
    mask[min_y:max_y, min_x:max_x] = _mask_blur(roi_mask)
    mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
    mouth_polygon = expanded
    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), mouth_polygon


def create_eyes_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """Create independent eye ellipses; the mouth slider is never consulted."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    eye_polygon = np.empty((0, 2), dtype=np.int32)
    _, eye_sets = _feature_landmarks(face)
    if not eye_sets or any(len(eye) == 0 for eye in eye_sets):
        return mask, eyes_cutout, (0, 0, 0, 0), eye_polygon

    scale = 1.0 + float(getattr(modules.globals, "mask_down_size", 0.1)) * float(
        getattr(modules.globals, "eyes_mask_size", 0.0)
    )
    all_points = np.vstack(eye_sets)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    padding = max(1, int(round(max(maxs - mins) * 0.20 * scale)))
    min_x = max(0, int(np.floor(mins[0] - padding)))
    min_y = max(0, int(np.floor(mins[1] - padding)))
    max_x = min(frame.shape[1], int(np.ceil(maxs[0] + padding + 1)))
    max_y = min(frame.shape[0], int(np.ceil(maxs[1] + padding + 1)))
    if max_x <= min_x or max_y <= min_y:
        return mask, eyes_cutout, (0, 0, 0, 0), eye_polygon

    yy, xx = np.ogrid[min_y:max_y, min_x:max_x]
    roi_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    polygons = []
    for eye in eye_sets:
        center = np.mean(eye, axis=0)
        axes = np.maximum(1.0, (np.ptp(eye, axis=0) * 0.5) * scale)
        inside = (((xx - center[0]) / axes[0]) ** 2 + ((yy - center[1]) / axes[1]) ** 2) <= 1.0
        roi_mask[inside] = 255
        t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        polygons.append(np.column_stack((center[0] + axes[0] * np.cos(t), center[1] + axes[1] * np.sin(t))))
    mask[min_y:max_y, min_x:max_x] = _mask_blur(roi_mask)
    eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
    eye_polygon = np.vstack(polygons).astype(np.int32)
    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eye_polygon


def create_curved_eyebrow(points):
    if len(points) >= 5:
        # Sort points by x-coordinate
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # Calculate dimensions
        x_min, y_min = np.min(sorted_points, axis=0)
        x_max, y_max = np.max(sorted_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        # Create more points for smoother curve
        num_points = 50
        x = np.linspace(x_min, x_max, num_points)
        
        # Fit quadratic curve through points for more natural arch
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
        y = np.polyval(coeffs, x)
        
        # Increased offsets to create more separation
        top_offset = height * 0.5  # Increased from 0.3 to shift up more
        bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
        
        # Create smooth curves
        top_curve = y - top_offset
        bottom_curve = y + bottom_offset
        
        # Create curved endpoints with more pronounced taper
        end_points = 5
        start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
        end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
        
        # Create tapered ends
        start_curve = np.column_stack((
            start_x,
            np.linspace(bottom_curve[0], top_curve[0], end_points)
        ))
        end_curve = np.column_stack((
            end_x,
            np.linspace(bottom_curve[-1], top_curve[-1], end_points)
        ))
        
        # Combine all points to form a smooth contour
        contour_points = np.vstack([
            start_curve,
            np.column_stack((x, top_curve)),
            end_curve,
            np.column_stack((x[::-1], bottom_curve[::-1]))
        ])
        
        # Add slight padding for better coverage
        center = np.mean(contour_points, axis=0)
        vectors = contour_points - center
        padded_points = center + vectors * 1.2  # Increased padding slightly
        
        return padded_points
    return points

def create_eyebrows_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Left eyebrow landmarks (97-105) and right eyebrow landmarks (43-51)
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)
        
        # Calculate centers and dimensions for each eyebrow
        left_center = np.mean(left_eyebrow, axis=0)
        right_center = np.mean(right_eyebrow, axis=0)
        
        # Calculate bounding box with padding adjusted by size
        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = modules.globals.eyebrows_mask_size
        min_x = np.min(all_points[:, 0]) - 25 * padding_factor
        max_x = np.max(all_points[:, 0]) + 25 * padding_factor
        min_y = np.min(all_points[:, 1]) - 20 * padding_factor
        max_y = np.max(all_points[:, 1]) + 15 * padding_factor
        
        # Ensure coordinates are within frame bounds
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame.shape[1], int(max_x))
        max_y = min(frame.shape[0], int(max_y))
        
        # Create mask for the eyebrows region
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        try:
            # Convert points to local coordinates
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            
            def create_curved_eyebrow(points):
                if len(points) >= 5:
                    # Sort points by x-coordinate
                    sorted_idx = np.argsort(points[:, 0])
                    sorted_points = points[sorted_idx]
                    
                    # Calculate dimensions
                    x_min, y_min = np.min(sorted_points, axis=0)
                    x_max, y_max = np.max(sorted_points, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Create more points for smoother curve
                    num_points = 50
                    x = np.linspace(x_min, x_max, num_points)
                    
                    # Fit quadratic curve through points for more natural arch
                    coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
                    y = np.polyval(coeffs, x)
                    
                    # Increased offsets to create more separation
                    top_offset = height * 0.5  # Increased from 0.3 to shift up more
                    bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
                    
                    # Create smooth curves
                    top_curve = y - top_offset
                    bottom_curve = y + bottom_offset
                    
                    # Create curved endpoints with more pronounced taper
                    end_points = 5
                    start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
                    end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
                    
                    # Create tapered ends
                    start_curve = np.column_stack((
                        start_x,
                        np.linspace(bottom_curve[0], top_curve[0], end_points)
                    ))
                    end_curve = np.column_stack((
                        end_x,
                        np.linspace(bottom_curve[-1], top_curve[-1], end_points)
                    ))
                    
                    # Combine all points to form a smooth contour
                    contour_points = np.vstack([
                        start_curve,
                        np.column_stack((x, top_curve)),
                        end_curve,
                        np.column_stack((x[::-1], bottom_curve[::-1]))
                    ])
                    
                    # Add slight padding for better coverage
                    center = np.mean(contour_points, axis=0)
                    vectors = contour_points - center
                    padded_points = center + vectors * 1.2  # Increased padding slightly
                    
                    return padded_points
                return points
            
            # Generate and draw eyebrow shapes
            left_shape = create_curved_eyebrow(left_local)
            right_shape = create_curved_eyebrow(right_local)
            
            # Apply multi-stage blurring for natural feathering (GPU-accelerated when available)
            # First, strong Gaussian blur for initial softening
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            
            # Second, medium blur for transition areas
            mask_roi = gpu_gaussian_blur(mask_roi, (11, 11), 3)
            
            # Finally, light blur for fine details
            mask_roi = gpu_gaussian_blur(mask_roi, (5, 5), 1)
            
            # Normalize mask values
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            # Place the mask ROI in the full-sized mask
            mask[min_y:max_y, min_x:max_x] = mask_roi
            
            # Extract the masked area from the frame
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            
            # Combine points for visualization
            eyebrows_polygon = np.vstack([
                left_shape + [min_x, min_y],
                right_shape + [min_x, min_y]
            ]).astype(np.int32)
            
        except Exception as e:
            # Fallback to simple polygons if curve fitting fails
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)
        
    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Create mask for the area
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        
        # Split points for left and right parts if needed
        if len(polygon) > 50:  # Arbitrary threshold to detect if we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply strong initial feathering (GPU-accelerated when available)
        polygon_mask = gpu_gaussian_blur(polygon_mask, (21, 21), 7)

        # Apply additional feathering
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (0, 0), feather_amount
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        # Apply additional smoothing to the mask edges
        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0))

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape)
        final_blend = blended * face_mask_3channel + roi * (np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame

def draw_mask_visualization(
    frame: Frame,
    mask_data: tuple,
    label: str,
    draw_method: str = "polygon"
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data

    vis_frame = frame.copy()

    # Ensure coordinates are within frame bounds
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if draw_method == "ellipse" and len(polygon) > 50:  # For eyes
        # Split points for left and right parts
        mid_point = len(polygon) // 2
        left_points = polygon[:mid_point]
        right_points = polygon[mid_point:]
        
        try:
            # Fit ellipses to points - need at least 5 points
            if len(left_points) >= 5 and len(right_points) >= 5:
                # Convert points to the correct format for ellipse fitting
                left_points = left_points.astype(np.float32)
                right_points = right_points.astype(np.float32)
                
                # Fit ellipses
                left_ellipse = cv2.fitEllipse(left_points)
                right_ellipse = cv2.fitEllipse(right_points)
                
                # Draw the ellipses
                cv2.ellipse(vis_frame, left_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vis_frame, right_ellipse, (0, 255, 0), 2)
        except Exception as e:
            # If ellipse fitting fails, draw simple rectangles as fallback
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
            cv2.rectangle(vis_frame, 
                        (left_rect[0], left_rect[1]), 
                        (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                        (0, 255, 0), 2)
            cv2.rectangle(vis_frame,
                        (right_rect[0], right_rect[1]),
                        (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]),
                        (0, 255, 0), 2)
    else:  # For mouth and eyebrows
        # Draw the polygon
        if len(polygon) > 50:  # If we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point]
            right_points = polygon[mid_point:]
            cv2.polylines(vis_frame, [left_points], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis_frame, [right_points], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Add label
    cv2.putText(
        vis_frame,
        label,
        (min_x, min_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_frame
