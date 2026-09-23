"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.platform_info import OPENVINO_PROVIDER_CONFIG
from modules.processors.frame.face_masking import (
    match_skin_tone_lab, aligned_feature_guard,
    create_bbox_safety_mask, harmonize_mask_boundary,
)
from modules.processors.frame.face_parser import parse_face_skin, skin_guard_for_crop

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))
_LIVE_ENHANCER_CACHE: dict[int, dict] = {}
_MASK_CACHE: dict[tuple, np.ndarray] = {}


def tensorrt_provider_config():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), ".cache", "tensorrt")
    os.makedirs(cache_dir, exist_ok=True)
    return ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
    })


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Use bare provider — ONNX Runtime's defaults are fastest on
            # modern GPUs (Blackwell/sm_120).  Custom options like
            # EXHAUSTIVE cudnn_conv_algo_search hurt performance on these
            # architectures.
            config.append(p)
        elif p == "TensorrtExecutionProvider":
            config.append(tensorrt_provider_config())
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        elif p == "OpenVINOExecutionProvider":
            # AUTO lets OpenVINO select the best device
            config.append(OPENVINO_PROVIDER_CONFIG)
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    providers = build_provider_config()
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _restoration_is_plausible(
    camera_crop: np.ndarray,
    restored_crop: np.ndarray,
    previous_restored: np.ndarray | None = None,
) -> bool:
    """Reject a live restoration that hallucinates, smears, or clips side hair."""
    if (not isinstance(camera_crop, np.ndarray)
            or not isinstance(restored_crop, np.ndarray)
            or camera_crop.size == 0 or restored_crop.size == 0
            or camera_crop.shape != restored_crop.shape
            or camera_crop.ndim != 3 or restored_crop.ndim != 3):
        return False
    camera_gray = cv2.cvtColor(camera_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    restored_gray = cv2.cvtColor(restored_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = camera_gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx + 0.5) / max(1.0, float(width))
    ny = (yy + 0.5) / max(1.0, float(height))
    face_region = (
        np.square((nx - 0.5) / 0.40) + np.square((ny - 0.52) / 0.43)
        <= 1.0
    )
    if int(np.count_nonzero(face_region)) < 64:
        return False

    # White or black side patches are a characteristic failed GPEN frame.
    side_region = face_region & (ny >= 0.12) & (ny <= 0.82) & (
        (nx <= 0.23) | (nx >= 0.77)
    )
    side_count = max(1, int(np.count_nonzero(side_region)))
    new_white = side_region & (restored_gray >= 246.0) & (camera_gray <= 215.0)
    new_black = side_region & (restored_gray <= 9.0) & (camera_gray >= 40.0)
    if ((np.count_nonzero(new_white) + np.count_nonzero(new_black))
            / side_count > 0.025):
        return False

    # Restoration may soften sensor noise, but it must not collapse all real
    # facial structure into a smooth patch.
    camera_sharpness = float(cv2.Laplacian(camera_gray, cv2.CV_32F).var())
    restored_sharpness = float(cv2.Laplacian(restored_gray, cv2.CV_32F).var())
    if (camera_sharpness >= 18.0
            and restored_sharpness < max(1.5, camera_sharpness * 0.055)):
        return False

    # Aligned crops from adjacent frames may change expression, but an abrupt
    # near-total structural replacement indicates a hallucinated identity.
    if (previous_restored is not None
            and isinstance(previous_restored, np.ndarray)
            and previous_restored.shape == restored_crop.shape):
        previous_gray = cv2.cvtColor(
            previous_restored, cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        sigma = max(2.0, min(height, width) / 48.0)
        current_low = cv2.GaussianBlur(restored_gray, (0, 0), sigma)
        previous_low = cv2.GaussianBlur(previous_gray, (0, 0), sigma)
        current_values = current_low[face_region]
        previous_values = previous_low[face_region]
        current_values -= np.median(current_values)
        previous_values -= np.median(previous_values)
        temporal_error = float(np.percentile(
            np.abs(current_values - previous_values), 90
        ))
        denom = float(np.linalg.norm(current_values) * np.linalg.norm(previous_values))
        correlation = (
            float(np.dot(current_values, previous_values) / denom)
            if denom > 1e-6 else 1.0
        )
        if temporal_error > 82.0 and correlation < 0.12:
            return False
    return True


def _closeup_detail_strength(
    base_strength: float,
    inverse_affine: np.ndarray,
    live: bool,
) -> float:
    """Increase real high-frequency detail when a 256px crop is enlarged."""
    base = float(np.clip(base_strength, 0.0, 1.0))
    if (not live or inverse_affine is None
            or np.asarray(inverse_affine).shape != (2, 3)):
        return base
    singular = np.linalg.svd(
        np.asarray(inverse_affine, dtype=np.float32)[:, :2],
        compute_uv=False,
    )
    output_scale = float(np.max(singular))
    closeup = float(np.clip((output_scale - 1.05) / 0.85, 0.0, 1.0))
    # Restore only camera high frequencies; blending the complete pre-enhancer
    # crop reintroduced its 128px low-frequency softness across the whole face.
    return min(0.92, base + (0.92 - base) * closeup)


def blend_high_frequency(
    camera_crop: np.ndarray,
    restored_crop: np.ndarray,
    strength: float | None = None,
) -> np.ndarray:
    """Put a controlled amount of camera texture back over a restoration.

    GPEN/GFPGAN deliberately removes sensor noise and can also smooth small
    features such as eyebrow hairs and expression lines.  We extract only the
    high-frequency residual (camera minus a 2 px Gaussian low-pass image), so
    colour and broad facial shape still come from the restoration model.  The
    residual is clipped before conversion back to uint8 to avoid haloing or
    wrap-around at bright/dark edges.
    """
    if strength is None:
        strength = modules.globals.detail_strength
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or camera_crop is None or restored_crop is None:
        return restored_crop
    if camera_crop.shape != restored_crop.shape:
        camera_crop = cv2.resize(
            camera_crop,
            (restored_crop.shape[1], restored_crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    camera = camera_crop.astype(np.float32)
    restored = restored_crop.astype(np.float32)
    low_frequency = cv2.GaussianBlur(camera, (0, 0), sigmaX=2.0, sigmaY=2.0)
    high_frequency = camera - low_frequency
    blended = np.clip(restored + high_frequency * strength, 0.0, 255.0)
    return blended.astype(np.uint8)


def add_adaptive_film_grain(
    camera_crop: np.ndarray,
    restored_crop: np.ndarray,
    strength: float | None = None,
) -> np.ndarray:
    """Add camera-matched, very fine grain to an enhanced face crop.

    Noise is estimated with a robust MAD statistic on a one-pixel high-pass
    residual so strong eyebrows or edges do not dominate the measurement.  A
    small ``cv2.randn`` field is then added to the restored crop.  The output
    is clipped to avoid coloured wrap-around at 0/255.
    """
    if strength is None:
        strength = modules.globals.film_grain_strength
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or camera_crop is None or restored_crop is None:
        return restored_crop
    if camera_crop.shape[:2] != restored_crop.shape[:2]:
        camera_crop = cv2.resize(
            camera_crop,
            (restored_crop.shape[1], restored_crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    gray = cv2.cvtColor(camera_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    measured_std = 1.4826 * mad
    # Typical webcam ISO grain is only a few 8-bit levels.  Keep a small floor
    # so an unusually clean synthetic crop still receives natural microtexture.
    noise_std = float(np.clip(measured_std, 2.0, 4.0)) * strength
    if noise_std <= 0.0:
        return restored_crop
    noise = np.zeros(restored_crop.shape[:2], dtype=np.float32)
    cv2.randn(noise, 0.0, noise_std)
    # Luminance grain is shared by the three channels to avoid coloured
    # speckles that would look like compression artefacts.
    noise = noise[:, :, None]
    return np.clip(restored_crop.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def apply_hairline_guard(mask: np.ndarray) -> np.ndarray:
    """Clip the upper part of an aligned enhancer mask above the hairline."""
    guard = float(getattr(modules.globals, "hairline_guard", 0.16))
    if guard <= 0.0 or mask is None or mask.ndim != 2:
        return mask
    size = mask.shape[0]
    boundary = max(0.0, min(0.35, guard)) * size
    yy = np.arange(size, dtype=np.float32)[:, None]
    xx = np.arange(mask.shape[1], dtype=np.float32)[None, :]
    half = max(1.0, size * 0.44)
    curve = boundary + 0.035 * size * np.square((xx - size * 0.5) / half)
    feather = max(1.0, float(getattr(modules.globals, "mask_blur", 1.5)) * 1.5)
    gate = np.clip((yy - curve + feather) / (2.0 * feather), 0.0, 1.0)
    return np.rint(mask.astype(np.float32) * gate).astype(np.uint8)


def get_enhancer_crop_mask(input_size: int) -> np.ndarray:
    """Return a scale-relative oval mask, never a rectangular GPEN crop."""
    key = ("anatomical-oval", input_size)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = np.mgrid[0:input_size, 0:input_size].astype(np.float32)
    x = (xx + 0.5) / float(input_size)
    y = (yy + 0.5) / float(input_size)
    radius = np.sqrt(
        np.square((x - 0.5) / 0.495)
        + np.square((y - 0.5) / 0.495)
    )
    # About 3% of crop width is feathered. Since the mask is inverse-warped,
    # the transition scales with the tracked face at every camera distance.
    alpha = np.clip((1.0 - radius) / 0.025, 0.0, 1.0)
    mask = np.rint(alpha * 255.0).astype(np.uint8)
    _MASK_CACHE[key] = mask
    return mask


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
    live: bool = False,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        print("[GPEN] Empty input frame; skipping enhancer", flush=True)
        return frame
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    # GPEN-256 must not alternate between freshly restored and cached looks
    # in the webcam stream. Keep its output synchronized to every frame even
    # if a legacy environment still requests an interval > 1.
    interval = 1 if input_size == 256 else (modules.globals.enhancer_interval if live else 1)
    cache = _LIVE_ENHANCER_CACHE.setdefault(
        id(session), {"count": 0, "enhanced": None, "reject_count": 0}
    )
    if live:
        cache["count"] += 1
    run_model = not live or interval == 1 or cache["enhanced"] is None or (cache["count"] - 1) % interval == 0
    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    if face_crop is None or face_crop.size == 0:
        print("[GPEN] Empty aligned crop; returning camera frame", flush=True)
        return frame
    if run_model:
        blob = preprocess_face(face_crop, input_size)
        try:
            with THREAD_SEMAPHORE:
                input_name = session.get_inputs()[0].name
                output = run_inference(session, input_name, blob)
            restored = postprocess_face(output)
        except Exception:
            if live:
                # A cached crop belongs to the previous pose. Re-warping it
                # with the current affine creates a different-looking face or
                # white side hair during fast motion, so retain the current
                # sharp face swap instead.
                print("[GPEN] Inference failed; keeping current swapped frame", flush=True)
                return frame
            raise
        if live and not _restoration_is_plausible(
            face_crop, restored, cache.get("enhanced")
        ):
            cache["reject_count"] = int(cache.get("reject_count", 0)) + 1
            # A source-image change can legitimately replace the aligned
            # identity. After three guarded frames, forget the old temporal
            # reference so the new stable identity can establish itself.
            if cache["reject_count"] >= 3:
                cache["enhanced"] = None
                cache["reject_count"] = 0
            print("[GPEN] Rejected unstable restoration; keeping current swapped frame", flush=True)
            return frame
        if live:
            cache["enhanced"] = restored
            cache["reject_count"] = 0
    else:
        restored = cache["enhanced"]

    detail_strength = _closeup_detail_strength(
        modules.globals.detail_strength, inv_M, live
    )
    enhanced = blend_high_frequency(face_crop, restored, detail_strength)
    enhanced = add_adaptive_film_grain(face_crop, enhanced, modules.globals.film_grain_strength)
    if not isinstance(enhanced, np.ndarray) or enhanced.size == 0:
        print("[GPEN] Empty enhanced crop; returning camera frame", flush=True)
        return frame

    # Restore the complete aligned GPEN crop. The proportional border follows
    # the affine scale instead of shrinking distant faces with a fixed ellipse.
    # Always retain a very wide oval as a safety envelope. The anatomical mask
    # remains the tighter normal boundary, but a bad landmark frame can no
    # longer reveal the square GPEN crop across the background.
    mask = get_enhancer_crop_mask(input_size)
    feature_guard = aligned_feature_guard(face, M, input_size)
    if feature_guard is not None:
        mask = cv2.multiply(mask, 255 - feature_guard, scale=1.0 / 255.0)
    if mask is None or mask.size == 0 or not np.any(mask):
        print("[GPEN] Alpha mask is empty; returning camera frame", flush=True)
        return frame

    # Match illumination within the small aligned crop. The mask keeps hair
    # and background out of the statistics and avoids a full-frame LAB pass.
    if getattr(modules.globals, "color_match", True):
        # The swap stage already matched the live skin tone. Keep this second
        # pass light so pitch/foreshortening cannot amplify cheek samples into
        # the red-grey cast seen while looking down.
        enhanced = match_skin_tone_lab(
            enhanced, face_crop, mask, strength=0.42
        )

    # Warp and blend only the face ROI; full-frame float conversion is costly
    # at 720p and is unnecessary for a 256px restoration crop.
    corners = np.array([[0, 0], [input_size, 0], [input_size, input_size], [0, input_size]], dtype=np.float32)
    transformed = corners @ inv_M[:, :2].T + inv_M[:, 2]
    bbox = getattr(face, "bbox", None)
    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            box_w = float(bbox[2] - bbox[0])
            box_h = float(bbox[3] - bbox[1])
            quad_min = transformed.min(axis=0)
            quad_max = transformed.max(axis=0)
            quad_w, quad_h = (quad_max - quad_min).astype(float)
            box_center = np.array(
                [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
            )
            quad_center = (quad_min + quad_max) * 0.5
            if (box_w <= 4.0 or box_h <= 4.0
                    or quad_w < 0.50 * box_w or quad_w > 2.60 * box_w
                    or quad_h < 0.50 * box_h or quad_h > 2.60 * box_h
                    or np.linalg.norm(quad_center - box_center)
                    > 0.70 * max(box_w, box_h)):
                return frame
    h, w = frame.shape[:2]
    x1, y1 = np.maximum(0, np.floor(transformed.min(axis=0) - 2).astype(int))
    x2, y2 = np.minimum([w, h], np.ceil(transformed.max(axis=0) + 2).astype(int))
    if x2 <= x1 or y2 <= y1:
        return frame
    roi_M = inv_M.copy()
    roi_M[:, 2] -= [x1, y1]
    size = (int(x2 - x1), int(y2 - y1))
    warped_face = cv2.warpAffine(enhanced, roi_M, size, flags=cv2.INTER_CUBIC)
    warped_mask = cv2.warpAffine(mask, roi_M, size, flags=cv2.INTER_LINEAR)
    parsed_skin = parse_face_skin(frame, face)
    bbox_guard = create_bbox_safety_mask(face, frame)
    if bbox_guard.shape[:2] == frame.shape[:2] and np.any(bbox_guard):
        candidate = np.minimum(
            warped_mask, bbox_guard[y1:y2, x1:x2]
        )
        retained = float(np.sum(candidate, dtype=np.float64)) / max(
            1.0, float(np.sum(warped_mask, dtype=np.float64))
        )
        if retained < 0.08:
            return frame
        warped_mask = candidate
    if parsed_skin is not None:
        parser_guard = skin_guard_for_crop(parsed_skin, x1, y1, x2, y2)
        candidate = np.minimum(warped_mask, parser_guard).astype(np.uint8)
        retained = float(np.sum(candidate, dtype=np.float64)) / max(
            1.0, float(np.sum(warped_mask, dtype=np.float64))
        )
        if retained >= 0.35:
            warped_mask = candidate
    if warped_mask is None or warped_mask.size == 0 or not np.any(warped_mask):
        print("[GPEN] Warped alpha mask is empty; returning camera frame", flush=True)
        return frame
    alpha = cv2.merge([warped_mask] * 3)
    roi = frame[y1:y2, x1:x2]
    warped_face = harmonize_mask_boundary(warped_face, roi, warped_mask)
    blended = cv2.add(
        cv2.multiply(warped_face, alpha, scale=1.0 / 255.0),
        cv2.multiply(roi, 255 - alpha, scale=1.0 / 255.0),
    )
    if blended is None or blended.size == 0 or blended.shape != roi.shape:
        print("[GPEN] Invalid blend result; returning camera frame", flush=True)
        return frame
    frame[y1:y2, x1:x2] = blended
    return frame
