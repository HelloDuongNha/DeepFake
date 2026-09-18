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

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))
_LIVE_ENHANCER_CACHE: dict[int, dict] = {}


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
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    interval = modules.globals.enhancer_interval if live else 1
    cache = _LIVE_ENHANCER_CACHE.setdefault(id(session), {"count": 0, "enhanced": None})
    if live:
        cache["count"] += 1
    run_model = not live or interval == 1 or cache["enhanced"] is None or (cache["count"] - 1) % interval == 0
    if run_model:
        face_crop = cv2.warpAffine(
            frame, M, (input_size, input_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        blob = preprocess_face(face_crop, input_size)
        with THREAD_SEMAPHORE:
            input_name = session.get_inputs()[0].name
            output = run_inference(session, input_name, blob)
        enhanced = postprocess_face(output)
        # Restore a small amount of the real camera's texture.  This is done
        # before caching so skipped live frames retain the same detail profile
        # without another CPU pass.
        enhanced = blend_high_frequency(
            face_crop, enhanced, modules.globals.detail_strength
        )
        enhanced = add_adaptive_film_grain(
            face_crop, enhanced, modules.globals.film_grain_strength
        )
        if live:
            cache["enhanced"] = enhanced
    else:
        enhanced = cache["enhanced"]
        # The expensive restoration stays cached, but the inexpensive
        # high-frequency residual follows the current frame so eyebrows and
        # expression lines do not look frozen during skipped frames.
        if live and modules.globals.detail_strength > 0.0:
            current_crop = cv2.warpAffine(
                frame, M, (input_size, input_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            enhanced = blend_high_frequency(
                current_crop, enhanced, modules.globals.detail_strength
            )
            enhanced = add_adaptive_film_grain(
                current_crop, enhanced, modules.globals.film_grain_strength
            )

    # Create mask for blending (feathered edges)
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])
    mask = apply_hairline_guard(np.rint(mask * 255.0).astype(np.uint8)).astype(np.float32) / 255.0

    h, w = frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)
