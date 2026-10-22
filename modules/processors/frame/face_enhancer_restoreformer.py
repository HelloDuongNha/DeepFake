"""GPEN-BFR-256 face enhancer — ONNX-based face restoration at 256x256."""

from typing import Any, List
import os
import hashlib
import threading

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

NAME = "DLC.FACE-ENHANCER-RESTOREFORMER"
INPUT_SIZE = 512
MODEL_MIRROR_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/restoreformer_plus_plus.onnx"
MODEL_FILE = "restoreformer_plus_plus.onnx"

ENHANCER = None
THREAD_LOCK = threading.Lock()

from modules.paths import MODELS_DIR

models_dir = MODELS_DIR


def _obtain_model():
    from modules.model_downloader import ensure_model

    model_path = ensure_model(MODEL_FILE)
    if model_path is not None:
        return model_path

    update_status(f"Retrying {MODEL_FILE} from the mirror...", NAME)
    from modules.utilities import conditional_download

    try:
        conditional_download(models_dir, [MODEL_MIRROR_URL])
    except Exception as error:
        update_status(f"Mirror download failed: {error}", NAME)
        return None
    fallback = os.path.join(models_dir, MODEL_FILE)
    return fallback if os.path.exists(fallback) else None


def pre_check() -> bool:
    if _obtain_model() is None:
        update_status(
            f"Could not obtain {MODEL_FILE}. Place it in the models folder "
            "manually or check your internet connection.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = _obtain_model()
            if model_path is None:
                raise FileNotFoundError(
                    f"Model file not found: {os.path.join(models_dir, MODEL_FILE)}"
                )
            print(f"{NAME}: Loading ONNX model from {model_path}")
            with open(model_path, "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != "164818b70d1745da4108a45c9263b05452236d14ac0ffab2202af0ac4a438195":
                raise ValueError("RestoreFormer++ model checksum mismatch")
            ENHANCER = create_onnx_session(model_path)
            warmup_session(ENHANCER)
            print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face, live: bool = False) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        from modules.processors.frame.face_swapper import FACE_SWAPPER, get_last_swap_status
        native_swapped = (
            getattr(FACE_SWAPPER, "native_resolution", 128) >= 256
            and get_last_swap_status() == "applied"
        )
        strength = .40 if native_swapped else 1.
        return enhance_face_onnx(
            temp_frame, face, session, INPUT_SIZE, live=live,
            restoration_strength=strength,
        )
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(
        temp_frame, target_face,
        live=detected_faces is not None and not modules.globals.many_faces,
    )


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
