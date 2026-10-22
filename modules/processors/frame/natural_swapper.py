"""Native 256px face swapping with the existing alignment/paste-back pipeline.

HyperSwap-1a weights: FaceFusion assets, models-3.3.0, ResearchRAIL license.
Input/output preprocessing follows the model's published FaceFusion contract.
"""

import hashlib

import cv2
import numpy as np
import onnxruntime
from insightface.utils import face_align

MODEL_FILE = "hyperswap_1a_256.onnx"
MODEL_SHA256 = "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"


def compact_skin_marks(crop: np.ndarray) -> np.ndarray:
    """Locate small, dark cheek marks while protecting facial feature shadows."""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    diameter = max(5, round(min(h, w) * .045) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter,) * 2)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    xx /= w
    yy /= h
    cheek = (yy > .48) & (yy < .78) & (xx > .12) & (xx < .88)
    nose = ((xx - .5) / .15) ** 2 + ((yy - .57) / .12) ** 2 < 1
    mouth = ((xx - .5) / .24) ** 2 + ((yy - .74) / .10) ** 2 < 1
    candidates = ((dark >= 12) & cheek & ~nose & ~mouth).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    mask = np.zeros((h, w), np.uint8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        aspect = max(width, height) / max(1, min(width, height))
        # A small real mole can cover just two horizontally adjacent pixels
        # after alignment; treating that as an elongated wrinkle misses it.
        max_aspect = 2.5 if area <= 4 else 1.9
        if (2 <= area <= max(6, h * w * .0012)
                and max(width, height) <= max(5, min(h, w) * .055)
                and aspect <= max_aspect):
            roi = mask[y:y + height, x:x + width]
            roi[labels[y:y + height, x:x + width] == label] = 255
    return mask


def remove_target_only_marks(generated: np.ndarray, camera: np.ndarray,
                             reference: np.ndarray | None) -> np.ndarray:
    """Remove a target cheek dot only where the source portrait has no mark."""
    if reference is None or reference.shape != camera.shape:
        return generated
    target_marks = compact_skin_marks(camera)
    if not np.any(target_marks):
        return generated
    source_marks = compact_skin_marks(reference)
    source_guard = cv2.dilate(source_marks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    removable = cv2.bitwise_and(target_marks, 255 - source_guard)
    if not np.any(removable):
        return generated
    removable = cv2.dilate(removable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.inpaint(generated, removable, 2., cv2.INPAINT_TELEA)


class Natural256Swapper:
    input_size = (256, 256)
    native_resolution = 256

    def __init__(self, model_path: str, providers, model_name='hyperswap_256'):
        from modules.swap_models import SWAP_MODELS
        self.model_name = model_name
        expected_hash = SWAP_MODELS[model_name][3]
        with open(model_path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError("HyperSwap model checksum mismatch; download the original model again")
        from modules.onnx_optimize import optimize_for_coreml

        self.model_file = optimize_for_coreml(model_path)
        self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def get(self, frame, target_face, source_face, paste_back=False):
        if paste_back:
            raise ValueError("Use the application's anatomical paste-back for Natural256Swapper")
        affine = face_align.estimate_norm(np.asarray(target_face.kps, np.float32), 128) * 2.
        camera = cv2.warpAffine(frame, affine, self.input_size,
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        target = camera[:, :, ::-1].astype(np.float32) / 127.5 - 1.
        target = np.ascontiguousarray(target.transpose(2, 0, 1)[None])
        identity = np.asarray(source_face.normed_embedding,
                              np.float32).reshape(1, 512)
        output = self.session.run(self.output_names, {"source": identity, "target": target})[0]
        if output.shape != (1, 3, 256, 256) or not np.isfinite(output).all():
            raise ValueError("Invalid HyperSwap face output")
        image = output[0].transpose(1, 2, 0)[:, :, ::-1]
        generated = np.rint(np.clip(
            (image + 1.) * 127.5, 0, 255
        )).astype(np.uint8)
        import modules.globals as settings
        if settings.nose_edge_repair and not settings.natural_nose:
            landmarks=getattr(target_face,"landmark_2d_106",None)
            kps=np.asarray(target_face.kps,np.float32)
            span=float(np.linalg.norm(kps[1]-kps[0]))
            yaw=abs(float(kps[2,0]-np.mean(kps[:2,0])))/max(1.,span)
            valid=(np.asarray(landmarks,np.float32) if landmarks is not None else None)
            if valid is not None and valid.shape==(106,2) and np.isfinite(valid).all() and yaw<.24:
                from modules.processors.frame.face_detail import suppress_model_nose_ridges
                points=cv2.transform(valid[None],affine)[0]
                generated=suppress_model_nose_ridges(generated,points)
        generated = remove_target_only_marks(
            generated, camera, getattr(source_face, "skin_reference", None)
        )
        return generated, affine
