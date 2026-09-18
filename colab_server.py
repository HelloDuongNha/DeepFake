"""Headless Gradio server for Deep-Live-Cam on Google Colab.

The notebook starts this file inside a Colab VM.  A browser on the Mac sends
webcam frames to Gradio, this process performs face analysis, INSwapper and an
optional ONNX face enhancer on the T4, then returns the processed frame.
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any

# Colab has no desktop session.  This must be set before importing modules that
# import Qt through the upstream UI module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DLC_PROFILE", "colab-cuda")

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

import modules.globals

STREAM_MAX_WIDTH = int(os.environ.get("COLAB_MAX_WIDTH", "640"))


def configure_cuda() -> list[str]:
    """Require CUDA and configure every ONNX session to prefer it."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu "
            "and restart the Colab runtime. Available providers: "
            f"{available}"
        )

    providers = ["CUDAExecutionProvider"]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    modules.globals.execution_providers = providers
    modules.globals.execution_threads = 2
    modules.globals.enhancer_interval = 1
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.poisson_blend = False
    modules.globals.mask_blur = 3.0
    modules.globals.mask_erosion = 3
    modules.globals.frame_processors = ["face_swapper"]
    print(f"[Colab] ONNX providers: {providers}", flush=True)
    return providers


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB webcam image")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if STREAM_MAX_WIDTH > 0 and image.shape[1] > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / image.shape[1]
        image = cv2.resize(
            image,
            (STREAM_MAX_WIDTH, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class LiveProcessor:
    """Thread-safe, lazy-loading webcam processor used by Gradio callbacks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.source_face: Any | None = None
        self.source_path: str | None = None
        self.enhancer_name = "None"
        self.last_error = ""
        self._swap = None

    def _ensure_swapper(self):
        if self._swap is None:
            from modules.processors.frame import face_swapper

            self._swap = face_swapper
            if face_swapper.get_face_swapper() is None:
                raise RuntimeError("Could not load inswapper_128.onnx")
        return self._swap

    def set_source(self, image: np.ndarray | None) -> str:
        if image is None:
            with self._lock:
                self.source_face = None
                self.source_path = None
            return "Source face cleared. Upload a clear, front-facing portrait."

        try:
            from modules.face_analyser import get_one_face

            bgr = _as_bgr(image)
            face = get_one_face(bgr)
            if face is None or getattr(face, "normed_embedding", None) is None:
                return "No usable face found in the source image."
            with self._lock:
                self.source_face = face
                self.source_path = "browser-upload"
            return "Source face ready. Start the webcam stream."
        except Exception as exc:
            self.last_error = str(exc)
            return f"Source error: {exc}"

    def set_enhancer(self, name: str) -> str:
        name = name or "None"
        if name not in {"None", "GPEN-512", "GFPGAN"}:
            name = "None"
        with self._lock:
            self.enhancer_name = name
        return f"Enhancer: {name}"

    def set_poisson(self, enabled: bool) -> str:
        modules.globals.poisson_blend = bool(enabled)
        return "Blend: Poisson / SeamlessClone" if enabled else "Blend: fast alpha"

    def clear(self) -> str:
        with self._lock:
            self.source_face = None
            self.source_path = None
        return "Source face cleared."

    def _enhance(self, frame: np.ndarray, target_face: Any) -> np.ndarray:
        if self.enhancer_name == "None":
            return frame

        if self.enhancer_name == "GPEN-512":
            from modules.processors.frame import face_enhancer_gpen512 as enhancer
        else:
            from modules.processors.frame import face_enhancer as enhancer

        modules.globals.enhancer_interval = 1
        return enhancer.process_frame(
            None,
            frame,
            detected_faces=[target_face],
        )

    def process(
        self,
        image: np.ndarray | None,
        enhancer_name: str,
        poisson: bool,
    ) -> np.ndarray | None:
        if image is None:
            return None

        with self._lock:
            if enhancer_name != self.enhancer_name:
                self.enhancer_name = enhancer_name or "None"
            modules.globals.poisson_blend = bool(poisson)
            source_face = self.source_face

        if source_face is None:
            return image

        try:
            from modules.face_analyser import get_one_face

            bgr = _as_bgr(image)
            target_face = get_one_face(bgr)
            if target_face is None:
                return image
            swapper = self._ensure_swapper()
            result = swapper.swap_face(source_face, target_face, bgr)
            result = self._enhance(result, target_face)
            return _as_rgb(result)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[Colab] frame error: {exc}", flush=True)
            return image


def build_demo(processor: LiveProcessor) -> gr.Blocks:
    """Construct the browser UI without creating a desktop window."""
    with gr.Blocks(title="Deep-Live-Cam · Colab T4") as demo:
        gr.Markdown(
            "## Deep-Live-Cam trên Google Colab\n"
            "Tải ảnh source, cấp quyền webcam cho trình duyệt rồi bật webcam. "
            "Khung hình được xử lý trên GPU T4."
        )
        with gr.Row():
            source = gr.Image(
                sources=["upload"], type="numpy", label="Source face",
            )
            webcam = gr.Image(
                sources=["webcam"], type="numpy", streaming=True,
                label="Webcam từ Mac",
            )
            preview = gr.Image(type="numpy", streaming=True, label="Preview")

        with gr.Row():
            enhancer = gr.Dropdown(
                ["None", "GPEN-512", "GFPGAN"], value="None",
                label="Enhancer (T4)",
            )
            poisson = gr.Checkbox(False, label="Poisson / SeamlessClone")
            clear = gr.Button("Clear source")
        status = gr.Markdown("Upload a source face to begin.")

        source.change(processor.set_source, inputs=source, outputs=status)
        enhancer.change(processor.set_enhancer, inputs=enhancer, outputs=status)
        poisson.change(processor.set_poisson, inputs=poisson, outputs=status)
        clear.click(processor.clear, outputs=status)
        webcam.stream(
            processor.process,
            inputs=[webcam, enhancer, poisson],
            outputs=preview,
            concurrency_limit=1,
            trigger_mode="always_last",
            stream_every=0.08,
            time_limit=3600,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true",
        default=os.environ.get("GRADIO_SHARE", "1") == "1",
        help="Create a public Gradio URL for the Mac browser.",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    configure_cuda()
    processor = LiveProcessor()
    demo = build_demo(processor)
    print("[Colab] Starting Gradio. Keep this cell running.", flush=True)
    demo.queue(max_size=2, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
