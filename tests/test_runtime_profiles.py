import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import modules.core as core
import modules.globals as settings
from modules.processors.frame import face_swapper
from modules.processors.frame import _onnx_enhancer as enhancer
from modules.processors.frame import face_enhancer as gfpgan


class RuntimeProfileTests(unittest.TestCase):
    def test_mask_erosion_reduces_coverage(self):
        with patch.object(settings, "mask_blur", 0.0):
            with patch.object(settings, "mask_erosion", 0):
                wide = face_swapper._get_soft_alpha(128).copy()
            with patch.object(settings, "mask_erosion", 4):
                narrow = face_swapper._get_soft_alpha(128).copy()
        self.assertEqual(int(narrow[64, 64]), 255)
        self.assertEqual(int(narrow[0, 0]), 0)
        self.assertLess(np.count_nonzero(narrow), np.count_nonzero(wide))

    def test_windows_provider_priority(self):
        available = ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]
        with patch.object(core.onnxruntime, "get_available_providers", return_value=available), \
             patch.object(core.platform, "system", return_value="Windows"), \
             patch.object(core, "has_nvidia_gpu", return_value=True), \
             patch.dict(core.os.environ, {"DLC_PREFER_TENSORRT": "0"}):
            self.assertEqual(core.auto_execution_providers(), ["cuda"])
            self.assertEqual(core.decode_execution_providers(["auto"]), ["CUDAExecutionProvider"])
        with patch.object(core.onnxruntime, "get_available_providers", return_value=available), \
             patch.object(core.platform, "system", return_value="Windows"), \
             patch.object(core, "has_nvidia_gpu", return_value=True), \
             patch.dict(core.os.environ, {"DLC_PREFER_TENSORRT": "1"}):
            self.assertEqual(core.decode_execution_providers(["auto"]), [
                "TensorrtExecutionProvider", "CUDAExecutionProvider"
            ])

    def test_live_gpen_runs_inference_every_third_frame(self):
        class Session:
            def get_inputs(self):
                return [type("Input", (), {"name": "input"})()]

        session = Session()
        calls = []
        identity = np.array([[1., 0., 0.], [0., 1., 0.]])
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)

        def fake_inference(_session, _name, _blob):
            calls.append(1)
            return np.zeros((1, 3, 16, 16), dtype=np.float32)

        with patch.object(settings, "enhancer_interval", 3), \
             patch.object(enhancer, "_get_face_affine", return_value=(identity, identity)), \
             patch.object(enhancer, "run_inference", side_effect=fake_inference):
            for _ in range(5):
                result = enhancer.enhance_face_onnx(frame, object(), session, 16, live=True)
                self.assertEqual(result.shape, frame.shape)
        self.assertEqual(len(calls), 2)

    def test_cached_gfpgan_follows_current_face_position(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input", shape=[1, 3, 16, 16])]

        session = Session()
        positions = []
        inference_calls = []
        affine_calls = []

        def align(_frame, _landmarks, output_size):
            affine_calls.append(1)
            pos = len(affine_calls)
            return np.zeros((16, 16, 3), dtype=np.uint8), np.array(
                [[1., 0., float(pos)], [0., 1., 0.]]
            )

        def infer(*_args):
            inference_calls.append(1)
            return np.zeros((1, 3, 16, 16), dtype=np.float32)

        def paste(_frame, _face, matrix, output_size):
            positions.append(matrix[0, 2])

        gfpgan._enh_live_cache.update({
            "enhanced_bgr": None, "affine_matrix": None,
            "align_size": 0, "frame_count": 0,
        })
        face = SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32))
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch.object(settings, "enhancer_interval", 3), \
             patch.object(gfpgan, "get_face_enhancer", return_value=session), \
             patch.object(gfpgan, "_align_face", side_effect=align), \
             patch.object(gfpgan, "_preprocess_face", return_value=np.zeros((1, 3, 16, 16))), \
             patch.object(gfpgan, "_postprocess_face", return_value=np.zeros((16, 16, 3), dtype=np.uint8)), \
             patch.object(gfpgan, "_paste_back", side_effect=paste), \
             patch.object(enhancer, "run_inference", side_effect=infer):
            for _ in range(3):
                gfpgan.enhance_face(frame, detected_faces=[face])
        self.assertEqual(len(inference_calls), 1)
        self.assertEqual(positions, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
