import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import modules.core as core
import modules.globals as settings
from modules.processors.frame import face_swapper
from modules.processors.frame import _onnx_enhancer as enhancer
from modules.processors.frame import face_enhancer as gfpgan
from modules.processors.frame import face_masking


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

    def test_detail_blend_restores_camera_texture_without_changing_shape(self):
        restored = np.full((32, 32, 3), 120, dtype=np.uint8)
        camera = restored.copy()
        camera[16, 16] = (255, 255, 255)
        blended = enhancer.blend_high_frequency(camera, restored, strength=1.0)
        self.assertEqual(blended.shape, restored.shape)
        self.assertGreater(int(blended[16, 16, 0]), int(restored[16, 16, 0]))
        self.assertTrue(np.all(blended <= 255))

    def test_hairline_guard_excludes_upper_aligned_rows(self):
        with patch.object(settings, "hairline_guard", 0.16), \
             patch.object(settings, "mask_blur", 1.5):
            mask = enhancer.apply_hairline_guard(np.full((128, 128), 255, dtype=np.uint8))
        self.assertEqual(int(mask[0, 64]), 0)
        self.assertGreater(int(mask[64, 64]), 240)

    def test_mouth_slider_does_not_touch_eye_rows(self):
        landmarks = np.zeros((106, 2), dtype=np.float32)
        left_eye = np.column_stack((np.linspace(48, 72, 10), np.full(10, 48)))
        right_eye = np.column_stack((np.linspace(128, 152, 10), np.full(10, 48)))
        mouth = np.column_stack((np.linspace(78, 122, 12), np.full(12, 132)))
        landmarks[33:43] = left_eye
        landmarks[87:97] = right_eye
        landmarks[52:64] = mouth
        face = SimpleNamespace(landmark_2d_106=landmarks)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch.object(settings, "mouth_mask_size", 100.0):
            mouth_mask, _, mouth_box, _ = face_masking.create_lower_mouth_mask(face, frame)
        eye_mask, _, eye_box, _ = face_masking.create_eyes_mask(face, frame)
        self.assertGreater(mouth_box[1], eye_box[3])
        self.assertEqual(int(np.count_nonzero(mouth_mask[:90])), 0)
        self.assertEqual(int(np.count_nonzero(eye_mask[90:])), 0)

    def test_68_point_mouth_and_eyes_stay_separate(self):
        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[36:42] = np.column_stack((np.linspace(40, 55, 6), np.full(6, 45)))
        landmarks[42:48] = np.column_stack((np.linspace(85, 100, 6), np.full(6, 45)))
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        landmarks[48:68] = np.column_stack((70 + 20 * np.cos(angles), 110 + 7 * np.sin(angles)))
        face = SimpleNamespace(landmark_2d_68=landmarks)
        frame = np.zeros((160, 140, 3), dtype=np.uint8)
        with patch.object(settings, "mouth_mask_size", 100.0):
            mouth, _, _, _ = face_masking.create_lower_mouth_mask(face, frame)
        eyes, _, _, _ = face_masking.create_eyes_mask(face, frame)
        self.assertGreater(int(np.count_nonzero(mouth)), 0)
        self.assertGreater(int(np.count_nonzero(eyes)), 0)
        self.assertEqual(int(np.count_nonzero((mouth > 0) & (eyes > 0))), 0)

    def test_pasting_mouth_preserves_eye_pixels(self):
        landmarks = np.zeros((106, 2), dtype=np.float32)
        landmarks[33:43] = np.column_stack((np.linspace(35, 55, 10), np.full(10, 40)))
        landmarks[87:97] = np.column_stack((np.linspace(85, 105, 10), np.full(10, 40)))
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        landmarks[52:64] = np.column_stack((70 + 18 * np.cos(angles), 108 + 6 * np.sin(angles)))
        face = SimpleNamespace(landmark_2d_106=landmarks)
        original = np.full((160, 140, 3), 40, dtype=np.uint8)
        swapped = np.full_like(original, 180)
        with patch.object(settings, "mouth_mask_size", 100.0):
            _, cutout, box, polygon = face_masking.create_lower_mouth_mask(face, original)
        result = face_swapper.apply_mouth_area(swapped, cutout, box, polygon)
        self.assertTrue(np.array_equal(result[:70], np.full_like(result[:70], 180)))
        self.assertLess(int(result[108, 70, 0]), 180)

    def test_adaptive_film_grain_adds_subtle_texture(self):
        camera = np.full((32, 32, 3), 120, dtype=np.uint8)
        restored = camera.copy()
        grain = enhancer.add_adaptive_film_grain(camera, restored, strength=0.5)
        self.assertEqual(grain.shape, restored.shape)
        self.assertGreater(float(np.std(grain.astype(np.int16))), 0.1)

    def test_masked_lab_color_match_changes_only_face_roi(self):
        source = np.zeros((16, 16, 3), dtype=np.uint8)
        source[:] = (180, 80, 50)
        target = np.zeros_like(source)
        target[:] = (50, 120, 180)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        corrected = face_masking.match_color_lab(source, target, mask)
        self.assertGreater(float(np.mean(corrected[4:12, 4:12, 2])), float(np.mean(source[4:12, 4:12, 2])))


if __name__ == "__main__":
    unittest.main()
