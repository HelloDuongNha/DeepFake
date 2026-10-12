import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from insightface.utils import face_align

# Import the real processor during discovery, before legacy detector tests
# install lightweight modules.typing stubs for their isolated imports.
import modules.core
from modules.processors.frame import face_enhancer_gpen256

from modules.processors.frame.natural_swapper import (
    Natural256Swapper, compact_skin_marks, remove_target_only_marks,
)


class NaturalSwapperTests(unittest.TestCase):
    def test_model_switch_clears_the_old_model_and_restoration_cache(self):
        import modules.core
        import modules.globals as settings
        from modules.processors.frame import face_swapper, _onnx_enhancer
        with (patch.object(face_swapper, 'FACE_SWAPPER', object()),
              patch.object(settings, 'swapper_model', 'hyperswap_256'),
              patch.dict(_onnx_enhancer._LIVE_ENHANCER_CACHE, {1: {'old': True}}, clear=True)):
            face_swapper.select_swapper_model('inswapper_128')
            self.assertIsNone(face_swapper.FACE_SWAPPER)
            self.assertEqual(settings.swapper_model, 'inswapper_128')
            self.assertFalse(_onnx_enhancer._LIVE_ENHANCER_CACHE)

    def test_live_model_change_is_blocked_before_clearing_active_sessions(self):
        import modules.core
        import modules.globals as settings
        from modules.ui import MainWindow
        from unittest.mock import Mock
        combo = Mock()
        window = SimpleNamespace(cb_swapper=combo)
        with (patch.object(settings, 'webcam_preview_running', True),
              patch.object(settings, 'swapper_model', 'hyperswap_256'),
              patch('modules.processors.frame.face_swapper.select_swapper_model') as select,
              patch('modules.ui.save_switch_states') as save,
              patch('modules.ui.update_status')):
            MainWindow._on_swapper_change(window, 'INSwapper 128')
            select.assert_not_called()
            save.assert_not_called()
            combo.setCurrentText.assert_called_once_with('Natural 256')

    def test_gpen_is_gentler_only_after_an_applied_native_swap(self):
        import modules.core
        from modules.processors.frame import face_swapper, face_enhancer_gpen256 as gpen
        frame = np.full((256, 256, 3), 150, np.uint8)
        with (patch.object(gpen, 'get_enhancer', return_value=object()),
              patch.object(face_swapper, 'FACE_SWAPPER', SimpleNamespace(native_resolution=256)),
              patch.object(face_swapper, 'get_last_swap_status', return_value='applied') as status,
              patch.object(gpen, 'enhance_face_onnx', return_value=frame) as enhance):
            gpen.enhance_face(frame, None, live=True)
            self.assertEqual(enhance.call_args.kwargs['restoration_strength'], .4)
            status.return_value = 'no_target'
            gpen.enhance_face(frame, None, live=True)
            self.assertEqual(enhance.call_args.kwargs['restoration_strength'], 1.)

    def test_two_pixel_under_eye_mole_survives_detection_downsampling(self):
        camera = np.full((256, 256, 3), 150, np.uint8)
        camera[124, 176:178] = 80
        marks = compact_skin_marks(camera)
        self.assertEqual(int(marks[124, 176]), 255)
        cleaned = remove_target_only_marks(camera, camera, np.full_like(camera, 150))
        self.assertGreater(int(cleaned[124, 176, 0]), 140)

    def test_camera_mark_is_removed_only_when_absent_from_source(self):
        camera = np.full((256, 256, 3), 150, np.uint8)
        cv2.circle(camera, (205, 155), 3, (40, 40, 40), -1)
        generated = camera.copy()
        reference = np.full_like(camera, 150)
        cleaned = remove_target_only_marks(generated, camera, reference)
        self.assertGreater(int(cleaned[155, 205, 0]), 140)
        np.testing.assert_array_equal(cleaned[:130], generated[:130])
        np.testing.assert_array_equal(
            remove_target_only_marks(generated, camera, camera.copy()), generated
        )
        np.testing.assert_array_equal(
            remove_target_only_marks(generated, camera, None), generated
        )

    def test_nostrils_lips_eyes_and_long_wrinkles_are_protected(self):
        camera = np.full((256, 256, 3), 150, np.uint8)
        for center in ((128, 146), (128, 189), (163, 103)):
            cv2.circle(camera, center, 3, (30, 30, 30), -1)
        cv2.line(camera, (170, 163), (219, 163), (40, 40, 40), 1)
        self.assertFalse(np.any(compact_skin_marks(camera)))

    def test_preprocessing_and_rgb_output_match_the_model_contract(self):
        model = Natural256Swapper.__new__(Natural256Swapper)
        model.output_names = ['output']
        recorded = {}
        def infer(names, inputs):
            recorded.update(inputs)
            return [inputs['target'].copy()]
        model.session = SimpleNamespace(run=infer)
        kps = np.array([[90, 105], [166, 104], [128, 147],
                        [97, 186], [160, 185]], np.float32)
        frame = np.full((256, 256, 3), (31, 100, 205), np.uint8)
        identity = np.ones(512, np.float32) / np.sqrt(512)
        generated, affine = model.get(frame, SimpleNamespace(kps=kps),
                                      SimpleNamespace(normed_embedding=identity))
        np.testing.assert_allclose(affine, face_align.estimate_norm(kps, 128) * 2)
        self.assertEqual(recorded['target'].shape, (1, 3, 256, 256))
        self.assertEqual(recorded['source'].shape, (1, 512))
        np.testing.assert_allclose(recorded['target'][0, :, 100, 100],
                                   np.array([205, 100, 31]) / 127.5 - 1, atol=1e-6)
        np.testing.assert_array_equal(generated[100, 100], frame[100, 100])

    def test_corrupt_weights_are_rejected_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'corrupt.onnx'
            file.write_bytes(b'corrupt model')
            with patch('modules.processors.frame.natural_swapper.onnxruntime.InferenceSession') as factory:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    Natural256Swapper(str(file), ['CPUExecutionProvider'])
                factory.assert_not_called()

    def test_nonfinite_model_output_is_rejected(self):
        model = Natural256Swapper.__new__(Natural256Swapper)
        model.output_names = ['output']
        model.session = SimpleNamespace(run=lambda *args: [np.full((1, 3, 256, 256), np.nan, np.float32)])
        kps = np.array([[90, 105], [166, 104], [128, 147],
                        [97, 186], [160, 185]], np.float32)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            model.get(np.full((256, 256, 3), 150, np.uint8), SimpleNamespace(kps=kps),
                      SimpleNamespace(normed_embedding=np.ones(512, np.float32)))


if __name__ == '__main__':
    unittest.main()
