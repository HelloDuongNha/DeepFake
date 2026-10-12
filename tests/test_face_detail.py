import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from modules.processors.frame.face_detail import (
    enhance_eye_clarity, skin_microtexture_gate,
)


def face():
    return SimpleNamespace(
        bbox=np.array([40, 20, 280, 290], np.float32),
        kps=np.array([[112, 130], [208, 130], [160, 184],
                      [131, 235], [189, 235]], np.float32),
    )


class FaceDetailTests(unittest.TestCase):
    def test_transfer_does_not_add_camera_mole_or_remove_source_mole(self):
        import modules.core
        from modules.processors.frame.face_swapper import _restore_camera_microtexture

        generated = np.full((128, 128, 3), 150, np.uint8)
        camera = generated.copy()
        cv2.circle(camera, (89, 79), 2, (60, 60, 60), -1)
        output = _restore_camera_microtexture(generated, camera, .45)
        self.assertGreaterEqual(int(output[79, 89, 0]), 149)
        cv2.circle(generated, (78, 98), 2, (60, 60, 60), -1)
        output = _restore_camera_microtexture(generated, np.full_like(camera, 150), .45)
        np.testing.assert_array_equal(output, generated)

    def test_camera_mole_is_excluded_without_erasing_a_long_wrinkle(self):
        camera = np.full((128, 128, 3), 150, np.uint8)
        cv2.circle(camera, (89, 79), 2, (60, 60, 60), -1)
        cv2.line(camera, (37, 86), (75, 86), (80, 80, 80), 1)
        gate = skin_microtexture_gate(camera)
        self.assertLess(float(gate[79, 89]), .01)
        self.assertGreater(float(gate[86, 55]), .99)
        self.assertGreater(float(gate[104, 64]), .99)

    def test_camera_eye_and_brow_contrast_is_not_copied(self):
        camera = np.full((128, 128, 3), 150, np.uint8)
        gate = skin_microtexture_gate(camera)
        self.assertLess(float(gate[59, 44]), .01)
        self.assertLess(float(gate[43, 84]), .01)
        self.assertGreater(float(gate[96, 64]), .99)

    def test_nostril_and_inner_mouth_shadows_are_not_treated_as_moles(self):
        camera = np.full((128, 128, 3), 150, np.uint8)
        for center in ((64, 73), (64, 95)):
            cv2.circle(camera, center, 2, (40, 40, 40), -1)
        gate = skin_microtexture_gate(camera)
        self.assertGreater(float(gate[73, 64]), .99)
        self.assertGreater(float(gate[95, 64]), .99)

    def test_zero_sharpness_preserves_all_pixels(self):
        image = np.random.default_rng(2).integers(0, 256, (320, 320, 3), np.uint8)
        np.testing.assert_array_equal(enhance_eye_clarity(image.copy(), face(), 0), image)

    def test_eye_edges_improve_without_changing_cheeks_nose_mouth_or_rim(self):
        image = np.full((320, 320, 3), (125, 145, 165), np.uint8)
        for x in (112, 208):
            cv2.ellipse(image, (x, 130), (23, 5), 0, 0, 360, (35, 35, 35), -1)
            cv2.line(image, (x - 27, 105), (x + 27, 103), (40, 40, 40), 4)
        image = cv2.GaussianBlur(image, (0, 0), 2.)
        original = image.copy()
        output = enhance_eye_clarity(image, face(), 5)
        before = cv2.cvtColor(original[97:138, 80:145], cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(output[97:138, 80:145], cv2.COLOR_BGR2GRAY)
        self.assertGreater(float(cv2.Laplacian(after, cv2.CV_32F).var()),
                           float(cv2.Laplacian(before, cv2.CV_32F).var()))
        for y, x in ((190, 160), (237, 160), (210, 80), (130, 42), (50, 160)):
            np.testing.assert_array_equal(output[y, x], original[y, x])
        self.assertLessEqual(int(np.abs(output.astype(int) - original.astype(int)).max()), 15)

    def test_same_input_is_deterministic_and_blink_uses_current_pixels(self):
        image = np.full((320, 320, 3), 140, np.uint8)
        cv2.line(image, (90, 130), (135, 130), (30, 30, 30), 2)
        first = enhance_eye_clarity(image.copy(), face(), 5)
        second = enhance_eye_clarity(image.copy(), face(), 5)
        np.testing.assert_array_equal(first, second)
        flat = np.full_like(image, 140)
        np.testing.assert_array_equal(enhance_eye_clarity(flat.copy(), face(), 5), flat)

    def test_invalid_geometry_and_profile_are_skipped(self):
        image = np.full((320, 320, 3), 140, np.uint8)
        invalid = face()
        invalid.kps[0] = np.nan
        np.testing.assert_array_equal(enhance_eye_clarity(image.copy(), invalid, 5), image)
        profile = face()
        profile.kps[2, 0] = 240
        np.testing.assert_array_equal(enhance_eye_clarity(image.copy(), profile, 5), image)


if __name__ == '__main__':
    unittest.main()
