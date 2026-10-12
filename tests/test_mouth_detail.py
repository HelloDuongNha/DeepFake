import unittest
from types import SimpleNamespace

import numpy as np

from modules.processors.frame.face_masking import preserve_open_mouth_interior


def _face(opening=True, turned=False):
    landmarks = np.zeros((106, 2), np.float32)
    locations = {
        52: (60, 104), 65: (65, 100), 66: (80, 96),
        62: (100, 96), 70: (120, 96), 69: (135, 100),
        61: (140, 104), 58: (135, 120), 57: (120, 122),
        60: (100, 124 if opening else 101), 54: (80, 122),
        55: (65, 120),
    }
    for index, point in locations.items():
        landmarks[index] = point
    keypoints = np.array([
        [70, 55], [130, 55], [150 if turned else 100, 80],
        [82, 120], [118, 120],
    ], np.float32)
    return SimpleNamespace(landmark_2d_106=landmarks, kps=keypoints)


class MouthDetailTests(unittest.TestCase):
    def setUp(self):
        self.swapped = np.full((200, 200, 3), 50, np.uint8)
        self.camera = np.full((200, 200, 3), 200, np.uint8)

    def test_open_mouth_keeps_camera_interior_only(self):
        result = preserve_open_mouth_interior(
            self.swapped.copy(), self.camera, _face()
        )
        self.assertGreater(result[108, 100, 0], 150)
        np.testing.assert_array_equal(result[85, 100], [50, 50, 50])
        np.testing.assert_array_equal(result[100, 30], [50, 50, 50])

    def test_closed_or_turned_mouth_stays_swapped(self):
        for face in (_face(opening=False), _face(turned=True)):
            with self.subTest(face=face):
                result = preserve_open_mouth_interior(
                    self.swapped.copy(), self.camera, face
                )
                np.testing.assert_array_equal(result, self.swapped)

    def test_missing_landmarks_stays_swapped(self):
        result = preserve_open_mouth_interior(
            self.swapped.copy(), self.camera,
            SimpleNamespace(kps=_face().kps),
        )
        np.testing.assert_array_equal(result, self.swapped)


if __name__ == "__main__":
    unittest.main()
