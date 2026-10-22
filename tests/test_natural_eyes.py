import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from modules.processors.frame.face_detail import preserve_eye_apertures


class NaturalEyesTests(unittest.TestCase):
    def test_iris_is_current_camera_and_no_pixels_escape_eyelids(self):
        points = np.zeros((106, 2), np.float32)
        contour = np.array([[30, 40], [34, 37], [40, 36], [46, 37], [50, 40],
                            [46, 43], [40, 44], [34, 43], [40, 40], [41, 40]], np.float32)
        points[33:43] = contour
        points[87:97] = contour + [40, 0]
        generated = np.full((100, 120, 3), 120, np.uint8)
        camera = np.full_like(generated, 220)
        camera[39:42, 39:42] = 25
        camera[39:42, 79:82] = 25
        result = preserve_eye_apertures(generated.copy(), camera, SimpleNamespace(landmark_2d_106=points))
        self.assertEqual(int(result[40, 40, 0]), 25)
        self.assertEqual(int(result[40, 80, 0]), 25)
        support = np.zeros((100, 120), np.uint8)
        for eye in (contour, contour + [40, 0]):
            cv2.fillConvexPoly(support, cv2.convexHull(eye.astype(np.int32)), 255)
        np.testing.assert_array_equal(result[support == 0], generated[support == 0])
        camera[39:42, 39:42] = 80
        updated = preserve_eye_apertures(generated.copy(), camera, SimpleNamespace(landmark_2d_106=points))
        self.assertEqual(int(updated[40, 40, 0]), 80)

    def test_invalid_landmarks_or_frame_shape_leave_output_unchanged(self):
        frame = np.full((100, 120, 3), 120, np.uint8)
        for landmarks in (None, np.full((106, 2), np.nan), np.zeros((5, 2))):
            np.testing.assert_array_equal(
                preserve_eye_apertures(frame.copy(), frame, SimpleNamespace(landmark_2d_106=landmarks)), frame)
