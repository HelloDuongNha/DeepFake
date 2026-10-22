import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from modules.processors.frame.face_detail import preserve_natural_nose


class NaturalNoseTests(unittest.TestCase):
    def test_lower_nostril_shadow_is_not_on_the_feather_boundary(self):
        points = np.zeros((106, 2), np.float32)
        points[74:87] = [[50,35],[38,45],[35,55],[37,61],[42,64],[47,66],
                          [50,67],[55,66],[60,64],[63,61],[65,55],[62,45],[50,55]]
        generated = np.full((100,100,3),140,np.uint8)
        camera = generated.copy()
        camera[66,50] = 25
        out = preserve_natural_nose(generated.copy(), camera,
                                    SimpleNamespace(landmark_2d_106=points))
        self.assertLess(int(out[66,50,0]), 40)

    def test_camera_nostril_is_current_and_outside_nose_is_unchanged(self):
        points = np.zeros((106, 2), np.float32)
        contour = np.array([[50, 35], [38, 45], [35, 55], [37, 61], [42, 64],
                            [47, 66], [50, 67], [55, 66], [60, 64], [63, 61],
                            [65, 55], [62, 45], [50, 55]], np.float32)
        points[74:87] = contour
        face = SimpleNamespace(landmark_2d_106=points)
        generated = np.full((100, 100, 3), 140, np.uint8)
        camera = generated.copy()
        cv2.circle(camera, (43, 58), 2, (25, 25, 25), -1)
        out = preserve_natural_nose(generated.copy(), camera, face)
        self.assertEqual(int(out[58, 43, 0]), 25)
        mask = np.zeros((100, 100), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(contour.astype(np.int32)), 255)
        np.testing.assert_array_equal(out[mask == 0], generated[mask == 0])
        cv2.circle(camera, (43, 58), 2, (60, 60, 60), -1)
        out = preserve_natural_nose(generated.copy(), camera, face)
        self.assertEqual(int(out[58, 43, 0]), 60)

    def test_missing_or_invalid_geometry_does_not_expose_camera(self):
        frame = np.full((100, 100, 3), 140, np.uint8)
        for points in (None, np.full((106, 2), np.nan), np.zeros((106, 2))):
            np.testing.assert_array_equal(preserve_natural_nose(
                frame.copy(), np.zeros_like(frame), SimpleNamespace(landmark_2d_106=points)), frame)
