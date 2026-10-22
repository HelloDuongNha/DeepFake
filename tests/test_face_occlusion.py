import unittest
import numpy as np
from modules.processors.frame.face_occlusion import restore_occluded_pixels


class OcclusionTests(unittest.TestCase):
    def test_foreground_overrides_generated_nose_and_uses_current_camera(self):
        affine = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
        camera = np.full((256, 256, 3), 130, np.uint8)
        generated = np.full_like(camera, 70)
        visible = np.ones((256, 256), np.float32)
        visible[120:256, 100:125] = 0.
        camera[120:256, 100:125] = [145, 170, 195]
        out = restore_occluded_pixels(generated.copy(), camera, affine, visible)
        np.testing.assert_array_equal(out[140:210, 104:120], camera[140:210, 104:120])
        np.testing.assert_array_equal(out[:100], generated[:100])
        camera[120:256, 100:125] = [180, 195, 210]
        out = restore_occluded_pixels(generated.copy(), camera, affine, visible)
        np.testing.assert_array_equal(out[140:210, 104:120], camera[140:210, 104:120])

    def test_no_occlusion_or_invalid_mask_keeps_swapped_pixels(self):
        affine = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
        generated = np.full((256, 256, 3), 70, np.uint8)
        camera = np.full_like(generated, 140)
        for visible in (np.ones((256, 256), np.float32), np.full((256, 256), np.nan)):
            np.testing.assert_array_equal(restore_occluded_pixels(
                generated.copy(), camera, affine, visible), generated)

