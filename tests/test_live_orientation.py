import unittest
from unittest.mock import patch

import numpy as np

import modules.globals as settings
from modules.ui import fit_live_preview, fill_image_to_size


class LiveOrientationTests(unittest.TestCase):
    def test_landscape_keeps_entire_frame_in_arbitrary_window(self):
        frame = np.zeros((90, 160, 3), np.uint8)
        frame[:, :10] = (255, 0, 0)
        frame[:, -10:] = (0, 255, 0)
        frame[:10, 10:-10] = (0, 0, 255)
        frame[-10:, 10:-10] = (255, 255, 255)
        before = frame.copy()
        with patch.object(settings, 'live_portrait', False):
            out = fit_live_preview(frame, 160, 240)
        # Same native width: no resizing and no missing edge pixels.
        np.testing.assert_array_equal(out, before)
        np.testing.assert_array_equal(frame, before)

    def test_portrait_preserves_existing_fill_behavior(self):
        frame = np.random.default_rng(7).integers(0, 256, (90, 160, 3), np.uint8)
        with patch.object(settings, 'live_portrait', True):
            out = fit_live_preview(frame, 45, 80)
        np.testing.assert_array_equal(out, fill_image_to_size(frame, 45, 80))
