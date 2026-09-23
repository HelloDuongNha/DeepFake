import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modules.ui import (
    _frame_or_camera_fallback,
    _is_valid_camera_frame,
    get_available_cameras,
    selected_camera_index,
)
from modules.video_capture import VideoCapturer


class CameraSelectionTests(unittest.TestCase):
    def test_macos_camera_list_is_plain_zero_and_one(self):
        with patch("modules.ui.platform.system", return_value="Darwin"):
            self.assertEqual(
                get_available_cameras(),
                ([0, 1], ["Camera 0", "Camera 1"]),
            )

    def test_combo_stores_plain_integer_camera_index(self):
        class Combo:
            def __init__(self, data):
                self.data = data

            def currentData(self, *_args):
                return self.data

        self.assertEqual(selected_camera_index(Combo(0)), 0)
        self.assertEqual(selected_camera_index(Combo(1)), 1)
        self.assertIsNone(selected_camera_index(Combo("Camera 0")))

    def test_video_capturer_normalizes_index_to_plain_int(self):
        self.assertIs(type(VideoCapturer(np.int64(1)).device_index), int)
        with self.assertRaises(TypeError):
            VideoCapturer("Camera 1")

    def test_macos_capture_uses_opencv_avfoundation_without_mjpg(self):
        class Capture:
            def __init__(self):
                self.set_properties = []

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.set_properties.append(prop)
                return True

            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return 640
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 360
                if prop == cv2.CAP_PROP_FPS:
                    return 30
                return 0

            def read(self):
                return True, np.full((4, 6, 3), 25, dtype=np.uint8)

            def release(self):
                pass

        capture = Capture()
        with patch("modules.video_capture.platform.system", return_value="Darwin"), \
             patch("modules.video_capture.cv2.VideoCapture", return_value=capture) as open_camera:
            video = VideoCapturer(np.int64(1))
            self.assertTrue(video.start(640, 360, 30))
        self.assertEqual(open_camera.call_args.args, (1, cv2.CAP_AVFOUNDATION))
        self.assertNotIn(cv2.CAP_PROP_FOURCC, capture.set_properties)

    def test_empty_processing_result_falls_back_to_raw_camera_frame(self):
        raw = np.full((12, 16, 3), 77, dtype=np.uint8)
        self.assertFalse(_is_valid_camera_frame(None))
        self.assertFalse(_is_valid_camera_frame(np.empty((0, 0, 3), dtype=np.uint8)))
        self.assertIs(_frame_or_camera_fallback(None, raw, "test"), raw)


if __name__ == "__main__":
    unittest.main()
