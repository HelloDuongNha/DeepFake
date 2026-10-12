import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from modules.processors.frame import eye_detail


def _face(nose_x=300):
    return SimpleNamespace(
        bbox=np.array([150, 80, 450, 400], np.float32),
        kps=np.array([[250, 180], [350, 180], [nose_x, 240],
                      [270, 290], [330, 290]], np.float32),
    )


class EyeDetailTests(unittest.TestCase):
    def tearDown(self):
        eye_detail.stop_live_eye_detail()

    def test_detail_preserves_pixels_outside_eyes(self):
        processed = np.full((500, 600, 3), 75, np.uint8)
        def restore(frame, *, detected_faces, use_temporal_cache):
            self.assertEqual(len(detected_faces), 1)
            self.assertFalse(use_temporal_cache)
            return np.full_like(frame, 180)

        with (
            patch.object(eye_detail.os.path, "isfile", return_value=True),
            patch.object(eye_detail, "enhance_face", side_effect=restore),
        ):
            result = eye_detail.apply_eye_detail(_face(), processed)

        self.assertGreater(result[180, 250, 0], processed[180, 250, 0])
        np.testing.assert_array_equal(result[240, 300], processed[240, 300])
        np.testing.assert_array_equal(result[450, 300], processed[450, 300])
        np.testing.assert_array_equal(processed[180, 250], [75, 75, 75])

    def test_detail_fades_out_on_turn(self):
        frame = np.full((500, 600, 3), 75, np.uint8)
        frontal = eye_detail._detail_weight(_face(), frame.shape)
        slight_turn = eye_detail._detail_weight(_face(nose_x=320), frame.shape)
        profile = eye_detail._detail_weight(_face(nose_x=360), frame.shape)
        self.assertGreater(frontal, slight_turn)
        self.assertGreater(slight_turn, profile)
        self.assertEqual(profile, 0.0)

    def test_live_detail_does_not_wait_for_restoration(self):
        frame = np.full((500, 600, 3), 75, np.uint8)
        started = threading.Event()
        release = threading.Event()
        source = SimpleNamespace(embedding=np.ones(512, np.float32))

        def restore(image, *, detected_faces, use_temporal_cache):
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test worker was not released")
            return np.full_like(image, 180)

        with (
            patch.object(eye_detail.os.path, "isfile", return_value=True),
            patch.object(eye_detail, "enhance_face", side_effect=restore),
        ):
            start = time.perf_counter()
            initial = eye_detail.apply_eye_detail_live(_face(), frame.copy(), source)
            elapsed = time.perf_counter() - start
            self.assertLess(elapsed, .10)
            np.testing.assert_array_equal(initial, frame)
            self.assertTrue(started.wait(timeout=1))
            release.set()
            eye_detail._live_future.result(timeout=2)
            result = eye_detail.apply_eye_detail_live(_face(), frame.copy(), source)
            self.assertGreater(result[180, 250, 0], frame[180, 250, 0])
            np.testing.assert_array_equal(result[240, 300], frame[240, 300])

    def test_live_detail_rejects_changed_expression(self):
        face = _face()
        frame = np.full((500, 600, 3), 75, np.uint8)
        mask = eye_detail._eye_mask(frame.shape, face)
        points = np.column_stack(np.nonzero(mask))
        y1, x1 = points.min(axis=0)
        y2, x2 = points.max(axis=0) + 1
        rect = (int(x1), int(y1), int(x2-x1), int(y2-y1))
        original = frame[y1:y2, x1:x2].copy()
        detail = eye_detail._LiveDetail(
            1, time.monotonic(), frame.shape, face.kps.copy(), rect,
            original, np.full_like(original, 100, dtype=np.float32),
            mask[y1:y2, x1:x2].copy(),
        )
        changed = frame.copy()
        changed[y1:y2, x1:x2] = 0
        before = changed.copy()
        result = eye_detail._apply_live_result(
            changed, face, detail, 1.0, time.monotonic(),
        )
        np.testing.assert_array_equal(result, before)

    def test_live_detail_rejects_stale_or_moved_face(self):
        face = _face()
        frame = np.full((500, 600, 3), 75, np.uint8)
        mask = eye_detail._eye_mask(frame.shape, face)
        x, y, width, height = cv2.boundingRect(cv2.findNonZero(mask))
        original = frame[y:y + height, x:x + width].copy()
        detail = eye_detail._LiveDetail(
            1, time.monotonic() - 1.0, frame.shape, face.kps.copy(),
            (x, y, width, height), original,
            np.full_like(original, 100, dtype=np.float32),
            mask[y:y + height, x:x + width].copy(),
        )
        np.testing.assert_array_equal(
            eye_detail._apply_live_result(frame.copy(), face, detail, 1.0,
                                          time.monotonic()), frame,
        )
        detail.captured_at = time.monotonic()
        moved = _face()
        moved.kps[0] += (80, 40)
        np.testing.assert_array_equal(
            eye_detail._apply_live_result(frame.copy(), moved, detail, 1.0,
                                          time.monotonic()), frame,
        )


if __name__ == "__main__":
    unittest.main()
