import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

import modules.core as core
import modules.face_analyser as face_analyser
import modules.globals as settings
from modules.processors.frame import face_swapper
from modules.processors.frame import _onnx_enhancer as enhancer
from modules.processors.frame import face_enhancer as gfpgan
from modules.processors.frame import face_masking
from modules.processors.frame import face_parser


class RuntimeProfileTests(unittest.TestCase):
    def test_face_geometry_gate_rejects_morphed_feature_layout(self):
        valid = SimpleNamespace(
            bbox=np.array([20., 20., 100., 120.], dtype=np.float32),
            kps=np.array([[42., 52.], [78., 52.], [60., 72.],
                          [48., 94.], [72., 94.]], dtype=np.float32),
        )
        self.assertTrue(face_masking.is_plausible_face_geometry(valid))
        broken = SimpleNamespace(
            bbox=valid.bbox.copy(),
            kps=valid.kps.copy(),
        )
        broken.kps[3:5, 1] = 38.0
        self.assertFalse(face_masking.is_plausible_face_geometry(broken))

    def test_face_geometry_gate_accepts_foreshortened_true_profile(self):
        bbox = np.array([20., 20., 120., 130.], dtype=np.float32)
        profile = SimpleNamespace(
            bbox=bbox,
            kps=np.array([[72., 52.], [78., 52.], [48., 72.],
                          [68., 98.], [74., 98.]], dtype=np.float32),
        )
        self.assertTrue(face_masking.is_plausible_face_geometry(profile))
        collapsed_frontal = SimpleNamespace(
            bbox=bbox,
            kps=np.array([[72., 52.], [78., 52.], [75., 72.],
                          [72., 98.], [78., 98.]], dtype=np.float32),
        )
        self.assertFalse(
            face_masking.is_plausible_face_geometry(collapsed_frontal)
        )

    def test_roi_detector_maps_reacquired_face_back_to_frame(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        reference = SimpleNamespace(
            bbox=np.array([70., 60., 130., 140.], dtype=np.float32)
        )
        local_bbox = np.array([[50., 50., 110., 130., 0.9]], dtype=np.float32)
        local_kps = np.array([[[60., 70.], [100., 70.], [80., 90.],
                               [66., 112.], [94., 112.]]], dtype=np.float32)
        detector = SimpleNamespace(
            detect=lambda *_args, **_kwargs: (local_bbox, local_kps)
        )
        analyser = SimpleNamespace(det_model=detector)
        with patch.object(face_analyser, "get_face_analyser",
                          return_value=analyser):
            recovered = face_analyser.detect_one_face_near(frame, reference)
        self.assertIsNotNone(recovered)
        # Expanded crop starts at x=22 and y=0 for this reference bbox.
        np.testing.assert_allclose(recovered.bbox, [72, 50, 132, 130])
        np.testing.assert_allclose(recovered.kps[0], [82, 70])

    def test_face_mask_extrapolates_forehead_and_insets_temples(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        for point_count in (68, 106):
            with self.subTest(point_count=point_count):
                landmarks = np.full((point_count, 2), [64.0, 66.0],
                                    dtype=np.float32)
                jaw_count = 17 if point_count == 68 else 33
                jaw_angles = np.linspace(np.pi, 0.0, jaw_count)
                landmarks[:jaw_count] = np.column_stack((
                    64.0 + 44.0 * np.cos(jaw_angles),
                    58.0 + 48.0 * np.sin(jaw_angles),
                ))
                if point_count == 68:
                    landmarks[17:22] = np.column_stack(
                        (np.linspace(38, 58, 5), np.full(5, 50))
                    )
                    landmarks[22:27] = np.column_stack(
                        (np.linspace(70, 90, 5), np.full(5, 50))
                    )
                else:
                    landmarks[43:52] = np.column_stack(
                        (np.linspace(38, 58, 9), np.full(9, 50))
                    )
                    landmarks[97:106] = np.column_stack(
                        (np.linspace(70, 90, 9), np.full(9, 50))
                    )
                face = SimpleNamespace(
                    landmark_2d_106=landmarks if point_count == 106 else None,
                    landmark_2d_68=landmarks if point_count == 68 else None,
                    kps=np.array([[48, 54], [80, 54], [64, 70],
                                  [52, 84], [76, 84]], dtype=np.float32),
                )
                face_masking.reset_forehead_height_ema()
                with patch.object(settings, "mask_blur", 1.5), \
                     patch.object(settings, "mask_erosion", 4):
                    mask = face_masking.create_hairline_safe_mask(face, frame)
                self.assertEqual(int(mask[64, 64]), 255)
                self.assertGreater(int(mask[36, 64]), 245)
                self.assertGreater(int(mask[24, 64]), 128)
                self.assertLess(int(mask[14, 64]), 16)
                self.assertLess(int(mask[58, 21]), 16)
                self.assertGreater(int(mask[58, 40]), 0)
                self.assertGreater(int(mask[58, 88]), 0)
                self.assertGreater(int(mask[90, 35]), 180)

    def test_forehead_height_ema_rejects_single_frame_jump(self):
        face_masking.reset_forehead_height_ema()
        self.assertEqual(face_masking._smooth_forehead_height(20.0, 100.0), 20.0)
        self.assertEqual(face_masking._smooth_forehead_height(40.0, 100.0), 25.0)

    def test_lateral_artifact_cleanup_inpaints_strip_not_temple_skin(self):
        generated = np.full((128, 128, 3), 130, dtype=np.uint8)
        generated[24:88, 8:14] = 15
        generated[24:88, 114:120] = 15
        cleaned = face_swapper._clean_lateral_hair_artifacts(generated)
        self.assertGreater(int(cleaned[50, 10, 0]), 90)
        self.assertGreater(int(cleaned[50, 117, 0]), 90)
        self.assertEqual(int(cleaned[50, 30, 0]), 130)
        self.assertEqual(int(cleaned[100, 10, 0]), 130)

    def test_artifact_cleanup_does_not_smooth_broad_forehead_region(self):
        generated = np.full((128, 128, 3), 140, dtype=np.uint8)
        generated[8:34, 30:98] = 12
        cleaned = face_swapper._clean_lateral_hair_artifacts(generated)
        self.assertEqual(int(cleaned[20, 64, 0]), 12)
        self.assertEqual(int(cleaned[54, 64, 0]), 140)

    def test_artifact_cleanup_never_erases_protected_eyebrow(self):
        generated = np.full((128, 128, 3), 140, dtype=np.uint8)
        generated[22:29, 42:70] = 12
        generated[24:88, 8:14] = 12
        protected = np.zeros((128, 128), dtype=np.uint8)
        protected[18:34, 38:74] = 255
        cleaned = face_swapper._clean_lateral_hair_artifacts(
            generated, protected_features=protected
        )
        self.assertLess(int(cleaned[25, 55, 0]), 30)
        self.assertGreater(int(cleaned[50, 10, 0]), 90)

    def test_artifact_cleanup_removes_blue_temple_streak(self):
        generated = np.full((128, 128, 3), (125, 140, 160), dtype=np.uint8)
        generated[24:92, 12:17] = (210, 70, 45)
        cleaned = face_swapper._clean_lateral_hair_artifacts(generated)
        original_distance = np.linalg.norm(
            generated[55, 14].astype(np.float32)
            - generated[55, 50].astype(np.float32)
        )
        cleaned_distance = np.linalg.norm(
            cleaned[55, 14].astype(np.float32)
            - cleaned[55, 50].astype(np.float32)
        )
        self.assertLess(cleaned_distance, original_distance * 0.45)

    def test_artifact_cleanup_does_not_inpaint_broad_shadowed_cheek(self):
        generated = np.full((128, 128, 3), (125, 140, 160), dtype=np.uint8)
        generated[45:108, 0:32] = (70, 85, 105)
        cleaned = face_swapper._clean_lateral_hair_artifacts(generated)
        np.testing.assert_array_equal(cleaned[70, 15], generated[70, 15])

    def test_artifact_cleanup_removes_faint_diagonal_temple_line(self):
        generated = np.full((128, 128, 3), (125, 140, 160), dtype=np.uint8)
        cv2.line(generated, (18, 24), (39, 91), (102, 114, 132), 3)
        before = np.linalg.norm(
            generated[58, 29].astype(np.float32)
            - generated[58, 55].astype(np.float32)
        )
        cleaned = face_swapper._clean_lateral_hair_artifacts(generated)
        after = np.linalg.norm(
            cleaned[58, 29].astype(np.float32)
            - cleaned[58, 55].astype(np.float32)
        )
        self.assertLess(after, before * 0.55)

    def test_swap_restores_camera_microtexture_without_copying_flat_tone(self):
        generated = np.full((128, 128, 3), 120, dtype=np.uint8)
        camera = generated.copy()
        camera[64, 35:94] = 210
        restored = face_swapper._restore_camera_microtexture(
            generated, camera, 0.36
        )
        self.assertGreater(int(restored[64, 64, 0]), 120)
        flat = face_swapper._restore_camera_microtexture(
            generated, generated.copy(), 0.36
        )
        np.testing.assert_array_equal(flat, generated)

    def test_skin_tone_match_uses_cheeks_without_whitening_face(self):
        source = np.full((128, 128, 3), (190, 205, 225), dtype=np.uint8)
        target = np.full((128, 128, 3), (80, 110, 145), dtype=np.uint8)
        target[96:, :] = 255  # bright shirt must not affect cheek sampling
        corrected = face_masking.match_skin_tone_lab(source, target)
        self.assertLess(float(np.mean(corrected[70:90])),
                        float(np.mean(source[70:90])))
        self.assertGreater(float(np.mean(corrected[70:90])),
                           float(np.mean(target[70:90])) - 25.0)

    def test_skin_tone_match_follows_low_frequency_side_lighting(self):
        source = np.full((128, 128, 3), (120, 140, 170), dtype=np.uint8)
        target = np.empty_like(source)
        target[:, :64] = (65, 85, 110)
        target[:, 64:] = (145, 165, 195)
        mask = np.full((128, 128), 255, dtype=np.uint8)
        corrected = face_masking.match_skin_tone_lab(source, target, mask)
        left = float(np.mean(corrected[55:85, 25:50]))
        right = float(np.mean(corrected[55:85, 78:103]))
        self.assertGreater(right - left, 8.0)

    def test_boundary_harmonization_removes_dark_edge_without_touching_center(self):
        source = np.full((96, 96, 3), 70, dtype=np.uint8)
        target = np.full((96, 96, 3), 150, dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 36, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        corrected = face_masking.harmonize_mask_boundary(source, target, mask)
        self.assertAlmostEqual(int(corrected[48, 12, 0]), 150, delta=2)
        self.assertEqual(int(corrected[48, 48, 0]), int(source[48, 48, 0]))
        self.assertEqual(int(corrected[0, 0, 0]), int(source[0, 0, 0]))

    def test_boundary_harmonization_handles_direct_sunlight_contrast(self):
        source = np.full((96, 96, 3), 35, dtype=np.uint8)
        target = np.full((96, 96, 3), 230, dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 36, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        corrected = face_masking.harmonize_mask_boundary(source, target, mask)
        self.assertAlmostEqual(int(corrected[48, 12, 0]), 230, delta=2)
        self.assertEqual(int(corrected[48, 48, 0]), 35)

    def test_boundary_harmonization_preserves_high_frequency_texture(self):
        checker = ((np.indices((96, 96)).sum(axis=0) % 2) * 30 - 15)
        source = np.clip(80 + checker[:, :, None], 0, 255).astype(np.uint8)
        source = np.repeat(source, 3, axis=2)
        target = np.full_like(source, 205)
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[10:86, 10:86] = 255
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        preserve = np.full((96, 96), 255, dtype=np.uint8)
        corrected = face_masking.harmonize_mask_boundary(
            source, target, mask, preserve_mask=preserve
        )
        # Illumination moves to the camera level while the alternating source
        # detail remains clearly present in the corrected edge strip.
        strip = corrected[20:76, 11, 0].astype(np.float32)
        self.assertGreater(float(strip.mean()), 90.0)
        self.assertLess(float(strip.mean()), 120.0)
        self.assertGreater(float(strip.std()), 8.0)

    def test_boundary_decontamination_keeps_protected_eyebrow(self):
        source = np.full((96, 96, 3), 130, dtype=np.uint8)
        target = np.full((96, 96, 3), 155, dtype=np.uint8)
        source[40:57, 9:14] = 5
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[10:86, 10:86] = 255
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        unprotected = face_masking.harmonize_mask_boundary(source, target, mask)
        preserve = np.zeros((96, 96), dtype=np.uint8)
        preserve[47:51, 10:16] = 255
        protected = face_masking.harmonize_mask_boundary(
            source, target, mask, preserve_mask=preserve
        )
        self.assertGreater(int(unprotected[48, 10, 0]), 100)
        self.assertLess(int(protected[48, 10, 0]), 70)

    def test_lower_chin_boundary_is_sealed_without_changing_face_center(self):
        source = np.full((96, 96, 3), 105, dtype=np.uint8)
        target = np.full((96, 96, 3), 145, dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 36, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        corrected = face_masking.harmonize_mask_boundary(source, target, mask)
        self.assertGreater(int(corrected[81, 48, 0]), 135)
        self.assertEqual(int(corrected[48, 48, 0]), 105)

    def test_both_cheek_boundaries_are_sealed_symmetrically(self):
        source = np.full((96, 96, 3), 105, dtype=np.uint8)
        target = np.full((96, 96, 3), 145, dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 36, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        corrected = face_masking.harmonize_mask_boundary(source, target, mask)
        self.assertGreater(int(corrected[48, 15, 0]), 130)
        self.assertGreater(int(corrected[48, 81, 0]), 130)
        self.assertAlmostEqual(
            int(corrected[48, 15, 0]), int(corrected[48, 81, 0]), delta=1
        )
        self.assertEqual(int(corrected[48, 48, 0]), 105)

    def test_both_temple_boundaries_use_the_same_edge_seal(self):
        source = np.full((96, 96, 3), 105, dtype=np.uint8)
        target = np.full((96, 96, 3), 145, dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (48, 48), 36, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        corrected = face_masking.harmonize_mask_boundary(source, target, mask)
        self.assertGreater(int(corrected[30, 19, 0]), 130)
        self.assertGreater(int(corrected[30, 77, 0]), 130)
        self.assertAlmostEqual(
            int(corrected[30, 19, 0]), int(corrected[30, 77, 0]), delta=1
        )
        self.assertEqual(int(corrected[48, 48, 0]), 105)

    def test_paste_back_uses_stable_bbox_mask_despite_dense_landmark_changes(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fake = np.full((32, 32, 3), 180, dtype=np.uint8)
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        stable_guard = np.zeros((64, 64), dtype=np.uint8)
        cv2.ellipse(stable_guard, (16, 16), (13, 14), 0, 0, 360, 255, -1)
        target = SimpleNamespace(
            landmark_2d_106=None,
            landmark_2d_68=np.full((68, 2), [16.0, 16.0], dtype=np.float32),
            bbox=np.array([2., 2., 30., 30.], dtype=np.float32),
        )
        with patch.object(face_swapper, "_get_soft_alpha",
                          return_value=np.full((32, 32), 255, dtype=np.uint8)) as fallback_mask, \
             patch.object(face_swapper, "create_bbox_safety_mask",
                          return_value=stable_guard) as build_mask:
            first = face_swapper._fast_paste_back(
                frame.copy(), fake, fake, identity, target_face=target
            )
            target.landmark_2d_68 = np.column_stack((
                np.linspace(-500., 800., 68), np.linspace(900., -300., 68)
            )).astype(np.float32)
            second = face_swapper._fast_paste_back(
                frame.copy(), fake, fake, identity, target_face=target
            )
        self.assertEqual(build_mask.call_count, 2)
        self.assertEqual(fallback_mask.call_count, 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first[16, 16, 0]), 180)
        self.assertEqual(int(first[0, 0, 0]), 0)

    def test_hairline_mask_clamps_corrupt_landmark_to_face_bbox(self):
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, 106, endpoint=False)
        landmarks = np.column_stack((
            80.0 + 38.0 * np.cos(angles),
            82.0 + 48.0 * np.sin(angles),
        )).astype(np.float32)
        landmarks[12] = [5000.0, -4000.0]
        face = SimpleNamespace(
            landmark_2d_106=landmarks,
            landmark_2d_68=None,
            kps=np.array([[62, 70], [98, 70], [80, 86],
                          [68, 104], [92, 104]], dtype=np.float32),
            bbox=np.array([40, 30, 120, 138], dtype=np.float32),
        )
        mask = face_masking.create_hairline_safe_mask(
            face, frame, blur_sigma=0, erosion_px=0
        )
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[0, 159]), 0)
        self.assertGreater(int(mask[82, 80]), 0)

    def test_bbox_safety_mask_excludes_hair_and_ears_without_a_box(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        face = SimpleNamespace(
            bbox=np.array([40., 30., 140., 155.], dtype=np.float32)
        )
        mask = face_masking.create_bbox_safety_mask(face, frame)
        self.assertEqual(int(mask[30, 90]), 0)   # hair above face
        self.assertLessEqual(int(mask[95, 40]), 16)   # left ear edge
        self.assertLessEqual(int(mask[95, 140]), 16)  # right ear edge
        self.assertGreater(int(mask[98, 90]), 245)
        # An ellipse cannot expose the four corners of an aligned crop box.
        self.assertEqual(int(mask[45, 50]), 0)
        self.assertEqual(int(mask[45, 130]), 0)

    def test_bbox_oval_uses_landmark_independent_eyebrow_envelope(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        landmarks = np.full((106, 2), [90., 100.], dtype=np.float32)
        landmarks[43:52] = np.column_stack((
            np.linspace(48, 73, 9), np.full(9, 60.0)
        ))
        landmarks[97:106] = np.column_stack((
            np.linspace(112, 135, 9), np.full(9, 60.0)
        ))
        face = SimpleNamespace(
            bbox=np.array([40., 30., 140., 155.], dtype=np.float32),
            kps=np.array([[65., 75.], [115., 75.], [90., 92.],
                          [75., 118.], [105., 118.]], dtype=np.float32),
            landmark_2d_106=landmarks,
            landmark_2d_68=None,
        )
        first = face_masking.create_bbox_safety_mask(face, frame)
        face.landmark_2d_106[:] = np.column_stack((
            np.linspace(-500, 800, 106), np.linspace(900, -300, 106)
        ))
        second = face_masking.create_bbox_safety_mask(face, frame)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first[60, 130]), 128)  # stable brow tail
        self.assertLess(int(first[95, 40]), 20)       # narrowed left temple
        self.assertLess(int(first[95, 140]), 20)      # ear still excluded
        self.assertEqual(int(first[30, 90]), 0)       # hair still excluded

    def test_strong_yaw_narrows_only_the_rear_side_of_bbox_oval(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        bbox = np.array([40., 30., 140., 155.], dtype=np.float32)
        base_kps = np.array([[65., 75.], [115., 75.], [90., 92.],
                             [75., 118.], [105., 118.]], dtype=np.float32)

        def make_mask(nose_x):
            kps = base_kps.copy()
            kps[2, 0] = nose_x
            face = SimpleNamespace(bbox=bbox, kps=kps)
            return face, face_masking.create_bbox_safety_mask(face, frame)

        frontal_face, frontal = make_mask(90.)
        right_rear_face, right_rear = make_mask(68.)
        left_rear_face, left_rear = make_mask(112.)
        self.assertEqual(face_masking._yaw_rear_side(frontal_face), (0, 0.0))
        self.assertEqual(face_masking._yaw_rear_side(right_rear_face)[0], 1)
        self.assertEqual(face_masking._yaw_rear_side(left_rear_face)[0], -1)
        frontal_x = np.flatnonzero(frontal[98] > 128)
        right_rear_x = np.flatnonzero(right_rear[98] > 128)
        left_rear_x = np.flatnonzero(left_rear[98] > 128)
        self.assertEqual(int(right_rear_x[0]), int(frontal_x[0]))
        self.assertLess(int(right_rear_x[-1]), int(frontal_x[-1]) - 4)
        self.assertGreater(int(left_rear_x[0]), int(frontal_x[0]) + 4)
        self.assertEqual(int(left_rear_x[-1]), int(frontal_x[-1]))
        self.assertGreater(int(right_rear[98, 90]), 245)
        self.assertGreater(int(left_rear[98, 90]), 245)

        # A detector refresh may shift every landmark slightly while the
        # smoothed bbox is unchanged. The cheek/temple cutoff must not follow
        # that landmark noise.
        jittered_kps = base_kps.copy()
        jittered_kps[:, 0] += 2.0
        jittered_kps[2, 0] = 70.0
        jittered = face_masking.create_bbox_safety_mask(
            SimpleNamespace(bbox=bbox, kps=jittered_kps), frame
        )
        jittered_x = np.flatnonzero(jittered[98] > 128)
        self.assertEqual(int(jittered_x[-1]), int(right_rear_x[-1]))

    def test_profile_cleanup_removes_only_rear_temple_color_stripe(self):
        generated = np.full((128, 128, 3), (105, 142, 178), dtype=np.uint8)
        generated[22:94, 29:33] = (220, 245, 210)
        generated[22:94, 95:99] = (220, 245, 210)
        cleaned = face_swapper._clean_lateral_hair_artifacts(
            generated, rear_side=1
        )
        skin = np.array([105, 142, 178], dtype=np.float32)
        right_before = np.mean(np.abs(
            generated[30:86, 96].astype(np.float32) - skin
        ))
        right_after = np.mean(np.abs(
            cleaned[30:86, 96].astype(np.float32) - skin
        ))
        self.assertLess(right_after, right_before * 0.35)
        # The visible/front side is byte-identical.
        np.testing.assert_array_equal(cleaned[:, :64], generated[:, :64])

    def test_profile_temple_exclusion_is_local_mirrored_and_yaw_only(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        bbox = np.array([40., 30., 140., 155.], dtype=np.float32)

        def exclusion(nose_x):
            return face_masking.create_profile_temple_exclusion_mask(
                SimpleNamespace(
                    bbox=bbox,
                    kps=np.array([
                        [65., 75.], [115., 75.], [nose_x, 92.],
                        [75., 118.], [105., 118.],
                    ], dtype=np.float32),
                    landmark_2d_106=None,
                    landmark_2d_68=None,
                ),
                frame,
            )

        frontal = exclusion(90.)
        rear_right = exclusion(68.)
        rear_left = exclusion(112.)
        self.assertEqual(int(np.max(frontal)), 0)
        self.assertGreater(int(rear_right[75, 135]), 245)
        self.assertEqual(int(rear_right[75, 45]), 0)
        self.assertGreater(int(rear_left[75, 45]), 245)
        self.assertEqual(int(rear_left[75, 135]), 0)
        # Central identity features never enter the camera-only patch.
        self.assertEqual(int(rear_right[92, 90]), 0)
        self.assertEqual(int(rear_left[92, 90]), 0)

    def test_profile_nostril_restore_is_tiny_mirrored_and_yaw_only(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        bbox = np.array([40., 30., 140., 155.], dtype=np.float32)

        def restore_mask(nose_x):
            face = SimpleNamespace(
                bbox=bbox,
                kps=np.array([
                    [65., 75.], [115., 75.], [nose_x, 92.],
                    [75., 118.], [105., 118.],
                ], dtype=np.float32),
            )
            return face_masking.create_profile_nostril_restore_mask(
                face, frame
            )

        frontal = restore_mask(90.)
        facing_left = restore_mask(68.)
        facing_right = restore_mask(112.)
        self.assertEqual(int(np.max(frontal)), 0)
        self.assertGreater(int(facing_left[96, 71]), 245)
        self.assertGreater(int(facing_right[96, 109]), 245)
        # The nose tip and central face remain generated; only the nostril is
        # restored from the camera.
        self.assertLess(int(facing_left[92, 68]), 128)
        self.assertEqual(int(facing_left[96, 90]), 0)
        self.assertEqual(int(facing_right[96, 90]), 0)

    def test_eyebrow_guard_does_not_protect_fake_temple_stroke(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        face = SimpleNamespace(
            bbox=np.array([40., 30., 140., 155.], dtype=np.float32),
            kps=np.array([[65., 75.], [115., 75.], [90., 92.],
                          [75., 118.], [105., 118.]], dtype=np.float32),
            landmark_2d_106=None,
            landmark_2d_68=None,
        )
        guard = face_masking.create_eyebrow_protection_mask(face, frame)
        self.assertGreater(int(guard[64, 76]), 128)  # eyebrow tail
        self.assertLess(int(guard[64, 85]), 16)      # adjacent temple stroke

    def test_parser_excludes_hair_without_punching_uncertain_face_holes(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

            def run(self, _outputs, _feed):
                logits = np.zeros((1, 19, 512, 512), dtype=np.float32)
                logits[:, 17, :256] = 10  # hair
                logits[:, 1, 256:] = 10   # skin
                logits[:, 1, 360:390, 240:270] = 0  # uncertain/background-labelled forehead
                return [logits]

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        face = SimpleNamespace(bbox=np.array([25, 25, 103, 103]))
        with patch.object(face_parser, "_get_session", return_value=Session()):
            parsed = face_parser.parse_face_skin(frame, face)
        self.assertIsNotNone(parsed)
        roi, skin = parsed
        self.assertEqual(int(skin[10, skin.shape[1] // 2]), 0)
        self.assertEqual(int(skin[-10, skin.shape[1] // 2]), 255)
        guard = face_parser.skin_guard_for_crop(parsed, 0, 0, 128, 128)
        self.assertEqual(int(guard[0, 0]), 255)
        self.assertEqual(int(guard[100, 64]), 255)
        self.assertEqual(int(skin[83, 55]), 255)

    def test_swap_paste_never_changes_parsed_hair_pixels(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        fake = np.full_like(frame, 240)
        skin = np.zeros((128, 128), dtype=np.uint8)
        skin[64:] = 255
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        result = face_swapper._fast_paste_back(
            frame, fake, fake, identity,
            parsed_skin=((0, 0, 128, 128), skin),
        )
        self.assertEqual(int(result[40, 64, 0]), 0)
        self.assertGreater(int(result[80, 64, 0]), 0)

    def test_large_yaw_does_not_disable_swap(self):
        turned = SimpleNamespace(kps=np.array(
            [[30, 40], [70, 40], [110, 60], [37, 80], [63, 80]],
            dtype=np.float32,
        ))
        class Swapper:
            input_size = (128, 128)

            def get(self, *_args, **_kwargs):
                return np.full((128, 128, 3), 200, dtype=np.uint8), np.array(
                    [[1., 0., 0.], [0., 1., 0.]], dtype=np.float32
                )

        target = SimpleNamespace(kps=turned.kps, bbox=np.array([8, 8, 120, 120]))
        source = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
        with patch.object(face_swapper, "get_face_swapper", return_value=Swapper()), \
             patch.object(face_swapper, "parse_face_skin", return_value=None), \
             patch.object(settings, "poisson_blend", False), \
             patch.object(settings, "mouth_mask", False), \
             patch.object(settings, "opacity", 1.0):
            result = face_swapper.swap_face(source, target, np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertGreater(int(result[64, 64, 0]), 0)

    def test_swap_mask_does_not_remove_fixed_lateral_strips(self):
        face_swapper._paste_cache["alpha_key"] = None
        with patch.object(settings, "mask_erosion", 0), \
             patch.object(settings, "mask_blur", 1.5):
            mask = face_swapper._get_soft_alpha(128).copy()
        self.assertGreater(int(mask[64, 16]), 0)
        self.assertGreater(int(mask[64, 111]), 0)
        self.assertGreater(int(mask[70, 64]), 245)

    def test_source_identity_is_extracted_from_original_portrait(self):
        image = np.full((120, 120, 3), 90, dtype=np.uint8)
        image[30:90, 20:30] = 245
        source = SimpleNamespace(
            bbox=np.array([20, 20, 100, 105]),
            kps=np.array([[43, 48], [77, 48], [60, 64],
                          [49, 80], [71, 80]], dtype=np.float32),
            normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        original = image.copy()
        with patch.object(face_analyser, "get_one_face", return_value=source) as analyse:
            self.assertIs(face_analyser.get_source_face(image), source)
        analyse.assert_called_once_with(image)
        np.testing.assert_array_equal(image, original)

    def test_temporal_landmarks_reduce_jitter_without_dropping_after_gap(self):
        smoother = face_masking.TemporalLandmarkSmoother()

        def face(dx):
            return SimpleNamespace(kps=np.array([[30, 40], [70, 40], [50, 60],
                                                 [37, 80], [63, 80]], dtype=np.float32)
                                   + np.array([dx, 0], dtype=np.float32))

        smoother.update(face(0), 1.0)
        slight = smoother.update(face(2), 1.1)
        self.assertGreater(float(slight.kps[0, 0]), 30.0)
        self.assertLess(float(slight.kps[0, 0]), 32.0)
        resumed = smoother.update(face(10), 2.0)
        self.assertGreater(float(resumed.kps[0, 0]), float(slight.kps[0, 0]))
        self.assertLessEqual(float(resumed.kps[0, 0]), 40.0)

    def test_temporal_landmarks_accept_coherent_distance_change(self):
        smoother = face_masking.TemporalLandmarkSmoother()
        points = np.array([[30, 40], [70, 40], [50, 60],
                           [37, 80], [63, 80]], dtype=np.float32)
        first = SimpleNamespace(kps=points.copy(),
                                bbox=np.array([20, 20, 80, 90], dtype=np.float32))
        smoother.update(first, 1.0)
        center = points.mean(axis=0)
        zoomed_points = (points - center) * 1.35 + center
        zoomed = SimpleNamespace(kps=zoomed_points,
                                 bbox=np.array([10, 10, 90, 105], dtype=np.float32))
        result = smoother.update(zoomed, 1.1)
        original_eye_width = np.linalg.norm(points[1] - points[0])
        new_eye_width = np.linalg.norm(result.kps[1] - result.kps[0])
        self.assertGreater(new_eye_width, original_eye_width * 1.10)

    def test_optical_flow_keeps_last_face_transform_during_detector_miss(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        face = SimpleNamespace(kps=points.copy(), bbox=np.array([20, 20, 80, 90], dtype=np.float32))
        moved = (points + np.array([3, -2], dtype=np.float32)).reshape(-1, 1, 2)
        gray = np.zeros((100, 100), dtype=np.uint8)
        with patch.object(face_masking.cv2, "calcOpticalFlowPyrLK",
                          return_value=(moved, np.ones((5, 1), dtype=np.uint8), None)):
            self.assertTrue(face_masking.track_face_landmarks(gray, gray, face))
        np.testing.assert_allclose(face.kps, points + [3, -2])
        np.testing.assert_allclose(face.bbox, [23, 18, 83, 88])

    def test_large_valid_turn_is_accepted_against_tracked_face(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        smoother = face_masking.TemporalLandmarkSmoother()
        first = SimpleNamespace(kps=points.copy(), bbox=np.array([20, 20, 80, 90], dtype=np.float32))
        smoother.update(first, 1.0)
        tracked = SimpleNamespace(kps=points + [2, 0], bbox=np.array([22, 20, 82, 90], dtype=np.float32))
        bad = SimpleNamespace(kps=points + [90, 0], bbox=np.array([110, 20, 170, 90], dtype=np.float32))
        result = smoother.update(bad, 1.1, reference_face=tracked)
        self.assertGreater(float(result.kps[0, 0]), float(tracked.kps[0, 0]) + 50.0)
        self.assertGreater(float(result.bbox[0]), 70.0)

    def test_detector_refresh_anchors_bbox_to_current_optical_flow(self):
        points = np.array([[30, 40], [70, 40], [50, 60],
                           [37, 80], [63, 80]], dtype=np.float32)
        smoother = face_masking.TemporalLandmarkSmoother()
        smoother.update(SimpleNamespace(
            kps=points.copy(), bbox=np.array([20, 20, 80, 90], dtype=np.float32)
        ), 1.0)
        tracked = SimpleNamespace(
            kps=points + [24, -6],
            bbox=np.array([44, 14, 104, 84], dtype=np.float32),
        )
        detected = SimpleNamespace(
            kps=points + [26, -5],
            bbox=np.array([46, 15, 106, 85], dtype=np.float32),
        )
        result = smoother.update(
            detected, 1.1, reference_face=tracked
        )
        # Refresh stays between the current tracked and detected boxes. It
        # must never jump back toward the detector box cached before motion.
        self.assertGreaterEqual(float(result.bbox[0]), 44.0)
        self.assertLessEqual(float(result.bbox[0]), 46.0)
        self.assertGreaterEqual(float(result.bbox[1]), 14.0)
        self.assertLessEqual(float(result.bbox[1]), 15.0)

    def test_single_landmark_outlier_is_replaced_without_dropping_face(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        smoother = face_masking.TemporalLandmarkSmoother()
        smoother.update(SimpleNamespace(kps=points.copy(), bbox=np.array([20, 20, 80, 90])), 1.0)
        tracked = SimpleNamespace(kps=points + [1, 0], bbox=np.array([21, 20, 81, 90]))
        detected = points + [2, 0]
        detected[2] += [80, 0]
        result = smoother.update(
            SimpleNamespace(kps=detected, bbox=np.array([22, 20, 82, 90])),
            1.1, reference_face=tracked,
        )
        self.assertLess(float(result.kps[2, 0]), float(tracked.kps[2, 0]) + 10.0)
        self.assertGreater(float(result.kps[0, 0]), float(tracked.kps[0, 0]))

    def test_valid_detection_reacquires_immediately(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        base_bbox = np.array([20, 20, 80, 90], dtype=np.float32)
        smoother = face_masking.TemporalLandmarkSmoother()
        smoother.update(SimpleNamespace(kps=points.copy(), bbox=base_bbox.copy()), 1.0)
        tracked = SimpleNamespace(kps=points.copy(), bbox=base_bbox.copy())
        outputs = []
        for i in range(3):
            candidate = SimpleNamespace(kps=points + [80, 0], bbox=base_bbox + [80, 0, 80, 0])
            outputs.append(smoother.update(candidate, 1.1 + 0.1 * i, reference_face=tracked).kps[0, 0])
        self.assertGreater(float(outputs[0]), 80.0)
        self.assertTrue(all(value > 80.0 for value in outputs))

    def test_optical_flow_rejects_one_morphing_landmark(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        moved = points + [3, 0]
        moved[2] += [80, 0]
        face = SimpleNamespace(kps=points.copy(), bbox=np.array([20, 20, 80, 90], dtype=np.float32))
        gray = np.zeros((100, 100), dtype=np.uint8)
        with patch.object(face_masking.cv2, "calcOpticalFlowPyrLK",
                          return_value=(moved.reshape(-1, 1, 2),
                                        np.ones((5, 1), dtype=np.uint8), None)):
            self.assertTrue(face_masking.track_face_landmarks(gray, gray, face))
        np.testing.assert_allclose(face.kps[2], points[2] + [3, 0], atol=1.0)

    def test_optical_flow_accepts_fast_whole_face_motion(self):
        points = np.array([[30, 40], [70, 40], [50, 60], [37, 80], [63, 80]], dtype=np.float32)
        face = SimpleNamespace(kps=points.copy(), bbox=np.array([20, 20, 80, 90], dtype=np.float32))
        gray = np.zeros((100, 100), dtype=np.uint8)
        with patch.object(face_masking.cv2, "calcOpticalFlowPyrLK",
                          return_value=((points + [80, 0]).reshape(-1, 1, 2),
                                        np.ones((5, 1), dtype=np.uint8), None)):
            self.assertTrue(face_masking.track_face_landmarks(gray, gray, face))
        self.assertAlmostEqual(float(face.kps[0, 0] - points[0, 0]), 80.0)

    def test_region_flow_carries_face_when_five_landmarks_fail(self):
        points = np.array([[30, 40], [70, 40], [50, 60],
                           [37, 80], [63, 80]], dtype=np.float32)
        face = SimpleNamespace(
            kps=points.copy(),
            bbox=np.array([20, 20, 80, 92], dtype=np.float32),
            landmark_2d_106=None,
        )
        features = np.array(
            [[[30., 35.]], [[45., 45.]], [[60., 55.]], [[35., 70.]],
             [[65., 75.]]], dtype=np.float32,
        )
        moved = features + np.array([[[6., -3.]]], dtype=np.float32)
        gray = np.zeros((110, 110), dtype=np.uint8)
        with patch.object(face_masking.cv2, "goodFeaturesToTrack",
                          return_value=features), \
             patch.object(face_masking.cv2, "calcOpticalFlowPyrLK",
                          side_effect=[(None, None, None),
                                       (moved, np.ones((5, 1), dtype=np.uint8), None)]):
            self.assertTrue(face_masking.track_face_landmarks(gray, gray, face))
        np.testing.assert_allclose(face.kps, points + [6, -3])

    def test_pose_affine_accepts_pitch_foreshortening_without_exploding(self):
        points = np.array([[30, 40], [70, 40], [50, 60],
                           [37, 80], [63, 80]], dtype=np.float32)
        expected = points.copy()
        expected[:, 0] = expected[:, 0] * 1.05 + 8.0
        expected[:, 1] = expected[:, 1] * 0.72 + 18.0
        transform = face_masking._estimate_pose_affine(points, expected)
        self.assertIsNotNone(transform)
        predicted = cv2.transform(points[None], transform)[0]
        np.testing.assert_allclose(predicted, expected, atol=1.0)

    def test_gpen_256_feature_guard_keeps_eye_and_mouth_separate(self):
        kps = np.array([[80, 85], [170, 85], [125, 130], [95, 175], [155, 175]], dtype=np.float32)
        landmarks = np.zeros((106, 2), dtype=np.float32)
        landmarks[33:43] = np.array([80, 85])
        landmarks[87:97] = np.array([170, 85])
        landmarks[52:64] = np.array([125, 175])
        face = SimpleNamespace(kps=kps, landmark_2d_106=landmarks)
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        with patch.object(settings, "mouth_mask", True):
            guarded = face_masking.aligned_feature_guard(face, identity, 256)
        self.assertGreater(int(guarded[85, 80]), 0)
        self.assertGreater(int(guarded[175, 125]), 0)
        self.assertEqual(int(guarded[130, 125]), 0)

    def test_gpen_guard_preserves_profile_nose_without_camera_cutout(self):
        face = SimpleNamespace(
            kps=np.array([[80., 85.], [170., 85.], [68., 130.],
                          [95., 175.], [155., 175.]], dtype=np.float32),
            landmark_2d_106=None,
            landmark_2d_68=None,
        )
        identity = np.array(
            [[1., 0., 0.], [0., 1., 0.]], dtype=np.float32
        )
        guard = face_masking.aligned_feature_guard(face, identity, 256)
        self.assertGreater(int(guard[130, 68]), 200)
        self.assertEqual(int(guard[150, 125]), 0)

    def test_aligned_guard_protects_raised_eyebrows(self):
        landmarks = np.zeros((106, 2), dtype=np.float32)
        landmarks[33:43] = np.array([80, 100])
        landmarks[87:97] = np.array([170, 100])
        landmarks[43:52] = np.array([80, 64])
        landmarks[97:106] = np.array([170, 64])
        face = SimpleNamespace(
            landmark_2d_106=landmarks,
            landmark_2d_68=None,
            kps=np.array([[80, 100], [170, 100], [125, 130],
                          [95, 175], [155, 175]], dtype=np.float32),
        )
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        protected = face_masking.aligned_brow_eye_guard(face, identity, 256)
        enhancer_guard = face_masking.aligned_feature_guard(face, identity, 256)
        self.assertGreater(int(protected[64, 80]), 0)
        self.assertGreater(int(enhancer_guard[64, 170]), 0)
        self.assertGreater(int(enhancer_guard[83, 125]), 0)

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

    def test_live_gpen256_runs_every_frame_and_skips_failed_result(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

        session = Session()
        calls = []
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        frame = np.full((256, 256, 3), 100, dtype=np.uint8)

        def inference(*_args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("transient model failure")
            return np.zeros((1, 3, 256, 256), dtype=np.float32)

        enhancer._LIVE_ENHANCER_CACHE.pop(id(session), None)
        with patch.object(settings, "enhancer_interval", 3), \
             patch.object(settings, "detail_strength", 0.0), \
             patch.object(settings, "film_grain_strength", 0.0), \
             patch.object(settings, "color_match", False), \
             patch.object(enhancer, "_get_face_affine", return_value=(identity, identity)), \
             patch.object(enhancer, "parse_face_skin", return_value=None), \
             patch.object(enhancer, "run_inference", side_effect=inference):
            values = [int(enhancer.enhance_face_onnx(frame.copy(), object(), session, 256, live=True)[128, 128, 0])
                      for _ in range(3)]
        self.assertEqual(len(calls), 3)
        self.assertEqual(values[1], 100)
        self.assertEqual(values[0], values[2])
        self.assertNotEqual(values[0], values[1])

    def test_live_restoration_rejects_white_side_hallucination(self):
        camera = np.full((128, 128, 3), 120, dtype=np.uint8)
        restored = camera.copy()
        restored[20:105, :28] = 255
        restored[20:105, 100:] = 255
        self.assertFalse(enhancer._restoration_is_plausible(camera, restored))

    def test_live_restoration_accepts_natural_detail_change(self):
        yy, xx = np.mgrid[0:128, 0:128]
        base = 105 + 18 * np.sin(xx / 8.0) + 12 * np.cos(yy / 11.0)
        camera = np.repeat(
            np.clip(base, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
        )
        restored = cv2.GaussianBlur(camera, (0, 0), 0.7)
        self.assertTrue(enhancer._restoration_is_plausible(camera, restored))

    def test_closeup_enhancer_increases_high_frequency_detail_only(self):
        checker = ((np.indices((128, 128)).sum(axis=0) % 2) * 80 + 80)
        camera = np.repeat(checker.astype(np.uint8)[:, :, None], 3, axis=2)
        softened = cv2.GaussianBlur(camera, (0, 0), 3.0)
        inverse = np.array([[1.8, 0., 0.], [0., 1.8, 0.]], dtype=np.float32)
        strength = enhancer._closeup_detail_strength(
            0.58, inverse, live=True
        )
        self.assertGreater(strength, 0.80)
        closeup = enhancer.blend_high_frequency(camera, softened, strength)
        self.assertGreater(
            float(cv2.Laplacian(closeup, cv2.CV_32F).var()),
            float(cv2.Laplacian(softened, cv2.CV_32F).var()) + 20.0,
        )
        normal_strength = enhancer._closeup_detail_strength(
            0.58,
            np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32),
            live=True,
        )
        self.assertAlmostEqual(normal_strength, 0.58)

    def test_live_restoration_rejects_severe_smear(self):
        checker = ((np.indices((128, 128)).sum(axis=0) % 2) * 120 + 60)
        camera = np.repeat(checker.astype(np.uint8)[:, :, None], 3, axis=2)
        restored = np.full_like(camera, 120)
        self.assertFalse(enhancer._restoration_is_plausible(camera, restored))

    def test_live_restoration_rejects_abrupt_identity_structure_change(self):
        previous = np.full((128, 128, 3), 40, dtype=np.uint8)
        previous[:, 64:] = 215
        restored = np.full((128, 128, 3), 215, dtype=np.uint8)
        restored[:, 64:] = 40
        camera = restored.copy()
        self.assertFalse(enhancer._restoration_is_plausible(
            camera, restored, previous_restored=previous
        ))

    def test_gpen_empty_alpha_returns_input_frame(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

        session = Session()
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        frame = np.full((256, 256, 3), 83, dtype=np.uint8)
        output = np.zeros((1, 3, 256, 256), dtype=np.float32)
        with patch.object(settings, "detail_strength", 0.0), \
             patch.object(settings, "film_grain_strength", 0.0), \
             patch.object(enhancer, "_get_face_affine", return_value=(identity, identity)), \
             patch.object(enhancer, "get_enhancer_crop_mask",
                          return_value=np.zeros((256, 256), dtype=np.uint8)), \
             patch.object(enhancer, "run_inference", return_value=output):
            result = enhancer.enhance_face_onnx(
                frame.copy(), object(), session, 256, live=True
            )
        np.testing.assert_array_equal(result, frame)

    def test_gpen_paste_is_clipped_to_stable_bbox_mask(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

        session = Session()
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        output = np.ones((1, 3, 256, 256), dtype=np.float32)
        stable_guard = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(stable_guard, (128, 132), (72, 100), 0, 0, 360, 255, -1)
        face = SimpleNamespace(
            landmark_2d_106=np.full((106, 2), [128., 128.], dtype=np.float32),
            landmark_2d_68=None,
            bbox=np.array([24., 20., 232., 244.], dtype=np.float32),
        )
        with patch.object(settings, "detail_strength", 0.0), \
             patch.object(settings, "film_grain_strength", 0.0), \
             patch.object(settings, "color_match", False), \
             patch.object(enhancer, "_get_face_affine",
                          return_value=(identity, identity)), \
             patch.object(enhancer, "get_enhancer_crop_mask",
                          return_value=np.full((256, 256), 255,
                                               dtype=np.uint8)) as fallback_mask, \
             patch.object(enhancer, "aligned_feature_guard", return_value=None), \
             patch.object(enhancer, "create_bbox_safety_mask",
                          return_value=stable_guard) as build_mask, \
             patch.object(enhancer, "parse_face_skin", return_value=None), \
             patch.object(enhancer, "run_inference", return_value=output):
            input_frame = frame.copy()
            result = enhancer.enhance_face_onnx(
                input_frame, face, session, 256, live=False
            )
        build_mask.assert_called_once()
        called_face, called_frame = build_mask.call_args.args
        self.assertIs(called_face, face)
        self.assertIs(called_frame, input_frame)
        fallback_mask.assert_called_once_with(256)
        self.assertEqual(int(result[132, 128, 0]), 255)
        self.assertEqual(int(result[20, 20, 0]), 0)

    def test_swapper_empty_alpha_returns_input_frame(self):
        frame = np.full((64, 64, 3), 91, dtype=np.uint8)
        fake = np.full((32, 32, 3), 200, dtype=np.uint8)
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        with patch.object(face_swapper, "_get_soft_alpha",
                          return_value=np.zeros((32, 32), dtype=np.uint8)):
            result = face_swapper._fast_paste_back(
                frame.copy(), fake, fake, identity
            )
        np.testing.assert_array_equal(result, frame)

    def test_disjoint_parser_roi_cannot_disable_face_swap(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fake = np.full((32, 32, 3), 180, dtype=np.uint8)
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        face = SimpleNamespace(
            bbox=np.array([0., 0., 32., 32.], dtype=np.float32),
            landmark_2d_106=None,
            landmark_2d_68=None,
        )
        parser_result = (
            (50, 50, 60, 60), np.full((10, 10), 255, dtype=np.uint8)
        )
        with patch.object(face_swapper, "_get_soft_alpha",
                          return_value=np.full((32, 32), 255,
                                               dtype=np.uint8)), \
             patch.object(face_swapper, "create_bbox_safety_mask",
                          return_value=np.full((64, 64), 255,
                                               dtype=np.uint8)):
            result = face_swapper._fast_paste_back(
                frame, fake, fake, identity, target_face=face,
                parsed_skin=parser_result,
            )
        self.assertEqual(int(result[16, 16, 0]), 180)

    def test_poisson_guard_returns_unmodified_alpha_frame_when_unsafe(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:54, 10:54] = 255
        swapped = np.full((64, 64, 3), 230, dtype=np.uint8)
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        with patch.object(face_swapper.cv2, "seamlessClone") as clone:
            result = face_swapper._safe_seamless_clone(
                swapped, original, mask, (32, 32), (10, 10, 54, 54)
            )
        clone.assert_not_called()
        self.assertIs(result, swapped)
        self.assertTrue(np.all(swapped == 230))

        mask[:20, 10:54] = 255
        close_color = np.full_like(original, 228)
        with patch.object(face_swapper.cv2, "seamlessClone") as clone:
            face_swapper._safe_seamless_clone(
                swapped, close_color, mask, (32, 32), (10, 0, 54, 54)
            )
        clone.assert_not_called()

        interior = np.zeros_like(mask)
        interior[10:54, 10:54] = 255
        before = swapped.copy()
        with patch.object(face_swapper.cv2, "seamlessClone", side_effect=cv2.error("clone failed")):
            result = face_swapper._safe_seamless_clone(
                swapped, close_color, interior, (32, 32), (10, 10, 54, 54)
            )
        self.assertIs(result, swapped)
        np.testing.assert_array_equal(swapped, before)

    def test_poisson_does_not_switch_masks_when_affine_path_fails(self):
        swapped = np.full((64, 64, 3), 120, dtype=np.uint8)
        original = np.full_like(swapped, 110)
        target = SimpleNamespace(bbox=np.array([10, 10, 54, 54]))
        affine = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        with patch.object(face_swapper.cv2, "invertAffineTransform", side_effect=cv2.error("bad affine")), \
             patch.object(face_swapper.cv2, "seamlessClone") as clone:
            result = face_swapper._apply_poisson_blend(
                swapped, original, target, affine, np.zeros((32, 32, 3), dtype=np.uint8)
            )
        clone.assert_not_called()
        self.assertIs(result, swapped)

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

        def paste(_frame, _enhanced, matrix, output_size, **_kwargs):
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

    def test_gpen_mask_is_oval_instead_of_crop_box(self):
        enhancer._MASK_CACHE.clear()
        mask = enhancer.get_enhancer_crop_mask(256)
        self.assertEqual(int(np.max(mask[0, :])), 0)
        self.assertEqual(int(np.max(mask[-1, :])), 0)
        self.assertEqual(int(np.max(mask[:, 0])), 0)
        self.assertEqual(int(np.max(mask[:, -1])), 0)
        self.assertEqual(int(mask[16, 16]), 0)
        self.assertEqual(int(mask[132, 128]), 255)
        self.assertGreater(int(mask[16, 128]), 0)
        self.assertTrue(np.any((mask[:, 128] > 0) & (mask[:, 128] < 255)))

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

    def test_skin_tone_second_pass_can_be_kept_conservative(self):
        source = np.full((128, 128, 3), (170, 105, 75), dtype=np.uint8)
        target = np.full((128, 128, 3), (65, 125, 185), dtype=np.uint8)
        mask = np.full((128, 128), 255, dtype=np.uint8)
        full = face_masking.match_skin_tone_lab(
            source, target, mask, strength=1.0
        )
        light = face_masking.match_skin_tone_lab(
            source, target, mask, strength=0.42
        )
        full_change = np.mean(np.abs(
            full.astype(np.float32) - source.astype(np.float32)
        ))
        light_change = np.mean(np.abs(
            light.astype(np.float32) - source.astype(np.float32)
        ))
        self.assertGreater(full_change, light_change * 1.5)


if __name__ == "__main__":
    unittest.main()
