import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from modules.processors.frame.upper_face import preserve_upper_face


def fixture(yaw=.32):
    p=np.zeros((106,2),np.float32)
    brow=np.array([[52,82],[60,78],[72,76],[83,78],[91,83],[84,85],[72,83],[61,85],[55,84]],np.float32)
    p[43:52]=brow;p[97:106]=brow+[80,0]
    eye=np.array([[54,105],[60,101],[70,100],[81,101],[87,105],[81,109],[70,110],[60,109],[70,105],[72,105]],np.float32)
    p[33:43]=eye;p[87:97]=eye+[80,0]
    kps=np.array([[70,105],[150,105],[110+yaw*80,150],[87,201],[147,201]],np.float32)
    return SimpleNamespace(kps=kps,bbox=np.array([20,15,220,255],np.float32),landmark_2d_106=p)


class UpperFaceTests(unittest.TestCase):
    def test_turn_removes_forehead_seam_without_touching_eyes_nose_or_mouth(self):
        face=fixture();generated=np.full((280,250,3),160,np.uint8)
        generated[45:48,50:200]=[60,70,80]
        camera=np.full_like(generated,120)
        out=preserve_upper_face(generated.copy(),camera,face)
        np.testing.assert_array_equal(out[46,110],camera[46,110])
        np.testing.assert_array_equal(out[80,110],camera[80,110])
        np.testing.assert_array_equal(out[100:],generated[100:])
        np.testing.assert_array_equal(out[105,70],generated[105,70])

    def test_frontal_and_invalid_geometry_do_not_change_output(self):
        generated=np.random.default_rng(2).integers(0,256,(280,250,3),np.uint8)
        camera=np.full_like(generated,120)
        for yaw in (0.,.05,.09):
            np.testing.assert_array_equal(preserve_upper_face(generated.copy(),camera,fixture(yaw)),generated)
        for points in (None,np.zeros((5,2)),np.full((106,2),np.nan),np.zeros((106,2))):
            f=fixture();f.landmark_2d_106=points
            np.testing.assert_array_equal(preserve_upper_face(generated.copy(),camera,f),generated)

    def test_roll_alone_is_not_a_turn_and_current_camera_is_used(self):
        f=fixture(0.);origin=f.kps[:2].mean(0)
        rotation=cv2.getRotationMatrix2D(tuple(origin),30,1)
        f.kps=cv2.transform(f.kps[None],rotation)[0]
        f.landmark_2d_106=cv2.transform(f.landmark_2d_106[None],rotation)[0]
        generated=np.full((280,250,3),160,np.uint8);camera=np.full_like(generated,120)
        np.testing.assert_array_equal(preserve_upper_face(generated.copy(),camera,f),generated)
        f=fixture();a=preserve_upper_face(generated.copy(),camera,f)
        camera[:]=90;b=preserve_upper_face(generated.copy(),camera,f)
        self.assertEqual(int(a[46,110,0]),120);self.assertEqual(int(b[46,110,0]),90)

    def test_strength_transition_is_continuous(self):
        generated=np.full((280,250,3),160,np.uint8);camera=np.full_like(generated,120)
        pixels=[int(preserve_upper_face(generated.copy(),camera,fixture(y))[46,110,0]) for y in np.linspace(.085,.165,81)]
        self.assertEqual(pixels[0],160);self.assertEqual(pixels[-1],120)
        self.assertTrue(all(a>=b for a,b in zip(pixels,pixels[1:])))
        self.assertLessEqual(max(abs(a-b) for a,b in zip(pixels,pixels[1:])),2)
