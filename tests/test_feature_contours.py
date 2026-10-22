from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from modules.processors.frame.face_detail import preserve_eye_apertures,repair_nose_edges

class FeatureContoursTests(unittest.TestCase):
    def test_eyelid_rim_replaces_double_line_and_keeps_iris_current(self):
        p=np.zeros((106,2),np.float32)
        eye=np.array([[30,40],[34,37],[40,36],[46,37],[50,40],[46,43],[40,44],[34,43],[40,40],[41,40]],np.float32)
        p[33:43]=eye;p[87:97]=eye+[40,0];face=SimpleNamespace(landmark_2d_106=p)
        camera=np.full((100,120,3),160,np.uint8);camera[38,40]=35;camera[40,40]=15
        generated=np.full_like(camera,160);generated[35,40]=30
        out=preserve_eye_apertures(generated.copy(),camera,face,repair_eyelids=True)
        self.assertEqual(int(out[40,40,0]),15)
        self.assertGreater(int(out[35,40,0]),130)
        np.testing.assert_array_equal(out[60:],generated[60:])
        np.testing.assert_array_equal(out[:25],generated[:25])
        camera[40,40]=80
        out2=preserve_eye_apertures(generated.copy(),camera,face,repair_eyelids=True)
        self.assertEqual(int(out2[40,40,0]),80)

    def test_nose_repair_is_local_and_does_not_touch_bridge_or_mouth(self):
        p=np.zeros((106,2),np.float32)
        p[74:87]=[[50,35],[38,45],[35,55],[37,61],[42,64],[47,66],[50,67],[55,66],[60,64],[63,61],[65,55],[62,45],[50,55]]
        generated=np.full((100,100,3),140,np.uint8);camera=generated.copy();camera[61,39]=20
        out=repair_nose_edges(generated.copy(),camera,SimpleNamespace(landmark_2d_106=p))
        self.assertLess(int(out[61,39,0]),70)
        np.testing.assert_array_equal(out[:40],generated[:40]);np.testing.assert_array_equal(out[75:],generated[75:])
        np.testing.assert_array_equal(out[:,:25],generated[:,:25])
        for points in (None,np.full((106,2),np.nan),np.zeros((106,2))):
            np.testing.assert_array_equal(repair_nose_edges(generated.copy(),camera,SimpleNamespace(landmark_2d_106=points)),generated)
