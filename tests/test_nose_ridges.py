import unittest
from unittest.mock import patch
import cv2
import numpy as np
from modules.processors.frame.face_detail import suppress_model_nose_ridges

def landmarks():
    p=np.zeros((106,2),np.float32)
    p[74:87]=[[50,35],[38,45],[35,55],[37,61],[42,64],[47,66],[50,67],[55,66],[60,64],[63,61],[65,55],[62,45],[50,55]]
    return p

class NoseRidgeTests(unittest.TestCase):
    def test_detected_bright_wing_ridge_is_reduced_without_erasing_nostrils(self):
        image=np.full((100,100,3),140,np.uint8)
        cv2.line(image,(39,48),(38,59),(220,220,220),1)
        cv2.circle(image,(41,62),2,(25,25,25),-1)
        out=suppress_model_nose_ridges(image,landmarks())
        self.assertLess(int(out[52,39,0]),int(image[52,39,0]))
        np.testing.assert_array_equal(out[62,41],image[62,41])
        np.testing.assert_array_equal(out[:44],image[:44])
        np.testing.assert_array_equal(out[68:],image[68:])
        np.testing.assert_array_equal(out[:,48:53],image[:,48:53])
        np.testing.assert_array_equal(out[:,:30],image[:,:30])

    def test_flat_skin_and_invalid_geometry_are_unchanged(self):
        image=np.full((100,100,3),140,np.uint8)
        for points in (landmarks(),np.zeros((5,2)),np.full((106,2),np.nan)):
            np.testing.assert_array_equal(suppress_model_nose_ridges(image,points),image)

    def test_camera_nose_edge_is_not_reintroduced_as_skin_texture(self):
        from modules.processors.frame.face_swapper import _restore_camera_microtexture
        generated=np.full((128,128,3),140,np.uint8);camera=generated.copy()
        cv2.line(camera,(52,65),(50,82),(230,230,230),1)
        before=_restore_camera_microtexture(generated,camera,.45)
        after=_restore_camera_microtexture(generated,camera,.45,protect_nose=True)
        self.assertGreater(int(np.max(cv2.absdiff(before,generated))),0)
        np.testing.assert_array_equal(after[65:82,48:55],generated[65:82,48:55])
