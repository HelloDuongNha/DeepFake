import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from modules import user_settings as prefs

def defaults():
    fields={k:True for k in prefs.BOOL_FIELDS}
    return SimpleNamespace(**fields,opacity=1.,sharpness=.65,fp_ui={},swapper_model='hyperswap_256',
        source_path=None,source_history=[],preferred_camera_index=None,window_geometry='',live_window_geometry='')

class PreferenceTests(unittest.TestCase):
    def test_full_roundtrip_and_history_dedup(self):
        with tempfile.TemporaryDirectory() as folder,patch('modules.paths.SETTINGS_PATH',str(Path(folder)/'settings.json')):
            image=Path(folder)/'tên ảnh.png';image.write_bytes(b'test')
            state=defaults();state.source_path=str(image);state.source_history=[str(image),str(image)]
            state.opacity=.37;state.sharpness=1.48;state.fp_ui={'face_enhancer_gpen256':True}
            state.preferred_camera_index=3;state.window_geometry='YWJj'
            self.assertTrue(prefs.save(state))
            restored=defaults();prefs.apply(prefs.read(),restored)
            self.assertEqual(restored.source_path,str(image));self.assertEqual(restored.source_history,[str(image)])
            self.assertEqual(restored.opacity,.37);self.assertEqual(restored.sharpness,1.48)
            self.assertTrue(restored.fp_ui['face_enhancer_gpen256']);self.assertEqual(restored.preferred_camera_index,3)
            self.assertEqual(restored.window_geometry,'YWJj')

    def test_corrupt_types_and_missing_file_are_safe(self):
        with tempfile.TemporaryDirectory() as folder,patch('modules.paths.SETTINGS_PATH',str(Path(folder)/'settings.json')):
            path=Path(folder)/'settings.json'
            for data in ('bad json','[]','null'):
                path.write_text(data);self.assertEqual(prefs.read(),{})
            state=defaults();prefs.apply({'opacity':float('nan'),'sharpness':'99','mask_enabled':'false','fp_ui':[],
                'source_path':str(Path(folder)/'missing.jpg'),'source_history':[None,[],123,'/missing.jpg']},state)
            self.assertEqual(state.opacity,1.);self.assertEqual(state.sharpness,.65);self.assertTrue(state.mask_enabled)
            self.assertIsNone(state.source_path);self.assertEqual(state.source_history,['/missing.jpg'])

    def test_failed_atomic_write_preserves_old_settings(self):
        with tempfile.TemporaryDirectory() as folder,patch('modules.paths.SETTINGS_PATH',str(Path(folder)/'settings.json')):
            path=Path(folder)/'settings.json';path.write_text('{"old":true}')
            with patch('modules.user_settings.os.replace',side_effect=OSError('disk full')):
                self.assertFalse(prefs.save(defaults()))
            self.assertEqual(json.loads(path.read_text()),{'old':True})
            self.assertEqual(len(list(Path(folder).iterdir())),1)
