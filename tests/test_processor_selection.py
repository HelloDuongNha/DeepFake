import unittest
from types import SimpleNamespace
from unittest.mock import patch

import modules.globals as settings
from modules.processors.frame import core
from modules.ui import MainWindow


class ProcessorSelectionTests(unittest.TestCase):
    def test_live_camera_ignores_stale_face_mapper_setting(self):
        camera = SimpleNamespace(currentText=lambda: "Camera 1")
        window = SimpleNamespace(cb_camera=camera)
        with patch.object(settings, "map_faces", True), \
             patch.object(settings, "source_path", "source.png"), \
             patch("modules.ui.selected_camera_index", return_value=1), \
             patch("modules.face_analyser.get_face_analyser"), \
             patch("modules.processors.frame.face_swapper.get_face_swapper"), \
             patch("modules.ui._open_webcam_preview") as preview, \
             patch("modules.ui._open_live_mapper_dialog") as mapper:
            MainWindow._on_live(window)
            preview.assert_called_once_with(1)
            mapper.assert_not_called()
            self.assertFalse(settings.map_faces)

    def test_enhancer_change_publishes_single_choice_without_mutating_active_flags(self):
        previous = {
            "face_enhancer": False,
            "face_enhancer_gpen256": True,
            "face_enhancer_gpen512": False,
        }
        with patch.object(settings, "fp_ui", previous), patch("modules.ui.save_switch_states"):
            MainWindow._on_enhancer_change(None, "GPEN-512")
            self.assertTrue(previous["face_enhancer_gpen256"])
            self.assertEqual([k for k, v in settings.fp_ui.items() if v],
                             ["face_enhancer_gpen512"])

    def test_in_flight_processor_snapshot_survives_enhancer_switch(self):
        names = ["face_swapper", "face_enhancer_gpen256", "face_enhancer_gpen512"]
        modules = {n: SimpleNamespace(__name__="modules.processors.frame." + n)
                   for n in names}
        processors = ["face_swapper"]
        with patch.object(core, "FRAME_PROCESSORS_MODULES", []), \
             patch.object(core, "load_frame_processor_module", side_effect=modules.get), \
             patch.object(settings, "frame_processors", processors), \
             patch.object(settings, "fp_ui", {"face_enhancer_gpen256": True,
                                               "face_enhancer_gpen512": False}):
            first = core.get_frame_processors_modules(processors)
            settings.fp_ui = {"face_enhancer_gpen256": False, "face_enhancer_gpen512": True}
            second = core.get_frame_processors_modules(processors)
            self.assertEqual(first, [modules["face_swapper"], modules["face_enhancer_gpen256"]])
            self.assertEqual(second, [modules["face_swapper"], modules["face_enhancer_gpen512"]])
            self.assertIsNot(first, core.FRAME_PROCESSORS_MODULES)
