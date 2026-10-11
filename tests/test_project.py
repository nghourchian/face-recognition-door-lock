import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from face_lock.antispoof import PassiveModels
from face_lock.config import Config
from face_lock.controller import LockController
from face_lock.event_log import EventLog
from face_lock.profile import load_profile, save_profile
from face_lock.ui import render


class ProjectTests(unittest.TestCase):
    def test_crops_at_frame_edges(self):
        for face in ((0, 0, 80, 100), (550, 380, 80, 100), (280, 180, 80, 100)):
            for scale in (2.7, 4.0):
                left, top, right, bottom = PassiveModels.crop_bounds(
                    (480, 640, 3), face, scale
                )
                self.assertTrue(0 <= left < right <= 640)
                self.assertTrue(0 <= top < bottom <= 480)

    def test_invalid_crop_is_rejected(self):
        with self.assertRaises(ValueError):
            PassiveModels.crop_bounds((480, 640, 3), [0, 0, 0, 10], 2.7)

    def test_profile_round_trip_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "enrolled_face.npz"
            save_profile(path, "Nima", [np.ones(128), np.ones(128) * 2])
            profile = load_profile(path)
            self.assertEqual(profile.name, "Nima")
            self.assertAlmostEqual(float(np.linalg.norm(profile.centroid)), 1, places=5)

    def test_invalid_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, name="Nima", model="wrong", centroid=np.ones(128))
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_configuration_rejects_invalid_values(self):
        for kwargs in (
            {"camera": -1},
            {"threshold": float("nan")},
            {"pad_threshold": 1.1},
        ):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_log_labels_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Config(workspace=Path(directory), label="photo_self"))
            log.write("START_LOCKED", LockController())
            log.close()
            with log.path.open() as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["presentation"], "photo_self")
            self.assertEqual(row["similarity"], "")
            self.assertEqual(row["pad_ready"], "False")

    def test_ui_renders_all_states_without_a_camera(self):
        controller = LockController()
        packet = dict(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            captured=10,
            score=0.8,
            problem=None,
            pad_scores=[0.9, 0.9],
            pad_ok=True,
        )
        for state in ("LOCKED", "UNLOCKED", "WAIT_CLEAR"):
            controller.state, controller.deadline = state, 12
            canvas = render(packet, controller, Config(), "Nima", 10, "", None)
            self.assertEqual(canvas.shape, (620, 1020, 3))
        render(None, controller, Config(), "Nima", 10, "", "Camera unavailable")
