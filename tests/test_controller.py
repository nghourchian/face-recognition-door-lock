import math
import unittest

from face_lock.controller import LockController, PassiveGate


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.lock = LockController()

    def observe(self, now, score=0.8, count=1, usable=True):
        return self.lock.observe(now, now, count, score, usable)

    def unlock(self):
        for t in (0, 0.25, 0.51):
            self.observe(t)
        self.assertEqual(self.lock.state, "UNLOCKED")

    def test_match_requires_duration_and_multiple_observations(self):
        self.observe(0)
        self.observe(0.51)
        self.assertEqual(self.lock.state, "LOCKED")
        self.observe(0.6)
        self.assertEqual(self.lock.state, "UNLOCKED")

    def test_mismatch_clears_partial_confirmation(self):
        self.observe(0)
        self.observe(0.3, score=0.1)
        self.observe(0.6)
        self.observe(0.9)
        self.assertEqual(self.lock.state, "LOCKED")
        self.observe(1.11)
        self.assertEqual(self.lock.state, "UNLOCKED")

    def test_relocks_then_requires_empty_frame(self):
        self.unlock()
        for t in (1, 1.5, 2, 2.5, 3, 3.5):
            self.observe(t)
        self.lock.tick(3.52)
        self.assertEqual(self.lock.state, "WAIT_CLEAR")
        for t in (3.6, 4, 4.5):
            self.observe(t)
        self.assertEqual(self.lock.unlocks, 1)
        for t in (4.6, 5.1, 5.61):
            self.observe(t, None, 0, False)
        self.assertEqual(self.lock.state, "LOCKED")
        for t in (5.7, 5.95, 6.21):
            self.observe(t)
        self.assertEqual(self.lock.unlocks, 2)

    def test_invalid_observations_do_not_unlock(self):
        for score, count, usable in (
            (0.2, 1, True),
            (0.9, 2, True),
            (None, 0, False),
            (0.9, 1, False),
            (math.nan, 1, True),
        ):
            with self.subTest(score=score, count=count, usable=usable):
                self.lock = LockController()
                for t in (0, 0.3, 0.6, 0.9):
                    self.observe(t, score, count, usable)
                self.assertEqual(self.lock.state, "LOCKED")

    def test_camera_stall_locks_and_requires_departure(self):
        self.unlock()
        self.lock.tick(1.52)
        self.assertTrue(self.lock.faulted)
        self.assertEqual(self.lock.state, "WAIT_CLEAR")
        self.observe(1.6)
        self.assertFalse(self.lock.faulted)
        self.assertEqual(self.lock.state, "WAIT_CLEAR")

    def test_stale_and_future_timestamps_fail_closed(self):
        for now, captured in ((2, 0), (0, 1)):
            self.lock = LockController()
            self.lock.observe(now, captured, 1, 0.9, True)
            self.assertTrue(self.lock.faulted)

    def test_duplicate_frames_cannot_accumulate_hits(self):
        self.observe(0)
        self.lock.observe(0.3, 0, 1, 0.9, True)
        self.lock.observe(0.6, 0, 1, 0.9, True)
        self.assertEqual(self.lock.hits, 1)


class GateTests(unittest.TestCase):
    def test_pass_requires_five_samples_over_one_second(self):
        gate = PassiveGate()
        for t in (0, 0.25, 0.5, 0.75):
            self.assertFalse(gate.update(t, True, [0.9, 0.9]))
        self.assertTrue(gate.update(1, True, [0.9, 0.9]))

    def test_bad_score_identity_loss_and_gap_reset(self):
        for identity, scores in (
            (False, [0.9, 0.9]),
            (True, [0.2, 0.9]),
            (True, [0.9, 0.2]),
            (True, None),
            (True, [math.nan, 0.9]),
        ):
            gate = PassiveGate()
            for i in range(6):
                gate.update(i * 0.2, True, [0.9, 0.9])
            self.assertFalse(gate.update(1.1, identity, scores))
            self.assertFalse(gate.update(1.2, True, [0.9, 0.9]))
            self.assertFalse(gate.update(2.2, True, [0.9, 0.9]))

    def test_identity_and_both_pad_models_are_required(self):
        for scores, expected in (([0.95, 0.95], 1), ([0.95, 0.2], 0), ([0.2, 0.95], 0)):
            gate, lock = PassiveGate(), LockController()
            for i in range(21):
                t = i * 0.1
                lock.observe(t, t, 1, 0.8, gate.update(t, True, scores))
            self.assertEqual(lock.unlocks, expected)
