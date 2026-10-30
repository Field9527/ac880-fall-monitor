import unittest
from unittest.mock import patch
import numpy as np
from semantic_features import region_descriptor, FeatureRegions, rotate_xyxy
from shared_detector import SemanticOptions
from tracked_fall import TrackedFallDetector
from fall_detect import FallState


class SharedRuntimeContracts(unittest.TestCase):
    def test_fast_roi_pool_matches_training_descriptor(self):
        rng = np.random.RandomState(5)
        for channels in (64, 96):
            features = [
                rng.normal(0, 10, (channels, h, w)).astype(np.float32)
                for h, w in ((24, 40), (12, 20), (6, 10))
            ]
            cache = FeatureRegions(features)
            for box in (
                [0, 0, 320, 192],
                [1, 2, 4, 5],
                [79.5, 23.1, 200.3, 99.7],
                [-10, -2, 25, 33],
            ):
                np.testing.assert_allclose(
                    cache.descriptor(box), region_descriptor(features, box), rtol=1e-5, atol=2e-5
                )

    def test_rotation_box_round_trip(self):
        original = [20, 30, 70, 80]
        for k in (1, 2, 3):
            rotated = rotate_xyxy(original, 640, 360, k)
            w, h = (360, 640) if k % 2 else (640, 360)
            np.testing.assert_allclose(rotate_xyxy(rotated, w, h, -k), original)

    def test_only_explicit_phone_mode_can_rotate_input(self):
        control = SemanticOptions()
        value = control.validate(dict(rotation=90))
        self.assertEqual(value["rotation"], 0)
        control.configure(dict(mode="phone_replay", rotation=90))
        self.assertEqual(control.configuration["generation"], 1)
        control.configure(control.configuration)
        self.assertEqual(control.configuration["generation"], 1)
        control.configure(dict(show_objects=False))
        self.assertEqual(control.configuration["generation"], 1)
        control.configure(dict(mode="live"))
        self.assertEqual(control.configuration["rotation"], 0)
        with self.assertRaises(ValueError):
            control.validate(dict(rotation=45))

    def test_different_targets_cannot_combine_trigger_counts(self):
        cfg = dict(
            ema_alpha=1,
            alarm_thr=0.7,
            score_th=0.8,
            quiet_frames=1,
            candidate_min_s=0,
            armed_min_s=0,
            trig_n=2,
        )
        tracker = TrackedFallDetector(cfg)
        with patch("fall_detect.time.monotonic", return_value=10):
            tracker.step_many([dict(bbox=[0, 0, 100, 50], score=0.95)], 1, 10, 10)
            result = tracker.step_many([dict(bbox=[500, 300, 100, 50], score=0.95)], 2, 10, 10)
        self.assertEqual(result["state"], FallState.IDLE)
        self.assertEqual(len(tracker.tracks), 2)

    def test_duplicate_or_stale_results_never_confirm_alarm(self):
        cfg = dict(
            ema_alpha=1,
            alarm_thr=0.7,
            score_th=0.8,
            quiet_frames=1,
            candidate_min_s=0,
            armed_min_s=0,
            trig_n=2,
        )
        tracker = TrackedFallDetector(cfg)
        d = [dict(bbox=[0, 0, 100, 50], score=0.95)]
        with patch("fall_detect.time.monotonic", return_value=10):
            tracker.step_many(d, 1, 10, 10)
            for _ in range(6):
                self.assertEqual(tracker.step_many(d, 1, 10, 10)["evidence_status"], "duplicate")
            self.assertEqual(tracker.step_many(d, 2, 1, 10)["evidence_status"], "stale")
        self.assertEqual(tracker._result()["state"], FallState.IDLE)

    def test_already_fallen_target_can_confirm_without_standing(self):
        cfg = dict(
            ema_alpha=1,
            alarm_thr=0.7,
            score_th=0.8,
            quiet_frames=1,
            candidate_min_s=0,
            armed_min_s=0,
            trig_n=1,
        )
        tracker = TrackedFallDetector(cfg)
        events = []
        tracker.on_event = events.append
        with patch("fall_detect.time.monotonic", return_value=10):
            for i in range(4):
                result = tracker.step_many([dict(bbox=[10, 10, 120, 40], score=0.95)], i, 10, 10)
        self.assertEqual(result["state"], FallState.ALARM)
        self.assertEqual(events[-1]["track_id"], 1)
        self.assertEqual(len(set(v["event_id"] for v in events)), 1)


if __name__ == "__main__":
    unittest.main()
