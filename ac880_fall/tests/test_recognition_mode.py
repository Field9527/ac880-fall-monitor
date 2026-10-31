"""验证拨码消抖、任务互斥和跨模式旧结果隔离。"""

from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch

from dual_inference import ConcurrentGestureInference
from recognition_mode import RecognitionSwitch
from shared_detector import SemanticOptions


class RecognitionModeTests(unittest.TestCase):
    def test_bounce_pauses_immediately_and_requires_stable_position(self):
        options = SemanticOptions()
        mmio = Mock()
        position = [0]
        mmio.read.side_effect = lambda offset: (
            RecognitionSwitch.MAGIC if offset == 0x110 else position[0]
        )
        switch = RecognitionSwitch(mmio, options)
        switch.poll(0)
        self.assertEqual(options.configuration["recognition_mode"], "paused")
        switch.poll(0.16)
        self.assertEqual(options.configuration["recognition_mode"], "fall")
        position[0] = 1
        switch.poll(1)
        self.assertEqual(options.configuration["recognition_mode"], "paused")
        position[0] = 0
        switch.poll(1.05)
        position[0] = 1
        switch.poll(1.1)
        switch.poll(1.2)
        self.assertEqual(switch.mode, "paused")
        switch.poll(1.26)
        self.assertEqual(switch.mode, "gesture")

    def test_missing_hardware_or_read_failure_cannot_enable_recognition(self):
        options, mmio = SemanticOptions(), Mock()
        switch = RecognitionSwitch(mmio, options)
        mmio.read.return_value = 0
        switch.poll(1)
        self.assertEqual(switch.mode, "paused")
        self.assertFalse(options.snapshot()["recognition_switch"]["available"])
        mmio.read.side_effect = OSError("read failed")
        switch.poll(2)
        self.assertEqual(switch.mode, "paused")

    def test_queued_web_settings_cannot_override_physical_selection(self):
        options = SemanticOptions()
        queued = options.configuration
        options.select_recognition("gesture", dict(available=True))
        generation = options.configuration["generation"]
        options.configure(queued)
        self.assertEqual(options.configuration["recognition_mode"], "gesture")
        self.assertEqual(options.configuration["generation"], generation)
        with self.assertRaises(ValueError):
            options.validate(dict(recognition_mode="fall"))

    def worker(self, mode):
        detector = Mock()
        detector.semantic_options = SemanticOptions()
        detector.semantic_options.select_recognition(mode, dict(available=True))
        detector.hand_interval_s = 0.5
        detector.gestures_enabled.return_value = True
        fall_pool, hand_pool = Mock(), Mock()
        fall_pool.submit.side_effect = lambda *args: Future()
        hand_pool.submit.side_effect = lambda *args: Future()
        with patch("fall_lite.ThreadPoolExecutor", return_value=fall_pool), patch(
            "dual_inference.ThreadPoolExecutor", return_value=hand_pool
        ):
            worker = ConcurrentGestureInference(detector)
        return worker, detector.semantic_options, fall_pool, hand_pool

    def test_gesture_mode_never_runs_fall_and_fall_mode_never_runs_hand(self):
        for mode in ("gesture", "fall", "paused"):
            worker, _, fall_pool, hand_pool = self.worker(mode)
            source = Mock(return_value="owned frame")
            try:
                worker.poll(source, 10)
                self.assertEqual(fall_pool.submit.call_count, int(mode == "fall"))
                self.assertEqual(hand_pool.submit.call_count, int(mode == "gesture"))
                self.assertEqual(source.call_count, int(mode != "paused"))
            finally:
                worker.close()

    def test_mode_switch_drains_old_task_and_discards_its_result(self):
        for old_mode, new_mode in (("fall", "gesture"), ("gesture", "fall")):
            worker, options, fall_pool, hand_pool = self.worker(old_mode)
            try:
                worker.poll("owned frame", 10)
                pending = worker.future if old_mode == "fall" else worker.hand_future
                options.select_recognition(new_mode, dict(available=True))
                observation, fresh = worker.poll("new frame", 10.1)
                self.assertFalse(fresh)
                self.assertIsNone(observation["bbox"])
                self.assertEqual(fall_pool.submit.call_count + hand_pool.submit.call_count, 1)
                pending.set_result(dict(ready=True, bbox=[1, 1, 20, 20], score=0.99))
                observation, fresh = worker.poll("new frame", 10.6)
                self.assertFalse(fresh)
                self.assertIsNone(observation["bbox"])
                self.assertIsNone(observation["hand"])
                self.assertEqual(fall_pool.submit.call_count + hand_pool.submit.call_count, 2)
            finally:
                worker.close()

    def test_round_trip_switch_does_not_reuse_old_fall_evidence(self):
        worker, options, _, _ = self.worker("fall")
        try:
            worker.poll("old frame", 10)
            pending = worker.future
            options.select_recognition("gesture", {})
            options.select_recognition("fall", {})
            pending.set_result(dict(ready=True, bbox=[1, 1, 20, 20], score=0.99))
            observation, fresh = worker.poll("new frame", 11)
            self.assertFalse(fresh)
            self.assertIsNone(observation["bbox"])
            self.assertEqual(worker.fall_completed, 0)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
