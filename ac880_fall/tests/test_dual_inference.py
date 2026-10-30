"""验证模型独立完成时仍保留证据时间和图像所有权。"""

from concurrent.futures import Future
from unittest.mock import Mock, patch
import unittest
import numpy as np
from dual_inference import ConcurrentGestureInference
from shared_detector import SemanticOptions


class DualInferenceTests(unittest.TestCase):
    def make_worker(self):
        detector = Mock()
        detector.semantic_options = SemanticOptions()
        detector.hand_interval_s = 0.5
        detector.gestures_enabled.return_value = True
        fall_pool = Mock()
        hand_pool = Mock()
        fall = Future()
        hand = Future()
        fall_pool.submit.side_effect = [fall, Future()]
        hand_pool.submit.side_effect = [hand, Future()]
        with patch("fall_lite.ThreadPoolExecutor", return_value=fall_pool), patch(
            "dual_inference.ThreadPoolExecutor", return_value=hand_pool
        ):
            worker = ConcurrentGestureInference(detector)
        return worker, detector, fall_pool, hand_pool, fall, hand

    def test_hand_completes_without_waiting_for_fall_and_never_reuses_dma_view(self):
        worker, _, fall_pool, hand_pool, fall, hand = self.make_worker()
        dma = np.ones((8, 12), np.uint16)
        source = Mock(side_effect=lambda: dma.copy())
        try:
            worker.poll(source, 10)
            self.assertEqual(source.call_count, 1)
            self.assertIs(fall_pool.submit.call_args.args[1], hand_pool.submit.call_args.args[1])
            dma.fill(0)
            self.assertTrue(fall_pool.submit.call_args.args[1].all())
            hand.set_result(
                dict(
                    hand=dict(name="fist"),
                    hand_sequence=1,
                    hand_sampled_at=10,
                    gesture_generation=0,
                    hand_semantic_generation=0,
                )
            )
            observation, fresh = worker.poll(source, 10.6)
            self.assertEqual(observation["hand"]["name"], "fist")
            self.assertFalse(fresh)
            self.assertEqual(worker.fall_completed, 0)
            self.assertEqual(worker.hand_completed, 1)
        finally:
            worker.close()

    def test_pending_models_do_not_copy_or_queue_another_frame(self):
        worker, _, fall_pool, hand_pool, fall, hand = self.make_worker()
        source = Mock(return_value=np.ones((4, 4), np.uint16))
        try:
            worker.poll(source, 10)
            for now in (10.1, 10.2, 10.3):
                worker.poll(source, now)
            self.assertEqual(source.call_count, 1)
            self.assertEqual(fall_pool.submit.call_count, 1)
            self.assertEqual(hand_pool.submit.call_count, 1)
        finally:
            worker.close()

    def test_old_hand_completion_after_mode_change_cannot_issue_control(self):
        worker, detector, _, _, fall, hand = self.make_worker()
        pixels = np.ones((4, 4), np.uint16)
        try:
            worker.poll(pixels, 10)
            detector.semantic_options.configure(dict(gesture_enabled=False))
            detector.gestures_enabled.return_value = False
            hand.set_result(
                dict(
                    hand=dict(name="victory"),
                    hand_sequence=1,
                    hand_sampled_at=10,
                    gesture_generation=0,
                    hand_semantic_generation=0,
                )
            )
            observation, fresh = worker.poll(pixels, 10.8)
            self.assertIsNone(observation["hand"])
            self.assertFalse(fresh)
        finally:
            worker.close()

    def test_fall_timestamp_is_independent_of_new_hand_sample(self):
        worker, _, _, _, fall, hand = self.make_worker()
        pixels = np.ones((4, 4), np.uint16)
        try:
            worker.poll(pixels, 10)
            fall.set_result(
                dict(ready=True, source_monotonic=10, fall_fresh=True, semantic_generation=0)
            )
            hand.set_result(
                dict(
                    hand=None,
                    hand_sequence=1,
                    hand_sampled_at=10.5,
                    gesture_generation=0,
                    hand_semantic_generation=0,
                )
            )
            observation, fresh = worker.poll(pixels, 11)
            self.assertTrue(fresh)
            self.assertEqual(observation["source_monotonic"], 10)
            self.assertEqual(observation["hand_sampled_at"], 10.5)
            self.assertEqual(worker.fall_completed, 1)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
