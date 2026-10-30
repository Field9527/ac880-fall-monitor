import unittest
import numpy as np
from gesture_control import GestureController, classify
from shared_detector import SemanticOptions


class GestureEvidenceTests(unittest.TestCase):
    def hand(self, name="open_palm", x=10):
        return dict(name=name, score=0.99, bbox=[x, 20, 100, 100])

    def test_one_fresh_sample_and_duplicate_frames_cannot_trigger(self):
        c = GestureController()
        self.assertIsNone(c.step(self.hand(), 1, 10, 10.2))
        for now in (10.5, 11, 12):
            self.assertIsNone(c.step(self.hand(), 1, 10, now))
        self.assertEqual(c.step(self.hand(), 2, 12, 12.2), {"led_mask": 255})

    def test_half_second_configuration_still_requires_two_fresh_stable_inputs(self):
        c = GestureController(hold_s=0.5)
        self.assertIsNone(c.step(self.hand(), 1, 10, 10))
        self.assertIsNone(c.step(self.hand(), 1, 10, 10.5))
        self.assertIsNone(c.step(self.hand(), 2, 10.4, 10.4))
        self.assertEqual(c.step(self.hand(), 3, 10.5, 10.5), {"led_mask": 255})
        self.assertIsNone(c.step(self.hand(), 4, 11, 11))

    def test_holding_v_is_one_bounded_pulse(self):
        c = GestureController()
        h = self.hand("victory")
        c.step(h, 1, 10, 10)
        self.assertEqual(c.step(h, 2, 11, 11), {"buzzer": {"on": True, "duration_ms": 500}})
        for i in range(3, 15):
            self.assertIsNone(c.step(h, i, 10 + i, 10 + i))
        c.step(None, 20, 30, 30)
        c.step(None, 21, 31, 31)
        c.step(h, 22, 32, 32)
        self.assertIsNotNone(c.step(h, 23, 33, 33))

    def test_confirmed_led_change_has_no_unrelated_buzzer_cooldown(self):
        c = GestureController()
        c.step(self.hand(), 1, 10, 10)
        self.assertEqual(c.step(self.hand(), 2, 10.9, 10.9), {"led_mask": 255})
        c.step(self.hand("fist"), 3, 11, 11)
        self.assertEqual(c.step(self.hand("fist"), 4, 11.9, 11.9), {"led_mask": 0})
        self.assertIsNone(c.step(self.hand("fist"), 5, 12, 12))

    def test_led_action_does_not_restart_v_pulse_interval(self):
        c = GestureController()
        c.step(self.hand(), 1, 10, 10)
        c.step(self.hand(), 2, 10.9, 10.9)
        c.step(self.hand("victory"), 3, 11, 11)
        self.assertEqual(
            c.step(self.hand("victory"), 4, 11.9, 11.9),
            {"buzzer": {"on": True, "duration_ms": 500}},
        )

    def test_removed_thumb_gesture_cannot_send_a_command(self):
        c = GestureController()
        h = self.hand("thumb_up")
        for i in range(5):
            self.assertIsNone(c.step(h, i, 10 + i, 10 + i))
        self.assertIsNone(c.last_action)

    def test_upright_thumb_is_unknown_not_a_fist(self):
        p = np.zeros((21, 3), dtype=float)
        p[0, :2] = [100, 200]
        for mcp, x in ((5, 70), (9, 100), (13, 120), (17, 140)):
            p[mcp : mcp + 4, :2] = [[x, 150], [x, 130], [x, 140], [x, 170]]
        p[1:5, :2] = [[80, 180], [65, 160], [55, 135], [40, 110]]
        self.assertEqual(classify(p), "unknown")
        p[1:5, :2] = [[80, 180], [70, 160], [85, 160], [100, 160]]
        self.assertEqual(classify(p), "fist")

    def test_v_with_crossed_or_extended_thumb_is_not_discarded(self):
        p = np.zeros((21, 3), dtype=float)
        p[0, :2] = [100, 200]
        for mcp, x in ((5, 70), (9, 110)):
            p[mcp : mcp + 4, :2] = [[x, 150], [x, 110], [x, 80], [x, 50]]
        for mcp, x in ((13, 130), (17, 150)):
            p[mcp : mcp + 4, :2] = [[x, 150], [x, 130], [x, 140], [x, 170]]
        p[1:5, :2] = [[80, 180], [65, 160], [55, 135], [40, 110]]
        self.assertEqual(classify(p), "victory")
        p[12] = p[8] + [5, 0, 0]
        self.assertEqual(classify(p), "unknown")

    def test_alternating_fall_job_does_not_erase_hand_confirmation(self):
        c = GestureController()
        h = self.hand()
        c.step(h, 1, 10, 11.8)
        # 下一次手势任务运行期间，旧叠加结果仍会过期。
        self.assertIsNone(c.step(None, 1, 10, 14))
        self.assertEqual(c.step(h, 2, 14, 15.8), {"led_mask": 255})

    def test_stale_disabled_unknown_and_different_hands_do_not_combine(self):
        for second in (
            dict(now=15),
            dict(enabled=False),
            dict(hand=self.hand("unknown")),
            dict(hand=self.hand(x=500)),
        ):
            c = GestureController()
            c.step(self.hand(), 1, 10, 10)
            args = dict(hand=self.hand(), sample_id=2, sampled_at=11, now=11, enabled=True)
            args.update(second)
            self.assertIsNone(c.step(**args))

    def test_enable_epoch_changes_without_resetting_fall_monitor(self):
        options = SemanticOptions()
        options.configure(dict(gesture_enabled=False))
        self.assertEqual(options.configuration["generation"], 0)
        self.assertEqual(options.configuration["gesture_generation"], 1)
        options.configure(options.configuration)
        self.assertEqual(options.configuration["gesture_generation"], 1)
        with self.assertRaises(ValueError):
            options.validate(dict(gesture_enabled=1))


if __name__ == "__main__":
    unittest.main()
