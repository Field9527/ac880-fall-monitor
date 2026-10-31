"""验证像素一致性和异步预览缓冲区所有权。"""

import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future
import numpy as np
import realtime


class DisplayTests(unittest.TestCase):
    def test_cached_bars_preserve_camera_pixels_and_reuse_stable_content(self):
        from display_ui import MonitorHUD

        hud = MonitorHUD()
        raw = np.random.RandomState(4).randint(0, 65536, (720, 1280)).astype(np.uint16)
        frame = raw.copy()
        result = dict(state=0, pe=0.1)
        observation = dict(ready=True)
        environment = dict(available=True, valid=True, temperature_c=25.1, humidity_rh=55.2)
        hud.draw(frame, result, observation, environment)
        header, footer = hud.header, hud.footer
        np.testing.assert_array_equal(frame[64:-84], raw[64:-84])
        hud.draw(frame, result, observation, dict(environment, age_s=2.0))
        self.assertIs(hud.header, header)
        self.assertIs(hud.footer, footer)
        hud.draw(frame, result, observation, dict(environment, temperature_c=25.2))
        self.assertIs(hud.header, header)
        self.assertIsNot(hud.footer, footer)

    def test_alarm_is_not_hidden_by_stale_result_and_replay_is_marked(self):
        from display_ui import MonitorHUD, RED

        hud = MonitorHUD()
        frame = np.zeros((720, 1280), dtype=np.uint16)
        observation = dict(ready=False, test_context="phone_screen_replay")
        self.assertEqual(hud.state(dict(state=3), observation), ("FALL ALERT", RED))
        hud.draw(frame, dict(state=3), observation)
        self.assertIn("PHONE REPLAY", hud.header_key)
        self.assertIn("Gesture paused", hud.footer_key[-1])

    def test_invalid_environment_does_not_show_stale_or_zero_values(self):
        from display_ui import MonitorHUD

        value = dict(available=True, valid=False, stale=True, temperature_c=26, humidity_rh=50)
        info = MonitorHUD.footer_info(value, {}, {})
        self.assertEqual(info[:3], ("--", "--", "Reading expired"))

    def test_hardware_selection_is_visible_and_fall_mode_explains_paused_hand(self):
        from display_ui import MonitorHUD

        hud = MonitorHUD()
        frame = np.zeros((720, 1280), dtype=np.uint16)
        hud.draw(frame, dict(state=3, pe=0.99), dict(ready=True, recognition_mode="gesture"))
        self.assertIn("SW0: GESTURE", hud.header_key)
        self.assertIn("Gesture control", hud.header_key)
        self.assertIsNone(hud.header_key[-2])
        self.assertEqual(frame[70, 0], 0)
        info = hud.footer_info({}, dict(recognition_mode="fall", gesture_enabled=False), {})
        self.assertEqual(info[3:], ("Gesture paused", "SW0 up: gesture mode"))

    def test_previous_command_is_not_reported_as_a_new_gesture_receipt(self):
        from display_ui import MonitorHUD
        from gesture_control import ACTIONS

        observation = dict(hand=dict(name="fist"))
        previous = dict(last_action=ACTIONS["open_palm"], last_action_age_s=0.5)
        palm = MonitorHUD.footer_info({}, dict(hand=dict(name="open_palm")), previous)
        self.assertEqual(palm[3:], ("Open palm", "Command submitted"))
        self.assertEqual(MonitorHUD.footer_info({}, observation, previous)[-1], "Hold steady")
        current = dict(previous, last_action=ACTIONS["fist"])
        self.assertEqual(MonitorHUD.footer_info({}, observation, current)[-1], "Command submitted")

    def test_in_place_annotation_matches_old_copy_bit_for_bit(self):
        raw = np.random.RandomState(3).randint(0, 65536, (720, 1280)).astype(np.uint16)
        original = raw.copy()
        result = dict(state=0, state_name="IDLE", pe=0.123)
        observation = dict(ready=True, bbox=(200, 160, 300, 450), backend="fall_lite")
        old = realtime.annotate(raw, result, observation)
        np.testing.assert_array_equal(raw, original)
        new = realtime.annotate(raw, result, observation, in_place=True)
        self.assertIs(new, raw)
        np.testing.assert_array_equal(new, old)

    def test_preview_has_owned_pixels_and_bounded_queue(self):
        pending = Future()
        executor = Mock()
        executor.submit.return_value = pending
        with patch("realtime.ThreadPoolExecutor", return_value=executor):
            writer = realtime.PreviewWriter()
        frame = np.full((12, 20), 0xABCD, np.uint16)
        self.assertTrue(writer.submit(frame, "unused.jpg"))
        frame[:] = 0
        self.assertFalse(writer.submit(frame, "unused.jpg"))
        captured = executor.submit.call_args[0][1]
        self.assertEqual(captured.shape, (6, 10))
        self.assertTrue(np.all(captured == 0xABCD))
        self.assertEqual(executor.submit.call_count, 1)
        pending.set_result(None)
        writer.close()


if __name__ == "__main__":
    unittest.main()
