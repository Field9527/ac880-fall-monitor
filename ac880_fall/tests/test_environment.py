"""验证环境读数有效性、读取限频和 HDMI 缓存绘制。"""

import unittest
from unittest.mock import patch
import numpy as np
from environment import EnvironmentMonitor, EnvironmentHUD, decode_payload


class Registers:
    def __init__(self):
        self.values = {
            0x120: 0x454E5631,
            0x124: 13,
            0x128: 0x37001A03,
            0x12C: 1,
            0x130: 100,
            0x134: 0,
            0x138: 1,
        }
        self.reads = 0

    def read(self, offset):
        self.reads += 1
        return self.values[offset]


class EnvironmentTests(unittest.TestCase):
    def test_real_dht11_payload_format_and_negative_temperature(self):
        self.assertEqual(decode_payload(0x37001A03), (26.3, 55.0))
        self.assertEqual(decode_payload(0x35000A81), (-10.1, 53.0))
        for bad in (0x65001A00, 0x370A1A00, 0x37001A0A):
            with self.assertRaises(ValueError):
                decode_payload(bad)

    def test_startup_does_not_publish_fake_zero_measurements(self):
        mmio = Registers()
        mmio.values.update({0x124: 8, 0x128: 0, 0x12C: 0, 0x130: 0xFFFFFFFF, 0x138: 0})
        value = EnvironmentMonitor(mmio, dict(enabled=True)).poll(1)
        self.assertFalse(value["valid"])
        self.assertIsNone(value["temperature_c"])
        self.assertIsNone(value["humidity_rh"])

    def test_periodic_reads_are_bounded(self):
        mmio = Registers()
        monitor = EnvironmentMonitor(mmio, dict(enabled=True))
        value = monitor.poll(1)
        reads = mmio.reads
        self.assertTrue(value["valid"])
        self.assertEqual(value["temperature_c"], 26.3)
        monitor.poll(1.1)
        monitor.poll(1.4)
        self.assertEqual(mmio.reads, reads)
        monitor.poll(1.6)
        self.assertGreater(mmio.reads, reads)

    def test_failures_keep_last_good_with_notice_then_hide_stale_values(self):
        mmio = Registers()
        monitor = EnvironmentMonitor(mmio, dict(enabled=True))
        mmio.values[0x124] = (4 << 8) | 9
        value = monitor.poll(1)
        self.assertTrue(value["valid"])
        self.assertFalse(value["last_read_ok"])
        self.assertIn("校验失败", value["message"])
        self.assertIn("最近有效值", value["message"])
        mmio.values[0x130] = 16000
        value = monitor.poll(2)
        self.assertFalse(value["valid"])
        self.assertTrue(value["stale"])
        self.assertIsNone(value["temperature_c"])
        self.assertIsNone(value["humidity_rh"])

    def test_absent_hardware_and_disabled_config_are_compatible(self):
        mmio = Registers()
        mmio.values[0x120] = 0
        monitor = EnvironmentMonitor(mmio, dict(enabled=True))
        self.assertFalse(monitor.poll()["available"])
        EnvironmentMonitor(None, {}).poll()

    def test_hud_reuses_strip_and_does_not_change_camera_below_it(self):
        import cv2

        frame = np.full((720, 1280), 0x7777, np.uint16)
        hud = EnvironmentHUD()
        value = EnvironmentMonitor(Registers(), dict(enabled=True)).poll(1)
        with patch("cv2.putText", wraps=cv2.putText) as draw:
            hud.draw(frame, value)
            hud.draw(frame, dict(value, age_s=1.1))
            self.assertEqual(draw.call_count, 1)
        self.assertTrue(np.all(frame[98:] == 0x7777))
        self.assertTrue(np.all(frame[:54] == 0x7777))
        self.assertTrue(np.any(frame[54:98] == 0))


if __name__ == "__main__":
    unittest.main()
