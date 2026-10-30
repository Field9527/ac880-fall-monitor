"""离线回归：DMA 协议、CNN ABI、倒地持续、负例和判决计时。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fall_detect as fd
from pose_rules import PoseRules, ForegroundDetector


class FakeMMIO:
    def __init__(self):
        self.reg = {fd.VDMA_MM2S_SR: 1, fd.VDMA_S2MM_SR: 1}
        self.writes = []

    def read(self, off):
        return self.reg.get(off, 0)

    def write(self, off, value):
        self.writes.append((off, value))
        self.reg[off] = value
        if off in (0, 0x30):
            if value & fd.VDMA_CR_RESET:
                self.reg[off] = 0
                self.reg[off + 4] = 1
            else:
                self.reg[off + 4] = 0 if value & 1 else 1


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf8"))

    def test_pl_clock_uses_linux_reference_and_rejects_frequency_mismatch(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            bit = Path(directory) / "clock.bit"
            bit.with_suffix(".hwh").write_text(
                '<ROOT><PARAMETER NAME="PCW_CRYSTAL_PERIPHERAL_FREQMHZ" VALUE="33.333333"/>'
                '<PORT NAME="fclk_clk0" CLKFREQUENCY="100000000"/></ROOT>'
            )
            overlay = SimpleNamespace(bitfile_name=str(bit))

            class ClockMMIO:
                def __init__(self):
                    self.control = 0x200500

                def read(self, offset):
                    return {0x170: self.control, 0x108: 0x1E000}[offset]

                def write(self, *args):
                    raise AssertionError("Clock audit must never change registers")

            mmio = ClockMMIO()
            summary = " ps_clk 3 3 0 33333333 0 0 50000\n fclk0 0 0 0 99999999 0 0 50000\n"
            report = fd.overlay_clock_evidence(overlay, mmio, summary)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["fclk0_mhz"], 100.0, places=4)
            mmio.control = 0x200300
            with self.assertRaisesRegex(RuntimeError, "Actual FCLK0"):
                fd.overlay_clock_evidence(overlay, mmio, summary)
            with self.assertRaisesRegex(RuntimeError, "reference disagrees"):
                fd.overlay_clock_evidence(overlay, mmio, summary.replace("33333333", "50000000"))

    def test_dma_circular_vsize_last_and_pointer_fields(self):
        mmio = FakeMMIO()
        dma = fd.VDMAManager(mmio, 1280, 720)

        class Buffer:
            def __init__(self, pa):
                self.physical_address = pa

        dma.frames = [Buffer(0x100000 + i * 0x200000) for i in range(3)]
        self.assertTrue(dma.start())
        for cr, vs in ((0, 0x50), (0x30, 0xA0)):
            vs_index = next(i for i, (off, value) in enumerate(mmio.writes) if off == vs)
            self.assertEqual(mmio.writes[vs_index - 1], (cr, 3))
        mmio.reg[0x28] = (2 << 24) | (1 << 16) | (1 << 8)
        self.assertEqual(dma.frame_indices(), (1, 2))
        self.assertTrue(dma.is_running(fd.VDMA_S2MM_SR))
        mmio.reg[0x34] = 1 << 12  # 正常帧中断不能被当成错误
        self.assertFalse(dma.s2mm_has_error())
        mmio.reg[0x34] = 1 << 15
        self.assertTrue(dma.s2mm_has_error())

    def test_dma_free_requires_stopped(self):
        mmio = FakeMMIO()
        dma = fd.VDMAManager(mmio, 1280, 720)

        class Buffer:
            def freebuffer(self):
                raise AssertionError("仍运行时不能释放")

        dma.frames = [Buffer()]
        with patch.object(dma, "stop", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                dma.free()

    def test_direct_display_preserves_pixels_and_buffer_ownership(self):
        class Buffer(np.ndarray):
            def invalidate(self):
                pass

            def flush(self):
                pass

        mmio = FakeMMIO()
        dma = fd.VDMAManager(mmio, 20, 12)
        raw = np.arange(240, dtype=np.uint16).reshape(12, 20)
        dma.frames = [raw.copy().view(Buffer) for _ in range(3)]
        dma.display_frames = [np.full_like(raw, 65535).view(Buffer) for _ in range(3)]
        # 显示读取帧 0，帧 1 待发布，相机正在写帧 1。
        mmio.reg[fd.VDMA_PARK_PTR] = (1 << 24) | 1
        target, image = dma.prepare_display_frame()
        self.assertEqual(target, 2)
        np.testing.assert_array_equal(image, raw)
        self.assertTrue(np.all(dma.display_frames[0] == 65535))
        self.assertTrue(np.all(dma.display_frames[1] == 65535))
        image[:2] = 0
        np.testing.assert_array_equal(dma.frames[0], raw)
        dma.commit_display_frame(target)
        self.assertEqual(mmio.reg[fd.VDMA_PARK_PTR] & 31, 2)
        with self.assertRaises(RuntimeError):
            dma.commit_display_frame(target)

    def test_direct_display_rejects_overwrite_and_long_copy(self):
        class Buffer(np.ndarray):
            def invalidate(self):
                pass

        dma = fd.VDMAManager(FakeMMIO(), 20, 12)
        dma.frames = [np.zeros((12, 20), np.uint16).view(Buffer) for _ in range(3)]
        dma.display_frames = [np.zeros((12, 20), np.uint16) for _ in range(3)]
        for following, elapsed, valid in ((1, 0.020, True), (2, 0.020, False), (0, 0.051, False)):
            with patch.object(
                dma, "frame_indices", side_effect=[(0, 0), (0, following)] * 2
            ), patch.object(fd.time, "monotonic", side_effect=[0, elapsed, 1, 1 + elapsed]):
                target, image = dma.prepare_display_frame()
            self.assertEqual(image is not None, valid)
            self.assertFalse(dma.mmio.writes)

    def test_cnn_uses_hls_ap_ctrl_and_true_threshold(self):
        class CNNMMIO(FakeMMIO):
            def read(self, off):
                if off == 0:
                    return 2 if self.reg.get(0) == 1 else 4
                if off == 0x10:
                    return 101
                return super().read(off)

        mmio = CNNMMIO()
        cnn = fd.CNNAccelIF(mmio)
        cnn.set_threshold(89)
        self.assertEqual(cnn.infer(), 101)
        self.assertIn((0x40, 89), mmio.writes)
        self.assertIn((0x20, 1), mmio.writes)
        self.assertIn((0, 1), mmio.writes)
        with self.assertRaises(NotImplementedError):
            cnn.load_weights("dummy.bin")

    def test_missing_camera_heartbeat_is_not_reported_as_ready(self):
        mmio = FakeMMIO()
        mmio.reg[fd.MOTION_REGS["VERSION"]] = 0x10000
        md = fd.MotionDetectIF(mmio)
        with self.assertRaises(TimeoutError):
            md.wait_frames(timeout=0.001)

    def test_fall_and_persistent_lying(self):
        rule = PoseRules(self.cfg["rules"])
        for i in range(10):
            self.assertLess(rule.update((200, 80, 100, 400), i * 0.1), 0.9)
        self.assertGreater(rule.update((190, 380, 380, 130), 1.0), 0.9)
        # 站立历史过期后，静止倒地仍保持可疑，不依赖逐帧运动。
        self.assertGreater(rule.update((190, 380, 380, 130), 10.0), 0.9)
        self.assertLess(rule.update((200, 80, 100, 400), 10.1), 0.9)

    def test_sitting_walking_and_initial_lying_are_negative(self):
        rule = PoseRules(self.cfg["rules"])
        self.assertLess(rule.update((200, 380, 380, 130), 0), 0.9)
        for i in range(20):
            self.assertLess(rule.update((100 + i * 5, 100, 100, 350), 1 + i * 0.1), 0.9)
        self.assertLess(rule.update((200, 280, 160, 200), 3), 0.9)

    def test_foreground_pipeline_synthetic_sequence(self):
        cfg = dict(self.cfg["rules"], warmup_s=0.5, processing_width=320)
        foreground = ForegroundDetector(cfg)
        empty = np.zeros((180, 320), np.uint16)
        for i in range(7):
            foreground.process(empty, i * 0.1)
        stand = empty.copy()
        stand[25:150, 130:160] = 0xFFFF
        for i in range(10):
            obs = foreground.process(stand, 1 + i * 0.1)
            self.assertLess(obs["score"], 0.9)
        lying = empty.copy()
        lying[125:160, 100:225] = 0xFFFF
        obs = foreground.process(lying, 2.1)
        self.assertGreater(obs["score"], 0.9)
        self.assertGreater(foreground.process(lying, 8)["score"], 0.9)
        self.assertLess(foreground.process(empty, 9)["score"], 0.9)
        foreground.reset()
        self.assertFalse(foreground.process(empty, 10)["ready"])
        self.assertFalse(foreground.pose.triggered)

    def test_fsm_alarm_and_recovery_with_monotonic_clock(self):
        det = fd.FallDetector(self.cfg["decision"])
        with patch.object(fd.time, "monotonic") as clock:
            states = []
            for i in range(50):
                clock.return_value = i * 0.1
                states.append(det.step(True, 0.95)["state"])
            self.assertIn(fd.FallState.ALARM, states)
            kinds = [ev["event"] for ev in det.events]
            self.assertEqual(kinds[:3], ["candidate_start", "alarm_armed", "alarm"])
            for i in range(50, 90):
                clock.return_value = i * 0.1
                det.step(False, 0)
            self.assertEqual(det.state, fd.FallState.IDLE)

    def test_short_spikes_do_not_accumulate_confirmation_time(self):
        cfg = dict(self.cfg["decision"], ema_alpha=1, trig_n=1)
        det = fd.FallDetector(cfg)
        with patch.object(fd.time, "monotonic") as clock:
            for i in range(100):
                clock.return_value = i * 0.1
                det.step(True, 0.95 if i % 3 else 0.65)
                self.assertNotEqual(det.state, fd.FallState.ALARM)

    def test_event_has_json_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = fd.EventLogger(str(Path(tmp) / "events.jsonl"), str(Path(tmp) / "images"))
            sink.log_event({"ts": 1, "event": "alarm"}, np.ones((96, 96), np.uint8))
            self.assertTrue(list(Path(tmp, "images").glob("*.png")))
            self.assertEqual(json.loads(Path(tmp, "events.jsonl").read_text())["event"], "alarm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
