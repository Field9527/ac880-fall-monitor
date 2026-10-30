"""验证图像 DMA 所有权、控制边界和原始模型输入。"""

import logging
import unittest
from unittest.mock import patch
import numpy as np
from image_processing import ImageFrameCopy
from home_web import HomeOutputs
from test_frame_copy import Buffer
from test_home_web import MMIO as HomeMMIO, LED


class DMA:
    def __init__(self):
        self.writes = []
        self.status = {4: 1, 0x34: 1}
        self.bus_error = False

    def read(self, offset):
        return self.status.get(offset, 0)

    def write(self, offset, value):
        self.writes.append((offset, value))
        if offset == 0 and value == 4:
            self.status = {4: 1, 0x34: 1}
        if offset == 0x58:
            self.status[0x34] = 2
        if offset == 0x28:
            self.status[4] = 0x20 if self.bus_error else 2


class Registers(HomeMMIO):
    def __init__(self):
        super().__init__()
        self.image = 0x00800000

    def read(self, offset):
        if offset == 0x140:
            return ImageFrameCopy.MAGIC
        if offset == 0x144:
            return self.image
        return super().read(offset)

    def write(self, offset, value):
        if offset == 0x144:
            self.image = value
        else:
            super().write(offset, value)


class ImageTests(unittest.TestCase):
    def test_late_completed_transfer_is_discarded_without_aborting_runtime(self):
        from fall_detect import VDMAManager

        class Clock:
            now = 0.0

            def sleep(self, _):
                self.now += 0.01

        clock = Clock()

        class DelayedDMA(DMA):
            def __init__(self):
                super().__init__()
                self.done = None

            def read(self, offset):
                if offset in (4, 0x34) and self.done is not None:
                    return 2 if clock.now >= self.done else 0
                return super().read(offset)

            def write(self, offset, value):
                super().write(offset, value)
                if offset == 0 and value == 4:
                    self.done = None
                if offset == 0x28:
                    self.done = clock.now + 0.075

        class Parked:
            def read(self, _):
                return 0

            def write(self, *_):
                raise AssertionError("Late frames must never be committed")

        with patch("image_processing.time.monotonic", lambda: clock.now), patch(
            "image_processing.time.sleep", clock.sleep
        ):
            copy = ImageFrameCopy(DelayedDMA(), Registers())
            manager = VDMAManager(Parked(), 1280, 720)
            manager.copy_engine = copy
            manager.frames = [Buffer(0x10000000 + i * 0x200000) for i in range(3)]
            manager.display_frames = [Buffer(0x11000000 + i * 0x200000) for i in range(3)]
            manager.frame_indices = lambda: (0, 1)
            self.assertEqual(manager.prepare_display_frame(), (None, None))
            self.assertEqual(copy.statistics()["slow_copies"], 2)
            self.assertFalse(copy.closed)

    def test_stalled_transfer_is_reset_before_destination_can_be_used(self):
        class Clock:
            now = 0.0

            def sleep(self, _):
                self.now += 0.01

        clock = Clock()

        class StalledDMA(DMA):
            def write(self, offset, value):
                super().write(offset, value)
                if offset == 0x28:
                    self.status = {4: 0, 0x34: 0}

        with patch("image_processing.time.monotonic", lambda: clock.now), patch(
            "image_processing.time.sleep", clock.sleep
        ):
            dma = StalledDMA()
            copy = ImageFrameCopy(dma, Registers())
            dest = Buffer(0x11000000)
            with self.assertRaisesRegex(TimeoutError, "250ms"):
                copy.copy(Buffer(0x10000000), dest)
            self.assertEqual(dma.writes[-1], (0, 4))
            self.assertEqual(dest.invalidated, 0)
            self.assertLess(clock.now, 0.3)

    def test_receive_is_armed_before_source_and_complete_frame_invalidated(self):
        regs = Registers()
        dma = DMA()
        copy = ImageFrameCopy(dma, regs)
        source = Buffer(0x10000000)
        dest = Buffer(0x11000000)
        copy.copy(source, dest)
        self.assertLess(dma.writes.index((0x58, 1843200)), dma.writes.index((0x28, 1843200)))
        self.assertEqual((source.flushed, dest.invalidated), (1, 1))
        self.assertFalse(copy.transformed)
        self.assertEqual(copy.statistics()["copies"], 1)

    def test_parameter_bounds_before_hardware_and_atomic_control(self):
        regs = Registers()
        copy = ImageFrameCopy(DMA(), regs)
        outputs = HomeOutputs(regs, LED(), logging.getLogger(), copy)
        for bad in (
            {"mode": []},
            {"mode": "blur"},
            {"brightness": 65},
            {"contrast": 49},
            {"brightness": True},
            {"unknown": 1},
        ):
            before = regs.image
            with self.assertRaises(ValueError):
                outputs.command(dict(led_mask=255, image=bad))
            self.assertEqual(regs.image, before)
            self.assertEqual(outputs.desired["led_mask"], 0)
        outputs.command(dict(image=dict(mode="edge", brightness=-64, contrast=200)))
        self.assertFalse(copy.transformed)  # HTTP 请求不能直接访问硬件。
        outputs.tick(False)
        self.assertTrue(copy.transformed)
        self.assertEqual(regs.image, 2 | (192 << 8) | (256 << 16))
        self.assertEqual(outputs.snapshot()["image"]["mode"], "edge")

    def test_bad_extent_and_bus_error_never_publish_an_output(self):
        dma = DMA()
        copy = ImageFrameCopy(dma, Registers())
        dest = Buffer(0x11000000)
        for source in (Buffer(0x10000001), Buffer(0x10000000, 16), Buffer(0x11000008)):
            with self.assertRaises(ValueError):
                copy.copy(source, dest)
        self.assertEqual(dma.writes, [(0, 4)])
        dma.bus_error = True
        with self.assertRaises(RuntimeError):
            copy.copy(Buffer(0x10000000), dest)
        self.assertEqual(dma.writes[-1], (0, 4))
        self.assertEqual(dest.invalidated, 0)

    def test_raw_frame_is_owned_and_comes_from_camera_not_processed_display(self):
        from fall_detect import VDMAManager

        class Frame(np.ndarray):
            def invalidate(self):
                pass

        capture = [np.full((8, 8), i + 123, np.uint16).view(Frame) for i in range(3)]
        manager = VDMAManager(None, 8, 8)
        manager.frames = capture
        manager.frame_indices = lambda: (0, 1)
        raw = manager.raw_inference_frame()
        self.assertTrue(np.all(raw == 123))
        self.assertFalse(np.shares_memory(raw, capture[0]))
        manager.frames[0][:] = 999
        self.assertTrue(np.all(raw == 123))


if __name__ == "__main__":
    unittest.main()
