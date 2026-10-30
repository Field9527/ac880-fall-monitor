"""验证 CDMA 启动顺序、边界和安全中止，无需连接硬件。"""

import unittest
from unittest.mock import patch
from frame_copy import FrameCopy


class Buffer:
    def __init__(self, address, size=1843200):
        self.physical_address = address
        self.nbytes = size
        self.flushed = 0
        self.invalidated = 0

    def flush(self):
        self.flushed += 1

    def invalidate(self):
        self.invalidated += 1


class Registers:
    def __init__(self):
        self.writes = []
        self.status = 2
        self.stuck = False
        self.error = False

    def write(self, offset, value):
        self.writes.append((offset, value))
        if offset == 0:
            self.status = 2
        if offset == 0x28:
            self.status = 0x20 if self.error else 0 if self.stuck else 0x1002

    def read(self, offset):
        return self.status if offset == 4 else 0


class CopyTests(unittest.TestCase):
    def test_transfer_order_exact_pixels_extent_and_invalidate(self):
        regs = Registers()
        copy = FrameCopy(regs)
        src = Buffer(0x10000000)
        dst = Buffer(0x11000000)
        copy.copy(src, dst)
        self.assertEqual(
            regs.writes,
            [
                (0, 4),
                (4, 0x7000),
                (0x18, src.physical_address),
                (0x20, dst.physical_address),
                (0x28, src.nbytes),
            ],
        )
        self.assertEqual((src.flushed, dst.invalidated), (1, 1))
        self.assertEqual(copy.statistics()["copies"], 1)

    def test_invalid_address_overlap_and_size_do_not_start_dma(self):
        regs = Registers()
        copy = FrameCopy(regs)
        for source, dest in [
            (Buffer(0x10000001), Buffer(0x11000000)),
            (Buffer(0x10000000), Buffer(0x10000008)),
            (Buffer(0x10000000), Buffer(0x40000000)),
            (Buffer(0x10000000), Buffer(0x11000000, 16)),
        ]:
            with self.assertRaises(ValueError):
                copy.copy(source, dest)
        self.assertEqual(regs.writes, [(0, 4)])

    def test_bus_error_and_timeout_reset_before_returning_failure(self):
        for mode in ("error", "stuck"):
            regs = Registers()
            copy = FrameCopy(regs)
            setattr(regs, mode, True)
            times = iter([100.0, 100.1, 100.2, 100.3])
            with patch("frame_copy.time.monotonic", side_effect=lambda: next(times)):
                with self.assertRaises((RuntimeError, TimeoutError)):
                    copy.copy(Buffer(0x10000000), Buffer(0x11000000))
            self.assertEqual(regs.writes[-1], (0, 4))


if __name__ == "__main__":
    unittest.main()
