"""验证 ABI、权重、缓存所有权和 DMA 排空。"""

import ctypes as ct
import unittest
import numpy as np
from pl_pointwise import PLPointwise, FLOAT_PTR


class Buffer(np.ndarray):
    def __new__(cls, shape, address):
        obj = np.full(shape, np.nan, dtype=np.float32).view(cls)
        obj.physical_address = address
        obj.freed = False
        obj.actions = []
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.physical_address = getattr(source, "physical_address", 0)
            self.actions = getattr(source, "actions", [])

    def flush(self):
        self.actions.append(("flush", self.nbytes))

    def invalidate(self):
        self.actions.append(("invalidate", self.nbytes))

    def freebuffer(self):
        self.freed = True


class Core:
    def __init__(self):
        self.buffers = {}
        self.registers = {}
        self.remaining = 0
        self.started = 0

    def allocate(self, shape, dtype):
        address = 0x1000000 + len(self.buffers) * 0x400000
        buffer = Buffer(shape, address)
        self.buffers[address] = buffer
        return buffer

    def read(self, offset):
        if self.remaining:
            self.remaining -= 1
            return 1
        return 6

    def write(self, offset, value):
        self.registers[offset] = value
        if offset != 0:
            return
        self.started += 1
        r = self.registers
        ci, co, pixels, ins, outs, relu, bias = [
            r[o] for o in (0x50, 0x58, 0x60, 0x68, 0x70, 0x78, 0x80)
        ]
        image = self.buffers[r[0x20]][: ci * ins].reshape(ci, ins)[:, :pixels]
        weights = self.buffers[r[0x30]]
        result = weights[: ci * co].reshape(co, ci) @ image
        if bias:
            result += weights[ci * co :, None]
        if relu:
            result = np.maximum(result, 0)
        self.buffers[r[0x40]][: co * outs].reshape(co, outs)[:, :pixels] = result


def pointer(array):
    return array.ctypes.data_as(FLOAT_PTR)


class PLDriverTests(unittest.TestCase):
    def setUp(self):
        self.core = Core()
        self.driver = PLPointwise(self.core, self.core.allocate)
        self.image = np.arange(12, dtype=np.float32).reshape(3, 4) - 4
        self.weights = np.array([[1, 2, -1], [-2, 0.5, 1]], np.float32)
        self.bias = np.array([0.1, 0.2], np.float32)
        self.out = np.full((2, 8), -999, np.float32)

    def run_layer(self):
        self.driver.compute(
            pointer(self.image),
            pointer(self.weights),
            pointer(self.bias),
            pointer(self.out),
            3,
            2,
            3,
            4,
            8,
            1,
            1,
        )

    def test_nchw_padding_bias_relu_and_weight_upload_cache(self):
        self.run_layer()
        self.run_layer()
        expected = np.maximum(self.weights @ self.image[:, :3] + self.bias[:, None], 0)
        np.testing.assert_allclose(self.out[:, :3], expected)
        np.testing.assert_array_equal(self.out[:, 3:], -999)
        self.assertEqual(len(self.driver.weights), 1)
        self.assertEqual(self.driver.statistics()["macs"], 36)
        self.assertIn(("flush", 48), self.driver.input.actions)
        self.assertIn(("invalidate", 64), self.driver.output.actions)
        self.driver.close()
        self.assertTrue(all(v.freed for v in self.core.buffers.values()))

    def test_invalid_dimensions_do_not_allocate_or_launch(self):
        with self.assertRaises(ValueError):
            self.driver.compute(None, None, None, None, 961, 2, 3, 4, 8, 1, 1)
        self.assertEqual(self.core.started, 0)
        self.assertFalse(self.core.buffers)

    def test_timeout_drains_dma_before_reporting_error(self):
        original = self.core.write

        def delayed(offset, value):
            original(offset, value)
            if offset == 0:
                self.core.remaining = 3

        self.core.write = delayed
        self.driver.timeout_s = -1
        with self.assertRaisesRegex(TimeoutError, "drained"):
            self.run_layer()
        self.assertFalse(self.driver.busy)
        self.assertEqual(self.core.remaining, 0)
        self.assertFalse(any(v.freed for v in self.core.buffers.values()))
        self.driver.close()

    def test_close_waits_for_idle_before_freeing(self):
        self.run_layer()
        self.driver.busy = True
        self.core.remaining = 3
        self.driver.close()
        self.assertEqual(self.core.remaining, 0)
        self.assertTrue(all(v.freed for v in self.core.buffers.values()))

    def test_clear_on_read_done_can_precede_idle(self):
        reads = iter((4, 2, 4))
        self.core.read = lambda offset: next(reads)
        self.run_layer()
        self.assertEqual(self.driver.calls, 1)
        self.assertFalse(self.driver.busy)
        self.driver.close()


if __name__ == "__main__":
    unittest.main()
