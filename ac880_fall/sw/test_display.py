"""验证像素一致性和异步预览缓冲区所有权。"""

import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future
import numpy as np
import realtime


class DisplayTests(unittest.TestCase):
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
