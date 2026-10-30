"""验证检测坐标与解码规则，无需 PYNQ 或下载模型。"""

import unittest
from pathlib import Path
import tempfile
import json
import hashlib
from concurrent.futures import Future
from unittest.mock import patch, Mock
import numpy as np
from fall_lite import anchors_for_shape, letterbox, decode, FallLiteDetector, LatestFrameInference


class FallLiteTests(unittest.TestCase):
    def test_diagnostic_capture_waits_for_arm_and_stops_at_twelve_unlabelled_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = FallLiteDetector.__new__(FallLiteDetector)
            detector.diagnostic_dir = Path(directory)
            detector.diagnostic_rows = []
            detector.diagnostic_last = -float("inf")
            detector.diagnostic_active = False
            detector.diagnostic_context = "live_scene"
            detector.model_sha256 = "fixture"
            frame = np.zeros((8, 12), np.uint16)
            with patch("fall_lite.time.monotonic", return_value=10):
                detector.record_diagnostic(frame, None)
            self.assertEqual(list(Path(directory).iterdir()), [])
            detector.diagnostic_active = True
            for i in range(15):
                with patch("fall_lite.time.monotonic", return_value=20 + i * 3):
                    detector.record_diagnostic(frame, None)
            manifest = json.loads((Path(directory) / "capture.json").read_text())
            self.assertEqual(len(manifest["samples"]), 12)
            self.assertEqual(len(list(Path(directory).glob("*.npy"))), 12)
            self.assertTrue(all(row["label"] == "unreviewed" for row in manifest["samples"]))

    def test_anchor_order_matches_three_nchw_heads(self):
        a = anchors_for_shape()
        self.assertEqual(a.shape, (7560, 4))
        np.testing.assert_equal(a[0, :2], [4, 4])
        np.testing.assert_equal(a[6, :2], [12, 4])
        np.testing.assert_equal(a[24 * 40 * 6, :2], [8, 8])

    def test_letterbox_retains_entire_portrait_and_landscape(self):
        for h, w in ((720, 1280), (1000, 333), (199, 301)):
            image = np.zeros((h, w, 3), np.uint8)
            boxed, (sx, sy, ox, oy) = letterbox(image)
            self.assertEqual(boxed.shape, (192, 320, 3))
            self.assertGreater(sx, 0)
            self.assertGreater(sy, 0)
            self.assertLessEqual(ox + w * sx, 320)
            self.assertLessEqual(oy + h * sy, 192)
            # 使用实际取整后的比例映射右下角。
            self.assertAlmostEqual((ox + w * sx - ox) / sx, w)
            self.assertAlmostEqual((oy + h * sy - oy) / sy, h)

    def test_decoding_and_nms_remove_duplicate_box(self):
        anchors = np.array([[160, 90, 100, 40], [160, 90, 100, 40], [40, 40, 20, 20]], np.float32)
        p = np.zeros((3, 5), np.float32)
        p[:, 0] = [4, 3, -10]
        found = decode(p, anchors)
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0]["box"], [110, 70, 100, 40])

    def test_nonfinite_or_wrong_shape_fails_instead_of_negative(self):
        for p in (np.array([[float("nan")] * 5]), np.zeros((2, 5))):
            with self.assertRaises(ValueError):
                decode(p, np.zeros((1, 4)))

    def test_untrained_and_mismatched_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.onnx"
            p.write_bytes(b"not an onnx")
            meta = p.with_suffix(".json")
            meta.write_text(json.dumps(dict(trained=False)))
            with self.assertRaisesRegex(ValueError, "trained-model"):
                FallLiteDetector(dict(model=str(p)))
            meta.write_text(json.dumps(dict(trained=True, onnx_sha256="incorrect")))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                FallLiteDetector(dict(model=str(p)))
            meta.write_text(json.dumps(dict(trained=True, label_policy_validated=False)))
            with self.assertRaisesRegex(ValueError, "label policy"):
                FallLiteDetector(dict(model=str(p)))

    def test_slow_model_has_one_pending_frame_and_one_evidence_per_result(self):
        first, second = Future(), Future()
        executor = Mock()
        executor.submit.side_effect = [first, second]
        with patch("fall_lite.ThreadPoolExecutor", return_value=executor):
            worker = LatestFrameInference(Mock())
        frame = np.zeros((12, 20), np.uint16)
        self.assertFalse(worker.poll(frame, 0.0)[1])
        self.assertFalse(worker.poll(frame, 0.1)[1])
        self.assertEqual(executor.submit.call_count, 1)
        first.set_result(
            dict(ready=True, bbox=(1, 2, 3, 4), score=0.95, area=12, reason="fall_box")
        )
        self.assertTrue(worker.poll(frame, 0.8)[1])
        self.assertFalse(worker.poll(frame, 0.9)[1])
        self.assertEqual(worker.completed, 1)
        self.assertEqual(executor.submit.call_count, 2)
        with self.assertRaises(TimeoutError):
            worker.poll(frame, 4.0)
        worker.close()

    def test_ncnn_and_pl_reject_mixed_artifact_manifests_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for name in (
                "model.onnx",
                "cpu.param",
                "weights.bin",
                "runtime.so",
                "pl.param",
                "plugin.so",
            ):
                (base / name).write_bytes(name.encode())
            sha = lambda name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            info = dict(
                source_onnx_sha256=sha("model.onnx"),
                weight_precision="fp32",
                param="cpu.param",
                weights="weights.bin",
                param_sha256=sha("cpu.param"),
                weights_sha256=sha("weights.bin"),
                library_sha256=sha("runtime.so"),
            )
            meta = dict(
                trained=True,
                onnx_sha256=sha("model.onnx"),
                input_width=320,
                input_height=192,
                recommended_threshold=0.7,
                runtimes=dict(ncnn=info),
            )
            cfg = dict(
                model=str(base / "model.onnx"),
                runtime_library=str(base / "runtime.so"),
                engine="ncnn_pl",
                pl_plugin=str(base / "plugin.so"),
            )
            manifest = base / "model.json"
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "PL partition manifest"):
                FallLiteDetector(cfg)
            meta["runtimes"]["ncnn_pl"] = dict(
                source_onnx_sha256=sha("model.onnx"),
                param="pl.param",
                param_sha256="incorrect",
                plugin_sha256=sha("plugin.so"),
            )
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "PL artifact SHA256"):
                FallLiteDetector(cfg)
            cfg["engine"] = "ncnn"
            (base / "weights.bin").write_bytes(b"wrong weights")
            with self.assertRaisesRegex(ValueError, "weights SHA256"):
                FallLiteDetector(cfg)

    def test_pl_overlay_pair_is_checked_before_attaching_dma_driver(self):
        with tempfile.TemporaryDirectory() as d:
            bit = Path(d) / "model.bit"
            bit.write_bytes(b"bit")
            hwh = bit.with_suffix(".hwh")
            hwh.write_bytes(b"hwh")
            detector = FallLiteDetector.__new__(FallLiteDetector)
            detector.engine = "ncnn_pl"
            detector.net = Mock()
            detector.pl_artifacts = dict(
                overlay_sha256="incorrect", hwh_sha256=hashlib.sha256(b"hwh").hexdigest()
            )
            overlay = Mock(bitfile_name=str(bit))
            with self.assertRaisesRegex(ValueError, "overlay SHA256"):
                detector.attach_pl(overlay)
            detector.net.attach_pl.assert_not_called()
            detector.pl_artifacts["overlay_sha256"] = hashlib.sha256(b"bit").hexdigest()
            with patch("fall_detect.overlay_clock_evidence", return_value=dict(passed=True)):
                detector.attach_pl(overlay)
            detector.net.attach_pl.assert_called_once_with(overlay, 5.0)
            detector.net.attach_pl.reset_mock()
            with patch(
                "fall_detect.overlay_clock_evidence", side_effect=RuntimeError("clock mismatch")
            ):
                with self.assertRaisesRegex(RuntimeError, "clock mismatch"):
                    detector.attach_pl(overlay)
            detector.net.attach_pl.assert_not_called()

    def test_lazy_frame_copy_is_owned_raw_and_only_taken_when_free(self):
        pending = Future()
        executor = Mock()
        executor.submit.return_value = pending
        with patch("fall_lite.ThreadPoolExecutor", return_value=executor):
            worker = LatestFrameInference(Mock())
        dma_frame = np.full((12, 20), 0x1234, np.uint16)
        provider = Mock(side_effect=lambda: np.array(dma_frame, copy=True))
        worker.poll(provider, 0.0)
        captured = executor.submit.call_args[0][1]
        dma_frame[:] = 0  # 显示标注和复用不能修改推理像素。
        worker.poll(provider, 0.1)
        self.assertEqual(provider.call_count, 1)
        self.assertTrue(np.all(captured == 0x1234))
        worker.close()

    def test_hand_job_keeps_fall_timestamp_and_cannot_count_as_new_fall(self):
        future = Future()
        executor = Mock()
        executor.submit.return_value = future
        with patch("fall_lite.ThreadPoolExecutor", return_value=executor):
            worker = LatestFrameInference(Mock())
        worker.poll(np.zeros((4, 4), np.uint16), 20)
        future.set_result(dict(source_monotonic=18, fall_fresh=False, hand_sequence=3))
        observation, fresh = worker.poll(np.zeros((4, 4), np.uint16), 21)
        self.assertTrue(fresh)
        self.assertEqual(observation["source_monotonic"], 18)
        self.assertEqual(worker.completed, 1)
        self.assertEqual(worker.fall_completed, 0)
        worker.close()


if __name__ == "__main__":
    unittest.main()
