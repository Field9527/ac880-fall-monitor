"""无背景摔倒检测，支持已校验的 OpenCV 与 ARM NCNN 后端。"""

import hashlib
import json
import math
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def rgb565(frame):
    pixels = frame.astype(np.uint32)
    return np.stack(
        (
            ((pixels >> 11) & 31) * 255 // 31,
            ((pixels >> 5) & 63) * 255 // 63,
            (pixels & 31) * 255 // 31,
        ),
        axis=-1,
    ).astype(np.uint8)


def anchors_for_shape(height=192, width=320):
    anchors = []
    for stride, sizes in ((8, (24, 38)), (16, (64, 96)), (32, (128, 192))):
        for y in range(math.ceil(height / stride)):
            for x in range(math.ceil(width / stride)):
                for size in sizes:
                    for ratio in (1.0, 2.0, 3.0):
                        anchors.append(
                            (
                                (x + 0.5) * stride,
                                (y + 0.5) * stride,
                                size * math.sqrt(ratio),
                                size / math.sqrt(ratio),
                            )
                        )
    return np.asarray(anchors, dtype=np.float32)


def letterbox(rgb, height=192, width=320, fill=114):
    import cv2

    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = (width - nw) // 2, (height - nh) // 2
    out = np.full((height, width, 3), fill, dtype=np.uint8)
    out[oy : oy + nh, ox : ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    # 按实际取整后的缩放比例恢复框坐标。
    return out, (nw / w, nh / h, ox, oy)


def flatten_heads(outputs):
    return np.concatenate([x.transpose(0, 2, 3, 1).reshape(-1, 5) for x in outputs])


def decode(prediction, anchors, threshold=0.5, nms=0.4, top_k=20):
    import cv2

    if prediction.shape != (len(anchors), 5) or not np.isfinite(prediction).all():
        raise ValueError("Non-finite or invalid detector output")
    scores = 1 / (1 + np.exp(-np.clip(prediction[:, 0], -40, 40)))
    ids = np.flatnonzero(scores >= threshold)
    ids = ids[np.argsort(scores[ids])[::-1][:300]]
    if not len(ids):
        return []
    a, p = anchors[ids], prediction[ids, 1:]
    centers = a[:, :2] + p[:, :2] * 0.1 * a[:, 2:]
    wh = a[:, 2:] * np.exp(np.clip(p[:, 2:] * 0.2, -5, 5))
    boxes = np.concatenate((centers - wh / 2, wh), 1)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores[ids].tolist(), threshold, nms)
    return [
        dict(box=boxes[int(k)].tolist(), score=float(scores[ids[int(k)]]))
        for k in np.asarray(keep).reshape(-1)[:top_k]
    ]


class FallLiteDetector:
    def __init__(self, cfg):
        import cv2

        model = Path(cfg["model"])
        meta_path = Path(cfg.get("metadata", str(model.with_suffix(".json"))))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("trained") is not True:
            raise ValueError("Model has no trained-model manifest")
        if meta.get("label_policy_validated") is False:
            raise ValueError("Model rejected: invalid training-label policy")
        actual = hashlib.sha256(model.read_bytes()).hexdigest()
        if actual != meta["onnx_sha256"]:
            raise ValueError("Model SHA256 does not match metadata")
        self.model_sha256 = actual
        self.training_run = meta.get("training_run", "not_recorded")
        self.width, self.height = meta["input_width"], meta["input_height"]
        self.threshold = float(cfg.get("score_threshold", meta["recommended_threshold"]))
        if not 0 < self.threshold < 1:
            raise ValueError("Invalid detector score threshold")
        self.anchors = anchors_for_shape(self.height, self.width)
        self.engine = cfg.get("engine", "opencv")
        self.diagnostic_dir = None
        self.diagnostic_rows = []
        self.diagnostic_last = -float("inf")
        self.diagnostic_context = cfg.get("test_context", "live_scene")
        if cfg.get("diagnostic_raw_dir"):
            self.diagnostic_dir = Path(cfg["diagnostic_raw_dir"]).resolve()
            self.diagnostic_dir.mkdir(parents=True, exist_ok=False)
        if self.engine == "opencv":
            self.net = cv2.dnn.readNetFromONNX(str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif self.engine in ("ncnn", "ncnn_pl"):
            from ncnn_backend import NCNNNet

            info = meta.get("runtimes", {}).get("ncnn", {})
            if info.get("source_onnx_sha256") != actual or info.get("weight_precision") != "fp32":
                raise ValueError("NCNN manifest is missing or belongs to another model")
            paths = {}
            for key in ("param", "weights"):
                paths[key] = meta_path.parent / info[key]
                if hashlib.sha256(paths[key].read_bytes()).hexdigest() != info[key + "_sha256"]:
                    raise ValueError("NCNN " + key + " SHA256 does not match metadata")
            library = Path(cfg["runtime_library"])
            if hashlib.sha256(library.read_bytes()).hexdigest() != info["library_sha256"]:
                raise ValueError("NCNN runtime library SHA256 mismatch")
            plugin = None
            if self.engine == "ncnn_pl":
                pl_info = meta.get("runtimes", {}).get("ncnn_pl", {})
                if pl_info.get("source_onnx_sha256") != actual:
                    raise ValueError("PL partition manifest is missing or belongs to another model")
                paths["param"] = meta_path.parent / pl_info["param"]
                plugin = Path(cfg["pl_plugin"])
                for path, key in ((paths["param"], "param_sha256"), (plugin, "plugin_sha256")):
                    if hashlib.sha256(path.read_bytes()).hexdigest() != pl_info[key]:
                        raise ValueError("PL artifact SHA256 mismatch: " + str(path))
                self.pl_artifacts = pl_info
            self.net = NCNNNet(
                library,
                paths["param"],
                paths["weights"],
                int(cfg.get("threads", 1)),
                pl_plugin=plugin,
            )
        else:
            raise ValueError("Unknown inference engine: " + str(self.engine))
        # OpenCV 仅负责缩放、NMS 和绘制，单独限制其线程数。
        cv2.setNumThreads(
            int(cfg.get("opencv_threads", cfg.get("threads", 2) if self.engine == "opencv" else 1))
        )

    def close(self):
        if hasattr(self.net, "close"):
            self.net.close()

    def attach_pl(self, overlay, timeout_s=5.0):
        if self.engine == "ncnn_pl":
            bit = Path(overlay.bitfile_name)
            for path, key in ((bit, "overlay_sha256"), (bit.with_suffix(".hwh"), "hwh_sha256")):
                if hashlib.sha256(path.read_bytes()).hexdigest() != self.pl_artifacts[key]:
                    raise ValueError("PL overlay SHA256 mismatch: " + str(path))
            from fall_detect import overlay_clock_evidence

            self.clock_evidence = overlay_clock_evidence(overlay)
            self.net.attach_pl(overlay, timeout_s)

    def reset(self):
        # 模型无需背景采集或背景预热。
        pass

    def predict_rgb(self, rgb):
        boxed, transform = letterbox(rgb, self.height, self.width)
        blob = boxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )[None, :, None, None]
        if self.engine == "opencv":
            self.net.setInput(blob)
            outputs = self.net.forward(["head8", "head16", "head32"])
        else:
            outputs = self.net.forward(blob)
        raw = flatten_heads(outputs)
        items = decode(raw, self.anchors, self.threshold)
        sx, sy, ox, oy = transform
        h, w = rgb.shape[:2]
        detections = []
        for item in items:
            x, y, bw, bh = item["box"]
            x1, y1 = np.clip((x - ox) / sx, 0, w), np.clip((y - oy) / sy, 0, h)
            x2, y2 = np.clip((x + bw - ox) / sx, 0, w), np.clip((y + bh - oy) / sy, 0, h)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                detections.append(
                    dict(bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), score=item["score"])
                )
        return detections

    def process(self, frame, now=None):
        # 先降采样，再展开 RGB 通道。
        start = time.monotonic()
        small = frame[::2, ::2]
        detections = self.predict_rgb(rgb565(small))
        sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
        for d in detections:
            x, y, w, h = d["bbox"]
            d["bbox"] = (round(x * sx), round(y * sy), round(w * sx), round(h * sy))
        best = max(detections, key=lambda d: d["score"]) if detections else None
        self.record_diagnostic(frame, best)
        roi = None
        if best:
            from PIL import Image

            x, y, w, h = best["bbox"]
            crop = frame[y : y + h, x : x + w]
            gray = (((crop >> 5) & 63) * 4).astype(np.uint8)
            roi = np.array(Image.fromarray(gray).resize((96, 96), Image.NEAREST))
        return dict(
            ready=True,
            bbox=best["bbox"] if best else None,
            score=best["score"] if best else 0.0,
            area=best["bbox"][2] * best["bbox"][3] if best else 0,
            reason="fall_box" if best else "no_fall_box",
            detections=detections,
            inference_ms=round((time.monotonic() - start) * 1000, 1),
            engine=self.engine,
            pl_statistics=(
                self.net.pl.statistics() if getattr(self.net, "pl", None) is not None else None
            ),
            backend="fall_lite",
            score_threshold=self.threshold,
            _roi=roi,
        )

    def record_diagnostic(self, frame, best):
        # 仅在限量诊断模式保存标注前的自有原始帧。
        # 采样帧不自动赋予类别。
        now = time.monotonic()
        if (
            self.diagnostic_dir is None
            or not getattr(self, "diagnostic_active", True)
            or len(self.diagnostic_rows) >= 12
            or now - self.diagnostic_last < 2.0
        ):
            return
        name = "input_%02d.npy" % len(self.diagnostic_rows)
        path = self.diagnostic_dir / name
        with path.with_suffix(".partial").open("wb") as output:
            np.save(output, frame)
        path.with_suffix(".partial").replace(path)
        self.diagnostic_rows.append(
            dict(
                file=name,
                monotonic_s=now,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                score=best["score"] if best else 0.0,
                bbox=best["bbox"] if best else None,
                label="unreviewed",
            )
        )
        manifest = dict(
            model_sha256=self.model_sha256,
            input_source="ov5640",
            test_context=self.diagnostic_context,
            purpose="Exact inference inputs for review; not labelled training data",
            samples=self.diagnostic_rows,
        )
        temp = self.diagnostic_dir / "capture.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(self.diagnostic_dir / "capture.json")
        self.diagnostic_last = now


class LatestFrameInference:
    """每次仅运行一个 CPU 推理任务，不排队或重复使用证据。空闲时才调用帧源，返回图像须自有，不能与 DMA 或显示标注共享内存。"""

    def __init__(self, detector, timeout_s=3.0, worker_initializer=None):
        self.detector = detector
        self.pool = (
            ThreadPoolExecutor(max_workers=1, initializer=worker_initializer)
            if worker_initializer is not None
            else ThreadPoolExecutor(max_workers=1)
        )
        self.future = None
        self.submitted_at = None
        self.timeout_s = timeout_s
        self.completed = 0
        self.fall_completed = 0
        self.observation = dict(
            ready=False, bbox=None, score=0.0, area=0, reason="model_starting", backend="fall_lite"
        )

    def poll(self, frame, now):
        fresh = False
        if self.future is not None:
            if self.future.done():
                self.observation = self.future.result()  # 推理失败向上传递，不能按正常场景返回。
                self.observation.setdefault("source_monotonic", self.submitted_at)
                self.future = None
                self.completed += 1
                fresh = True
                self.fall_completed += int(self.observation.get("fall_fresh", True))
            elif now - self.submitted_at > self.timeout_s:
                raise TimeoutError(
                    "Model inference exceeded %.1f seconds (%s)"
                    % (self.timeout_s, getattr(self.detector, "runtime_stage", "inference"))
                )
        if self.future is None:
            frame = frame() if callable(frame) else frame
            self.submitted_at = now
            self.future = self.pool.submit(self.detector.process, frame, now)
        return self.observation, fresh

    def close(self):
        self.pool.shutdown(wait=True)
        if hasattr(self.detector, "close"):
            self.detector.close()
