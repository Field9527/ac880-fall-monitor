"""通过已校验的 NCNN 接口运行固定版本的掌部和 21 点模型。"""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
from gesture_control import classify
from hand_geometry import HandGeometry
from ncnn_backend import NCNNNet


class HandAdapter:
    def __init__(self, net):
        self.net = net
        self.blob = None

    def setInput(self, blob):
        self.blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))

    def getUnconnectedOutLayersNames(self):
        return None

    def forward(self, names):
        return [v.reshape(1, -1) for v in self.net.forward(self.blob)]


class HandModels:
    def __init__(self, directory, library, threads=1):
        self.nets = []
        root = Path(directory)
        meta = json.loads((root / "gesture.json").read_text())
        self.reset()
        if meta.get("architecture") != "mediapipe_hands" or not meta.get("pc_parity"):
            raise ValueError("Checked gesture export required")
        if hashlib.sha256(Path(library).read_bytes()).hexdigest() != meta.get("library_sha256"):
            raise ValueError("Gesture runtime SHA mismatch")
        try:
            for name in ("palm", "hand"):
                cfg = meta["models"][name]
                for key in ("param", "weights"):
                    if (
                        hashlib.sha256((root / cfg[key]).read_bytes()).hexdigest()
                        != cfg[key + "_sha256"]
                    ):
                        raise ValueError("Gesture model SHA mismatch")
                self.nets.append(
                    NCNNNet(
                        library,
                        root / cfg["param"],
                        root / cfg["weights"],
                        threads,
                        output_names=cfg["output_names"],
                        output_layouts=cfg["output_layouts"],
                    )
                )
            self.geometry = HandGeometry(HandAdapter(self.nets[1]), confThreshold=0.85)
            self.anchors = np.asarray(
                [
                    ((x + 0.5) / grid, (y + 0.5) / grid)
                    for grid, repeats in ((24, 2), (12, 6))
                    for y in range(grid)
                    for x in range(grid)
                    for _ in range(repeats)
                ],
                np.float32,
            )
        except Exception:
            self.close()
            raise

    def warmup(self):
        self.nets[0].forward(np.zeros((1, 3, 192, 192), np.float32))
        self.nets[1].forward(np.zeros((1, 3, 224, 224), np.float32))

    def detect(self, rgb):
        import cv2

        self.last_status = "searching"
        now = time.monotonic()
        # 握拳时掌部检测可能丢失，短时复用已确认的腕部和指根。
        if self.track_points is not None and now - self.track_at < 6.0 and self.track_steps < 3:
            points = self.track_points
            low, high = points.min(axis=0), points.max(axis=0)
            result = self._infer(rgb, np.r_[low, high, points.reshape(-1), 1.0])
            if result is not None and self._overlap(result["bbox"], self.track_bbox) > 0.15:
                self._track(result, now, self.track_steps + 1)
                return result
            self.reset()
        h, w = rgb.shape[:2]
        scale = 192.0 / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        left, top = (192 - nw) // 2, (192 - nh) // 2
        boxed = np.zeros((192, 192, 3), np.uint8)
        boxed[top : top + nh, left : left + nw] = cv2.resize(rgb, (nw, nh))
        out = self.nets[0].forward(
            np.ascontiguousarray(boxed.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        )
        delta = out[0].reshape(2016, 18)
        score = 1 / (1 + np.exp(-np.clip(out[1].reshape(2016), -60, 60)))
        index = np.flatnonzero(score >= 0.8)
        if not len(index):
            self.last_status = "no_palm"
            self.reset()
            return None
        d = delta[index]
        center = (d[:, :2] / 192 + self.anchors[index]) * max(h, w) - [left / scale, top / scale]
        size = d[:, 2:4] / 192 * max(h, w)
        boxes = np.c_[center - size / 2, size]
        valid = (size.min(axis=1) > 5) & np.isfinite(boxes).all(axis=1)
        index = index[valid]
        boxes = boxes[valid]
        if not len(index):
            self.last_status = "no_palm"
            self.reset()
            return None
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), score[index].tolist(), 0.8, 0.3, top_k=20)
        if not len(keep):
            self.last_status = "no_palm"
            self.reset()
            return None
        # 多手结果不明确时不发出控制命令。
        kept = np.asarray(keep).reshape(-1)
        i = index[kept[0]]
        xyxy = np.r_[boxes[kept[0], :2], boxes[kept[0], :2] + boxes[kept[0], 2:]]
        points = (delta[i, 4:].reshape(7, 2) / 192 + self.anchors[i]) * max(h, w) - [
            left / scale,
            top / scale,
        ]
        palm = np.r_[xyxy, points.reshape(-1), score[i]]
        result = self._infer(rgb, palm)
        if result is not None:
            if len(kept) > 1:
                result.update(name="unknown", multiple_hands=True)
                self.last_status = "multiple_hands"
                self.reset()
            else:
                self._track(result, now, 0)
        else:
            self.reset()
        return result

    def _infer(self, rgb, palm):
        h, w = rgb.shape[:2]
        self.last_status = "uncertain_landmarks"
        try:
            hand = self.geometry.infer(np.ascontiguousarray(rgb[:, :, ::-1]), palm)
        except ValueError:
            return None
        if hand is None or not np.isfinite(hand).all():
            return None
        landmarks = hand[4:67].reshape(21, 3)
        b = hand[:4].clip([0, 0, 0, 0], [w, h, w, h])
        bbox = [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
        if min(bbox[2:]) < 20:
            self.last_status = "too_small"
            return None
        if np.any(landmarks[:, :2] < [-w * 0.1, -h * 0.1]) or np.any(
            landmarks[:, :2] > [w * 1.1, h * 1.1]
        ):
            self.last_status = "hand_clipped"
            return None
        name = classify(landmarks)
        self.last_status = "unknown_pose" if name == "unknown" else "recognized"
        return dict(
            name=name,
            score=float(min(palm[-1], hand[-1])),
            palm_score=float(palm[-1]),
            landmark_score=float(hand[-1]),
            bbox=bbox,
            landmarks=landmarks[:, :2].round(1).tolist(),
            multiple_hands=False,
        )

    @staticmethod
    def _overlap(a, b):
        from gesture_control import iou

        return iou(a, b)

    def _track(self, result, now, steps):
        points = np.asarray(result["landmarks"])
        self.track_points = points[[0, 5, 9, 13, 17, 1, 2]]
        self.track_bbox = result["bbox"]
        self.track_at = now
        self.track_steps = steps

    def reset(self):
        self.track_points = None
        self.track_bbox = None
        self.track_at = -float("inf")
        self.track_steps = 0

    def close(self):
        for net in self.nets:
            net.close()
        self.nets = []
