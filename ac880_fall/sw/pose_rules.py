"""固定相机、单人场景的可解释基线。分数是规则置信度，不是 CNN 概率。"""

from collections import deque
import time
import numpy as np


class PoseRules:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = deque()
        self.triggered = False
        self.last_bbox = None
        self.last_seen = None

    def reset(self):
        self.history.clear()
        self.triggered = False
        self.last_bbox = None
        self.last_seen = None

    def update(self, bbox, now=None):
        now = time.monotonic() if now is None else now
        window = self.cfg.get("transition_window_s", 4.0)
        while self.history and now - self.history[0][0] > window:
            self.history.popleft()
        if bbox is None:
            if self.last_seen is not None and now - self.last_seen > 0.5:
                self.reset()
            return 0.0
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return 0.0
        # 大幅跳变的目标不能沿用上一个人的站立参考。
        if self.last_bbox is not None:
            px, py, pw, ph = self.last_bbox
            distance = ((x + w / 2 - px - pw / 2) ** 2 + (y + h / 2 - py - ph / 2) ** 2) ** 0.5
            if distance > 1.5 * max(ph, h):
                self.reset()
        self.last_seen = now
        self.last_bbox = bbox
        aspect = w / h
        if aspect <= self.cfg.get("upright_ratio", 0.85):
            self.history.append((now, bbox))
            self.triggered = False
        horizontal = aspect >= self.cfg.get("lying_ratio", 1.25)
        if horizontal and len(self.history) >= 3:
            samples = list(self.history)
            # 至少 0.3 秒的直立证据，排除单帧噪声。
            if samples[-1][0] - samples[0][0] >= 0.3:
                ref = max((b for _, b in samples), key=lambda b: b[3])
                _, ry, _, rh = ref
                dropped = (y + h / 2) - (ry + rh / 2)
                if h <= rh * self.cfg.get("height_ratio", 0.72) and dropped >= rh * self.cfg.get(
                    "drop_ratio", 0.12
                ):
                    self.triggered = True
        if self.triggered and horizontal:
            return 0.95
        # 躺卧变为坐姿/站姿或丢失目标时解除规则锁存。
        if not horizontal:
            self.triggered = False
        return 0.05


class ForegroundDetector:
    def __init__(self, cfg):
        import cv2

        cv2.setNumThreads(1)
        self.cv = cv2
        self.cfg = cfg
        self.background = None
        self.started = None
        self.ready = False
        self.pose = PoseRules(cfg)
        self.kernel = np.ones((3, 3), np.uint8)

    def reset(self):
        """机位或场景改变后重新建立无人背景。"""
        self.background = None
        self.started = None
        self.ready = False
        self.pose.reset()

    def process(self, frame565, now=None):
        now = time.monotonic() if now is None else now
        cv = self.cv
        fh, fw = frame565.shape
        sw = self.cfg.get("processing_width", 320)
        sh = int(round(fh * sw / fw))
        # 只在小图上处理；绿色通道有 6 bit，与 PL 一级检测一致。
        small = cv.resize(frame565, (sw, sh), interpolation=cv.INTER_NEAREST)
        gray = (((small >> 5) & 63) * 4).astype(np.uint8)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        if self.started is None:
            self.started = now
            self.background = gray.astype(np.float32)
        if not self.ready:
            cv.accumulateWeighted(gray, self.background, 0.15)
            self.ready = now - self.started >= self.cfg.get("warmup_s", 5.0)
            return {
                "ready": self.ready,
                "bbox": None,
                "score": 0.0,
                "area": 0,
                "reason": "background_warmup",
            }
        delta = gray.astype(np.float32) - self.background
        delta -= float(np.median(delta))
        mask = (np.abs(delta) >= self.cfg.get("foreground_threshold", 28)).astype(np.uint8) * 255
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel, iterations=2)
        occupancy = np.count_nonzero(mask) / mask.size
        if occupancy > self.cfg.get("max_foreground_fraction", 0.5):
            self.pose.reset()
            return {
                "ready": True,
                "bbox": None,
                "score": 0.0,
                "area": 0,
                "reason": "scene_changed_recalibrate",
            }
        # 只更新背景像素，保留倒地后不动的人。
        cv.accumulateWeighted(gray, self.background, 0.01, mask=cv.bitwise_not(mask))
        count, labels, stats, _ = cv.connectedComponentsWithStats(mask)
        bbox, area = None, 0
        if count > 1:
            idx = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
            x, y, w, h, area = (int(a) for a in stats[idx])
            if area >= self.cfg.get("min_foreground_area", 180) and h >= 12:
                sx, sy = fw / sw, fh / sh
                bbox = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
            else:
                area = 0
        score = self.pose.update(bbox, now)
        return {
            "ready": True,
            "bbox": bbox,
            "area": area,
            "score": score,
            "reason": "fall_transition" if score > 0.9 else "tracking",
        }
