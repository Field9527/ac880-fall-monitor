"""根据新鲜、稳定的单手证据生成一次受限控制命令。"""

import math
import numpy as np

NAMES = {"open_palm": "张开手掌", "fist": "握拳", "victory": "V 手势", "unknown": "未确认"}
COMMANDS = {
    "open_palm": {"led_mask": 255},
    "fist": {"led_mask": 0},
    "victory": {"buzzer": {"on": True, "duration_ms": 500}},
}
ACTIONS = {"open_palm": "全部开灯", "fist": "全部关灯", "victory": "短鸣 0.5 秒"}


def classify(landmarks):
    p = np.asarray(landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return "unknown"
    scale = np.linalg.norm(p[9] - p[0])
    if scale < 8:
        return "unknown"

    def angle(a, b):
        return math.degrees(
            math.acos(
                float(
                    np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8), -1, 1)
                )
            )
        )

    straight = []
    folded = []
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        bend = angle(p[pip] - p[mcp], p[tip] - p[dip])
        distance = np.linalg.norm(p[tip] - p[0]) / max(np.linalg.norm(p[pip] - p[0]), 1e-8)
        straight.append(bend < 40 and distance > 1.13)
        folded.append(bend > 65 or distance < 0.95)
    thumb = angle(p[2] - p[1], p[4] - p[3]) < 40 and np.linalg.norm(p[4] - p[5]) > scale * 0.55
    if all(straight) and thumb:
        return "open_palm"
    if all(folded):
        # 竖拇指不当作握拳，也不执行控制。
        if thumb and p[4, 1] < p[2, 1] - scale * 0.25:
            return "unknown"
        return "fist"
    # V 手势仅检查两指伸展分开及其余两指折叠。
    # 拇指可能被遮挡，不作为 V 的必要条件。
    spread = np.linalg.norm(p[8] - p[12]) / scale
    if straight[:2] == [True, True] and all(folded[2:]) and spread > 0.35:
        return "victory"
    return "unknown"


def iou(a, b):
    x, y, w, h = a
    u, v, r, s = b
    inter = max(0, min(x + w, u + r) - max(x, u)) * max(0, min(y + h, v + s) - max(y, v))
    return inter / max(1, w * h + r * s - inter)


class GestureController:
    def __init__(self, hold_s=0.8, min_samples=2, max_age_s=3.0, max_gap_s=5.0, cooldown_s=1.5):
        self.hold_s, self.min_samples, self.max_age_s, self.max_gap_s, self.cooldown_s = (
            hold_s,
            min_samples,
            max_age_s,
            max_gap_s,
            cooldown_s,
        )
        self.last_id = None
        self.last_action = None
        self.last_action_at = -float("inf")
        self.latched = None
        self.clear()
        self.last_pulse_at = -float("inf")

    def clear(self):
        self.name = None
        self.bbox = None
        self.since = None
        self.previous = None
        self.count = 0
        self.neutral_since = None

    def step(self, hand, sample_id, sampled_at, now, enabled=True):
        if not enabled:
            self.clear()
            self.latched = None
            self.last_id = sample_id
            return None
        if sample_id == self.last_id:
            return None
        self.last_id = sample_id
        if sampled_at is None or not 0 <= now - sampled_at <= self.max_age_s:
            self.clear()
            return None
        name = hand.get("name") if hand else None
        bbox = hand.get("bbox") if hand else None
        if name not in COMMANDS or bbox is None or float(hand.get("score", 0)) < 0.85:
            if self.neutral_since is None:
                self.neutral_since = sampled_at
            if sampled_at - self.neutral_since >= 0.5:
                self.latched = None
            self.name = None
            self.count = 0
            self.bbox = None
            self.previous = sampled_at
            return None
        self.neutral_since = None
        contiguous = self.previous is not None and 0 < sampled_at - self.previous <= self.max_gap_s
        same = name == self.name and self.bbox is not None and iou(bbox, self.bbox) >= 0.15
        if not contiguous or not same:
            self.name = name
            self.since = sampled_at
            self.count = 0
        self.previous = sampled_at
        self.bbox = bbox
        self.count += 1
        cooling = name == "victory" and now - self.last_pulse_at < self.cooldown_s
        if (
            self.count < self.min_samples
            or sampled_at - self.since < self.hold_s
            or cooling
            or self.latched == name
        ):
            return None
        self.latched = name
        self.last_action_at = now
        self.last_action = dict(name=name, label=ACTIONS[name], at=now)
        if name == "victory":
            self.last_pulse_at = now
        # 返回独立命令，由硬件主线程执行。
        return {k: dict(v) if isinstance(v, dict) else v for k, v in COMMANDS[name].items()}

    def snapshot(self, now):
        return dict(
            candidate=self.name,
            samples=self.count,
            hold_s=self.hold_s,
            last_action=self.last_action["label"] if self.last_action else None,
            last_action_age_s=round(now - self.last_action_at, 1) if self.last_action else None,
        )
