"""实验用证据路由，不操作 GPIO 或正式报警。未检出人体仅表示不确定；屏幕区域必须来自已核验内容，不能由手机物体框推断。"""

import math


def checked_box(box):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Invalid XYXY box")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty XYXY box")
    return x1, y1, x2, y2


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def coverage(a, b):
    return intersection(a, b) / ((a[2] - a[0]) * (a[3] - a[1]))


def iou(a, b):
    value = intersection(a, b)
    return value / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - value)


class SemanticPolicy:
    def __init__(self, max_age_s=3, person_threshold=0.4, fall_threshold=0.7):
        if max_age_s <= 0 or not 0 < person_threshold <= 1 or not 0 < fall_threshold <= 1:
            raise ValueError("Invalid policy parameters")
        self.max_age_s = max_age_s
        self.person_threshold = person_threshold
        self.fall_threshold = fall_threshold
        self.last_result_id = -1

    def evaluate(
        self, result_id, sampled_at, now, persons, falls, objects=(), screens=(), mode="live"
    ):
        if mode not in ("live", "phone_replay"):
            raise ValueError("Unknown monitoring mode")
        if not isinstance(result_id, int) or not all(math.isfinite(v) for v in (sampled_at, now)):
            raise ValueError("Invalid result identity/time")
        if result_id <= self.last_result_id:
            return dict(status="duplicate_or_out_of_order", items=[])
        self.last_result_id = result_id
        if now < sampled_at or now - sampled_at > self.max_age_s:
            return dict(status="stale", items=[])

        def candidates(values, threshold):
            items = []
            for value in values:
                box = checked_box(value["box"])
                score = float(value["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid confidence")
                if score >= threshold:
                    items.append(dict(box=box, score=score))
            return items

        people = candidates(persons, self.person_threshold)
        fallen = candidates(falls, self.fall_threshold)
        verified = []
        for screen in screens:
            box = checked_box(screen["box"])
            if screen.get("verified") is True:
                verified.append(box)
        # 通用物体框仅作上下文，不能直接证明屏幕、安全床区或无人。
        for item in objects:
            checked_box(item["box"])
        routed = []
        for item in fallen:
            person = max(people, key=lambda p: iou(item["box"], p["box"]), default=None)
            linked = person is not None and iou(item["box"], person["box"]) >= 0.25
            if (
                mode == "live"
                and linked
                and any(coverage(person["box"], s) >= 0.95 for s in verified)
            ):
                state, reason = "screen_content", "verified_screen_person"
            elif linked:
                state, reason = "risk_candidate", "person_and_fall_evidence"
            else:
                state, reason = "needs_review", "person_not_confirmed"
            routed.append(
                dict(
                    box=list(item["box"]),
                    score=item["score"],
                    status=state,
                    reason=reason,
                    source="phone_screen_replay" if mode == "phone_replay" else "ov5640",
                )
            )
        return dict(
            status="fresh",
            items=routed,
            confirmed_alarm=False,
            note="Risk candidates still require calibrated model and per-person temporal confirmation",
        )
