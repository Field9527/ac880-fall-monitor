"""仅复核与已确认手部高度重合的倒地候选。"""

import math
from gesture_control import iou


def review_hand_region(observation, now):
    hand = observation.get("hand")
    hand_at = observation.get("hand_sampled_at")
    fall_at = observation.get("source_monotonic")
    if not hand or hand_at is None or fall_at is None:
        return observation
    if not all(math.isfinite(v) for v in (now, hand_at, fall_at)):
        return observation
    if not 0 <= now - hand_at <= 3.0 or abs(hand_at - fall_at) > 2.0:
        return observation
    if hand.get("palm_score", 0) < 0.8 or hand.get("landmark_score", 0) < 0.95:
        return observation
    box = hand.get("bbox")
    points = hand.get("landmarks", [])
    if not box or len(box) != 4 or min(box[2:]) < 20 or len(points) != 21:
        return observation
    retained = []
    reviewed = []
    for candidate in observation.get("detections", []):
        overlap = iou(candidate["bbox"], box)
        if overlap >= 0.65:
            reviewed.append(
                dict(
                    candidate,
                    region="verified_hand_region",
                    reviewed_nonfall=True,
                    hand_overlap=round(overlap, 3),
                    hand_sampled_at=hand_at,
                )
            )
        else:
            retained.append(candidate)
    if not reviewed:
        return observation
    result = dict(
        observation,
        detections=retained,
        reviewed_candidates=list(observation.get("reviewed_candidates", [])) + reviewed,
    )
    best = max(retained, key=lambda d: d["score"]) if retained else None
    result.update(
        bbox=best["bbox"] if best else None,
        score=best["score"] if best else 0.0,
        area=best["bbox"][2] * best["bbox"][3] if best else 0,
        reason="fall_box" if best else "hand_region_reviewed_nonfall",
    )
    # 删除最优候选后，不把旧裁剪图关联到其他目标。
    result.pop("_roi", None)
    return result
