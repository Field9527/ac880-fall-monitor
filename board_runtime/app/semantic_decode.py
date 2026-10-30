"""按类别解码 SSD 物体框，与报警判决分离。"""

import numpy as np
from semantic_features import OBJECT_CLASSES


def box_iou(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    area = np.maximum(
        np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0
    ).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return area / np.maximum(aa[:, None] + bb[None, :] - area, 1e-8)


def decode_objects(outputs, anchors, threshold=0.4, limit=30):
    import cv2

    k = len(OBJECT_CLASSES)
    raw = np.concatenate([x.transpose(0, 2, 3, 1).reshape(-1, k + 4) for x in outputs])
    if raw.shape != (len(anchors), k + 4) or not np.isfinite(raw).all():
        raise ValueError("Invalid object output")
    # 先按 logit 阈值筛选，再计算指数。
    # 减少 ARMv7 上无效的逐标量指数计算。
    labels = raw[:, :k].argmax(1)
    logits = raw[np.arange(len(raw)), labels]
    cutoff = np.log(threshold / (1 - threshold))
    ids = np.flatnonzero(logits >= cutoff)
    ids = ids[np.argsort(logits[ids])[::-1][:400]]
    if not len(ids):
        return []
    scores = np.zeros(len(raw), np.float32)
    scores[ids] = 1 / (1 + np.exp(-np.clip(logits[ids], -50, 50)))
    anchor = anchors[ids]
    offset = raw[ids, k:]
    center = anchor[:, :2] + offset[:, :2] * 0.1 * anchor[:, 2:]
    size = anchor[:, 2:] * np.exp(np.clip(offset[:, 2:] * 0.2, -5, 5))
    xywh = np.c_[center - size / 2, size]
    result = []
    for label in np.unique(labels[ids]):
        group = np.flatnonzero(labels[ids] == label)
        keep = cv2.dnn.NMSBoxes(xywh[group].tolist(), scores[ids[group]].tolist(), threshold, 0.45)
        for local in np.asarray(keep).reshape(-1):
            index = group[int(local)]
            x, y, w, h = xywh[index]
            result.append(
                dict(
                    label=int(label),
                    name=OBJECT_CLASSES[label],
                    score=float(scores[ids[index]]),
                    box=[float(x), float(y), float(x + w), float(y + h)],
                )
            )
    return sorted(result, key=lambda v: v["score"], reverse=True)[:limit]


def matched_count(expected, detected, threshold=0.5):
    if not len(expected) or not detected:
        return 0
    overlaps = box_iou([v["box"] for v in detected], expected)
    remaining = set(range(len(expected)))
    count = 0
    for index in sorted(range(len(detected)), key=lambda i: detected[i]["score"], reverse=True):
        if not remaining:
            break
        best = max(remaining, key=lambda k: overlaps[index, k])
        if overlaps[index, best] >= threshold:
            remaining.remove(best)
            count += 1
    return count
