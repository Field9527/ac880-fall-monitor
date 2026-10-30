"""独立 NanoDet 可行性探针，解码遵循官方 NCNN 距离分布示例；不作为摔倒分类或正式报警策略。"""

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ncnn_backend import NCNNNet


OBJECT_NAMES = {
    0: "person",
    15: "cat",
    16: "dog",
    39: "bottle",
    41: "cup",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    67: "cell phone",
    73: "book",
}


def preprocess(bgr, target=320):
    import cv2

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    height, width = bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Empty image")
    scale = target / max(height, width)
    w, h = max(1, int(width * scale)), max(1, int(height * scale))
    pw, ph = (-w) % 32, (-h) % 32
    ox, oy = pw // 2, ph // 2
    image = np.zeros((h + ph, w + pw, 3), np.float32)
    image[oy : oy + h, ox : ox + w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    image = (image - np.array([103.53, 116.28, 123.675], np.float32)) / np.array(
        [57.375, 57.12, 58.395], np.float32
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None]), (w / width, h / height, ox, oy)


def overlap(a, b):
    lo = np.maximum(a[:2], b[:, :2])
    hi = np.minimum(a[2:], b[:, 2:])
    area = np.maximum(hi - lo, 0).prod(1)
    aa = np.maximum(a[2:] - a[:2], 0).prod()
    ba = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return area / np.maximum(aa + ba - area, 1e-8)


def decode(outputs, height, width, mapping, original_shape, threshold=0.4, nms=0.5):
    if len(outputs) != 6:
        raise ValueError("Expected six explicitly named NanoDet matrices")
    boxes = []
    scores = []
    labels = []
    for k, stride in enumerate((8, 16, 32)):
        cls, dist = outputs[2 * k : 2 * k + 2]
        gh, gw = height // stride, width // stride
        if cls.shape != (gh * gw, 80) or dist.shape != (gh * gw, 32):
            raise ValueError("Grid/output mismatch")
        if (
            not np.isfinite(cls).all()
            or not np.isfinite(dist).all()
            or cls.min() < 0
            or cls.max() > 1
        ):
            raise ValueError("Invalid sigmoid scores or distances")
        label = cls.argmax(1)
        score = cls[np.arange(len(cls)), label]
        ids = np.flatnonzero(score >= threshold)
        if not len(ids):
            continue
        logits = dist[ids].reshape(-1, 4, 8)
        values = np.exp(logits - logits.max(2, keepdims=True))
        values /= values.sum(2, keepdims=True)
        distance = (values * np.arange(8, dtype=np.float32)).sum(2) * stride
        centers = np.stack(((ids % gw + 0.5) * stride, (ids // gw + 0.5) * stride), 1)
        boxes.append(np.concatenate((centers - distance[:, :2], centers + distance[:, 2:]), 1))
        scores.append(score[ids])
        labels.append(label[ids])
    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    sx, sy, ox, oy = mapping
    boxes = (boxes - np.array([ox, oy, ox, oy])) / np.array([sx, sy, sx, sy])
    oh, ow = original_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)
    valid = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    kept = []
    # 分类别 NMS 保留手机与人体重叠框，供后续关联。
    for label in np.unique(labels[valid]):
        ids = valid[labels[valid] == label]
        ids = ids[np.argsort(scores[ids])[::-1][:300]]
        while len(ids):
            best = ids[0]
            kept.append(int(best))
            ids = ids[1:][overlap(boxes[best], boxes[ids[1:]]) <= nms]
    kept = sorted(kept, key=lambda k: float(scores[k]), reverse=True)[:100]
    return [
        dict(
            label=int(labels[k]),
            name=OBJECT_NAMES.get(int(labels[k]), "class_" + str(labels[k])),
            score=float(scores[k]),
            box=boxes[k].tolist(),
        )
        for k in kept
    ]


class NanoDetProbe:
    def __init__(self, directory, library, threads=1):
        directory = Path(directory)
        meta = json.loads((directory / "source_manifest.json").read_text(encoding="utf8"))
        entries = {v["file"]: v for v in meta["files"]}
        for name in ("nanodet-m.param", "nanodet-m.bin"):
            if (
                hashlib.sha256((directory / name).read_bytes()).hexdigest()
                != entries[name]["sha256"]
            ):
                raise ValueError("Probe source hash mismatch: " + name)
        names = [kind + "_pred_stride_" + str(s) for s in (8, 16, 32) for kind in ("cls", "dis")]
        self.net = NCNNNet(
            library,
            directory / "nanodet-m.param",
            directory / "nanodet-m.bin",
            threads=threads,
            output_names=names,
            flat_outputs=True,
        )

    def detect(self, bgr):
        start = time.monotonic()
        blob, mapping = preprocess(bgr)
        prepared = time.monotonic()
        outputs = self.net.forward(blob)
        finished = time.monotonic()
        objects = decode(outputs, *blob.shape[2:], mapping, bgr.shape)
        return dict(
            objects=objects,
            seconds=dict(
                preprocess=prepared - start,
                inference=finished - prepared,
                postprocess=time.monotonic() - finished,
                total=time.monotonic() - start,
            ),
            input_shape=list(blob.shape),
            output_shapes=[list(v.shape) for v in outputs],
        )

    def close(self):
        self.net.close()
