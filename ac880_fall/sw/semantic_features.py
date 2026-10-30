"""复用骨干特征构造 ROI 描述并进行 FP32 复核。"""

from pathlib import Path
import numpy as np

OBJECT_CLASSES = (
    "person",
    "cell phone",
    "tv",
    "chair",
    "couch",
    "bed",
    "bottle",
    "cup",
    "book",
    "laptop",
)
REGION_CLASSES = (
    "fallen",
    "normal_person",
    "hand_region",
    "partial_body",
    "nonhuman_object",
    "reviewed_nonfall",
)
OUTPUT_NAMES = tuple(
    kind + str(stride) for kind in ("fall", "object", "feature") for stride in (8, 16, 32)
)
NANO_SHARED_OUTPUT_NAMES = (
    tuple("fall" + str(s) for s in (8, 16, 32))
    + tuple(kind + str(s) for s in (8, 16, 32) for kind in ("cls", "dis"))
    + tuple("feature" + str(s) for s in (8, 16, 32))
)
FALL_REGION_OUTPUT_NAMES = tuple(kind + str(s) for kind in ("fall", "feature") for s in (8, 16, 32))


def rotate_xyxy(box, width, height, quarter_turns):
    x1, y1, x2, y2 = map(float, box)
    for _ in range(quarter_turns % 4):
        x1, y1, x2, y2 = y1, width - x2, y2, width - x1
        width, height = height, width
    return [x1, y1, x2, y2]


def region_descriptor(features, box, height=192, width=320):
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Expected finite nonempty XYXY region in model pixels")
    box = box.clip([0, 0, 0, 0], [width, height, width, height])
    if np.any(box[2:] <= box[:2]):
        raise ValueError("Region is outside the image")
    if len(features) != 3:
        raise ValueError("Expected three shared feature levels")
    result = []
    for feature in features:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature[0]
        if feature.ndim != 3 or feature.shape[0] not in (64, 96) or not np.isfinite(feature).all():
            raise ValueError("Expected checked shared CHW features")
        _, h, w = feature.shape
        x1, y1 = np.floor(box[:2] * [w / width, h / height]).astype(int)
        x2, y2 = np.ceil(box[2:] * [w / width, h / height]).astype(int)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        crop = feature[:, y1:y2, x1:x2]
        # 显式平均池化分块，兼容不同 OpenCV 的通道数限制。
        ch, cw = crop.shape[1:]
        pooled = np.empty((feature.shape[0], 3, 3), np.float32)
        for iy in range(3):
            for ix in range(3):
                ys, ye = iy * ch // 3, ((iy + 1) * ch + 2) // 3
                xs, xe = ix * cw // 3, ((ix + 1) * cw + 2) // 3
                pooled[:, iy, ix] = crop[:, ys:ye, xs:xe].mean((1, 2))
        result.extend((pooled.reshape(-1), feature.mean((1, 2))))
    return np.concatenate(result).astype(np.float32)


class RegionVerifier:
    def __init__(self, path):
        with np.load(str(Path(path)), allow_pickle=False) as values:
            self.values = {k: values[k].copy() for k in ("mean", "std", "w1", "b1", "w2", "b2")}
        v = self.values
        dimension = len(v["mean"])
        if (
            dimension not in (1920, 2880)
            or v["std"].shape != (dimension,)
            or v["w1"].shape != (128, dimension)
            or v["b1"].shape != (128,)
            or v["w2"].shape != (len(REGION_CLASSES), 128)
            or v["b2"].shape != (len(REGION_CLASSES),)
        ):
            raise ValueError("Verifier shape mismatch")
        if (
            not all(np.isfinite(x).all() and x.dtype == np.float32 for x in v.values())
            or (v["std"] <= 0).any()
        ):
            raise ValueError("Invalid FP32 verifier parameters")

    def predict(self, features, box):
        return self.predict_many(features, [box])[0]

    def predict_many(self, features, boxes):
        if not boxes:
            return []
        prepared = FeatureRegions(features)
        vectors = np.stack([prepared.descriptor(box) for box in boxes])
        v = self.values
        x = (vectors - v["mean"]) / v["std"]
        hidden = np.maximum(x @ v["w1"].T + v["b1"], 0)
        logits = hidden @ v["w2"].T + v["b2"]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return [dict(zip(REGION_CLASSES, map(float, row))) for row in p]


class FeatureRegions:
    """每层特征只构造一份积分图，供候选区域共享。"""

    def __init__(self, features):
        self.levels = []
        if len(features) != 3:
            raise ValueError("Expected three feature levels")
        for feature in features:
            f = np.asarray(feature, dtype=np.float32)
            if f.ndim == 4 and f.shape[0] == 1:
                f = f[0]
            if f.ndim != 3 or f.shape[0] not in (64, 96) or not np.isfinite(f).all():
                raise ValueError("Invalid shared features")
            h, w = f.shape[1:]
            integral = np.zeros((f.shape[0], h + 1, w + 1), np.float64)
            integral[:, 1:, 1:] = f.cumsum(1, dtype=np.float64).cumsum(2, dtype=np.float64)
            self.levels.append((integral, h, w))

    def descriptor(self, box):
        box = np.asarray(box, np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Invalid region")
        box = box.clip([0, 0, 0, 0], [320, 192, 320, 192])
        if np.any(box[2:] <= box[:2]):
            raise ValueError("Empty region")
        result = []
        for integral, h, w in self.levels:
            x1, y1 = np.floor(box[:2] * [w / 320, h / 192]).astype(int)
            x2, y2 = np.ceil(box[2:] * [w / 320, h / 192]).astype(int)
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            ch, cw = y2 - y1, x2 - x1
            iy, ix = np.mgrid[:3, :3]
            iy = iy.ravel()
            ix = ix.ravel()
            ys = y1 + iy * ch // 3
            ye = y1 + ((iy + 1) * ch + 2) // 3
            xs = x1 + ix * cw // 3
            xe = x1 + ((ix + 1) * cw + 2) // 3
            pooled = (
                integral[:, ye, xe]
                - integral[:, ys, xe]
                - integral[:, ye, xs]
                + integral[:, ys, xs]
            ) / ((ye - ys) * (xe - xs))
            result.extend(
                (
                    pooled.astype(np.float32).reshape(-1),
                    (integral[:, -1, -1] / (h * w)).astype(np.float32),
                )
            )
        return np.concatenate(result)
