"""为 v5 导入已核验公共标注及相机负例。"""

import cache_env
import hashlib
import json
from pathlib import Path
import struct
import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import rgb565


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "models/fall_lite_v5_public"
    output.mkdir(exist_ok=True)
    generated = root / "data/derived/v5"
    generated.mkdir(parents=True, exist_ok=True)
    items = json.loads((root / "models/fall_lite_v3_fpn/split.json").read_text())["items"]
    items = [
        dict(
            v,
            source="original_falldataset",
            kind="original_positive" if v["boxes"] else "original_negative",
        )
        for v in items
    ]
    # 不同 JPEG 字节可能解码出相同像素。
    # 保留原验证记录，排除匹配的训练副本。
    validation_pixels = {}
    excluded = []

    def pixel_key(item):
        image = cv2.imread(item["path"])
        if image is None:
            raise ValueError("Unreadable original image: " + item["path"])
        return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()

    for item in items:
        if item["split"] == "val":
            validation_pixels[pixel_key(item)] = item
    retained = []
    for item in items:
        match = validation_pixels.get(pixel_key(item)) if item["split"] == "train" else None
        if match is not None:
            if item["boxes"] != match["boxes"]:
                raise ValueError("Duplicate has conflicting source labels")
            excluded.append(
                dict(
                    path=item["path"],
                    matches_validation=match["path"],
                    reason="Identical full decoded pixels despite different JPEG bytes",
                    pixel_sha256=pixel_key(item),
                )
            )
        else:
            retained.append(item)
    items = retained
    public = root / "data/external/caucafall"
    download = json.loads((public / "download_report.json").read_text())
    if not download["complete"]:
        raise ValueError("CAUCAFall subset download incomplete")
    source = json.loads((public / "source_manifest.json").read_text())
    source_files = {(v["subject"], v["activity"], v["filename"]): v for v in source["files"]}
    for entry in source["files"]:
        if not entry["filename"].endswith(".png"):
            continue
        imagepath = public / "raw" / entry["subject"] / entry["activity"] / entry["filename"]
        labelpath = imagepath.with_suffix(".txt")
        for path in (imagepath, labelpath):
            expected = source_files[(entry["subject"], entry["activity"], path.name)]["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Source SHA mismatch: " + str(path))
        image = cv2.imread(str(imagepath))
        h, w = image.shape[:2]
        boxes = []
        classes = []
        for line in labelpath.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError("Invalid source YOLO label: " + str(labelpath))
            category = int(values[0])
            x, y, bw, bh = map(float, values[1:])
            classes.append(category)
            if category not in (0, 1) or not (
                0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
            ):
                raise ValueError("Unsupported source class/coordinates")
            if category == 1:
                boxes.append(
                    [
                        max(0, (x - bw / 2) * w),
                        max(0, (y - bh / 2) * h),
                        min(w, (x + bw / 2) * w),
                        min(h, (y + bh / 2) * h),
                    ]
                )
        subject = int(entry["subject"].split(".")[-1])
        split = "train" if subject <= 7 else "val" if subject == 8 else "test"
        items.append(
            dict(
                path=str(imagepath),
                boxes=boxes,
                split=split,
                source="CAUCAFall_v4",
                subject=subject,
                group=entry["subject"] + "/" + entry["activity"],
                source_classes=classes,
                kind="public_positive" if boxes else "public_negative",
                label_status="author_yolo_class_1_fall",
            )
        )
    capture_items = []
    captures = [
        (
            root / "out/false_alarm_speed_20261016/negative_dorm_20261016",
            "dorm_chair",
            "User confirmed chair with hanging black clothes; visually reviewed",
            False,
        ),
        (
            root / "out/validation_20261017/captures_clothing_picture_20261017",
            "clothing_picture",
            "User clarified black clothing picture triggered ALARM; assistant reviewed all 12 captures",
            True,
        ),
    ]
    for directory, scene, label_source, hold in captures:
        for index, path in enumerate(sorted(directory.glob("*.npy"))):
            rgb = rgb565(np.load(str(path))[::2, ::2])
            dest = generated / (scene + "_" + path.stem + ".png")
            cv2.imwrite(str(dest), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            item = dict(
                path=str(dest),
                boxes=[],
                split="val" if hold and index % 3 == 2 else "train",
                source="user_camera",
                kind="camera_negative",
                group=scene,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                label_source=label_source,
                label_status="no_visible_fallen_person",
                evaluation_scope="same-scene regression; not independent field holdout",
            )
            items.append(item)
            capture_items.append(item)
    fashion = root / "data/external/fashion_mnist"
    for split in ("train", "test"):
        raw = (fashion / (split + ".idx")).read_bytes()
        magic, n, h, w = struct.unpack(">IIII", raw[:16])
        if magic != 2051 or len(raw) != 16 + n * h * w:
            raise ValueError("Invalid Fashion-MNIST subset")
        images = np.frombuffer(raw[16:], np.uint8).reshape(n, h, w)
        for index, gray in enumerate(images):
            rng = np.random.RandomState(index + int(split == "test") * 100000)
            canvas = np.full((192, 320, 3), int(rng.randint(0, 256)), np.uint8)
            size = int(rng.randint(50, 190))
            garment = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
            if index % 2:
                garment = 255 - garment
            x = int(rng.randint(0, 321 - size))
            y = int(rng.randint(0, 193 - size))
            canvas[y : y + size, x : x + size] = garment[:, :, None]
            dest = generated / ("fashion_%s_%04d.png" % (split, index))
            cv2.imwrite(str(dest), canvas)
            items.append(
                dict(
                    path=str(dest),
                    boxes=[],
                    split=split,
                    source="Fashion-MNIST",
                    kind="fashion_negative",
                    group="official_" + split,
                    source_index=index,
                    label_status="garment/product only; auxiliary negative",
                    evaluation_scope="low-resolution clothing auxiliary; not field accuracy",
                )
            )
    for path in sorted(
        (root / "out/false_alarm_speed_20261016/negative_images").glob("synthetic_*.png")
    ):
        items.append(
            dict(
                path=str(path),
                boxes=[],
                split="train",
                source="synthetic",
                kind="synthetic_negative",
                group="seed94",
            )
        )
    counts = {}
    for item in items:
        key = (
            item["source"]
            + "/"
            + item["split"]
            + "/"
            + ("positive" if item["boxes"] else "negative")
        )
        counts[key] = counts.get(key, 0) + 1
    report = dict(
        items=items,
        counts=counts,
        source_caucafall="https://data.mendeley.com/datasets/7w7fccy7ky/4",
        original_policy="Retain validation split; exclude pixel-identical training copies; no missing XML inferred as negative",
        excluded_original_training_duplicates=excluded,
        new_policy="CAUCAFall subject 1-7 train, 8 validation, 9-10 test; whole subjects never split by frames",
        limitations=[
            "Source labels are author annotations, not independently relabelled",
            "Inherited checkpoints saw a training duplicate of one original validation image; original validation is a regression set, not a clean independent benchmark",
            "Subject separation does not guarantee different rooms/cameras",
            "Camera captures are training/selection regressions; not field generalization",
        ],
    )
    (output / "split.json").write_text(json.dumps(report, indent=2))
    (output / "negative_labels.json").write_text(
        json.dumps(dict(items=capture_items, scene_count=2), indent=2)
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
