"""独立冻结 v6 数据，仅向训练划分增加公共时间采样。"""

import cache_env
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import rgb565


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "models/fall_lite_v6_varied"
    out.mkdir(exist_ok=True)
    generated = root / "data/derived/v6"
    generated.mkdir(parents=True, exist_ok=True)
    report = json.loads((root / "models/fall_lite_v5_public/split.json").read_text())
    items = report["items"]
    public = root / "data/external/caucafall"
    downloaded = json.loads((public / "training_extension_download.json").read_text())
    if not downloaded["complete"]:
        raise ValueError("Training extension not complete")
    manifest = json.loads((public / "training_extension_manifest.json").read_text())
    files = {(v["subject"], v["activity"], v["filename"]): v for v in manifest["files"]}
    for entry in manifest["files"]:
        if not entry["filename"].endswith(".png"):
            continue
        subject = int(entry["subject"].split(".")[-1])
        if not 1 <= subject <= 7:
            raise ValueError("Extension must contain training subjects only")
        imagepath = public / "raw" / entry["subject"] / entry["activity"] / entry["filename"]
        labelpath = imagepath.with_suffix(".txt")
        for path in (imagepath, labelpath):
            expected = files[(entry["subject"], entry["activity"], path.name)]["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Source SHA mismatch")
        image = cv2.imread(str(imagepath))
        h, w = image.shape[:2]
        boxes = []
        classes = []
        for line in labelpath.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError("Invalid author YOLO label")
            category = int(values[0])
            x, y, bw, bh = map(float, values[1:])
            classes.append(category)
            if category not in (0, 1) or not (
                0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
            ):
                raise ValueError("Invalid class/coordinates")
            if category == 1:
                boxes.append(
                    [
                        max(0, (x - bw / 2) * w),
                        max(0, (y - bh / 2) * h),
                        min(w, (x + bw / 2) * w),
                        min(h, (y + bh / 2) * h),
                    ]
                )
        items.append(
            dict(
                path=str(imagepath),
                boxes=boxes,
                split="train",
                source="CAUCAFall_v4",
                subject=subject,
                group=entry["subject"] + "/" + entry["activity"],
                source_classes=classes,
                kind="public_positive" if boxes else "public_negative",
                label_status="author_yolo_class_1_fall",
                temporal_extension=True,
            )
        )
    captures = [
        (
            "captures_unreviewed_v5_20261017",
            "dark_or_covered",
            "Assistant reviewed all12 nearly-black raw captures; no visible fallen person",
        ),
        (
            "captures_clothing_v6_20261017",
            "phone_garment_with_partial_torso",
            "User confirmed phone garment image; all15 visually reviewed, background partial torso, no visible fallen person",
        ),
        (
            "captures_seated_v6_20261017",
            "seated_partial_torso",
            "User confirmed sitting in black shirt with only chest visible; no fall",
        ),
    ]
    added = []
    for directory, scene, provenance in captures:
        capture = root / "out/validation_20261017" / directory
        if not (capture / "capture.json").exists():
            raise ValueError("Required reviewed capture missing: " + str(capture))
        for index, path in enumerate(sorted(capture.glob("*.npy"))):
            image = rgb565(np.load(str(path))[::2, ::2])
            dest = generated / (scene + "_" + path.stem + ".png")
            cv2.imwrite(str(dest), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            row = dict(
                path=str(dest),
                boxes=[],
                split="val" if index % 3 == 2 else "train",
                source="user_camera",
                kind="camera_negative",
                group=scene,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                label_source=provenance,
                label_status="no_visible_fallen_person",
                input_source="ov5640",
                test_context="phone_screen_replay" if scene.startswith("phone_") else "live_scene",
                evaluation_scope="User-labelled training/selection scene; not independent field acceptance",
            )
            items.append(row)
            added.append(row)
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
    report.update(
        items=items,
        counts=counts,
        training_extension_manifest_sha256=hashlib.sha256(
            (public / "training_extension_manifest.json").read_bytes()
        ).hexdigest(),
    )
    report["limitations"].append(
        "v5 test subjects/products were previously evaluated; v6 repeat is a regression, not a fresh independent test"
    )
    (out / "split.json").write_text(json.dumps(report, indent=2))
    (out / "new_negative_labels.json").write_text(
        json.dumps(dict(items=added, capture_group_count=3, independent_scene_count=None), indent=2)
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
