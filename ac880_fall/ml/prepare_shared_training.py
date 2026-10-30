"""训练新分支前冻结部分标注掩码与分组划分。"""

import cache_env
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from semantic_features import OBJECT_CLASSES, REGION_CLASSES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def prepare(item, manual):
    path = Path(item["path"])
    sha = digest(path)
    if item.get("sha256") and item["sha256"] != sha:
        raise ValueError("Input image changed")
    with Image.open(path) as image:
        w, h = image.size
    source = item["source"]
    group = item.get("original_group")
    split = item.get("legacy_split")
    if source == "COCO_train2017":
        split = item["development_split"]
    elif source == "user_camera":
        split = "development" if group in ("seated_partial_torso", "dark_or_covered") else "train"
    elif split == "val":
        split = "development"
    if split not in ("train", "development", "test"):
        raise ValueError("Unspecified partition")
    rois = []
    objects = []
    people = [v["box"] for v in item["objects"] if v["label"] == "person"]

    def region(box, label, provenance):
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid region")
        rois.append(dict(box=box, label=label, provenance=provenance))

    for obj in item["objects"]:
        if obj["label"] in OBJECT_CLASSES:
            objects.append(
                dict(
                    label=OBJECT_CLASSES.index(obj["label"]),
                    box=obj["box"],
                    crowd=obj.get("crowd", False),
                )
            )
        if obj["label"] == "person" and obj.get("posture") in ("fallen", "not_fallen"):
            label = 0 if obj["posture"] == "fallen" else 1
            region(obj["box"], label, "source_posture_annotation")
            if label == 1:
                x1, y1, x2, y2 = obj["box"]
                region(
                    [x1, y1, x2, y1 + 0.55 * (y2 - y1)],
                    3,
                    "derived_upper_body_of_author_normal_person",
                )
        elif obj["label"] == "hand_region":
            region(obj["box"], 2, "reviewed_coarse_author_joint_region")
        elif obj["label"] in OBJECT_CLASSES and obj["label"] != "person" and not obj.get("crowd"):
            area = max(1, (obj["box"][2] - obj["box"][0]) * (obj["box"][3] - obj["box"][1]))
            if not any(intersection(obj["box"], person) / area > 0.1 for person in people):
                region(obj["box"], 4, "author_object_box_without_person_overlap")
    fallback = None
    if source == "Fashion-MNIST":
        fallback = 4
    elif source == "synthetic" or item["supervision"].get("fall_absent") is True:
        fallback = 5
    if source == "user_camera" and group == "seated_partial_torso":
        region([0, 0, w, h], 3, "user_confirmed_seated_partial_torso")
    if sha in manual:
        value = manual[sha]
        box = value["normalized_box"]
        box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        region(box, value["label"], "assistant_visual_camera_review_20261028")
    if fallback is not None:
        region([0, 0, w, h], fallback, "explicit_no_fall_or_author_nonhuman_source")
    known = item["supervision"].get("background_known_classes", [])
    return dict(
        path=str(path),
        sha256=sha,
        width=w,
        height=h,
        source=source,
        group=group,
        split=split,
        rotation=0,
        objects=objects,
        regions=rois,
        known_background=[name in known for name in OBJECT_CLASSES],
        fallback_region_class=fallback,
        fall_absent=item["supervision"].get("fall_absent"),
        original_split=item.get("legacy_split"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--manual-camera-review", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    manual = {r["sha256"]: r for r in json.loads(a.manual_camera_review.read_text())["items"]}
    data = json.loads(a.labels.read_text(encoding="utf8"))["items"]
    items = [prepare(v, manual) for v in data]
    partitions = defaultdict(set)
    for item in items:
        partitions[item["sha256"]].add(item["split"])
    priority = {"train": 0, "development": 1, "test": 2}
    relocated = 0
    for item in items:
        chosen = max(partitions[item["sha256"]], key=lambda s: priority[s])
        if item["split"] != chosen:
            relocated += 1
            item["split"] = chosen
    # 只旋转手部来源图，不把旋转人体标为正常。
    variants = []
    for item in items:
        if item["source"] == "FreiHAND_research" and item["split"] == "train":
            variants.extend(dict(item, rotation=angle) for angle in (90, 180, 270))
    items.extend(variants)
    audit = dict(
        records=len(items),
        unique_images=len(partitions),
        rotation_variants=len(variants),
        byte_duplicate_partition_moves=relocated,
        counts=dict(Counter(r["source"] + "/" + r["split"] for r in items)),
        object_classes=OBJECT_CLASSES,
        region_classes=REGION_CLASSES,
        labels_sha256=digest(a.labels),
        manual_camera_review_sha256=digest(a.manual_camera_review),
        limitations=[
            "Old FallDataset split is exact-image grouped, not scene/person independent",
            "Camera groups and FreiHAND selection blocks do not prove subject independence",
            "Frozen v6 backbone has previously seen some current dev/test scenes",
            "No phone-rotation or absent-person hard veto is trained",
            "COCO person pose remains unknown; coarse hand boxes have no box regression loss",
        ],
    )
    (a.out / "index.json").write_text(json.dumps(dict(audit=audit, items=items), indent=2) + "\n")
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
