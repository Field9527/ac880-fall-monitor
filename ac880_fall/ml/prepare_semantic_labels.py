"""准备部分标注监督，缺失标注不自动补为负例。"""

import cache_env
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image


def legacy_record(item):
    path = Path(item["path"])
    objects = [
        dict(label="person", box=list(box), posture="fallen", source="existing_fall_annotation")
        for box in item["boxes"]
    ]
    yolo = path.with_suffix(".txt")
    if str(item.get("source", "")).startswith("CAUCA") and yolo.is_file():
        with Image.open(path) as image:
            w, h = image.size
        objects = []
        for line in yolo.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError("Invalid source person annotation")
            label = int(values[0])
            x, y, bw, bh = map(float, values[1:])
            if label not in (0, 1) or not (
                0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
            ):
                raise ValueError("Invalid CAUCA class/box")
            box = [
                max(0, (x - bw / 2) * w),
                max(0, (y - bh / 2) * h),
                min(w, (x + bw / 2) * w),
                min(h, (y + bh / 2) * h),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Empty source person box")
            objects.append(
                dict(
                    label="person",
                    box=box,
                    posture="fallen" if label == 1 else "not_fallen",
                    source="author_yolo_class",
                )
            )
    reviewed_negative = not item["boxes"] and item.get("label_status") in {
        "no_visible_fallen_person",
        "assistant_visual_review_negative",
        "author_yolo_class_1_fall",
    }
    return dict(
        path=str(path),
        source=item.get("source", "FallDataset"),
        original_group=item.get("group"),
        legacy_split=item.get("split"),
        objects=objects,
        supervision=dict(
            object_background_known=False,
            fall_absent=True if reviewed_negative else None,
            posture_only_on_annotated_boxes=True,
        ),
        scope="legacy training/development/regression; not independent field acceptance",
    )


def coco_record(item, categories, directory):
    objects = []
    for ann in item["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        objects.append(
            dict(
                label=categories[ann["category_id"]],
                box=[x, y, x + w, y + h],
                posture=None,
                crowd=bool(ann["iscrowd"]),
                source="COCO_author",
            )
        )
    return dict(
        path=str(directory / item["path"]),
        sha256=item["sha256"],
        source="COCO_train2017",
        original_group="coco_" + str(item["id"]),
        legacy_split=None,
        objects=objects,
        development_split=item["split"],
        license=item["license"],
        supervision=dict(
            object_background_known=True,
            fall_absent=None,
            background_known_classes=sorted(categories.values()),
            additional_classes_background_known=False,
            posture_only_on_annotated_boxes=True,
            crowd_requires_ignore_mask=True,
        ),
        scope="generic object supervision; no normal/fall posture supervision",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--hand-regions", type=Path)
    p.add_argument("--coco", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    index = a.workspace / "models/fall_lite_v6_varied/split.json"
    old = json.loads(index.read_text())["items"]
    records = []
    missing = []
    hand_regions = {}
    if a.hand_regions:
        for region in json.loads(a.hand_regions.read_text())["items"]:
            if (
                region["review_status"] != "assistant_visual_reviewed"
                or region["train_eligible"] is not True
            ):
                raise ValueError("Derived hand regions require visual review")
            if region["sha256"] in hand_regions:
                raise ValueError("Duplicate hand-region identity")
            hand_regions[region["sha256"]] = region
    for item in old:
        if not Path(item["path"]).is_file():
            missing.append(item["path"])
            continue
        records.append(legacy_record(item))
    reviewed = a.workspace / "out/clarity_hardneg_20261019/reviewed_negatives.json"
    negatives = json.loads(reviewed.read_text())["items"]
    for item in negatives:
        path = Path(item["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Reviewed negative changed")
        if item["label_status"] != "no_visible_fallen_person":
            raise ValueError("Negative was not reviewed")
        row = legacy_record(item)
        row.update(
            sha256=item["sha256"],
            kind=item["kind"],
            terms=item.get("terms"),
            label_source=item["label_source"],
        )
        # 图像级含手标签不能推导手部或人体检测框。
        row["weak_image_label"] = (
            "hand_present" if item["kind"] == "hand_negative" else "reviewed_partial_no_fall"
        )
        if item["kind"] == "hand_negative" and item["sha256"] in hand_regions:
            region = hand_regions[item["sha256"]]
            row["objects"].append(
                dict(
                    label="hand_region",
                    box=region["box"],
                    posture=None,
                    source="FreiHAND_author_joint_projection",
                    quality=region["quality"],
                    derived=True,
                    supervision_weight=0.5,
                    complete_object_box=False,
                    regression_supervision=False,
                    roi_classification_supervision=True,
                )
            )
            row["supervision"]["hand_background_known"] = False
        records.append(row)
    if a.coco:
        coco = json.loads((a.coco / "manifest.json").read_text())
        if not coco["complete"]:
            raise ValueError("COCO pilot incomplete")
        categories = {v["id"]: v["name"] for v in coco["categories"]}
        records.extend(coco_record(v, categories, a.coco) for v in coco["items"])
    groups = defaultdict(set)
    for row in records:
        if row.get("original_group"):
            groups[(row["source"], str(row["original_group"]))].add(row.get("legacy_split"))
    crossing = [
        dict(source=k[0], group=k[1], splits=sorted(str(x) for x in v))
        for k, v in groups.items()
        if len(v) > 1
    ]
    counts = Counter(row["source"] for row in records)
    summary = dict(
        records=len(records),
        sources=dict(counts),
        missing_files=len(missing),
        missing_person_or_object_boxes=sum(not v["objects"] for v in records),
        missing_complete_detection_boxes=sum(
            not any(not b.get("derived", False) for b in r["objects"]) for r in records
        ),
        derived_hand_regions=sum(
            any(v["label"] == "hand_region" for v in r["objects"]) for r in records
        ),
        cross_split_legacy_groups=crossing,
        independent_field_acceptance=False,
        warnings=[
            "Do not use missing person/object annotations as detection background",
            "COCO person is not a negative fall label",
            "Legacy split and camera-scene regression are not a new independent test",
            "Hand-presence weak labels require boxes or a separate image/ROI loss",
            "Source/image-ID grouping alone does not prove person/scene independence",
        ],
    )
    (a.out / "labels.json").write_text(
        json.dumps(dict(items=records), ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    (a.out / "audit.json").write_text(
        json.dumps(dict(summary, missing=missing), ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
