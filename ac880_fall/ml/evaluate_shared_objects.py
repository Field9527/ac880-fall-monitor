"""在冻结的 COCO 开发子集统计已知类别检测结果。"""

import cache_env
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from shared_semantic_model import SharedSemantic
from semantic_decode import decode_objects, box_iou
from semantic_features import OBJECT_CLASSES
from fall_lite import anchors_for_shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    records = json.loads((a.cache / "records.json").read_text())["items"]
    features = [np.load(a.cache / ("feature%d.npy" % i), mmap_mode="r") for i in range(3)]
    checkpoint = torch.load(a.model / "object_head.pt", map_location="cpu")
    model = (
        SharedSemantic(a.checkpoint, object_neck=checkpoint.get("object_neck", False)).cuda().eval()
    )
    model.objects.load_state_dict(checkpoint["objects"])
    anchors = anchors_for_shape()
    thresholds = (0.3, 0.5)
    counts = {str(t): {name: dict(gt=0, tp=0, fp=0) for name in OBJECT_CLASSES} for t in thresholds}
    images = 0
    with torch.no_grad():
        for i, row in enumerate(records):
            if row["source"] != "COCO_train2017" or row["split"] != "development":
                continue
            images += 1
            outputs = [
                v.cpu().numpy()
                for v in model.object_forward(
                    [
                        torch.from_numpy(np.asarray(f[i : i + 1], np.float32)).cuda()
                        for f in features
                    ]
                )
            ]
            for threshold in thresholds:
                found = decode_objects(outputs, anchors, threshold, limit=100)
                for k, name in enumerate(OBJECT_CLASSES):
                    gt = [v["box"] for v in row["objects"] if v["label"] == k and not v["crowd"]]
                    candidates = [v for v in found if v["label"] == k]
                    remaining = set(range(len(gt)))
                    c = counts[str(threshold)][name]
                    c["gt"] += len(gt)
                    for detection in candidates:
                        overlap = box_iou([detection["box"]], gt)[0] if gt else np.array([])
                        best = max(remaining, key=lambda j: overlap[j]) if remaining else None
                        if best is not None and overlap[best] >= 0.5:
                            c["tp"] += 1
                            remaining.remove(best)
                        else:
                            c["fp"] += 1
    for group in counts.values():
        for r in group.values():
            r.update(
                precision=r["tp"] / max(1, r["tp"] + r["fp"]), recall=r["tp"] / max(1, r["gt"])
            )
    report = dict(
        images=images,
        metrics=counts,
        iou=0.5,
        scope="Small COCO source development subset, not home-camera or independent field acceptance",
    )
    (a.model / "object_development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
