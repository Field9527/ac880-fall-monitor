"""仅对训练 ROI 进行相机宽高比与 RGB565 增强。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import letterbox, anchors_for_shape, flatten_heads, decode, rgb565
from semantic_features import FALL_REGION_OUTPUT_NAMES, FeatureRegions
from semantic_decode import box_iou


def main():
    p = argparse.ArgumentParser()
    for name in ("cache", "model", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    records = json.loads((a.cache / "records.json").read_text(encoding="utf8"))["items"]
    rng = np.random.RandomState(1029)
    ids = []
    for source, limit in (
        ("original_falldataset", 1000),
        ("CAUCAFall_v4", 2000),
        ("FreiHAND_research", 400),
        ("user_camera", 100),
    ):
        group = [
            i for i, r in enumerate(records) if r["split"] == "train" and r["source"] == source
        ]
        ids.extend(rng.permutation(group)[:limit].tolist())
    net = cv2.dnn.readNetFromONNX(str(a.model / "shared_semantic.onnx"))
    anchors = anchors_for_shape()
    vectors = []
    rows = []
    for counter, index in enumerate(ids):
        item = records[index]
        assert item["split"] == "train"
        bgr = cv2.imread(item["path"])
        if bgr is None:
            raise RuntimeError("Missing source training image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if item["rotation"]:
            rgb = np.ascontiguousarray(np.rot90(rgb, item["rotation"] // 90))
        height, width = rgb.shape[:2]
        scale = min(640 / width, 360 / height)
        nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (640 - nw) // 2, (360 - nh) // 2
        canvas = np.full((360, 640, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(
            rgb, (nw, nh), interpolation=cv2.INTER_LINEAR
        )
        packed = (
            ((canvas[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((canvas[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (canvas[:, :, 2].astype(np.uint16) >> 3)
        )
        image = rgb565(packed)
        boxed, (sx, sy, ox, oy) = letterbox(image)
        previous = np.asarray(item["input_transform"])

        def mapped(box):
            native = (np.asarray(box) - [previous[2], previous[3], previous[2], previous[3]]) / [
                previous[0],
                previous[1],
                previous[0],
                previous[1],
            ]
            camera = native * [nw / width, nh / height, nw / width, nh / height] + [
                left,
                top,
                left,
                top,
            ]
            return (camera * [sx, sy, sx, sy] + [ox, oy, ox, oy]).clip(
                [0, 0, 0, 0], [320, 192, 320, 192]
            )

        regions = [dict(r, box=mapped(r["box"])) for r in item["regions"]]
        blob = (
            boxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
        ) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        net.setInput(blob)
        outputs = net.forward(list(FALL_REGION_OUTPUT_NAMES))
        features = FeatureRegions(outputs[3:])
        candidates = decode(flatten_heads(outputs[:3]), anchors, 0.7, top_k=16)
        rois = [dict(box=r["box"], label=r["label"], kind="ground_region") for r in regions]
        for candidate in candidates:
            bx, by, bw, bh = candidate["box"]
            box = np.asarray([bx, by, bx + bw, by + bh]).clip([0, 0, 0, 0], [320, 192, 320, 192])
            label = -1
            fallen = [r for r in regions if r["label"] == 0]
            if fallen and box_iou([box], [r["box"] for r in fallen]).max() >= 0.25:
                label = 0
            elif regions:
                overlaps = box_iou([box], [r["box"] for r in regions])[0]
                best = int(overlaps.argmax())
                if overlaps[best] >= 0.4:
                    label = regions[best]["label"]
            if label < 0 and item["fallback_region_class"] is not None:
                label = item["fallback_region_class"]
            if label >= 0:
                rois.append(dict(box=box, label=label, kind="candidate", score=candidate["score"]))
        for roi in rois:
            if min(np.asarray(roi["box"])[2:] - np.asarray(roi["box"])[:2]) < 1:
                continue
            vectors.append(features.descriptor(roi["box"]))
            rows.append(
                dict(
                    source=item["source"],
                    split="train",
                    label=roi["label"],
                    kind=roi["kind"],
                    source_image_id=index,
                    box=np.asarray(roi["box"]).tolist(),
                )
            )
        if counter % 200 == 0:
            print("CAMERA_FEATURES", counter, len(ids), len(rows), flush=True)
    np.save(a.out / "vectors.npy", np.asarray(vectors, np.float32), allow_pickle=False)
    (a.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf8")
    manifest = dict(
        complete=True,
        source_records_sha256=hashlib.sha256((a.cache / "records.json").read_bytes()).hexdigest(),
        selected_train_images=len(ids),
        roi_count=len(rows),
        scope="Training split only; camera 16:9 canvas and RGB565 quantization; unknown candidates excluded; no rotated-normal negative relabeling",
    )
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print("CAMERA_FEATURES_FINISHED", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
