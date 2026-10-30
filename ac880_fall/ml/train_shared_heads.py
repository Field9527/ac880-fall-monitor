"""在冻结 v6 特征上训练物体和 ROI 头，不修改摔倒权重。"""

import cache_env
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from shared_semantic_model import SharedSemantic, RegionHead, flatten_objects, export_shared

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import anchors_for_shape
from semantic_features import OBJECT_CLASSES, REGION_CLASSES, region_descriptor
from semantic_decode import box_iou, decode_objects, matched_count


def assign_objects(item, anchors):
    n = len(anchors)
    k = len(OBJECT_CLASSES)
    target = np.full((n, k), -1, np.int8)
    target[:, np.asarray(item["known_background"], bool)] = 0
    offsets = np.zeros((n, 4), np.float32)
    positive = np.zeros(n, bool)
    rect = np.c_[anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2]
    valid = [v for v in item["objects"] if not v["crowd"]]
    for obj in item["objects"]:
        if not obj["crowd"]:
            continue
        b = np.asarray(obj["box"])
        area = np.maximum(np.minimum(rect[:, 2:], b[2:]) - np.maximum(rect[:, :2], b[:2]), 0).prod(
            1
        )
        target[area / np.maximum(anchors[:, 2:].prod(1), 1e-8) > 0.5, obj["label"]] = -1
    if not valid:
        return target, offsets, positive
    boxes = np.asarray([v["box"] for v in valid], np.float32)
    labels = np.asarray([v["label"] for v in valid])
    overlaps = box_iou(rect, boxes)
    matched = overlaps.argmax(1)
    best = overlaps.max(1)
    for j, label in enumerate(labels):
        target[overlaps[:, j] >= 0.25, label] = -1
    positive = best >= 0.45
    used = set()
    for j in np.argsort(overlaps.max(0))[::-1]:
        values = overlaps[:, j].copy()
        if used:
            values[list(used)] = -1
        index = int(values.argmax())
        used.add(index)
        positive[index] = True
        matched[index] = j
    ids = np.flatnonzero(positive)
    target[ids, labels[matched[ids]]] = 1
    chosen = boxes[matched[ids]]
    center = (chosen[:, :2] + chosen[:, 2:]) / 2
    size = np.maximum(chosen[:, 2:] - chosen[:, :2], 1)
    offsets[ids, :2] = (center - anchors[ids, :2]) / anchors[ids, 2:] * 10
    offsets[ids, 2:] = np.log(size / anchors[ids, 2:]) * 5
    return target, offsets, positive


def object_loss(pred, target, offsets, positive):
    k = len(OBJECT_CLASSES)
    bce = F.binary_cross_entropy_with_logits(pred[:, :, :k], (target > 0).float(), reduction="none")
    negatives = bce.masked_fill(target != 0, -1).flatten(1)
    top = negatives.topk(256, dim=1).values
    count = (positive.sum(1) * 3).clamp(min=20, max=256)
    selected = torch.arange(256, device=pred.device)[None] < count[:, None]
    denom = positive.sum().clamp(min=1)
    classification = (
        bce[target > 0].sum() + top.clamp(min=0).masked_fill(~selected, 0).sum()
    ) / denom
    regression = (
        F.smooth_l1_loss(pred[:, :, k:][positive], offsets[positive], reduction="sum") / denom
    )
    return classification + regression


def prepare_rois(cache, records, features):
    path = cache / "roi_vectors.npy"
    if path.exists():
        info = json.loads((cache / "roi_records.json").read_text())
        return np.load(path, mmap_mode="r"), info
    rows = []
    vectors = []
    rng = np.random.RandomState(628)
    for index, item in enumerate(records):
        fs = [np.asarray(v[index], np.float32) for v in features]
        rois = []
        for region in item["regions"]:
            rois.append(dict(box=region["box"], label=region["label"], kind="ground_region"))
            if item["split"] == "train" and region["label"] in (0, 1, 2, 3):
                b = np.asarray(region["box"])
                center = (b[:2] + b[2:]) / 2
                size = b[2:] - b[:2]
                for _ in range(3 if region["label"] == 0 else 1):
                    scale = rng.uniform(0.8, 1.2, 2)
                    shift = rng.uniform(-0.08, 0.08, 2) * size
                    rois.append(
                        dict(
                            box=np.r_[
                                center + shift - size * scale / 2, center + shift + size * scale / 2
                            ].tolist(),
                            label=region["label"],
                            kind="jitter",
                        )
                    )
        for candidate in item["base_candidates"]:
            label = -1
            # A labelled fallen body takes priority over overlapping small parts.
            fallen = [r for r in item["regions"] if r["label"] == 0]
            if fallen and box_iou([candidate["box"]], [v["box"] for v in fallen]).max() >= 0.25:
                label = 0
            elif item["regions"]:
                overlaps = box_iou([candidate["box"]], [r["box"] for r in item["regions"]])[0]
                best = int(overlaps.argmax())
                if overlaps[best] >= 0.4:
                    label = item["regions"][best]["label"]
            if label < 0 and item["fallback_region_class"] is not None:
                label = item["fallback_region_class"]
            rois.append(
                dict(box=candidate["box"], label=label, kind="candidate", score=candidate["score"])
            )
        for roi in rois:
            b = np.asarray(roi["box"]).clip([0, 0, 0, 0], [320, 192, 320, 192])
            if np.any(b[2:] - b[:2] < 1):
                continue
            vectors.append(region_descriptor(fs, b))
            rows.append(
                dict(
                    roi, image_id=index, box=b.tolist(), source=item["source"], split=item["split"]
                )
            )
        if index % 1000 == 0:
            print("ROI_CACHE", index, len(rows), flush=True)
    data = np.asarray(vectors, np.float32)
    np.save(path, data, allow_pickle=False)
    (cache / "roi_records.json").write_text(json.dumps(rows) + "\n")
    return data, rows


@torch.no_grad()
def roi_metrics(
    probabilities, rows, records, split="development", confidence=0.98, aggregate=False
):
    predicted = defaultdict(list)
    for p, row in zip(probabilities, rows):
        if row["split"] != split or row["kind"] != "candidate" or row["score"] < 0.7:
            continue
        reject = bool(p[0] <= 1 - confidence and (aggregate or p[1:].max() >= confidence))
        predicted[row["image_id"]].append(dict(box=row["box"], score=row["score"], reject=reject))
    groups = {}
    for index, item in enumerate(records):
        if item["split"] != split:
            continue
        key = item["source"]
        r = groups.setdefault(
            key,
            dict(gt=0, base_tp=0, kept_tp=0, negative_images=0, base_fp_images=0, kept_fp_images=0),
        )
        gt = [
            v["box"]
            for v in item["regions"]
            if v["label"] == 0 and v["provenance"] == "source_posture_annotation"
        ]
        raw = predicted[index]
        kept = [d for d in raw if not d["reject"]]
        reference = [v for v in item.get("reference_v6_candidates", raw) if v["score"] >= 0.7]
        r["gt"] += len(gt)
        r["base_tp"] += matched_count(gt, reference)
        r["kept_tp"] += matched_count(gt, kept)
        if not gt and (item["fall_absent"] is True or item["fallback_region_class"] is not None):
            r["negative_images"] += 1
            r["base_fp_images"] += bool(reference)
            r["kept_fp_images"] += bool(kept)
    return groups


@torch.no_grad()
def all_probabilities(model, x):
    model.eval()
    return torch.cat([model(v).softmax(1).cpu() for v in x.split(1024)]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--object-epochs", type=int, default=35)
    p.add_argument("--roi-epochs", type=int, default=40)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(628)
    np.random.seed(628)
    random.seed(628)
    info = json.loads((a.cache / "records.json").read_text())
    assert info["complete"]
    records = info["items"]
    features = [np.load(a.cache / ("feature%d.npy" % i), mmap_mode="r") for i in range(3)]
    vectors, rois = prepare_rois(a.cache, records, features)
    known = np.array([r["split"] == "train" and r["label"] >= 0 for r in rois])
    labels = np.array([r["label"] for r in rois])
    mean = np.asarray(vectors[known]).mean(0)
    std = np.maximum(np.asarray(vectors[known]).std(0), 0.1)
    x = torch.from_numpy((np.asarray(vectors) - mean) / std).cuda()
    y = torch.from_numpy(labels).long().cuda()
    counts = Counter(labels[known])
    print("ROI_TRAIN_COUNTS", dict(counts), flush=True)
    if set(counts) != set(range(len(REGION_CLASSES))):
        raise ValueError("Missing verifier class")
    fractions = np.array([0.4, 0.15, 0.18, 0.12, 0.1, 0.05])
    weights = np.zeros(len(rois), np.float64)
    for i, row in enumerate(rois):
        if known[i]:
            weights[i] = (
                fractions[row["label"]]
                / counts[row["label"]]
                * (5 if row["source"] == "user_camera" else 1)
            )
    weights = torch.from_numpy(weights).cuda()
    head = RegionHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    best = None
    for epoch in range(a.roi_epochs):
        head.train()
        losses = []
        ids = torch.multinomial(weights, 128 * 100, replacement=True).split(128)
        start = time.perf_counter()
        for selected in ids:
            optimizer.zero_grad()
            logits = head(x[selected])
            loss = F.cross_entropy(logits, y[selected], label_smoothing=0.01)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        probs = all_probabilities(head, x)
        metrics = roi_metrics(probs, rois, records)
        qualified = all(r["kept_tp"] >= r["base_tp"] for r in metrics.values() if r["gt"])
        retained = sum(r["kept_tp"] for r in metrics.values())
        fp = sum(r["kept_fp_images"] for r in metrics.values())
        key = (int(qualified), retained, -fp, -float(np.mean(losses)))
        if best is None or key > best["key"]:
            best = dict(key=key, epoch=epoch + 1, metrics=metrics, qualified=qualified)
            torch.save(dict(state=head.state_dict(), mean=mean, std=std), a.out / "region_head.pt")
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            seconds=time.perf_counter() - start,
            metrics=metrics,
            qualified=qualified,
        )
        history.append(row)
        (a.out / "roi_history.json").write_text(
            json.dumps(dict(best=best, epochs=history), indent=2) + "\n"
        )
        print("ROI_EPOCH", json.dumps(row), flush=True)
    selected = torch.load(a.out / "region_head.pt", map_location="cpu")
    head.cpu().load_state_dict(selected["state"])
    np.savez(
        a.out / "region_verifier.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        w1=head.layers[0].weight.detach().numpy(),
        b1=head.layers[0].bias.detach().numpy(),
        w2=head.layers[3].weight.detach().numpy(),
        b2=head.layers[3].bias.detach().numpy(),
    )
    head.cuda()
    probs = all_probabilities(head, x)
    (a.out / "roi_regression.json").write_text(
        json.dumps(
            {
                split: roi_metrics(probs, rois, records, split)
                for split in ("train", "development", "test")
            },
            indent=2,
        )
        + "\n"
    )
    del x, y, head, probs
    torch.cuda.empty_cache()
    anchors = anchors_for_shape()
    model = SharedSemantic(a.checkpoint).cuda()
    targets = {}
    valid = [
        i
        for i, r in enumerate(records)
        if r["split"] == "train" and (r["objects"] or any(r["known_background"]))
    ]
    for i in valid:
        targets[i] = assign_objects(records[i], anchors)
    fractions = {"COCO_train2017": 0.65, "CAUCAFall_v4": 0.25, "original_falldataset": 0.1}
    counts = Counter(records[i]["source"] for i in valid)
    weights = np.array(
        [fractions.get(records[i]["source"], 0.01) / counts[records[i]["source"]] for i in valid]
    )
    weights /= weights.sum()
    optimizer = torch.optim.AdamW(model.objects.parameters(), lr=0.001, weight_decay=0.0001)
    object_history = []
    for epoch in range(a.object_epochs):
        model.train()
        losses = []
        start = time.perf_counter()
        for _ in range(75):
            selected = np.random.choice(valid, 24, p=weights)
            fs = [torch.from_numpy(np.asarray(f[selected], np.float32)).cuda() for f in features]
            cls = torch.from_numpy(np.stack([targets[i][0] for i in selected])).cuda()
            off = torch.from_numpy(np.stack([targets[i][1] for i in selected])).cuda()
            pos = torch.from_numpy(np.stack([targets[i][2] for i in selected])).cuda()
            optimizer.zero_grad()
            prediction = flatten_objects(model.object_forward(fs))
            loss = object_loss(prediction, cls, off, pos)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite semantic object loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.objects.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        row = dict(
            epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.perf_counter() - start
        )
        object_history.append(row)
        torch.save(
            dict(
                objects=model.objects.state_dict(),
                base_checkpoint_sha256=info["base_checkpoint_sha256"],
                epoch=epoch + 1,
            ),
            a.out / "object_head.pt",
        )
        (a.out / "object_history.json").write_text(json.dumps(object_history, indent=2) + "\n")
        print("OBJECT_EPOCH", json.dumps(row), flush=True)
    export_shared(model, a.out)
    metadata = dict(
        trained=True,
        production_qualified=False,
        training_run=a.out.name,
        object_classes=OBJECT_CLASSES,
        region_classes=REGION_CLASSES,
        input_height=192,
        input_width=320,
        base_checkpoint_sha256=info["base_checkpoint_sha256"],
        index_sha256=info["index_sha256"],
        frozen_fall_output_error=info["fall_output_max_abs_error"],
        roi_selection=best,
        verifier_sha256=hashlib.sha256((a.out / "region_verifier.npz").read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256((a.out / "shared_semantic.onnx").read_bytes()).hexdigest(),
        strong_nonfall_threshold=0.98,
        fall_probability_ceiling=0.02,
        safety="No person-absence veto; uncertain ROI retains original fall candidate",
        limitations=[
            "Development/regression only; no fresh field acceptance",
            "Coarse hand ROIs never used for box regression",
        ],
    )
    (a.out / "shared_semantic.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("SHARED_TRAINING_EXPORTED", a.out, flush=True)


if __name__ == "__main__":
    main()
