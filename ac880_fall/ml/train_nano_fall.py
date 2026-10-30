"""保留预训练 NanoDet 物体知识，训练摔倒分支。"""

import cache_env
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanodet_graph import NanoFall
from fall_lite import anchors_for_shape, flatten_heads, decode
from semantic_decode import box_iou, matched_count


def target_for(row, anchors):
    gt = [
        v["box"]
        for v in row["regions"]
        if v["label"] == 0 and v["provenance"] == "source_posture_annotation"
    ]
    n = len(anchors)
    cls = np.full(n, -1, np.int8)
    offsets = np.zeros((n, 4), np.float32)
    if not gt:
        if row["fall_absent"] is True or row["fallback_region_class"] is not None:
            cls[:] = 0
        return cls, offsets
    gt = np.asarray(gt, np.float32)
    rect = np.c_[anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2]
    overlaps = box_iou(rect, gt)
    matched = overlaps.argmax(1)
    positive = overlaps.max(1) >= 0.45
    used = set()
    for j in np.argsort(overlaps.max(0))[::-1]:
        scores = overlaps[:, j].copy()
        if used:
            scores[list(used)] = -1
        i = int(scores.argmax())
        used.add(i)
        positive[i] = True
        matched[i] = j
    ids = np.flatnonzero(positive)
    cls[ids] = 1
    selected = gt[matched[ids]]
    offsets[ids, :2] = (
        ((selected[:, :2] + selected[:, 2:]) / 2 - anchors[ids, :2]) / anchors[ids, 2:] * 10
    )
    offsets[ids, 2:] = (
        np.log(np.maximum(selected[:, 2:] - selected[:, :2], 1) / anchors[ids, 2:]) * 5
    )
    return cls, offsets


def flatten(outputs):
    return torch.cat([v.permute(0, 2, 3, 1).reshape(v.shape[0], -1, 5) for v in outputs], 1)


def masked_loss(pred, cls, offset):
    positive = cls > 0
    bce = F.binary_cross_entropy_with_logits(pred[:, :, 0], positive.float(), reduction="none")
    npos = positive.sum(1).clamp(min=1)
    posloss = bce.masked_fill(~positive, 0).sum(1) / npos
    neg = bce.masked_fill(cls != 0, -1).topk(80, dim=1).values.clamp(min=0).mean(1)
    regression = (
        F.smooth_l1_loss(pred[:, :, 1:], offset, reduction="none")
        .sum(2)
        .masked_fill(~positive, 0)
        .sum(1)
        / npos
    )
    return (posloss + neg + regression).mean()


def sampling_group(row, cls):
    """Keep scarce normal people separate from plentiful clothing images."""
    return row["source"] + ("/fallen" if (cls > 0).any() else "/nonfall")


def sampling_weights(rows, targets):
    fractions = {
        "original_falldataset/fallen": 0.30,
        "CAUCAFall_v4/fallen": 0.18,
        "original_falldataset/nonfall": 0.12,
        "CAUCAFall_v4/nonfall": 0.12,
        "FreiHAND_research/nonfall": 0.12,
        "user_camera/nonfall": 0.10,
        "Fashion-MNIST/nonfall": 0.04,
        "synthetic/nonfall": 0.02,
    }
    groups = {i: sampling_group(rows[i], target[0]) for i, target in targets.items()}
    counts = Counter(groups.values())
    if set(counts) - set(fractions):
        raise ValueError("Unreviewed training source: " + str(set(counts) - set(fractions)))
    ids = list(targets)
    weights = np.array([fractions[groups[i]] / counts[groups[i]] for i in ids])
    return ids, weights / weights.sum(), counts


def selection_key(metrics):
    """Require each positive domain to retain recall before optimizing FP."""
    ratios = [r["nano_tp"] / r["reference_tp"] for r in metrics.values() if r["reference_tp"]]
    retained = all(r["nano_tp"] >= r["reference_tp"] for r in metrics.values() if r["gt"])
    cost = sum(
        r["nano_fp"] / r["negative_images"] for r in metrics.values() if r["negative_images"]
    )
    tp = sum(r["nano_tp"] for r in metrics.values())
    return (1, -cost, tp) if retained else (0, min(ratios, default=0), -cost, tp)


@torch.no_grad()
def assess(model, features, rows, indices, anchors):
    model.eval()
    groups = {}
    candidates = {}
    for offset in range(0, len(indices), 32):
        ids = indices[offset : offset + 32]
        fs = [torch.from_numpy(np.asarray(v[ids], np.float32)).cuda() for v in features]
        outputs = [v.cpu().numpy() for v in [head(f) for head, f in zip(model.falls, fs)]]
        for k, i in enumerate(ids):
            raw = flatten_heads([v[k : k + 1] for v in outputs])
            found = decode(raw, anchors, 0.3, top_k=16)
            found = [
                dict(
                    score=v["score"],
                    box=[
                        v["box"][0],
                        v["box"][1],
                        v["box"][0] + v["box"][2],
                        v["box"][1] + v["box"][3],
                    ],
                )
                for v in found
            ]
            candidates[i] = found
            visible = [v for v in found if v["score"] >= 0.7]
            row = rows[i]
            gt = [
                v["box"]
                for v in row["regions"]
                if v["label"] == 0 and v["provenance"] == "source_posture_annotation"
            ]
            r = groups.setdefault(
                row["source"],
                dict(gt=0, reference_tp=0, nano_tp=0, negative_images=0, reference_fp=0, nano_fp=0),
            )
            r["gt"] += len(gt)
            r["reference_tp"] += matched_count(
                gt, [v for v in row["reference_v6_candidates"] if v["score"] >= 0.7]
            )
            r["nano_tp"] += matched_count(gt, visible)
            if not gt and (row["fall_absent"] is True or row["fallback_region_class"] is not None):
                r["negative_images"] += 1
                r["reference_fp"] += bool(
                    [v for v in row["reference_v6_candidates"] if v["score"] >= 0.7]
                )
                r["nano_fp"] += bool(visible)
    return groups, candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(631)
    np.random.seed(631)
    info = json.loads((a.cache / "records.json").read_text())
    rows = info["items"]
    anchors = anchors_for_shape()
    features = [np.load(a.cache / ("feature%d.npy" % k), mmap_mode="r") for k in range(3)]
    targets = {}
    for i, row in enumerate(rows):
        if row["split"] != "train":
            continue
        cls, offset = target_for(row, anchors)
        if not (cls >= 0).any():
            continue
        targets[i] = (cls, offset)
    valid, weights, counts = sampling_weights(rows, targets)
    print(
        "NANO_SAMPLING",
        json.dumps(
            dict(
                counts=counts,
                source_mass={
                    group: float(
                        sum(
                            w
                            for i, w in zip(valid, weights)
                            if sampling_group(rows[i], targets[i][0]) == group
                        )
                    )
                    for group in counts
                },
            )
        ),
        flush=True,
    )
    model = NanoFall(a.source).cuda()
    optimizer = torch.optim.AdamW(model.falls.parameters(), lr=0.001, weight_decay=0.0001)
    dev = [
        i
        for i, r in enumerate(rows)
        if r["split"] == "development" and r["source"] != "COCO_train2017"
    ]
    history = []
    best = None
    for epoch in range(a.epochs):
        model.train()
        losses = []
        start = time.perf_counter()
        for _ in range(100):
            ids = np.random.choice(valid, 24, p=weights)
            fs = [torch.from_numpy(np.asarray(v[ids], np.float32)).cuda() for v in features]
            cls = torch.from_numpy(np.stack([targets[i][0] for i in ids])).cuda()
            off = torch.from_numpy(np.stack([targets[i][1] for i in ids])).cuda()
            optimizer.zero_grad()
            pred = flatten([head(f) for head, f in zip(model.falls, fs)])
            loss = masked_loss(pred, cls, off)
            if not torch.isfinite(loss):
                raise ValueError("Invalid fall loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.falls.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        row = dict(
            epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.perf_counter() - start
        )
        if (epoch + 1) % 5 == 0:
            metrics, _ = assess(model, features, rows, dev, anchors)
            retained = all(r["nano_tp"] >= r["reference_tp"] for r in metrics.values() if r["gt"])
            key = selection_key(metrics)
            if best is None or key > best["key"]:
                best = dict(key=key, epoch=epoch + 1, metrics=metrics, retained_reference=retained)
                torch.save(
                    dict(falls=model.falls.state_dict(), epoch=epoch + 1), a.out / "fall_head.pt"
                )
            row.update(metrics=metrics, retained_reference=retained)
        history.append(row)
        (a.out / "fall_history.json").write_text(
            json.dumps(dict(epochs=history, best=best), indent=2) + "\n"
        )
        if epoch + 1 in (35, 50):
            for group in optimizer.param_groups:
                group["lr"] *= 0.4
        print("NANO_FALL_EPOCH", json.dumps(row), flush=True)
    model.falls.load_state_dict(torch.load(a.out / "fall_head.pt")["falls"])
    _, candidates = assess(model, features, rows, list(range(len(rows))), anchors)
    for i, row in enumerate(rows):
        row["base_candidates"] = candidates[i]
    info.update(items=rows, fall_head_epoch=best["epoch"])
    (a.cache / "records_with_nano_fall.json").write_text(json.dumps(info, indent=2) + "\n")
    print("NANO_FALL_TRAINED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
