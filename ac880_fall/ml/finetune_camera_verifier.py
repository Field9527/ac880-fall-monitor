"""使用已复核相机负例训练复核器，同时守住原倒地检出；新采样只作训练，不算现场验收。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from shared_semantic_model import RegionHead
from train_shared_heads import all_probabilities, roi_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import rgb565, letterbox, decode, flatten_heads, anchors_for_shape
from semantic_features import FALL_REGION_OUTPUT_NAMES, region_descriptor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(729)
    rng = np.random.RandomState(729)
    review = json.loads(a.review.read_text())
    manifest = json.loads((a.capture / "capture.json").read_text())
    if review.get("fall_absent") is not True or review.get("split") != "train":
        raise ValueError("Explicit reviewed nonfall training sequence required")
    if set(review["files"]) != {r["file"] for r in manifest["samples"]}:
        raise ValueError("Review must cover the exact captured files")
    net = cv2.dnn.readNetFromONNX(str(a.model / "shared_semantic.onnx"))
    anchors = anchors_for_shape()
    extra = []
    new_rows = []
    for row in manifest["samples"]:
        path = a.capture / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Capture SHA mismatch")
        image = rgb565(np.load(path, allow_pickle=False)[::2, ::2])
        for view in ("original", "mirror", "dim"):
            rgb = (
                image
                if view == "original"
                else (
                    image[:, ::-1].copy()
                    if view == "mirror"
                    else (image.astype(np.float32) * 0.75).astype(np.uint8)
                )
            )
            boxed, _ = letterbox(rgb)
            blob = boxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            blob = (
                blob - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
            ) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
            net.setInput(blob)
            out = net.forward(list(FALL_REGION_OUTPUT_NAMES))
            candidates = decode(flatten_heads(out[:3]), anchors, 0.7, top_k=16)
            for item in candidates:
                x, y, w, h = item["box"]
                box = np.asarray([x, y, x + w, y + h]).clip([0, 0, 0, 0], [320, 192, 320, 192])
                if min(box[2:] - box[:2]) < 1:
                    continue
                extra.append(region_descriptor(out[3:], box))
                new_rows.append(
                    dict(
                        file=row["file"],
                        view=view,
                        box=box.tolist(),
                        score=item["score"],
                        label=5,
                        split="train",
                        provenance="visually reviewed no fallen person; no hand box or pose label inferred",
                    )
                )
    if not extra:
        raise ValueError("No hard candidates captured")
    vectors = np.load(a.cache / "roi_vectors.npy", mmap_mode="r")
    rows = json.loads((a.cache / "roi_records.json").read_text())
    records = json.loads((a.cache / "records.json").read_text())["items"]
    init = np.load(a.model / "region_verifier.npz")
    mean = init["mean"]
    std = init["std"]
    head = RegionHead(len(mean)).cuda()
    with torch.no_grad():
        for layer, k in ((head.layers[0], "1"), (head.layers[3], "2")):
            layer.weight.copy_(torch.from_numpy(init["w" + k]))
            layer.bias.copy_(torch.from_numpy(init["b" + k]))
    x = torch.from_numpy((np.asarray(vectors) - mean) / std).cuda()
    extra_x = torch.from_numpy((np.asarray(extra) - mean) / std).cuda()
    labels = np.asarray([r["label"] for r in rows])
    y = torch.from_numpy(labels).long().cuda()
    known = np.asarray([r["split"] == "train" and r["label"] >= 0 for r in rows])
    weights = np.zeros(len(rows), np.float64)
    for label, fraction in enumerate((0.5, 0.13, 0.13, 0.09, 0.1, 0.05)):
        ids = known & (labels == label)
        weights[ids] = fraction / max(1, ids.sum())
    weights = torch.from_numpy(weights).cuda()
    teacher = RegionHead(len(mean)).cuda()
    teacher.load_state_dict(head.state_dict())
    teacher.eval()
    baseline = roi_metrics(
        all_probabilities(head, x), rows, records, confidence=0.98, aggregate=True
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001, weight_decay=0.0001)
    history = []
    best = None

    def save():
        np.savez(
            a.out / "region_verifier.npz",
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            w1=head.layers[0].weight.detach().cpu().numpy(),
            b1=head.layers[0].bias.detach().cpu().numpy(),
            w2=head.layers[3].weight.detach().cpu().numpy(),
            b2=head.layers[3].bias.detach().cpu().numpy(),
        )

    for epoch in range(15):
        head.train()
        losses = []
        for _ in range(80):
            ids = torch.multinomial(weights, 96, replacement=True)
            new = torch.from_numpy(rng.randint(len(extra), size=32).astype(np.int64)).cuda()
            logits = head(x[ids])
            new_logits = head(extra_x[new])
            target = torch.full((32,), 5, dtype=torch.long, device="cuda")
            with torch.no_grad():
                old = teacher(x[ids]).softmax(1)
            loss = (
                F.cross_entropy(logits, y[ids], label_smoothing=0.005)
                + F.cross_entropy(new_logits, target, label_smoothing=0.005)
                + 0.5 * F.kl_div(logits.log_softmax(1), old, reduction="batchmean")
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        probabilities = all_probabilities(head, x)
        new_p = all_probabilities(head, extra_x)
        metrics = roi_metrics(probabilities, rows, records, confidence=0.98, aggregate=True)
        qualified = all(
            m["kept_tp"] >= baseline[k]["kept_tp"] for k, m in metrics.items() if m["gt"]
        )
        retained_new = int((new_p[:, 0] > 0.02).sum())
        fp = sum(m["kept_fp_images"] for m in metrics.values())
        key = (int(qualified), -retained_new, -fp)
        if best is None or key > best["key"]:
            best = dict(
                key=key,
                epoch=epoch + 1,
                qualified=qualified,
                new_candidates=len(extra),
                new_retained=retained_new,
                metrics=metrics,
            )
            save()
        row = dict(
            epoch=epoch + 1,
            qualified=qualified,
            new_retained=retained_new,
            old_dev_fp=fp,
            loss=float(np.mean(losses)),
        )
        history.append(row)
        print(json.dumps(row), flush=True)
    selected = np.load(a.out / "region_verifier.npz")
    with torch.no_grad():
        for layer, k in ((head.layers[0], "1"), (head.layers[3], "2")):
            layer.weight.copy_(torch.from_numpy(selected["w" + k]))
            layer.bias.copy_(torch.from_numpy(selected["b" + k]))
    final = all_probabilities(head, x)
    new_p = all_probabilities(head, extra_x)
    report = dict(
        best=best,
        history=history,
        regression={
            split: roi_metrics(final, rows, records, split, confidence=0.98, aggregate=True)
            for split in ("development", "test")
        },
        new_training_candidates=len(extra),
        new_training_retained=int((new_p[:, 0] > 0.02).sum()),
        scope="Reviewed camera sequence is training only; no new field acceptance; fall backbone unchanged",
    )
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.out / "new_rows.json").write_text(json.dumps(new_rows, indent=2) + "\n")
    (a.out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print("CAMERA_VERIFIER_TRAINED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
