"""冻结骨干，仅用已标注训练负例细化 ROI 复核；开发选择保留原倒地正例，相机训练重放不算现场验收。"""

import cache_env
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from shared_semantic_model import RegionHead
from train_shared_heads import all_probabilities, roi_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import rgb565, letterbox, decode, flatten_heads, anchors_for_shape
from semantic_features import FALL_REGION_OUTPUT_NAMES, region_descriptor


def main():
    p = argparse.ArgumentParser()
    for name in ("cache", "model", "capture", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--augmentation-cache", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.manual_seed(1029)
    np.random.seed(1029)
    meta = json.loads((a.model / "shared_semantic.json").read_text(encoding="utf8"))
    ceiling = meta["fall_probability_ceiling"]
    records = json.loads((a.cache / "records.json").read_text(encoding="utf8"))["items"]
    rows = json.loads((a.cache / "roi_records.json").read_text(encoding="utf8"))
    init = np.load(a.model / "region_verifier.npz", allow_pickle=False)
    mean, std = init["mean"], init["std"]
    x = torch.from_numpy(
        (np.asarray(np.load(a.cache / "roi_vectors.npy", mmap_mode="r")) - mean) / std
    ).cuda()
    labels = np.asarray([r["label"] for r in rows])
    y = torch.from_numpy(labels).long().cuda()
    head = RegionHead(len(mean)).cuda()
    with torch.no_grad():
        for layer, k in ((head.layers[0], "1"), (head.layers[3], "2")):
            layer.weight.copy_(torch.from_numpy(init["w" + k]))
            layer.bias.copy_(torch.from_numpy(init["b" + k]))
    teacher = copy.deepcopy(head).eval()
    before = all_probabilities(head, x)
    teacher_p = torch.from_numpy(before).cuda()
    baseline = roi_metrics(before, rows, records, confidence=1 - ceiling, aggregate=True)
    review = json.loads((a.capture / "review.json").read_text(encoding="utf8"))
    manifest = json.loads((a.capture / "capture.json").read_text(encoding="utf8"))
    assert review["fall_absent"] is True and review["split"] == "train"
    assert set(review["files"]) == {r["file"] for r in manifest["samples"]}
    net = cv2.dnn.readNetFromONNX(str(a.model / "shared_semantic.onnx"))
    anchors = anchors_for_shape()
    extra = []
    for row in manifest["samples"]:
        path = a.capture / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        image = rgb565(np.load(path, allow_pickle=False)[::2, ::2])
        for rgb in (
            image,
            image[:, ::-1].copy(),
            (image.astype(np.float32) * 0.75).astype(np.uint8),
        ):
            boxed, _ = letterbox(rgb)
            blob = (
                boxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
            ) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
            net.setInput(blob)
            outputs = net.forward(list(FALL_REGION_OUTPUT_NAMES))
            for candidate in decode(flatten_heads(outputs[:3]), anchors, 0.7, top_k=16):
                bx, by, bw, bh = candidate["box"]
                box = np.asarray([bx, by, bx + bw, by + bh]).clip(
                    [0, 0, 0, 0], [320, 192, 320, 192]
                )
                if min(box[2:] - box[:2]) >= 1:
                    extra.append(region_descriptor(outputs[3:], box))
    if not extra:
        raise RuntimeError("Expected reviewed camera training candidates")
    extra_x = torch.from_numpy((np.asarray(extra) - mean) / std).cuda()
    known = np.asarray([r["split"] == "train" and r["label"] >= 0 for r in rows])
    weights = np.zeros(len(rows), np.float64)
    for label, fraction in enumerate((0.45, 0.12, 0.22, 0.08, 0.07, 0.06)):
        ids = known & (labels == label)
        weights[ids] = fraction / max(1, ids.sum())
    # 仅在训练集挖掘高置信度误报和易漏正例。
    weights[known & (labels != 0)] *= 1 + np.minimum(20, before[known & (labels != 0), 0] / ceiling)
    weights[known & (labels == 0)] *= 1 + 4 * (before[known & (labels == 0), 0] < 0.2)
    weights = torch.from_numpy(weights).cuda()
    aug_x = aug_y = aug_weights = None
    aug_positive_floor = 0
    if a.augmentation_cache:
        aug_rows = json.loads((a.augmentation_cache / "rows.json").read_text(encoding="utf8"))
        assert all(r["split"] == "train" and r["label"] >= 0 for r in aug_rows)
        aug_labels = np.asarray([r["label"] for r in aug_rows])
        aug_x = torch.from_numpy(
            (np.load(a.augmentation_cache / "vectors.npy", allow_pickle=False) - mean) / std
        ).cuda()
        aug_y = torch.from_numpy(aug_labels).long().cuda()
        aug_before = all_probabilities(head, aug_x)
        aug_positive_floor = int((aug_before[aug_labels == 0, 0] > ceiling).sum())
        probabilities = np.zeros(len(aug_rows), np.float64)
        for label, fraction in enumerate((0.45, 0.13, 0.22, 0.08, 0.06, 0.06)):
            chosen = aug_labels == label
            probabilities[chosen] = fraction / max(1, chosen.sum())
        probabilities[aug_labels != 0] *= 1 + np.minimum(
            20, aug_before[aug_labels != 0, 0] / ceiling
        )
        aug_weights = torch.from_numpy(probabilities).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.00015, weight_decay=0.0001)

    def cost(metrics):
        return sum(
            v["kept_fp_images"] * (3 if k in ("FreiHAND_research", "user_camera") else 1)
            for k, v in metrics.items()
        )

    initial_aug_fp = (
        int((aug_before[aug_labels != 0, 0] > ceiling).sum()) if aug_x is not None else 0
    )
    best_cost = cost(baseline)
    best_key = (best_cost, initial_aug_fp, float("inf"))
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    history = []
    print("BASELINE", json.dumps(baseline), "extra_training_candidates", len(extra), flush=True)
    for epoch in range(a.epochs):
        head.train()
        losses = []
        for _ in range(100):
            ids = torch.multinomial(weights, 96, replacement=True)
            ex = torch.randint(len(extra_x), (32,), device="cuda")
            logits = head(x[ids])
            extra_logits = head(extra_x[ex])
            negative = y[ids] != 0
            positive = ~negative
            binary = logits[:, 0] - torch.logsumexp(logits[:, 1:], 1)
            loss = F.cross_entropy(logits, y[ids], label_smoothing=0.005)
            if negative.any():
                loss = loss + 0.35 * F.softplus(binary[negative] + 4.6).mean()
            if positive.any():
                loss = loss + 0.1 * F.kl_div(
                    logits[positive].log_softmax(1), teacher_p[ids][positive], reduction="batchmean"
                )
            loss = loss + 0.5 * F.cross_entropy(
                extra_logits,
                torch.full((32,), 5, dtype=torch.long, device="cuda"),
                label_smoothing=0.005,
            )
            if aug_x is not None:
                aug_ids = torch.multinomial(aug_weights, 96, replacement=True)
                aug_logits = head(aug_x[aug_ids])
                loss = loss + 0.8 * F.cross_entropy(
                    aug_logits, aug_y[aug_ids], label_smoothing=0.005
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        probabilities = all_probabilities(head, x)
        metrics = roi_metrics(probabilities, rows, records, confidence=1 - ceiling, aggregate=True)
        camera = all_probabilities(head, extra_x)
        qualified = all(
            v["kept_tp"] >= baseline[k]["kept_tp"] for k, v in metrics.items() if v["gt"]
        )
        qualified = qualified and not (camera[:, 0] > ceiling).any()
        aug_fp = 0
        if aug_x is not None:
            aug_after = all_probabilities(head, aug_x)
            aug_fp = int((aug_after[aug_labels != 0, 0] > ceiling).sum())
            qualified = (
                qualified
                and int((aug_after[aug_labels == 0, 0] > ceiling).sum()) >= aug_positive_floor
            )
        row = dict(
            epoch=epoch + 1,
            qualified=bool(qualified),
            cost=cost(metrics),
            augmented_training_fp=aug_fp,
            loss=float(np.mean(losses)),
            metrics=metrics,
        )
        history.append(row)
        key = (cost(metrics), aug_fp, float(np.mean(losses)))
        if qualified and key < best_key:
            best_state = copy.deepcopy(head.state_dict())
            best_key = key
            best_cost = cost(metrics)
            best_epoch = epoch + 1
        print(
            "EPOCH",
            epoch + 1,
            "qualified",
            bool(qualified),
            "FP_cost",
            cost(metrics),
            "best",
            best_cost,
            flush=True,
        )
    head.load_state_dict(best_state)
    after = all_probabilities(head, x)
    np.savez(
        a.out / "region_verifier.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        w1=head.layers[0].weight.detach().cpu().numpy(),
        b1=head.layers[0].bias.detach().cpu().numpy(),
        w2=head.layers[3].weight.detach().cpu().numpy(),
        b2=head.layers[3].bias.detach().cpu().numpy(),
    )
    for name in ("shared_semantic.onnx", meta["ncnn"]["param"], meta["ncnn"]["weights"]):
        shutil.copy2(a.model / name, a.out / name)
    if meta.get("reference_verifier"):
        shutil.copy2(
            a.model / meta["reference_verifier"]["file"], a.out / meta["reference_verifier"]["file"]
        )
    report = dict(
        baseline=baseline,
        best_epoch=best_epoch,
        candidate_selected=best_epoch > 0,
        improved=best_cost < cost(baseline),
        fall_probability_ceiling=ceiling,
        extra_reviewed_training_candidates=len(extra),
        history=history,
        regression={
            split: roi_metrics(after, rows, records, split, confidence=1 - ceiling, aggregate=True)
            for split in ("train", "development", "test")
        },
        augmentation_cache=str(a.augmentation_cache) if a.augmentation_cache else None,
        reviewed_capture=str(a.capture),
        capture_manifest_sha256=hashlib.sha256(
            (a.capture / "capture.json").read_bytes()
        ).hexdigest(),
        capture_frames=len(manifest["samples"]),
        previous_model=str(a.model),
        scope="Cached labelled training data and old reviewed camera training sequence; camera augmentation uses training split only; no new field acceptance",
    )
    (a.out / "refinement_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    meta.update(
        training_run="v6_regions_balanced_hardneg_20261029",
        production_qualified=False,
        verifier_sha256=hashlib.sha256((a.out / "region_verifier.npz").read_bytes()).hexdigest(),
        refinement=dict(
            best_epoch=best_epoch,
            threshold_unchanged=True,
            backbone_unchanged=True,
            scope=report["scope"],
            reviewed_capture=report["reviewed_capture"],
            capture_frames=report["capture_frames"],
            capture_manifest_sha256=report["capture_manifest_sha256"],
            previous_model=report["previous_model"],
        ),
    )
    (a.out / "shared_semantic.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    print(
        "REFINEMENT_FINISHED",
        json.dumps(
            dict(
                best_epoch=best_epoch,
                candidate_selected=best_epoch > 0,
                improved=report["improved"],
                regression=report["regression"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
