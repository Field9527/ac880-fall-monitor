"""为 NanoDet 共享特征训练小型 ROI 复核器。"""

import cache_env
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from shared_semantic_model import RegionHead
from train_shared_heads import prepare_rois, roi_metrics, all_probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(632)
    info = json.loads((a.cache / "records.json").read_text())
    records = info["items"]
    features = [np.load(a.cache / ("feature%d.npy" % i), mmap_mode="r") for i in range(3)]
    vectors, rows = prepare_rois(a.cache, records, features)
    labels = np.array([r["label"] for r in rows])
    known = np.array([r["split"] == "train" and r["label"] >= 0 for r in rows])
    mean = np.asarray(vectors[known]).mean(0)
    std = np.maximum(np.asarray(vectors[known]).std(0), 0.1)
    x = torch.from_numpy((np.asarray(vectors) - mean) / std).cuda()
    y = torch.from_numpy(labels).long().cuda()
    counts = Counter(labels[known])
    if set(counts) != set(range(6)):
        raise ValueError("Missing ROI class")
    fractions = (0.4, 0.15, 0.18, 0.12, 0.1, 0.05)
    weights = np.zeros(len(rows), np.float64)
    for i, r in enumerate(rows):
        if known[i]:
            weights[i] = (
                fractions[r["label"]]
                / counts[r["label"]]
                * (5 if r["source"] == "user_camera" else 1)
            )
    weights = torch.from_numpy(weights).cuda()
    head = RegionHead(2880).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    best = None
    for epoch in range(a.epochs):
        head.train()
        losses = []
        for ids in torch.multinomial(weights, 128 * 100, replacement=True).split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(head(x[ids]), y[ids], label_smoothing=0.01)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        probabilities = all_probabilities(head, x)
        metrics = roi_metrics(probabilities, rows, records, aggregate=True)
        retained = all(r["kept_tp"] >= r["base_tp"] for r in metrics.values() if r["gt"])
        tp = sum(r["kept_tp"] for r in metrics.values())
        fp = sum(r["kept_fp_images"] for r in metrics.values())
        key = (int(retained), tp, -fp, -float(np.mean(losses)))
        if best is None or key > best["key"]:
            best = dict(key=key, epoch=epoch + 1, metrics=metrics, retained_reference=retained)
            torch.save(
                dict(state=head.state_dict(), mean=mean, std=std), a.model / "region_head.pt"
            )
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), metrics=metrics)
        history.append(row)
        (a.model / "roi_history.json").write_text(
            json.dumps(dict(epochs=history, best=best), indent=2) + "\n"
        )
        print("NANO_ROI_EPOCH", json.dumps(row), flush=True)
    chosen = torch.load(a.model / "region_head.pt", map_location="cpu")
    head.cpu().load_state_dict(chosen["state"])
    np.savez(
        a.model / "region_verifier.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        w1=head.layers[0].weight.detach().numpy(),
        b1=head.layers[0].bias.detach().numpy(),
        w2=head.layers[3].weight.detach().numpy(),
        b2=head.layers[3].bias.detach().numpy(),
    )
    print("NANO_ROI_TRAINED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
