"""冻结摔倒和 ROI 文件，训练轻量非线性物体分支。"""

import cache_env
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from shared_semantic_model import SharedSemantic, flatten_objects, export_shared
from train_shared_heads import assign_objects, object_loss
from fall_lite import anchors_for_shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--region-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(629)
    np.random.seed(629)
    info = json.loads((a.cache / "records.json").read_text())
    assert info["complete"]
    rows = info["items"]
    features = [np.load(a.cache / ("feature%d.npy" % i), mmap_mode="r") for i in range(3)]
    anchors = anchors_for_shape()
    valid = [i for i, r in enumerate(rows) if r["split"] == "train"]
    targets = {i: assign_objects(rows[i], anchors) for i in valid}
    model = SharedSemantic(a.checkpoint, object_neck=True).cuda()
    optimizer = torch.optim.AdamW(model.objects.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    for epoch in range(a.epochs):
        model.train()
        losses = []
        start = time.perf_counter()
        selected = np.random.permutation(valid)
        for offset in range(0, len(selected), 24):
            ids = selected[offset : offset + 24]
            fs = [torch.from_numpy(np.asarray(f[ids], np.float32)).cuda() for f in features]
            target = torch.from_numpy(np.stack([targets[i][0] for i in ids])).cuda()
            off = torch.from_numpy(np.stack([targets[i][1] for i in ids])).cuda()
            pos = torch.from_numpy(np.stack([targets[i][2] for i in ids])).cuda()
            optimizer.zero_grad()
            pred = flatten_objects(model.object_forward(fs))
            loss = object_loss(pred, target, off, pos)
            if not torch.isfinite(loss):
                raise RuntimeError("Invalid object loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.objects.parameters(), 5)
            optimizer.step()
            losses.append(float(loss))
        row = dict(
            epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.perf_counter() - start
        )
        history.append(row)
        (a.out / "object_history.json").write_text(json.dumps(history, indent=2) + "\n")
        if (epoch + 1) % 20 == 0:
            torch.save(
                dict(objects=model.objects.state_dict(), object_neck=True, epoch=epoch + 1),
                a.out / ("object_epoch_%d.pt" % (epoch + 1)),
            )
        if epoch + 1 in (40, 65):
            for group in optimizer.param_groups:
                group["lr"] *= 0.4
        print("OBJECT_NECK_EPOCH", json.dumps(row), flush=True)
    torch.save(
        dict(objects=model.objects.state_dict(), object_neck=True, epoch=a.epochs),
        a.out / "object_head.pt",
    )
    for name in (
        "region_head.pt",
        "region_verifier.npz",
        "roi_calibration.json",
        "roi_history.json",
        "roi_regression.json",
    ):
        shutil.copy2(a.region_model / name, a.out / name)
    export_shared(model, a.out)
    meta = json.loads((a.region_model / "shared_semantic.json").read_text())
    meta.pop("ncnn", None)
    meta.update(
        training_run=a.out.name,
        object_neck=True,
        object_training_index_sha256=info["index_sha256"],
        object_training_scope="COCO subset only; no unknown-background negatives",
        onnx_sha256=hashlib.sha256((a.out / "shared_semantic.onnx").read_bytes()).hexdigest(),
        production_qualified=False,
    )
    (a.out / "shared_semantic.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("SHARED_V2_EXPORTED", a.out, flush=True)


if __name__ == "__main__":
    main()
