"""使用明确复核的新负例微调原 v6 网络；阈值保持 0.7，既有测试仅作回归。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from fall_lite_model import FallLite, flatten_heads, export_model
from train_fall_lite import Images, normalize, loss_fn, evaluate
from train_public_v5 import VariedNegatives, measure
from select_hardneg import records
from finetune_hard_negatives import scores


class NegativeViews(VariedNegatives):
    def __getitem__(self, i):
        image, cls, offsets, index = super().__getitem__(i)
        row = self.items[i]
        if self.train and row["kind"] in ("hand_negative", "current_negative"):
            rgb = image.permute(1, 2, 0).numpy()
            if random.random() < 0.7:
                h, w = rgb.shape[:2]
                scale = random.uniform(0.55, 1.0)
                ch, cw = int(h * scale), int(w * scale)
                y, x = random.randrange(h - ch + 1), random.randrange(w - cw + 1)
                rgb = cv2.resize(rgb[y : y + ch, x : x + cw], (w, h))
            # 训练增强覆盖适度对比度和增益变化。
            # 不改变正例框，也不增加在线计算。
            gain = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            rgb = np.clip(((rgb.astype(np.float32) - 128) * contrast + 128) * gain, 0, 255).astype(
                np.uint8
            )
            image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        return image, cls, offsets, index


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True)
    p.add_argument("--run-name", default="fall_lite_v7_hands")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--hard-negative-weight", type=float, default=0.0)
    p.add_argument("--distill-weight", type=float, default=0.0)
    a = p.parse_args()
    out = root / "models" / a.run_name
    out.mkdir(exist_ok=False)
    source = root / "models/fall_lite_v6_varied"
    review_path = Path(a.review)
    review = json.loads(review_path.read_text())
    extra = review["items"]
    if not extra:
        raise ValueError("No reviewed negatives")
    for row in extra:
        if (
            row["boxes"]
            or row["label_status"] != "no_visible_fallen_person"
            or not row.get("label_source")
        ):
            raise ValueError("Every added image must have explicit negative review provenance")
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Reviewed image changed")
    split = json.loads((source / "split.json").read_text())
    items = split["items"] + extra
    split.update(
        items=items,
        new_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
        new_data_policy=review["scope"],
    )
    split["counts"] = {}
    for row in items:
        key = (
            row["source"] + "/" + row["split"] + "/" + ("positive" if row["boxes"] else "negative")
        )
        split["counts"][key] = split["counts"].get(key, 0) + 1
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(219)
    np.random.seed(219)
    torch.manual_seed(219)
    trainitems = [r for r in items if r["split"] == "train"]
    original = Images(
        [r for r in items if r["source"] == "original_falldataset" and r["split"] == "val"],
        192,
        320,
        False,
    )
    public = Images(
        [r for r in items if r["source"] == "CAUCAFall_v4" and r["split"] == "val"], 192, 320, False
    )
    camera = Images([r for r in items if r["source"] == "user_camera"], 192, 320, False)
    added = Images(extra, 192, 320, False)
    train = NegativeViews(trainitems, 192, 320, True)
    loader = lambda d: DataLoader(d, batch_size=24, num_workers=0, pin_memory=True)
    fractions = dict(
        original_positive=0.45,
        original_negative=0.07,
        public_positive=0.13,
        public_negative=0.10,
        camera_negative=0.07,
        current_negative=0.06,
        hand_negative=0.07,
        fashion_negative=0.04,
        synthetic_negative=0.01,
    )
    counts = {k: sum(r["kind"] == k for r in trainitems) for k in fractions}
    if not all(counts.values()):
        raise ValueError("Missing training category: " + str(counts))
    sampler = WeightedRandomSampler(
        [fractions[r["kind"]] / counts[r["kind"]] for r in trainitems], 3911, replacement=True
    )
    trainloader = DataLoader(train, batch_size=24, sampler=sampler, num_workers=0, pin_memory=True)
    baseline = torch.load(source / "best.pt", map_location="cpu")["model"]
    model = FallLite(depth=16).cuda()
    model.load_state_dict(baseline)
    teacher = None
    if a.distill_weight:
        teacher = FallLite(depth=16).cuda()
        teacher.load_state_dict(baseline)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    def assess():
        return dict(
            original=measure(records(model, loader(original), original), 0),
            public=measure(records(model, loader(public), public), 0),
            camera_scores=scores(model, camera, "cuda"),
            added_scores=scores(model, added, "cuda"),
        )

    base = assess()
    report = dict(
        baseline=base,
        source_sha256=hashlib.sha256((source / "best.pt").read_bytes()).hexdigest(),
        fractions=fractions,
        counts=counts,
        epochs=[],
        threshold=0.7,
        selection_scope=review["scope"],
        hard_negative_weight=a.hard_negative_weight,
        distill_weight=a.distill_weight,
    )
    print("BASELINE", json.dumps(base), flush=True)
    rate = 1e-5 if a.hard_negative_weight else 3e-6
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.features.parameters(), lr=rate),
            dict(
                params=[v for k, v in model.named_parameters() if not k.startswith("features.")],
                lr=rate * 10,
            ),
        ],
        weight_decay=1e-4,
    )
    best = None
    for epoch in range(a.epochs):
        start = time.monotonic()
        model.train()
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for parameter in model.features.parameters():
            parameter.requires_grad_(epoch >= 1)
        losses = []
        for image, cls, offsets, ids in trainloader:
            optimizer.zero_grad(set_to_none=True)
            x = normalize(image, "cuda", True)
            pred = flatten_heads(model(x))
            loss, _, _ = loss_fn(pred, cls.cuda(), offsets.cuda())
            if a.hard_negative_weight:
                mask = torch.tensor(
                    [
                        trainitems[i]["kind"] in ("hand_negative", "current_negative")
                        for i in ids.tolist()
                    ],
                    device="cuda",
                )
                if mask.any():
                    logits = pred[mask, :, 0].topk(20, dim=1).values
                    loss += a.hard_negative_weight * F.softplus(logits).mean()
            if teacher is not None:
                positive = (cls > 0).any(1).cuda()
                if positive.any():
                    with torch.no_grad():
                        target = flatten_heads(teacher(x[positive]))
                    keep = target[:, :, 0] > 0.0
                    if keep.any():
                        loss += a.distill_weight * F.mse_loss(pred[positive][keep], target[keep])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            losses.append(loss.item())
        result = assess()
        qualified = (
            all(
                result[k]["tp"] >= base[k]["tp"] and result[k]["fp"] <= base[k]["fp"]
                for k in ("original", "public")
            )
            and max(result["camera_scores"]) < 0.5
            and max(result["added_scores"]) < 0.5
        )
        row = dict(
            epoch=epoch + 1,
            seconds=time.monotonic() - start,
            loss=sum(losses) / len(losses),
            qualified=qualified,
            **result,
        )
        key = (result["original"]["tp"], result["public"]["tp"], -max(result["added_scores"]))
        torch.save(
            dict(model=model.state_dict(), depth=16, height=192, width=320, epoch=epoch + 1),
            out / ("epoch_%02d.pt" % (epoch + 1)),
        )
        if qualified and (best is None or key > best["key"]):
            best = dict(key=key, **row)
            torch.save(
                dict(model=model.state_dict(), depth=16, height=192, width=320, epoch=epoch + 1),
                out / "best.pt",
            )
        report["epochs"].append(row)
        report["best"] = best
        (out / "training_history.json").write_text(json.dumps(report, indent=2) + "\n")
        print("EPOCH", json.dumps(row), flush=True)
        if epoch == a.epochs * 2 // 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.4
    if best is None:
        raise RuntimeError(
            "No candidate passes positive-retention and negative checks; v6 remains active"
        )
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model"])
    original_metrics = evaluate(model, loader(original), original, "cuda")
    public_metrics = evaluate(model, loader(public), public, "cuda")
    test = Images(
        [r for r in items if r["source"] == "CAUCAFall_v4" and r["split"] == "test"],
        192,
        320,
        False,
    )
    test_metrics = evaluate(model, loader(test), test, "cuda")
    (out / "regression.json").write_text(
        json.dumps(
            dict(
                original=original_metrics,
                public=public_metrics,
                test=test_metrics,
                test_scope="Previously evaluated CAUCA subjects 9/10; repeated regression, not independent field acceptance",
            ),
            indent=2,
        )
        + "\n"
    )
    export_model(model, out / "fall_lite.onnx")
    meta = json.loads((source / "fall_lite.json").read_text())
    meta.update(
        training_run=a.run_name,
        epoch=best["epoch"],
        recommended_threshold=0.7,
        validation=original_metrics,
        public_validation=public_metrics,
        public_test=test_metrics,
        onnx_sha256=hashlib.sha256((out / "fall_lite.onnx").read_bytes()).hexdigest(),
        runtimes={},
        negative_review_sha256=split["new_review_sha256"],
        negative_review_scope=review["scope"],
        selection=best,
    )
    meta["training_data"] = dict(
        split_sha256=hashlib.sha256((out / "split.json").read_bytes()).hexdigest(),
        counts=split["counts"],
        added_reviewed_negatives=len(extra),
        extra_source_terms="FreiHAND ICCV 2019: research only, non-commercial, cite authors",
    )
    (out / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("QUALIFIED", out, flush=True)


if __name__ == "__main__":
    main()
