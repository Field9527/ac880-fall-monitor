"""扩充倒地、日常活动与服饰训练数据，完整保留新测试人员。"""

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from fall_lite_model import FallLite, flatten_heads, export_model
from train_fall_lite import Images, normalize, loss_fn, evaluate
from select_hardneg import records
from finetune_hard_negatives import scores


class VariedNegatives(Images):
    """Change garment framing while preserving all positive boxes/geometry."""

    def __getitem__(self, index):
        image, cls, offsets, item = super().__getitem__(index)
        row = self.items[index]
        if (
            self.train
            and not row["boxes"]
            and row["source"] in ("user_camera", "Fashion-MNIST")
            and random.random() < 0.8
        ):
            rgb = image.permute(1, 2, 0).numpy()
            h, w = rgb.shape[:2]
            scale = random.uniform(0.40, 1.0)
            ch = max(2, int(h * scale))
            cw = max(2, int(w * scale * random.uniform(0.8, 1.2)))
            cw = min(cw, w)
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            rgb = cv2.resize(
                rgb[top : top + ch, left : left + cw], (w, h), interpolation=cv2.INTER_LINEAR
            )
            if random.random() < 0.35:
                rgb = cv2.GaussianBlur(rgb, (3, 3), random.uniform(0.2, 1.2))
            gamma = random.uniform(0.7, 1.4)
            lut = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
            rgb = cv2.LUT(rgb, lut)
            image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        return image, cls, offsets, item


def measure(rows, delta):
    threshold = 1 / (1 + np.exp(-(np.log(0.7 / 0.3) + delta)))
    tp = fp = gtcount = neg = 0
    for confidence, iou, ngt in rows:
        gtcount += ngt
        keep = np.flatnonzero(confidence >= threshold)
        if not ngt:
            neg += 1
            fp += int(bool(len(keep)))
            continue
        used = set()
        for k in keep:
            match = int(iou[k].argmax())
            if iou[k, match] >= 0.5 and match not in used:
                used.add(match)
                tp += 1
    return dict(tp=tp, fp=fp, gt=gtcount, negative_images=neg, recall=tp / max(1, gtcount))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="fall_lite_v5_public")
    parser.add_argument("--source-dir", default="models/fall_lite_v4_hardneg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--varied-negatives", action="store_true")
    parser.add_argument("--previously-evaluated-test", action="store_true")
    args = parser.parse_args()
    out = root / "models" / args.run_name
    split = json.loads((out / "split.json").read_text())
    items = split["items"]
    audit = json.loads((out / "duplicate_audit.json").read_text())
    if audit["canonical_rgb64_cross_split_matches"] or audit["possible_near_matches"]:
        raise ValueError("Resolve public-data duplicate audit before training")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    trainitems = [v for v in items if v["split"] == "train"]
    original = [v for v in items if v["split"] == "val" and v["source"] == "original_falldataset"]
    public = [v for v in items if v["split"] == "val" and v["source"] == "CAUCAFall_v4"]
    harditems = [v for v in items if v["source"] == "user_camera"]
    train = (VariedNegatives if args.varied_negatives else Images)(trainitems, 192, 320, True)
    oldval = Images(original, 192, 320, False)
    newval = Images(public, 192, 320, False)
    hard = Images(harditems, 192, 320, False)
    loader = lambda data: DataLoader(data, batch_size=24, num_workers=0, pin_memory=True)
    oldloader, newloader = loader(oldval), loader(newval)
    fractions = dict(
        original_positive=0.50,
        original_negative=0.10,
        public_positive=0.12,
        public_negative=0.12,
        camera_negative=0.10,
        fashion_negative=0.05,
        synthetic_negative=0.01,
    )
    counts = {kind: sum(v["kind"] == kind for v in trainitems) for kind in fractions}
    if not all(counts.values()):
        raise ValueError("Missing required training category: " + str(counts))
    sampler = WeightedRandomSampler(
        [fractions[v["kind"]] / counts[v["kind"]] for v in trainitems], 3911, replacement=True
    )
    trainloader = DataLoader(train, batch_size=24, sampler=sampler, num_workers=0, pin_memory=True)
    source = root / args.source_dir / "best.pt"
    baseline = torch.load(source, map_location="cpu")["model"]
    model = FallLite(depth=16).cuda()
    model.load_state_dict(baseline)
    base_old = measure(records(model, oldloader, oldval), 0)
    base_public = measure(records(model, newloader, newval), 0)
    base_hard = scores(model, hard, "cuda")
    report = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        baseline_original=base_old,
        baseline_public=base_public,
        baseline_camera_scores=base_hard,
        sampler=dict(fractions=fractions, counts=counts),
        epochs=[],
    )
    print("BASELINE", json.dumps(report), flush=True)
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.features.parameters(), lr=1e-5),
            dict(
                params=[v for k, v in model.named_parameters() if not k.startswith("features.")],
                lr=1e-4,
            ),
        ],
        weight_decay=1e-4,
    )
    candidates = []
    for epoch in range(args.epochs):
        started = time.monotonic()
        model.train()
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for parameter in model.features.parameters():
            parameter.requires_grad_(epoch >= 1)
        total = 0.0
        batches = 0
        for image, cls, offsets, _ in trainloader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_fn(
                flatten_heads(model(normalize(image, "cuda", True))), cls.cuda(), offsets.cuda()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total += loss.item()
            batches += 1
        old = measure(records(model, oldloader, oldval), 0)
        new = measure(records(model, newloader, newval), 0)
        hs = scores(model, hard, "cuda")
        row = dict(
            epoch=epoch + 1,
            seconds=round(time.monotonic() - started, 2),
            loss=total / batches,
            original=old,
            public=new,
            camera_max=max(hs),
            camera_scores=hs,
        )
        report["epochs"].append(row)
        path = out / ("epoch_%02d.pt" % (epoch + 1))
        torch.save(dict(model=model.state_dict(), epoch=epoch + 1), path)
        row["key"] = old["tp"] + 4 * new["tp"] - 12 * old["fp"] - 4 * new["fp"]
        candidates.append((row["key"], path))
        (out / "training_history.json").write_text(json.dumps(report, indent=2))
        print("EPOCH", json.dumps(row), flush=True)
        if epoch == 4:
            for group in optimizer.param_groups:
                group["lr"] *= 0.4
    # 检查点与校准选择不读取人员 9、10 或服饰测试集。
    best = None
    trials = []
    for _, path in sorted(candidates, reverse=True)[:3]:
        checkpoint = torch.load(path, map_location="cpu")
        for alpha in (1.0, 0.8, 0.65, 0.5, 0.35):
            state = {
                k: (baseline[k] * (1 - alpha) + v * alpha if v.is_floating_point() else v)
                for k, v in checkpoint["model"].items()
            }
            model.load_state_dict(state)
            oldrows = records(model, oldloader, oldval)
            newrows = records(model, newloader, newval)
            hs = np.clip(np.asarray(scores(model, hard, "cuda"), np.float64), 1e-8, 1 - 1e-8)
            negatives = sorted(
                [float(v.max()) if len(v) else 0.0 for v, _, ngt in oldrows if not ngt],
                reverse=True,
            )
            boundary = max(
                0.0,
                float(
                    np.log(max(negatives[2], 1e-8) / max(1 - negatives[2], 1e-8))
                    - np.log(0.7 / 0.3)
                    + 1e-4
                ),
            )
            for delta in sorted(
                set((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, boundary, boundary + 0.03))
            ):
                old, new = measure(oldrows, delta), measure(newrows, delta)
                hardmax = float((1 / (1 + np.exp(-(np.log(hs / (1 - hs)) - delta)))).max())
                qualified = (
                    old["tp"] >= base_old["tp"]
                    and old["fp"] <= base_old["fp"]
                    and new["tp"] >= base_public["tp"]
                    and new["fp"] <= base_public["fp"]
                    and hardmax < 0.5
                )
                row = dict(
                    epoch=checkpoint["epoch"],
                    alpha=alpha,
                    logit_offset=delta,
                    original=old,
                    public=new,
                    camera_max=hardmax,
                    qualified=qualified,
                )
                trials.append(row)
                key = (new["tp"] - 2 * new["fp"], old["tp"], -old["fp"], -hardmax)
                if qualified and (best is None or key > best["key"]):
                    best = dict(row, key=key)
                    selected = {k: v.clone() for k, v in state.items()}
                    for head in range(3):
                        selected["heads.%d.1.bias" % head][::5] -= delta
                    torch.save(
                        dict(
                            model=selected,
                            depth=16,
                            height=192,
                            width=320,
                            epoch=checkpoint["epoch"],
                            selection=best,
                        ),
                        out / "best.pt",
                    )
            print("SELECT", path.name, alpha, "best=", json.dumps(best), flush=True)
    (out / "selection.json").write_text(json.dumps(dict(best=best, trials=trials), indent=2))
    if best is None:
        raise RuntimeError(
            "No v5 candidate both preserves original validation and repairs camera negatives"
        )
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model"])
    original_metrics = evaluate(model, oldloader, oldval, "cuda")
    selected = next(v for v in original_metrics if v["threshold"] == 0.7)
    hard_scores = scores(model, hard, "cuda")
    if (
        selected["true_positive_boxes"] < base_old["tp"]
        or selected["false_positive_negative_images"] > base_old["fp"]
        or max(hard_scores) >= 0.5
    ):
        raise RuntimeError("Final calibrated checkpoint verification failed")
    public_metrics = evaluate(model, newloader, newval, "cuda")
    # 冻结检查点选择后，仅执行一次最终评估。
    testitems = [v for v in items if v["split"] == "test" and v["source"] == "CAUCAFall_v4"]
    testdata = Images(testitems, 192, 320, False)
    test_metrics = evaluate(model, loader(testdata), testdata, "cuda")
    fashiontest = Images(
        [v for v in items if v["split"] == "test" and v["source"] == "Fashion-MNIST"],
        192,
        320,
        False,
    )
    fashion_scores = scores(model, fashiontest, "cuda")
    final = dict(
        public_test=test_metrics,
        fashion_test=dict(
            images=len(fashion_scores),
            false_positive_images=sum(v >= 0.7 for v in fashion_scores),
            max_score=max(fashion_scores),
        ),
        camera_scores=hard_scores,
        test_policy=(
            "Repeated evaluation of previously evaluated v5 test subjects/products; regression, not a fresh independent holdout"
            if args.previously_evaluated_test
            else "One final evaluation, no checkpoint selection on test subjects/products"
        ),
    )
    (out / "final_evaluation.json").write_text(json.dumps(final, indent=2))
    export_model(model, out / "fall_lite.onnx")
    meta = json.loads((root / args.source_dir / "fall_lite.json").read_text())
    meta.update(
        training_run=args.run_name,
        epoch=best["epoch"],
        recommended_threshold=0.7,
        onnx_sha256=hashlib.sha256((out / "fall_lite.onnx").read_bytes()).hexdigest(),
        validation=original_metrics,
        public_validation=public_metrics,
        public_test=test_metrics,
        selection=best,
        hard_negative_training_scores=hard_scores,
        runtimes={},
        training_data=dict(
            split_sha256=hashlib.sha256((out / "split.json").read_bytes()).hexdigest(),
            counts=split["counts"],
            sources=[
                "Existing FallDataset",
                "CAUCAFall v4 CC BY 4.0",
                "Fashion-MNIST MIT (auxiliary negatives)",
                "Reviewed OV5640 camera regression scenes",
            ],
            camera_capture_group_count=len(set(v["group"] for v in harditems)),
            negative_framing_augmentation=args.varied_negatives,
        ),
    )
    meta["limitations"] += split["limitations"]
    if args.previously_evaluated_test:
        meta["limitations"].append(final["test_policy"])
    (out / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("V5_EXPORTED", json.dumps(dict(selection=best, final=final)), flush=True)


if __name__ == "__main__":
    main()
