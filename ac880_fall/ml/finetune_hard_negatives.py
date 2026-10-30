"""使用已标注服饰和座椅误报微调，并检查正例保留；单一相机场景只作训练，原验证图不参与优化。"""

import cache_env
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_fall_lite import Images, loss_fn, normalize, evaluate
from fall_lite_model import FallLite, flatten_heads, export_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import decode, FallLiteDetector
from video_test import rgb565


@torch.no_grad()
def scores(model, data, device):
    model.eval()
    all_scores = []
    for start in range(0, len(data), 24):
        batch = torch.stack([data[i][0] for i in range(start, min(start + 24, len(data)))])
        pred = flatten_heads(model(normalize(batch, device))).cpu().numpy()
        for p in pred:
            prob = 1 / (1 + np.exp(-np.clip(p[:, 0], -40, 40)))
            all_scores.append(float(prob.max()))
    return all_scores


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "out/false_alarm_speed_20261016"
    out = root / "models/fall_lite_v4_hardneg"
    out.mkdir(exist_ok=True)
    generated = evidence / "negative_images"
    generated.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    original = json.loads((root / "models/fall_lite_v3_fpn/split.json").read_text())
    items = original["items"]
    new = []
    labels = []
    for file in sorted((evidence / "negative_dorm_20261016").glob("*.npy")):
        image = rgb565(np.load(str(file))[::2, ::2])
        path = generated / (file.stem + ".png")
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        item = dict(
            path=str(path.resolve()),
            boxes=[],
            split="train",
            negative_kind="dorm",
            group="one_user_confirmed_dorm_scene",
        )
        new.append(item)
        labels.append(
            dict(
                item,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                label="no_visible_person",
                label_source="User confirmed chair with hanging black clothes; assistant inspected captures",
            )
        )
    rng = np.random.RandomState(94)
    for i in range(128):
        base = int(rng.randint(0, 256))
        image = np.clip(rng.normal(base, rng.uniform(0, 8), (180, 320, 3)), 0, 255).astype(np.uint8)
        if i % 3:
            for _ in range(int(rng.randint(1, 7))):
                x, y = int(rng.randint(0, 300)), int(rng.randint(0, 160))
                cv2.rectangle(
                    image,
                    (x, y),
                    (
                        min(319, x + int(rng.randint(10, 100))),
                        min(179, y + int(rng.randint(5, 60))),
                    ),
                    tuple(int(v) for v in rng.randint(0, 256, 3)),
                    -1,
                )
        path = generated / ("synthetic_%03d.png" % i)
        cv2.imwrite(str(path), image)
        new.append(
            dict(
                path=str(path.resolve()),
                boxes=[],
                split="train",
                negative_kind="synthetic",
                group="synthetic_seed94",
            )
        )
    (out / "negative_labels.json").write_text(
        json.dumps(
            dict(
                scene_count=1,
                items=labels,
                synthetic_count=128,
                synthetic_seed=94,
                scope="Training negatives; same-scene replay is a regression, not an independent field holdout",
            ),
            indent=2,
        )
        + "\n"
    )
    trainitems = [i for i in items if i["split"] == "train"] + new
    train = Images(trainitems, 192, 320, True)
    val = Images([i for i in items if i["split"] == "val"], 192, 320, False)
    hard = Images(new[: len(labels)], 192, 320, False)
    counts = {
        k: sum(
            1
            for i in trainitems
            if ("positive" if i["boxes"] else i.get("negative_kind", "source_negative")) == k
        )
        for k in ("positive", "source_negative", "dorm", "synthetic")
    }
    fractions = dict(positive=0.65, source_negative=0.15, dorm=0.15, synthetic=0.05)
    weights = [
        fractions["positive" if i["boxes"] else i.get("negative_kind", "source_negative")]
        / counts["positive" if i["boxes"] else i.get("negative_kind", "source_negative")]
        for i in trainitems
    ]
    sampler = WeightedRandomSampler(
        weights, len([i for i in items if i["split"] == "train"]), replacement=True
    )
    loader = DataLoader(train, batch_size=24, sampler=sampler, num_workers=0, pin_memory=True)
    valloader = DataLoader(val, batch_size=24, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = FallLite(depth=16).to(device)
    source = root / "models/fall_lite_v3_fpn/best.pt"
    ck = torch.load(source, map_location="cpu")
    model.load_state_dict(ck["model"])
    baseline = next(m for m in evaluate(model, valloader, val, device) if m["threshold"] == 0.7)
    baseline_hard = scores(model, hard, device)
    print("BASELINE", json.dumps(dict(validation=baseline, hard_scores=baseline_hard)), flush=True)
    history = []
    best = None
    candidate = None
    baseline_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.features.parameters(), lr=0.00001),
            dict(
                params=[v for n, v in model.named_parameters() if not n.startswith("features.")],
                lr=0.00015,
            ),
        ],
        weight_decay=0.0001,
    )
    for epoch in range(10):
        started = time.monotonic()
        model.train()
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for v in model.features.parameters():
            v.requires_grad_(epoch >= 1)
        total = 0.0
        n = 0
        for image, cls, offsets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            image = normalize(image, device, True)
            loss, _, _ = loss_fn(flatten_heads(model(image)), cls.to(device), offsets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total += loss.item()
            n += 1
        metrics = evaluate(model, valloader, val, device)
        selected = next(m for m in metrics if m["threshold"] == 0.7)
        hard_scores = scores(model, hard, device)
        row = dict(
            epoch=epoch + 1,
            seconds=round(time.monotonic() - started, 2),
            loss=total / n,
            validation=selected,
            hard_scores=hard_scores,
        )
        history.append(row)
        # 保留原工作阈值，不允许验证召回下降。
        # 不通过抬高阈值掩盖误检。
        qualifies = (
            selected["true_positive_boxes"] >= baseline["true_positive_boxes"]
            and selected["false_positive_negative_images"]
            <= baseline["false_positive_negative_images"]
            and max(hard_scores) < 0.5
        )
        key = selected["true_positive_boxes"] - 8 * selected["false_positive_negative_images"]
        if max(hard_scores) < 0.5 and (candidate is None or key > candidate["key"]):
            candidate = dict(key=key, epoch=epoch + 1, validation=selected, hard_scores=hard_scores)
            torch.save(
                dict(
                    model=model.state_dict(),
                    depth=16,
                    height=192,
                    width=320,
                    epoch=epoch + 1,
                    metrics=metrics,
                    selected=selected,
                ),
                out / "candidate.pt",
            )
        if qualifies and (best is None or key > best["key"]):
            best = dict(key=key, epoch=epoch + 1, validation=selected, hard_scores=hard_scores)
            torch.save(
                dict(
                    model=model.state_dict(),
                    depth=16,
                    height=192,
                    width=320,
                    epoch=epoch + 1,
                    metrics=metrics,
                    selected=selected,
                    training_run="fall_lite_v4_hardneg",
                ),
                out / "best.pt",
            )
        print("EPOCH", json.dumps(dict(row, qualifies=qualifies, best=best)), flush=True)
        (out / "history.json").write_text(
            json.dumps(
                dict(
                    baseline_validation=baseline,
                    baseline_hard_scores=baseline_hard,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    history=history,
                    best=best,
                ),
                indent=2,
            )
        )
        if epoch == 5:
            for group in optimizer.param_groups:
                group["lr"] *= 0.4
    # 与原检查点插值检查，保留已确认行为。
    # 验证图不进入优化器或训练加载器。
    ck_candidate = torch.load(out / "candidate.pt", map_location="cpu")
    blends = []
    for alpha in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8):
        state = {
            k: (baseline_state[k] * (1 - alpha) + v * alpha if v.is_floating_point() else v)
            for k, v in ck_candidate["model"].items()
        }
        model.load_state_dict(state)
        metrics = evaluate(model, valloader, val, device)
        selected = next(m for m in metrics if m["threshold"] == 0.7)
        hard_scores = scores(model, hard, device)
        qualifies = (
            selected["true_positive_boxes"] >= baseline["true_positive_boxes"]
            and selected["false_positive_negative_images"]
            <= baseline["false_positive_negative_images"]
            and max(hard_scores) < 0.5
        )
        key = selected["true_positive_boxes"] - 8 * selected["false_positive_negative_images"]
        row = dict(alpha=alpha, validation=selected, hard_scores=hard_scores, qualifies=qualifies)
        blends.append(row)
        print("BLEND", json.dumps(row), flush=True)
        if qualifies and (best is None or key > best["key"]):
            best = dict(
                key=key,
                epoch=ck_candidate["epoch"],
                interpolation_alpha=alpha,
                validation=selected,
                hard_scores=hard_scores,
            )
            torch.save(
                dict(
                    model=state,
                    depth=16,
                    height=192,
                    width=320,
                    epoch=ck_candidate["epoch"],
                    metrics=metrics,
                    selected=selected,
                ),
                out / "best.pt",
            )
    (out / "selection.json").write_text(
        json.dumps(dict(candidate=candidate, blends=blends, best=best), indent=2)
    )
    if best is None:
        raise RuntimeError(
            "No candidate preserves positive recall and fixes the hard negatives; original model retained"
        )
    ck = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    export_model(model, out / "fall_lite.onnx")
    meta = json.loads((root / "models/active/fall_lite.json").read_text())
    meta.update(
        training_run="fall_lite_v4_hardneg",
        epoch=best["epoch"],
        recommended_threshold=0.7,
        validation=ck["metrics"],
        onnx_sha256=hashlib.sha256((out / "fall_lite.onnx").read_bytes()).hexdigest(),
        hard_negative_training=dict(
            scene_count=1,
            captured_images=len(labels),
            synthetic_images=128,
            label_manifest_sha256=hashlib.sha256(
                (out / "negative_labels.json").read_bytes()
            ).hexdigest(),
        ),
        hard_negative_regression=best["hard_scores"],
    )
    meta["limitations"].append(
        "User-labelled clothing/chair negatives from one dorm training scene; general-scene acceptance remains pending"
    )
    (out / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("HARD_NEGATIVE_TRAINING_COMPLETE", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
