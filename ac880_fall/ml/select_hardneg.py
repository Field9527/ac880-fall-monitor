"""结合已记录的分数校准选择权重；原验证标签只作选择，不参与优化，单一场景不算泛化测试。"""

import cache_env
import hashlib, json
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from fall_lite_model import FallLite, export_model, flatten_heads
from train_fall_lite import Images, normalize, evaluate
from finetune_hard_negatives import scores
from fall_lite import decode


@torch.no_grad()
def records(model, loader, data):
    model.eval()
    rows = []
    for images, _, _, ids in loader:
        prediction = flatten_heads(model(normalize(images, "cuda"))).cpu().numpy()
        for p, i in zip(prediction, ids.tolist()):
            gt = data.cache[i][3]
            items = decode(p, data.anchors.numpy(), 0.3)
            b = np.asarray([d["box"] for d in items], np.float32).reshape(-1, 4)
            if len(b):
                b[:, 2:] += b[:, :2]
            iou = box_iou(torch.from_numpy(b), torch.from_numpy(gt.astype(np.float32))).numpy()
            rows.append((np.asarray([d["score"] for d in items]), iou, len(gt)))
    return rows


def metric(rows, delta):
    tp = fp = 0
    for score, iou, ngt in rows:
        keep = np.flatnonzero(score >= 1 / (1 + np.exp(-(np.log(0.7 / 0.3) + delta))))
        if not ngt:
            fp += bool(len(keep))
            continue
        used = set()
        for k in keep:
            j = int(iou[k].argmax())
            if iou[k, j] >= 0.5 and j not in used:
                used.add(j)
                tp += 1
    return dict(true_positive_boxes=tp, false_positive_negative_images=fp, recall=tp / 698)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "models/fall_lite_v4_hardneg"
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    source = torch.load(root / "models/fall_lite_v3_fpn/best.pt", map_location="cpu")["model"]
    ck = torch.load(out / "candidate.pt", map_location="cpu")
    model = FallLite(depth=16).cuda()
    split = json.loads((root / "models/fall_lite_v3_fpn/split.json").read_text())["items"]
    val = Images([i for i in split if i["split"] == "val"], 192, 320, False)
    harditems = json.loads((out / "negative_labels.json").read_text())["items"]
    hard = Images(harditems, 192, 320, False)
    loader = DataLoader(val, batch_size=24, num_workers=0, pin_memory=True)
    best = None
    trials = []
    for alpha in (1.0, 0.8, 0.65, 0.6, 0.55, 0.5):
        state = {
            k: (source[k] * (1 - alpha) + v * alpha if v.is_floating_point() else v)
            for k, v in ck["model"].items()
        }
        model.load_state_dict(state)
        rows = records(model, loader, val)
        hs = scores(model, hard, "cuda")
        negatives = sorted(
            [float(v.max()) if len(v) else 0.0 for v, _, ngt in rows if not ngt], reverse=True
        )
        third = negatives[2]
        boundary = max(
            0.0, float(np.log(max(third, 1e-9) / max(1 - third, 1e-9)) - np.log(0.7 / 0.3) + 1e-5)
        )
        for delta in sorted(
            set((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, boundary, boundary + 0.02))
        ):
            m = metric(rows, delta)
            calibrated = [float(1 / (1 + np.exp(-(np.log(s / (1 - s)) - delta)))) for s in hs]
            ok = (
                m["true_positive_boxes"] >= 546
                and m["false_positive_negative_images"] <= 2
                and max(calibrated) < 0.5
            )
            row = dict(
                alpha=alpha, logit_offset=delta, metric=m, hard_max=max(calibrated), qualified=ok
            )
            trials.append(row)
            if ok and (
                best is None
                or (m["true_positive_boxes"], -m["false_positive_negative_images"])
                > (
                    best["metric"]["true_positive_boxes"],
                    -best["metric"]["false_positive_negative_images"],
                )
            ):
                best = row
                selected = {k: v.clone() for k, v in state.items()}
                for head in range(3):
                    selected["heads.%d.1.bias" % head][::5] -= delta
                torch.save(
                    dict(
                        model=selected,
                        depth=16,
                        height=192,
                        width=320,
                        epoch=ck["epoch"],
                        calibration=row,
                    ),
                    out / "best.pt",
                )
        print("SELECTION", json.dumps(dict(alpha=alpha, best=best)), flush=True)
    (out / "calibration_selection.json").write_text(
        json.dumps(dict(trials=trials, best=best), indent=2)
    )
    if best is None:
        raise RuntimeError("No repaired candidate preserves validation recall/negatives")
    chosen = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(chosen["model"])
    metrics = evaluate(model, loader, val, "cuda")
    hard_scores = scores(model, hard, "cuda")
    export_model(model, out / "fall_lite.onnx")
    meta = json.loads((root / "models/active/fall_lite.json").read_text())
    meta.update(
        training_run="fall_lite_v4_hardneg",
        epoch=ck["epoch"],
        recommended_threshold=0.7,
        onnx_sha256=hashlib.sha256((out / "fall_lite.onnx").read_bytes()).hexdigest(),
        validation=metrics,
        hard_negative_training=dict(
            scene_count=1,
            captured_images=len(harditems),
            synthetic_images=128,
            label_manifest_sha256=hashlib.sha256(
                (out / "negative_labels.json").read_bytes()
            ).hexdigest(),
        ),
        selection=best,
        hard_negative_training_scores=hard_scores,
    )
    meta["limitations"].append(
        "One user-labelled dorm training scene; classification logits calibrated using original selected validation set; not field generalization acceptance"
    )
    (out / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        "REPAIRED_MODEL_SELECTED", json.dumps(dict(best=best, hard_scores=hard_scores)), flush=True
    )


if __name__ == "__main__":
    main()
