"""按已记录的去重划分训练 SSD 倒地框模型；图像标签不等于连续跌倒事件真值。沿用教师模型 seed=0、val_ratio=0.15 的划分。"""

import argparse
import cache_env  # 导入模型库前配置临时、编译和下载缓存。
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import anchors_for_shape, letterbox, decode
from fall_lite_model import FallLite, flatten_heads, export_model


def read_item(path):
    xml = path.parent.parent / "xml" / (path.stem + ".xml")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Unreadable image " + str(path))
    h, w = image.shape[:2]
    boxes = []
    if xml.exists():
        root = ET.parse(xml).getroot()
        for obj in root.findall("object"):
            if (obj.findtext("name") or "").strip().lower() != "fall":
                continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes.append([x1, y1, x2, y2])
    return dict(
        path=str(path.resolve()),
        md5=hashlib.md5(path.read_bytes()).hexdigest(),
        boxes=boxes,
        has_xml=xml.exists(),
    )


def split_index(root, output, review_path):
    paths = sorted((root / "img").glob("*.jpg")) + sorted((root / "img").glob("*.png"))
    with ThreadPoolExecutor(8) as pool:
        items = list(pool.map(read_item, paths))
    groups = {}
    for item in items:
        groups.setdefault(item["md5"], []).append(item)
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    valkeys = set(keys[: max(1, int(len(keys) * 0.15))])
    for item in items:
        item["split"] = "val" if item["md5"] in valkeys else "train"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    reviewed = {v["filename"]: v for v in review["items"]}
    selected = []
    excluded = 0
    for item in items:
        if item["has_xml"]:
            item["label_status"] = "source_xml"
            selected.append(item)
        elif Path(item["path"]).name in reviewed:
            label = reviewed[Path(item["path"]).name]
            if label["label"] != "no_visible_fallen_person":
                raise ValueError("Invalid negative review label")
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != label["sha256"]:
                raise ValueError("Review hash mismatch")
            if item["split"] != label["split"]:
                raise ValueError("Review split mismatch")
            item["label_status"] = "assistant_visual_review_negative"
            selected.append(item)
        else:
            excluded += 1
    report = dict(
        seed=0,
        val_ratio=0.15,
        grouping="exact image MD5; near duplicates and shared scenes remain possible",
        missing_xml_policy="excluded unless explicitly visually reviewed as negative",
        negative_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
        excluded_unlabelled=excluded,
        items=selected,
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return selected


def xyxy(anchors):
    return torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)


def assign(boxes, anchors):
    cls = torch.zeros(len(anchors), dtype=torch.float32)
    offsets = torch.zeros(len(anchors), 4)
    if not len(boxes):
        return cls, offsets
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    ious = box_iou(xyxy(anchors), boxes)
    best, matched = ious.max(1)
    cls[(best >= 0.25) & (best < 0.45)] = -1
    cls[best >= 0.45] = 1
    forced = ious.argmax(0)
    cls[forced] = 1
    matched[forced] = torch.arange(len(boxes))
    target = boxes[matched]
    centers = (target[:, :2] + target[:, 2:]) / 2
    wh = (target[:, 2:] - target[:, :2]).clamp(min=1)
    offsets[:, :2] = (centers - anchors[:, :2]) / anchors[:, 2:] * 10
    offsets[:, 2:] = torch.log(wh / anchors[:, 2:]) * 5
    return cls, offsets


class Images(Dataset):
    def __init__(self, items, height, width, train):
        self.items = items
        self.train = train
        self.height = height
        self.width = width
        self.anchors = torch.from_numpy(anchors_for_shape(height, width))
        self.cache = []
        for item in items:
            rgb = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
            image, (sx, sy, ox, oy) = letterbox(rgb, height, width)
            boxes = np.array(item["boxes"], np.float32).reshape(-1, 4)
            boxes = boxes * np.array([sx, sy, sx, sy]) + np.array([ox, oy, ox, oy])
            flipped = boxes.copy()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            self.cache.append(
                (image, assign(boxes, self.anchors), assign(flipped, self.anchors), boxes)
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, normal, flipped, boxes = self.cache[i]
        flip = self.train and random.random() < 0.5
        image = np.ascontiguousarray(image[:, ::-1] if flip else image)
        cls, offsets = flipped if flip else normal
        return torch.from_numpy(image).permute(2, 0, 1), cls, offsets, i


def loss_fn(pred, cls, offsets):
    positive = cls > 0
    bce = F.binary_cross_entropy_with_logits(pred[:, :, 0], positive.float(), reduction="none")
    neg = bce.detach().masked_fill(cls != 0, -1)
    ranks = neg.argsort(1, descending=True).argsort(1)
    nneg = torch.maximum(positive.sum(1) * 3, torch.full_like(positive.sum(1), 20))
    selected = positive | ((cls == 0) & (ranks < nneg[:, None]))
    denom = positive.sum().clamp(min=1)
    classification = bce[selected].sum() / denom
    regression = (
        F.smooth_l1_loss(pred[:, :, 1:][positive], offsets[positive], reduction="sum") / denom
    )
    return classification + regression, classification.detach(), regression.detach()


def normalize(x, device, augment=False):
    x = x.to(device, non_blocking=True).float() / 255.0
    if augment:
        strength = torch.rand(len(x), 1, 1, 1, device=device) * 0.4 + 0.8
        x = (x * strength).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    return (x - mean) / std


@torch.no_grad()
def evaluate(model, loader, data, device):
    model.eval()
    records = []
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    for images, _, _, ids in loader:
        pred = flatten_heads(model(normalize(images, device))).cpu().numpy()
        for p, i in zip(pred, ids.tolist()):
            gt = data.cache[i][3]
            det = decode(p, data.anchors.numpy(), 0.3)
            boxes = np.array([d["box"] for d in det], np.float32).reshape(-1, 4)
            if len(boxes):
                boxes[:, 2:] += boxes[:, :2]
            scores = np.array([d["score"] for d in det])
            overlaps = box_iou(
                torch.from_numpy(boxes), torch.from_numpy(gt.astype(np.float32))
            ).numpy()
            records.append((scores, overlaps, len(gt)))
    metrics = []
    for t in thresholds:
        tp = gtcount = fp_images = negcount = matched_images = poscount = 0
        for scores, ious, ngt in records:
            keep = np.flatnonzero(scores >= t)
            gtcount += ngt
            if ngt:
                poscount += 1
                used = set()
                any_match = False
                for k in keep:
                    j = int(ious[k].argmax())
                    if ious[k, j] >= 0.5 and j not in used:
                        tp += 1
                        used.add(j)
                        any_match = True
                matched_images += int(any_match)
            else:
                negcount += 1
                fp_images += int(len(keep) > 0)
        metrics.append(
            dict(
                threshold=t,
                box_recall_iou50=tp / max(1, gtcount),
                positive_image_recall=matched_images / max(1, poscount),
                negative_image_fp_rate=fp_images / max(1, negcount),
                true_positive_boxes=tp,
                gt_boxes=gtcount,
                positive_images=poscount,
                negative_images=negcount,
                false_positive_negative_images=fp_images,
            )
        )
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument(
        "--review",
        type=Path,
        required=True,
        help="Explicit reviewed negatives; missing XML alone is never a negative label",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--depth", type=int, choices=(13, 16), default=13)
    p.add_argument(
        "--resume", type=Path, help="继续同一数据划分下的检查点；--epochs 为本轮追加轮数"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(16)
    np.random.seed(16)
    torch.manual_seed(16)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device, flush=True)
    items = split_index(a.dataset, a.out / "split.json", a.review)
    train = Images([i for i in items if i["split"] == "train"], a.height, a.width, True)
    val = Images([i for i in items if i["split"] == "val"], a.height, a.width, False)
    print("DATA", len(train), len(val), "cached", flush=True)
    positive = sum(bool(i["boxes"]) for i in train.items)
    negative = len(train) - positive
    if not negative:
        raise ValueError("No reviewed negative training images")
    weights = [0.25 / negative if not i["boxes"] else 0.75 / positive for i in train.items]
    sampler = WeightedRandomSampler(weights, len(train), replacement=True)
    trainloader = DataLoader(
        train, batch_size=a.batch, sampler=sampler, num_workers=0, pin_memory=True
    )
    valloader = DataLoader(val, batch_size=a.batch, num_workers=0, pin_memory=True)
    model = FallLite(a.teacher, a.depth).to(device)
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.features.parameters(), lr=0.0001),
            dict(
                params=[v for n, v in model.named_parameters() if not n.startswith("features.")],
                lr=0.001,
            ),
        ],
        weight_decay=0.0001,
    )
    history = []
    best = -float("inf")
    start_epoch = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        if (ck["height"], ck["width"]) != (a.height, a.width):
            raise ValueError("Resume input size mismatch")
        if ck.get("depth", 13) != a.depth:
            raise ValueError("Resume backbone depth mismatch")
        model.load_state_dict(ck["model"])
        start_epoch = ck["epoch"]
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if (a.out / "history.json").exists():
            history = json.loads((a.out / "history.json").read_text())
        if (a.out / "best.pt").exists():
            b = torch.load(a.out / "best.pt", map_location="cpu")["selected"]
            best = b["box_recall_iou50"] - b["negative_image_fp_rate"]
    for epoch in range(start_epoch, start_epoch + a.epochs):
        start = time.monotonic()
        model.train()
        # 小样本微调时保留教师模型的有效统计量。
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for v in model.features.parameters():
            v.requires_grad_(epoch >= 2)
        sums = np.zeros(3)
        count = 0
        for images, cls, offsets, _ in trainloader:
            images = normalize(images, device, True)
            optimizer.zero_grad(set_to_none=True)
            loss, cl, reg = loss_fn(
                flatten_heads(model(images)), cls.to(device), offsets.to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            sums += np.array([loss.item(), cl.item(), reg.item()])
            count += 1
        metrics = evaluate(model, valloader, val, device)
        selected = max(metrics, key=lambda m: m["box_recall_iou50"] - m["negative_image_fp_rate"])
        score = selected["box_recall_iou50"] - selected["negative_image_fp_rate"]
        row = dict(
            epoch=epoch + 1,
            loss=(sums / count).tolist(),
            seconds=round(time.monotonic() - start, 1),
            metrics=metrics,
            selected=selected,
        )
        history.append(row)
        (a.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        ck = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            depth=a.depth,
            height=a.height,
            width=a.width,
            epoch=epoch + 1,
            metrics=metrics,
            teacher_sha256=hashlib.sha256(Path(a.teacher).read_bytes()).hexdigest(),
            selected=selected,
        )
        torch.save(ck, a.out / "last.pt")
        if score > best:
            best = score
            torch.save(ck, a.out / "best.pt")
        print(
            json.dumps(
                dict(
                    epoch=epoch + 1,
                    loss=row["loss"],
                    seconds=row["seconds"],
                    selected=selected,
                    best=round(best, 4),
                )
            ),
            flush=True,
        )
        if epoch - start_epoch == a.epochs * 2 // 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.3
    ck = torch.load(a.out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    export_model(model, a.out / "fall_lite.onnx", a.height, a.width)
    architecture = (
        "fall_lite_mnv3_depth16_fpn_ssd_v1" if a.depth == 16 else "fall_lite_mnv3_depth13_ssd_v1"
    )
    meta = dict(
        architecture=architecture,
        trained=True,
        input_width=a.width,
        input_height=a.height,
        normalization="RGB /255 then ImageNet mean/std; letterbox fill=114",
        epoch=ck["epoch"],
        recommended_threshold=ck["selected"]["threshold"],
        validation=ck["metrics"],
        teacher_sha256=ck["teacher_sha256"],
        onnx_sha256=hashlib.sha256((a.out / "fall_lite.onnx").read_bytes()).hexdigest(),
        label_policy="missing XML excluded unless reviewed; assistant-reviewed negatives, not independent human ground truth",
        label_policy_validated=True,
        negative_review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),
        limitations=[
            "image holdout only; source/scene overlap may remain",
            "not a validated temporal fall-event model",
            "small selected negative set; labels not independently human-audited",
            "no general-scene acceptance yet",
        ],
    )
    (a.out / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("TRAIN_EXPORT_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
