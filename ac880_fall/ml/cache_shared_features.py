"""在 E 盘缓存冻结骨干特征与 v6 回归候选框。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
import torch
from shared_semantic_model import SharedSemantic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from fall_lite import letterbox, anchors_for_shape, flatten_heads, decode
from semantic_features import region_descriptor


def rotate_box(box, width, height, angle):
    x1, y1, x2, y2 = box
    for _ in range(angle // 90):
        x1, y1, x2, y2 = y1, width - x2, y2, width - x1
        width, height = height, width
    return [x1, y1, x2, y2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=24)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.out.drive.upper() != "E:":
        raise ValueError("Feature caches must stay on E:")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    manifest = json.loads(a.index.read_text())
    items = manifest["items"]
    n = len(items)
    model = SharedSemantic(a.checkpoint).cuda().eval()
    anchors = anchors_for_shape()
    maps = [
        np.lib.format.open_memmap(
            str(a.out / ("feature%d.npy" % i)), mode="w+", dtype=np.float16, shape=(n, 64, h, w)
        )
        for i, (h, w) in enumerate(((24, 40), (12, 20), (6, 10)))
    ]
    records = []
    start = time.perf_counter()
    max_error = 0.0
    with torch.no_grad():
        for offset in range(0, n, a.batch_size):
            batch = []
            entries = []
            for item in items[offset : offset + a.batch_size]:
                image = cv2.imread(item["path"])
                if image is None:
                    raise ValueError("Unreadable training image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if item["rotation"]:
                    image = np.ascontiguousarray(np.rot90(image, item["rotation"] // 90))
                boxed, (sx, sy, ox, oy) = letterbox(image)

                def mapped(box):
                    b = rotate_box(box, item["width"], item["height"], item["rotation"])
                    return (np.array(b) * [sx, sy, sx, sy] + [ox, oy, ox, oy]).tolist()

                entry = dict(
                    item,
                    objects=[dict(v, box=mapped(v["box"])) for v in item["objects"]],
                    regions=[dict(v, box=mapped(v["box"])) for v in item["regions"]],
                    input_transform=[sx, sy, ox, oy],
                )
                entries.append(entry)
                batch.append(boxed)
            x = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2)).cuda().float() / 255.0
            x = (
                x - torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
            features = model.pyramid(x)
            falls = tuple(head(f) for head, f in zip(model.base.heads, features))
            if offset == 0:
                expected = model.base(x)
                max_error = max(float((a - b).abs().max()) for a, b in zip(expected, falls))
                if max_error > 1e-6:
                    raise ValueError("Frozen fall output changed")
            for k, f in enumerate(features):
                maps[k][offset : offset + len(batch)] = f.cpu().numpy().astype(np.float16)
            raw = flatten_heads([v.cpu().numpy() for v in falls]) if len(batch) == 1 else None
            for k, entry in enumerate(entries):
                pred = flatten_heads([v[k : k + 1].cpu().numpy() for v in falls])
                predictions = decode(pred, anchors, 0.3, top_k=8)
                entry["base_candidates"] = [
                    dict(
                        score=v["score"],
                        box=[
                            v["box"][0],
                            v["box"][1],
                            v["box"][0] + v["box"][2],
                            v["box"][1] + v["box"][3],
                        ],
                    )
                    for v in predictions
                ]
                records.append(entry)
            if offset % (a.batch_size * 10) == 0:
                print(
                    json.dumps(
                        dict(
                            cached=len(records),
                            total=n,
                            seconds=round(time.perf_counter() - start, 1),
                        )
                    ),
                    flush=True,
                )
    for m in maps:
        m.flush()
    info = dict(
        index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(),
        base_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        fall_output_max_abs_error=max_error,
        items=records,
        complete=True,
        dtype="float16_feature_cache_only; runtime FP32",
    )
    (a.out / "records.json").write_text(json.dumps(info, indent=2) + "\n")
    print("FEATURE_CACHE_COMPLETE", len(records), round(time.perf_counter() - start, 1), flush=True)


if __name__ == "__main__":
    main()
