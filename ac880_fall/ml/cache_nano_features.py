"""缓存固定版本 NanoDet 特征，沿用已冻结的数据划分。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
import cv2
import numpy as np
import torch
from nanodet_graph import NanoGraph
from fall_lite import letterbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.out.drive.upper() != "E:":
        raise ValueError("Keep features on E:")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    info = json.loads(a.records.read_text())
    items = info["items"]
    n = len(items)
    model = NanoGraph(a.source).cuda().eval()
    maps = [
        np.lib.format.open_memmap(
            str(a.out / ("feature%d.npy" % i)), mode="w+", dtype=np.float16, shape=(n, 96, h, w)
        )
        for i, (h, w) in enumerate(((24, 40), (12, 20), (6, 10)))
    ]
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, n, 24):
            batch = []
            for item in items[offset : offset + 24]:
                bgr = cv2.imread(item["path"])
                if bgr is None:
                    raise ValueError("Image is unreadable")
                if item["rotation"]:
                    bgr = np.ascontiguousarray(np.rot90(bgr, item["rotation"] // 90))
                image, (sx, sy, ox, oy) = letterbox(bgr)
                nh, nw = round(bgr.shape[0] * sy), round(bgr.shape[1] * sx)
                image[:oy] = 0
                image[oy + nh :] = 0
                image[:, :ox] = 0
                image[:, ox + nw :] = 0
                batch.append(image)
            x = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2)).cuda().float()
            x = (
                x - torch.tensor([103.53, 116.28, 123.675], device="cuda")[None, :, None, None]
            ) / torch.tensor([57.375, 57.12, 58.395], device="cuda")[None, :, None, None]
            objects, features = model(x)
            for k, f in enumerate(features):
                maps[k][offset : offset + len(batch)] = f.cpu().numpy().astype(np.float16)
            if offset % 240 == 0:
                print(
                    "NANO_CACHE",
                    offset + len(batch),
                    n,
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    for m in maps:
        m.flush()
    for item in items:
        item["reference_v6_candidates"] = item["base_candidates"]
    info.update(
        architecture="nanodet_shared",
        feature_channels=96,
        preprocess="nanodet_bgr_zero_pad",
        nano_source_manifest_sha256=hashlib.sha256(
            (a.source / "source_manifest.json").read_bytes()
        ).hexdigest(),
        complete=True,
    )
    (a.out / "records.json").write_text(json.dumps(info, indent=2) + "\n")
    print("NANO_FEATURES_COMPLETE", n, flush=True)


if __name__ == "__main__":
    main()
