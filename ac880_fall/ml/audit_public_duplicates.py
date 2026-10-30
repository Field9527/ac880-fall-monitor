"""检查旧数据与按人员划分的新数据是否重复；近似图仅列为复核项，不自动标为负例。"""

import cache_env
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def fingerprint(item):
    image = cv2.imread(item["path"])
    if image is None:
        raise ValueError("Unreadable image: " + item["path"])
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(
        cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    frequencies = cv2.dct(gray.astype(np.float32))[:8, :8].reshape(-1)
    perceptual = np.packbits(frequencies > np.median(frequencies[1:]))
    return hashlib.sha256(small.tobytes()).hexdigest(), perceptual, small


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/fall_lite_v5_public")
    folder = root / parser.parse_args().model_dir
    manifest = json.loads((folder / "split.json").read_text())
    items = manifest["items"]
    relevant = [v for v in items if v["source"] in ("original_falldataset", "CAUCAFall_v4")]
    with ThreadPoolExecutor(6) as pool:
        features = list(pool.map(fingerprint, relevant))
    original = [i for i, v in enumerate(relevant) if v["source"] == "original_falldataset"]
    public = [i for i, v in enumerate(relevant) if v["source"] == "CAUCAFall_v4"]
    oldhash = np.stack([features[i][1] for i in original])
    lookup = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    canonical = {}
    exact = []
    near = []
    for index, item in enumerate(relevant):
        digest = features[index][0]
        if digest in canonical:
            other = canonical[digest]
            if relevant[other]["split"] != item["split"]:
                exact.append(
                    dict(
                        a=relevant[other]["path"],
                        a_split=relevant[other]["split"],
                        b=item["path"],
                        b_split=item["split"],
                    )
                )
        canonical[digest] = index
    for i in public:
        distance = lookup[np.bitwise_xor(oldhash, features[i][1])].sum(1)
        for position in np.flatnonzero(distance <= 4):
            j = original[int(position)]
            mae = float(np.abs(features[i][2].astype(np.float32) - features[j][2]).mean())
            if mae < 8:
                near.append(
                    dict(
                        public=relevant[i]["path"],
                        public_split=relevant[i]["split"],
                        original=relevant[j]["path"],
                        original_split=relevant[j]["split"],
                        phash_distance=int(distance[position]),
                        rgb64_mae=mae,
                    )
                )
    report = dict(
        canonical_rgb64_cross_split_matches=exact,
        possible_near_matches=near,
        public_images=len(public),
        original_images=len(original),
        policy="Inspect possible near matches; do not claim new subject tests are unseen until audit is resolved",
    )
    (folder / "duplicate_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                public=len(public),
                original=len(original),
                exact_cross_split=len(exact),
                near=len(near),
            )
        ),
        flush=True,
    )
    if exact:
        raise RuntimeError("Cross-split duplicate images require regrouping before training")


if __name__ == "__main__":
    main()
