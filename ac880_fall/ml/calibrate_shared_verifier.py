"""仅使用开发数据选择倒地与非倒地区域的工作点。"""

import cache_env
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from shared_semantic_model import RegionHead
from train_shared_heads import roi_metrics, all_probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    records = json.loads((a.cache / "records.json").read_text())["items"]
    rows = json.loads((a.cache / "roi_records.json").read_text())
    selected = torch.load(a.model / "region_head.pt", map_location="cpu")
    head = RegionHead(len(selected["mean"])).cuda()
    head.load_state_dict(selected["state"])
    values = np.load(a.cache / "roi_vectors.npy", mmap_mode="r")
    x = torch.from_numpy((np.asarray(values) - selected["mean"]) / selected["std"]).cuda()
    probabilities = all_probabilities(head, x)
    trials = []
    best = None
    for ceiling in (0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05):
        metrics = roi_metrics(probabilities, rows, records, confidence=1 - ceiling, aggregate=True)
        retained = all(r["kept_tp"] >= r["base_tp"] for r in metrics.values() if r["gt"])
        cost = sum(
            r["kept_fp_images"] * (3 if name in ("FreiHAND_research", "user_camera") else 1)
            for name, r in metrics.items()
        )
        trial = dict(
            fall_probability_ceiling=ceiling,
            retained_development_boxes=retained,
            weighted_fp_cost=cost,
            metrics=metrics,
        )
        trials.append(trial)
        key = (int(retained), -cost, -ceiling)
        if best is None or key > best[0]:
            best = (key, trial)
    chosen = best[1]
    if not chosen["retained_development_boxes"]:
        raise RuntimeError("No operating point retains baseline true positives")
    report = dict(
        chosen=chosen,
        trials=trials,
        scope="Development calibration; repeated legacy regressions are not fresh field acceptance",
        groups={
            split: roi_metrics(
                probabilities,
                rows,
                records,
                split,
                confidence=1 - chosen["fall_probability_ceiling"],
                aggregate=True,
            )
            for split in ("train", "development", "test")
        },
    )
    (a.model / "roi_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = json.loads((a.model / "shared_semantic.json").read_text())
    metadata.update(
        nonfall_aggregation="sum_of_nonfall_classes",
        fall_probability_ceiling=chosen["fall_probability_ceiling"],
        strong_nonfall_threshold=1 - chosen["fall_probability_ceiling"],
        descriptor_abi="pyramid%d_adaptive3_global_v1" % (len(selected["mean"]) // 30),
        calibration_scope="Development only; no fresh field acceptance",
    )
    (a.model / "shared_semantic.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
