"""从现有模型选择划分生成确定性 CPU 与 PL 数值样本，不作为新准确率保留集。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from fall_lite_model import FallLite
from train_fall_lite import Images, normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import decode, flatten_heads, anchors_for_shape


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/fall_lite_v4_hardneg")
    parser.add_argument("--output", default="out/validation_20261017/model_cases")
    args = parser.parse_args()
    modeldir = root / args.model_dir
    output = root / args.output
    output.mkdir(exist_ok=True)
    items = json.loads((root / "models/fall_lite_v3_fpn/split.json").read_text())["items"]
    val = [v for v in items if v["split"] == "val"]
    rng = random.Random(17)
    chosen = rng.sample([v for v in val if v["boxes"]], 12) + rng.sample(
        [v for v in val if not v["boxes"]], 4
    )
    images = Images(chosen, 192, 320, False)
    model = FallLite(depth=16)
    model.load_state_dict(torch.load(modeldir / "best.pt", map_location="cpu")["model"])
    model.eval()
    torch.set_num_threads(4)
    rows = []
    with torch.no_grad():
        for i, item in enumerate(chosen):
            x = normalize(images[i][0][None], "cpu")
            heads = [a.numpy() for a in model(x)]
            filename = "case_%02d.npz" % i
            np.savez(
                output / filename,
                image=x.numpy(),
                **dict(zip(("head8", "head16", "head32"), heads)),
            )
            detections = decode(flatten_heads(heads), anchors_for_shape(), 0.7)
            rows.append(
                dict(
                    file=filename,
                    source_sha256=hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(),
                    source_file=Path(item["path"]).name,
                    source_box_count=len(item["boxes"]),
                    reference_detections=detections,
                )
            )
    report = dict(
        purpose="Numerical/runtime regression, existing selection split; not field acceptance",
        model_sha256=hashlib.sha256((modeldir / "fall_lite.onnx").read_bytes()).hexdigest(),
        cases=rows,
    )
    (output / "cases.json").write_text(json.dumps(report, indent=2))
    print("Exported %d real-image fixtures to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
