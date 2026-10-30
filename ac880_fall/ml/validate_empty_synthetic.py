"""确定性暗场与空白图回归，不作为真实场景基准。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import letterbox, flatten_heads


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = root / args.model_dir / "fall_lite.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    net = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(1427)
    images = [
        ("uniform_%d" % value, np.full((360, 640, 3), value, np.uint8))
        for value in (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 254, 255)
    ]
    for i in range(12):
        image = np.clip(rng.normal(i * 2, 1 + i * 0.3, (360, 640, 3)), 0, 255).astype(np.uint8)
        images.append(("low_light_noise_%d" % i, image))
    rows = []
    for name, image in images:
        frame, _ = letterbox(image)
        blob = frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )[None, :, None, None]
        outputs = net.run(None, {net.get_inputs()[0].name: blob})
        raw = flatten_heads(sorted(outputs, key=lambda v: v.shape[-1] * v.shape[-2], reverse=True))
        if not np.isfinite(raw).all():
            raise RuntimeError("Non-finite output")
        score = float((1 / (1 + np.exp(-np.clip(raw[:, 0], -40, 40)))).max())
        rows.append(dict(name=name, max_score=score, triggered=score >= 0.7))
    report = dict(
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        execution="ONNX Runtime FP32, synthetic blank/low-light inputs only",
        passed=not any(v["triggered"] for v in rows),
        samples=rows,
        max_score=max(v["max_score"] for v in rows),
    )
    (root / args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Synthetic empty regression failed")


if __name__ == "__main__":
    main()
