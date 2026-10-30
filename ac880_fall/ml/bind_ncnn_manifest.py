"""将 FP32 NCNN 模型与 ONNX 及已校验的 ARM 库绑定。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    a = p.parse_args()
    directory = Path(a.directory)
    root = Path(__file__).resolve().parents[1]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta = json.loads((directory / "fall_lite.json").read_text())
    if meta["onnx_sha256"] != sha(directory / "fall_lite.onnx"):
        raise ValueError("ONNX metadata mismatch")
    meta["runtimes"] = dict(
        ncnn=dict(
            param="fall_lite.ncnn.param",
            weights="fall_lite.ncnn.bin",
            source_onnx_sha256=meta["onnx_sha256"],
            weight_precision="fp32",
            param_sha256=sha(directory / "fall_lite.ncnn.param"),
            weights_sha256=sha(directory / "fall_lite.ncnn.bin"),
            library_sha256=sha(root / "sw/runtime/libncnn.so.1"),
            ncnn_tag="20260526",
            source_commit="e54f7b1f88434e1d844ea0551b880a1cfb079ce1",
        )
    )
    (directory / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    (directory / "ADDITIONAL_DATA.md").write_text(
        """# Additional hand negatives, v7

FreiHAND: Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell,
Max Argus and Thomas Brox. "FreiHAND: A Dataset for Markerless Capture of
Hand Pose and Shape from Single RGB Images", ICCV 2019.
https://lmbweb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html

Author terms: research purposes only, without warranty; commercial use
prohibited; cite the paper. The model is a research-project artifact.
96 RGB images selected only from the original training-image variant:
72 training, 24 selection. All manually reviewed as hands/forearms without
a visible fallen body. No author evaluation image was downloaded or used.
Selection index blocks do not establish subject or scene independence.

15 newly reviewed OV5640 raw frames: hands, black-shirt chest, partial
upright people, and dorm furniture. User clarified that the black garment
was their own chest. These are training-scene regressions, not field tests.
Full inputs, review hashes, selection metrics and captures stay on E:.
""",
        encoding="utf8",
    )
    print(meta["training_run"], meta["onnx_sha256"])


if __name__ == "__main__":
    main()
