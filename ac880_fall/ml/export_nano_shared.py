"""导出冻结的物体骨干与已训练摔倒分支，记录模型来源。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from nanodet_graph import NanoFall, FEATURE_BLOBS
from semantic_features import NANO_SHARED_OUTPUT_NAMES, REGION_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    model = NanoFall(a.source).eval()
    checkpoint = torch.load(a.model / "fall_head.pt", map_location="cpu")
    model.falls.load_state_dict(checkpoint["falls"])
    sample = torch.from_numpy(
        np.random.RandomState(17).normal(size=(1, 3, 192, 320)).astype(np.float32)
    )
    with torch.no_grad():
        values = [v.numpy() for v in model(sample)]
    torch.jit.trace(model, sample, strict=True).save(str(a.model / "shared_semantic.pt"))
    torch.onnx.export(
        model,
        sample,
        str(a.model / "shared_semantic.onnx"),
        opset_version=11,
        input_names=["image"],
        output_names=list(NANO_SHARED_OUTPUT_NAMES),
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(a.model / "shared_semantic.onnx"))
    onnx.checker.check_model(graph)
    np.savez(
        a.model / "export_reference.npz",
        input=sample.numpy(),
        **{"out" + str(i): v for i, v in enumerate(values)},
    )
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    info = json.loads((a.cache / "records.json").read_text())
    selection = json.loads((a.model / "fall_history.json").read_text())["best"]
    metadata = dict(
        trained=True,
        production_qualified=False,
        architecture="nanodet_shared",
        training_run=a.model.name,
        input_height=192,
        input_width=320,
        recommended_threshold=0.7,
        descriptor_abi="pyramid96_adaptive3_global_v1",
        preprocess="BGR mean103.53,116.28,123.675 std57.375,57.12,58.395; fixed320x192 zero letterbox; INTER_AREA",
        region_classes=REGION_CLASSES,
        source_manifest=json.loads((a.source / "source_manifest.json").read_text()),
        feature_blobs=FEATURE_BLOBS,
        index_sha256=info["index_sha256"],
        cache_records_sha256=sha(a.cache / "records.json"),
        fall_head_epoch=checkpoint["epoch"],
        fall_selection=selection,
        onnx_sha256=sha(a.model / "shared_semantic.onnx"),
        nonfall_aggregation="sum_of_nonfall_classes",
        fall_probability_ceiling=0.02,
        strong_nonfall_threshold=0.98,
        safety="No person-absence veto; unknown or unverified regions retain fall evidence",
        limitations=[
            "Development calibration and repeated regression are not fresh field acceptance",
            "Phone replay requires explicit orientation; static images do not identify physical gravity",
            "Coarse hand regions are ROI supervision, never exact box regression",
        ],
    )
    if (a.model / "region_verifier.npz").exists():
        metadata["verifier_sha256"] = sha(a.model / "region_verifier.npz")
    (a.model / "shared_semantic.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                model=str(a.model), shapes=[list(v.shape) for v in values], fall_selection=selection
            )
        )
    )


if __name__ == "__main__":
    main()
