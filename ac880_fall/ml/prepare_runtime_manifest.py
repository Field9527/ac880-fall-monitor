"""绑定 FP32 NCNN、ONNX 与可选已验证 PL 文件。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
from partition_pointwise import partition


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--overlay")
    p.add_argument("--remote-model", default="candidates/v5")
    p.add_argument("--config-name", default="config_candidate_v5.json")
    a = p.parse_args()
    directory = root / a.model_dir
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta = json.loads((directory / "fall_lite.json").read_text())
    digest = sha(directory / "fall_lite.onnx")
    if meta["onnx_sha256"] != digest:
        raise ValueError("Export metadata SHA mismatch")
    info = dict(
        param="fall_lite.ncnn.param",
        weights="fall_lite.ncnn.bin",
        weight_precision="fp32",
        source_onnx_sha256=digest,
        param_sha256=sha(directory / "fall_lite.ncnn.param"),
        weights_sha256=sha(directory / "fall_lite.ncnn.bin"),
        library_sha256=sha(root / "sw/runtime/libncnn.so.1"),
        ncnn_tag="20260526",
        source_commit="e54f7b1f88434e1d844ea0551b880a1cfb079ce1",
    )
    graph, report = partition(directory / "fall_lite.ncnn.param")
    if len(report["selected"]) != 31 or report["pl_convolution_macs"] != 228311040:
        raise ValueError("Changed partition requires explicit capacity/performance validation")
    (directory / "fall_lite.pl.param").write_text(graph)
    (directory / "pl_partition.json").write_text(json.dumps(report, indent=2))
    meta.setdefault("runtimes", {})["ncnn"] = info
    cfg = json.loads((root / "sw/config_fall_lite_cpu.json").read_text())
    cfg["fall_lite"].update(
        model=a.remote_model + "/fall_lite.onnx", metadata=a.remote_model + "/fall_lite.json"
    )
    cfg["fall_lite"].pop("pl_plugin", None)
    if a.overlay:
        bit = root / a.overlay
        hwh = bit.with_suffix(".hwh")
        meta["runtimes"]["ncnn_pl"] = dict(
            param="fall_lite.pl.param",
            param_sha256=sha(directory / "fall_lite.pl.param"),
            source_onnx_sha256=digest,
            plugin_sha256=sha(root / "sw/runtime/libplconv.so"),
            overlay_sha256=sha(bit),
            hwh_sha256=sha(hwh),
            abi="pointwise_fp32_v1",
            selected_layers=31,
            pl_convolution_macs=report["pl_convolution_macs"],
            total_convolution_macs=report["total_convolution_macs"],
        )
        cfg["overlay_path"] = "/home/xilinx/fall/candidates/pl17/" + bit.name
        cfg["fall_lite"].update(engine="ncnn_pl", pl_plugin="runtime/libplconv.so")
    (directory / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
    (root / "out/validation_20261017" / a.config_name).write_text(json.dumps(cfg, indent=2) + "\n")
    print("Prepared", directory, "engine", cfg["fall_lite"]["engine"], digest, flush=True)


if __name__ == "__main__":
    main()
