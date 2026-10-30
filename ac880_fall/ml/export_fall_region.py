"""导出不变的 v6 摔倒张量与特征图，去除未采用的物体头。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from torch import nn
from shared_semantic_model import SharedSemantic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from semantic_features import FALL_REGION_OUTPUT_NAMES


class FallRegion(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.model = SharedSemantic(checkpoint)

    def forward(self, x):
        fs = self.model.pyramid(x)
        return tuple(head(f) for head, f in zip(self.model.base.heads, fs)) + fs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--verifier-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = FallRegion(a.checkpoint).eval()
    sample = torch.from_numpy(
        np.random.RandomState(17).normal(size=(1, 3, 192, 320)).astype(np.float32)
    )
    with torch.no_grad():
        values = [v.numpy() for v in model(sample)]
    torch.jit.trace(model, sample, strict=True).save(str(a.out / "shared_semantic.pt"))
    torch.onnx.export(
        model,
        sample,
        str(a.out / "shared_semantic.onnx"),
        opset_version=8,
        input_names=["image"],
        output_names=list(FALL_REGION_OUTPUT_NAMES),
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(a.out / "shared_semantic.onnx"))
    aliases = {}
    kept = []
    for node in graph.graph.node:
        for k, name in enumerate(node.input):
            node.input[k] = aliases.get(name, name)
        if node.op_type == "Identity":
            aliases[node.output[0]] = node.input[0]
        else:
            kept.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(kept)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(a.out / "shared_semantic.onnx"))
    np.savez(
        a.out / "export_reference.npz",
        input=sample.numpy(),
        **{"out" + str(i): v for i, v in enumerate(values)},
    )
    for name in ("region_verifier.npz", "roi_calibration.json"):
        shutil.copyfile(a.verifier_model / name, a.out / name)
    meta = json.loads((a.verifier_model / "shared_semantic.json").read_text())
    meta.pop("ncnn", None)
    meta.update(
        architecture="v6_roi",
        production_qualified=False,
        training_run="v6_regions_scene_v1",
        onnx_sha256=hashlib.sha256((a.out / "shared_semantic.onnx").read_bytes()).hexdigest(),
        base_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        object_policy="Optional pinned pretrained NanoDet labels every4s; never an alarm veto; no second concurrent worker",
    )
    (a.out / "shared_semantic.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
