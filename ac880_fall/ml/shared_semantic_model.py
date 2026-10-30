"""冻结原骨干与摔倒头，仅训练共享特征上的新增分支。"""

import cache_env
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from fall_lite_model import FallLite, depthwise_block
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
from semantic_features import OBJECT_CLASSES, REGION_CLASSES, OUTPUT_NAMES


class SharedSemantic(nn.Module):
    def __init__(self, checkpoint, object_neck=False):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu")
        if state["depth"] != 16:
            raise ValueError("This model requires the accepted depth-16 pyramid")
        self.base = FallLite(depth=16)
        self.base.load_state_dict(state["model"], strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.objects = nn.ModuleList(
            (
                nn.Sequential(
                    depthwise_block(64, 64), nn.Conv2d(64, 6 * (len(OBJECT_CLASSES) + 4), 1)
                )
                if object_neck
                else nn.Conv2d(64, 6 * (len(OBJECT_CLASSES) + 4), 1)
            )
            for _ in range(3)
        )
        for branch in self.objects:
            head = branch[-1] if object_neck else branch
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                weights = head.weight.view(6, len(OBJECT_CLASSES) + 4, 64, 1, 1)
                bias = head.bias.view(6, len(OBJECT_CLASSES) + 4)
                # 金字塔和旧 SSD 投影的通道基不同。
                weights[:, len(OBJECT_CLASSES) :].zero_()
                bias[:, : len(OBJECT_CLASSES)].fill_(-4.595)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def pyramid(self, x):
        features = []
        for i, block in enumerate(self.base.features):
            x = block(x)
            if i in (6, 12):
                features.append(x)
        features.append(x)
        p3 = self.base.lateral[2](features[2])
        p2 = self.base.lateral[1](features[1]) + F.interpolate(
            p3, size=tuple(int(v) for v in features[1].shape[-2:]), mode="nearest"
        )
        p1 = self.base.lateral[0](features[0]) + F.interpolate(
            p2, size=tuple(int(v) for v in features[0].shape[-2:]), mode="nearest"
        )
        return (p1, p2, p3)

    def forward(self, x):
        features = self.pyramid(x)
        falls = tuple(head(f) for head, f in zip(self.base.heads, features))
        objects = tuple(head(f) for head, f in zip(self.objects, features))
        return falls + objects + features

    def object_forward(self, features):
        return tuple(head(f) for head, f in zip(self.objects, features))


class RegionHead(nn.Module):
    def __init__(self, dimension=1920):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, len(REGION_CLASSES)),
        )

    def forward(self, x):
        return self.layers(x)


def flatten_objects(outputs):
    count = len(OBJECT_CLASSES) + 4
    return torch.cat([v.permute(0, 2, 3, 1).reshape(v.shape[0], -1, count) for v in outputs], 1)


def export_shared(model, directory):
    directory = Path(directory)
    model.cpu().eval()
    sample = torch.zeros(1, 3, 192, 320)
    torch.jit.trace(model, sample, strict=True).save(str(directory / "shared_semantic.pt"))
    torch.onnx.export(
        model,
        sample,
        str(directory / "shared_semantic.onnx"),
        opset_version=8,
        input_names=["image"],
        output_names=list(OUTPUT_NAMES),
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(directory / "shared_semantic.onnx"))
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
    onnx.save(graph, str(directory / "shared_semantic.onnx"))
