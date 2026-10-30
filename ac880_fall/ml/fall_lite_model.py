"""轻量倒地框检测网络，仅用于训练与导出。沿用 MobileNetV3-Large 前 13 块与三个 SSD 检测头，不依赖前景候选。"""

import math
import cache_env  # 必须先于 Torch 和 ONNX 导入。
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.ops.misc import SqueezeExcitation
import torch.nn.functional as F


class PortableSE(SqueezeExcitation):
    def forward(self, x):
        # 旧版 OpenCV 不支持运行时通道缩放广播到空间维度。
        size = (int(x.shape[-2]), int(x.shape[-1]))
        scale = F.interpolate(self._scale(x), size=size, mode="nearest")
        return x * scale


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class HardSwish(nn.Module):
    def forward(self, x):
        return x * (torch.clamp(x + 3.0, 0.0, 6.0) / 6.0)


def portable_activations(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Hardswish):
            setattr(module, name, HardSwish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, HardSigmoid())
        else:
            if isinstance(child, SqueezeExcitation):
                child.__class__ = PortableSE
            portable_activations(child)


def depthwise_block(ci, co, stride=1):
    return nn.Sequential(
        nn.Conv2d(ci, ci, 3, stride, 1, groups=ci, bias=False),
        nn.BatchNorm2d(ci),
        nn.ReLU(),
        nn.Conv2d(ci, co, 1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(),
    )


class FallLite(nn.Module):
    def __init__(self, teacher=None, depth=13):
        super().__init__()
        if depth not in (13, 16):
            raise ValueError("Backbone depth must be 13 or 16")
        self.depth = depth
        self.features = mobilenet_v3_large(weights=None).features[:depth]
        if teacher:
            ck = torch.load(teacher, map_location="cpu")
            prefix = "backbone.body."
            state = {k[len(prefix) :]: v for k, v in ck["model"].items() if k.startswith(prefix)}
            self.features.load_state_dict(
                {k: v for k, v in state.items() if int(k.split(".")[0]) < depth}, strict=True
            )
        portable_activations(self.features)
        self.extra = depthwise_block(112, 96, 2) if depth == 13 else nn.Identity()
        if depth == 16:
            self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (40, 112, 160)])
        self.heads = nn.ModuleList(
            [
                nn.Sequential(depthwise_block(c, 64), nn.Conv2d(64, 30, 1))
                for c in ((40, 112, 96) if depth == 13 else (64, 64, 64))
            ]
        )
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
            with torch.no_grad():
                head[-1].bias.view(6, 5)[:, 0].fill_(-math.log(99.0))

    def forward(self, x):
        features = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 6:
                features.append(x)
            elif i == 12:
                features.append(x)
        features.append(self.extra(x))
        if self.depth == 16:
            p3 = self.lateral[2](features[2])
            size16 = tuple(int(v) for v in features[1].shape[-2:])
            p2 = self.lateral[1](features[1]) + F.interpolate(p3, size=size16, mode="nearest")
            size8 = tuple(int(v) for v in features[0].shape[-2:])
            p1 = self.lateral[0](features[0]) + F.interpolate(p2, size=size8, mode="nearest")
            features = (p1, p2, p3)
        outputs = [head(f) for head, f in zip(self.heads, features)]
        # 保留空间输出，避免旧 OpenCV 的动态变形与拼接。
        return tuple(outputs)


def flatten_heads(outputs):
    return torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 5) for x in outputs], 1)


def export_model(model, path, height=192, width=320):
    model.cpu().eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, height, width),
        str(path),
        opset_version=8,
        input_names=["image"],
        output_names=["head8", "head16", "head32"],
        do_constant_folding=True,
    )
    # 解析共享常量的 Identity 别名，兼容 OpenCV 4.2。
    import onnx

    graph = onnx.load(str(path))
    aliases = {}
    kept = []
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = aliases.get(name, name)
        if node.op_type == "Identity":
            aliases[node.output[0]] = node.input[0]
        else:
            kept.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(kept)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--teacher")
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=int, default=13)
    a = p.parse_args()
    model = FallLite(a.teacher, a.depth)
    export_model(model, a.output)
    print("EXPORT_ONLY: detection heads are untrained; not a recognition model")
    print("parameters", sum(x.numel() for x in model.parameters()))
