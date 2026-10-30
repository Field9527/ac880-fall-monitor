"""按固定 NCNN 图重建可训练 PyTorch 网络；仅接受已核验算子，将权重作为数据读取，使用前需数值一致性验证。"""

import cache_env
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fall_lite_model import depthwise_block


NANO_OUTPUTS = tuple(
    kind + "_pred_stride_" + str(s) for s in (8, 16, 32) for kind in ("cls", "dis")
)
FEATURE_BLOBS = ("785", "807", "829")


def read_param(path):
    lines = Path(path).read_text().splitlines()
    if lines[0] != "7767517":
        raise ValueError("Unexpected NCNN format")
    result = []
    for line in lines[2:]:
        fields = line.split()
        op, name = fields[:2]
        ni, no = map(int, fields[2:4])
        inputs = fields[4 : 4 + ni]
        outputs = fields[4 + ni : 4 + ni + no]
        params = {}
        for field in fields[4 + ni + no :]:
            key, value = field.split("=")
            key = int(key)
            if key <= -23300:
                values = value.split(",")
                count = int(values[0])
                values = [float(v) for v in values[1:]]
                if len(values) != count:
                    raise ValueError("Bad NCNN array parameter")
                params[-key - 23300] = values
            else:
                params[key] = (
                    float(value) if any(c in value.lower() for c in (".", "e")) else int(value)
                )
        result.append((op, name, inputs, outputs, params))
    if len(result) != int(lines[1].split()[0]):
        raise ValueError("Layer count mismatch")
    return result


class NanoGraph(nn.Module):
    def __init__(self, source):
        super().__init__()
        source = Path(source)
        manifest = json.loads((source / "source_manifest.json").read_text())
        expected = {v["file"]: v["sha256"] for v in manifest["files"]}
        for name in ("nanodet-m.param", "nanodet-m.bin"):
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected[name]:
                raise ValueError("Pinned NanoDet source changed")
        self.rows = read_param(source / "nanodet-m.param")
        self.convolutions = nn.ModuleDict()
        raw = (source / "nanodet-m.bin").read_bytes()
        offset = 0
        for i, (op, name, inputs, outputs, p) in enumerate(self.rows):
            if op not in {
                "Input",
                "Convolution",
                "ConvolutionDepthWise",
                "Pooling",
                "Concat",
                "ShuffleChannel",
                "Slice",
                "Split",
                "Interp",
                "BinaryOp",
                "Sigmoid",
                "Reshape",
                "Permute",
            }:
                raise ValueError("Unreviewed operator: " + op)
            if op not in ("Convolution", "ConvolutionDepthWise"):
                continue
            co = int(p[0])
            kw = int(p.get(1, 1))
            kh = int(p.get(11, kw))
            group = int(p.get(7, 1)) if op == "ConvolutionDepthWise" else 1
            count = int(p[6])
            ci = count // (co * kh * kw) * group
            tag = struct.unpack_from("<I", raw, offset)[0]
            offset += 4
            if tag == 0x01306B47:
                weight = np.frombuffer(raw, dtype="<f2", count=count, offset=offset).astype(
                    np.float32
                )
                offset += (count * 2 + 3) // 4 * 4
            elif tag == 0:
                weight = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy()
                offset += count * 4
            else:
                raise ValueError("Unsupported weight storage: " + hex(tag))
            layer = nn.Conv2d(
                ci,
                co,
                (kh, kw),
                stride=(int(p.get(13, p.get(3, 1))), int(p.get(3, 1))),
                padding=(int(p.get(14, p.get(4, 0))), int(p.get(4, 0))),
                dilation=int(p.get(2, 1)),
                groups=group,
                bias=bool(p.get(5, 0)),
            )
            layer.weight.data.copy_(torch.from_numpy(weight.reshape(co, ci // group, kh, kw)))
            if layer.bias is not None:
                bias = np.frombuffer(raw, dtype="<f4", count=co, offset=offset).copy()
                offset += co * 4
                layer.bias.data.copy_(torch.from_numpy(bias))
            self.convolutions[str(i)] = layer
        if offset != len(raw):
            raise ValueError("Weights did not consume the checked source exactly")
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image, capture=False):
        values = {}
        for i, (op, name, inputs, outputs, p) in enumerate(self.rows):
            if op == "Input":
                values[outputs[0]] = image
                continue
            x = values[inputs[0]]
            if op in ("Convolution", "ConvolutionDepthWise"):
                x = self.convolutions[str(i)](x)
                activation = int(p.get(9, 0))
                if activation == 1:
                    x = F.relu(x)
                elif activation == 2:
                    x = F.leaky_relu(x, float(p.get(10, [0.1])[0]))
                elif activation != 0:
                    raise ValueError("Unknown fused activation")
            elif op == "Pooling":
                if int(p.get(0, 0)) != 0:
                    raise ValueError("Unknown pooling")
                x = F.max_pool2d(x, int(p.get(1, 1)), int(p.get(2, 1)), int(p.get(3, 0)))
            elif op == "Concat":
                x = torch.cat([values[v] for v in inputs], dim=1)
            elif op == "ShuffleChannel":
                batch, c, h, w = x.shape
                group = int(p.get(0, 1))
                x = (
                    x.reshape(batch, group, c // group, h, w)
                    .transpose(1, 2)
                    .reshape(batch, c, h, w)
                )
            elif op == "Slice":
                sizes = [int(v) for v in p[0]]
                remaining = x.shape[1] - sum(v for v in sizes if v >= 0)
                sizes = [remaining if v == -233 else v for v in sizes]
                for out, value in zip(outputs, x.split(sizes, dim=1)):
                    values[out] = value
                continue
            elif op == "Split":
                for out in outputs:
                    values[out] = x
                continue
            elif op == "Interp":
                if int(p[0]) != 2:
                    raise ValueError("Unsupported resize mode")
                x = F.interpolate(
                    x, scale_factor=(float(p[1]), float(p[2])), mode="bilinear", align_corners=False
                )
            elif op == "BinaryOp":
                if int(p.get(0, 0)) != 0:
                    raise ValueError("Unsupported binary operation")
                x = x + values[inputs[1]]
            elif op == "Sigmoid":
                x = x.sigmoid()
            elif op == "Reshape":
                if int(p[0]) != -1 or int(p[1]) not in (80, 32):
                    raise ValueError("Unexpected prediction reshape")
                x = x.reshape(x.shape[0], int(p[1]), -1)
            elif op == "Permute":
                if int(p[0]) != 1 or x.ndim != 3:
                    raise ValueError("Unexpected prediction permutation")
                x = x.transpose(1, 2)
            values[outputs[0]] = x
        if capture:
            return values
        return tuple(values[name] for name in NANO_OUTPUTS), tuple(
            values[name] for name in FEATURE_BLOBS
        )


class NanoFall(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.graph = NanoGraph(source)
        self.falls = nn.ModuleList(
            nn.Sequential(depthwise_block(96, 64), nn.Conv2d(64, 30, 1)) for _ in range(3)
        )
        for branch in self.falls:
            nn.init.normal_(branch[-1].weight, std=0.01)
            nn.init.zeros_(branch[-1].bias)
            with torch.no_grad():
                branch[-1].bias.view(6, 5)[:, 0].fill_(-4.595)

    def train(self, mode=True):
        super().train(mode)
        self.graph.eval()
        return self

    def forward(self, image):
        objects, features = self.graph(image)
        return tuple(head(f) for head, f in zip(self.falls, features)) + objects + features
