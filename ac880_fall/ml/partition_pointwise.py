"""推导 NCNN 静态形状，并标记实际由 PL 执行的 1×1 卷积。"""

import cache_env
import json
from pathlib import Path


def partition(path, minimum_macs=2000000):
    lines = path.read_text().splitlines()
    shapes = {}
    selected = []
    all_macs = 0
    new = lines[:2]
    for line in lines[2:]:
        parts = line.split()
        kind, name = parts[:2]
        ni, no = map(int, parts[2:4])
        inputs = parts[4 : 4 + ni]
        outputs = parts[4 + ni : 4 + ni + no]
        params = {
            int(k): float(v)
            for p in parts[4 + ni + no :]
            if "=" in p
            for k, v in [p.split("=", 1)]
            if not k.startswith("-")
        }
        get = lambda k, d: int(params.get(k, d))
        before = shapes[inputs[0]] if inputs else None
        if kind == "Input":
            after = (3, 192, 320)
        elif kind in ("Convolution", "ConvolutionDepthWise"):
            ci, h, w = before
            co = get(0, 0)
            kw = get(1, 0)
            kh = get(11, kw)
            sw = get(3, 1)
            sh = get(13, sw)
            dw = get(2, 1)
            dh = get(12, dw)
            left = get(4, 0)
            top = get(14, left)
            right = get(15, left)
            bottom = get(16, top)
            if min(left, top, right, bottom) < 0:
                raise ValueError("Unhandled auto-padding")
            oh = (h + top + bottom - dh * (kh - 1) - 1) // sh + 1
            ow = (w + left + right - dw * (kw - 1) - 1) // sw + 1
            after = (co, oh, ow)
            groups = get(7, 1)
            mac = co * oh * ow * (ci // groups) * kw * kh
            all_macs += mac
            if (
                kind == "Convolution"
                and kw == kh == sw == sh == 1
                and not any((left, top, right, bottom))
                and get(9, 0) in (0, 1)
                and mac >= minimum_macs
                and max(ci, co) <= 960
            ):
                parts[0] = "PLPointwise"
                line = " ".join(parts)
                selected.append(
                    dict(
                        name=name,
                        input_shape=before,
                        output_shape=after,
                        macs=mac,
                        weight_values=ci * co,
                        bias=get(5, 0),
                        activation=get(9, 0),
                    )
                )
        elif kind == "Pooling" and get(4, 0) == 1:
            after = (before[0], 1, 1)
        elif kind == "Reshape":
            w, h, c = get(0, -1), get(1, -1), get(2, -1)
            if c == -1 and w > 0 and h > 0:
                c = before[0] * before[1] * before[2] // (w * h)
            if min(w, h, c) < 1:
                raise ValueError("Unhandled reshape " + line)
            after = (c, h, w)
        elif kind == "Interp":
            c, h, w = before
            after = (
                c,
                get(3, 0) or round(h * params.get(1, 1)),
                get(4, 0) or round(w * params.get(2, 1)),
            )
        elif kind == "BinaryOp" and len(inputs) == 2:
            a, b = [shapes[k] for k in inputs]
            for x, y in zip(a, b):
                if x != y and x != 1 and y != 1:
                    raise ValueError("Invalid broadcast")
            after = tuple(max(x, y) for x, y in zip(a, b))
        elif kind in ("Split", "HardSwish", "HardSigmoid", "ReLU", "Sigmoid", "Clip", "BinaryOp"):
            after = before
        else:
            raise ValueError("Unimplemented shape rule: " + kind)
        for output in outputs:
            shapes[output] = after
        new.append(line)
    assert [shapes["out%d" % i] for i in range(3)] == [(30, 24, 40), (30, 12, 20), (30, 6, 10)]
    return "\n".join(new) + "\n", dict(
        selected=selected,
        total_convolution_macs=all_macs,
        pl_convolution_macs=sum(v["macs"] for v in selected),
        shape_inference="static NCHW, fixed 320x192 model",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    p = root / "models/fall_lite_v4_hardneg/fall_lite.ncnn.param"
    text, report = partition(p)
    p.with_name("fall_lite.pl.param").write_text(text)
    p.with_name("pl_partition.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                layers=len(report["selected"]),
                macs=report["pl_convolution_macs"],
                fraction=report["pl_convolution_macs"] / report["total_convolution_macs"],
            ),
            indent=2,
        )
    )
