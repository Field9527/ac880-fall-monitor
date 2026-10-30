"""统一 ONNX 输入布局，等价替换不支持的通道填充。"""

import cache_env
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    changes = []
    for name, size in (("palm", 192), ("hand", 224)):
        model = onnx.load(str(a.directory / (name + ".onnx")))
        graph = model.graph
        input_name = graph.input[0].name
        first = [
            n for n in graph.node if n.op_type == "Transpose" and list(n.input) == [input_name]
        ]
        if len(first) != 1 or list(first[0].attribute[0].ints) != [0, 3, 1, 2]:
            raise ValueError("Unexpected source input layout")
        old = first[0].output[0]
        graph.node.remove(first[0])
        for node in graph.node:
            for i, v in enumerate(node.input):
                if v == old:
                    node.input[i] = input_name
        for d, value in zip(graph.input[0].type.tensor_type.shape.dim, (1, 3, size, size)):
            d.dim_value = value
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
        shapes = {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(graph.input) + list(graph.value_info)
        }
        initial = {v.name: numpy_helper.to_array(v) for v in graph.initializer}
        nodes = []
        count = 0
        for node in graph.node:
            if node.op_type != "Pad":
                nodes.append(node)
                continue
            attrs = {v.name: helper.get_attribute_value(v) for v in node.attribute}
            if attrs.get("mode", b"constant") != b"constant" or (
                len(node.input) > 2 and float(initial[node.input[2]]) != 0.0
            ):
                raise ValueError("Only exact zero padding is supported")
            pads = initial[node.input[1]].tolist()
            shape = shapes[node.input[0]]
            if (
                len(shape) != 4
                or len(pads) != 8
                or any(pads[i] for i in (0, 2, 3, 4, 6, 7))
                or pads[1] != 0
            ):
                raise ValueError("Unreviewed Pad operation")
            ci = shape[1]
            co = ci + pads[5]
            weight = np.zeros((co, ci, 1, 1), np.float32)
            weight[np.arange(ci), np.arange(ci), 0, 0] = 1
            wname = "exact_channel_pad_" + str(count)
            graph.initializer.append(numpy_helper.from_array(weight, wname))
            nodes.append(
                helper.make_node(
                    "Conv",
                    [node.input[0], wname],
                    list(node.output),
                    name=wname,
                    kernel_shape=[1, 1],
                )
            )
            count += 1
        del graph.node[:]
        graph.node.extend(nodes)
        used = {v for n in graph.node for v in n.input}
        kept = [v for v in graph.initializer if v.name in used]
        del graph.initializer[:]
        graph.initializer.extend(kept)
        onnx.checker.check_model(model)
        onnx.save(model, str(a.directory / (name + "_nchw.onnx")))
        changes.append(
            dict(model=name, input_layout="NCHW", exact_identity_channel_expansions=count)
        )
    (a.directory / "conversion_notes.json").write_text(json.dumps(changes, indent=2) + "\n")
    print(json.dumps(changes))


if __name__ == "__main__":
    main()
