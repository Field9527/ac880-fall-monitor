import cache_env
from pathlib import Path
import re
import tarfile
import json

src = Path("E:/Zynq_cache/ncnn-20260526")
out = Path(__file__).resolve().parents[1] / "out/false_alarm_speed_20261016"
if not (out / "ncnn_source.tar.gz").exists():
    with tarfile.open(out / "ncnn_source.tar.gz", "w:gz") as t:
        for name in ("CMakeLists.txt", "LICENSE.txt", "cmake", "src"):
            t.add(src / name, arcname="ncnn/" + name)
keep = {
    "AbsVal",
    "BinaryOp",
    "Clip",
    "Convolution",
    "ConvolutionDepthWise",
    "Crop",
    "Flatten",
    "GELU",
    "HardSigmoid",
    "HardSwish",
    "Input",
    "Interp",
    "Mish",
    "Packing",
    "Padding",
    "Pooling",
    "ReLU",
    "Reshape",
    "Sigmoid",
    "Split",
    "Swish",
    "TanH",
    "UnaryOp",
}
names = re.findall(r"ncnn_add_layer\((\w+)", (src / "src/CMakeLists.txt").read_text())
flags = ["-DWITH_LAYER_%s=%s" % (name.lower(), "ON" if name in keep else "OFF") for name in names]
script = """#!/bin/sh
set -eu
cd /home/xilinx/fall/ncnn_build
tar -xmzf ncnn_source.tar.gz
cmake -S ncnn -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_VERSION=20260526 \\
 -DNCNN_SHARED_LIB=ON -DNCNN_C_API=ON -DNCNN_OPENMP=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_VFPV4=OFF \\
 -DNCNN_VULKAN=OFF -DNCNN_PIXEL=OFF -DNCNN_PIXEL_ROTATE=OFF -DNCNN_PIXEL_AFFINE=OFF -DNCNN_PIXEL_DRAWING=OFF \\
 -DNCNN_INT8=OFF -DNCNN_BF16=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \\
 '-DCMAKE_C_FLAGS=-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard' \\
 '-DCMAKE_CXX_FLAGS=-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard' %s
cmake --build build -j 2
cmake --build build --target install
echo NCNN_BUILD_COMPLETE
""" % " ".join(
    flags
)
(out / "build_ncnn.sh").write_bytes(script.encode("utf8"))
(out / "ncnn_build_source.json").write_text(
    json.dumps(
        dict(
            tag="20260526",
            commit="e54f7b1f88434e1d844ea0551b880a1cfb079ce1",
            source="https://github.com/Tencent/ncnn",
            cortex="A9 ARMv7 NEON FP32, no VFPv4",
            enabled_layers=sorted(keep),
        ),
        indent=2,
    )
)
print("prepared", (out / "ncnn_source.tar.gz").stat().st_size)
