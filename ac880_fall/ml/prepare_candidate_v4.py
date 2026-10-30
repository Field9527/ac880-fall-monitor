import cache_env
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from fall_lite_model import FallLite

root = Path(__file__).resolve().parents[1]
modeldir = root / "models/fall_lite_v4_hardneg"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = json.loads((modeldir / "fall_lite.json").read_text())
meta["runtimes"] = {
    "ncnn": dict(
        param="fall_lite.ncnn.param",
        weights="fall_lite.ncnn.bin",
        weight_precision="fp32",
        source_onnx_sha256=meta["onnx_sha256"],
        param_sha256=sha(modeldir / "fall_lite.ncnn.param"),
        weights_sha256=sha(modeldir / "fall_lite.ncnn.bin"),
        library_sha256=sha(root / "sw/runtime/libncnn.so.1"),
        ncnn_tag="20260526",
        source_commit="e54f7b1f88434e1d844ea0551b880a1cfb079ce1",
    )
}
(modeldir / "fall_lite.json").write_text(json.dumps(meta, indent=2) + "\n")
torch.set_num_threads(4)
model = FallLite(depth=16)
model.load_state_dict(torch.load(modeldir / "best.pt", map_location="cpu")["model"])
model.eval()
x = np.random.RandomState(8).normal(size=(1, 3, 192, 320)).astype(np.float32)
with torch.no_grad():
    heads = model(torch.from_numpy(x))
np.savez(
    root / "out/false_alarm_speed_20261016/golden_v4.npz",
    image=x,
    **{name: y.numpy() for name, y in zip(("head8", "head16", "head32"), heads)},
)
cfg = json.loads((root / "sw/config_fall_lite.json").read_text())
cfg["fall_lite"].update(
    model="candidates/v4/fall_lite.onnx",
    metadata="candidates/v4/fall_lite.json",
    engine="ncnn",
    runtime_library="runtime/libncnn.so.1",
    threads=1,
)
(root / "out/false_alarm_speed_20261016/config_candidate_v4.json").write_text(
    json.dumps(cfg, indent=2) + "\n"
)
print("Candidate prepared", meta["onnx_sha256"])
