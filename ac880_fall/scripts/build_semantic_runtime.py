"""打包手动启动运行目录，不含训练数据和旧 PL 推理核。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile


def build(a):
    root = Path(__file__).resolve().parents[1]
    target = a.destination.resolve()
    meta = json.loads((a.model / "shared_semantic.json").read_text())
    if meta.get("architecture") != "v6_roi" or not meta.get("ncnn"):
        raise ValueError("Checked fall-region export required")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for filename, key in (
        ("shared_semantic.onnx", "onnx_sha256"),
        ("region_verifier.npz", "verifier_sha256"),
    ):
        if sha(a.model / filename) != meta[key]:
            raise ValueError("Model hash mismatch")
    if sha(a.library) != meta["ncnn"]["library_sha256"]:
        raise ValueError("Runtime hash mismatch")
    modules = ET.parse(a.hardware.with_suffix(".hwh")).findall(".//MODULE")
    if any("pointwise:" in m.get("VLNV", "") or "cnn_accel:" in m.get("VLNV", "") for m in modules):
        raise ValueError("Use the checked video/peripheral-only overlay")
    target.mkdir(parents=True, exist_ok=False)
    for folder in (
        "app/web",
        "assets/models/gestures",
        "assets/licenses",
        "data/logs",
        "data/snapshots",
    ):
        (target / folder).mkdir(parents=True, exist_ok=True)
    names = (
        "manage",
        "fall_detect",
        "realtime",
        "pose_rules",
        "fall_lite",
        "ncnn_backend",
        "frame_copy",
        "image_processing",
        "home_web",
        "environment",
        "semantic_features",
        "semantic_decode",
        "shared_detector",
        "tracked_fall",
        "hand_models",
        "hand_geometry",
        "gesture_control",
        "dual_inference",
        "hand_region_review",
    )
    for name in names:
        shutil.copyfile(root / "sw" / (name + ".py"), target / "app" / (name + ".py"))
    for path in (root / "sw/web").iterdir():
        if path.is_file():
            shutil.copyfile(path, target / "app/web" / path.name)
    for name in (
        "shared_semantic.onnx",
        "shared_semantic.json",
        "shared_semantic.ncnn.param",
        "shared_semantic.ncnn.bin",
        "region_verifier.npz",
    ):
        shutil.copyfile(a.model / name, target / "assets/models" / name)
    if meta.get("reference_verifier"):
        reference = meta["reference_verifier"]
        name = reference["file"]
        if Path(name).name != name or sha(a.model / name) != reference["sha256"]:
            raise ValueError("Reference verifier hash mismatch")
        shutil.copyfile(a.model / name, target / "assets/models" / name)
    gesture = json.loads((a.gesture / "gesture.json").read_text())
    if not gesture.get("pc_parity") or gesture.get("library_sha256") != sha(a.library):
        raise ValueError("Checked gesture model/runtime required")
    for cfg_model in gesture["models"].values():
        for key in ("param", "weights"):
            if sha(a.gesture / cfg_model[key]) != cfg_model[key + "_sha256"]:
                raise ValueError("Gesture artifact changed")
            shutil.copyfile(
                a.gesture / cfg_model[key], target / "assets/models/gestures" / cfg_model[key]
            )
    for name in ("gesture.json", "source_manifest.json"):
        shutil.copyfile(a.gesture / name, target / "assets/models/gestures" / name)
    for ext in (".bit", ".hwh"):
        shutil.copyfile(a.hardware.with_suffix(ext), target / "assets" / ("system" + ext))
    shutil.copyfile(a.library, target / "assets/libncnn.so.1")
    shutil.copyfile(
        root / "sw/runtime/licenses/ncnn-LICENSE.txt", target / "assets/licenses/ncnn-LICENSE.txt"
    )
    shutil.copyfile(
        root / "sw/runtime/licenses/opencv-zoo-Apache-2.0.txt",
        target / "assets/licenses/opencv-zoo-Apache-2.0.txt",
    )
    shutil.copyfile(root / "sw/runtime/DATA_SOURCES.md", target / "assets/DATA_SOURCES.md")
    with (target / "assets/DATA_SOURCES.md").open("a", encoding="utf8") as f:
        f.write(
            "\nHand detection and 21 landmarks: OpenCV Zoo commit48b10fa505aeb51fbd127fc60a5c660040e823b7, Apache-2.0; source hashes in models/gestures/source_manifest.json. "
            "FreiHAND is used for coarse hand-region research supervision under its research/noncommercial terms; no original images are packaged.\n"
        )
    cfg = json.loads(a.config.read_text(encoding="utf8"))
    board = a.board_root.rstrip("/")
    cfg.pop("rules", None)
    cfg.pop("cnn_accel", None)
    cfg["_comment"] = "Fall-region and gesture-control trial; manual start; no empty background"
    cfg["video"]["poll_hz"] = 12
    cfg["overlay_path"] = board + "/assets/system.bit"
    cfg["logging"].update(
        events_file=board + "/data/events.jsonl",
        snapshot_dir=board + "/data/snapshots",
        log_file=board + "/data/logs/run.log",
    )
    cfg["output"].pop("http_webhook", None)
    cfg["output"].pop("mqtt_topic", None)
    cfg["fall_lite"] = dict(
        model="assets/models/shared_semantic.onnx",
        metadata="assets/models/shared_semantic.json",
        runtime_library="assets/libncnn.so.1",
        engine="semantic",
        allow_experimental=True,
        threads=1,
        score_threshold=0.7,
        max_verify=16,
        gesture_model="assets/models/gestures",
        gesture_interval_s=0.5,
        partition_cpus=False,
    )
    cfg["image_processing"] = dict(mode="color", brightness=0, contrast=100)
    cfg["gesture_control"] = dict(hold_s=0.5)
    (target / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    script = """#!/bin/bash
set -e
cd "$(dirname "$0")"
export XILINX_XRT=/usr PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
action=run
case "${1:-}" in status|stop) action="$1"; shift;; esac
py=/usr/local/share/pynq-venv/bin/python3
if [ "$(id -u)" -eq 0 ]; then exec "$py" app/manage.py "$action" "$@"; fi
exec sudo -E env XILINX_XRT=/usr PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" app/manage.py "$action" "$@"
"""
    (target / "run.sh").write_bytes(script.encode("utf8"))
    for name, options in (("run_home.sh", ""), ("run_fall.sh", "--no-web ")):
        (target / name).write_bytes(
            ('#!/bin/bash\ncd "$(dirname "$0")"\nexec ./run.sh ' + options + '"$@"\n').encode(
                "utf8"
            )
        )
    (target / "README.md").write_text(
        """# AC880 摔倒监护与手势控制试运行

./run.sh 手动启动；./run.sh status 查看；./run.sh stop 安全停止。
run_home.sh 等同 run.sh；run_fall.sh 仅 HDMI。不开机自启。
OV5640 720p、PL 图像处理、DHT11 与 LED/蜂鸣器控制保留。
CPU 摔倒分支沿用 v6 权重；小型共享特征复核器判断手部、局部人体等。
张开手掌：全部开灯；握拳：全部关灯；V手势：短鸣0.5秒；停止鸣叫请用网页按钮。
单手完整入镜，保持手势至连续确认；保持不动只执行一次。网页可关闭手势控制。
手机回放模式暂停手势控制；网页手动控制始终可用。手部检测不能直接否决摔倒报警。
网页手机回放模式可手动设置输入方向；不会自动猜测重力方向。
重复回归通过不代表现场验收通过。原始训练数据和检查点不包含在运行包中。
""",
        encoding="utf8",
    )
    entries = {
        p.relative_to(target).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in target.rglob("*")
        if p.is_file()
    }
    (target / "manifest.json").write_text(
        json.dumps(
            dict(
                files=entries,
                training_run=meta["training_run"],
                model_sha256=meta["onnx_sha256"],
                experimental=True,
            ),
            indent=2,
        )
        + "\n"
    )
    with zipfile.ZipFile(target.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(target).as_posix())
    print(
        json.dumps(
            dict(
                directory=str(target),
                files=len(entries),
                bytes=sum(v["bytes"] for v in entries.values()),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gesture", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--hardware", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--board-root", default="/home/xilinx/fall_semantic_trial")
    build(p.parse_args())
