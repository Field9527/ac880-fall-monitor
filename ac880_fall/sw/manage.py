"""开发板手动启动与进程诊断；运行前不写硬件。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


def process_pid(root):
    path = root / "data/logs/detector.pid"
    try:
        pid = int(path.read_text())
        command = Path("/proc/%d/cmdline" % pid).read_bytes()
        if str(root / "app/fall_detect.py").encode() in command.split(b"\0"):
            return pid
    except (OSError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "status", "stop"))
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parent.parent
    cfg = root / "config.json"
    pid = process_pid(root)
    if args.action == "status":
        status = {}
        try:
            status = json.loads((root / "data/logs/status.json").read_text())
        except (OSError, ValueError):
            pass
        fresh = False
        if pid:
            log = root / "data/logs"
            fresh = (log / "status.json").exists() and (log / "status.json").stat().st_mtime >= (
                log / "detector.pid"
            ).stat().st_mtime
        print(
            json.dumps(
                dict(
                    running=pid is not None,
                    pid=pid,
                    camera_ready=bool(pid and fresh and status.get("ready")),
                    state=status.get("state_name") if fresh else "STARTING" if pid else "STOPPED",
                    error=status.get("error"),
                    web="http://192.168.2.99:8080" if pid else None,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "stop":
        if pid:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 25
            while process_pid(root) and time.monotonic() < deadline:
                time.sleep(0.1)
            if process_pid(root):
                raise RuntimeError("程序尚未退出；不能强杀仍持有DMA的进程")
        print("识别已停止。")
        return 0
    if pid:
        print("识别已经运行（PID %d），无需重复启动。用 ./run.sh status 查看状态。" % pid)
        return 0
    settings = json.loads(cfg.read_text(encoding="utf8"))
    for name in (
        "config.json",
        "assets/system.bit",
        "assets/system.hwh",
        settings["fall_lite"]["model"],
        settings["fall_lite"]["metadata"],
        "assets/libncnn.so.1",
    ):
        if not (root / name).is_file():
            raise RuntimeError("运行文件缺失: " + name)
    path = Path("/run/lock/ac880_fall.lock")
    with path.open("r" if path.exists() else "x") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("另一程序正在使用摄像头/DMA，请先停止它。", file=sys.stderr)
            return 2
    print("运行文件检查通过；启动OV5640、HDMI与识别模型。", flush=True)
    os.chdir(root)
    os.environ["XILINX_XRT"] = "/usr"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    (root / "data/logs").mkdir(parents=True, exist_ok=True)
    (root / "data/logs/detector.pid").write_text(str(os.getpid()))
    os.execv(
        sys.executable, [sys.executable, str(root / "app/fall_detect.py"), "-c", str(cfg)] + extra
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as exc:
        print("启动失败: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
