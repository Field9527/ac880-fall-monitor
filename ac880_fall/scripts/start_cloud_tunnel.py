"""手动启动并维护电脑到云端的隧道，不启动开发板。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import cache_env
from cloud_ssh import CloudSSH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-host", default="47.238.203.222")
    parser.add_argument("--cloud-user", default="root")
    parser.add_argument("--key", type=Path, default=Path("E:/Zynq_keys/aliyun/ac880_home_hk.pem"))
    parser.add_argument("--known-hosts", type=Path, default=Path("E:/Zynq_keys/aliyun/known_hosts"))
    parser.add_argument("--jump-host")
    parser.add_argument("--jump-user", default="ecs-user")
    parser.add_argument("--jump-key", type=Path, default=Path("E:/Zynq_keys/aliyun/ac880_home.pem"))
    parser.add_argument("--gateway-port", type=int, default=8089)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1024 <= args.gateway_port <= 65535 or args.gateway_port == 18089:
        parser.error("Invalid gateway port")
    args.output.mkdir(parents=True, exist_ok=True)
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open("http://127.0.0.1:%d/healthz" % args.gateway_port, timeout=5) as response:
        if not json.load(response).get("ok"):
            raise RuntimeError("Local web gateway is not ready")
    ssh = CloudSSH(
        args.cloud_host,
        args.cloud_user,
        args.key,
        args.known_hosts,
        args.output / "tunnel_transport.log",
        jump_host=args.jump_host,
        jump_user=args.jump_user,
        jump_key=args.jump_key,
    )
    command = ssh.command[:-1] + [
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-R",
        "127.0.0.1:18088:127.0.0.1:%d" % args.gateway_port,
        "-L",
        "127.0.0.1:18089:127.0.0.1:18089",
        ssh.command[-1],
    ]
    stopped = threading.Event()

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (args.output / "tunnel_supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    delay = 2
    while not stopped.is_set():
        print("Connecting cloud tunnel to " + args.cloud_host, flush=True)
        started = time.monotonic()
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        (args.output / "tunnel_ssh.pid").write_text(str(process.pid) + "\n", encoding="ascii")
        try:
            while process.poll() is None and not stopped.wait(0.5):
                pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if stopped.is_set():
            break
        delay = 2 if time.monotonic() - started >= 60 else min(delay * 2, 30)
        print("Tunnel exited (%s); retry in %ss" % (process.returncode, delay), flush=True)
        stopped.wait(delay)


if __name__ == "__main__":
    main()
