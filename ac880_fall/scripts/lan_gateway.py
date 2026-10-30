"""共享 RGB565 接收和 JPEG 编码的网页网关。"""

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import cache_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "board_runtime/app"))
import numpy as np
import requests
from home_web import HomeWeb
from alarm_store import AlarmStore


def session():
    value = requests.Session()
    value.trust_env = False
    return value


class RemoteOutputs:
    def __init__(self, base):
        self.base = base
        self.lock = threading.Lock()
        self.latest = None

    def command(self, payload):
        try:
            with session() as client:
                state = client.get(self.base + "/api/state", timeout=4)
                state.raise_for_status()
                result = client.post(
                    self.base + "/api/control",
                    json=payload,
                    headers={"X-Home-Token": state.json()["control_token"], "Origin": self.base},
                    timeout=4,
                )
                if not result.ok:
                    raise ValueError(result.json().get("error", "开发板控制失败"))
                return result.json()["revision"]
        except requests.RequestException as exc:
            raise ValueError("开发板暂时无法连接，请检查供电与网线") from exc

    def snapshot(self):
        with self.lock:
            if self.latest:
                return self.latest["controls"]
        return dict(
            desired=dict(ps_led="auto", led_auto=True, buzzer_auto=False),
            applied=dict(led_mask=0, ps_led=False, ps_led_available=False, buzzer=False),
            revision=0,
            applied_revision=0,
            events=[],
            alarm_count=0,
            muted_seconds=0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--board", default="http://192.168.2.99:8080")
    parser.add_argument(
        "--data", default=str(Path(__file__).resolve().parents[1] / "data/home_events")
    )
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--lan-port", type=int)
    args = parser.parse_args()
    base = args.board.rstrip("/")
    outputs = RemoteOutputs(base)
    archive = AlarmStore(args.data)
    web = HomeWeb(
        outputs,
        dict(
            host=args.host,
            port=args.port,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            max_width=args.max_width,
        ),
        archive=archive,
    )
    if args.lan_port is not None:
        web.add_listener("0.0.0.0", args.lan_port)
    with web.lock:
        web.state = dict(
            state_name="STARTING", running=False, ready=False, reason="board_connecting"
        )
    stopped = threading.Event()

    def poll_state():
        with session() as client:
            while not stopped.is_set():
                try:
                    response = client.get(base + "/api/state", timeout=4)
                    response.raise_for_status()
                    state = response.json()
                    with outputs.lock:
                        outputs.latest = state
                    with web.lock:
                        web.state = dict(state["detector"])
                        web.state_stamp = time.monotonic()
                        web.state["environment"] = state.get("environment")
                        if state.get("status_age_s", 0) > 3:
                            web.state.update(ready=False, reason="camera_status_stale")
                        if state.get("status_age_s", 0) > 3:
                            web.state["environment"] = None
                    with web.video.condition:
                        jpeg, stamp = web.video.jpeg, web.video.published_at
                    if archive.observe(state, jpeg, stamp):
                        web.video.frame(timeout=0.0)
                except (requests.RequestException, ValueError):
                    with web.lock:
                        web.state = dict(
                            state_name="STOPPED",
                            running=False,
                            ready=False,
                            reason="board_disconnected",
                        )
                        web.state_stamp = time.monotonic()
                stopped.wait(0.5)

    def video_source():
        with session() as client:
            while not stopped.is_set():
                if (
                    web.video.jpeg is not None
                    and web.video.viewers == 0
                    and time.monotonic() > web.video.demand_until
                ):
                    stopped.wait(0.2)
                    continue
                try:
                    with client.get(
                        base + "/stream.rgb565", stream=True, timeout=(4, 4)
                    ) as response:
                        response.raise_for_status()
                        stream = response.raw
                        while not stopped.is_set():
                            line = stream.readline(1024)
                            if not line:
                                break
                            if line.strip() != b"--homeframe":
                                continue
                            header_received_at = time.monotonic()
                            headers = {}
                            while True:
                                line = stream.readline(1024)
                                if line in (b"\r\n", b"\n", b""):
                                    break
                                key, _, value = line.partition(b":")
                                headers[key.strip().lower()] = value.strip()
                            length = int(headers.get(b"content-length", b"0"))
                            width = int(headers.get(b"x-width", b"1280"))
                            height = int(headers.get(b"x-height", b"720"))
                            if (width, height) not in (
                                (1280, 720),
                                (640, 360),
                            ) or length != width * height * 2:
                                raise ValueError("Unexpected RGB565 frame dimensions")
                            data = stream.read(length)
                            if len(data) != length:
                                raise ValueError("Truncated RGB565 frame")
                            source_at = (
                                header_received_at
                                - float(headers.get(b"x-frame-age-ms", b"0")) / 1000.0
                            )
                            web.video.submit(
                                np.frombuffer(data, dtype="<u2").reshape(height, width),
                                captured_at=source_at,
                            )
                            if web.video.viewers == 0 and time.monotonic() > web.video.demand_until:
                                break
                except (requests.RequestException, OSError, ValueError):
                    stopped.wait(1.0)

    threads = [threading.Thread(target=fn, daemon=True) for fn in (poll_state, video_source)]
    for thread in threads:
        thread.start()

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        "HOME_LAN_GATEWAY http://%s:%d -> %s (shared PC JPEG encoding)"
        % (args.host, args.port, base),
        flush=True,
    )
    try:
        stopped.wait()
    finally:
        web.close()
        for thread in threads:
            thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
