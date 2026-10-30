"""仅监听回环地址，按需拉取一路 MJPEG 并共享给观看端。"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class VideoRelay:
    def __init__(self, upstream="http://127.0.0.1:18088", stale_after=2.0):
        self.upstream = upstream.rstrip("/")
        self.stale_after = stale_after
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.viewers = 0
        self.sequence = 0
        self.jpeg = None
        self.published_at = 0.0
        self.capture_unix_ms = None
        self.connected = False
        self.error = None
        self.upstream_connections = 0
        self.state_error = None
        self.state_fetch_ms = None
        self.worker = threading.Thread(target=self.collect, name="video-upstream", daemon=True)
        self.worker.start()

    def client(self):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def collect(self):
        while not self.stopped.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.viewers > 0 or self.stopped.is_set(), 1.0)
                if self.stopped.is_set():
                    return
                if not self.viewers:
                    continue
            try:
                with self.client().open(self.upstream + "/stream.mjpg", timeout=5) as response:
                    with self.condition:
                        self.connected = True
                        self.error = None
                        self.upstream_connections += 1
                    while not self.stopped.is_set():
                        with self.condition:
                            if not self.viewers:
                                break
                        while True:
                            line = response.readline(1024)
                            if not line:
                                raise ConnectionError("Upstream video ended")
                            if line.strip() == b"--homeframe":
                                break
                        fields = {}
                        while True:
                            line = response.readline(1024)
                            if line in (b"\r\n", b"\n"):
                                break
                            if not line:
                                raise ConnectionError("Truncated video header")
                            key, _, value = line.partition(b":")
                            fields[key.lower()] = value.strip()
                        length = int(fields.get(b"content-length", b"0"))
                        if not 0 < length <= 5000000:
                            raise ValueError("Invalid upstream frame size")
                        jpeg = response.read(length)
                        if (
                            len(jpeg) != length
                            or jpeg[:2] != b"\xff\xd8"
                            or jpeg[-2:] != b"\xff\xd9"
                        ):
                            raise ValueError("Invalid upstream JPEG")
                        with self.condition:
                            self.jpeg = jpeg
                            self.sequence += 1
                            self.published_at = time.monotonic()
                            self.capture_unix_ms = (
                                float(fields[b"x-capture-unix-ms"])
                                if b"x-capture-unix-ms" in fields
                                else None
                            )
                            self.condition.notify_all()
            except (OSError, ValueError, urllib.error.URLError) as exc:
                with self.condition:
                    self.error = type(exc).__name__
                self.stopped.wait(1.0)
            finally:
                with self.condition:
                    self.connected = False
                    self.condition.notify_all()

    def latest(self, after, timeout=1.0, with_source=False):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopped.is_set()
                or self.sequence > after
                and time.monotonic() - self.published_at < self.stale_after,
                timeout,
            )
            if self.sequence <= after or time.monotonic() - self.published_at >= self.stale_after:
                return None
            value = (self.sequence, self.jpeg)
            return value + (self.capture_unix_ms,) if with_source else value

    def stats(self):
        with self.condition:
            return dict(
                viewers=self.viewers,
                sequence=self.sequence,
                upstream_connected=self.connected,
                upstream_connections=self.upstream_connections,
                error=self.error,
                state_error=self.state_error,
                state_fetch_ms=self.state_fetch_ms,
                source_age_s=(
                    round(time.time() - self.capture_unix_ms / 1000, 3)
                    if self.capture_unix_ms is not None
                    else None
                ),
                frame_age_s=round(time.monotonic() - self.published_at, 3) if self.jpeg else None,
            )

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=6)


def make_server(relay, address=("127.0.0.1", 18087)):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(8)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)

        def log_message(self, *_):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/healthz":
                return self.send_json(dict(ok=True, **relay.stats()))
            if path == "/api/state":
                started = time.monotonic()
                try:
                    with relay.client().open(relay.upstream + "/api/state", timeout=8) as response:
                        state = json.load(response)
                    with relay.condition:
                        relay.state_error = None
                        relay.state_fetch_ms = round((time.monotonic() - started) * 1000, 1)
                    state["cloud_video"] = relay.stats()
                    state["video"]["viewers"] = state["cloud_video"]["viewers"]
                    return self.send_json(state)
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    with relay.condition:
                        relay.state_error = type(exc).__name__
                        relay.state_fetch_ms = round((time.monotonic() - started) * 1000, 1)
                    return self.send_json(dict(error="Device gateway unavailable"), 503)
            if path != "/stream.mjpg":
                return self.send_json(dict(error="Not found"), 404)
            with relay.condition:
                if relay.viewers >= 12:
                    return self.send_json(dict(error="Too many viewers"), 429)
                relay.viewers += 1
                relay.condition.notify_all()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=homeframe")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                sequence = 0
                while not relay.stopped.is_set():
                    frame = relay.latest(sequence, with_source=True)
                    if frame is None:
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and not self.connection.recv(1, socket.MSG_PEEK):
                            break
                        continue
                    sequence, jpeg, captured = frame
                    timing = (
                        "X-Capture-Unix-Ms: %.1f\r\n" % captured if captured is not None else ""
                    )
                    header = (
                        "--homeframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\nX-Sequence: %d\r\n%s\r\n"
                        % (len(jpeg), sequence, timing)
                    ).encode()
                    self.wfile.write(header + jpeg + b"\r\n")
            except (OSError, TimeoutError):
                pass
            finally:
                with relay.condition:
                    relay.viewers -= 1
                    relay.condition.notify_all()
                self.close_connection = True

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return Server(address, Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()
    relay = VideoRelay()
    server = make_server(relay, ("127.0.0.1", args.port))
    stopped = threading.Event()

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        stopped.wait()
    finally:
        relay.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
