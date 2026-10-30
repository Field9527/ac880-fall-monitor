"""在外部设备验证 TLS、登录保护和新视频帧。凭据从标准输入传入，输出不含密码或控制令牌。"""

import base64
from datetime import datetime, timezone
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request


def verify(account):
    domain = "shuaidao.online"
    base = "https://" + domain
    auth = (
        "Basic "
        + base64.b64encode((account["username"] + ":" + account["password"]).encode()).decode()
    )
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        url=base,
        verification_location="external cloud instance over public HTTPS",
        mobile_network_verified=False,
    )
    with socket.create_connection((domain, 443), timeout=8) as tcp:
        with ssl.create_default_context().wrap_socket(tcp, server_hostname=domain) as tls:
            cert = tls.getpeercert()
            result.update(
                tls_verified=True, certificate_expires=cert["notAfter"], tls_version=tls.version()
            )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    plain = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        plain.open("http://" + domain + "/", timeout=8)
        raise AssertionError("HTTP must redirect to HTTPS")
    except urllib.error.HTTPError as exc:
        assert exc.code == 301 and exc.headers["Location"] == base + "/", exc.code
        result["http_redirect"] = exc.code
    routes = {}
    for path in (
        "/",
        "/app.js",
        "/app.css",
        "/api/state",
        "/snapshot.jpg",
        "/stream.mjpg",
        "/api/incidents",
        "/healthz",
    ):
        try:
            with client.open(base + path, timeout=8) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 401, (path, status)
        routes[path] = status
    result["unauthenticated_routes"] = routes
    for path in ("/", "/app.js", "/app.css", "/api/state", "/api/incidents", "/healthz"):
        request = urllib.request.Request(base + path, headers={"Authorization": auth})
        with client.open(request, timeout=8) as response:
            data = response.read()
            assert response.status == 200, (path, response.status)
        if path == "/":
            assert "实时设备监控".encode() in data and b"__HOME_TOKEN__" not in data
        if path == "/app.js":
            assert "本地连接".encode() not in data
        if path == "/api/state":
            state = json.loads(data)
            detector = state["detector"]
            assert detector["running"] and detector["ready"], "Board not ready"
            result.update(
                board_running=True,
                board_ready=True,
                board_state=detector.get("state_name"),
                status_age_s=state.get("status_age_s"),
            )
    request = urllib.request.Request(
        base + "/api/control",
        data=b"{}",
        method="POST",
        headers={"Authorization": auth, "Content-Type": "application/json", "Origin": base},
    )
    try:
        with client.open(request, timeout=8) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 403, status
    result["control_without_token"] = status
    request = urllib.request.Request(base + "/snapshot.jpg", headers={"Authorization": auth})
    with client.open(request, timeout=10) as response:
        image = response.read()
        assert image[:2] == b"\xff\xd8" and image[-2:] == b"\xff\xd9"
        result["snapshot_bytes"] = len(image)
    request = urllib.request.Request(base + "/stream.mjpg", headers={"Authorization": auth})
    frames, sizes = [], []
    started = time.monotonic()
    with client.open(request, timeout=10) as response:
        for _ in range(3):
            while True:
                line = response.readline(1024)
                if not line:
                    raise RuntimeError("Video stream ended")
                if line.strip() == b"--homeframe":
                    break
            headers = {}
            while True:
                line = response.readline(1024)
                if line in (b"\r\n", b"\n"):
                    break
                if not line:
                    raise RuntimeError("Video header truncated")
                key, _, value = line.partition(b":")
                headers[key.lower()] = value.strip()
            length = int(headers[b"content-length"])
            assert 0 < length < 5000000
            image = response.read(length)
            assert len(image) == length and image[:2] == b"\xff\xd8"
            frames.append(int(headers[b"x-sequence"]))
            sizes.append(length)
    assert frames[0] < frames[1] < frames[2], frames
    result.update(
        live_video_verified=True,
        mjpeg_sequences=frames,
        mjpeg_frame_bytes=sizes,
        stream_sample_seconds=round(time.monotonic() - started, 3),
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify(json.load(sys.stdin))))
