"""局域网监控页面，共享视频编码，硬件控制由主线程执行。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

import numpy as np


class HomeOutputs:
    MAGIC = 0x484F4D31

    def __init__(self, mmio, ps_led, logger=None, image_processing=None, semantic=None):
        if mmio.read(0x100) != self.MAGIC:
            raise RuntimeError("Home I/O bitstream required (HOM1 register missing)")
        self.mmio, self.ps_led = mmio, ps_led
        self.logger = logger or logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.desired = dict(led_mask=0, led_auto=True, ps_led="auto", buzzer_auto=False)
        self.revision = self.applied_revision = 0
        self.manual_until = self.auto_until = self.muted_until = 0.0
        self.last_alarm = False
        self.alarm_count = 0
        self.last_write = -float("inf")
        self.last_control = None
        self.applied = dict(led_mask=0, ps_led=False, buzzer=False, watchdog_expired=False)
        self.events = deque(maxlen=24)
        self.image_processing = image_processing
        self.image_desired = dict(image_processing.configuration) if image_processing else None
        self.semantic = semantic
        self.semantic_desired = dict(semantic.configuration) if semantic else None

    def command(self, value):
        if not isinstance(value, dict) or not value:
            raise ValueError("请输入有效控制命令")
        allowed = {
            "led",
            "led_mask",
            "led_auto",
            "ps_led",
            "buzzer",
            "buzzer_auto",
            "image",
            "semantic",
        }
        if set(value) - allowed:
            raise ValueError("不支持的控制项目")
        # 完整校验请求后再修改共享状态。
        update = {}
        if "led" in value:
            led = value["led"]
            if (
                not isinstance(led, dict)
                or set(led) != {"index", "on"}
                or type(led["index"]) is not int
                or not 0 <= led["index"] <= 7
                or type(led["on"]) is not bool
            ):
                raise ValueError("LED编号应为0～7，状态应为开或关")
        if "led_mask" in value:
            if type(value["led_mask"]) is not int or not 0 <= value["led_mask"] <= 255:
                raise ValueError("LED掩码应为0～255")
            update["led_mask"] = value["led_mask"]
            update["led_auto"] = False
        if "led_auto" in value:
            if type(value["led_auto"]) is not bool:
                raise ValueError("自动模式必须为布尔值")
            update["led_auto"] = value["led_auto"]
        if "ps_led" in value:
            if value["ps_led"] not in ("auto", "on", "off"):
                raise ValueError("报警灯模式无效")
            update["ps_led"] = value["ps_led"]
        if "buzzer_auto" in value:
            if type(value["buzzer_auto"]) is not bool:
                raise ValueError("报警联动必须为布尔值")
            update["buzzer_auto"] = value["buzzer_auto"]
        if "buzzer" in value:
            buzzer = value["buzzer"]
            if (
                not isinstance(buzzer, dict)
                or set(buzzer) - {"on", "duration_ms"}
                or type(buzzer.get("on")) is not bool
            ):
                raise ValueError("蜂鸣器命令无效")
            duration = buzzer.get("duration_ms", 2000)
            if type(duration) is not int or not 100 <= duration <= 5000:
                raise ValueError("鸣叫时间应为100～5000毫秒")
        with self.lock:
            image_update = None
            semantic_update = None
            if "image" in value:
                if self.image_processing is None:
                    raise ValueError("当前硬件不支持图像处理")
                image_update = self.image_processing.validate(value["image"], self.image_desired)
            if "semantic" in value:
                if self.semantic is None:
                    raise ValueError("当前模型不支持语义设置")
                semantic_update = self.semantic.validate(value["semantic"], self.semantic_desired)
            now = time.monotonic()
            if "led" in value:
                # 从当前实际输出开始手动控制，避免使用旧掩码。
                mask = (
                    self.applied["led_mask"]
                    if self.desired["led_auto"]
                    else self.desired["led_mask"]
                )
                bit = 1 << value["led"]["index"]
                update["led_mask"] = (mask | bit) if value["led"]["on"] else (mask & ~bit)
                update["led_auto"] = False
            self.desired.update(update)
            if image_update is not None:
                self.image_desired = image_update
            if semantic_update is not None:
                self.semantic_desired = semantic_update
            if "buzzer" in value:
                if value["buzzer"]["on"]:
                    self.manual_until = now + value["buzzer"].get("duration_ms", 2000) / 1000.0
                    self.muted_until = 0.0
                else:
                    self.manual_until = self.auto_until = 0.0
                    self.muted_until = now + 60.0
            if value.get("buzzer_auto") is False:
                self.auto_until = 0.0
            self.revision += 1
            self.events.appendleft(dict(kind="control", message="设备控制已提交", at=now))
            self.logger.info(
                "HOME_CONTROL revision=%d %s", self.revision, json.dumps(value, ensure_ascii=False)
            )
            return self.revision

    def tick(self, alarm):
        # 仅相机主线程访问 MMIO 和 GPIO。
        with self.lock:
            now = time.monotonic()
            if self.image_processing is not None:
                self.image_processing.configure(self.image_desired)
            if self.semantic is not None:
                self.semantic.configure(self.semantic_desired)
            if alarm and not self.last_alarm:
                self.alarm_count += 1
                self.events.appendleft(
                    dict(kind="alarm", message="检测到持续倒地，触发报警", at=now)
                )
                if self.desired["buzzer_auto"] and now >= self.muted_until:
                    self.auto_until = now + 2.0
            self.last_alarm = alarm
            sounding = now < max(self.manual_until, self.auto_until) and now >= self.muted_until
            control = (
                self.desired["led_mask"]
                | (0 if self.desired["led_auto"] else 1 << 9)
                | (int(sounding) << 8)
            )
            if control != self.last_control or now - self.last_write >= 0.5:
                self.mmio.write(0x104, control)
                if self.mmio.read(0x104) != control:
                    raise RuntimeError("Home output register readback mismatch")
                self.last_write = now
                self.last_control = control
            mode = self.desired["ps_led"]
            self.ps_led.set(alarm if mode == "auto" else mode == "on")
            actual = self.mmio.read(0x108)
            self.applied = dict(
                led_mask=actual & 255,
                ps_led=bool(self.ps_led.state),
                ps_led_available=self.ps_led.path is not None,
                buzzer=bool(actual & 256),
                watchdog_expired=bool(actual & 512),
            )
            self.applied_revision = self.revision

    def gesture_command(self, value, label):
        # 时序确认后由相机主线程执行控制。
        revision = self.command(value)
        with self.lock:
            self.events.appendleft(
                dict(kind="gesture", message="手势：" + label, at=time.monotonic())
            )
        return revision

    def recognition_changed(self, mode):
        with self.lock:
            # 切换时结束旧报警或手势短鸣，保留用户 LED 的当前状态。
            self.manual_until = self.auto_until = 0.0
            self.last_alarm = False
            names = {"fall": "摔倒模式", "gesture": "手势模式", "paused": "识别暂停"}
            self.events.appendleft(
                dict(kind="mode", message="拨码切换：" + names[mode], at=time.monotonic())
            )

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            return dict(
                desired=dict(self.desired),
                applied=dict(self.applied),
                revision=self.revision,
                applied_revision=self.applied_revision,
                alarm_count=self.alarm_count,
                image=(
                    self.image_processing.snapshot()
                    if self.image_processing
                    else dict(available=False)
                ),
                semantic=self.semantic.snapshot() if self.semantic else dict(available=False),
                muted_seconds=max(0, round(self.muted_until - now)),
                events=[
                    dict(kind=e["kind"], message=e["message"], age_s=round(now - e["at"]))
                    for e in self.events
                ],
            )

    def close(self):
        self.mmio.write(0x104, 0)
        self.ps_led.set(False)


class SharedVideo:
    """最多保留一个在途图像，慢观看端跳过旧帧。"""

    def __init__(self, fps=6, quality=82, max_clients=12, codec="jpeg", max_width=None):
        self.fps, self.quality, self.max_clients = float(fps), int(quality), int(max_clients)
        if codec not in ("jpeg", "rgb565"):
            raise ValueError("Unsupported stream codec")
        self.codec = codec
        self.max_width = max_width
        if max_width is not None and (type(max_width) is not int or not 320 <= max_width <= 1280):
            raise ValueError("Invalid video width")
        if not 1 <= self.fps <= 30 or not 50 <= self.quality <= 95:
            raise ValueError("Invalid LAN video settings")
        self.condition = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.jpeg = None
        self.sequence = 0
        self.viewers = 0
        self.closed = False
        self.captured_at = 0.0
        self.resolution = [1280, 720]
        self.last_submit = -float("inf")
        self.next_submit = -float("inf")
        self.published_at = 0.0
        self.demand_until = 0.0
        self.publish_times = deque(maxlen=36)
        self.encode_ms = 0.0
        self.error = None
        self.skipped = 0

    def submit(self, frame, captured_at=None):
        now = time.monotonic()
        captured_at = now if captured_at is None else captured_at
        with self.condition:
            needed = self.jpeg is None or self.viewers > 0 or now < self.demand_until
        if self.closed or not needed or now < self.next_submit:
            return
        if self.future is not None and not self.future.done():
            self.skipped += 1
            return
        self.last_submit = now
        period = 1 / self.fps
        self.next_submit = (
            now + period if now - self.next_submit > period else self.next_submit + period
        )
        if self.codec == "rgb565":
            # 在显示缓冲区释放前固定图像字节。
            # 原始传输无需压缩线程或中间数组。
            self._encode(frame, captured_at)
        else:
            owned = np.array(frame, copy=True)
            self.future = self.pool.submit(self._encode, owned, captured_at)

    def _encode(self, frame, captured_at=None):
        try:
            before = time.monotonic()
            if frame.dtype != np.uint16 or frame.ndim != 2:
                raise ValueError("Expected RGB565 display frame")
            if self.codec == "rgb565":
                if self.max_width is not None and frame.shape[1] > self.max_width:
                    stride = (frame.shape[1] + self.max_width - 1) // self.max_width
                    # 先复制完整选定行，再按列抽样。
                    # 避免逐像素读取非缓存内存，同时保持像素与所有权。
                    rows = np.array(frame[::stride, :], copy=True, order="C")
                    frame = np.ascontiguousarray(rows[:, ::stride])
                encoded = frame.astype("<u2", copy=False).tobytes()
            else:
                import cv2

                if self.max_width is not None and frame.shape[1] > self.max_width:
                    height = round(frame.shape[0] * self.max_width / frame.shape[1])
                    # RGB565 先展开通道再缩放，避免位域插值损坏颜色。
                    bgr = cv2.cvtColor(
                        frame.view(np.uint8).reshape(frame.shape + (2,)), cv2.COLOR_BGR5652BGR
                    )
                    bgr = cv2.resize(bgr, (self.max_width, height), interpolation=cv2.INTER_AREA)
                else:
                    bgr = cv2.cvtColor(
                        frame.view(np.uint8).reshape(frame.shape + (2,)), cv2.COLOR_BGR5652BGR
                    )
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError("JPEG encoder failed")
                encoded = jpeg.tobytes()
            with self.condition:
                self.jpeg = encoded
                self.sequence += 1
                self.published_at = time.monotonic()
                self.captured_at = self.published_at if captured_at is None else captured_at
                self.resolution = (
                    [frame.shape[1], frame.shape[0]]
                    if self.codec == "rgb565"
                    else [bgr.shape[1], bgr.shape[0]]
                )
                self.encode_ms = (self.published_at - before) * 1000
                if self.publish_times and self.published_at - self.publish_times[-1] > 2:
                    self.publish_times.clear()
                self.publish_times.append(self.published_at)
                self.error = None
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def frame(self, after=0, timeout=3.0, with_source=False):
        with self.condition:
            self.demand_until = time.monotonic() + 5.0
            self.condition.wait_for(lambda: self.sequence > after or self.closed, timeout)
            value = (self.sequence, self.jpeg, self.published_at)
            return value + (self.captured_at,) if with_source else value

    def stats(self):
        with self.condition:
            now = time.monotonic()
            times = list(self.publish_times)
            fps = (
                (len(times) - 1) / (times[-1] - times[0])
                if len(times) > 1 and now - self.published_at < 2
                else 0.0
            )
            return dict(
                viewers=self.viewers,
                sequence=self.sequence,
                fps=round(fps, 2),
                limit_fps=self.fps,
                resolution=list(self.resolution),
                quality=self.quality,
                encode_ms=round(self.encode_ms, 2),
                capture_age_s=round(now - self.captured_at, 3) if self.jpeg else None,
                codec=self.codec,
                frame_age_s=round(now - self.published_at, 2) if self.jpeg else None,
                jpeg_bytes=len(self.jpeg) if self.jpeg else 0,
                error=self.error,
                skipped=self.skipped,
            )

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.pool.shutdown(wait=True)


class HomeWeb:
    def __init__(self, outputs, config, logger=None, archive=None):
        self.outputs = outputs
        self.logger = logger or logging.getLogger(__name__)
        self.token = secrets.token_urlsafe(24)
        self.run_id = secrets.token_hex(12)
        self.archive = archive
        self.video = SharedVideo(
            config.get("fps", 6),
            config.get("jpeg_quality", 82),
            config.get("max_viewers", 12),
            config.get("codec", "jpeg"),
            config.get("max_width"),
        )
        self.jpeg_fallback = None
        self.lock = threading.Lock()
        self.state = dict(state_name="STARTING", running=True, ready=False)
        self.state_stamp = time.monotonic()
        self.assets = Path(__file__).with_name("web")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(5.0)
                # 立即发送较小的分段头尾，避免等待延迟确认。
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if owner.video.codec == "jpeg":
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)

            def log_message(self, fmt, *args):
                owner.logger.debug("HOME_HTTP " + fmt, *args)

            def send_bytes(self, data, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "same-origin")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(data)

            def send_json(self, value, status=200):
                self.send_bytes(
                    json.dumps(value, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                    status,
                )

            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path in ("/", "/app.css", "/app.js", "/video_stream.js"):
                        name = "index.html" if path == "/" else path[1:]
                        data = (owner.assets / name).read_bytes()
                        if name == "index.html":
                            data = data.replace(b"__HOME_TOKEN__", owner.token.encode())
                        kind = {
                            "index.html": "text/html",
                            "app.css": "text/css",
                            "app.js": "text/javascript",
                            "video_stream.js": "text/javascript",
                        }[name]
                        return self.send_bytes(data, kind + "; charset=utf-8")
                    if path == "/api/state":
                        return self.send_json(owner.snapshot())
                    if path == "/api/incidents":
                        return self.send_json(
                            dict(
                                items=owner.archive.list() if owner.archive else [],
                                available=owner.archive is not None,
                            )
                        )
                    if (
                        path.startswith("/api/incidents/")
                        and path.endswith("/snapshot.jpg")
                        and owner.archive
                    ):
                        try:
                            data = owner.archive.snapshot(path.split("/")[3])
                        except ValueError:
                            return self.send_json(dict(error="报警编号无效"), 400)
                        return (
                            self.send_bytes(data, "image/jpeg")
                            if data
                            else self.send_json(dict(error="暂无报警截图"), 404)
                        )
                    if path == "/healthz":
                        return self.send_json(dict(ok=True))
                    if path == "/snapshot.jpg":
                        hub = owner.jpeg_hub()
                        with hub.condition:
                            after = hub.sequence if time.monotonic() - hub.published_at > 1.0 else 0
                        _, data, stamp = hub.frame(after, timeout=2.0)
                        return (
                            self.send_bytes(data, "image/jpeg")
                            if data and time.monotonic() - stamp < 2.0
                            else self.send_json(dict(error="等待摄像头新画面"), 503)
                        )
                    if path == "/stream.mjpg":
                        return self.stream(owner.jpeg_hub())
                    if path == "/stream.rgb565" and owner.video.codec == "rgb565":
                        return self.stream(owner.video)
                    return self.send_json(dict(error="页面不存在"), 404)
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    ConnectionAbortedError,
                    TimeoutError,
                    socket.timeout,
                ):
                    return

            def stream(self, hub):
                with hub.condition:
                    if hub.viewers >= hub.max_clients:
                        return self.send_json(dict(error="观看设备较多，请稍后重试"), 429)
                    hub.viewers += 1
                try:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=homeframe"
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sequence = 0
                    while not hub.closed:
                        current, jpeg, stamp, captured_at = hub.frame(sequence, with_source=True)
                        if not jpeg or current == sequence:
                            continue
                        sequence = current
                        if time.monotonic() - stamp >= 2.0:
                            continue
                        kind = "image/jpeg" if hub.codec == "jpeg" else "application/octet-stream"
                        age = max(0.0, time.monotonic() - captured_at)
                        capture_unix_ms = (time.time() - age) * 1000
                        width, height = hub.resolution
                        head = (
                            "--homeframe\r\nContent-Type: %s\r\nContent-Length: %d\r\nX-Sequence: %d\r\nX-Width: %d\r\nX-Height: %d\r\nX-Frame-Age-Ms: %.1f\r\nX-Capture-Unix-Ms: %.1f\r\n\r\n"
                            % (kind, len(jpeg), current, width, height, age * 1000, capture_unix_ms)
                        ).encode()
                        self.wfile.write(head)
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                finally:
                    with hub.condition:
                        hub.viewers -= 1
                    self.close_connection = True

            def do_POST(self):
                path = urlsplit(self.path).path
                review = (
                    path.startswith("/api/incidents/")
                    and path.endswith("/review")
                    and owner.archive is not None
                )
                if path != "/api/control" and not review:
                    return self.send_json(dict(error="接口不存在"), 404)
                origin = self.headers.get("Origin")
                if (
                    not secrets.compare_digest(self.headers.get("X-Home-Token", ""), owner.token)
                    or origin
                    and urlsplit(origin).netloc != self.headers.get("Host")
                ):
                    # 关闭前限量读完请求体，避免复位掩盖 403 响应。
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if 0 < length <= 2048:
                            self.rfile.read(length)
                    except (ValueError, TimeoutError, socket.timeout):
                        pass
                    self.close_connection = True
                    return self.send_json(dict(error="请从本机网页重新打开控制面板"), 403)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2048:
                        raise ValueError("请求长度不合法")
                    if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        raise ValueError("需要JSON控制请求")
                    value = json.loads(self.rfile.read(length).decode("utf8"))
                    if review:
                        return self.send_json(owner.archive.review(path.split("/")[3], value))
                    revision = owner.outputs.command(value)
                    return self.send_json(dict(accepted=True, revision=revision), 202)
                except (ValueError, UnicodeError) as exc:
                    return self.send_json(dict(error=str(exc)), 400)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(
            (config.get("host", "0.0.0.0"), int(config.get("port", 8080))), Handler
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="home-http", daemon=True
        )
        self.thread.start()
        self.listeners = []
        self.logger.info("局域网控制面板: http://0.0.0.0:%d", self.server.server_port)

    def add_listener(self, host, port):
        """不同访问端口共用状态、视频和编码线程。"""
        server = type(self.server)((host, port), self.server.RequestHandlerClass)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.listeners.append((server, thread))
        return server.server_port

    def publish(self, frame, status=None):
        self.video.submit(frame)
        if self.jpeg_fallback is not None:
            self.jpeg_fallback.submit(frame)
        if status is not None:
            with self.lock:
                self.state = dict(status)
                self.state_stamp = time.monotonic()

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            age = time.monotonic() - self.state_stamp
        environment = state.pop("environment", None)
        if environment is not None and (not state.get("running") or age > 3):
            environment = dict(
                environment,
                valid=False,
                stale=True,
                temperature_c=None,
                humidity_rh=None,
                message="开发板环境数据已离线",
            )
        hub = (
            self.jpeg_fallback
            if self.video.codec == "rgb565" and self.jpeg_fallback is not None
            else self.video
        )
        return dict(
            detector=state,
            controls=self.outputs.snapshot(),
            video=hub.stats(),
            acquisition_video=self.video.stats(),
            environment=environment,
            control_token=self.token,
            run_id=self.run_id,
            status_age_s=round(age, 2),
        )

    def jpeg_hub(self):
        if self.video.codec == "jpeg":
            return self.video
        with self.lock:
            if self.jpeg_fallback is None:
                self.jpeg_fallback = SharedVideo(
                    fps=2, quality=self.video.quality, max_clients=self.video.max_clients
                )
            return self.jpeg_fallback

    def close(self):
        self.video.close()
        if self.jpeg_fallback:
            self.jpeg_fallback.close()
        for server, thread in self.listeners:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3.0)
