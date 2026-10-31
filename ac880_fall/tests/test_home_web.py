"""验证网页协议、输出所有权和共享视频，无需连接硬件。"""

import http.client
from io import BytesIO
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from home_web import HomeOutputs, HomeWeb, SharedVideo


class MMIO:
    def __init__(self):
        self.control = 0
        self.writes = []

    def read(self, offset):
        return {
            0x100: 0x484F4D31,
            0x104: self.control,
            0x108: (self.control & 255 if self.control & 512 else 1) | (self.control & 256),
        }[offset]

    def write(self, offset, value):
        self.control = value
        self.writes.append((threading.get_ident(), offset, value))


class LED:
    state = False
    path = "fake"

    def set(self, value):
        self.state = bool(value)


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.mmio = MMIO()
        self.outputs = HomeOutputs(self.mmio, LED())

    def test_gateway_ports_share_pixels_state_and_video_client_code(self):
        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0))
        extra = web.add_listener("127.0.0.1", 0)
        clients = [
            http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            for port in (web.server.server_port, extra)
        ]
        try:
            web.publish(None, dict(running=True, ready=True, state_name="IDLE"))
            values = []
            for client in clients:
                client.request("GET", "/api/state")
                response = client.getresponse()
                self.assertEqual(response.status, 200)
                values.append(json.loads(response.read()))
                client.request("GET", "/video_stream.js")
                response = client.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("javascript", response.getheader("Content-Type"))
                self.assertIn(b"LatestDecoder", response.read())
            self.assertEqual(values[0]["control_token"], values[1]["control_token"])
            self.assertEqual(values[0]["run_id"], values[1]["run_id"])
        finally:
            for client in clients:
                client.close()
            web.close()

    def test_raw_video_snapshot_survives_owner_reuse_and_strided_input(self):
        video = SharedVideo(fps=30, codec="rgb565")
        pixels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
        view = pixels[::2, ::2]
        expected = view.astype("<u2", copy=False).tobytes()
        try:
            video.submit(view)
            pixels.fill(0)
            sequence, data, _ = video.frame(timeout=0.5)
            self.assertEqual(sequence, 1)
            self.assertEqual(data, expected)
        finally:
            video.close()

    def test_remote_resize_preserves_source_timestamp_and_reports_real_dimensions(self):
        video = SharedVideo(fps=30, max_width=640)
        captured = time.monotonic() - 0.4
        try:
            video.submit(np.full((720, 1280), 0xF800, np.uint16), captured_at=captured)
            _, data, _, source = video.frame(timeout=2, with_source=True)
            self.assertEqual(source, captured)
            self.assertEqual(Image.open(BytesIO(data)).size, (640, 360))
            self.assertEqual(video.stats()["resolution"], [640, 360])
            self.assertGreaterEqual(video.stats()["capture_age_s"], 0.4)
        finally:
            video.close()

    def test_raw_downsample_sends_expected_pixels_without_changing_owner(self):
        video = SharedVideo(fps=30, codec="rgb565", max_width=640)
        pixels = np.arange(720 * 1280, dtype=np.uint16).reshape(720, 1280)
        expected = pixels[::2, ::2].astype("<u2", copy=False).tobytes()
        try:
            video.submit(pixels)
            pixels.fill(0)
            _, data, _ = video.frame(timeout=1)
            self.assertEqual(data, expected)
            self.assertEqual(video.stats()["resolution"], [640, 360])
        finally:
            video.close()

    def test_confirmed_gesture_uses_existing_owner_and_bounded_buzzer(self):
        from gesture_control import GestureController, ACTIONS

        now = [100.0]
        controller = GestureController()
        hand = dict(name="victory", score=0.99, bbox=[10, 10, 80, 80])
        with patch("home_web.time.monotonic", side_effect=lambda: now[0]):
            self.assertIsNone(controller.step(hand, 1, 99, 99))
            value = controller.step(hand, 2, 100, 100)
            self.outputs.gesture_command(value, ACTIONS["victory"])
            self.assertEqual(self.mmio.writes, [])
            self.outputs.tick(False)
            self.assertTrue(self.outputs.snapshot()["applied"]["buzzer"])
            now[0] = 100.6
            self.outputs.tick(False)
            self.assertFalse(self.outputs.snapshot()["applied"]["buzzer"])
            self.assertIsNone(controller.step(hand, 3, 101, 101))
            self.assertEqual(self.outputs.snapshot()["events"][0]["kind"], "gesture")

    def test_semantic_commands_are_atomic_and_applied_only_by_owner_tick(self):
        from shared_detector import SemanticOptions

        semantic = SemanticOptions()
        self.outputs = HomeOutputs(self.mmio, LED(), semantic=semantic)
        with self.assertRaises(ValueError):
            self.outputs.command(
                dict(led_mask=255, semantic=dict(mode="phone_replay", rotation=45))
            )
        self.assertEqual(self.outputs.desired["led_mask"], 0)
        self.outputs.command(dict(semantic=dict(mode="phone_replay", rotation=90)))
        self.assertEqual(semantic.configuration["mode"], "live")
        self.outputs.tick(False)
        self.assertEqual(semantic.configuration["rotation"], 90)
        self.assertEqual(semantic.configuration["generation"], 1)
        for _ in range(10):
            self.outputs.tick(False)
        self.assertEqual(semantic.configuration["generation"], 1)

    def test_switching_mode_ends_old_alarm_and_gesture_buzzer(self):
        self.outputs.command(dict(buzzer_auto=True, led_mask=255))
        self.outputs.tick(True)
        self.assertTrue(self.outputs.snapshot()["applied"]["buzzer"])
        self.outputs.recognition_changed("gesture")
        self.outputs.tick(False)
        value = self.outputs.snapshot()
        self.assertFalse(value["applied"]["buzzer"])
        self.assertFalse(value["applied"]["ps_led"])
        self.assertEqual(value["applied"]["led_mask"], 255)
        self.outputs.command(dict(buzzer=dict(on=True, duration_ms=500)))
        self.outputs.recognition_changed("fall")
        self.outputs.tick(False)
        self.assertFalse(self.outputs.snapshot()["applied"]["buzzer"])

    def test_environment_api_does_not_keep_live_values_when_board_status_is_stale(self):
        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0))
        try:
            reading = dict(
                available=True, valid=True, stale=False, temperature_c=26.3, humidity_rh=55.0
            )
            web.publish(None, dict(running=True, ready=True, environment=reading))
            data = web.snapshot()
            self.assertEqual(data["environment"]["temperature_c"], 26.3)
            self.assertNotIn("environment", data["detector"])
            with web.lock:
                web.state_stamp -= 4
            data = web.snapshot()
            self.assertFalse(data["environment"]["valid"])
            self.assertTrue(data["environment"]["stale"])
            self.assertIsNone(data["environment"]["temperature_c"])
            self.assertIsNone(data["environment"]["humidity_rh"])
        finally:
            web.close()

    def test_snapshot_waits_for_fresh_pixels_instead_of_returning_old_cache(self):
        from concurrent.futures import ThreadPoolExecutor

        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0, fps=30))
        client = http.client.HTTPConnection("127.0.0.1", web.server.server_port, timeout=3)
        try:
            with web.video.condition:
                web.video.jpeg = b"old"
                web.video.sequence = 1
                web.video.published_at = time.monotonic() - 20

            def snapshot():
                client.request("GET", "/snapshot.jpg")
                response = client.getresponse()
                return response.status, response.read()

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(snapshot)
                time.sleep(0.1)
                self.assertFalse(future.done())
                web.video.submit(np.full((36, 64), 0xF800, np.uint16))
                status, data = future.result(timeout=3)
            self.assertEqual(status, 200)
            self.assertEqual(Image.open(BytesIO(data)).size, (64, 36))
        finally:
            client.close()
            web.close()

    def test_stream_waits_for_new_pixels_after_idle_cache_expires(self):
        from concurrent.futures import ThreadPoolExecutor

        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0, fps=30))
        client = http.client.HTTPConnection("127.0.0.1", web.server.server_port, timeout=3)
        try:
            with web.video.condition:
                web.video.jpeg = b"old"
                web.video.sequence = 1
                web.video.published_at = time.monotonic() - 20

            def first_frame():
                client.request("GET", "/stream.mjpg")
                response = client.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.readline(), b"--homeframe\r\n")
                fields = {}
                while True:
                    line = response.readline()
                    if line == b"\r\n":
                        break
                    if not line:
                        raise AssertionError("Truncated video header")
                    key, _, value = line.partition(b":")
                    fields[key.lower()] = value.strip()
                return int(fields[b"x-sequence"]), response.read(int(fields[b"content-length"]))

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(first_frame)
                time.sleep(0.1)
                self.assertFalse(future.done())
                web.video.submit(np.full((36, 64), 0xF800, np.uint16))
                sequence, data = future.result(timeout=3)
            self.assertEqual(sequence, 2)
            self.assertEqual(Image.open(BytesIO(data)).size, (64, 36))
        finally:
            client.close()
            web.close()

    def test_all_or_nothing_validation_and_led_manual_from_actual(self):
        self.outputs.tick(False)
        with self.assertRaises(ValueError):
            self.outputs.command(dict(led_mask=255, buzzer=dict(on=True, duration_ms=99999)))
        self.assertEqual(self.outputs.desired["led_mask"], 0)
        self.outputs.command(dict(led=dict(index=1, on=True)))
        self.assertEqual(self.outputs.desired["led_mask"], 3)
        self.outputs.tick(False)
        self.assertEqual(self.outputs.snapshot()["applied"]["led_mask"], 3)

    def test_buzzer_deadline_mute_and_hardware_readback(self):
        now = [100.0]
        with patch("home_web.time.monotonic", side_effect=lambda: now[0]):
            self.outputs.command(dict(buzzer=dict(on=True, duration_ms=2000)))
            self.outputs.tick(False)
            self.assertTrue(self.outputs.snapshot()["applied"]["buzzer"])
            now[0] = 102.1
            self.outputs.tick(False)
            self.assertFalse(self.outputs.snapshot()["applied"]["buzzer"])
            self.outputs.command(dict(buzzer_auto=True))
            self.outputs.command(dict(buzzer=dict(on=False)))
            self.outputs.tick(True)
            self.assertFalse(self.outputs.snapshot()["applied"]["buzzer"])
            self.assertEqual(self.outputs.alarm_count, 1)
            now[0] = 163.0
            self.outputs.tick(False)
            self.outputs.tick(True)
            self.assertTrue(self.outputs.snapshot()["applied"]["buzzer"])
            self.outputs.close()
            self.assertEqual(self.mmio.control, 0)

    def test_mismatched_bit_rejected(self):
        with patch.object(self.mmio, "read", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "HOM1"):
                HomeOutputs(self.mmio, LED())

    def test_http_csrf_validation_owner_thread_and_static(self):
        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0))
        connection = http.client.HTTPConnection("127.0.0.1", web.server.server_port, timeout=2.0)
        try:
            connection.request("GET", "/")
            res = connection.getresponse()
            html = res.read()
            self.assertEqual(res.status, 200)
            self.assertIn(web.token.encode(), html)
            self.assertNotIn(b"__HOME_TOKEN__", html)
            connection.request(
                "POST",
                "/api/control",
                json.dumps(dict(led_mask=85)),
                {"Content-Type": "application/json"},
            )
            res = connection.getresponse()
            res.read()
            self.assertEqual(res.status, 403)
            connection.close()
            connection = http.client.HTTPConnection(
                "127.0.0.1", web.server.server_port, timeout=2.0
            )
            headers = {
                "Content-Type": "application/json",
                "X-Home-Token": web.token,
                "Origin": "http://evil.example",
            }
            connection.request("POST", "/api/control", json.dumps(dict(led_mask=85)), headers)
            res = connection.getresponse()
            res.read()
            self.assertEqual(res.status, 403)
            connection.close()
            connection = http.client.HTTPConnection(
                "127.0.0.1", web.server.server_port, timeout=2.0
            )
            headers.pop("Origin")
            connection.request("POST", "/api/control", json.dumps(dict(led_mask=85)), headers)
            res = connection.getresponse()
            body = json.loads(res.read())
            self.assertEqual(res.status, 202)
            self.assertEqual(len(self.mmio.writes), 0)
            self.outputs.tick(False)
            self.assertEqual(self.outputs.applied_revision, body["revision"])
            self.assertEqual(self.mmio.writes[-1][0], threading.get_ident())
            connection.request("GET", "/../../secret")
            res = connection.getresponse()
            res.read()
            self.assertEqual(res.status, 404)
            connection.request("GET", "/api/state")
            res = connection.getresponse()
            state = json.loads(res.read())
            self.assertEqual(state["controls"]["applied"]["led_mask"], 85)
        finally:
            connection.close()
            web.close()

    def test_shared_video_owns_pixels_encodes_once_and_multiple_clients(self):
        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0, fps=6))
        clients = []
        try:
            raw = np.full((36, 64), 0xF800, np.uint16)
            web.publish(raw, dict(running=True, ready=True))
            raw[:] = 0x7E0
            seq, jpeg, _ = web.video.frame(timeout=2.0)
            self.assertEqual(seq, 1)
            color = Image.open(BytesIO(jpeg)).convert("RGB").getpixel((10, 10))
            self.assertGreater(color[0], 200)
            self.assertLess(color[1], 20)
            for _ in range(2):
                conn = http.client.HTTPConnection("127.0.0.1", web.server.server_port, timeout=2.0)
                conn.request("GET", "/stream.mjpg")
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"--homeframe", response.read(80))
                clients.append((conn, response))
            self.assertEqual(web.video.stats()["viewers"], 2)
            self.assertEqual(web.video.stats()["sequence"], 1)
        finally:
            for conn, response in clients:
                response.close()
                conn.close()
            web.close()

    def test_rgb565_stream_owned_pixels_endianness_and_lazy_jpeg(self):
        web = HomeWeb(self.outputs, dict(host="127.0.0.1", port=0, codec="rgb565"))
        conn = None
        try:
            raw = np.array([[0xF800, 0x07E0], [0x001F, 0xFFFF]], dtype=np.uint16)
            expected = raw.astype("<u2").tobytes()
            web.publish(raw)
            raw[:] = 0
            seq, data, _ = web.video.frame(timeout=2.0)
            self.assertEqual(seq, 1)
            self.assertEqual(data, expected)
            self.assertIsNone(web.jpeg_fallback)
            conn = http.client.HTTPConnection("127.0.0.1", web.server.server_port, timeout=2.0)
            conn.request("GET", "/stream.rgb565")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            self.assertEqual(res.fp.readline(), b"--homeframe\r\n")
            headers = {}
            while True:
                line = res.fp.readline()
                if line == b"\r\n":
                    break
                key, _, value = line.partition(b":")
                headers[key.lower()] = value.strip()
            self.assertEqual(headers[b"content-type"], b"application/octet-stream")
            self.assertEqual(res.read(int(headers[b"content-length"])), expected)
            self.assertIsNone(web.jpeg_fallback)
            self.assertIs(web.jpeg_hub(), web.jpeg_hub())
        finally:
            if conn:
                conn.close()
            web.close()


if __name__ == "__main__":
    unittest.main()
