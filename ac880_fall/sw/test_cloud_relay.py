"""通过独立 HTTP 测试服务验证分发、重连和帧新鲜度。"""

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))
from video_relay import VideoRelay, make_server


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.count = 0
        self.max_frames = 10000
        self.stop = threading.Event()
        owner = self

        class Source(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/api/state":
                    data = json.dumps(
                        dict(video=dict(viewers=99), control_token="fixture")
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                owner.count += 1
                self.send_response(200)
                self.end_headers()
                try:
                    for index in range(owner.max_frames):
                        if owner.stop.wait(0.03):
                            break
                        jpeg = b"\xff\xd8frame" + str(index).encode() + b"\xff\xd9"
                        self.wfile.write(
                            ("--homeframe\r\nContent-Length: %d\r\n\r\n" % len(jpeg)).encode()
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except OSError:
                    pass

        self.source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
        self.source.daemon_threads = True
        self.source_thread = threading.Thread(target=self.source.serve_forever, daemon=True)
        self.source_thread.start()
        self.relay = VideoRelay("http://127.0.0.1:%d" % self.source.server_port)
        self.server = make_server(self.relay, ("127.0.0.1", 0))
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self):
        self.stop.set()
        self.relay.close()
        self.server.shutdown()
        self.server.server_close()
        self.source.shutdown()
        self.source.server_close()
        self.server_thread.join()
        self.source_thread.join()

    def read_frames(self, count=4):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)
        try:
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            sequences = []
            for _ in range(count):
                while True:
                    line = response.readline()
                    if not line:
                        raise AssertionError("Relay ended before the requested frames")
                    if line.strip() == b"--homeframe":
                        break
                fields = {}
                while True:
                    line = response.readline()
                    if not line:
                        raise AssertionError("Relay returned an incomplete frame header")
                    if line == b"\r\n":
                        break
                    key, _, value = line.partition(b":")
                    fields[key.lower()] = value.strip()
                jpeg = response.read(int(fields[b"content-length"]))
                self.assertTrue(jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"))
                sequences.append(int(fields[b"x-sequence"]))
            return sequences
        finally:
            connection.close()

    def test_two_viewers_share_one_upstream_and_receive_new_frames(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.read_frames) for _ in range(2)]
            sequences = [f.result() for f in futures]
        self.assertEqual(self.count, 1)
        self.assertTrue(all(all(b > a for a, b in zip(values, values[1:])) for values in sequences))

    def test_source_reconnect_preserves_monotonic_relay_sequence(self):
        self.max_frames = 2
        sequences = self.read_frames(4)
        self.assertGreaterEqual(self.count, 2)
        self.assertTrue(all(b > a for a, b in zip(sequences, sequences[1:])))

    def test_stale_pixels_are_not_returned_and_state_token_is_preserved(self):
        with self.relay.condition:
            self.relay.sequence = 1
            self.relay.jpeg = b"\xff\xd8old\xff\xd9"
            self.relay.published_at = time.monotonic() - 20
        self.assertIsNone(self.relay.latest(0, timeout=0.01))
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request("GET", "/api/state")
            response = connection.getresponse()
            value = json.loads(response.read())
            self.assertEqual(value["control_token"], "fixture")
            self.assertEqual(value["video"]["viewers"], 0)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
