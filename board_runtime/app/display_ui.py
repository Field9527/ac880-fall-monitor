"""缓存 RGB565 状态栏，只在信息变化时重新排版。"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gesture_control import ACTIONS


FONT = Path(__file__).resolve().parents[1] / "assets/ui/monitor.ttf"
BG = (17, 27, 40)
LINE = (44, 61, 80)
WHITE = (239, 244, 249)
MUTED = (147, 166, 188)
BLUE = (100, 174, 245)
AMBER = (246, 169, 91)
RED = (255, 108, 117)
HAND_NAMES = {"open_palm": "Open palm", "fist": "Closed fist", "victory": "V sign"}


def rgb565(color):
    r, g, b = color
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def packed(image):
    pixels = np.asarray(image, dtype=np.uint16)
    return ((pixels[:, :, 0] >> 3) << 11) | ((pixels[:, :, 1] >> 2) << 5) | (pixels[:, :, 2] >> 3)


@lru_cache(maxsize=16)
def font(size):
    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        return ImageFont.load_default()


def text_width(value, face):
    if hasattr(face, "getbbox"):
        bounds = face.getbbox(value)
        return bounds[2] - bounds[0]
    return face.getsize(value)[0]


class MonitorHUD:
    def __init__(self):
        self.header_key = self.footer_key = None
        self.header = self.footer = None

    @staticmethod
    def state(result, observation):
        if observation.get("recognition_mode") == "paused":
            return "Recognition paused", MUTED
        if observation.get("recognition_mode") == "gesture":
            return "Gesture control", BLUE
        state = int(result.get("state", 0))
        if state == 3:
            return "FALL ALERT", RED
        if not observation.get("ready"):
            return "Initializing", MUTED
        return {
            0: ("Monitoring", BLUE),
            1: ("Checking posture", AMBER),
            2: ("Confirming fall", AMBER),
        }.get(state, ("Monitoring", BLUE))

    def draw(self, frame, result, observation, environment=None, gesture=None):
        if frame.ndim != 2 or frame.dtype != np.uint16:
            raise ValueError("HDMI 界面需要 RGB565 图像")
        height, width = frame.shape
        if height < 160 or width < 480:
            return
        scale = min(1.5, max(0.5, width / 1280.0))
        top, bottom = round(64 * scale), round(84 * scale)
        self.targets(frame, result, observation, top, height - bottom, scale)
        title, accent = self.state(result, observation)
        mode = "LIVE CAMERA" if observation.get("ready") else "STARTING"
        if observation.get("test_only"):
            mode = "TEST INPUT"
        if observation.get("test_context") == "phone_screen_replay":
            mode = "PHONE REPLAY"
        elif observation.get("recognition_mode") in ("fall", "gesture", "paused"):
            mode = {"fall": "SW0: FALL", "gesture": "SW0: GESTURE", "paused": "SW0: WAIT"}[
                observation["recognition_mode"]
            ]
        limited = bool(observation.get("coverage_limited") or result.get("tracking_limited"))
        fall_active = observation.get("recognition_mode", "both") in ("both", "fall")
        score = round(float(result.get("pe", 0)), 2) if fall_active else None
        key = (width, top, title, accent, mode, score, limited)
        if key != self.header_key:
            self.header = self.make_header(key, scale)
            self.header_key = key
        info = self.footer_info(environment, observation, gesture)
        key = (width, height, bottom, mode, info)
        if key != self.footer_key:
            self.footer = self.make_footer(key, scale)
            self.footer_key = key
        frame[:top] = self.header
        frame[-bottom:] = self.footer
        if fall_active and int(result.get("state", 0)) == 3:
            cv2.rectangle(frame, (0, top), (width - 1, height - bottom - 1), rgb565(RED), 3)

    @staticmethod
    def make_header(key, scale):
        width, height, title, accent, mode, score, limited = key
        base = MonitorHUD.header_template((width, height, title, accent, mode, limited), scale)
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.text(
            (width - round(332 * scale), round(29 * scale)),
            "%.2f" % score if score is not None else "--",
            font=font(round(19 * scale)),
            fill=WHITE,
        )
        return packed(image)

    @staticmethod
    @lru_cache(maxsize=8)
    def header_template(key, scale):
        width, height, title, accent, mode, limited = key
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        p = lambda n: round(n * scale)
        text = lambda xy, value, size, color: draw.text(
            (p(xy[0]), p(xy[1])), value, font=font(p(size)), fill=color
        )
        draw.rectangle((p(22), p(16), p(52), p(47)), fill=(37, 62, 88))
        draw.line(
            [
                (p(30), p(24)),
                (p(37), p(21)),
                (p(44), p(24)),
                (p(43), p(35)),
                (p(37), p(41)),
                (p(31), p(35)),
                (p(30), p(24)),
            ],
            fill=BLUE,
            width=p(2),
        )
        text((66, 11), "FALL GUARD", 21, WHITE)
        text((67, 38), "AC880  /  EDGE MONITOR", 10, MUTED)
        x = round(width * 0.37)
        draw.line((x - p(24), p(18), x - p(24), height - p(18)), fill=LINE)
        draw.ellipse((x, p(26), x + p(8), p(34)), fill=accent)
        draw.text((x + p(21), p(16)), title, font=font(p(23)), fill=accent)
        label = "COVERAGE LIMITED" if limited else "FALL SCORE"
        draw.text(
            (width - p(332), p(13)), label, font=font(p(10)), fill=AMBER if limited else MUTED
        )
        box = (width - p(173), p(18), width - p(22), p(47))
        draw.rectangle(box, fill=(29, 46, 65), outline=LINE)
        mode_color = AMBER if mode in ("TEST INPUT", "PHONE REPLAY") else BLUE
        label_width = text_width(mode, font(p(11)))
        draw.text(
            ((box[0] + box[2] - label_width) / 2, p(25)), mode, font=font(p(11)), fill=mode_color
        )
        draw.line(
            (0, height - 1, width, height - 1),
            fill=RED if title == "FALL ALERT" else LINE,
            width=p(3) if title == "FALL ALERT" else 1,
        )
        return image

    @staticmethod
    def footer_info(environment, observation, gesture):
        value = environment or {}
        if value.get("available") and value.get("valid"):
            temperature = "%.1f" % value["temperature_c"]
            humidity = "%.1f" % value["humidity_rh"]
            env_note = (
                "Latest valid reading" if not value.get("last_read_ok", True) else "DHT11 sensor"
            )
        else:
            temperature = humidity = "--"
            env_note = (
                "Reading expired"
                if value.get("stale")
                else ("Waiting for sensor" if value.get("available") else "Sensor unavailable")
            )
        hand = observation.get("hand") or {}
        name = hand.get("name")
        hand_title = HAND_NAMES.get(name, "Uncertain gesture" if hand else "No hand detected")
        hint = "Hold steady" if name in HAND_NAMES else "Keep a full hand in view"
        status = observation.get("hand_status")
        if status == "multiple_hands":
            hint = "One hand at a time"
        elif status == "too_small":
            hint = "Move your hand closer"
        control = gesture or {}
        age = control.get("last_action_age_s")
        if (
            age is not None
            and 0 <= age <= 3
            and name in HAND_NAMES
            and control.get("last_action") == ACTIONS[name]
        ):
            hint = "Command submitted"
        if observation.get("test_context") == "phone_screen_replay":
            hand_title, hint = "Gesture paused", "Phone replay mode"
        if observation.get("gesture_enabled") is False:
            hand_title, hint = "Gesture disabled", "Enable from the web page"
        if observation.get("recognition_mode") == "fall":
            hand_title, hint = "Gesture paused", "SW0 up: gesture mode"
        elif observation.get("recognition_mode") == "paused":
            hand_title, hint = "Recognition paused", "Check SW0 / FPGA configuration"
        return temperature, humidity, env_note, hand_title, hint

    @staticmethod
    def make_footer(key, scale):
        width, frame_height, height, mode, info = key
        temperature, humidity, env_note, hand_title, hint = info
        base = MonitorHUD.footer_template(
            (width, frame_height, height, mode, env_note, hand_title, hint), scale
        )
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for x, number in ((round(24 * scale), temperature), (round(width * 0.24), humidity)):
            draw.text((x, round(30 * scale)), number, font=font(round(30 * scale)), fill=WHITE)
        return packed(image)

    @staticmethod
    @lru_cache(maxsize=8)
    def footer_template(key, scale):
        width, frame_height, height, mode, env_note, hand_title, hint = key
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        p = lambda n: round(n * scale)
        draw.line((0, 0, width, 0), fill=LINE)
        columns = [p(24), round(width * 0.24), round(width * 0.48), round(width * 0.80)]
        for x in columns[1:]:
            draw.line((x - p(20), p(18), x - p(20), height - p(18)), fill=LINE)
        for x, label, unit in (
            (columns[0], "TEMPERATURE", "C"),
            (columns[1], "HUMIDITY", "% RH"),
        ):
            draw.text((x, p(13)), label, font=font(p(10)), fill=MUTED)
            draw.text((x + p(111), p(44)), unit, font=font(p(12)), fill=MUTED)
        x = columns[2]
        draw.text((x, p(13)), "GESTURE CONTROL", font=font(p(10)), fill=MUTED)
        draw.text((x, p(31)), hand_title, font=font(p(19)), fill=WHITE)
        draw.text((x, p(58)), hint, font=font(p(10)), fill=BLUE)
        x = columns[3]
        draw.text((x, p(13)), "VIDEO INPUT", font=font(p(10)), fill=MUTED)
        draw.text(
            (x, p(32)),
            "OV5640" if mode not in ("TEST INPUT", "PHONE REPLAY") else "TEST / REPLAY",
            font=font(p(16)),
            fill=WHITE,
        )
        draw.text((x, p(58)), "%d x %d" % (width, frame_height), font=font(p(10)), fill=MUTED)
        if env_note != "DHT11 sensor":
            # 失效或沿用读数时显式提示，不显示伪造的零值。
            draw.text((columns[0] + p(110), p(14)), env_note, font=font(p(9)), fill=AMBER)
        return image

    @staticmethod
    def targets(frame, result, observation, top, bottom, scale):
        state = int(result.get("state", 0))
        color = rgb565(RED if state == 3 else AMBER if state else BLUE)
        width = frame.shape[1]

        def box(bounds, value, thickness=2):
            x, y, w, h = (int(v) for v in bounds)
            left, right = max(0, x), min(width - 1, x + w)
            upper, lower = max(top + 1, y), min(bottom - 1, y + h)
            if right > left and lower > upper:
                cv2.rectangle(frame, (left, upper), (right, lower), value, thickness)
                corner = max(5, round(15 * scale))
                for a, b, dx, dy in (
                    (left, upper, 1, 1),
                    (right, upper, -1, 1),
                    (left, lower, 1, -1),
                    (right, lower, -1, -1),
                ):
                    cv2.line(frame, (a, b), (a + dx * corner, b), value, thickness + 1)
                    cv2.line(frame, (a, b), (a, b + dy * corner), value, thickness + 1)

        if observation.get("bbox") and observation.get("recognition_mode", "both") in (
            "both",
            "fall",
        ):
            box(observation["bbox"], color)
        if observation.get("show_objects"):
            for obj in observation.get("objects", [])[:8]:
                box(obj["bbox"], rgb565(MUTED), 1)
        for obj in observation.get("reviewed_candidates", [])[:3]:
            box(obj["bbox"], rgb565((111, 126, 142)), 1)
        hand = observation.get("hand")
        if hand:
            box(hand["bbox"], rgb565(BLUE), 1)
            points = hand.get("landmarks", [])
            chains = (
                (0, 1, 2, 3, 4),
                (0, 5, 6, 7, 8),
                (5, 9, 10, 11, 12),
                (9, 13, 14, 15, 16),
                (13, 17, 18, 19, 20),
                (0, 17),
            )
            if len(points) == 21:
                for chain in chains:
                    for a, b in zip(chain, chain[1:]):
                        pa, pb = tuple(map(int, points[a])), tuple(map(int, points[b]))
                        if top < pa[1] < bottom and top < pb[1] < bottom:
                            cv2.line(frame, pa, pb, rgb565(BLUE), 1)
            for u, v in points:
                if top < v < bottom:
                    cv2.circle(frame, (int(u), int(v)), max(2, round(2 * scale)), rgb565(WHITE), -1)


HUD = MonitorHUD()
