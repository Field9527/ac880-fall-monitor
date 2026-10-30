"""限频读取 PL 环境寄存器，并缓存 HDMI 温湿度显示条。"""

import time


ERRORS = {
    0: "",
    1: "DHT11未响应，请检查接线",
    2: "DHT11数据线持续低电平",
    3: "DHT11通信时序异常",
    4: "DHT11校验失败",
    5: "DHT11数据格式异常",
}


def decode_payload(value):
    hi, hd, ti, td = ((value >> n) & 255 for n in (24, 16, 8, 0))
    humidity = hi + hd / 10.0
    temperature = ti + (td & 127) / 10.0
    if td & 128:
        temperature = -temperature
    if hd > 9 or (td & 127) > 9 or not 0 <= humidity <= 100 or not -80 <= temperature <= 80:
        raise ValueError("Invalid DHT11 payload")
    return round(temperature, 1), round(humidity, 1)


class EnvironmentMonitor:
    MAGIC = 0x454E5631

    def __init__(self, mmio, config=None):
        config = config or {}
        self.mmio = mmio
        self.interval = max(0.25, float(config.get("read_interval_s", 0.5)))
        self.stale_after = float(config.get("stale_after_s", 15.0))
        self.last_read = -float("inf")
        self.available = bool(config.get("enabled", False) and mmio.read(0x120) == self.MAGIC)
        self.value = dict(
            available=self.available,
            valid=False,
            stale=False,
            temperature_c=None,
            humidity_rh=None,
            age_s=None,
            error_code=0,
            message="等待DHT11首个有效读数" if self.available else "环境采集尚未启用",
        )

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.available or now - self.last_read < self.interval:
            return self.value
        self.last_read = now
        try:
            for _ in range(2):
                seq = self.mmio.read(0x12C)
                status = self.mmio.read(0x124)
                raw = self.mmio.read(0x128)
                age = self.mmio.read(0x130)
                errors = self.mmio.read(0x134)
                attempts = self.mmio.read(0x138)
                if seq == self.mmio.read(0x12C):
                    break
            else:
                raise ValueError("环境读数正在更新")
            good = bool(status & 1)
            fresh = good and age <= self.stale_after * 1000
            temperature, humidity = decode_payload(raw) if good else (None, None)
            code = (status >> 8) & 255
            ok = bool(status & 4)
            message = ERRORS.get(code, "DHT11通信异常")
            if fresh:
                message = "读数正常" if ok else message + "；显示最近有效值"
            elif good:
                message = "温湿度读数已过期；" + (message or "等待新读数")
            elif not message:
                message = "等待DHT11首个有效读数"
            self.value = dict(
                available=True,
                valid=fresh,
                stale=good and not fresh,
                temperature_c=temperature if fresh else None,
                humidity_rh=humidity if fresh else None,
                age_s=round(age / 1000.0, 1) if good else None,
                last_read_ok=ok,
                error_code=code,
                message=message,
                sample_seq=seq,
                errors=errors,
                attempts=attempts,
                busy=bool(status & 2),
            )
        except (OSError, ValueError) as error:
            self.value = dict(
                self.value,
                valid=False,
                stale=True,
                temperature_c=None,
                humidity_rh=None,
                message=str(error),
            )
        return self.value


class EnvironmentHUD:
    """在 44 行 RGB565 缓冲区内缓存温湿度大字。"""

    def __init__(self):
        self.key = None
        self.strip = None

    def draw(self, frame, value):
        if not value or not value.get("available") or frame.shape[0] < 98:
            return
        if value.get("valid"):
            text = "TEMP: %.1f C      HUMIDITY: %.1f %%RH" % (
                value["temperature_c"],
                value["humidity_rh"],
            )
            if not value.get("last_read_ok", True):
                text += "   DHT11: RETRY"
        else:
            reason = (
                "STALE"
                if value.get("stale")
                else {1: "NO RESPONSE", 2: "LINE LOW", 3: "TIMING", 4: "CHECKSUM", 5: "FORMAT"}.get(
                    value.get("error_code"), "WAITING"
                )
            )
            text = "DHT11: %s     TEMP: --     HUMIDITY: --" % reason
        key = (frame.shape[1], text)
        if key != self.key:
            import cv2
            import numpy as np

            self.strip = np.zeros((44, frame.shape[1]), dtype=np.uint16)
            cv2.putText(
                self.strip, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0xFFFF, 2, cv2.LINE_8
            )
            self.key = key
        frame[54:98, :] = self.strip
