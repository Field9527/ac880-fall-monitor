"""读取 SW0，消抖期间暂停识别，稳定后只运行选定的任务。"""

import time


class RecognitionSwitch:
    MAGIC = 0x53574D31

    def __init__(self, mmio, options, config=None):
        self.mmio, self.options = mmio, options
        self.debounce_s = float((config or {}).get("debounce_s", 0.15))
        if not 0.05 <= self.debounce_s <= 1.0:
            raise ValueError("拨码消抖时间应为 0.05～1 秒")
        self.candidate = self.since = None
        self.mode = "paused"
        self.options.select_recognition("paused", dict(available=False, message="等待 SW0"))

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        try:
            if self.mmio.read(0x110) != self.MAGIC:
                raise ValueError("当前 FPGA 配置不支持 SW0，请更新完整运行包")
            raw = self.mmio.read(0x114) & 1
            if raw != self.candidate:
                self.candidate, self.since = raw, now
            stable = now - self.since >= self.debounce_s
            mode = ("gesture" if raw else "fall") if stable else "paused"
            status = dict(
                available=True,
                position=raw,
                stable=stable,
                message="SW0 向上：手势模式" if raw else "SW0 向下：摔倒模式",
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.candidate = self.since = None
            mode, status = "paused", dict(available=False, stable=False, message=str(error))
        changed = mode != self.mode
        self.mode = mode
        self.options.select_recognition(mode, status)
        return changed
