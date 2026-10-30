"""按 PG021 简单模式驱动 RGB565 图像滤波 DMA，仅由主线程调用。"""

import time


class ImageFrameCopy:
    MAGIC = 0x494D4731
    MODES = {"color": 0, "grayscale": 1, "edge": 2}
    ERROR_MASK = 0x770

    def __init__(self, dma, registers, options=None):
        if registers.read(0x140) != self.MAGIC:
            raise RuntimeError("IMG1 hardware required")
        self.mmio, self.registers = dma, registers
        self.closed = False
        self.copies = self.slow_copies = 0
        self.total_ms = self.maximum_ms = 0.0
        self.configuration = dict(mode="color", brightness=0, contrast=100)
        self.control = None
        self.reset()
        self.configure(self.validate(options or {}, self.configuration))

    @classmethod
    def validate(cls, value, previous=None):
        if not isinstance(value, dict) or set(value) - {"mode", "brightness", "contrast"}:
            raise ValueError("图像处理参数无效")
        result = dict(previous or dict(mode="color", brightness=0, contrast=100))
        result.update(value)
        if type(result["mode"]) is not str or result["mode"] not in cls.MODES:
            raise ValueError("图像模式应为原图、灰度或边缘")
        if type(result["brightness"]) is not int or not -64 <= result["brightness"] <= 64:
            raise ValueError("亮度应为-64～64")
        if type(result["contrast"]) is not int or not 50 <= result["contrast"] <= 200:
            raise ValueError("对比度应为50～200%")
        return result

    @property
    def transformed(self):
        return self.control != 0x00800000

    def configure(self, value):
        value = self.validate(value)
        contrast = (value["contrast"] * 128 + 50) // 100
        word = self.MODES[value["mode"]] | ((value["brightness"] & 255) << 8) | (contrast << 16)
        if word != self.control:
            self.registers.write(0x144, word)
            if self.registers.read(0x144) != word:
                raise RuntimeError("Image control readback mismatch")
            self.control = word
        self.configuration = dict(value)

    def reset(self):
        # 主复位同时影响 DMA 双通道和图像滤波器。
        self.mmio.write(0, 4)
        deadline = time.monotonic() + 1.0
        while self.mmio.read(0) & 4:
            if time.monotonic() > deadline:
                raise TimeoutError("Image DMA reset incomplete; retain buffers")
            time.sleep(0.0002)
        if any(self.mmio.read(offset) & self.ERROR_MASK for offset in (4, 0x34)):
            raise RuntimeError("Image DMA reset error")

    def copy(self, source, destination):
        if self.closed:
            raise RuntimeError("Image DMA closed")
        size = int(source.nbytes)
        src, dst = int(source.physical_address), int(destination.physical_address)
        if (
            size != 1843200
            or destination.nbytes != size
            or src % 8
            or dst % 8
            or not 0 <= src < src + size <= 0x40000000
            or not 0 <= dst < dst + size <= 0x40000000
            or max(src, dst) < min(src + size, dst + size)
        ):
            raise ValueError("Image DMA requires disjoint aligned 1280x720 RGB565 frames")
        for offset in (4, 0x34):
            status = self.mmio.read(offset)
            if status & self.ERROR_MASK or not status & 3:
                raise RuntimeError("Image DMA busy or unhealthy")
        started = time.monotonic()
        source.flush()
        self.mmio.write(4, 0x7000)
        self.mmio.write(0x34, 0x7000)
        # 先准备接收端，再启动发送端；BTT 最后写入。
        self.mmio.write(0x30, 1)
        self.mmio.write(0x48, dst)
        self.mmio.write(0x58, size)
        self.mmio.write(0, 1)
        self.mmio.write(0x18, src)
        self.mmio.write(0x28, size)
        # 此处为硬件停滞看门狗，帧发布仍受 50ms 所有权限制。
        deadline = time.monotonic() + 0.250
        while True:
            statuses = [self.mmio.read(offset) for offset in (4, 0x34)]
            if any(v & self.ERROR_MASK for v in statuses):
                self.reset()
                raise RuntimeError("Image DMA AXI/packet error: " + str(statuses))
            if all(v & 2 for v in statuses):
                break
            if time.monotonic() > deadline:
                self.reset()
                raise TimeoutError("Image DMA stalled for 250ms; transaction reset before return")
            time.sleep(0.0002)
        destination.invalidate()
        elapsed = (time.monotonic() - started) * 1000
        self.copies += 1
        self.slow_copies += int(elapsed >= 50.0)
        self.total_ms += elapsed
        self.maximum_ms = max(self.maximum_ms, elapsed)

    def statistics(self):
        return dict(
            engine="pl_image_dma",
            copies=self.copies,
            mean_ms=round(self.total_ms / max(self.copies, 1), 3),
            maximum_ms=round(self.maximum_ms, 3),
            slow_copies=self.slow_copies,
            configuration=dict(self.configuration),
        )

    def snapshot(self):
        return dict(available=True, **self.configuration)

    def close(self):
        if not self.closed:
            self.reset()
            self.closed = True
