"""按 PG034 简单模式执行对齐且有边界检查的 CDMA 帧复制。"""

import time


class FrameCopy:
    ERROR_MASK = 0x770

    def __init__(self, mmio):
        self.mmio = mmio
        self.closed = False
        self.copies = 0
        self.total_ms = 0.0
        self.maximum_ms = 0.0
        self.reset()

    def reset(self):
        # 复位先排空已提交的 AXI 事务，再清除复位位。
        self.mmio.write(0, 4)
        deadline = time.monotonic() + 1.0
        while self.mmio.read(0) & 4:
            if time.monotonic() > deadline:
                raise TimeoutError("CDMA reset incomplete; retain DMA buffers")
            time.sleep(0.0002)
        if self.mmio.read(4) & self.ERROR_MASK:
            raise RuntimeError("CDMA reset error")

    def copy(self, source, destination):
        if self.closed:
            raise RuntimeError("CDMA engine already closed")
        size = int(source.nbytes)
        src, dst = int(source.physical_address), int(destination.physical_address)
        if (
            size != destination.nbytes
            or not 0 < size <= 0x3FFFFFF
            or size % 8
            or src % 8
            or dst % 8
            or not 0 <= src < src + size <= 0x40000000
            or not 0 <= dst < dst + size <= 0x40000000
            or max(src, dst) < min(src + size, dst + size)
        ):
            raise ValueError("CDMA requires disjoint aligned equal-size buffers inside PS DDR")
        status = self.mmio.read(4)
        if not status & 2 or status & self.ERROR_MASK:
            raise RuntimeError("CDMA not idle/healthy before frame copy: 0x%x" % status)
        started = time.monotonic()
        source.flush()
        self.mmio.write(4, 0x7000)
        self.mmio.write(0x18, src)
        self.mmio.write(0x20, dst)
        self.mmio.write(0x28, size)  # 最后写 BTT，启动传输。
        deadline = started + 0.050
        while True:
            status = self.mmio.read(4)
            if status & self.ERROR_MASK:
                self.reset()
                raise RuntimeError("CDMA frame error: 0x%x" % status)
            if status & 2:
                break
            if time.monotonic() > deadline:
                self.reset()
                raise TimeoutError("CDMA frame copy exceeded 50ms")
            time.sleep(0.0002)
        destination.invalidate()
        elapsed = (time.monotonic() - started) * 1000
        self.copies += 1
        self.total_ms += elapsed
        self.maximum_ms = max(self.maximum_ms, elapsed)

    def statistics(self):
        return dict(
            engine="pl_axi_cdma",
            copies=self.copies,
            mean_ms=round(self.total_ms / max(self.copies, 1), 3),
            maximum_ms=round(self.maximum_ms, 3),
        )

    def close(self):
        if not self.closed:
            self.reset()
            self.closed = True
