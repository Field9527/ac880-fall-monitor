"""使用自有 FP32 CMA 缓冲区及 AXI-Lite 逐点卷积驱动。仅单个推理线程访问；超时后仍须等待 AXI 主设备停止访存才能释放内存。"""

import ctypes as ct
import time
import numpy as np


FLOAT_PTR = ct.POINTER(ct.c_float)
COMPUTE_CALLBACK = ct.CFUNCTYPE(
    ct.c_int, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, *([ct.c_uint] * 7)
)
MAX_FLOATS = 983040


class PLPointwise:
    def __init__(self, mmio, allocate, timeout_s=5.0):
        self.mmio, self.allocate = mmio, allocate
        self.timeout_s = float(timeout_s)
        self.input = self.output = None
        self.weights = {}
        self.busy = False
        self.last_error = None
        self.calls = self.macs = 0
        self.seconds = self.transfer_seconds = 0.0
        self.callback = COMPUTE_CALLBACK(self._callback)
        if not self.mmio.read(0) & 4:
            raise RuntimeError("PL pointwise core is not idle before ownership")

    @classmethod
    def from_overlay(cls, overlay, timeout_s=5.0):
        from pynq import MMIO, allocate

        entries = [
            v
            for k, v in overlay.ip_dict.items()
            if "pointwise" in k.lower() and "hls:pointwise:" in v.get("type", "")
        ]
        if len(entries) != 1:
            raise RuntimeError("Overlay must contain exactly one pointwise HLS core")
        entry = entries[0]
        # 先校验 HWH 与固定 ABI，再启动 DMA。
        regs = entry.get("registers", {})
        for name, offset in {
            "ci": 0x50,
            "co": 0x58,
            "pixels": 0x60,
            "input_step": 0x68,
            "output_step": 0x70,
            "relu": 0x78,
            "bias_term": 0x80,
        }.items():
            if name not in regs or int(regs[name]["address_offset"]) != offset:
                raise RuntimeError("Pointwise HWH register ABI mismatch: " + name)
        return cls(MMIO(entry["phys_addr"], entry["addr_range"]), allocate, timeout_s)

    def _address(self, offset, buffer):
        address = int(buffer.physical_address)
        if address < 0 or address + buffer.nbytes > 0x40000000:
            raise ValueError("PL buffer is outside mapped PS DDR")
        self.mmio.write(offset, address & 0xFFFFFFFF)
        self.mmio.write(offset + 4, address >> 32)

    @staticmethod
    def _array(pointer, count):
        if not pointer:
            raise ValueError("Null PL tensor pointer")
        return np.ctypeslib.as_array(pointer, shape=(count,))

    def _callback(
        self,
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        ci,
        co,
        pixels,
        input_step,
        output_step,
        relu,
        bias_term,
    ):
        try:
            self.compute(
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                ci,
                co,
                pixels,
                input_step,
                output_step,
                relu,
                bias_term,
            )
            return 0
        except Exception as exc:
            self.last_error = str(exc)
            return -320

    def compute(
        self,
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        ci,
        co,
        pixels,
        input_step,
        output_step,
        relu,
        bias_term,
    ):
        if self.last_error:
            raise RuntimeError(self.last_error)
        if not (
            1 <= ci <= 960
            and 1 <= co <= 960
            and 1 <= pixels <= 15360
            and pixels <= input_step <= 15360
            and pixels <= output_step <= 15360
            and ci * input_step <= MAX_FLOATS
            and co * output_step <= MAX_FLOATS
            and relu in (0, 1)
            and bias_term in (0, 1)
        ):
            raise ValueError("Unsupported PL pointwise dimensions")
        if self.busy or not self.mmio.read(0) & 4:
            raise RuntimeError("PL pointwise core unexpectedly busy")
        start = time.monotonic()
        if self.input is None:
            self.input = self.allocate(shape=(MAX_FLOATS,), dtype=np.float32)
            self.output = self.allocate(shape=(MAX_FLOATS,), dtype=np.float32)
            self.output.flush()
        key = (ct.cast(weight_ptr, ct.c_void_p).value, ci, co, bias_term)
        if key not in self.weights:
            weights = self.allocate(shape=(ci * co + co,), dtype=np.float32)
            # 立即保留缓冲区所有权，覆盖后续复制失败路径。
            self.weights[key] = weights
            weights[: ci * co] = self._array(weight_ptr, ci * co)
            weights[ci * co :] = self._array(bias_ptr, co) if bias_term else 0.0
            weights.flush()
        self.input[: ci * input_step] = self._array(input_ptr, ci * input_step)
        self.input[: ci * input_step].flush()
        self._address(0x20, self.input)
        self._address(0x30, self.weights[key])
        self._address(0x40, self.output)
        for offset, value in zip(
            (0x50, 0x58, 0x60, 0x68, 0x70, 0x78, 0x80),
            (ci, co, pixels, input_step, output_step, relu, bias_term),
        ):
            self.mmio.write(offset, int(value))
        launch = time.monotonic()
        self.busy = True
        self.mmio.write(0, 1)
        expired = False
        seen_done = False
        while True:
            control = self.mmio.read(0)
            # ap_done 读后清零，等待 ap_idle 时保留完成状态。
            seen_done = seen_done or bool(control & 2)
            if seen_done and control & 4:
                self.busy = False
                break
            if time.monotonic() - launch > self.timeout_s:
                expired = True
            # 超时后继续排空，不释放活动 AXI 缓冲区。
            time.sleep(0.0002)
        kernel_seconds = time.monotonic() - launch
        self.seconds += kernel_seconds
        if expired:
            raise TimeoutError("PL kernel exceeded deadline; DMA drained before return")
        self.output[: co * output_step].invalidate()
        destination = self._array(output_ptr, co * output_step).reshape(co, output_step)
        destination[:, :pixels] = self.output[: co * output_step].reshape(co, output_step)[
            :, :pixels
        ]
        self.calls += 1
        self.macs += ci * co * pixels
        self.transfer_seconds += time.monotonic() - start - kernel_seconds

    def statistics(self):
        return dict(
            calls=self.calls,
            macs=self.macs,
            kernel_seconds=round(self.seconds, 6),
            transfer_seconds=round(self.transfer_seconds, 6),
            cached_weight_layers=len(self.weights),
            error=self.last_error,
        )

    def close(self):
        if self.busy:
            # 覆盖启动寄存器已写入后的异常路径。
            # 确认主设备停止访存后才释放缓冲区。
            while not self.mmio.read(0) & 4:
                time.sleep(0.001)
            self.busy = False
        for buffer in list(self.weights.values()) + [self.input, self.output]:
            if buffer is not None:
                buffer.freebuffer()
        self.weights.clear()
        self.input = self.output = None
