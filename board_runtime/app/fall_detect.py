#!/usr/bin/env python3
"""
AC880 摔倒检测系统 - 硬件接口与时序判决
=================================
根据《AC880 实现详细方案》第 9 章设计

当前入口委托 realtime.py，默认运行无背景 fall_lite 模型。
本模块保留 VDMA、PL 运动接口、生成的 HLS ABI 和四级时序判决。
当前 CNN 尚无已验证的训练权重，运行程序不使用它作报警判据。

使用方法：
  sudo -E python3 fall_detect.py --config config.json

依赖：
  - pynq 2.7.0
  - numpy
  - PIL (Pillow)
"""

import os
import sys
import time
import json
import math
import argparse
import logging
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image

# 在 PC 上也能运行模拟、回放和单元测试；硬件依赖在 main 中加载。
Overlay = allocate = MMIO = None


def acquire_hardware_lock(path="/run/lock/ac880_fall.lock"):
    """普通用户与 sudo 共用文件锁；已有锁只读打开，避免截断及 protected_regular 限制。"""
    import fcntl

    try:
        lock = open(path, "r")
    except FileNotFoundError:
        try:
            lock = open(path, "x")
        except FileExistsError:
            lock = open(path, "r")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        lock.close()
        raise
    return lock


# ============================================================
# 常量定义
# ============================================================

# motion_detect 寄存器偏移 (与 motion_regfile.v 一致)
MOTION_REGS = {
    "CTRL": 0x00,  # 写入控制位：EN、SW_RST、CLEAR_BG、ROI_EN、FORCE_TRIG
    "STATUS": 0x04,  # 只读状态位：BUSY、BG_VALID、ROI_ACTIVE、ROI_DONE、MOTION、BBOX_UPD
    "THRESH": 0x08,  # R/W: 帧差阈值 (默认24)
    "MIN_AREA": 0x0C,  # R/W: 最小运动面积 (默认200, cell数)
    "BBOX_X": 0x10,  # R: BBOX 左上角 X
    "BBOX_Y": 0x14,  # R: BBOX 左上角 Y
    "BBOX_W": 0x18,  # R: BBOX 宽度
    "BBOX_H": 0x1C,  # R: BBOX 高度
    "AREA": 0x20,  # R: 当前帧运动面积
    "ENERGY": 0x24,  # R: 当前帧运动能量
    "FRAME_CNT": 0x28,  # R: 帧计数器
    "ROI_TRIG": 0x2C,  # R/W: ROI 触发计数
    "DS_FACTOR": 0x30,  # R/W: 降采样因子 (默认8)
    "MORPH_CFG": 0x34,  # R/W: 形态学配置 (默认3=开运算)
    "TRIG_N": 0x38,  # R/W: 连续触发帧数 (默认3)
    "MODE": 0x3C,  # R/W: 0=帧差, 1=背景减除
    "BG_ALPHA": 0x40,  # R/W: 背景更新 alpha 位移 (默认4)
    "ROI_PAD": 0x44,  # R/W: ROI 边距 (默认8)
    "PROC_LAT": 0x48,  # R: 处理延迟 (周期数)
    "PIX_W": 0x4C,  # R/W: 图像宽度 (默认1280)
    "PIX_H": 0x50,  # R/W: 图像高度 (默认720)
    "ROI_SENT": 0x54,  # R: ROI 发送帧数
    "VERSION": 0x58,  # R: 版本号
    "SNAP_SEQ": 0x5C,  # R: 快照序号 (跨时钟域一致性)
    "ROW_TH": 0x60,  # R/W: 行投影阈值
    "COL_TH": 0x64,  # R/W: 列投影阈值
    "FRAME_MS": 0x68,  # R: 帧周期 (ms)
}

# cnn_accel 寄存器偏移 (与 HLS cnn_accel.cpp 的 s_axilite bundle=CTRL 一致)
# 以 solution1/impl/ip/drivers/.../xcnn_accel_hw.h 为准。
# 0x04/08/0c 是中断寄存器，不是 DONE/IDLE/READY。
CNN_REGS = {
    "AP_CTRL": 0x00,
    "SCORE": 0x10,
    "SCORE_VALID": 0x14,
    "CTRL": 0x18,
    "START": 0x20,
    "DONE": 0x28,
    "IDLE": 0x30,
    "READY": 0x38,
    "THRESH_Q7": 0x40,
}

# VDMA 寄存器偏移 (AXI VDMA v6.3 直接模式, PG020 官方映射, 2021-11-18 实测核对)
# 注意: VSIZE/HSIZE/STRIDE/START_ADDRESS 全部在 0x50 之后, 0x28/0x2C 是
# PARK_PTR(RO)/VERSION(RO), 0x48/0x4C 是保留位(写入无效)
VDMA_MM2S_CR = 0x00
VDMA_MM2S_SR = 0x04
VDMA_MM2S_VSIZE = 0x50
VDMA_MM2S_HSIZE = 0x54
VDMA_MM2S_STRIDE = 0x58
VDMA_MM2S_START = 0x5C  # +4*i, i=0..帧数-1

VDMA_S2MM_CR = 0x30
VDMA_S2MM_SR = 0x34
VDMA_S2MM_VSIZE = 0xA0
VDMA_S2MM_HSIZE = 0xA4
VDMA_S2MM_STRIDE = 0xA8
VDMA_S2MM_START = 0xAC  # 帧地址步长为 4 字节。

VDMA_CR_RUN_STOP = 1 << 0
VDMA_CR_CIRCULAR = 1 << 1
VDMA_CR_RESET = 1 << 2
VDMA_SR_HALTED = 1 << 0
# PG020 v6.3: bit12 是正常帧中断，EOLLate 是 bit15。
VDMA_SR_ERROR_MASK = 0x000089F0
VDMA_SR_CLEAR_MASK = 0x0000F9F0
VDMA_PARK_PTR = 0x28


# ============================================================
# 状态机定义
# ============================================================


class FallState:
    IDLE = 0  # 空闲，无运动或概率低
    CANDIDATE = 1  # 候选，连续触发且概率超过告警阈值
    ALARM_ARMED = 2  # 预警，候选持续足够长时间
    ALARM = 3  # 报警，确认摔倒事件


STATE_NAMES = ["IDLE", "CANDIDATE", "ALARM_ARMED", "ALARM"]


# ============================================================
# 工具函数
# ============================================================


def pl_boot_fix(logger=None):
    """看门狗复位恢复后 FCLK 节流器处于停止态 (THR_STA.RUNNING=0)，PL 无时钟；
    POR 上电则默认运行。按 UG585 25.7.2 配方重启全部 FCLK 连续输出 + 解除 PL 复位。
    本函数幂等，任何时候调用都安全。"""
    import mmap as _mm
    import struct as _st

    try:
        f = open("/dev/mem", "r+b")
    except PermissionError:
        if logger:
            logger.warning("pl_boot_fix: 无 /dev/mem 权限（需 root），跳过")
        return False

    def rd(a):
        o = a & 0xFFF
        m = _mm.mmap(f.fileno(), 0x1000, offset=a & ~0xFFF)
        v = _st.unpack("<I", m[o : o + 4])[0]
        m.close()
        return v

    def wr(a, v):
        o = a & 0xFFF
        m = _mm.mmap(f.fileno(), 0x1000, offset=a & ~0xFFF)
        m[o : o + 4] = _st.pack("<I", v)
        m.close()

    wr(0xF8000008, 0xDF0D)  # 解锁 SLCR。
    wr(0xF8000240, 0x00000000)  # FPGA_RST_CTRL: 解除全部 PL 复位
    for base in (0xF8000170, 0xF8000180, 0xF8000190, 0xF80001A0):
        wr(base + 8, 0x0)  # THR_CNT = 0 连续模式
        wr(base + 4, 0x4)  # 预置时钟控制，保持 CPU_START 为 0。
        wr(base + 4, 0x5)  # CPU_START 0->1 正边沿 -> 启动
    wr(0xF8000004, 0x767B)  # 锁定 SLCR。
    running = [(rd(b + 12) >> 16) & 1 for b in (0xF8000170, 0xF8000180, 0xF8000190, 0xF80001A0)]
    f.close()
    if logger:
        logger.info(f"pl_boot_fix: FCLK0-3 RUNNING={running}")
    return all(running)


def pl_lvl_shifter_fix(logger=None):
    """fpga_manager (zynq-fpga 驱动) 加载比特流后不使能电平移位器
    (LVL_SHFTR_EN=0)，AXI 事务无法进入 PL —— PCFG_DONE=1 也会挂死总线。
    手动写 0xF 使能全部电平移位器。幂等，随时可调。"""
    import mmap as _mm
    import struct as _st

    try:
        f = open("/dev/mem", "r+b")
    except PermissionError:
        if logger:
            logger.warning("pl_lvl_shifter_fix: 无 /dev/mem 权限（需 root），跳过")
        return False

    def rd(a):
        o = a & 0xFFF
        m = _mm.mmap(f.fileno(), 0x1000, offset=a & ~0xFFF)
        v = _st.unpack("<I", m[o : o + 4])[0]
        m.close()
        return v

    def wr(a, v):
        o = a & 0xFFF
        m = _mm.mmap(f.fileno(), 0x1000, offset=a & ~0xFFF)
        m[o : o + 4] = _st.pack("<I", v)
        m.close()

    wr(0xF8000008, 0xDF0D)  # 解锁 SLCR。
    wr(0xF8000900, 0x0000000F)  # 使能全部 PS 与 PL 电平移位器。
    wr(0xF8000004, 0x767B)  # 锁定 SLCR。
    val = rd(0xF8000900)
    f.close()
    if logger:
        logger.info(f"pl_lvl_shifter_fix: LVL_SHFTR_EN=0x{val:08X}")
    return val == 0xF


def overlay_clock_evidence(overlay, mmio=None, clock_summary=None):
    """用 Linux 实际 PS 参考时钟核对 HWH。板载 PYNQ 的 50MHz 显示与实际 33.333333MHz 不符，不能据此调整物理分频。仅硬件主线程读取寄存器。"""
    from pathlib import Path
    import re
    import xml.etree.ElementTree as ET

    hwh = ET.parse(Path(overlay.bitfile_name).with_suffix(".hwh"))
    refs = {
        float(v.attrib["VALUE"])
        for v in hwh.findall(".//PARAMETER[@NAME='PCW_CRYSTAL_PERIPHERAL_FREQMHZ']")
    }
    expected = {
        float(v.attrib["CLKFREQUENCY"]) / 1e6
        for v in hwh.findall(".//PORT[@NAME='fclk_clk0']")
        if "CLKFREQUENCY" in v.attrib
    }
    if len(refs) != 1 or len(expected) != 1:
        raise RuntimeError("Missing unique PS reference/FCLK0 rate in HWH")
    if clock_summary is None:
        clock_summary = Path("/sys/kernel/debug/clk/clk_summary").read_text()
    rates = {
        name: int(rate) / 1e6
        for name, rate in re.findall(
            r"^\s*(ps_clk|fclk0)\s+\d+\s+\d+\s+\d+\s+(\d+)\s", clock_summary, re.M
        )
    }
    if set(rates) != {"ps_clk", "fclk0"} or abs(rates["ps_clk"] - next(iter(refs))) > 0.001:
        raise RuntimeError("Linux PS reference disagrees with HWH or is unavailable")
    if mmio is None:
        from pynq import MMIO as HardwareMMIO

        mmio = HardwareMMIO(0xF8000000, 0x1000)
    control = mmio.read(0x170)
    source = (control >> 4) & 3
    pll_offset = (0x108, 0x108, 0x100, 0x104)[source]
    pll = mmio.read(pll_offset)
    div0, div1 = (control >> 8) & 63, (control >> 20) & 63
    feedback = (pll >> 12) & 127
    if not div0 or not div1 or not feedback or pll & 0x11:
        raise RuntimeError("Unsupported PLL reset/bypass or zero divider")
    actual = rates["ps_clk"] * feedback / div0 / div1
    target = next(iter(expected))
    if abs(actual - rates["fclk0"]) > 0.001 or abs(actual - target) > target * 0.001:
        raise RuntimeError("Actual FCLK0 rate disagrees with Linux or routed HWH")
    return dict(
        reference_mhz=rates["ps_clk"],
        fclk0_mhz=round(actual, 6),
        expected_fclk0_mhz=target,
        linux_fclk0_mhz=rates["fclk0"],
        fclk0_control=hex(control),
        source_pll_control=hex(pll),
        source="Linux ps_clk reference and read-only SLCR PLL/dividers",
        passed=True,
    )


def setup_logger(log_file):
    """初始化日志"""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logger = logging.getLogger("fall_detect")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    fh = logging.FileHandler(log_file, encoding="utf8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def resolve_overlay_path(path, config_path, logger):
    """
    校验比特流路径，必要时在常见目录中回退搜索。
    同时检查同名 .hwh 是否成对存在（PYNQ 加载 Overlay 的硬性要求）。
    返回可用的 .bit 路径；失败返回 None。
    """
    base = os.path.basename(path)
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = [
        path,
        os.path.join(cfg_dir, base),
        os.path.join("/home/xilinx/fall", base),
        os.path.join("/home/xilinx", base),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            hwh = os.path.splitext(cand)[0] + ".hwh"
            if not os.path.exists(hwh):
                logger.error(f"找到比特流 {cand}，但缺少配套文件 {hwh}")
                logger.error("PYNQ 要求 .bit 与同目录同名 .hwh 成对存在，请一并传输")
                return None
            if cand != path:
                logger.warning(f"配置的比特流路径不存在: {path}")
                logger.warning(f"已自动回退使用: {cand}")
            return cand
    logger.error(f"比特流文件不存在: {path}")
    logger.error(f"已尝试以下位置: {candidates}")
    logger.error("请确认 ac880_fall.bit 和 ac880_fall.hwh 已传输到开发板")
    return None


def q7_to_float(q7_val):
    """Q7 格式转浮点概率"""
    return q7_val / 127.0


def float_to_q7(p):
    """浮点概率转 Q7 格式"""
    return int(max(0.0, min(1.0, p)) * 127.0 + 0.5)


# ============================================================
# VDMA 管理器
# ============================================================


class VDMAManager:
    """
    AXI VDMA 管理器
    负责视频采集 (S2MM) 和显示 (MM2S)
    """

    def __init__(self, mmio, width, height, bpp=16, num_frames=3):
        if bpp != 16 or width <= 0 or height <= 0 or num_frames != 3:
            raise ValueError("当前硬件要求 RGB565、正数尺寸和 3 个帧缓冲")
        self.mmio = mmio
        self.width = width
        self.height = height
        self.bpp_bytes = bpp // 8
        self.num_frames = num_frames
        self.frames = None
        self.display_frames = None
        self.copy_engine = None

    def _read(self, offset):
        return self.mmio.read(offset)

    def _write(self, offset, value):
        self.mmio.write(offset, value)

    def allocate_frames(self):
        """分配帧缓冲"""
        self.frames = []
        for i in range(self.num_frames):
            buf = allocate(
                shape=(self.height, self.width),
                dtype=np.uint16 if self.bpp_bytes == 2 else np.uint32,
                cacheable=False,
            )
            self.frames.append(buf)
            buf[:] = 0
            buf.flush()
        return self.frames

    def _wait(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.001)
        return False

    def allocate_display(self, cacheable=False):
        """独立显示缓冲，CPU 标注不会写入相机正在采集的帧。"""
        self.display_frames = []
        for _ in range(3):
            buf = allocate(shape=(self.height, self.width), dtype=np.uint16, cacheable=cacheable)
            buf[:] = 0
            buf.flush()
            self.display_frames.append(buf)

    def show_frame(self, image):
        if self.display_frames is None:
            return
        park = self._read(VDMA_PARK_PTR)
        reading = (park >> 16) & 0x1F
        pending = park & 0x1F
        target = next(i for i in range(3) if i not in (reading, pending))
        self.display_frames[target][:] = image
        self.display_frames[target].flush()
        self._write(VDMA_PARK_PTR, (park & 0x1F00) | target)
        self._write(VDMA_MM2S_CR, self._read(VDMA_MM2S_CR) & ~VDMA_CR_CIRCULAR)

    def prepare_display_frame(self):
        """将已完成采集帧复制到空闲显示缓冲区。三帧缓冲且相机不超过 30FPS 时，复制必须短于 50ms，写指针不能前进两帧。冲突或超时均丢弃，完成标注后才发布。"""
        park = self._read(VDMA_PARK_PTR)
        if self._read(VDMA_MM2S_CR) & VDMA_CR_CIRCULAR:
            raise RuntimeError("Direct display requires parked MM2S")
        target = next(i for i in range(3) if i not in ((park >> 16) & 31, park & 31))
        image = self.display_frames[target]
        for _ in range(2):
            started = time.monotonic()
            writing = self.frame_indices()[1]
            if writing >= self.num_frames:
                raise RuntimeError("VDMA returned invalid frame store")
            source = self.frames[(writing - 1) % self.num_frames]
            source.invalidate()
            if self.copy_engine is not None:
                self.copy_engine.copy(source, image)
            else:
                np.copyto(image, source)
            following = self.frame_indices()[1]
            if (
                time.monotonic() - started < 0.050
                and following < self.num_frames
                and (following - writing) % self.num_frames <= 1
            ):
                return target, image
        return None, None

    def commit_display_frame(self, target):
        """发布已完成绘制的缓冲区，VDMA 在帧边界切换。"""
        park = self._read(VDMA_PARK_PTR)
        if target in ((park >> 16) & 31, park & 31):
            raise RuntimeError("Display buffer is already owned by VDMA")
        self.display_frames[target].flush()
        self._write(VDMA_PARK_PTR, (park & 0x1F00) | target)

    def raw_inference_frame(self):
        """取得稳定的原始彩色帧，模型不读取显示滤镜结果。"""
        for _ in range(2):
            started = time.monotonic()
            writing = self.frame_indices()[1]
            if writing >= self.num_frames:
                raise RuntimeError("Invalid raw frame store")
            source = self.frames[(writing - 1) % self.num_frames]
            source.invalidate()
            owned = np.array(source, copy=True)
            following = self.frame_indices()[1]
            if (
                time.monotonic() - started < 0.050
                and following < self.num_frames
                and (following - writing) % self.num_frames <= 1
            ):
                return owned
        raise RuntimeError("Cannot obtain a complete raw inference frame")

    def reset_channel(self, cr_offset):
        """复位通道"""
        self._write(cr_offset, VDMA_CR_RESET)
        if not self._wait(lambda: not self._read(cr_offset) & VDMA_CR_RESET):
            raise TimeoutError("VDMA 通道复位超时: 0x%x" % cr_offset)

    def config_channel(self, cr_offset, start_off, hsize_off, stride_off):
        """配置通道 (PG020 直接模式): 停机 -> 写各帧存起始地址 -> HSIZE -> STRIDE
        注意 VSIZE 必须最后写 (启动时写), 此处不碰"""
        stride = self.width * self.bpp_bytes

        # 确保停机 (RS=0)
        self._write(cr_offset, 0)
        if not self._wait(lambda: self._read(cr_offset + 4) & VDMA_SR_HALTED):
            raise TimeoutError("VDMA 配置前停机超时")

        buffers = (
            self.display_frames
            if cr_offset == VDMA_MM2S_CR and self.display_frames is not None
            else self.frames
        )
        for i, buf in enumerate(buffers):
            self._write(start_off + 4 * i, buf.physical_address)
        self._write(hsize_off, self.width * self.bpp_bytes)
        self._write(stride_off, stride)

    def start_channel(self, cr_offset, vsize_off, sr_offset):
        """启动通道 (PG020): 置 RS=1 后最后写 VSIZE,
        该写操作把参数组载入内部寄存器块并启动传输"""
        cr = self._read(cr_offset)
        cr |= VDMA_CR_RUN_STOP | VDMA_CR_CIRCULAR
        self._write(cr_offset, cr)
        self._write(vsize_off, self.height)

        # PG020 v6.3: SR bit0=Halted，bit1 保留，不能用作运行判据。
        for _ in range(500):
            sr = self._read(sr_offset)
            if not (sr & VDMA_SR_HALTED):
                return True
            time.sleep(0.001)
        return False

    def is_running(self, sr_offset):
        return not bool(self._read(sr_offset) & VDMA_SR_HALTED)

    def has_error(self, sr_offset):
        sr = self._read(sr_offset)
        return bool(sr & VDMA_SR_ERROR_MASK)

    def frame_indices(self):
        """PG020: 高半字是当前帧，低半字是停泊目标。"""
        park = self._read(VDMA_PARK_PTR)
        return (park >> 16) & 0x1F, (park >> 24) & 0x1F

    def read_frame(self):
        """复制刚完成的采集帧；若复制期间写指针变化则重试。"""
        for _ in range(4):
            _, writing = self.frame_indices()
            if writing >= self.num_frames:
                raise RuntimeError("VDMA 返回无效帧号")
            buf = self.frames[(writing - 1) % self.num_frames]
            buf.invalidate()
            result = np.array(buf, copy=True)
            if self.frame_indices()[1] == writing:
                return result
        return None

    def s2mm_has_error(self):
        """S2MM 通道是否有任何错误位 (PG020 bit4..bit15)"""
        return bool(self._read(VDMA_S2MM_SR) & VDMA_SR_ERROR_MASK)

    def recover_s2mm(self):
        """S2MM 错误恢复: 复位 -> 清错 -> 重配帧存 -> 重启。
        用于启动时与视频源同步导致的 SOF/EOL 锁存错误。"""
        # 复位通道
        self.reset_channel(VDMA_S2MM_CR)
        # 清错误位 (写1清除)
        self._write(VDMA_S2MM_SR, VDMA_SR_CLEAR_MASK)
        # 重配帧存与行参数
        for i, buf in enumerate(self.frames):
            self._write(VDMA_S2MM_START + 4 * i, buf.physical_address)
        self._write(VDMA_S2MM_HSIZE, self.width * self.bpp_bytes)
        self._write(VDMA_S2MM_STRIDE, self.width * self.bpp_bytes)
        # 重启 (VSIZE 最后写)
        return self.start_channel(VDMA_S2MM_CR, VDMA_S2MM_VSIZE, VDMA_S2MM_SR)

    def start(self):
        """启动 VDMA：先显示后采集 (VSIZE 最后写以启动)"""
        self.reset_channel(VDMA_MM2S_CR)
        self.reset_channel(VDMA_S2MM_CR)

        self.config_channel(VDMA_MM2S_CR, VDMA_MM2S_START, VDMA_MM2S_HSIZE, VDMA_MM2S_STRIDE)
        self.config_channel(VDMA_S2MM_CR, VDMA_S2MM_START, VDMA_S2MM_HSIZE, VDMA_S2MM_STRIDE)

        mm2s_ok = self.start_channel(VDMA_MM2S_CR, VDMA_MM2S_VSIZE, VDMA_MM2S_SR)
        time.sleep(0.05)
        s2mm_ok = self.start_channel(VDMA_S2MM_CR, VDMA_S2MM_VSIZE, VDMA_S2MM_SR)

        if mm2s_ok and self.display_frames is not None:
            # CPU 写入前固定显示缓冲区的所有权。
            park = self._read(VDMA_PARK_PTR)
            self._write(VDMA_PARK_PTR, park & 0x1F00)
            self._write(VDMA_MM2S_CR, self._read(VDMA_MM2S_CR) & ~VDMA_CR_CIRCULAR)
            if not self._wait(lambda: self.frame_indices()[0] == 0):
                raise TimeoutError("VDMA display parking timed out")

        return mm2s_ok and s2mm_ok

    def stop(self):
        """停机并等待总线完成；确认停止前不能释放物理内存。"""
        if self.copy_engine is not None:
            self.copy_engine.close()
        for cr_off in (VDMA_S2MM_CR, VDMA_MM2S_CR):
            self._write(cr_off, self._read(cr_off) & ~VDMA_CR_RUN_STOP)
        for cr_off in (VDMA_S2MM_CR, VDMA_MM2S_CR):
            if not self._wait(lambda: self._read(cr_off + 4) & VDMA_SR_HALTED):
                self.reset_channel(cr_off)
                if not self._wait(lambda: self._read(cr_off + 4) & VDMA_SR_HALTED):
                    raise TimeoutError("VDMA 未停止，保留帧缓存")

    def install_exit_guard(self):
        """退出前先停 VDMA，再释放缓冲区。正常退出、SIGTERM 和 SIGINT 均需保护；SIGKILL 无法拦截，须重启开发板。"""
        import atexit
        import os as _os
        import signal as _sig

        vdma = self

        def _halt():
            try:
                vdma.stop()
                # 再复位两个通道, 确保没有残留的 AXI 写事务
                vdma._write(VDMA_S2MM_CR, VDMA_CR_RESET)
                vdma._write(VDMA_MM2S_CR, VDMA_CR_RESET)
                time.sleep(0.02)
                vdma._write(VDMA_S2MM_CR, 0)
                vdma._write(VDMA_MM2S_CR, 0)
            except Exception:
                pass

        def _sig_handler(signum, _frame):
            _halt()
            _os._exit(1)

        atexit.register(_halt)
        for s in (_sig.SIGTERM, _sig.SIGINT, _sig.SIGHUP):
            try:
                _sig.signal(s, _sig_handler)
            except Exception:
                pass
        self._exit_guard = _halt
        return True

    def free(self):
        """释放帧缓冲"""
        if self.frames:
            self.stop()
            for buf in self.frames:
                buf.freebuffer()
            self.frames = None
        if self.display_frames:
            self.stop()
            for buf in self.display_frames:
                buf.freebuffer()
            self.display_frames = None


# ============================================================
# 运动检测接口
# ============================================================


class MotionDetectIF:
    """
    一级运动检测引擎接口
    通过 AXI-Lite 寄存器访问

    注意：本设计中运动检测以"外部 AXI-Lite 端口"形式引出
    （hwh 中为 m_axi_lite_md，基地址 0x44A10000），
    不会出现在 Overlay.ip_dict 中，必须用裸 MMIO 访问。
    """

    def __init__(self, mmio):
        self.mmio = mmio

    def read(self, name):
        return self.mmio.read(MOTION_REGS[name])

    def write(self, name, value):
        self.mmio.write(MOTION_REGS[name], value)

    def wait_frames(self, timeout=10.0):
        """等待相机实际出帧，避免 SCCB 初始化期间复位无时钟的 S2MM。"""
        if self.read("VERSION") != 0x10000:
            raise RuntimeError("运动寄存器版本不匹配")
        deadline = time.monotonic() + timeout
        previous = self.read("FRAME_CNT")
        changes = 0
        while time.monotonic() < deadline:
            current = self.read("FRAME_CNT")
            if current != previous:
                changes += 1
                previous = current
                if changes >= 2:
                    return True
            time.sleep(0.02)
        raise TimeoutError("OV5640 帧心跳未就绪，请检查相机连接与初始化")

    def configure(self, cfg, width=1280, height=720):
        """根据配置字典设置寄存器"""
        self.write("THRESH", cfg.get("thresh", 24))
        self.write("MIN_AREA", cfg.get("min_area", 200))
        self.write("DS_FACTOR", cfg.get("ds_factor", 8))
        self.write("MORPH_CFG", cfg.get("morph_cfg", 3))
        self.write("TRIG_N", cfg.get("trig_n", 3))
        self.write("MODE", cfg.get("mode", 0))
        self.write("BG_ALPHA", cfg.get("bg_alpha", 4))
        self.write("ROI_PAD", cfg.get("roi_pad", 8))
        self.write("ROW_TH", cfg.get("row_th", 3))
        self.write("COL_TH", cfg.get("col_th", 3))
        self.write("PIX_W", width)
        self.write("PIX_H", height)

    def enable(self):
        """使能检测"""
        ctrl = self.read("CTRL")
        ctrl |= 0x01  # 置位 EN。
        self.write("CTRL", ctrl)

    def disable(self):
        """禁用检测"""
        ctrl = self.read("CTRL")
        ctrl &= ~0x01
        self.write("CTRL", ctrl)

    def sw_reset(self):
        """软件复位"""
        ctrl = self.read("CTRL")
        ctrl |= 0x02  # 触发软件复位。
        self.write("CTRL", ctrl)
        time.sleep(0.01)
        ctrl &= ~0x02
        self.write("CTRL", ctrl)

    def clear_bg(self):
        """清除背景"""
        ctrl = self.read("CTRL")
        ctrl |= 0x04
        self.write("CTRL", ctrl)
        ctrl &= ~0x04
        self.write("CTRL", ctrl)

    def get_snapshot(self):
        """
        读取一帧运动状态快照
        使用 snap_seq 协议保证一致性：
          读 SEQ → 读数据 → 再读 SEQ
          两次 SEQ 相同则数据一致
        """
        for _ in range(3):
            seq1 = self.read("SNAP_SEQ")
            area = self.read("AREA")
            energy = self.read("ENERGY")
            bbox_x = self.read("BBOX_X")
            bbox_y = self.read("BBOX_Y")
            bbox_w = self.read("BBOX_W")
            bbox_h = self.read("BBOX_H")
            frame_cnt = self.read("FRAME_CNT")
            seq2 = self.read("SNAP_SEQ")
            if seq1 == seq2:
                return {
                    "seq": seq1,
                    "area": area,
                    "energy": energy,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "frame_cnt": frame_cnt,
                }
        return None  # 连续撕裂，放弃


# ============================================================
# CNN 加速器接口
# ============================================================


class CNNAccelIF:
    """
    二级 CNN 加速器接口
    通过 AXI-Lite 控制，通过 AXI DMA 输入 ROI
    """

    def __init__(self, mmio, dma_send=None):
        self.mmio = mmio
        self.dma_send = dma_send  # AXI DMA 发送通道 MM2S。

    def read(self, name):
        return self.mmio.read(CNN_REGS[name])

    def write(self, name, value):
        self.mmio.write(CNN_REGS[name], value)

    def set_threshold(self, q7_val):
        self.write("THRESH_Q7", q7_val)

    def load_weights(self, weights_file, logger=None):
        raise NotImplementedError("当前 HLS IP 没有运行时权重加载接口")

    def infer(self, roi_buf=None):
        """
        执行一次推理
        roi_buf: PYNQ allocate buffer (96x96 uint8) — 当前硬件管线自动送 ROI，参数保留

        返回: score (Q7 uint8)
        """
        if not (self.read("AP_CTRL") & 0x4):
            raise RuntimeError("CNN 忙碌，不能再次启动")

        # 写 START 启动推理
        self.write("START", 0x01)
        self.write("AP_CTRL", 0x01)

        # 等 DONE
        t0 = time.monotonic()
        while not (self.read("AP_CTRL") & 0x2):
            if time.monotonic() - t0 > 1.0:
                raise TimeoutError("CNN 未完成推理，不能把超时当成正常负例")
            time.sleep(0.0001)

        # 读 SCORE
        score = self.read("SCORE") & 0xFF
        return score


# ============================================================
# 判决状态机
# ============================================================


class FallDetector:
    """
    摔倒检测判决状态机
    四级状态：IDLE → CANDIDATE → ALARM_ARMED → ALARM

    判决逻辑（与方案第 4 章、第 9.9 节一致）：
    - IDLE: pe >= alarm_thr 且 motion 连续 trig_n 帧 → CANDIDATE
    - CANDIDATE: 累计 pe >= score_th 持续 candidate_min_s → ALARM_ARMED
    - CANDIDATE: pe < alarm_thr 连续 quiet_frames 帧 → IDLE
    - ALARM_ARMED: 持续 armed_min_s 仍满足 → ALARM
    - ALARM_ARMED: pe < alarm_thr 连续 quiet_frames 帧 → IDLE
    - ALARM: 人工复位 或 pe < alarm_thr 连续 2*quiet_frames 帧 → IDLE
    """

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

        # 状态
        self.state = FallState.IDLE
        self.pe = 0.0  # EMA 平滑后的摔倒概率
        self.peak_p = 0.0  # 本次事件的峰值概率

        # 计数器
        self.quiet_count = 0  # 连续无事件帧计数
        self.motion_count = 0  # 连续运动触发计数
        self.state_time = 0.0  # 进入当前状态的时间
        self.t_start = None  # 事件起始时间

        # 事件记录
        self.events = deque(maxlen=1000)
        self.last_roi = None  # 最近一帧 ROI (numpy)
        self.last_bbox = None  # 最近的 bbox
        self.event_id = 0
        self.on_event = None  # 事件回调（用于持久化），由外部注册

        # 参数
        self.ema_alpha = cfg["ema_alpha"]
        self.alarm_thr = cfg["alarm_thr"]
        self.score_th = cfg["score_th"]
        self.quiet_frames = cfg["quiet_frames"]
        self.candidate_min_s = cfg["candidate_min_s"]
        self.armed_min_s = cfg["armed_min_s"]

    def _log_event(self, kind, reason=None):
        """记录事件"""
        now = time.time()
        ev = {
            "ts": now,
            "ts_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_id": self.event_id,
            "event": kind,
            "state": STATE_NAMES[self.state],
            "peak_p": round(self.peak_p, 3),
            "pe": round(self.pe, 3),
            "duration_s": (
                round(time.monotonic() - self.t_start, 2) if self.t_start is not None else 0
            ),
            "bbox": self.last_bbox,
        }
        if reason:
            ev["reason"] = reason
        self.events.append(ev)

        if self.logger:
            self.logger.info(
                f"EVENT [{kind}] pe={self.pe:.3f} peak={self.peak_p:.3f} "
                f"state={STATE_NAMES[self.state]}"
            )

        if self.on_event:
            try:
                self.on_event(ev)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"事件回调执行失败: {e}")

        return ev

    def step(self, motion, score_p, bbox=None, roi=None):
        """
        推进一个检测周期

        Args:
            motion: bool, 是否有运动触发
            score_p: float, CNN 输出的摔倒概率 (0~1)，无触发时为 0
            bbox: tuple (x, y, w, h)，运动目标框
            roi: numpy array, 96x96 ROI 图像

        Returns:
            dict: 当前状态信息
        """
        # 更新 EMA
        self.pe = (1 - self.ema_alpha) * self.pe + self.ema_alpha * score_p

        # 更新峰值
        if self.state != FallState.IDLE and score_p > self.peak_p:
            self.peak_p = score_p

        # 保存最近的 bbox 和 ROI
        if bbox:
            self.last_bbox = bbox
        if roi is not None:
            self.last_roi = roi.copy()

        now = time.monotonic()
        prev_state = self.state

        # ---- 状态转换 ----
        if self.state == FallState.IDLE:
            if motion and self.pe >= self.alarm_thr:
                self.motion_count += 1
                if self.motion_count >= self.cfg.get("trig_n", 3):
                    self._enter_candidate(now)
            else:
                self.motion_count = 0

        elif self.state == FallState.CANDIDATE:
            if self.pe < self.score_th:
                self.state_time = now
            if self.pe < self.alarm_thr:
                self.quiet_count += 1
                if self.quiet_count >= self.quiet_frames:
                    self._enter_idle("quiet in candidate")
            else:
                self.quiet_count = 0
                # 低于确认阈值时重新计时，避免零散高分累积为持续摔倒。
                elapsed = now - self.state_time
                if elapsed >= self.candidate_min_s and self.pe >= self.score_th:
                    self._enter_armed()

        elif self.state == FallState.ALARM_ARMED:
            if self.pe < self.score_th:
                self.state_time = now
            if self.pe < self.alarm_thr:
                self.quiet_count += 1
                if self.quiet_count >= self.quiet_frames:
                    self._enter_idle("quiet in armed")
            else:
                self.quiet_count = 0
                elapsed = now - self.state_time
                if elapsed >= self.armed_min_s and self.pe >= self.score_th:
                    self._enter_alarm()

        elif self.state == FallState.ALARM:
            if self.pe < self.alarm_thr:
                self.quiet_count += 1
                if self.quiet_count >= self.quiet_frames * 2:
                    self._enter_idle("quiet in alarm")
            else:
                self.quiet_count = 0

        # 状态变化日志
        if self.state != prev_state and self.logger:
            self.logger.info(
                f"State: {STATE_NAMES[prev_state]} → {STATE_NAMES[self.state]} "
                f"pe={self.pe:.3f} peak={self.peak_p:.3f}"
            )

        return {
            "state": self.state,
            "state_name": STATE_NAMES[self.state],
            "pe": self.pe,
            "peak_p": self.peak_p,
            "motion": motion,
            "score": score_p,
            "bbox": self.last_bbox,
        }

    def _enter_candidate(self, now):
        self.state = FallState.CANDIDATE
        self.state_time = now
        self.t_start = now
        self.peak_p = self.pe
        self.quiet_count = 0
        self.event_id += 1
        self._log_event("candidate_start")

    def _enter_armed(self):
        self.state = FallState.ALARM_ARMED
        self.state_time = time.monotonic()
        self.quiet_count = 0
        self._log_event("alarm_armed")

    def _enter_alarm(self):
        self.state = FallState.ALARM
        self.state_time = time.monotonic()
        self.quiet_count = 0
        self._log_event("alarm")

    def _enter_idle(self, reason=""):
        prev = self.state
        self.state = FallState.IDLE
        self.quiet_count = 0
        self.motion_count = 0
        if prev in (FallState.CANDIDATE, FallState.ALARM_ARMED, FallState.ALARM):
            self._log_event("end", reason)
        self.t_start = None
        self.peak_p = 0.0

    def reset(self):
        """手动复位"""
        self._enter_idle("manual_reset")


# ============================================================
# 事件持久化
# ============================================================


class EventLogger:
    """事件记录器：写 JSONL 日志 + 保存 ROI 快照"""

    def __init__(self, events_file, snapshot_dir, save_snapshots=True, snapshot_scale=2):
        self.events_file = events_file
        self.snapshot_dir = snapshot_dir
        self.save_snapshots = save_snapshots
        self.snapshot_scale = snapshot_scale

        os.makedirs(os.path.dirname(os.path.abspath(events_file)), exist_ok=True)
        if save_snapshots:
            os.makedirs(snapshot_dir, exist_ok=True)

    def log_event(self, event, roi_img=None):
        """记录事件"""
        # 写 JSONL
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # 保存快照
        if self.save_snapshots and roi_img is not None:
            try:
                ts_ms = int(event["ts"] * 1000)
                fname = f"{event['event']}_{ts_ms}.png"
                fpath = os.path.join(self.snapshot_dir, fname)

                # roi_img 是 96x96 uint8 灰度
                if isinstance(roi_img, np.ndarray):
                    h, w = roi_img.shape[:2]
                    new_h, new_w = h * self.snapshot_scale, w * self.snapshot_scale
                    img = Image.fromarray(roi_img).resize((new_w, new_h), Image.NEAREST)
                    img.save(fpath)
            except Exception as e:
                print(f"保存快照失败: {e}", file=sys.stderr)


# ============================================================
# 主程序
# ============================================================


def main():
    from realtime import main as run_realtime

    return run_realtime()


if __name__ == "__main__":
    raise SystemExit(main())
