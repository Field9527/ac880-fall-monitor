#!/usr/bin/env python3
"""分阶段自检，使用与运行程序相同的寄存器和 DMA 生命周期。"""
import argparse
import fcntl
import json
import time

import fall_detect as fd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        type=int,
        choices=range(6),
        nargs="?",
        default=0,
        help="0:overlay 1:CNN寄存器 2:VDMA复位 3:视频轮转 4:帧心跳 5:仅VDMA",
    )
    p.add_argument("-c", "--config", default="config.json")
    args = p.parse_args()
    from pynq import Overlay, allocate, MMIO

    fd.allocate = allocate
    with open(args.config) as f:
        cfg = json.load(f)
    lock = fd.acquire_hardware_lock()
    log = fd.setup_logger("logs/hw_check.log")
    dma = None
    try:
        log.info("[MARK] 加载 Overlay")
        fd.pl_boot_fix(log)
        overlay = Overlay(cfg["overlay_path"])
        fd.pl_boot_fix(log)
        fd.pl_lvl_shifter_fix(log)
        if args.stage == 0:
            return 0
        if args.stage != 5:
            log.info("[MARK] CNN AP_CTRL/THRESH_Q7 寄存器回读，不启动未训练网络")
            entry = next(v for k, v in overlay.ip_dict.items() if "cnn_accel" in k)
            cnn = fd.CNNAccelIF(MMIO(entry["phys_addr"], entry["addr_range"]))
            log.info("AP_CTRL=0x%x", cnn.read("AP_CTRL"))
            old = cnn.read("THRESH_Q7")
            try:
                cnn.set_threshold(89)
                if cnn.read("THRESH_Q7") != 89:
                    raise RuntimeError("CNN THRESH_Q7 回读不符")
            finally:
                cnn.set_threshold(old)
        if args.stage == 1:
            return 0
        md = fd.MotionDetectIF(MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000))
        log.info("[MARK] 等待相机帧心跳")
        md.wait_frames()
        log.info("[MARK] VDMA 复位")
        entry = next(v for k, v in overlay.ip_dict.items() if "vdma" in k)
        v = cfg["video"]
        dma = fd.VDMAManager(MMIO(entry["phys_addr"], entry["addr_range"]), v["width"], v["height"])
        dma.reset_channel(fd.VDMA_S2MM_CR)
        dma.reset_channel(fd.VDMA_MM2S_CR)
        if args.stage in (2, 5):
            return 0
        log.info("[MARK] 分配帧缓冲并启动")
        dma.allocate_frames()
        dma.install_exit_guard()
        if not dma.start():
            raise RuntimeError("VDMA 未启动")
        time.sleep(2)
        dma._write(fd.VDMA_S2MM_SR, fd.VDMA_SR_CLEAR_MASK)
        seen = set()
        for _ in range(120):
            seen.add(dma.frame_indices()[1])
            if dma.s2mm_has_error():
                raise RuntimeError("VDMA 状态寄存器报错")
            time.sleep(0.017)
        if seen != {0, 1, 2}:
            raise RuntimeError("未观察到三个采集帧号轮转: %s" % seen)
        log.info("采集帧号轮转 PASS: %s", seen)
        if args.stage == 4:
            log.info("[MARK] 运动检测版本与相机帧心跳")
            md = fd.MotionDetectIF(MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000))
            ver = md.read("VERSION")
            a = md.get_snapshot()
            time.sleep(0.2)
            b = md.get_snapshot()
            if ver != 0x10000 or a is None or b is None or a["frame_cnt"] == b["frame_cnt"]:
                raise RuntimeError("版本或帧心跳异常")
        return 0
    except Exception:
        log.exception("硬件自检失败")
        return 1
    finally:
        if dma:
            dma.free()
        lock.close()
        log.info("自检结束；以退出码和具体检查为准")


if __name__ == "__main__":
    raise SystemExit(main())
