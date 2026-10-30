#!/usr/bin/env python3
"""HDMI 空间映射诊断：输出四个不同色块和四角坐标，排除相机内容影响。"""
import argparse
import fcntl
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fall_detect as fd


def pattern(width=1280, height=720):
    im = Image.new("RGB", (width, height))
    d = ImageDraw.Draw(im)
    colors = [(180, 30, 30), (30, 150, 30), (30, 30, 180), (180, 140, 30)]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    for idx, color in enumerate(colors):
        x, y = (idx % 2) * (width // 2), (idx // 2) * (height // 2)
        d.rectangle((x, y, x + width // 2 - 1, y + height // 2 - 1), fill=color)
        d.text(
            (x + 25, y + 35),
            "%d: %s %s"
            % (idx + 1, "TOP" if idx < 2 else "BOTTOM", "LEFT" if idx % 2 == 0 else "RIGHT"),
            fill="white",
            font=font,
        )
    d.rectangle((0, 0, width - 1, height - 1), outline="white", width=8)
    d.line((0, height // 2, width, height // 2), fill="white", width=4)
    d.line((width // 2, 0, width // 2, height), fill="white", width=4)
    a = np.array(im).astype(np.uint16)
    pixels = ((a[:, :, 0] >> 3) << 11) | ((a[:, :, 1] >> 2) << 5) | (a[:, :, 2] >> 3)
    return im, pixels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("--seconds", type=float, default=90)
    args = p.parse_args()
    from pynq import Overlay, allocate, MMIO

    fd.allocate = allocate
    lock = fd.acquire_hardware_lock()
    with open(args.config) as f:
        cfg = json.load(f)
    fd.pl_boot_fix()
    ol = Overlay(cfg["overlay_path"])
    fd.pl_boot_fix()
    fd.pl_lvl_shifter_fix()
    entry = next(v for k, v in ol.ip_dict.items() if "vdma" in k.lower())
    dma = fd.VDMAManager(MMIO(entry["phys_addr"], entry["addr_range"]), 1280, 720)
    try:
        dma.allocate_frames()
        dma.install_exit_guard()
        im, pixels = pattern()
        im.save("display_expected.png")
        for frame in dma.frames:
            frame[:] = pixels
            frame.flush()
        dma.reset_channel(fd.VDMA_MM2S_CR)
        dma.config_channel(
            fd.VDMA_MM2S_CR, fd.VDMA_MM2S_START, fd.VDMA_MM2S_HSIZE, fd.VDMA_MM2S_STRIDE
        )
        if not dma.start_channel(fd.VDMA_MM2S_CR, fd.VDMA_MM2S_VSIZE, fd.VDMA_MM2S_SR):
            raise RuntimeError("MM2S 启动失败")
        print("EXPECTED: top red/green; bottom blue/yellow; labels 1/2/3/4 once each", flush=True)
        time.sleep(args.seconds)
    finally:
        dma.free()
        lock.close()


if __name__ == "__main__":
    main()
