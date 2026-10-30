#!/usr/bin/env python3
"""OV5640 -> DDR -> HDMI 有限时长自检；结束时停 DMA 并释放帧缓存。"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image
import fall_detect as fd


def rgb565(frame):
    p = frame.astype(np.uint32)
    return np.stack(
        (((p >> 11) & 31) * 255 // 31, ((p >> 5) & 63) * 255 // 63, (p & 31) * 255 // 31), axis=-1
    ).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--output", default="video_check")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()
    if args.seconds < 3:
        p.error("--seconds 必须 >= 3")
    from pynq import Overlay, allocate, MMIO
    import fcntl

    lock = fd.acquire_hardware_lock()
    fd.allocate = allocate
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.output, exist_ok=True)
    fd.pl_boot_fix()
    ol = Overlay(cfg["overlay_path"], download=not args.skip_download)
    fd.pl_boot_fix()
    fd.pl_lvl_shifter_fix()
    entry = next(v for k, v in ol.ip_dict.items() if "vdma" in k.lower())
    v = cfg["video"]
    dma = fd.VDMAManager(
        MMIO(entry["phys_addr"], entry["addr_range"]),
        v["width"],
        v["height"],
        v["bpp"],
        v["num_frames"],
    )
    with open(cfg["overlay_path"], "rb") as f:
        bit_hash = hashlib.sha256(f.read()).hexdigest()
    report = {"passed": False, "seconds_requested": args.seconds, "bit_sha256": bit_hash}
    try:
        md = fd.MotionDetectIF(MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000))
        md.wait_frames()
        dma.allocate_frames()
        dma.install_exit_guard()
        patterns = [0x1234, 0x5678, 0x9ABC]
        for buf, pattern in zip(dma.frames, patterns):
            buf[:] = pattern
            buf.flush()
        if not dma.start():
            raise RuntimeError("VDMA 启动失败")
        time.sleep(2)
        dma._write(fd.VDMA_S2MM_SR, fd.VDMA_SR_CLEAR_MASK)
        dma._write(fd.VDMA_MM2S_SR, fd.VDMA_SR_CLEAR_MASK)
        rows, seen_w, seen_r = [], set(), set()
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            ri, wi = dma.frame_indices()
            seen_w.add(wi)
            seen_r.add(ri)
            row = {
                "t": round(time.monotonic() - t0, 3),
                "write_frame": wi,
                "read_frame": ri,
                "s2mm_sr": dma._read(fd.VDMA_S2MM_SR),
                "mm2s_sr": dma._read(fd.VDMA_MM2S_SR),
            }
            rows.append(row)
            if len(rows) % 10 == 0:
                print(json.dumps(row), flush=True)
            time.sleep(0.017)
        dma.stop()
        overwritten = []
        for i, (buf, pattern) in enumerate(zip(dma.frames, patterns)):
            buf.invalidate()
            frame = np.array(buf)
            overwritten.append(int(np.count_nonzero(frame != pattern)))
            Image.fromarray(rgb565(frame)).save(os.path.join(args.output, "frame%d.png" % i))
        clean = all(
            not (r["s2mm_sr"] | r["mm2s_sr"]) & fd.VDMA_SR_ERROR_MASK
            and not (r["s2mm_sr"] | r["mm2s_sr"]) & fd.VDMA_SR_HALTED
            for r in rows
        )
        report.update(
            samples=rows,
            write_frames=sorted(seen_w),
            read_frames=sorted(seen_r),
            overwritten_pixels=overwritten,
            errors_clean=clean,
            passed=clean
            and seen_w == {0, 1, 2}
            and seen_r == {0, 1, 2}
            and all(n > v["width"] * v["height"] * 0.95 for n in overwritten),
        )
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            dma.free()
        finally:
            with open(os.path.join(args.output, "report.json"), "w") as f:
                json.dump(report, f, indent=2)
    print(
        "PASS" if report["passed"] else "FAIL",
        {k: val for k, val in report.items() if k != "samples"},
        flush=True,
    )
    lock.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
