#!/usr/bin/env python3
"""AC880：相机/录像 -> 检测后端 -> 时序判决 -> HDMI/LED/JSONL。"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import logging
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import fall_detect as fd
from pose_rules import ForegroundDetector


class AlarmLED:
    """AC880 PS LED D34=MIO8，高电平亮；仅使用已复用为 GPIO 的引脚。"""

    def __init__(self, enabled, logger):
        self.path = None
        self.exported = False
        self.previous = None
        self.gpio = None
        self.slcr = None
        self.clock_was_enabled = True
        self.state = None
        if not enabled:
            return
        try:
            self.slcr = fd.MMIO(0xF8000000, 0x1000)
            # 本机镜像 APER_CLK_CTRL=0x00a00440，GPIO AMBA 时钟被关。
            # UG585: bit22 是 GPIO_CPU_1XCLKACT，与 PL FCLK 无关。
            aper = self.slcr.read(0x12C)
            self.clock_was_enabled = bool(aper & (1 << 22))
            if not self.clock_was_enabled:
                self.slcr.write(0x008, 0xDF0D)
                self.slcr.write(0x12C, aper | (1 << 22))
                self.slcr.write(0x004, 0x767B)
            # 本地 AC880 硬件手册第22页；不更改 PS 引脚复用。
            if self.slcr.read(0x720) & 0xFE:
                raise RuntimeError("MIO8 当前不是 GPIO，不能驱动 LED")
            chips = [
                p
                for p in glob.glob("/sys/class/gpio/gpiochip*")
                if "zynq_gpio" in Path(p, "label").read_text()
            ]
            if len(chips) != 1:
                raise RuntimeError("未找到 Zynq GPIO")
            self.gpio = int(Path(chips[0], "base").read_text()) + 8
            p = Path("/sys/class/gpio/gpio%d" % self.gpio)
            self.path = p
            if not p.exists():
                Path("/sys/class/gpio/export").write_text(str(self.gpio))
                self.exported = True
            else:
                self.previous = (
                    Path(p, "direction").read_text().strip(),
                    Path(p, "value").read_text().strip(),
                )
            Path(p, "direction").write_text("low")
            logger.info("报警 LED: D34 / MIO8 / gpio%d", self.gpio)
        except Exception as exc:
            try:
                self.close()
            except OSError:
                pass
            logger.warning("LED 不可用: %s；HDMI 与事件报警仍可用", exc)

    def set(self, alarm):
        if self.path is not None and self.state != bool(alarm):
            Path(self.path, "value").write_text("1" if alarm else "0")
            self.state = bool(alarm)

    def close(self):
        try:
            if self.path is not None and self.path.exists():
                Path(self.path, "value").write_text("0")
                if self.exported:
                    Path("/sys/class/gpio/unexport").write_text(str(self.gpio))
                elif self.previous is not None:
                    direction, value = self.previous
                    Path(self.path, "direction").write_text(direction)
                    if direction == "out":
                        Path(self.path, "value").write_text(value)
        finally:
            self.path = None
            if self.slcr is not None and not self.clock_was_enabled:
                aper = self.slcr.read(0x12C)
                self.slcr.write(0x008, 0xDF0D)
                self.slcr.write(0x12C, aper & ~(1 << 22))
                self.slcr.write(0x004, 0x767B)
                self.clock_was_enabled = True


def annotate(frame, result, observation, in_place=False, environment=None, gesture=None):
    from display_ui import HUD

    image = frame if in_place else frame.copy()
    HUD.draw(image, result, observation, environment, gesture)
    return image


def save_preview(frame, path, already_small=False):
    from fall_lite import rgb565

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    # 先缩小 RGB565 再展开三通道，避免 ARM 为预览反复分配全帧 uint32。
    small = frame if already_small else frame[::2, ::2]
    Image.fromarray(rgb565(small)).save(str(tmp), format="JPEG")
    os.replace(str(tmp), str(p))


class PreviewWriter:
    """最多一个 JPEG 在途，不积压预览帧。"""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.skipped = 0

    def submit(self, frame, path, required=False):
        if self.future is not None:
            if not self.future.done():
                if not required:
                    self.skipped += 1
                    return False
                # 报警截图必须保留；最多等待一个在途 JPEG。
                self.future.result()
            self.future.result()
        # 显示缓冲区复用前复制所需像素。
        small = np.array(frame[::2, ::2], copy=True)
        self.future = self.pool.submit(save_preview, small, path, True)
        return True

    def close(self):
        self.pool.shutdown(wait=True)
        if self.future is not None:
            self.future.result()


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    os.replace(str(tmp), str(p))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="config_fall_lite.json")
    p.add_argument("-b", "--overlay")
    p.add_argument(
        "--backend", choices=("rules", "fall_lite"), help="rules 联调基线 / fall_lite 无背景模型"
    )
    p.add_argument(
        "--test-context",
        choices=("live_scene", "phone_screen_replay"),
        default="live_scene",
        help="标记实际场景或 OV5640 拍摄手机回放，写入画面与事件记录",
    )
    p.add_argument("--duration", type=float, default=0, help="运行秒数，0=持续运行")
    p.add_argument("--display-fps", type=float, help="显示更新上限 1..30；不改变分辨率或模型输入")
    p.add_argument("--simulate", action="store_true", help="纯软件注入状态机测试")
    p.add_argument("--replay", help="用录像验证同一套前景/规则算法，无需开发板")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--preview-only", action="store_true", help="仅预览 OV5640，不做识别或背景采集")
    p.add_argument(
        "--diagnostic-raw-dir", help="显式留存最多12个高分原始推理帧，目录须不存在；仅用于样本复核"
    )
    p.add_argument(
        "--diagnostic-start-paused",
        action="store_true",
        help="诊断采样等 SIGUSR2 后开始，仍最多12帧",
    )
    p.add_argument(
        "--wait-background", action="store_true", help="先预览；SIGUSR1 开始或重新采集无人背景"
    )
    p.add_argument("--no-motion", action="store_true", help="不启用 PL 运动计算，仍读取帧心跳")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-web", action="store_true", help="仅HDMI识别，不启动网页接口")
    p.add_argument("--inference-engine", choices=("cpu", "pl"), help="CPU NCNN或PS+PL协同推理")
    p.add_argument(
        "--image-mode",
        choices=("color", "grayscale", "edge"),
        help="显示图像模式；识别始终读取原始彩色帧",
    )
    p.add_argument("--alarm-test", action="store_true", help="仅测试 D34 报警 LED 一秒后退出")
    p.add_argument(
        "--test-alarm-output",
        action="store_true",
        help="注入测试分数验证板上 HDMI/LED/事件输出，不能用于验收识别准确率",
    )
    args = p.parse_args()
    if args.duration < 0:
        p.error("--duration 不能为负数")
    if args.simulate and args.replay:
        p.error("--simulate 和 --replay 只能选一项")
    hardware = not (args.simulate or args.replay)
    if args.test_context == "phone_screen_replay" and not hardware:
        p.error("phone_screen_replay 指 OV5640 拍摄屏幕，需使用开发板")
    if args.preview_only and (
        not hardware or args.wait_background or args.alarm_test or args.test_alarm_output
    ):
        p.error("--preview-only 只用于单独的板端相机预览")
    if args.test_alarm_output and not hardware:
        p.error("--test-alarm-output 需要开发板，不能与模拟/回放混用")
    if args.wait_background and (not hardware or args.test_alarm_output or args.alarm_test):
        p.error("--wait-background 只用于正常板端识别")
    if args.test_alarm_output and not args.duration:
        args.duration = 18
    if hardware and args.no_video and not args.alarm_test:
        p.error("实时识别需要视频；离线检查使用 --simulate 或 --replay")
    with open(args.config, encoding="utf8") as f:
        cfg = json.load(f)
    if args.inference_engine:
        if cfg.get("fall_lite", {}).get("engine") == "semantic":
            if args.inference_engine == "pl":
                p.error("共享模型采用CPU推理；PL保留视频与图像处理")
        else:
            cfg["fall_lite"]["engine"] = "ncnn_pl" if args.inference_engine == "pl" else "ncnn"
    if args.image_mode:
        cfg.setdefault("image_processing", {})["mode"] = args.image_mode
    if args.no_web:
        cfg.setdefault("web", {})["enabled"] = False
    if args.display_fps is not None:
        if not 1 <= args.display_fps <= 30:
            p.error("--display-fps 必须在 1..30 之间")
        cfg["video"]["poll_hz"] = args.display_fps
    backend = args.backend or cfg.get("backend", "rules")
    if backend not in ("rules", "fall_lite"):
        p.error("未知识别后端；PL CNN 尚无经验证的训练权重")
    if args.diagnostic_raw_dir and (
        not hardware
        or backend != "fall_lite"
        or args.preview_only
        or args.test_alarm_output
        or args.alarm_test
    ):
        p.error("--diagnostic-raw-dir 只用于真实 OV5640 模型推理")
    if args.wait_background and backend != "rules":
        p.error("fall_lite 不需要背景采集；不要传 --wait-background")
    if args.preview_only:
        backend = "preview"
    decision = cfg["decision"]
    gesture_hold_s = float(cfg.get("gesture_control", {}).get("hold_s", 0.8))
    if not 0.5 <= gesture_hold_s <= 2.0:
        p.error("手势稳定确认时间应为0.5～2秒")
    if not (
        0 < decision["ema_alpha"] <= 1
        and 0 <= decision["alarm_thr"] <= decision["score_th"] <= 1
        and decision["candidate_min_s"] >= 0
        and decision["armed_min_s"] >= 0
        and decision["quiet_frames"] >= 1
        and 0 < cfg["video"]["poll_hz"] <= 30
    ):
        p.error("判决阈值、持续时间、静默帧数或轮询频率配置不合法")
    if not hardware:
        # 离线输出固定放在当前目录，避免写入板端绝对路径。
        base = Path.cwd() / "offline_run"
        cfg["logging"] = dict(
            cfg["logging"],
            log_file=str(base / "run.log"),
            events_file=str(base / "events.jsonl"),
            snapshot_dir=str(base / "snapshots"),
        )
    log_cfg = cfg["logging"]
    if args.test_alarm_output:
        base = Path(log_cfg["log_file"]).parent / "output_test"
        log_cfg = dict(
            log_cfg,
            log_file=str(base / "run.log"),
            events_file=str(base / "events.jsonl"),
            snapshot_dir=str(base / "snapshots"),
        )
    logger = fd.setup_logger(log_cfg["log_file"])
    evt = fd.EventLogger(
        log_cfg["events_file"],
        log_cfg["snapshot_dir"],
        log_cfg.get("save_snapshots", True),
        log_cfg.get("snapshot_scale", 2),
    )
    detector = fd.FallDetector(cfg["decision"], logger)
    source = (
        "synthetic_output_test"
        if args.test_alarm_output
        else "synthetic" if args.simulate else "replay" if args.replay else "ov5640"
    )
    detector.on_event = lambda event: evt.log_event(
        dict(
            event,
            backend=backend,
            input_source=source,
            test_context=event.get("test_context", args.test_context),
            model_sha256=getattr(segmenter, "model_sha256", None),
        ),
        detector.last_roi,
    )
    segmenter = None
    if not (args.simulate or args.alarm_test or args.test_alarm_output or args.preview_only):
        if backend == "fall_lite":
            from fall_lite import FallLiteDetector

            model_cfg = dict(cfg["fall_lite"])
            if args.diagnostic_raw_dir:
                model_cfg.update(
                    diagnostic_raw_dir=args.diagnostic_raw_dir, test_context=args.test_context
                )
            # 相对模型路径以配置文件所在目录为准。
            for key in ("model", "metadata", "runtime_library", "pl_plugin"):
                if key in model_cfg and not Path(model_cfg[key]).is_absolute():
                    model_cfg[key] = str(Path(args.config).resolve().parent / model_cfg[key])
            if model_cfg.get("engine") == "semantic":
                from shared_detector import SharedFallDetector
                from tracked_fall import TrackedFallDetector

                if args.test_context == "phone_screen_replay":
                    model_cfg["monitor_mode"] = "phone_replay"
                segmenter = SharedFallDetector(model_cfg)
                callback = detector.on_event
                detector = TrackedFallDetector(cfg["decision"], logger)
                detector.on_event = callback
            else:
                segmenter = FallLiteDetector(model_cfg)
            if args.diagnostic_raw_dir:
                segmenter.diagnostic_active = not args.diagnostic_start_paused
            logger.info(
                "模型 SHA256=%s，训练记录=%s", segmenter.model_sha256, segmenter.training_run
            )
        else:
            segmenter = ForegroundDetector(cfg["rules"])
    dma = md = led = capture = lock = None
    hardware_owned = False
    inference_worker = None
    preview_writer = None
    home_web = home_outputs = None
    environment = None
    recognition_switch = None
    last_home_status = -float("inf")
    timings = {k: deque(maxlen=4096) for k in ("copy_ms", "draw_ms", "present_ms", "interval_ms")}
    first_camera_count = first_camera_time = last_present = None
    copy_rejected = 0
    performance = {}
    fps = float(cfg["video"].get("poll_hz", 10))
    start = last_status = last_camera = time.monotonic()
    last_frame_count = None
    processed = 0
    status_file = str(Path(log_cfg["log_file"]).parent / "status.json")
    preview_file = str(Path(log_cfg["log_file"]).parent / "latest.jpg")
    stopped = False
    diagnostic_requested = False
    failure = None
    last_state = fd.FallState.IDLE
    waiting_background = args.wait_background
    recalibrate = False
    pid_file = Path(log_cfg["log_file"]).parent / "detector.pid"

    def stop_signal(signum, frame):
        nonlocal stopped
        stopped = True

    def diagnostic_signal(signum, frame):
        nonlocal diagnostic_requested
        diagnostic_requested = True

    def calibrate_signal(signum, frame):
        nonlocal recalibrate
        recalibrate = True

    def performance_report(now):
        report = dict(
            display_fps=round(processed / max(now - start, 0.001), 2),
            display_limit_fps=fps,
            copy_rejected=copy_rejected,
            resolution=[cfg["video"]["width"], cfg["video"]["height"]],
            pixel_format="RGB565",
            camera_fps=(
                round(
                    ((last_frame_count - first_camera_count) & 0xFFFFFFFF)
                    / max(last_camera - first_camera_time, 0.001),
                    2,
                )
                if first_camera_count is not None
                else None
            ),
            inference_fps=(
                round(inference_worker.fall_completed / max(now - start, 0.001), 2)
                if inference_worker
                else None
            ),
            stage_ms={
                k: dict(
                    mean=round(float(np.mean(v)), 2),
                    p95=round(float(np.percentile(v, 95)), 2),
                    maximum=round(float(max(v)), 2),
                )
                for k, v in timings.items()
                if v
            },
        )
        report.update(
            engine=getattr(segmenter, "engine", backend),
            model_sha256=getattr(segmenter, "model_sha256", None),
            input_source=source,
            test_context=args.test_context,
        )
        if getattr(segmenter, "clock_evidence", None):
            report["clock_evidence"] = segmenter.clock_evidence
        if dma is not None and dma.copy_engine is not None:
            report["frame_copy"] = dma.copy_engine.statistics()
        return report

    try:
        if hardware:
            import fcntl
            from pynq import Overlay, allocate, MMIO

            fd.Overlay, fd.allocate, fd.MMIO = Overlay, allocate, MMIO
            lock = fd.acquire_hardware_lock()
            hardware_owned = True
            led = AlarmLED(cfg["output"].get("gpio_alarm", True), logger)
            if args.alarm_test:
                if led.path is None:
                    raise RuntimeError("LED 测试失败")
                led.set(True)
                gpio_mmio = MMIO(0xE000A000, 0x1000)
                high = (gpio_mmio.read(0x40) >> 8) & 1
                time.sleep(1)
                led.set(False)
                low = (gpio_mmio.read(0x40) >> 8) & 1
                if (high, low) != (1, 0):
                    raise RuntimeError("LED 输出回读失败")
                logger.info(
                    "ALARM_LED_REGISTER_TEST_PASS: MIO8 输出锁存 1 -> 0；实际灯光需目视确认"
                )
                return 0
            bit = fd.resolve_overlay_path(args.overlay or cfg["overlay_path"], args.config, logger)
            if bit is None:
                return 1
            fd.pl_boot_fix(logger)
            overlay = Overlay(bit, download=not args.skip_download)
            fd.pl_boot_fix(logger)
            fd.pl_lvl_shifter_fix(logger)
            if segmenter is not None and hasattr(segmenter, "attach_pl"):
                segmenter.attach_pl(overlay)
            md = fd.MotionDetectIF(MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000))
            from environment import EnvironmentMonitor

            environment = EnvironmentMonitor(
                MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000), cfg.get("environment")
            )
            if cfg.get("environment", {}).get("enabled"):
                logger.info(
                    "温湿度采集: %s",
                    "ENV1 / P7-31 DHT11" if environment.available else "当前硬件未提供ENV1",
                )
            logger.info("等待 OV5640 实际出帧...")
            md.wait_frames()
            entry = next(v for k, v in overlay.ip_dict.items() if "vdma" in k.lower())
            v = cfg["video"]
            if (v["width"], v["height"], v["bpp"], v["num_frames"]) != (1280, 720, 16, 3):
                raise ValueError("当前比特流固定为 1280x720 RGB565 / 3 帧缓存")
            dma = fd.VDMAManager(
                MMIO(entry["phys_addr"], entry["addr_range"]),
                v["width"],
                v["height"],
                v["bpp"],
                v["num_frames"],
            )
            if v.get("pl_frame_copy", False):
                from frame_copy import FrameCopy

                copy_entry = overlay.ip_dict.get("frame_copy")
                if copy_entry is None:
                    raise RuntimeError("PL video bitstream with frame_copy IP required")
                fd.overlay_clock_evidence(overlay)
                if "axi_dma:" in copy_entry.get("type", ""):
                    from image_processing import ImageFrameCopy

                    dma.copy_engine = ImageFrameCopy(
                        MMIO(copy_entry["phys_addr"], copy_entry["addr_range"]),
                        MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000),
                        cfg.get("image_processing"),
                    )
                    logger.info("PL image DMA: original/grayscale/Sobel, brightness/contrast")
                else:
                    dma.copy_engine = FrameCopy(
                        MMIO(copy_entry["phys_addr"], copy_entry["addr_range"])
                    )
                    logger.info("PL AXI CDMA hardware frame copy enabled")
            dma.allocate_frames()
            dma.allocate_display(cacheable=v.get("display_cacheable", False))
            dma.install_exit_guard()
            if not dma.start():
                raise RuntimeError("VDMA 启动失败")
            md.configure(cfg["motion_detect"], v["width"], v["height"])
            md.disable() if args.no_motion or backend != "rules" else md.enable()
            if args.test_alarm_output:
                logger.warning("TEST INPUT: 人工分数测试输出链路，不能作为真实识别通过记录")
            elif args.preview_only:
                logger.info("仅预览 OV5640，不做识别或背景采集")
            elif backend == "fall_lite":
                logger.info("无背景模型已启动；从首帧直接检测倒地人体")
            elif not waiting_background:
                logger.info("视频已启动；前 %.1f 秒保持无人以初始化背景", cfg["rules"]["warmup_s"])
            logger.warning("识别后端=%s；现场泛化精度与 PL 模型加速尚未验收", backend)
            time.sleep(2)
            dma._write(fd.VDMA_S2MM_SR, fd.VDMA_SR_CLEAR_MASK)
            dma._write(fd.VDMA_MM2S_SR, fd.VDMA_SR_CLEAR_MASK)
            if backend == "fall_lite" and segmenter is not None:
                from fall_lite import LatestFrameInference

                if getattr(segmenter, "independent_gestures", False):
                    from dual_inference import ConcurrentGestureInference

                    inference_worker = ConcurrentGestureInference(
                        segmenter,
                        timeout_s=5.0,
                        partition_cpus=cfg["fall_lite"].get("partition_cpus", False),
                    )
                else:
                    inference_worker = LatestFrameInference(
                        segmenter,
                        timeout_s=(
                            10.0
                            if segmenter.engine == "ncnn_pl"
                            else 5.0 if segmenter.engine == "semantic" else 3.0
                        ),
                    )
            if (
                cfg.get("recognition_switch", {}).get("enabled", False)
                and getattr(segmenter, "semantic_options", None) is not None
            ):
                from recognition_mode import RecognitionSwitch

                recognition_switch = RecognitionSwitch(
                    MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000),
                    segmenter.semantic_options,
                    cfg["recognition_switch"],
                )
            preview_writer = PreviewWriter()
            if cfg.get("web", {}).get("enabled", False) or recognition_switch is not None:
                from home_web import HomeOutputs, HomeWeb

                home_outputs = HomeOutputs(
                    MMIO(int(cfg["motion_detect"]["base_addr"], 0), 0x1000),
                    led,
                    logger,
                    image_processing=(
                        dma.copy_engine if hasattr(dma.copy_engine, "configuration") else None
                    ),
                    semantic=getattr(segmenter, "semantic_options", None),
                )
                if cfg.get("web", {}).get("enabled", False):
                    home_web = HomeWeb(home_outputs, cfg["web"], logger)
        elif args.replay:
            import cv2

            capture = cv2.VideoCapture(args.replay)
            if not capture.isOpened():
                raise RuntimeError("无法打开录像: " + args.replay)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop_signal)
        if hardware and backend == "rules" and segmenter is not None and hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, calibrate_signal)
        if hardware and args.diagnostic_raw_dir and hasattr(signal, "SIGUSR2"):
            signal.signal(signal.SIGUSR2, diagnostic_signal)
        if hardware:
            pid_file.write_text(str(os.getpid()))
        if waiting_background:
            logger.info("预览模式：确认画面无人后，用 SIGUSR1 开始背景初始化")
        start = last_status = last_camera = time.monotonic()
        result = dict(
            state=fd.FallState.IDLE,
            state_name="IDLE",
            pe=0.0,
            peak_p=0.0,
            motion=False,
            score=0.0,
            bbox=None,
        )
        semantic_generation = None
        from gesture_control import GestureController, ACTIONS

        gesture_controller = GestureController(hold_s=gesture_hold_s)
        gesture_epoch = None
        while not stopped and (not args.duration or time.monotonic() - start < args.duration):
            loop = time.monotonic()
            if recognition_switch is not None and recognition_switch.poll(loop):
                logger.info("SW0 识别模式：%s", recognition_switch.mode)
                if home_outputs:
                    home_outputs.recognition_changed(recognition_switch.mode)
            if diagnostic_requested:
                if segmenter is not None and getattr(segmenter, "diagnostic_dir", None) is not None:
                    segmenter.diagnostic_active = True
                    logger.info("DIAGNOSTIC: finite raw capture armed, maximum 12 frames")
                diagnostic_requested = False
            new_evidence = True
            if recalibrate:
                segmenter.reset()
                detector.reset()
                waiting_background = False
                recalibrate = False
                logger.info("重新初始化无人背景，保持画面无人 %.1f 秒", cfg["rules"]["warmup_s"])
            frame = None
            display_target = None
            if args.simulate:
                t = (loop - start) % 18
                score = 0.95 if 2 < t < 12 else 0
                observation = dict(
                    ready=True,
                    bbox=(100, 300, 360, 150) if score else None,
                    score=score,
                    area=1000 if score else 0,
                    reason="synthetic",
                )
            else:
                if capture is not None:
                    import cv2

                    ok, bgr = capture.read()
                    if not ok:
                        break
                    frame = (
                        (bgr[:, :, 2].astype(np.uint16) >> 3) << 11
                        | (bgr[:, :, 1].astype(np.uint16) >> 2) << 5
                        | (bgr[:, :, 0].astype(np.uint16) >> 3)
                    )
                else:
                    if dma.has_error(fd.VDMA_S2MM_SR) or dma.has_error(fd.VDMA_MM2S_SR):
                        raise RuntimeError("VDMA 出现错误，停止识别以防使用不完整帧")
                    snap = md.get_snapshot()
                    if snap is None or snap["frame_cnt"] == last_frame_count:
                        if loop - last_camera > 2:
                            raise RuntimeError("相机帧心跳超过 2 秒未更新")
                        time.sleep(0.01)
                        continue
                    last_frame_count = snap["frame_cnt"]
                    last_camera = loop
                    if first_camera_count is None:
                        first_camera_count, first_camera_time = last_frame_count, loop
                    copied_at = time.monotonic()
                    display_target, frame = dma.prepare_display_frame()
                    timings["copy_ms"].append((time.monotonic() - copied_at) * 1000)
                    if frame is None:
                        copy_rejected += 1
                        continue
                if args.test_alarm_output:
                    t = loop - start
                    score = 0.95 if 2 < t < 12 else 0
                    observation = dict(
                        ready=True,
                        bbox=(100, 300, 360, 150) if score else None,
                        score=score,
                        area=0,
                        reason="synthetic_output_test",
                        test_only=True,
                    )
                elif args.preview_only:
                    observation = dict(
                        ready=True,
                        bbox=None,
                        score=0.0,
                        area=0,
                        reason="preview_only",
                        backend="preview",
                    )
                elif waiting_background:
                    observation = dict(
                        ready=False, bbox=None, score=0.0, area=0, reason="await_background"
                    )
                elif inference_worker is not None:
                    observation, new_evidence = inference_worker.poll(
                        lambda: (
                            dma.raw_inference_frame()
                            if dma is not None and getattr(dma.copy_engine, "transformed", False)
                            else np.array(frame, copy=True)
                        ),
                        loop,
                    )
                else:
                    observation = segmenter.process(frame, loop)
            observation = dict(observation)
            if getattr(segmenter, "semantic_options", None) is not None:
                options = segmenter.semantic_options.configuration
                if semantic_generation != options["generation"]:
                    detector.reset()
                    semantic_generation = options["generation"]
                    result = dict(
                        state=fd.FallState.IDLE,
                        state_name="IDLE",
                        pe=0.0,
                        peak_p=0.0,
                        motion=False,
                        score=0.0,
                        bbox=None,
                    )
                if observation.get("semantic_generation") != semantic_generation:
                    new_evidence = False
                    observation.update(
                        ready=False, bbox=None, score=0.0, reason="model_mode_switch"
                    )
                observation["show_objects"] = options["show_objects"]
                observation["recognition_mode"] = options["recognition_mode"]
                if options["mode"] == "phone_replay":
                    observation["test_context"] = "phone_screen_replay"
                if options["recognition_mode"] not in ("fall", "both"):
                    new_evidence = False
                    observation.update(bbox=None, score=0.0, detections=[])
            bbox = observation["bbox"]
            roi = None
            if frame is not None and bbox and "_roi" not in observation and new_evidence:
                x, y, w, h = bbox
                crop = frame[y : y + h, x : x + w]
                gray = (((crop >> 5) & 63) * 4).astype(np.uint8)
                roi = np.array(Image.fromarray(gray).resize((96, 96), Image.NEAREST))
            if "_roi" in observation:
                roi = observation["_roi"]
            active = (
                observation["score"] >= cfg["decision"]["score_th"]
                if backend == "fall_lite"
                else observation["score"] > 0.9
            )
            if new_evidence and observation.get("fall_fresh", True):
                if hasattr(detector, "step_many"):
                    result = detector.step_many(
                        observation.get("detections", []),
                        inference_worker.completed if inference_worker else processed,
                        observation.get("source_monotonic", loop),
                        loop,
                        roi,
                        observation["bbox"],
                    )
                    if result.get("evidence_status") == "stale":
                        observation["ready"] = False
                    if result["state"] != fd.FallState.IDLE:
                        observation["bbox"] = result["bbox"]
                else:
                    result = detector.step(active, observation["score"], bbox, roi)
            observation.setdefault("test_context", args.test_context)
            if home_outputs:
                if getattr(segmenter, "hands", None) is not None:
                    options = segmenter.semantic_options.configuration
                    epoch = (options["generation"], options["gesture_generation"])
                    if gesture_epoch != epoch:
                        gesture_controller = GestureController(hold_s=gesture_hold_s)
                        gesture_epoch = epoch
                    hand_at = observation.get("hand_sampled_at")
                    enabled = (
                        options["gesture_enabled"]
                        and options["mode"] == "live"
                        and options["recognition_mode"] in ("both", "gesture")
                        and observation.get("gesture_generation") == epoch[1]
                        and observation.get("semantic_generation") == epoch[0]
                    )
                    command = gesture_controller.step(
                        observation.get("hand"),
                        observation.get("hand_sequence"),
                        hand_at,
                        loop,
                        enabled,
                    )
                    if command:
                        home_outputs.gesture_command(command, ACTIONS[observation["hand"]["name"]])
                    if hand_at is None or loop - hand_at > 3.0:
                        observation["hand"] = None
                home_outputs.tick(result["state"] == fd.FallState.ALARM)
            elif led:
                led.set(result["state"] == fd.FallState.ALARM)
            observation["gesture_enabled"] = bool(
                getattr(segmenter, "gestures_enabled", lambda: False)()
            )
            drawing_at = time.monotonic()
            environment_value = environment.poll(loop) if environment is not None else None
            shown = (
                annotate(
                    frame,
                    result,
                    observation,
                    in_place=dma is not None,
                    environment=environment_value,
                    gesture=gesture_controller.snapshot(loop),
                )
                if frame is not None
                else None
            )
            timings["draw_ms"].append((time.monotonic() - drawing_at) * 1000)
            if home_web is not None and shown is not None:
                status = None
                if loop - last_home_status >= 0.25:
                    elapsed = max(loop - start, 0.001)
                    status = dict(
                        result,
                        running=True,
                        ready=observation["ready"],
                        engine=getattr(segmenter, "engine", backend),
                        test_context=observation["test_context"],
                        model_sha256=getattr(segmenter, "model_sha256", None),
                        input_source=source,
                        uptime_s=round(elapsed, 1),
                        display_fps=round(processed / elapsed, 2),
                        inference_fps=(
                            round(inference_worker.fall_completed / elapsed, 2)
                            if inference_worker
                            else None
                        ),
                        inference_ms=observation.get("inference_ms"),
                        bbox=observation.get("bbox"),
                        pl_statistics=observation.get("pl_statistics"),
                        environment=environment_value,
                        objects=observation.get("objects", []),
                        hand=observation.get("hand"),
                        gesture_control=gesture_controller.snapshot(loop),
                        hand_status=observation.get("hand_status"),
                        gesture_ms=observation.get("gesture_ms"),
                        gesture_fps=round(
                            getattr(inference_worker, "hand_completed", 0) / elapsed, 2
                        ),
                        hand_age_s=(
                            round(loop - observation["hand_sampled_at"], 2)
                            if observation.get("hand_sampled_at") is not None
                            else None
                        ),
                        object_age_s=(
                            round(max(0, loop - observation["object_sampled_at"]), 2)
                            if observation.get("object_sampled_at") is not None
                            else None
                        ),
                        reviewed_candidates=observation.get("reviewed_candidates", []),
                        coverage_limited=observation.get("coverage_limited", False)
                        or result.get("tracking_limited", False),
                        monitor_mode=observation.get("monitor_mode"),
                        recognition_mode=observation.get("recognition_mode", "both"),
                        replay_rotation=observation.get("replay_rotation", 0),
                        evidence_age_s=(
                            round(loop - observation["source_monotonic"], 2)
                            if observation.get("source_monotonic") is not None
                            else None
                        ),
                    )
                    last_home_status = loop
                home_web.publish(shown, status)
            if dma:
                presenting_at = time.monotonic()
                dma.commit_display_frame(display_target)
                presented_at = time.monotonic()
                timings["present_ms"].append((presented_at - presenting_at) * 1000)
                if last_present is not None:
                    timings["interval_ms"].append((presented_at - last_present) * 1000)
                last_present = presented_at
            if (
                shown is not None
                and result["state"] == fd.FallState.ALARM
                and last_state != fd.FallState.ALARM
            ):
                alarm_path = str(
                    Path(log_cfg["snapshot_dir"]) / ("alarm_display_%d.jpg" % int(time.time()))
                )
                if preview_writer is not None:
                    preview_writer.submit(shown, alarm_path, required=True)
                else:
                    save_preview(shown, alarm_path)
            last_state = result["state"]
            processed += 1
            now = time.monotonic()
            if now - last_status >= 2:
                performance = performance_report(now)
                status = dict(
                    result,
                    bbox=observation["bbox"],
                    running=True,
                    backend=backend,
                    ready=observation["ready"],
                    input_source=source,
                    test_context=args.test_context,
                    model_sha256=getattr(segmenter, "model_sha256", None),
                    reason=observation["reason"],
                    area=observation["area"],
                    processed=processed,
                    fps=round(processed / (now - start), 2),
                    uptime_s=round(now - start, 2),
                    timestamp=time.time(),
                )
                status.update(performance)
                status["environment"] = environment_value
                status["recognition_mode"] = observation.get("recognition_mode", "both")
                if "inference_ms" in observation:
                    status.update(
                        inference_ms=observation["inference_ms"],
                        detections=observation["detections"],
                        engine=observation.get("engine", "opencv"),
                        pl_statistics=observation.get("pl_statistics"),
                    )
                if inference_worker is not None:
                    status.update(
                        inference_fps=round(inference_worker.fall_completed / (now - start), 2),
                        result_age_s=round(now - observation.get("source_monotonic", now), 2),
                    )
                atomic_json(status_file, status)
                if shown is not None:
                    if preview_writer is not None:
                        preview_writer.submit(shown, preview_file)
                    else:
                        save_preview(shown, preview_file)
                logger.info(
                    "fps=%.1f state=%s score_ema=%.3f area=%d ready=%s reason=%s",
                    status["fps"],
                    result["state_name"],
                    result["pe"],
                    observation["area"],
                    observation["ready"],
                    observation["reason"],
                )
                last_status = now
            time.sleep(max(0, 1 / fps - (time.monotonic() - loop)))
        logger.info("运行结束: processed=%d events=%d", processed, len(detector.events))
        return 0
    except Exception as exc:
        failure = str(exc)
        logger.exception("实时检测退出")
        return 1
    finally:
        ended_at = time.monotonic()
        final_pl = getattr(getattr(segmenter, "net", None), "pl", None)
        if processed:
            performance = performance_report(ended_at)
        try:
            if inference_worker:
                inference_worker.close()
            elif segmenter is not None and hasattr(segmenter, "close"):
                segmenter.close()
            if preview_writer:
                preview_writer.close()
            if home_web:
                home_web.close()
            if home_outputs:
                home_outputs.close()
            if led:
                led.close()
            if md:
                md.disable()
        finally:
            # 推理任务排空最后一个 AXI 事务后再统计卸载量。
            if processed and final_pl is not None:
                performance["pl_statistics"] = final_pl.statistics()
            try:
                if dma:
                    dma.free()
            finally:
                if capture:
                    capture.release()
                if (
                    hardware
                    and pid_file.exists()
                    and pid_file.read_text().strip() == str(os.getpid())
                ):
                    pid_file.unlink()
                if lock:
                    lock.close()
                if not hardware or hardware_owned:
                    if processed:
                        performance.update(
                            processed=processed,
                            uptime_s=round(ended_at - start, 2),
                            preview_skipped=preview_writer.skipped if preview_writer else 0,
                            error=failure,
                        )
                        atomic_json(
                            str(Path(status_file).with_name("performance.json")), performance
                        )
                    atomic_json(
                        status_file,
                        {
                            "state_name": "STOPPED",
                            "running": False,
                            "processed": processed,
                            "backend": backend,
                            "test_context": args.test_context,
                            "error": failure,
                            "performance": performance,
                            "timestamp": time.time(),
                        },
                    )


if __name__ == "__main__":
    raise SystemExit(main())
