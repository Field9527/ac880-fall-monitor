"""在电脑回放公共视频，并按板端节拍推进时序；深度图不参与推理，结果不作为 OV5640 现场验收。"""

import cache_env
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import FallLiteDetector, rgb565
from fall_detect import FallDetector


def phash(image):
    gray = cv2.cvtColor(
        cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    value = cv2.dct(gray.astype(np.float32))[:8, :8].reshape(-1)
    return np.packbits(value > np.median(value[1:]))


def original_feature(item):
    image = cv2.imread(item["path"])
    if image is None:
        raise ValueError("Missing training source: " + item["path"])
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return phash(image), small


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-directory", default="models/active")
    args = parser.parse_args()
    folder = root / "data/external/urfall_validation"
    manifest = json.loads((folder / "manifest.json").read_text())
    for item in manifest["files"]:
        if hashlib.sha256((folder / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source hash mismatch")
    cfg = json.loads((root / "sw/config_home_video12.json").read_text())
    modeldir = root / args.model_directory
    model = FallLiteDetector(
        dict(
            model=str(modeldir / "fall_lite.onnx"),
            metadata=str(modeldir / "fall_lite.json"),
            engine="opencv",
            threads=1,
            score_threshold=cfg["fall_lite"]["score_threshold"],
        )
    )
    split = json.loads((root / "models/fall_lite_v6_varied/split.json").read_text())
    # 逐张核验记录的原始及公共训练图，不仅比较文件名。
    originals = [v for v in split["items"] if v["split"] == "train"]
    with ThreadPoolExecutor(6) as pool:
        features = list(pool.map(original_feature, originals))
    hashes = np.stack([v[0] for v in features])
    lookup = np.array([bin(v).count("1") for v in range(256)], np.uint8)
    reports = []
    overlaps = []
    contacts = []
    for name in manifest["sequences"]:
        cap = cv2.VideoCapture(str(folder / (name + "-cam0.mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count < 1:
            raise ValueError("Unreadable continuous video: " + name)
        # 每 0.25 秒固定一个得分，节拍扫描只消费新结果。
        # 重复结果不作为新鲜时序证据。
        samples = []
        next_time = 0.0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = index / fps
            index += 1
            if stamp + 1e-6 < next_time:
                continue
            next_time += 0.25
            rgb = frame[:, frame.shape[1] // 2 :]
            # 模拟相机的 720p RGB565 及半尺寸预处理。
            # 保持原图宽高比。
            height, width = rgb.shape[:2]
            scale = min(1280 / width, 720 / height)
            image = np.zeros((720, 1280, 3), np.uint8)
            w, h = round(width * scale), round(height * scale)
            x, y = (1280 - w) // 2, (720 - h) // 2
            image[y : y + h, x : x + w] = cv2.resize(rgb, (w, h))
            raw = (
                ((image[:, :, 2].astype(np.uint16) >> 3) << 11)
                | ((image[:, :, 1].astype(np.uint16) >> 2) << 5)
                | (image[:, :, 0].astype(np.uint16) >> 3)
            )
            observation = model.process(raw, stamp)
            samples.append(dict(t=stamp, score=observation["score"], bbox=observation["bbox"]))
            # 感知近似匹配仅标记相关帧，留待复核。
            feature = phash(rgb)
            distance = lookup[np.bitwise_xor(hashes, feature)].sum(1)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= 4:
                small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
                mae = float(np.abs(small.astype(np.float32) - features[nearest][1]).mean())
                if mae < 12:
                    overlaps.append(
                        dict(
                            sequence=name,
                            time_s=stamp,
                            training_path=originals[nearest]["path"],
                            distance=int(distance[nearest]),
                            mae=mae,
                        )
                    )
            if len(contacts) < 100 and abs(stamp - round(stamp)) < 0.02:
                thumb = cv2.resize(rgb, (256, 192))
                cv2.putText(
                    thumb,
                    "%s %.1fs" % (name, stamp),
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
                contacts.append(thumb)
        cap.release()
        sweeps = []
        for cadence in (0.7, 1.0, 2.0):
            detector = FallDetector(cfg["decision"])
            next_tick = 0.0
            alarms = []
            states = []
            for sample in samples:
                if sample["t"] + 1e-6 < next_tick:
                    continue
                next_tick = sample["t"] + 1 / cadence
                with patch("fall_detect.time.monotonic", return_value=sample["t"]):
                    result = detector.step(
                        sample["score"] >= cfg["decision"]["score_th"],
                        sample["score"],
                        sample["bbox"],
                    )
                if result["state_name"] == "ALARM" and (not states or states[-1] != "ALARM"):
                    alarms.append(sample["t"])
                states.append(result["state_name"])
            sweeps.append(
                dict(requested_inference_hz=cadence, observations=len(states), alarm_times_s=alarms)
            )
        reports.append(
            dict(
                sequence=name,
                fall_event=name.startswith("fall"),
                duration_s=count / fps,
                source_fps=fps,
                frames=count,
                samples=samples,
                cadences=sweeps,
            )
        )
        print(name, "processed", len(samples), flush=True)
    summaries = []
    for slot, hz in enumerate((0.7, 1.0, 2.0)):
        positive = [v for v in reports if v["fall_event"]]
        negative = [v for v in reports if not v["fall_event"]]
        tp = sum(bool(v["cadences"][slot]["alarm_times_s"]) for v in positive)
        false = sum(len(v["cadences"][slot]["alarm_times_s"]) for v in negative)
        seconds = sum(v["duration_s"] for v in negative)
        summaries.append(
            dict(
                requested_inference_hz=hz,
                fall_sequences_detected=tp,
                fall_sequences=len(positive),
                false_alarm_events=false,
                negative_exposure_s=seconds,
                observed_false_alarms_per_hour=false * 3600 / seconds,
                caveat="Short pilot exposure; zero events is not proof of a low household false-alarm rate",
            )
        )
    report = dict(
        scope="PC continuous public-video regression with RGB565 preprocessing; no camera/HDMI/network latency measurement",
        model_sha256=model.model_sha256,
        source=manifest["source"],
        summary=summaries,
        sequences=reports,
        possible_training_overlap=overlaps,
        independence=(
            "No flagged near duplicates in audited training frames"
            if not overlaps
            else "Possible overlap: exclude affected sequences from independent acceptance"
        ),
        latency_status="No manually verified video onset labels yet; do not report onset-to-alarm or recovery latency from depth CSV frame numbers",
        promotion="Current v6 unchanged; pilot results alone do not justify replacement or household acceptance",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if contacts:
        columns = 5
        black = np.zeros_like(contacts[0])
        contacts += [black] * ((-len(contacts)) % columns)
        cv2.imwrite(
            str(output.with_suffix(".jpg")),
            np.concatenate(
                [
                    np.concatenate(contacts[i : i + columns], 1)
                    for i in range(0, len(contacts), columns)
                ],
                0,
            ),
        )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
