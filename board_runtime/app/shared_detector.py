"""共享骨干输出摔倒候选及 ROI 特征，并进行限量复核。"""

import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
from fall_lite import FallLiteDetector, anchors_for_shape, letterbox, flatten_heads, decode
from semantic_features import (
    RegionVerifier,
    rotate_xyxy,
    OUTPUT_NAMES,
    NANO_SHARED_OUTPUT_NAMES,
    FALL_REGION_OUTPUT_NAMES,
)
from semantic_decode import decode_objects


class SemanticOptions:
    def __init__(self, mode="live", rotation=0):
        self.lock = threading.Lock()
        self.value = dict(
            mode=mode,
            rotation=rotation,
            show_objects=False,
            gesture_enabled=True,
            gesture_generation=0,
            generation=0,
            recognition_mode="both",
        )
        self.switch_status = None

    @property
    def configuration(self):
        with self.lock:
            return dict(self.value)

    def validate(self, value, current=None):
        if not isinstance(value, dict) or set(value) - {
            "mode",
            "rotation",
            "show_objects",
            "gesture_enabled",
        }:
            raise ValueError("Unknown recognition setting")
        proposed = dict(current or self.configuration)
        proposed.update(value)
        if proposed["mode"] not in ("live", "phone_replay"):
            raise ValueError("Unknown monitoring mode")
        if type(proposed["rotation"]) is not int or proposed["rotation"] not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0/90/180/270")
        if proposed["mode"] == "live":
            proposed["rotation"] = 0
        if type(proposed["show_objects"]) is not bool:
            raise ValueError("Object visibility must be boolean")
        if type(proposed["gesture_enabled"]) is not bool:
            raise ValueError("Gesture control must be boolean")
        return proposed

    def configure(self, value):
        with self.lock:
            proposed = self.validate(
                {
                    k: v
                    for k, v in value.items()
                    if k not in ("generation", "gesture_generation", "recognition_mode")
                },
                self.value,
            )
            generation = self.value["generation"] + int(
                any(proposed[k] != self.value[k] for k in ("mode", "rotation"))
            )
            epoch = self.value["gesture_generation"] + int(
                proposed["gesture_enabled"] != self.value["gesture_enabled"]
            )
            self.value = dict(proposed, generation=generation, gesture_generation=epoch)

    def select_recognition(self, mode, status):
        if mode not in ("fall", "gesture", "paused"):
            raise ValueError("无效的拨码识别模式")
        with self.lock:
            if mode != self.value["recognition_mode"]:
                self.value["recognition_mode"] = mode
                self.value["generation"] += 1
            self.switch_status = dict(status)

    def snapshot(self):
        with self.lock:
            return dict(available=True, recognition_switch=self.switch_status, **self.value)


class SharedFallDetector(FallLiteDetector):
    def __init__(self, cfg):
        # 在线程启动前导入 OpenCV，避免绘制线程等待导入锁。
        import cv2

        cv2.setNumThreads(1)
        path = Path(cfg["metadata"])
        self.meta = json.loads(path.read_text(encoding="utf8"))
        root = path.parent
        if not self.meta.get("trained") or (
            not self.meta.get("production_qualified") and not cfg.get("allow_experimental")
        ):
            raise ValueError("Shared model requires validation or an explicit experimental trial")
        self.nano = self.meta.get("architecture") == "nanodet_shared"
        self.roi_only = self.meta.get("architecture") == "v6_roi"
        self.feature_start = 9 if self.nano else 3 if self.roi_only else 6
        self.output_names = (
            NANO_SHARED_OUTPUT_NAMES
            if self.nano
            else FALL_REGION_OUTPUT_NAMES if self.roi_only else OUTPUT_NAMES
        )
        if self.meta.get("descriptor_abi") != (
            "pyramid96_adaptive3_global_v1" if self.nano else "pyramid64_adaptive3_global_v1"
        ):
            raise ValueError("Unsupported shared feature ABI")
        self.width, self.height = 320, 192
        self.threshold = float(
            cfg.get("score_threshold", self.meta.get("recommended_threshold", 0.7))
        )
        self.anchors = anchors_for_shape()
        self.training_run = self.meta["training_run"]
        self.model_sha256 = self.meta["onnx_sha256"]
        self.engine = cfg.get("engine", "semantic")
        self.semantic_options = SemanticOptions(
            cfg.get("monitor_mode", "live"), cfg.get("replay_rotation", 0)
        )
        self.object_threshold = float(cfg.get("object_threshold", 0.5))
        self.max_verify = int(cfg.get("max_verify", 8))
        if (
            not 1 <= self.max_verify <= 16
            or not 0 < self.threshold < 1
            or not 0 < self.object_threshold < 1
        ):
            raise ValueError("Invalid shared detector limits")
        self.verifier = RegionVerifier(root / "region_verifier.npz")
        if (
            hashlib.sha256((root / "region_verifier.npz").read_bytes()).hexdigest()
            != self.meta["verifier_sha256"]
        ):
            raise ValueError("Verifier SHA mismatch")
        self.reference_verifier = None
        if self.meta.get("reference_verifier"):
            reference = self.meta["reference_verifier"]
            name = Path(reference["file"])
            if name.name != str(name) or name.is_absolute():
                raise ValueError("Invalid reference verifier path")
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("Reference verifier SHA mismatch")
            self.reference_verifier = RegionVerifier(root / name)
        if self.engine == "opencv_shared":
            import cv2

            if (
                hashlib.sha256((root / "shared_semantic.onnx").read_bytes()).hexdigest()
                != self.meta["onnx_sha256"]
            ):
                raise ValueError("ONNX SHA mismatch")
            self.net = cv2.dnn.readNetFromONNX(str(root / "shared_semantic.onnx"))
        elif self.engine == "semantic":
            from ncnn_backend import NCNNNet

            runtime = self.meta["ncnn"]
            for key in ("param", "weights"):
                if (
                    hashlib.sha256((root / runtime[key]).read_bytes()).hexdigest()
                    != runtime[key + "_sha256"]
                ):
                    raise ValueError("NCNN artifact SHA mismatch")
            if (
                hashlib.sha256(Path(cfg["runtime_library"]).read_bytes()).hexdigest()
                != runtime["library_sha256"]
            ):
                raise ValueError("NCNN runtime SHA mismatch")
            layouts = [3] * 3 + [2] * 6 + [3] * 3 if self.nano else [3] * len(self.output_names)
            if runtime.get("output_layouts", layouts) != layouts:
                raise ValueError("NCNN output ABI mismatch")
            self.net = NCNNNet(
                cfg["runtime_library"],
                root / runtime["param"],
                root / runtime["weights"],
                threads=int(cfg.get("threads", 1)),
                output_names=runtime["output_names"],
                output_layouts=layouts,
            )
        else:
            raise ValueError("Unsupported shared inference engine")
        self.diagnostic_dir = None
        self.diagnostic_rows = []
        self.diagnostic_last = -float("inf")
        self.diagnostic_context = cfg.get("test_context", "live_scene")
        if cfg.get("diagnostic_raw_dir"):
            self.diagnostic_dir = Path(cfg["diagnostic_raw_dir"]).resolve()
            self.diagnostic_dir.mkdir(parents=True, exist_ok=False)
        self.last_semantic = {}
        self.scene_probe = None
        self.scene_objects = []
        self.scene_sampled_at = None
        self.scene_generation = -1
        if self.roi_only and cfg.get("scene_model"):
            from semantic_probe import NanoDetProbe

            self.scene_probe = NanoDetProbe(
                cfg["scene_model"], cfg["runtime_library"], threads=int(cfg.get("threads", 1))
            )
        self.scene_interval_s = max(2.0, float(cfg.get("scene_interval_s", 4.0)))
        self.hands = None
        self.hand = None
        self.hand_sampled_at = None
        self.hand_sequence = 0
        self.hand_epoch = None
        self.independent_gestures = True
        self.hand_interval_s = max(0.25, float(cfg.get("gesture_interval_s", 0.5)))
        if cfg.get("gesture_model"):
            from hand_models import HandModels

            self.hands = HandModels(
                cfg["gesture_model"], cfg["runtime_library"], threads=int(cfg.get("threads", 1))
            )
        # 加载 PL 和视频前完成 CPU 内核初始化。
        zero = np.zeros((1, 3, 192, 320), np.float32)
        if self.engine == "opencv_shared":
            self.net.setInput(zero)
            warm = self.net.forward(list(self.output_names))
        else:
            warm = self.net.forward(zero)
        self.verifier.predict_many(warm[self.feature_start :], [[0, 0, 320, 192]])
        if self.scene_probe is not None:
            self.scene_probe.detect(np.zeros((192, 320, 3), np.uint8))
        if self.hands is not None:
            self.hands.warmup()

    def predict_rgb(self, rgb):
        options = self.semantic_options.configuration
        rotation = options["rotation"] if options["mode"] == "phone_replay" else 0
        original_height, original_width = rgb.shape[:2]
        if rotation:
            rgb = np.ascontiguousarray(np.rot90(rgb, rotation // 90))
        h, w = rgb.shape[:2]
        boxed, (sx, sy, ox, oy) = letterbox(
            rgb[:, :, ::-1] if self.nano else rgb, fill=0 if self.nano else 114
        )
        blob = boxed.transpose(2, 0, 1)[None].astype(np.float32)
        if self.nano:
            blob = (
                blob - np.array([103.53, 116.28, 123.675], np.float32)[None, :, None, None]
            ) / np.array([57.375, 57.12, 58.395], np.float32)[None, :, None, None]
        else:
            blob = (
                blob / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
            ) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        blob = np.ascontiguousarray(blob)
        self.runtime_stage = "fall_network"
        start = time.perf_counter()
        if self.engine == "opencv_shared":
            self.net.setInput(blob)
            outputs = self.net.forward(list(self.output_names))
        else:
            outputs = self.net.forward(blob)
        if len(outputs) != len(self.output_names):
            raise ValueError("Shared output count changed")
        network_ms = (time.perf_counter() - start) * 1000
        raw = decode(flatten_heads(outputs[:3]), self.anchors, self.threshold, top_k=16)

        def display_box(box):
            b = (np.asarray(box) - [ox, oy, ox, oy]) / [sx, sy, sx, sy]
            b = b.clip([0, 0, 0, 0], [w, h, w, h])
            if rotation:
                b = np.asarray(rotate_xyxy(b, w, h, -rotation // 90))
            b = b.clip(
                [0, 0, 0, 0], [original_width, original_height, original_width, original_height]
            )
            return [
                int(b[0]),
                int(b[1]),
                max(0, int(b[2]) - int(b[0])),
                max(0, int(b[3]) - int(b[1])),
            ]

        self.runtime_stage = "region_verification"
        detections = []
        reviewed = []
        verify_start = time.perf_counter()
        boxes = []
        for item in raw:
            x, y, bw, bh = item["box"]
            boxes.append(np.array([x, y, x + bw, y + bh]).clip([0, 0, 0, 0], [320, 192, 320, 192]))
        valid = [
            i for i, box in enumerate(boxes[: self.max_verify]) if np.all(box[2:] - box[:2] >= 1)
        ]
        values = self.verifier.predict_many(
            outputs[self.feature_start :], [boxes[i] for i in valid]
        )
        probability_map = dict(zip(valid, values))
        reference_map = {}
        if self.reference_verifier is not None:
            accepted = [
                i
                for i in valid
                if probability_map[i]["fallen"] > self.meta["fall_probability_ceiling"]
            ]
            references = self.reference_verifier.predict_many(
                outputs[self.feature_start :], [boxes[i] for i in accepted]
            )
            reference_map = dict(zip(accepted, references))
        for index, item in enumerate(raw):
            box = boxes[index]
            if np.any(box[2:] - box[:2] < 1):
                continue
            probabilities = probability_map.get(index)
            reference = reference_map.get(index)
            reference_rejected = (
                reference is not None
                and reference["fallen"] <= self.meta["fall_probability_ceiling"]
            )
            rejected = bool(
                probabilities is not None
                and probabilities["fallen"] <= self.meta["fall_probability_ceiling"]
                or reference_rejected
            )
            deciding = reference if reference_rejected else probabilities
            reason = max(deciding, key=deciding.get) if deciding else "verification_budget_exceeded"
            entry = dict(
                bbox=display_box(box),
                score=item["score"],
                region=reason,
                region_probabilities=probabilities,
                reviewed_nonfall=rejected,
                reference_probabilities=reference,
                test_context=(
                    "phone_screen_replay" if options["mode"] == "phone_replay" else "live_scene"
                ),
            )
            if entry["bbox"][2] < 2 or entry["bbox"][3] < 2:
                continue
            (reviewed if rejected else detections).append(entry)
        verification_ms = (time.perf_counter() - verify_start) * 1000
        objects = []
        if self.nano:
            from semantic_probe import decode as decode_nano, OBJECT_NAMES

            proposals = decode_nano(
                outputs[3:9], 192, 320, (1, 1, 0, 0), (192, 320), threshold=self.object_threshold
            )
            proposals = [v for v in proposals if v["label"] in OBJECT_NAMES][:20]
        elif not self.roi_only:
            proposals = decode_objects(outputs[3:6], self.anchors, self.object_threshold, limit=20)
        else:
            proposals = []
        for obj in proposals:
            objects.append(dict(name=obj["name"], score=obj["score"], bbox=display_box(obj["box"])))
        object_age = None
        scene_ms = 0.0
        if self.scene_probe is not None:
            now = time.monotonic()
            if self.scene_generation != options["generation"]:
                self.scene_objects = []
                self.scene_sampled_at = None
                self.scene_generation = options["generation"]
            # 可选场景标签在同一推理任务内串行计算。
            # 摔倒证据优先于场景标签。
            if (
                options["show_objects"]
                and not detections
                and (
                    self.scene_sampled_at is None
                    or now - self.scene_sampled_at >= self.scene_interval_s
                )
            ):
                self.runtime_stage = "scene_labels"
                start = time.perf_counter()
                found = self.scene_probe.detect(rgb[:, :, ::-1].copy())["objects"]
                self.scene_sampled_at = now
                self.scene_objects = []
                from semantic_probe import OBJECT_NAMES

                for obj in found:
                    if obj["label"] not in OBJECT_NAMES or obj["score"] < self.object_threshold:
                        continue
                    b = obj["box"]
                    if rotation:
                        b = rotate_xyxy(b, w, h, -rotation // 90)
                    b = np.asarray(b).clip(
                        [0, 0, 0, 0],
                        [original_width, original_height, original_width, original_height],
                    )
                    self.scene_objects.append(
                        dict(
                            name=obj["name"],
                            score=obj["score"],
                            bbox=[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])],
                        )
                    )
                scene_ms = (time.perf_counter() - start) * 1000
            object_age = (
                time.monotonic() - self.scene_sampled_at
                if self.scene_sampled_at is not None
                else None
            )
            objects = (
                self.scene_objects
                if object_age is not None and object_age < self.scene_interval_s + 3.0
                else []
            )
        self.last_semantic = dict(
            objects=objects,
            reviewed_candidates=reviewed,
            coverage_limited=len(raw) > self.max_verify,
            semantic_generation=options["generation"],
            show_objects=options["show_objects"],
            hand=self.hand,
            hand_sequence=self.hand_sequence,
            hand_sampled_at=self.hand_sampled_at,
            gesture_generation=options["gesture_generation"],
            monitor_mode=options["mode"],
            replay_rotation=rotation,
            test_context=(
                "phone_screen_replay" if options["mode"] == "phone_replay" else "live_scene"
            ),
            object_age_s=round(object_age, 2) if object_age is not None else None,
            object_sampled_at=self.scene_sampled_at,
            scene_ms=round(scene_ms, 2),
            network_ms=round(network_ms, 2),
            verification_ms=round(verification_ms, 2),
        )
        self.diagnostic_context = self.last_semantic["test_context"]
        self.runtime_stage = "complete"
        return detections

    def process(self, frame, now=None):
        self.frame_time = now if now is not None else time.monotonic()
        observation = super().process(frame, now)
        observation.update(source_monotonic=self.frame_time, fall_fresh=True)
        info = dict(self.last_semantic)
        sx = frame.shape[1] / frame[::2, ::2].shape[1]
        sy = frame.shape[0] / frame[::2, ::2].shape[0]
        for key in ("objects", "reviewed_candidates"):
            info[key] = [
                dict(v, bbox=[round(v["bbox"][i] * (sx if i % 2 == 0 else sy)) for i in range(4)])
                for v in info[key]
            ]
        if not observation["bbox"] and info["reviewed_candidates"]:
            observation["reason"] = "region_reviewed_nonfall"
        observation.update(info)
        return observation

    def gestures_enabled(self):
        options = self.semantic_options.configuration
        return (
            self.hands is not None
            and options["gesture_enabled"]
            and options["mode"] == "live"
            and options["recognition_mode"] in ("both", "gesture")
        )

    def process_hand(self, frame, now=None):
        # 仅独立手势任务访问关键点网络和跟踪器。
        from fall_lite import rgb565

        options = self.semantic_options.configuration
        epoch = (options["generation"], options["gesture_generation"])
        if epoch != self.hand_epoch:
            self.hand_epoch = epoch
            self.hands.reset()
        started = time.monotonic()
        small = frame[::2, ::2]
        hand = self.hands.detect(rgb565(small))
        if hand:
            sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
            hand = dict(
                hand,
                bbox=[round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(hand["bbox"])],
                landmarks=[[round(x * sx, 1), round(y * sy, 1)] for x, y in hand["landmarks"]],
            )
        self.hand_sequence += 1
        return dict(
            hand=hand,
            hand_sequence=self.hand_sequence,
            hand_sampled_at=now if now is not None else started,
            hand_status=getattr(self.hands, "last_status", None),
            gesture_ms=round((time.monotonic() - started) * 1000, 1),
            gesture_generation=epoch[1],
            hand_semantic_generation=epoch[0],
        )

    def attach_pl(self, overlay, timeout_s=5.0):
        return None

    def reset(self):
        self.last_semantic = {}
        self.scene_objects = []
        self.scene_sampled_at = None
        self.hand = None
        self.hand_sampled_at = None

    def close(self):
        self.net.close() if hasattr(self.net, "close") else None
        if self.scene_probe is not None:
            self.scene_probe.close()
        if self.hands is not None:
            self.hands.close()
