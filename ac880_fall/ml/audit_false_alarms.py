"""先验证已知负例，再评估阈值调整。"""

import cache_env
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
import cv2
import numpy as np
from fall_lite import FallLiteDetector

root = Path(__file__).resolve().parents[1]
out = root / "out/false_alarm_speed_20261016"
det = FallLiteDetector(
    dict(model=str(root / "models/active/fall_lite.onnx"), threads=1, score_threshold=0.3)
)
images = {}
for value in (0, 4, 8, 16, 32, 64, 114, 128, 192, 255):
    images["uniform_%03d" % value] = np.full((360, 640, 3), value, np.uint8)
rng = np.random.RandomState(38)
for value in (4, 16, 40, 80):
    images["dark_noise_%03d" % value] = rng.randint(0, value + 1, (360, 640, 3), dtype=np.uint8)
rows = []
for name, img in images.items():
    detections = det.predict_rgb(img)
    rows.append(
        dict(
            name=name,
            max_score=max([d["score"] for d in detections] + [0.0]),
            detections=detections,
        )
    )
(out / "synthetic_before.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
print(json.dumps([dict(name=r["name"], max_score=r["max_score"]) for r in rows], indent=2))
