"""核对数值一致性、保留集预测和可视化误差。"""

import cache_env
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import FallLiteDetector, letterbox, flatten_heads
from fall_lite_model import FallLite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    ck = torch.load(a.model_dir / "best.pt", map_location="cpu")
    model = FallLite(depth=ck.get("depth", 13))
    model.load_state_dict(ck["model"])
    model.eval()
    detector = FallLiteDetector(
        dict(model=str(a.model_dir / "fall_lite.onnx"), score_threshold=0.7)
    )
    x = np.random.RandomState(16).normal(size=(1, 3, ck["height"], ck["width"])).astype(np.float32)
    with torch.no_grad():
        reference = [t.numpy() for t in model(torch.from_numpy(x))]
    detector.net.setInput(x)
    actual = detector.net.forward(["head8", "head16", "head32"])
    errors = [float(np.abs(g - y).max()) for g, y in zip(reference, actual)]
    for g, y in zip(reference, actual):
        np.testing.assert_allclose(g, y, rtol=0.002, atol=0.002)
    np.savez(
        a.out / "golden.npz", image=x, head8=reference[0], head16=reference[1], head32=reference[2]
    )
    print("PC_EQUIVALENCE", errors, flush=True)
    items = json.loads((a.model_dir / "split.json").read_text())["items"]
    records = []
    false_positive = []
    true_positive = []
    missed = []
    for item in items:
        if item["split"] != "val":
            continue
        rgb = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
        detections = detector.predict_rgb(rgb)
        record = dict(
            path=item["path"], gt=item["boxes"], has_xml=item["has_xml"], detections=detections
        )
        records.append(record)
        if not item["boxes"] and detections:
            false_positive.append(record)
        elif item["boxes"] and detections:
            true_positive.append(record)
        elif item["boxes"]:
            missed.append(record)
    false_positive.sort(key=lambda r: r["detections"][0]["score"], reverse=True)
    (a.out / "holdout_predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    for name, examples in [
        ("false_positive_labels", false_positive),
        ("positive_predictions", true_positive),
        ("missed_labels", missed),
    ]:
        sheet = Image.new("RGB", (3 * 320, 3 * 210), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(examples[:9]):
            rgb = np.array(Image.open(r["path"]).convert("RGB"))
            boxed, (sx, sy, ox, oy) = letterbox(rgb, 180, 320)
            tile = Image.fromarray(boxed)
            draw = ImageDraw.Draw(tile)
            for box in r["gt"]:
                x1, y1, x2, y2 = box
                draw.rectangle(
                    (x1 * sx + ox, y1 * sy + oy, x2 * sx + ox, y2 * sy + oy),
                    outline=(0, 255, 0),
                    width=2,
                )
            for det in r["detections"]:
                x, y, w, h = det["bbox"]
                draw.rectangle(
                    (x * sx + ox, y * sy + oy, (x + w) * sx + ox, (y + h) * sy + oy),
                    outline=(255, 0, 0),
                    width=2,
                )
            col, row = i % 3, i // 3
            sheet.paste(tile, (col * 320, row * 210))
            score = max([v["score"] for v in r["detections"]], default=0)
            d.text(
                (col * 320 + 3, row * 210 + 182),
                Path(r["path"]).name + " %.3f xml=%s" % (score, r["has_xml"]),
                fill="white",
            )
        sheet.save(a.out / (name + ".jpg"))
    report = dict(
        pc_max_abs_error=errors,
        validation_images=len(records),
        threshold=0.7,
        negative_image_predictions=len(false_positive),
        visual_audit="pending human inspection",
        note="Green: source annotation; red: model detection. Positive_predictions is not an IoU metric.",
    )
    (a.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
