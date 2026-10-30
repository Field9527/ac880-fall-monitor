"""限量下载 COCO train2017 并保留类别及许可；人体框不提供姿态真值，倒地监督保持未知。"""

import cache_env
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile
import requests
from PIL import Image


BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
CLASSES = (
    "person",
    "cell phone",
    "chair",
    "couch",
    "bed",
    "bottle",
    "cup",
    "book",
    "laptop",
    "tv",
    "cat",
    "dog",
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            with requests.get(
                url,
                headers={"Range": "bytes=%d-" % offset} if offset else {},
                stream=True,
                timeout=(15, 90),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(
                    "bytes %d-" % offset
                ):
                    raise ValueError("Unexpected HTTP range")
                with partial.open("ab" if append else "wb") as output:
                    for block in response.iter_content(256 * 1024):
                        if block:
                            output.write(block)
            partial.replace(path)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--images", type=int, default=600)
    a = p.parse_args()
    if not 12 <= a.images <= 2000:
        raise ValueError("Pilot must remain bounded to 12..2000 images")
    a.out.mkdir(parents=True, exist_ok=True)
    archive = Path("E:/Zynq_cache/datasets/coco/annotations_trainval2017.zip")
    url = BUCKET + "/annotations/annotations_trainval2017.zip"
    print("Fetching official COCO train annotations to E: cache", flush=True)
    download(url, archive)
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("annotations/instances_train2017.json") as source:
            data = json.load(source)
    categories = {v["name"]: v["id"] for v in data["categories"]}
    wanted = {categories[name] for name in CLASSES}
    candidates = {c: set() for c in wanted}
    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
        if ann["category_id"] in wanted and not ann["iscrowd"]:
            candidates[ann["category_id"]].add(ann["image_id"])
    pools = {}
    for cat, ids in candidates.items():
        values = sorted(ids)
        random.Random(20261028 + cat).shuffle(values)
        pools[cat] = iter(values)
    selected = set()
    while len(selected) < a.images:
        before = len(selected)
        for cat in sorted(wanted):
            value = next(pools[cat], None)
            if value is not None:
                selected.add(value)
            if len(selected) >= a.images:
                break
        if len(selected) == before:
            raise ValueError("Insufficient source images")
    images = {v["id"]: v for v in data["images"] if v["id"] in selected}
    licenses = {v["id"]: v for v in data["licenses"]}

    def fetch(image_id):
        image = images[image_id]
        path = a.out / "images" / image["file_name"]
        source = BUCKET + "/train2017/" + image["file_name"]
        download(source, path)
        with Image.open(path) as decoded:
            if decoded.size != (image["width"], image["height"]):
                raise ValueError("COCO dimensions mismatch")
            decoded.verify()
        return dict(
            id=image_id,
            path="images/" + image["file_name"],
            source=source,
            sha256=digest(path),
            width=image["width"],
            height=image["height"],
            license=licenses[image["license"]],
            original_url=image.get("flickr_url"),
            annotations=by_image[image_id],
            split=(
                "development"
                if int(hashlib.sha256(str(image_id).encode()).hexdigest()[:8], 16) % 5 == 0
                else "train"
            ),
            person_posture_supervision="unknown",
            field_acceptance=False,
        )

    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(fetch, k): k for k in sorted(selected)}
        for future in as_completed(tasks):
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append(dict(image_id=tasks[future], error=str(exc)))
            if (len(records) + len(errors)) % 50 == 0:
                print("images", len(records), "errors", len(errors), flush=True)
    records.sort(key=lambda v: v["id"])
    counts = {
        name: sum(
            any(v["category_id"] == categories[name] and not v["iscrowd"] for v in r["annotations"])
            for r in records
        )
        for name in CLASSES
    }
    manifest = dict(
        source="COCO train2017",
        annotation_url=url,
        annotation_archive_sha256=digest(archive),
        selection_seed=20261028,
        classes=list(CLASSES),
        categories=data["categories"],
        items=records,
        errors=errors,
        counts=counts,
        complete=len(records) == a.images,
        limitations=[
            "COCO person is not a normal/fall posture label",
            "Unknown or unselected object boxes and crowds must not become negative fall labels",
            "Image-ID split is development only; scene and near-duplicate independence not established",
        ],
        terms_url="https://cocodataset.org/#termsofuse",
    )
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(dict(images=len(records), errors=len(errors), counts=counts)), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
