"""从作者资源获取按人员划分的子集；类别 0 为非倒地、1 为倒地，只下载数据，不执行下载内容。"""

import cache_env
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import threading
import requests

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "out/validation_20261017/online_data"
OUTPUT = ROOT / "data/external/caucafall"
CACHE = Path("E:/Zynq_cache/datasets/caucafall")
TEMP = Path("E:/Zynq_cache/temp")
API = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
THREAD = threading.local()


def session():
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
    return THREAD.session


def listing(folder):
    cache = CACHE / (folder["id"] + ".json")
    if cache.exists():
        return folder, json.loads(cache.read_text())
    files = []
    start = 0
    while True:
        for attempt in range(3):
            try:
                response = session().get(
                    API + "/files",
                    params=dict(
                        folder_id=folder["id"], version=4, **{"$start": start, "$limit": 1000}
                    ),
                    headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
                    timeout=(15, 30),
                )
                response.raise_for_status()
                page = response.json()
                break
            except Exception:
                if attempt == 2:
                    raise
        files.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    cache.write_text(json.dumps(files, indent=2))
    return folder, files


def download(entry):
    destination = OUTPUT / "raw" / entry["subject"] / entry["activity"] / entry["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = entry["sha256"]
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    temporary = TEMP / ("caucafall-" + entry["id"] + ".part")
    for attempt in range(3):
        try:
            started = time.monotonic()
            digest = hashlib.sha256()
            with session().get(entry["url"], stream=True, timeout=(15, 25)) as response:
                response.raise_for_status()
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(65536):
                        if time.monotonic() - started > 120:
                            raise TimeoutError("Dataset file deadline exceeded")
                        f.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("Dataset SHA256 mismatch")
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("Download failed")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    folders = json.loads((EVIDENCE / "caucafall_folders.json").read_text())
    subjects = {v["id"]: v["name"] for v in folders if v["name"].startswith("Subject.")}
    activities = [
        dict(v, subject=subjects[v["parent_id"]]) for v in folders if v.get("parent_id") in subjects
    ]
    entries = []
    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(listing, v) for v in activities]
        for number, future in enumerate(as_completed(futures), 1):
            folder, files = future.result()
            by_name = {v["filename"]: v for v in files}
            images = sorted(v["filename"] for v in files if v["filename"].lower().endswith(".png"))
            if not images:
                raise RuntimeError("No source images: " + str(folder))
            # 每项活动取四个时间点，同一人员全部留在同一划分。
            # 两个相机及正常、倒地姿态不跨划分。
            selected = sorted(
                set(round((len(images) - 1) * fraction) for fraction in (0.08, 0.40, 0.72, 0.96))
            )
            for index in selected:
                image = images[index]
                for name in (image, Path(image).stem + ".txt"):
                    if name not in by_name:
                        raise RuntimeError("Missing source label: " + name)
                    f = by_name[name]
                    details = f["content_details"]
                    entries.append(
                        dict(
                            id=f["id"],
                            filename=name,
                            subject=folder["subject"],
                            activity=folder["name"],
                            url=details["download_url"],
                            sha256=details["sha256_hash"],
                            size=f["size"],
                        )
                    )
            if number % 10 == 0:
                print("Indexed activities %d/%d" % (number, len(activities)), flush=True)
    manifest = dict(
        source="https://data.mendeley.com/datasets/7w7fccy7ky/4",
        doi="10.17632/7w7fccy7ky.4",
        authors="Jose Camilo Eraso, Elena Muñoz, Mariela Muñoz, Jesus Pinto",
        license="CC BY 4.0",
        class_order=["nofall", "fall"],
        class_order_evidence="Author Figure6 content of different txt files",
        split_policy="Subjects 1-7 train, 8 validation, 9-10 final test. No random frame splitting.",
        sampling="Four temporal positions (8%,40%,72%,96%) per complete activity sequence",
        files=entries,
    )
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Downloading %d original files, %.1f MB"
        % (len(entries), sum(v["size"] for v in entries) / 1e6),
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(download, v): v for v in entries}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(file=futures[future], error=str(exc)))
            if number % 40 == 0:
                print(
                    "Downloaded %d/%d, failed=%d" % (number, len(entries), len(failures)),
                    flush=True,
                )
    (OUTPUT / "download_report.json").write_text(
        json.dumps(dict(files=len(entries), failures=failures, complete=not failures), indent=2)
    )
    if failures:
        raise RuntimeError("%d dataset files failed; rerun resumes verified files" % len(failures))
    print("CAUCAFALL_SUBSET_READY", flush=True)


if __name__ == "__main__":
    main()
