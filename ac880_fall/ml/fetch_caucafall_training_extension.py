"""仅增加训练人员的时间采样，保持原验证与测试划分不变。"""

import cache_env
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from fetch_caucafall import ROOT, OUTPUT, EVIDENCE, CACHE, TEMP, listing, download


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    folders = json.loads((EVIDENCE / "caucafall_folders.json").read_text())
    subjects = {
        v["id"]: v["name"]
        for v in folders
        if v["name"].startswith("Subject.") and int(v["name"].split(".")[-1]) <= 7
    }
    activities = [
        dict(v, subject=subjects[v["parent_id"]]) for v in folders if v.get("parent_id") in subjects
    ]
    original = json.loads((OUTPUT / "source_manifest.json").read_text())
    known = {v["id"] for v in original["files"]}
    entries = []
    for folder in activities:
        _, files = listing(folder)
        by_name = {v["filename"]: v for v in files}
        images = sorted(v["filename"] for v in files if v["filename"].lower().endswith(".png"))
        fractions = (0.02, 0.14, 0.20, 0.26, 0.32, 0.48, 0.54, 0.60, 0.66, 0.80, 0.86, 0.92)
        for index in sorted(set(round((len(images) - 1) * f) for f in fractions)):
            image = images[index]
            for name in (image, Path(image).stem + ".txt"):
                file = by_name[name]
                if file["id"] in known:
                    continue
                known.add(file["id"])
                details = file["content_details"]
                entries.append(
                    dict(
                        id=file["id"],
                        filename=name,
                        subject=folder["subject"],
                        activity=folder["name"],
                        url=details["download_url"],
                        sha256=details["sha256_hash"],
                        size=file["size"],
                    )
                )
    manifest = {key: value for key, value in original.items() if key != "files"}
    manifest.update(
        files=entries,
        sampling="12 additional temporal positions, training subjects1-7 only",
        split_policy="Training only; validation/test files remain frozen in v5 manifest",
    )
    (OUTPUT / "training_extension_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "TRAIN_EXTENSION",
        len(entries),
        "files",
        round(sum(v["size"] for v in entries) / 1e6, 1),
        "MB",
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
            if number % 100 == 0:
                print("Downloaded", number, "/", len(entries), "failed", len(failures), flush=True)
    (OUTPUT / "training_extension_download.json").write_text(
        json.dumps(dict(files=len(entries), failures=failures, complete=not failures), indent=2)
    )
    if failures:
        raise RuntimeError("Training extension download incomplete; rerun reuses verified files")
    print("TRAIN_EXTENSION_READY", flush=True)


if __name__ == "__main__":
    main()
