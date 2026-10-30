"""下载固定版本官方 NanoDet 探针，记录 Git 对象和 SHA256。"""

import cache_env
import argparse
import hashlib
import json
from pathlib import Path
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--revision", default="321276bf9fd465896dd685d64ab0bb08dd68179b")
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = "https://api.github.com/repos/nihui/ncnn-android-nanodet"
    with requests.Session() as client:
        response = client.get(source + "/commits/" + args.revision, timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
        records = []
        for path in (
            "app/src/main/assets/nanodet-m.param",
            "app/src/main/assets/nanodet-m.bin",
            "README.md",
            "app/src/main/jni/nanodet.cpp",
            "app/src/main/jni/nanodetncnn.cpp",
        ):
            response = client.get(
                source + "/contents/" + path, params={"ref": revision}, timeout=30
            )
            response.raise_for_status()
            info = response.json()
            response = client.get(info["download_url"], timeout=90)
            response.raise_for_status()
            content = response.content
            gitsha = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if gitsha != info["sha"] or len(content) != info["size"]:
                raise ValueError("Official Git object verification failed: " + path)
            name = Path(path).name
            dest = root / name
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError("Refusing to overwrite a different probe: " + name)
            dest.write_bytes(content)
            records.append(
                dict(
                    path=path,
                    file=name,
                    bytes=len(content),
                    git_blob_sha=gitsha,
                    sha256=hashlib.sha256(content).hexdigest(),
                    url=info["download_url"],
                )
            )
    manifest = dict(
        repository="nihui/ncnn-android-nanodet",
        revision=revision,
        files=records,
        purpose="CPU-only feasibility probe, not a trained fall monitor",
        production_deployed=False,
    )
    (root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(dict(directory=str(root), revision=revision, files=len(records))))


if __name__ == "__main__":
    main()
