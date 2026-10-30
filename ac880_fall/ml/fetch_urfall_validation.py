"""获取作者提供的连续 RGB 序列，仅用于评估。"""

import cache_env
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import requests
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/external/urfall_validation"
BASE = "https://fenix.ur.edu.pl/~mkepski/ds/"
SEQUENCES = ["fall-%02d" % i for i in (1, 2, 3)] + ["adl-%02d" % i for i in (1, 2, 3)]


def fetch(name):
    destination = DEST / name
    if not destination.exists():
        temporary = Path("E:/Zynq_cache/temp") / ("urfall-" + name + ".part")
        before = time.monotonic()
        with requests.get(BASE + "data/" + name, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(65536):
                    if time.monotonic() - before > 180:
                        raise TimeoutError(name)
                    output.write(chunk)
        temporary.replace(destination)
    if name.endswith(".zip"):
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
            if not all(
                not Path(v).is_absolute() and ".." not in Path(v).parts for v in archive.namelist()
            ):
                raise ValueError("Unsafe archive path")
    return dict(
        file=name,
        url=BASE + "data/" + name,
        bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    )


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    response = requests.get(BASE + "uf.html", timeout=30)
    response.raise_for_status()
    (DEST / "source.html").write_text(response.text, encoding="utf8")
    # 从作者页面解析链接，不猜测姿态 CSV 地址。
    labels = []
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        links = re.findall(r'href\s*=\s*["\']?([^\s"\'>]+)', response.text, re.I)
        link = next(v for v in links if v.endswith(name))
        url = requests.compat.urljoin(BASE + "uf.html", link)
        result = requests.get(url, timeout=30)
        result.raise_for_status()
        (DEST / name).write_bytes(result.content)
        labels.append(dict(file=name, url=url, sha256=hashlib.sha256(result.content).hexdigest()))
    names = [v + "-cam0.mp4" for v in SEQUENCES] + [v + "-data.csv" for v in SEQUENCES]
    with ThreadPoolExecutor(3) as pool:
        files = list(pool.map(fetch, names))
    manifest = dict(
        source=BASE + "uf.html",
        author="Michal Kepski / University of Rzeszow",
        license="CC BY-NC-SA 4.0; author academic-use terms in source.html",
        citation="Kwolek and Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, 2014",
        purpose="Frozen continuous-sequence validation; never added to training",
        sequences=SEQUENCES,
        video_layout="Author MP4: depth left, RGB right; evaluate right 320x240 half only",
        timing="MP4 timestamps only; depth CSV labels must not be assumed to align exactly with encoded video frames",
        overlap_audit_required=True,
        files=files + labels,
    )
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(dict(sequences=len(SEQUENCES), bytes=sum(v["bytes"] for v in files))), flush=True
    )


if __name__ == "__main__":
    main()
