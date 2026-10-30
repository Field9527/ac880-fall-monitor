"""从作者 FreiHAND 训练压缩包按范围读取限量 RGB 子集，不下载评测图或执行源代码。"""

import cache_env
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
import requests

URL = "https://lmbweb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
PAGE = "https://lmbweb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html"
ROOT = Path(__file__).resolve().parents[1] / "data/external/freihand_research"


def read_range(start, end):
    if not 0 < end - start + 1 <= 2 * 1024 * 1024:
        raise ValueError("Range exceeds the 2 MiB download bound")
    for attempt in range(3):
        try:
            started = time.monotonic()
            parts = []
            with requests.get(
                URL, headers={"Range": "bytes=%d-%d" % (start, end)}, timeout=20, stream=True
            ) as r:
                r.raise_for_status()
                if r.status_code != 206:
                    raise ValueError("Server did not honor the byte range")
                for part in r.iter_content(65536):
                    parts.append(part)
                    if time.monotonic() - started > 100:
                        raise ValueError("Bounded range timed out")
            data = b"".join(parts)
            if len(data) != end - start + 1:
                raise ValueError("Wrong byte range length")
            return data
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tail = requests.get(URL, headers={"Range": "bytes=-65536"}, timeout=30)
    tail.raise_for_status()
    if tail.status_code != 206:
        raise ValueError("Range support required")
    pos = tail.content.rfind(b"PK\x05\x06")
    fields = struct.unpack_from("<4s4H2LH", tail.content, pos)
    count, size, offset = fields[4:7]

    def index_part(start):
        end = min(offset + size - 1, start + 1024 * 1024 - 1)
        path = ROOT / ("index_%d.bin" % start)
        if not path.exists():
            path.write_bytes(read_range(start, end))
        data = path.read_bytes()
        if len(data) != end - start + 1:
            raise ValueError("Partial index cache")
        print("INDEX", start, len(data), flush=True)
        return data

    with ThreadPoolExecutor(8) as pool:
        central = b"".join(pool.map(index_part, range(offset, offset + size, 1024 * 1024)))
    entries = {}
    pos = 0
    # ZIP64 的普通 EOCD 计数已饱和为 65535。
    while pos < len(central):
        row = struct.unpack_from("<4s6H3L5H2L", central, pos)
        if row[0] != b"PK\x01\x02":
            raise ValueError("Bad central directory")
        name = central[pos + 46 : pos + 46 + row[10]].decode("utf8")
        entries[name] = dict(
            method=row[4], crc=row[7], compressed=row[8], size=row[9], offset=row[16]
        )
        pos += 46 + row[10] + row[11] + row[12]
    names = sorted(n for n in entries if n.startswith("training/rgb/") and n.endswith(".jpg"))
    if len(names) != 130240:
        raise ValueError("Unexpected author archive layout: " + str(len(names)))

    # 只取原始 32560 张图中的 12 个互斥区段。
    # 排除四种加工变体，避免重复手部画面。
    # 该划分仅用于选择，尚未证明人员独立。
    def block(job):
        group, start = job
        selected = names[start : start + 8]
        result = []
        for name in selected:
            info = entries[name]
            dest = ROOT / "rgb" / Path(name).name
            dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                image = dest.read_bytes()
            else:
                begin = info["offset"]
                end = begin + 30 + len(name.encode()) + info["compressed"] + 256
                data = read_range(begin, end)
                p = 0
                local = struct.unpack_from("<4s5H3L2H", data, p)
                if local[0] != b"PK\x03\x04":
                    raise ValueError("Bad local ZIP header")
                p += 30 + local[9] + local[10]
                packed = data[p : p + info["compressed"]]
                image = zlib.decompress(packed, -15) if info["method"] == 8 else packed
            if len(image) != info["size"] or zlib.crc32(image) != info["crc"]:
                raise ValueError("Image CRC mismatch")
            dest.write_bytes(image)
            result.append(
                dict(
                    path=str(dest),
                    archive_entry=name,
                    sha256=hashlib.sha256(image).hexdigest(),
                    group="freihand_block_%02d" % group,
                    split="val" if group % 4 == 3 else "train",
                    label="unreviewed",
                )
            )
        print("RGB_BLOCK", group, len(result), flush=True)
        return result

    jobs = list(enumerate(400 + 2600 * i for i in range(12)))
    with ThreadPoolExecutor(8) as pool:
        rows = sum(list(pool.map(block, jobs)), [])
    report = dict(
        source_url=URL,
        source_page=PAGE,
        source_archive_bytes=int(tail.headers["Content-Range"].split("/")[-1]),
        terms="Research only; commercial use prohibited; cite FreiHAND ICCV 2019. See source page.",
        citation="Christian Zimmermann et al., FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images, ICCV 2019.",
        policy="96 author training images only; manual image review required before negative labelling. Selection blocks do not establish independent subjects.",
        items=rows,
    )
    (ROOT / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(images=len(rows), directory=str(ROOT))), flush=True)


if __name__ == "__main__":
    main()
