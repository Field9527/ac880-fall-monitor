"""由作者相机及关节点标注生成带边距的手部粗框；不是作者人工检测框，训练前必须复核。"""

import cache_env
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import zlib
import numpy as np
from PIL import Image, ImageDraw
import fetch_hand_negatives as archive_source
from fetch_hand_negatives import read_range, PAGE


def project_region(xyz, intrinsic, width, height):
    xyz = np.asarray(xyz, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if (
        xyz.shape != (21, 3)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(xyz).all()
        or not np.isfinite(intrinsic).all()
        or (xyz[:, 2] <= 0).any()
    ):
        raise ValueError("Invalid author camera/joint annotation")
    camera = xyz @ intrinsic.T
    if (camera[:, 2] <= 0).any():
        raise ValueError("Invalid projected depth")
    uv = camera[:, :2] / camera[:, 2:]
    lo, hi = uv.min(0), uv.max(0)
    margin = np.maximum(6, (hi - lo) * 0.15)
    box = np.r_[np.maximum(lo - margin, 0), np.minimum(hi + margin, [width, height])]
    if not np.isfinite(box).all() or np.any(box[2:] - box[:2] < 2):
        raise ValueError("Invalid projected hand region")
    return box.tolist(), uv.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument(
        "--archive-url",
        default="https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
    )
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    archive_source.URL = a.archive_url
    central = b"".join(
        v.read_bytes()
        for v in sorted(a.source.glob("index_*.bin"), key=lambda v: int(v.stem.split("_")[1]))
    )
    entries = {}
    offset = 0
    while offset < len(central):
        row = struct.unpack_from("<4s6H3L5H2L", central, offset)
        if row[0] != b"PK\x01\x02":
            raise ValueError("Invalid cached central directory")
        name = central[offset + 46 : offset + 46 + row[10]].decode("utf8")
        if name in ("training_K.json", "training_xyz.json"):
            entries[name] = dict(
                method=row[4], crc=row[7], compressed=row[8], size=row[9], offset=row[16]
            )
        offset += 46 + row[10] + row[11] + row[12]

    def load(name):
        info = entries[name]
        cache = Path("E:/Zynq_cache/datasets/freihand") / name
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            raw = cache.read_bytes()
        else:
            header = read_range(info["offset"], info["offset"] + 29)
            local = struct.unpack("<4s5H3L2H", header)
            if local[0] != b"PK\x03\x04" or local[3] != info["method"]:
                raise ValueError("ZIP local-header mismatch")
            start = info["offset"] + 30 + local[9] + local[10]
            end = start + info["compressed"]
            ranges = [(i, min(end, i + 1024 * 1024) - 1) for i in range(start, end, 1024 * 1024)]
            with ThreadPoolExecutor(4) as pool:
                packed = b"".join(pool.map(lambda v: read_range(*v), ranges))
            raw = zlib.decompress(packed, -15) if info["method"] == 8 else packed
            cache.write_bytes(raw)
        if len(raw) != info["size"] or zlib.crc32(raw) != info["crc"]:
            raise ValueError("Author metadata CRC mismatch")
        return json.loads(raw), dict(
            file=name, sha256=hashlib.sha256(raw).hexdigest(), crc32=info["crc"]
        )

    intrinsics, k_source = load("training_K.json")
    joints, xyz_source = load("training_xyz.json")
    if len(intrinsics) != 32560 or len(joints) != 32560:
        raise ValueError("Unexpected training annotation count")
    manifest = json.loads((a.source / "download_manifest.json").read_text())
    items = []
    tiles = []
    for item in manifest["items"]:
        image = Path(item["path"])
        if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source RGB changed")
        index = int(image.stem)
        if not 0 <= index < 32560:
            raise ValueError("Only original author training views are allowed")
        with Image.open(image) as source:
            rgb = source.convert("RGB")
        box, uv = project_region(joints[index], intrinsics[index], rgb.width, rgb.height)
        items.append(
            dict(
                path=str(image),
                sha256=item["sha256"],
                box=box,
                joints_uv=uv,
                label="hand_region",
                source_index=index,
                group=item["group"],
                quality="author_joints_projected_with_15_percent_margin",
                review_status="pending_visual_review",
                train_eligible=False,
            )
        )
        draw = ImageDraw.Draw(rgb)
        draw.rectangle(box, outline="red", width=2)
        for x, y in uv:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="lime")
        draw.rectangle((0, 0, 100, 14), fill="black")
        draw.text((2, 1), str(index), fill="white")
        tiles.append(rgb)
    for page in range((len(tiles) + 23) // 24):
        canvas = Image.new("RGB", (4 * 224, 6 * 224), "#333333")
        for n, tile in enumerate(tiles[page * 24 : (page + 1) * 24]):
            canvas.paste(tile, (n % 4 * 224, n // 4 * 224))
        canvas.save(a.out / ("review_%02d.jpg" % page), quality=90)
    result = dict(
        items=items,
        source_url=a.archive_url,
        source_page=PAGE,
        author_metadata=[k_source, xyz_source],
        projection_reference="https://github.com/lmb-freiburg/freihand/blob/master/utils/fh_utils.py",
        terms=manifest["terms"],
        citation=manifest["citation"],
        limitations=[
            "Derived joint-region boxes, not author-drawn full-hand object boxes",
            "No person/background completeness or scene-independent validation implied",
            "Visual review is required before train_eligible may be set",
        ],
    )
    (a.out / "hand_regions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(derived_regions=len(items), review_pages=(len(tiles) + 23) // 24)))


if __name__ == "__main__":
    main()
