"""从官方 Fashion-MNIST 获取服饰辅助负例，保留训练测试边界；未保留完整压缩包，不声明其完整文件 MD5 已校验。"""

import cache_env
import gzip
import hashlib
import json
from pathlib import Path
import struct
import requests


def main():
    output = Path(__file__).resolve().parents[1] / "data/external/fashion_mnist"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for split, name, count in (
        ("train", "train-images-idx3-ubyte.gz", 2000),
        ("test", "t10k-images-idx3-ubyte.gz", 1000),
    ):
        url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" + name
        destination = output / (split + ".idx")
        if not destination.exists():
            with requests.get(url, stream=True, timeout=(15, 25)) as response:
                response.raise_for_status()
                source = gzip.GzipFile(fileobj=response.raw)
                magic, total, height, width = struct.unpack(">IIII", source.read(16))
                if (magic, height, width) != (2051, 28, 28) or total < count:
                    raise ValueError("Unexpected Fashion-MNIST IDX header")
                pixels = source.read(count * 784)
                if len(pixels) != count * 784:
                    raise ValueError("Incomplete IDX records")
                destination.write_bytes(struct.pack(">IIII", 2051, count, 28, 28) + pixels)
        records.append(
            dict(
                split=split,
                url=url,
                first_source_index=0,
                count=count,
                subset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                integrity="Complete IDX records from a source-file prefix; full gzip MD5 not verified",
            )
        )
        print("Fashion-MNIST", split, count, flush=True)
    license_response = requests.get(
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE",
        timeout=(15, 25),
    )
    license_response.raise_for_status()
    (output / "LICENSE").write_bytes(license_response.content)
    (output / "source_manifest.json").write_text(
        json.dumps(
            dict(
                source="https://github.com/zalandoresearch/fashion-mnist",
                authors="Zalando Research, Han Xiao, Kashif Rasul, Roland Vollgraf",
                license="MIT",
                use="Auxiliary garment/object negatives; 28x28 grayscale cannot replace real camera fall/ADL data",
                records=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
