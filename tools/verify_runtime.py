"""校验板端运行文件，不连接开发板或下载硬件。"""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1] / 'board_runtime'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('校验路径超出运行目录：' + name)
        if not path.is_file():
            raise FileNotFoundError(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('文件校验失败：' + name)
    print('运行文件校验通过，共 %d 项。' % len(manifest['files']))


if __name__ == '__main__':
    main()
