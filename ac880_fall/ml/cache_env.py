"""在导入机器学习库前，将项目缓存与临时目录配置到 E 盘。"""

import os
from pathlib import Path
import tempfile


def configure():
    root = (
        Path("E:/Zynq_cache") if os.name == "nt" else Path(__file__).resolve().parents[2] / ".cache"
    )
    paths = {
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "PIP_CACHE_DIR": "pip",
        "CUDA_CACHE_PATH": "cuda",
        "XDG_CACHE_HOME": "xdg",
        "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba",
        "TRITON_CACHE_DIR": "triton",
        "JOBLIB_TEMP_FOLDER": "temp",
        "TEMP": "temp",
        "TMP": "temp",
    }
    for key, relative in paths.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    tempfile.tempdir = str(root / "temp")
    return root


configure()
