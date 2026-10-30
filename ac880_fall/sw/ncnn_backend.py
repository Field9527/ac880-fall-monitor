"""NCNN C 接口封装，ARMv7 无需安装 Python 推理包。"""

import ctypes as ct
from pathlib import Path
import numpy as np


class NCNNNet:
    def __init__(
        self,
        library,
        param,
        weights,
        threads=1,
        pl_plugin=None,
        output_names=None,
        flat_outputs=False,
        output_layouts=None,
    ):
        self.pl = self.plugin = None
        self.flat_outputs = bool(flat_outputs)
        self.named_outputs = output_names is not None
        count = len(output_names) if output_names is not None else 3
        self.output_layouts = (
            list(output_layouts)
            if output_layouts is not None
            else [2 if flat_outputs else 3] * count
        )
        if len(self.output_layouts) != count or any(
            type(v) is not int or v not in (1, 2, 3) for v in self.output_layouts
        ):
            raise ValueError("Each checked output requires a vector, matrix or CHW layout")
        self.lib = ct.CDLL(str(Path(library).resolve()))
        ptr = ct.c_void_p
        integer = ct.c_int
        size = ct.c_size_t
        string = ct.c_char_p
        signatures = {
            "ncnn_net_create": (ptr, []),
            "ncnn_net_destroy": (None, [ptr]),
            "ncnn_net_get_option": (ptr, [ptr]),
            "ncnn_option_set_num_threads": (None, [ptr, integer]),
            "ncnn_net_load_param": (integer, [ptr, string]),
            "ncnn_net_load_model": (integer, [ptr, string]),
            "ncnn_net_get_input_count": (integer, [ptr]),
            "ncnn_net_get_output_count": (integer, [ptr]),
            "ncnn_net_get_input_name": (string, [ptr, integer]),
            "ncnn_net_get_output_name": (string, [ptr, integer]),
            "ncnn_mat_create_external_3d": (ptr, [integer, integer, integer, ptr, ptr]),
            "ncnn_mat_destroy": (None, [ptr]),
            "ncnn_mat_get_data": (ptr, [ptr]),
            "ncnn_mat_get_cstep": (size, [ptr]),
            "ncnn_mat_get_elemsize": (size, [ptr]),
            "ncnn_extractor_create": (ptr, [ptr]),
            "ncnn_extractor_destroy": (None, [ptr]),
            "ncnn_extractor_input": (integer, [ptr, string, ptr]),
            "ncnn_extractor_extract": (integer, [ptr, string, ct.POINTER(ptr)]),
        }
        for key in ("dims", "w", "h", "d", "c", "elempack"):
            signatures["ncnn_mat_get_" + key] = (integer, [ptr])
        for key in (
            "fp16_packed",
            "fp16_storage",
            "fp16_arithmetic",
            "bf16_storage",
            "int8_packed",
            "int8_storage",
            "int8_arithmetic",
            "vulkan_compute",
        ):
            signatures["ncnn_option_set_use_" + key] = (None, [ptr, integer])
        for name, (result, args) in signatures.items():
            f = getattr(self.lib, name)
            f.restype = result
            f.argtypes = args
        self.net = self.lib.ncnn_net_create()
        if not self.net:
            raise MemoryError("NCNN net allocation failed")
        try:
            if pl_plugin is not None:
                self.plugin = ct.CDLL(str(Path(pl_plugin).resolve()))
                self.plugin.ac880_register_pl.argtypes = [ptr]
                self.plugin.ac880_register_pl.restype = integer
                self.plugin.ac880_set_compute_callback.argtypes = [ptr]
                self.plugin.ac880_set_compute_callback.restype = None
                if self.plugin.ac880_register_pl(self.net) != 0:
                    raise RuntimeError("NCNN PL layer registration failed")
            opt = self.lib.ncnn_net_get_option(self.net)
            self.lib.ncnn_option_set_num_threads(opt, int(threads))
            for key in (
                "fp16_packed",
                "fp16_storage",
                "fp16_arithmetic",
                "bf16_storage",
                "int8_packed",
                "int8_storage",
                "int8_arithmetic",
                "vulkan_compute",
            ):
                getattr(self.lib, "ncnn_option_set_use_" + key)(opt, 0)
            for function, path in (
                ("ncnn_net_load_param", param),
                ("ncnn_net_load_model", weights),
            ):
                if getattr(self.lib, function)(self.net, str(Path(path).resolve()).encode()) != 0:
                    raise RuntimeError("NCNN model load failed: " + str(path))
            count = len(output_names) if output_names is not None else 3
            if (
                self.lib.ncnn_net_get_input_count(self.net),
                self.lib.ncnn_net_get_output_count(self.net),
            ) != (1, count):
                raise ValueError("Unexpected NCNN input/output count")
            self.input_name = self.lib.ncnn_net_get_input_name(self.net, 0)
            available = [self.lib.ncnn_net_get_output_name(self.net, i) for i in range(count)]
            if output_names is None:
                self.output_names = available
            else:
                requested = [name.encode("utf8") for name in output_names]
                if len(set(requested)) != count or set(requested) != set(available):
                    raise ValueError("Requested outputs do not match the checked graph")
                self.output_names = requested
        except Exception:
            self.close()
            raise

    def forward(self, blob):
        if self.plugin is not None and self.pl is None:
            raise RuntimeError("PL model cannot run until hardware is attached")
        if blob.dtype != np.float32 or blob.ndim != 4 or blob.shape[0] != 1:
            raise ValueError("NCNN requires one FP32 NCHW image")
        image = np.ascontiguousarray(blob[0])
        channels, height, width = image.shape
        # 外部 Mat 引用自有且已对齐的数组，直到提取完成。
        # 数组不与 DMA 缓冲区共享。
        if width * height % 4:
            raise ValueError("Unaligned NCNN input dimensions")
        mat = self.lib.ncnn_mat_create_external_3d(width, height, channels, image.ctypes.data, None)
        ex = self.lib.ncnn_extractor_create(self.net)
        if not mat or not ex:
            raise MemoryError("NCNN extractor allocation failed")
        outputs = []
        try:
            if self.lib.ncnn_extractor_input(ex, self.input_name, mat) != 0:
                raise RuntimeError("NCNN input failed")
            for name, layout in zip(self.output_names, self.output_layouts):
                result = ct.c_void_p()
                try:
                    if self.lib.ncnn_extractor_extract(ex, name, ct.byref(result)) != 0:
                        detail = self.pl.last_error if self.pl is not None else None
                        raise RuntimeError(
                            "NCNN extraction failed" + (": " + detail if detail else "")
                        )
                    dims = self.lib.ncnn_mat_get_dims(result)
                    if (
                        dims != layout
                        or self.lib.ncnn_mat_get_elempack(result) != 1
                        or self.lib.ncnn_mat_get_elemsize(result) != 4
                    ):
                        raise ValueError("NCNN output must be unpacked FP32 CHW")
                    w = self.lib.ncnn_mat_get_w(result)
                    h = self.lib.ncnn_mat_get_h(result)
                    c = self.lib.ncnn_mat_get_c(result)
                    step = self.lib.ncnn_mat_get_cstep(result)
                    data = self.lib.ncnn_mat_get_data(result)
                    if not data or min(w, h, c) <= 0 or step < w * h:
                        raise ValueError("Invalid NCNN output layout")
                    if layout == 1:
                        view = np.ctypeslib.as_array(
                            ct.cast(data, ct.POINTER(ct.c_float)), shape=(w,)
                        )
                        outputs.append(view.copy().reshape(1, w))
                    elif layout == 2:
                        view = np.ctypeslib.as_array(
                            ct.cast(data, ct.POINTER(ct.c_float)), shape=(h * w,)
                        )
                        outputs.append(view.copy().reshape(h, w))
                    else:
                        view = np.ctypeslib.as_array(
                            ct.cast(data, ct.POINTER(ct.c_float)), shape=(c * step,)
                        )
                        outputs.append(view.reshape(c, step)[:, : h * w].copy().reshape(1, c, h, w))
                finally:
                    if result:
                        self.lib.ncnn_mat_destroy(result)
        finally:
            self.lib.ncnn_extractor_destroy(ex)
            self.lib.ncnn_mat_destroy(mat)
        # 按 head8、head16、head32 排序，不依赖图输出顺序。
        return (
            outputs
            if self.flat_outputs or self.named_outputs
            else sorted(outputs, key=lambda v: v.shape[2] * v.shape[3], reverse=True)
        )

    def attach_pl(self, overlay, timeout_s=5.0):
        if self.plugin is None or self.pl is not None:
            raise RuntimeError("PL attachment requires an unattached PL model")
        from pl_pointwise import PLPointwise

        self.pl = PLPointwise.from_overlay(overlay, timeout_s)
        self.plugin.ac880_set_compute_callback(ct.cast(self.pl.callback, ct.c_void_p))

    def close(self):
        if getattr(self, "pl", None) is not None:
            self.pl.close()
            self.plugin.ac880_set_compute_callback(None)
            self.pl = None
        if getattr(self, "net", None):
            self.lib.ncnn_net_destroy(self.net)
            self.net = None

    def __del__(self):
        self.close()
