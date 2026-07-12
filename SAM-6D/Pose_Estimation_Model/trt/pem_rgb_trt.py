"""
TensorRT runner for PEM ``ViT_AE`` dense branch (ONNX I/O: ``images`` / ``dense_feat``).

Expects a fixed **batch=1** engine (see ``export_pem_rgb_net_onnx.py``). ``forward_dense``
loops per instance when ``B>1``. Requires **TensorRT** Python bindings matching the
engine build (e.g. 10.x) and **CUDA** tensors **float32** **NCHW** aligned with training
preprocess (ImageNet norm, 224).
"""

from __future__ import annotations

import os

import torch

INPUT_NAME = "images"
OUTPUT_NAME = "dense_feat"
SHAPE_IN = (1, 3, 224, 224)
SHAPE_OUT = (1, 256, 224, 224)


class PemRgbNetTrt:
    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt  # type: ignore[import-not-found]

        self._trt = trt
        path = os.path.abspath(os.fspath(engine_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PEM RGB TensorRT engine not found: {path}")

        with open(path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"TensorRT deserialize_cuda_engine failed: {path}")

        self.context = self.engine.create_execution_context()
        self.input_tensor_name = self._resolve_tensor_name(is_input=True)
        self.output_tensor_name = self._resolve_tensor_name(is_input=False)

        dev = torch.device("cuda", torch.cuda.current_device())
        self._infer_batch1(torch.zeros(SHAPE_IN, device=dev, dtype=torch.float32))

    def _resolve_tensor_name(self, *, is_input: bool) -> str:
        trt = self._trt
        engine = self.engine
        want = INPUT_NAME if is_input else OUTPUT_NAME
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and is_input and name == want:
                return name
            if mode == trt.TensorIOMode.OUTPUT and (not is_input) and name == want:
                return name
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if is_input and mode == trt.TensorIOMode.INPUT:
                return name
            if (not is_input) and mode == trt.TensorIOMode.OUTPUT:
                return name
        raise RuntimeError("TensorRT engine: could not resolve input/output tensor names")

    def forward_dense(self, img: torch.Tensor) -> torch.Tensor:
        """``img`` Bx3x224x224 float32 CUDA -> Bx256x224x224 float32 CUDA."""
        if img.dim() != 4 or img.size(1) != 3 or img.size(2) != 224 or img.size(3) != 224:
            raise ValueError(f"PemRgbNetTrt expected Nx3x224x224, got {tuple(img.shape)}")
        if img.dtype != torch.float32:
            raise ValueError("PemRgbNetTrt expects float32 input")
        if not img.is_cuda:
            raise ValueError("PemRgbNetTrt expects a CUDA tensor")
        img = img.contiguous()
        b = img.size(0)
        if b == 1:
            return self._infer_batch1(img)
        return torch.cat([self._infer_batch1(img[i : i + 1]) for i in range(b)], dim=0)

    def _infer_batch1(self, x1: torch.Tensor) -> torch.Tensor:
        ctx = self.context
        out = torch.empty(SHAPE_OUT, device=x1.device, dtype=torch.float32)
        ctx.set_tensor_address(self.input_tensor_name, x1.data_ptr())
        ctx.set_tensor_address(self.output_tensor_name, out.data_ptr())

        stream = torch.cuda.current_stream(device=x1.device)
        stream_ptr = int(stream.cuda_stream) if hasattr(stream, "cuda_stream") else 0

        if not hasattr(ctx, "execute_async_v3"):
            raise RuntimeError(
                "TensorRT IExecutionContext has no execute_async_v3; upgrade TensorRT (>= 8.6 / 10.x)."
            )
        ok = ctx.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        stream.synchronize()
        return out


def resolve_engine_path(raw: str) -> str:
    """If ``raw`` is not an existing file, try ``Pose_Estimation_Model/<raw>``."""
    raw = raw.strip().strip('"').strip("'")
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    pem_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.abspath(os.path.join(pem_root, raw))
