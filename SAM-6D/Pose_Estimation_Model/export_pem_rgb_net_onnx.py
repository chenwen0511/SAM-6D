"""
Export PEM ``ViT_AE`` (``feature_extraction.rgb_net``) dense branch to ONNX.

PyTorch 2.0's ONNX exporter does not support ``aten::scaled_dot_product_attention``.
This script patches ``F.scaled_dot_product_attention`` to a matmul/softmax path for
export only (see ``_patch_sdpa_for_onnx_export``). Backend SDPA toggles alone are
often insufficient on 2.0.x.

Usage (from ``Pose_Estimation_Model`` directory)::

    python export_pem_rgb_net_onnx.py
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))

_orig_scaled_dot_product_attention: Any = None


def _setup_path() -> None:
    os.chdir(ROOT)
    for p in ("provider", "utils", "model", os.path.join("model", "pointnet2")):
        sys.path.insert(0, os.path.join(ROOT, p))
    sys.path.insert(0, ROOT)


def _disable_fused_sdpa_for_onnx_export() -> None:
    """Best-effort: prefer decomposable SDPA backends (often still insufficient on PT 2.0 ONNX)."""
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def _sdpa_export_safe(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Math attention path ONNX can trace (replaces fused ``scaled_dot_product_attention``)."""
    d = query.size(-1)
    scale_factor = (d**-0.5) if scale is None else float(scale)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    if is_causal:
        L, S = scores.size(-2), scores.size(-1)
        causal = torch.triu(
            torch.ones(L, S, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=False)
    return torch.matmul(attn, value)


def _patch_sdpa_for_onnx_export() -> None:
    global _orig_scaled_dot_product_attention
    _orig_scaled_dot_product_attention = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_export_safe  # type: ignore[assignment]


def _restore_sdpa() -> None:
    global _orig_scaled_dot_product_attention
    if _orig_scaled_dot_product_attention is not None:
        F.scaled_dot_product_attention = _orig_scaled_dot_product_attention  # type: ignore[assignment]
        _orig_scaled_dot_product_attention = None


class DenseOnly(nn.Module):
    def __init__(self, rgb_net: nn.Module) -> None:
        super().__init__()
        self.rgb_net = rgb_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense, _cls = self.rgb_net(x)
        return dense


def main() -> None:
    _setup_path()
    _disable_fused_sdpa_for_onnx_export()
    _patch_sdpa_for_onnx_export()
    try:
        cfg = gorilla.Config.fromfile(os.path.join(ROOT, "config", "base.yaml"))
        cfg.model_name = "pose_estimation_model"
        cfg.gpus = "0"
        gorilla.utils.set_cuda_visible_devices(gpu_ids="0")

        MODEL = importlib.import_module(cfg.model_name)
        net = MODEL.Net(cfg.model).cuda().eval()
        ckpt = os.path.join(ROOT, "checkpoints", "sam-6d-pem-base.pth")
        gorilla.solver.load_checkpoint(model=net, filename=ckpt)

        m = DenseOnly(net.feature_extraction.rgb_net).cuda().eval()
        dummy = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)

        onnx_path = os.path.join(ROOT, "checkpoints", "pem_rgb_net_b1_224.onnx")
        torch.onnx.export(
            m,
            dummy,
            onnx_path,
            input_names=["images"],
            output_names=["dense_feat"],
            opset_version=17,
            do_constant_folding=True,
        )
        print("ONNX:", onnx_path)
    finally:
        _restore_sdpa()


if __name__ == "__main__":
    main()
