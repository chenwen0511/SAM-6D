"""
Compare PEM ``ViT_AE`` dense map ``rgb_net(img)[0]`` (PyTorch) vs ``PemRgbNetTrt`` (TensorRT).

Aligns with ``vit_acc.md`` §9.6: same normalized ``Nx3x224x224`` inputs as training/inference
(``ToTensor`` + ImageNet ``Normalize``), then reports dense and optional ``get_chosen_pixel_feats`` errors.

Usage (from ``Pose_Estimation_Model``)::

    python trt/pose_acc/accurate_val.py --engine checkpoints/pem_rgb_net_b1_224_fp16.engine

If ``--engine`` is omitted, uses ``SAM6D_PEM_RGB_TRT_ENGINE`` when set; otherwise
``checkpoints/pem_rgb_net_b1_224_fp16.engine`` under ``Pose_Estimation_Model``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import gorilla
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _setup_path() -> None:
    os.chdir(ROOT)
    for p in ("provider", "utils", "model", os.path.join("model", "pointnet2")):
        sys.path.insert(0, os.path.join(ROOT, p))
    sys.path.insert(0, ROOT)


def _imagenet_norm_random(n: int, device: torch.device, *, seed: int) -> torch.Tensor:
    """Same distribution family as ``rgb_transform`` (uniform in [0,1) then Normalize)."""
    g = torch.Generator()
    g.manual_seed(seed)
    u = torch.rand((n, 3, 224, 224), generator=g, dtype=torch.float32).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (u - mean) / std


def _resolve_engine_path(arg: str | None) -> str:
    from trt.pem_rgb_trt import resolve_engine_path

    raw = (arg or os.environ.get("SAM6D_PEM_RGB_TRT_ENGINE") or "").strip()
    if not raw:
        raw = os.path.join("checkpoints", "pem_rgb_net_b1_224_fp16.engine")
    return resolve_engine_path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="PEM rgb_net PyTorch vs TensorRT dense accuracy")
    parser.add_argument(
        "--engine",
        default=None,
        help="TensorRT engine path (default: env SAM6D_PEM_RGB_TRT_ENGINE or checkpoints/pem_rgb_net_b1_224_fp16.engine)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "sam-6d-pem-base.pth"),
        help="PEM checkpoint (default: checkpoints/sam-6d-pem-base.pth)",
    )
    parser.add_argument("--config", default=os.path.join("config", "base.yaml"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8, help="Number of random inputs for dense stats")
    parser.add_argument(
        "--n-choose",
        type=int,
        default=2048,
        help="Points for get_chosen_pixel_feats (match fine_npoint)",
    )
    parser.add_argument(
        "--fail-if-max-abs-above",
        type=float,
        default=None,
        help="If set, exit 1 when any sample max |pt-trt| exceeds this (e.g. 0.5 for FP16 engine smoke test)",
    )
    args = parser.parse_args()

    _setup_path()
    engine_path = _resolve_engine_path(args.engine)
    os.environ.pop("SAM6D_PEM_RGB_TRT_ENGINE", None)

    gorilla.utils.set_cuda_visible_devices(gpu_ids=args.gpus)
    device = torch.device("cuda", 0)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = gorilla.Config.fromfile(os.path.join(ROOT, args.config))
    cfg.model_name = "pose_estimation_model"
    cfg.gpus = args.gpus

    MODEL = importlib.import_module(cfg.model_name)
    net = MODEL.Net(cfg.model).to(device).eval()
    ckpt = os.path.join(ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    gorilla.solver.load_checkpoint(model=net, filename=ckpt)

    rgb_net = net.feature_extraction.rgb_net
    from trt.pem_rgb_trt import PemRgbNetTrt

    trt_runner = PemRgbNetTrt(engine_path)

    from model_utils import get_chosen_pixel_feats

    max_abs_all = 0.0
    mean_abs_all = 0.0
    rms_rel_all = 0.0
    max_g_all = 0.0

    print(f"=> engine: {engine_path}")
    print(f"=> checkpoint: {ckpt}")
    print(f"=> samples: {args.samples}, n_choose: {args.n_choose}, seed: {args.seed}")

    for i in range(args.samples):
        x = _imagenet_norm_random(1, device, seed=args.seed + i)
        with torch.no_grad():
            pt = rgb_net(x)[0]
            tt = trt_runner.forward_dense(x)
        diff = (pt - tt).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        denom = float(pt.abs().mean().item()) + 1e-8
        rms_rel = float((diff.pow(2).mean().sqrt().item()) / denom)
        max_abs_all = max(max_abs_all, max_abs)
        mean_abs_all += mean_abs
        rms_rel_all += rms_rel

        g_cpu = torch.Generator()
        g_cpu.manual_seed(args.seed + 10000 + i)
        choose = torch.randint(
            0,
            224 * 224,
            (1, args.n_choose),
            generator=g_cpu,
            dtype=torch.long,
        ).to(device)
        with torch.no_grad():
            g_pt = get_chosen_pixel_feats(pt, choose)
            g_tt = get_chosen_pixel_feats(tt, choose)
        g_diff = (g_pt - g_tt).abs().max().item()
        max_g_all = max(max_g_all, float(g_diff))

        print(f"  sample {i}: dense max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms~={rms_rel:.6g} | gather max_abs={g_diff:.6g}")

    mean_abs_all /= max(args.samples, 1)
    rms_rel_all /= max(args.samples, 1)

    print("=> summary (dense, over samples):")
    print(f"   max_abs (worst): {max_abs_all:.6g}")
    print(f"   mean_abs (avg): {mean_abs_all:.6g}")
    print(f"   rel_rms vs |pt|_mean (avg): {rms_rel_all:.6g}")
    print(f"   get_chosen_pixel_feats max_abs (worst): {max_g_all:.6g}")

    b_test = 3
    xb = _imagenet_norm_random(b_test, device, seed=args.seed + 99999)
    with torch.no_grad():
        pt_b = torch.cat([rgb_net(xb[i : i + 1])[0] for i in range(b_test)], dim=0)
        tt_b = trt_runner.forward_dense(xb)
    db = (pt_b - tt_b).abs()
    print(f"=> batch B={b_test}: dense max_abs={float(db.max().item()):.6g} mean_abs={float(db.mean().item()):.6g}")

    if args.fail_if_max_abs_above is not None and max_abs_all > args.fail_if_max_abs_above:
        raise SystemExit(
            f"max_abs {max_abs_all} > threshold {args.fail_if_max_abs_above} (set looser or fix engine/export)"
        )


if __name__ == "__main__":
    main()

