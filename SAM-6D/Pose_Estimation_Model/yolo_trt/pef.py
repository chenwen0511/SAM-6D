"""
Benchmark YOLO segmentation: PyTorch (.pt) vs TensorRT (.engine) predict latency.

Aligns with HTTP stack: task=segment, same args as yolo_seg_backend.run_yolo_segmentation.

Usage (Ubuntu, from repo root or PEM dir):

  cd /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model/yolo_trt
  python pef.py \\
    --pt /path/to/best.pt \\
    --engine /path/to/best.engine \\
    --rgb /path/to/rgb.png \\
    --warmup 10 \\
    --runs 50 \\
    --imgsz 640 \\
    --conf 0.25 \\
    --device 0 \\
    --baseline-s 0.16

One-time `yolo export ... format=engine` is NOT included in predict timings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from ultralytics import YOLO


def _sync_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _predict_once(model: YOLO, rgb: str, *, conf: float, imgsz: int, device: str) -> None:
    model.predict(
        rgb,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        task="segment",
    )


def bench_weights(
    weights: Path,
    rgb_path: Path,
    *,
    warmup: int,
    runs: int,
    conf: float,
    imgsz: int,
    device: str,
) -> List[float]:
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb not found: {rgb_path}")

    rgb = str(rgb_path.resolve())
    model = YOLO(str(weights.resolve()), task="segment")

    for _ in range(warmup):
        _predict_once(model, rgb, conf=conf, imgsz=imgsz, device=device)
        _sync_cuda()

    samples: List[float] = []
    for _ in range(runs):
        _sync_cuda()
        t0 = time.perf_counter()
        _predict_once(model, rgb, conf=conf, imgsz=imgsz, device=device)
        _sync_cuda()
        samples.append(time.perf_counter() - t0)

    return samples


def _summarize_ms(times_s: List[float]) -> dict:
    ms = [t * 1000.0 for t in times_s]
    return {
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "std_ms": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
        "runs": len(ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO seg .pt vs .engine predict benchmark")
    parser.add_argument("--pt", type=Path, help="Path to best.pt")
    parser.add_argument("--engine", type=Path, help="Path to best.engine")
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--baseline-s",
        type=float,
        default=None,
        help="Optional baseline latency in seconds (e.g. 0.16 from your HTTP log) for speedup print",
    )
    args = parser.parse_args()

    if not args.pt and not args.engine:
        print("Provide at least one of --pt or --engine", file=sys.stderr)
        sys.exit(2)

    results: dict = {}
    if args.pt:
        t = bench_weights(
            args.pt,
            args.rgb,
            warmup=args.warmup,
            runs=args.runs,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        results["pt"] = _summarize_ms(t)
    if args.engine:
        t = bench_weights(
            args.engine,
            args.rgb,
            warmup=args.warmup,
            runs=args.runs,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        results["engine"] = _summarize_ms(t)

    print("[pef] rgb:", args.rgb.resolve())
    print("[pef] warmup:", args.warmup, "runs:", args.runs, "imgsz:", args.imgsz, "conf:", args.conf, "device:", args.device)
    for name, s in results.items():
        print(f"[pef] {name}: mean={s['mean_ms']:.3f} ms median={s['median_ms']:.3f} ms "
              f"min={s['min_ms']:.3f} max={s['max_ms']:.3f} std={s['std_ms']:.3f} (n={s['runs']})")

    if "pt" in results and "engine" in results:
        pt_m = results["pt"]["mean_ms"]
        en_m = results["engine"]["mean_ms"]
        if en_m > 0:
            print(f"[pef] engine vs pt mean: {pt_m / en_m:.2f}x faster (pt/engine)")
        if pt_m > 0:
            print(f"[pef] pt vs engine mean: {en_m / pt_m:.2f}x faster (engine/pt)")

    if args.baseline_s is not None:
        base_ms = args.baseline_s * 1000.0
        print(f"[pef] baseline (given): {base_ms:.3f} ms")
        for name, s in results.items():
            if s["mean_ms"] > 0:
                print(f"[pef] baseline / {name} mean = {base_ms / s['mean_ms']:.2f}x")


if __name__ == "__main__":
    main()
