import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
PEM_DIR = THIS_DIR.parent
if str(PEM_DIR) not in sys.path:
    sys.path.insert(0, str(PEM_DIR))

import numpy as np
import torch

from run_warmup_inference_custom import preload_default_pem_model, run_pose_inference  # noqa: E402


def _seed_all(rd_seed: int) -> None:
    """Align with PEM ``run_pose_inference`` (random / torch / numpy / CUDA). Call before preload."""
    random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    np.random.seed(rd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rd_seed)


def _rotation_matrix_to_euler_zyx(rotation: List[List[float]]) -> List[float]:
    """rx, ry, rz in radians; ZYX order: R = Rz(rz) * Ry(ry) * Rx(rx). Same as sam6d_http_service."""
    r00, r10 = float(rotation[0][0]), float(rotation[1][0])
    r20, r21, r22 = float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])
    r01, r11 = float(rotation[0][1]), float(rotation[1][1])

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r01, r11)
        ry = math.atan2(-r20, sy)
        rz = 0.0
    return [rx, ry, rz]


def _serialize_pose_snapshot(out: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten PEM outputs for JSON comparison against TRT runs (align fields with HTTP PEM JSON)."""
    bd = out.get("best_detection") or {}
    R = bd.get("R")
    t = bd.get("t")
    if R is not None:
        R = [[float(x) for x in row] for row in R]
    if t is not None:
        t = [float(x) for x in t]

    euler_zyx_rad = _rotation_matrix_to_euler_zyx(R) if R else None
    xyz_mm = t
    xyzrxryrz = (
        list(xyz_mm) + list(euler_zyx_rad)
        if euler_zyx_rad is not None and xyz_mm is not None
        else None
    )

    snap: Dict[str, Any] = {
        "score": float(bd.get("score", 0.0)),
        "xyz_mm": xyz_mm,
        "rotation_euler_zyx_rad": euler_zyx_rad,
        "rotation_order": "zyx",
        "pose_convention": (
            "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians."
        ),
        "xyzrxryrz": xyzrxryrz,
        "xyzrxryrz_unit": "mm_rad",
        "best_idx": int(out.get("best_idx", -1)),
        "t_mm": xyz_mm,
        "R": R,
        "num_detections": len(out.get("detections") or []),
    }
    ps = out.get("pose_scores")
    if ps is not None:
        try:
            import numpy as np

            snap["pose_scores"] = ps.tolist() if hasattr(ps, "tolist") else list(ps)
        except Exception:
            snap["pose_scores"] = None
    snap["detection_pem_path"] = out.get("detection_pem_path")
    return snap


def _now_ns() -> int:
    return time.perf_counter_ns()


def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def _run_once(args) -> Tuple[float, Dict[str, Any]]:
    t0 = _now_ns()
    out = run_pose_inference(
        output_dir=args.output_dir,
        cad_path=args.cad_path,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        cam_path=args.cam_path,
        seg_path=args.seg_path,
        det_score_thresh=args.det_score_thresh,
        gpus=args.gpus,
        verbose=False,
        save_visualization=False,
    )
    t1 = _now_ns()
    return _ns_to_ms(t1 - t0), out


def _write_csv(csv_path: Path, rows: List[List[object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _write_results_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _default_results_path(csv_path: str) -> Path:
    p = Path(csv_path).resolve()
    return p.with_name(p.stem + "_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PEM baseline benchmark with warmup and repeated runs.")
    parser.add_argument("--output_dir", required=True, help="SAM-6D output dir (must contain templates/).")
    parser.add_argument("--cad_path", required=True, help="Path to CAD .ply.")
    parser.add_argument("--rgb_path", required=True, help="Path to RGB image.")
    parser.add_argument("--depth_path", required=True, help="Path to depth image.")
    parser.add_argument("--cam_path", required=True, help="Path to camera json.")
    parser.add_argument("--seg_path", required=True, help="Path to detection_ism.json (or equivalent).")
    parser.add_argument("--det_score_thresh", type=float, default=0.0, help="Detection threshold for PEM input.")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value used by PEM.")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of warmup inference runs.")
    parser.add_argument("--benchmark_runs", type=int, default=30, help="Number of measured runs.")
    parser.add_argument(
        "--csv_path",
        default=str(THIS_DIR / "baseline.csv"),
        help="Output CSV path for raw and summary metrics.",
    )
    parser.add_argument(
        "--results_json",
        default=None,
        help="Output JSON path for pose/score snapshots (default: same stem as csv -> *_results.json).",
    )
    parser.add_argument(
        "--rd_seed",
        type=int,
        default=1,
        help="Global RNG seed before PEM preload/inference; keep equal to PEM config rd_seed (default in config/base.yaml is 1). "
        "Each run_pose_inference still re-seeds from cfg.rd_seed internally.",
    )
    args = parser.parse_args()

    results_json_path = Path(args.results_json).resolve() if args.results_json else _default_results_path(args.csv_path)

    _seed_all(args.rd_seed)
    print(f"[baseline] rd_seed={args.rd_seed} (random/torch/numpy/cuda)")

    # 1) 先做模型预加载，避免首轮加载成本污染基线。
    preload_default_pem_model(gpus=args.gpus, verbose=True)

    # 2) 进行推理预热，稳定 CUDA kernel/allocator 状态。
    print(f"[baseline] warmup runs: {args.warmup_runs}")
    for i in range(args.warmup_runs):
        ms, _ = _run_once(args)
        print(f"[baseline] warmup {i + 1}/{args.warmup_runs}: {ms:.3f} ms")

    # 3) 正式计时，多次调用取平均。
    print(f"[baseline] benchmark runs: {args.benchmark_runs}")
    samples_ms: List[float] = []
    per_run_snapshots: List[Dict[str, Any]] = []
    last_out: Optional[Dict[str, Any]] = None
    for i in range(args.benchmark_runs):
        ms, out = _run_once(args)
        samples_ms.append(ms)
        last_out = out
        per_run_snapshots.append(
            {
                "run_index": i + 1,
                "latency_ms": ms,
                "pose": _serialize_pose_snapshot(out),
            }
        )
        print(f"[baseline] run {i + 1}/{args.benchmark_runs}: {ms:.3f} ms")

    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    p95_ms = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)]
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0

    run_columns = [
        "run_index",
        "latency_ms",
        "score",
        "x_mm",
        "y_mm",
        "z_mm",
        "rx_rad",
        "ry_rad",
        "rz_rad",
    ]
    rows: List[List[object]] = [
        ["metric", "value_ms"],
        ["mean", f"{mean_ms:.6f}"],
        ["median", f"{median_ms:.6f}"],
        ["p95", f"{p95_ms:.6f}"],
        ["min", f"{min_ms:.6f}"],
        ["max", f"{max_ms:.6f}"],
        ["std", f"{std_ms:.6f}"],
        [],
        run_columns,
    ]
    for item in per_run_snapshots:
        pose = item["pose"]
        ms = item["latency_ms"]
        xyzr = pose.get("xyzrxryrz")
        sc = pose.get("score")
        if xyzr is not None and len(xyzr) == 6:
            x_mm, y_mm, z_mm, rx, ry, rz = xyzr
            rows.append(
                [
                    item["run_index"],
                    f"{ms:.6f}",
                    f"{float(sc):.9f}",
                    f"{float(x_mm):.9f}",
                    f"{float(y_mm):.9f}",
                    f"{float(z_mm):.9f}",
                    f"{float(rx):.9f}",
                    f"{float(ry):.9f}",
                    f"{float(rz):.9f}",
                ]
            )
        else:
            rows.append(
                [
                    item["run_index"],
                    f"{ms:.6f}",
                    f"{float(sc):.9f}" if sc is not None else "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    csv_path = Path(args.csv_path).resolve()
    _write_csv(csv_path, rows)

    results_payload: Dict[str, Any] = {
        "schema": "sam6d_pem_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "output_dir": os.fspath(args.output_dir),
            "cad_path": os.fspath(args.cad_path),
            "rgb_path": os.fspath(args.rgb_path),
            "depth_path": os.fspath(args.depth_path),
            "cam_path": os.fspath(args.cam_path),
            "seg_path": os.fspath(args.seg_path),
            "det_score_thresh": args.det_score_thresh,
            "gpus": args.gpus,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "rd_seed": args.rd_seed,
        },
        "timing_summary_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
            "min": min_ms,
            "max": max_ms,
            "std": std_ms,
        },
        "reference_for_trt_compare": None,
        "per_run": per_run_snapshots,
        "note": (
            "Primary compare fields: score and xyzrxryrz [x_mm,y_mm,z_mm,rx,ry,rz_rad] (same convention as sam6d HTTP PEM). "
            "reference_for_trt_compare is the last benchmark run. If per_run differs across runs, check CUDA nondeterminism."
        ),
    }
    if last_out is not None:
        results_payload["reference_for_trt_compare"] = _serialize_pose_snapshot(last_out)

    _write_results_json(results_json_path, results_payload)

    print("[baseline] done")
    print(f"[baseline] mean={mean_ms:.3f} ms median={median_ms:.3f} ms p95={p95_ms:.3f} ms")
    print(f"[baseline] csv saved: {csv_path}")
    print(f"[baseline] results json (pose/score for TRT compare): {results_json_path}")


if __name__ == "__main__":
    main()
