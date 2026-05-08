import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List


THIS_DIR = Path(__file__).resolve().parent
PEM_DIR = THIS_DIR.parent
if str(PEM_DIR) not in sys.path:
    sys.path.insert(0, str(PEM_DIR))

from run_warmup_inference_custom import preload_default_pem_model, run_pose_inference  # noqa: E402


def _now_ns() -> int:
    return time.perf_counter_ns()


def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def _run_once(args) -> float:
    t0 = _now_ns()
    run_pose_inference(
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
    return _ns_to_ms(t1 - t0)


def _write_csv(csv_path: Path, rows: List[List[object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


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
    args = parser.parse_args()

    # 1) 先做模型预加载，避免首轮加载成本污染基线。
    preload_default_pem_model(gpus=args.gpus, verbose=True)

    # 2) 进行推理预热，稳定 CUDA kernel/allocator 状态。
    print(f"[baseline] warmup runs: {args.warmup_runs}")
    for i in range(args.warmup_runs):
        ms = _run_once(args)
        print(f"[baseline] warmup {i + 1}/{args.warmup_runs}: {ms:.3f} ms")

    # 3) 正式计时，多次调用取平均。
    print(f"[baseline] benchmark runs: {args.benchmark_runs}")
    samples_ms: List[float] = []
    for i in range(args.benchmark_runs):
        ms = _run_once(args)
        samples_ms.append(ms)
        print(f"[baseline] run {i + 1}/{args.benchmark_runs}: {ms:.3f} ms")

    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    p95_ms = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.95) - 1)]
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0

    rows: List[List[object]] = [
        ["metric", "value_ms"],
        ["mean", f"{mean_ms:.6f}"],
        ["median", f"{median_ms:.6f}"],
        ["p95", f"{p95_ms:.6f}"],
        ["min", f"{min_ms:.6f}"],
        ["max", f"{max_ms:.6f}"],
        ["std", f"{std_ms:.6f}"],
        [],
        ["run_index", "latency_ms"],
    ]
    for idx, ms in enumerate(samples_ms, start=1):
        rows.append([idx, f"{ms:.6f}"])

    csv_path = Path(args.csv_path).resolve()
    _write_csv(csv_path, rows)

    print("[baseline] done")
    print(f"[baseline] mean={mean_ms:.3f} ms median={median_ms:.3f} ms p95={p95_ms:.3f} ms")
    print(f"[baseline] csv saved: {csv_path}")


if __name__ == "__main__":
    main()
