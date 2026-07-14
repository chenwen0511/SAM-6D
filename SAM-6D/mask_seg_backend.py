"""Convert user-provided binary mask PNG to detection_ism.json for PEM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = cocomask.encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _bbox_xywh_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("mask is empty")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def _reference_hw(depth_path: Path) -> Tuple[int, int]:
    depth = imageio.imread(str(depth_path))
    if depth.ndim < 2:
        raise RuntimeError(f"depth image must be 2D, got shape {depth.shape}")
    return int(depth.shape[0]), int(depth.shape[1])


def _load_binary_mask(mask_path: Path, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    gray = np.array(Image.open(mask_path).convert("L"))
    target_h, target_w = target_hw
    resized = False
    if gray.shape[0] != target_h or gray.shape[1] != target_w:
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized = True
    binary = gray > 127
    if not np.any(binary):
        raise RuntimeError("mask has no foreground pixels after binarization (threshold > 127)")
    return binary, resized


def _draw_overlay(rgb_path: Path, mask: np.ndarray, bbox_xywh: List[int], output_path: Path, score: float) -> None:
    image = np.array(Image.open(rgb_path).convert("RGB"))
    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    x, y, w, h = bbox_xywh
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"user_mask {score:.3f}",
        (x, max(0, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    Image.fromarray(overlay).save(output_path)


def run_user_mask_segmentation(
    mask_path: Path,
    depth_path: Path,
    output_dir: Path,
    *,
    rgb_path: Optional[Path] = None,
    score: float = 1.0,
    draw_vis: bool = True,
) -> Path:
    mask_path = mask_path.expanduser().resolve()
    depth_path = depth_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not mask_path.is_file():
        raise FileNotFoundError(f"mask not found: {mask_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"depth not found: {depth_path}")

    t0 = time.perf_counter()
    target_hw = _reference_hw(depth_path)
    mask, resized = _load_binary_mask(mask_path, target_hw)
    if resized:
        print(
            f"[mask_seg_backend] resized mask to depth shape HxW={target_hw[0]}x{target_hw[1]}"
        )

    bbox_xywh = _bbox_xywh_from_mask(mask)
    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"
    detection = {
        "scene_id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": bbox_xywh,
        "score": float(score),
        "time": 0.0,
        "segmentation": _mask_to_rle(mask),
    }
    json_path.write_text(json.dumps([detection]), encoding="utf-8")

    if draw_vis and rgb_path is not None and rgb_path.is_file():
        _draw_overlay(rgb_path, mask, bbox_xywh, sam6d_results / "vis_ism.png", float(score))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[mask_seg_backend] wrote {json_path} score={score:.4f} "
        f"bbox={bbox_xywh} elapsed_ms={elapsed_ms:.3f}"
    )
    return json_path
