"""
SAM3 text-prompt segmentation backend (subprocess).

Intermediate layout matches ``yolo_seg_backend``::

    {output_dir}/sam6d_results/detection_ism.json

Environment (optional)::

    SAM6D_SAM3_ROOT          default /home/ubuntu/stephen/01-code/sam3
    SAM6D_SAM3_PYTHON
    SAM6D_SAM3_INFER_SCRIPT  default {SAM3_ROOT}/scripts/infer.py
    SAM6D_SAM3_PROMPT
    SAM6D_SAM3_THRESHOLD
    SAM6D_SAM3_MASK_THRESHOLD
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

DEFAULT_SAM3_ROOT = "/home/ubuntu/stephen/01-code/sam3"
DEFAULT_SAM3_PYTHON = "/home/ubuntu/miniconda3/envs/sam3/bin/python"
DEFAULT_SAM3_INFER_SCRIPT = f"{DEFAULT_SAM3_ROOT}/scripts/infer.py"


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
        raise RuntimeError("SAM3 mask is empty")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def _load_mask_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    gray = np.array(Image.open(path).convert("L"))
    if gray.shape[0] != height or gray.shape[1] != width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
    return gray > 127


def _decode_rle_dict(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    size = rle["size"]
    h, w = int(size[0]), int(size[1])
    decoded = cocomask.decode({"counts": counts, "size": [h, w]})
    return decoded.astype(bool)


def _pick_from_pred_json(
    data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, List[int]]:
    scores = data.get("pred_scores") or data.get("scores")
    masks = data.get("pred_masks") or data.get("masks")
    boxes = data.get("pred_boxes") or data.get("bbox") or data.get("bboxes")

    if masks is None and "segmentation" in data:
        seg = data["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
            score = float(data.get("score", data.get("confidence", 1.0)))
            if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    image_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            return mask, score, _bbox_xywh_from_mask(mask)

    if isinstance(masks, list) and masks:
        idx = 0
        if scores is not None and len(scores):
            idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            score = float(scores[idx])
        else:
            score = float(data.get("confidence", 1.0))
        item = masks[idx]
        if isinstance(item, dict) and "counts" in item:
            mask = _decode_rle_dict(item)
        elif isinstance(item, (list, np.ndarray)):
            mask = np.asarray(item, dtype=bool)
        else:
            raise RuntimeError(f"unsupported mask entry type in SAM3 json: {type(item)}")
        if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                image_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(0.0 <= float(v) <= 1.0 for v in box):
                    w, h = image_size
                    x, y, bw, bh = [float(v) for v in box]
                    bbox = [int(x * w), int(y * h), int(bw * w), int(bh * h)]
                else:
                    bbox = [int(round(v)) for v in box]
            else:
                bbox = _bbox_xywh_from_mask(mask)
        else:
            bbox = _bbox_xywh_from_mask(mask)
        return mask, score, bbox

    raise RuntimeError("SAM3 json has no usable mask fields")


def _parse_sam3_output(work_dir: Path, rgb_path: Path) -> Tuple[np.ndarray, float, List[int]]:
    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)

    for jf in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            try:
                return _pick_from_pred_json(data, image_size)
            except RuntimeError:
                continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "segmentation" in data[0] or "bbox" in data[0]:
                det = max(data, key=lambda d: float(d.get("score", 0.0)))
                seg = det["segmentation"]
                mask = _decode_rle_dict(seg) if isinstance(seg, dict) else np.asarray(seg, dtype=bool)
                if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return mask, float(det.get("score", 1.0)), list(det.get("bbox", _bbox_xywh_from_mask(mask)))

    mask_candidates = [
        p
        for p in work_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "mask" in p.name.lower()
    ]
    if not mask_candidates:
        mask_candidates = list(work_dir.rglob("*.png"))
    if mask_candidates:
        path = sorted(mask_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mask = _load_mask_image(path, image_size)
        return mask, 1.0, _bbox_xywh_from_mask(mask)

    raise RuntimeError(f"no mask or json found under SAM3 output dir: {work_dir}")


def _sam3_root() -> Path:
    return Path(os.environ.get("SAM6D_SAM3_ROOT", DEFAULT_SAM3_ROOT)).expanduser().resolve()


def _sam3_python() -> str:
    return os.environ.get("SAM6D_SAM3_PYTHON", DEFAULT_SAM3_PYTHON)


def _sam3_infer_script() -> Path:
    override = os.environ.get("SAM6D_SAM3_INFER_SCRIPT")
    if override:
        return Path(override).expanduser().resolve()
    return (_sam3_root() / "scripts" / "infer.py").resolve()


def run_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    infer_script: Optional[Path] = None,
    python_exe: Optional[str] = None,
) -> Path:
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sam3_root = _sam3_root()
    script = (infer_script or _sam3_infer_script()).expanduser().resolve()
    py = python_exe or _sam3_python()
    prompt_text = prompt if prompt is not None else os.environ.get("SAM6D_SAM3_PROMPT", "white plate")
    thresh = (
        float(threshold)
        if threshold is not None
        else float(os.environ.get("SAM6D_SAM3_THRESHOLD", "0.41"))
    )
    mask_thresh = (
        float(mask_threshold)
        if mask_threshold is not None
        else float(os.environ.get("SAM6D_SAM3_MASK_THRESHOLD", "0.50"))
    )

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"

    cmd = [
        py,
        str(script),
        "--image",
        str(rgb_path),
        "--prompt",
        prompt_text,
        "--output-dir",
        str(output_dir),
        "--threshold",
        str(thresh),
        "--mask-threshold",
        str(mask_thresh),
    ]
    print(f"[sam3_seg_backend] sam3_root={sam3_root}")
    print(f"[sam3_seg_backend] cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    if os.environ.get("SAM6D_CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["SAM6D_CUDA_VISIBLE_DEVICES"]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(sam3_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[sam3_seg_backend] elapsed_ms={elapsed_ms:.3f} returncode={proc.returncode}")
    if proc.stdout:
        print("[sam3_seg_backend] stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-40:]))
    if proc.returncode != 0:
        raise RuntimeError(
            f"SAM3 infer failed with exit code {proc.returncode}\n"
            f"{proc.stdout[-4000:] if proc.stdout else ''}"
        )

    if json_path.is_file():
        dets = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(dets, list):
            raise RuntimeError(f"SAM3 detection_ism.json must be a list, got {type(dets).__name__}")
        print(f"[sam3_seg_backend] using {json_path} n_detections={len(dets)}")
        return json_path

    print(f"[sam3_seg_backend] {json_path} missing, fallback parse under {output_dir}")
    mask, score, bbox_xywh = _parse_sam3_output(output_dir, rgb_path)
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
    print(f"[sam3_seg_backend] wrote {json_path} score={score:.4f}")
    return json_path

