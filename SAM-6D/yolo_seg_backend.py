import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
from ultralytics import YOLO


_MODEL_CACHE: Dict[str, YOLO] = {}


def _load_model(weights_path: Path) -> YOLO:
    key = str(weights_path.resolve())
    print(f"[yolo_seg_backend] load request: {weights_path} (resolved: {key})")
    if key not in _MODEL_CACHE:
        if not weights_path.is_file():
            print(f"[yolo_seg_backend] weights not found: {weights_path}")
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        print(f"[yolo_seg_backend] loading YOLO model: {key}")
        # Force segmentation task to avoid TRT engine auto-guess as detect.
        _MODEL_CACHE[key] = YOLO(key, task="segment")
        print(f"[yolo_seg_backend] model loaded and cached: {key}")
    else:
        print(f"[yolo_seg_backend] model cache hit: {key}")
    return _MODEL_CACHE[key]


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = cocomask.encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _xyxy_to_xywh(box: np.ndarray) -> List[int]:
    x1, y1, x2, y2 = box.tolist()
    return [
        int(round(x1)),
        int(round(y1)),
        int(round(max(0.0, x2 - x1))),
        int(round(max(0.0, y2 - y1))),
    ]


def _resize_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def _draw_overlay(rgb_path: Path, mask: np.ndarray, bbox_xywh: List[int], output_path: Path, score: float) -> None:
    image = np.array(Image.open(rgb_path).convert("RGB"))
    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

    x, y, w, h = bbox_xywh
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"tray {score:.3f}",
        (x, max(0, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    Image.fromarray(overlay).save(output_path)


def run_yolo_segmentation(
    weights_path: Path,
    rgb_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    class_id: Optional[int] = 0,
) -> Path:
    t0 = time.perf_counter()
    t_load0 = time.perf_counter()
    model = _load_model(weights_path)
    t_load1 = time.perf_counter()
    print(f"[yolo_seg_backend] _load_model elapsed_ms={(t_load1 - t_load0) * 1000:.3f}")

    t_pred0 = time.perf_counter()
    results = model.predict(str(rgb_path), conf=conf, imgsz=imgsz, verbose=False, task="segment")
    t_pred1 = time.perf_counter()
    print(f"[yolo_seg_backend] model.predict elapsed_ms={(t_pred1 - t_pred0) * 1000:.3f}")
    print(f"[yolo_seg_backend] load+predict elapsed_ms={(t_pred1 - t0) * 1000:.3f}")
    if not results:
        raise RuntimeError("YOLO returned no results")

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO returned no segmentation masks")

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy()

    candidate_indexes = list(range(len(scores)))
    if class_id is not None:
        candidate_indexes = [idx for idx in candidate_indexes if classes[idx] == class_id]
    if not candidate_indexes:
        raise RuntimeError(f"YOLO returned no masks for class_id={class_id}")

    best_idx = max(candidate_indexes, key=lambda idx: float(scores[idx]))
    # with Image.open(rgb_path) as image:
    #     image_size = image.size
    image_size = (480, 640)
    mask = _resize_mask(masks[best_idx], image_size)
    bbox_xywh = _xyxy_to_xywh(boxes[best_idx])
    score = float(scores[best_idx])

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"
    detection = {
        "scene_id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": bbox_xywh,
        "score": score,
        "time": 0.0,
        "segmentation": _mask_to_rle(mask),
    }
    json_path.write_text(json.dumps([detection]), encoding="utf-8")
    # _draw_overlay(rgb_path, mask, bbox_xywh, sam6d_results / "vis_yolo_seg.png", score)
    # _draw_overlay(rgb_path, mask, bbox_xywh, sam6d_results / "vis_ism.png", score)
    return json_path


def preload_yolo_model(
    weights_path: Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    class_id: Optional[int] = 0,
) -> None:
    """Preload YOLO model into cache and run one dummy warmup predict."""
    t0 = time.perf_counter()
    model = _load_model(weights_path)
    t1 = time.perf_counter()
    print(f"[yolo_seg_backend] preload _load_model elapsed_ms={(t1 - t0) * 1000:.3f}")

    # Dummy image with fixed camera shape (H, W, C) = 640x480x3.
    dummy = np.zeros((640, 480, 3), dtype=np.uint8)
    t2 = time.perf_counter()
    results = model.predict(dummy, conf=conf, imgsz=imgsz, verbose=False, task="segment")
    t3 = time.perf_counter()
    print(f"[yolo_seg_backend] preload warmup predict elapsed_ms={(t3 - t2) * 1000:.3f}")

    if results:
        result = results[0]
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"[yolo_seg_backend] preload warmup detections={n} class_filter={class_id}")
    print(f"[yolo_seg_backend] preload total elapsed_ms={(t3 - t0) * 1000:.3f}")

