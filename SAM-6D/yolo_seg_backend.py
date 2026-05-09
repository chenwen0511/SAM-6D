import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


_MODEL_CACHE: Dict[str, YOLO] = {}


def _load_model(weights_path: Path) -> YOLO:
    key = str(weights_path.resolve())
    if key not in _MODEL_CACHE:
        if not weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    counts: List[int] = []
    last_elem = 0
    running_length = 0
    for elem in mask.ravel(order="F"):
        if int(elem) == last_elem:
            running_length += 1
        else:
            counts.append(running_length)
            running_length = 1
            last_elem = int(elem)
    counts.append(running_length)
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
    model = _load_model(weights_path)
    results = model.predict(str(rgb_path), conf=conf, imgsz=imgsz, verbose=False)
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
    with Image.open(rgb_path) as image:
        image_size = image.size
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
    _draw_overlay(rgb_path, mask, bbox_xywh, sam6d_results / "vis_yolo_seg.png", score)
    _draw_overlay(rgb_path, mask, bbox_xywh, sam6d_results / "vis_ism.png", score)
    return json_path
