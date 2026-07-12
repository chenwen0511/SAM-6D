import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_dataset = root / "user_data/yolo_dataset"
    parser = argparse.ArgumentParser(description="Convert Labelme polygon labels to YOLO segmentation format.")
    parser.add_argument("--labelme-dir", default=str(default_dataset / "labelme_json"))
    parser.add_argument("--image-dir", default=str(default_dataset / "raw_images"))
    parser.add_argument("--output-dir", default=str(default_dataset))
    parser.add_argument("--class-name", default="tray")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _find_image(json_path: Path, image_dir: Path, image_path_value: Optional[str]) -> Path:
    candidates: List[Path] = []
    if image_path_value:
        image_path = Path(image_path_value)
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append(json_path.parent / image_path)
            candidates.append(image_dir / image_path.name)

    for suffix in IMAGE_SUFFIXES:
        candidates.append(image_dir / f"{json_path.stem}{suffix}")

    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found for {json_path.name}")


def _shape_to_points(shape: Dict[str, object]) -> Optional[List[Tuple[float, float]]]:
    shape_type = str(shape.get("shape_type", "polygon"))
    raw_points = shape.get("points", [])
    if not isinstance(raw_points, list):
        return None

    points = [(float(point[0]), float(point[1])) for point in raw_points if len(point) >= 2]
    if shape_type == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if shape_type == "polygon" and len(points) >= 3:
        return points
    return None


def _normalise_points(points: Sequence[Tuple[float, float]], width: int, height: int) -> List[float]:
    values: List[float] = []
    for x, y in points:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        values.extend([nx, ny])
    return values


def _convert_one(json_path: Path, image_dir: Path, class_name: str) -> Tuple[Path, List[str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = _find_image(json_path, image_dir, data.get("imagePath"))
    with Image.open(image_path) as image:
        width, height = image.size

    label_lines: List[str] = []
    for shape in data.get("shapes", []):
        if str(shape.get("label")) != class_name:
            continue
        points = _shape_to_points(shape)
        if not points:
            continue
        normalised = _normalise_points(points, width, height)
        coords = " ".join(f"{value:.6f}" for value in normalised)
        label_lines.append(f"0 {coords}")

    return image_path, label_lines


def _clear_split_dirs(output_dir: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        path = output_dir / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _write_data_yaml(output_dir: Path, class_name: str) -> None:
    content = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def _copy_sample(image_path: Path, label_lines: Iterable[str], output_dir: Path, split: str) -> None:
    target_image = output_dir / "images" / split / image_path.name
    target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
    shutil.copy2(image_path, target_image)
    target_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    labelme_dir = Path(args.labelme_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    json_paths = sorted(labelme_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No Labelme JSON files found in {labelme_dir}")

    samples: List[Tuple[Path, List[str]]] = []
    skipped: List[str] = []
    for json_path in json_paths:
        image_path, label_lines = _convert_one(json_path, image_dir, args.class_name)
        if label_lines:
            samples.append((image_path, label_lines))
        else:
            skipped.append(json_path.name)

    if not samples:
        raise SystemExit(f"No '{args.class_name}' polygon labels found.")

    random.Random(args.seed).shuffle(samples)
    val_count = max(1, int(round(len(samples) * args.val_ratio))) if len(samples) > 1 else 0
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    if not train_samples:
        train_samples, val_samples = val_samples, []

    _clear_split_dirs(output_dir)
    for image_path, label_lines in train_samples:
        _copy_sample(image_path, label_lines, output_dir, "train")
    for image_path, label_lines in val_samples:
        _copy_sample(image_path, label_lines, output_dir, "val")
    _write_data_yaml(output_dir, args.class_name)

    print(f"Converted {len(samples)} samples to {output_dir}")
    print(f"train={len(train_samples)} val={len(val_samples)} skipped={len(skipped)}")
    if skipped:
        print("Skipped JSON without matching labels:")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
