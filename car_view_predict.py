from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO


MODEL_PATH = Path("./models/yolo11m-cls.pt")
DATA_ROOT = Path("./datasets/car-view-cls")
SOURCE_DIR = DATA_ROOT / "test"
OUTPUT_DIR = Path("./results/car-view-cls")
IMGSZ = 224


IMAGE_SUFFIXES = {
    ".jpeg",
    ".jpg",
    ".jpe",
    ".png",
}


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def _resolve_existing_path(path: Path, fallbacks: list[Path]) -> Path:
    if path.exists():
        return path
    for fb in fallbacks:
        if fb.exists():
            return fb
    return path


def _list_classes_from_dataset(data_root: Path) -> list[str]:
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        return []
    names = [p.name for p in train_dir.iterdir() if p.is_dir()]
    return sorted(names)


def _safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = dst.stem
    suffix = dst.suffix
    for i in range(1, 10_000):
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"目标文件名冲突过多: {dst}")


def predict_and_split(
    model_path: str | Path = MODEL_PATH,
    source_dir: str | Path = SOURCE_DIR,
    data_root: str | Path = DATA_ROOT,
    output_dir: str | Path = OUTPUT_DIR,
    imgsz: int = IMGSZ,
    device: str | int | None = None,
) -> dict[str, int]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    source_dir = Path(source_dir)
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    data_root = _resolve_existing_path(
        data_root,
        fallbacks=[Path(str(data_root).replace("car-view.cls", "car-view-cls"))],
    )
    source_dir = _resolve_existing_path(
        source_dir,
        fallbacks=[data_root / "test"],
    )

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    if device is None:
        device = _default_device()

    model = YOLO(str(model_path))

    names: list[str] | None = None
    try:
        if hasattr(model, "names") and model.names:
            if isinstance(model.names, dict):
                names = [model.names[i] for i in range(len(model.names))]
            else:
                names = list(model.names)
    except Exception:
        names = None

    if not names:
        names = _list_classes_from_dataset(data_root)

    if not any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in source_dir.rglob("*")
    ):
        raise RuntimeError(f"No images found under: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    results_iter = model.predict(
        source=str(source_dir),
        imgsz=int(imgsz),
        device=device,
        stream=True,
        verbose=False,
    )

    for result in results_iter:
        raw_path = getattr(result, "path", "")
        src_path = Path(raw_path) if raw_path else Path()
        if not src_path.exists() and raw_path:
            candidate = source_dir / Path(raw_path).name
            if candidate.exists():
                src_path = candidate
        if not src_path.exists():
            continue

        probs = getattr(result, "probs", None)
        if probs is None:
            pred_idx = -1
        else:
            pred_idx = int(getattr(probs, "top1", -1))

        if names and 0 <= pred_idx < len(names):
            cls_name = str(names[pred_idx])
        else:
            cls_name = f"class_{pred_idx}" if pred_idx >= 0 else "unknown"

        dst_path = output_dir / cls_name / src_path.name
        _safe_copy(src_path, dst_path)
        counts[cls_name] += 1

    return dict(counts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify images with a YOLO classifier and copy them into class folders."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Model weights path, e.g. ./runs/classify/car_view/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(SOURCE_DIR),
        help="Images directory, e.g. ./datasets/car-view-cls/test",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DATA_ROOT),
        help="Dataset root (used to infer class names if model has no names)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory, e.g. ./results/car-view-cls",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(IMGSZ),
        help="Inference image size (classification), default 224",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device, e.g. "0" or "cpu" (default: auto)',
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device: str | int | None
    if args.device is None:
        device = None
    elif args.device.lower() == "cpu":
        device = "cpu"
    else:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    counts = predict_and_split(
        model_path=args.model,
        source_dir=args.source,
        data_root=args.data_root,
        output_dir=args.out_dir,
        imgsz=args.imgsz,
        device=device,
    )

    total = sum(counts.values())
    print(f"Done. Copied {total} images.")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
