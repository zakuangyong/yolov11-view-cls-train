from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


DATA_DIR = Path("./datasets/car-view-cls")
BASE_MODEL = Path("./models/yolo11m-cls.pt")
PROJECT_DIR = Path("./runs/classify")
EPOCHS = 50
IMGSZ = 224
BATCH = 64
WORKERS = 8
RUN_NAME = "car_view"
PRETRAINED = True
SAVE_PERIOD = 10
VERBOSE = True


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def _list_classes(train_dir: Path) -> list[str]:
    if not train_dir.is_dir():
        return []
    names = [p.name for p in train_dir.iterdir() if p.is_dir()]
    return sorted(names)


def train_car_view_cls(
    data_dir: str | Path = "./datasets/car-view-cls",
    base_model: str | Path = "./models/yolo11m-cls.pt",
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 64,
    workers: int = 8,
    device: str | int | None = None,
    project: str | Path = "./runs/classify",
    name: str = "car_view",
    pretrained: bool = True,
    save_period: int = 10,
    verbose: bool = True,
) -> None:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"验证集目录不存在: {val_dir}")

    classes = _list_classes(train_dir)
    if not classes:
        raise RuntimeError(f"训练集目录下未发现类别子目录: {train_dir}")

    val_classes = _list_classes(val_dir)
    missing = sorted(set(classes) - set(val_classes))
    if missing:
        raise FileNotFoundError(
            "验证集目录缺少以下类别子目录:\n"
            + "\n".join(f"- {name}" for name in missing)
            + f"\n请在 {val_dir} 下补齐对应类别目录。"
        )

    base_model = Path(base_model)
    if not base_model.exists():
        raise FileNotFoundError(
            f"基础模型不存在: {base_model}\n"
            "请确认 ./models/yolo11m-cls.pt 已放置到位，或用 --base_model 指定正确路径。"
        )

    if device is None:
        device = _default_device()

    model = YOLO(str(base_model))
    model.train(
        data=str(data_dir),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device=device,
        project=str(project),
        name=str(name),
        pretrained=bool(pretrained),
        save=True,
        save_period=int(save_period),
        verbose=bool(verbose),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO classifier for car view")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--base_model", type=str, default=str(BASE_MODEL))
    parser.add_argument("--epochs", type=int, default=int(EPOCHS))
    parser.add_argument("--imgsz", type=int, default=int(IMGSZ))
    parser.add_argument("--batch", type=int, default=int(BATCH))
    parser.add_argument("--workers", type=int, default=int(WORKERS))
    parser.add_argument("--device", type=str, default=None, help='e.g. "0" or "cpu" (default: auto)')
    parser.add_argument("--project", type=str, default=str(PROJECT_DIR))
    parser.add_argument("--name", type=str, default=str(RUN_NAME))
    parser.add_argument("--pretrained", action="store_true", default=bool(PRETRAINED))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--save_period", type=int, default=int(SAVE_PERIOD))
    parser.add_argument("--verbose", action="store_true", default=bool(VERBOSE))
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    args = parser.parse_args()

    device: str | int | None
    if args.device is None:
        device = _default_device()
    elif args.device.lower() == "cpu":
        device = "cpu"
    else:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    train_car_view_cls(
        data_dir=args.data_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        save_period=args.save_period,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
