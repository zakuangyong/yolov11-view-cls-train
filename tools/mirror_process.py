from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".tif",
    ".tiff",
}


MIRROR_DIR_MAP = {
    "back_left_side45": "back_right_side45",
    "front_left_side45": "front_right_side45",
    "left_side": "right_side",
}


@dataclass(frozen=True)
class MirrorStats:
    scanned: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    missing_dir: int = 0


def _iter_images(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def _build_mirror_writer() -> tuple[str, Callable[[Path, Path], None]]:
    try:
        from PIL import Image, ImageOps
    except Exception:
        Image = None
        ImageOps = None

    if Image is not None and ImageOps is not None:

        def _write_pillow(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = ImageOps.mirror(im)
                if dst.suffix.lower() in {".jpg", ".jpeg", ".jpe"} and im.mode in {
                    "RGBA",
                    "LA",
                    "P",
                }:
                    im = im.convert("RGB")
                im.save(dst)

        return "pillow", _write_pillow

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "No image backend found. Install `pillow` (recommended) or `opencv-python`."
        ) from e

    def _write_cv2(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        flipped = cv2.flip(img, 1)
        ok = cv2.imwrite(str(dst), flipped)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")

    return "opencv", _write_cv2


def mirror_dir(
    *,
    src_dir: Path,
    dst_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> MirrorStats:
    if not src_dir.exists():
        return MirrorStats(missing_dir=1)

    _, writer = _build_mirror_writer()

    scanned = 0
    written = 0
    skipped = 0
    failed = 0

    for src in _iter_images(src_dir):
        scanned += 1
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        try:
            if not dry_run:
                writer(src, dst)
            written += 1
        except Exception:
            failed += 1

    return MirrorStats(
        scanned=scanned, written=written, skipped=skipped, failed=failed, missing_dir=0
    )


def run_mirror_process(*, root: str, overwrite: bool = False, dry_run: bool = False) -> dict:
    root_dir = Path(root)
    if not root_dir.exists():
        raise FileNotFoundError(f"root not found: {root_dir}")

    backend, _ = _build_mirror_writer()
    print(f"backend={backend}")

    total = MirrorStats()
    per_dir: dict[str, MirrorStats] = {}

    for src_name, dst_name in MIRROR_DIR_MAP.items():
        src_dir = root_dir / src_name
        dst_dir = root_dir / dst_name
        stats = mirror_dir(
            src_dir=src_dir,
            dst_dir=dst_dir,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        per_dir[src_name] = stats
        total = MirrorStats(
            scanned=total.scanned + stats.scanned,
            written=total.written + stats.written,
            skipped=total.skipped + stats.skipped,
            failed=total.failed + stats.failed,
            missing_dir=total.missing_dir + stats.missing_dir,
        )

    for src_name, stats in per_dir.items():
        dst_name = MIRROR_DIR_MAP[src_name]
        print(
            " ".join(
                [
                    f"src={src_name}",
                    f"dst={dst_name}",
                    f"scanned={stats.scanned}",
                    f"written={stats.written}",
                    f"skipped={stats.skipped}",
                    f"failed={stats.failed}",
                    f"missing_dir={stats.missing_dir}",
                ]
            )
        )

    return {
        "root": str(root_dir),
        "backend": backend,
        "scanned": total.scanned,
        "written": total.written,
        "skipped": total.skipped,
        "failed": total.failed,
        "missing_dir": total.missing_dir,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mirror selected view directories to their right-side counterparts."
    )
    p.add_argument(
        "--root",
        type=str,
        default="./results/car-view-cls",
        help="Root directory that contains back_left_side45/front_left_side45/left_side",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images (default: skip if exists)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only count what would be processed, do not write files",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stats = run_mirror_process(
        root=args.root,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(
        " ".join(
            [
                f"scanned={stats['scanned']}",
                f"written={stats['written']}",
                f"skipped={stats['skipped']}",
                f"failed={stats['failed']}",
                f"missing_dir={stats['missing_dir']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
