from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


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


def _iter_images(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _dhash64(path: Path) -> tuple[int, tuple[int, int]]:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "Pillow not installed. Install it via `pip install pillow` or use --mode exact."
        ) from e

    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        im = im.resize((9, 8), resample=Image.Resampling.BILINEAR)
        if hasattr(im, "get_flattened_data"):
            px = list(im.get_flattened_data())
        else:
            px = list(im.getdata())

    bits = 0
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
    return bits, (w, h)


def _safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    stem = dst.stem
    suffix = dst.suffix
    for i in range(1, 10_000):
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
    raise RuntimeError(f"Too many name collisions under: {dst.parent}")


def _keep_key(path: Path, strategy: str) -> tuple:
    stat = path.stat()
    if strategy == "first":
        return (0,)
    if strategy == "largest":
        return (-stat.st_size,)
    if strategy == "smallest":
        return (stat.st_size,)
    if strategy == "newest":
        return (-stat.st_mtime_ns,)
    if strategy == "oldest":
        return (stat.st_mtime_ns,)
    if strategy == "shortest_name":
        return (len(path.name), path.name)
    if strategy == "longest_name":
        return (-len(path.name), path.name)
    return (0,)


@dataclass(frozen=True)
class DupItem:
    group_id: str
    kept: Path
    removed: Path
    reason: str


def _dedup_exact(files: list[Path], same_dir_only: bool, keep: str) -> tuple[list[Path], list[DupItem]]:
    by_key: dict[tuple[str, int], list[Path]] = {}
    for p in files:
        base = str(p.parent) if same_dir_only else "__all__"
        by_key.setdefault((base, p.stat().st_size), []).append(p)

    kept: list[Path] = []
    dups: list[DupItem] = []

    for (base, size), paths in by_key.items():
        if len(paths) == 1:
            kept.append(paths[0])
            continue

        hashes: dict[str, list[Path]] = {}
        for p in paths:
            hashes.setdefault(_sha256(p), []).append(p)

        for h, hp in hashes.items():
            if len(hp) == 1:
                kept.append(hp[0])
                continue

            hp_sorted = sorted(hp, key=lambda x: _keep_key(x, keep))
            keep_path = hp_sorted[0]
            kept.append(keep_path)

            gid = f"exact:{size}:{h[:16]}:{os.path.basename(base)}"
            for r in hp_sorted[1:]:
                dups.append(
                    DupItem(group_id=gid, kept=keep_path, removed=r, reason="sha256")
                )

    return kept, dups


class _BKNode:
    __slots__ = ("key", "idx", "children")

    def __init__(self, key: int, idx: int) -> None:
        self.key = key
        self.idx = idx
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, key: int, idx: int) -> None:
        if self.root is None:
            self.root = _BKNode(key, idx)
            return

        node = self.root
        while True:
            d = _hamming(key, node.key)
            child = node.children.get(d)
            if child is None:
                node.children[d] = _BKNode(key, idx)
                return
            node = child

    def search(self, key: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        out: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = _hamming(key, node.key)
            if d <= threshold:
                out.append(node.idx)
            lo = d - threshold
            hi = d + threshold
            for cd, child in node.children.items():
                if lo <= cd <= hi:
                    stack.append(child)
        return out


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _dedup_dhash(
    files: list[Path],
    threshold: int,
    same_dir_only: bool,
    keep: str,
) -> tuple[list[Path], list[DupItem]]:
    if threshold < 0 or threshold > 64:
        raise ValueError("threshold must be between 0 and 64")

    groups: dict[str, list[Path]]
    if same_dir_only:
        groups = {}
        for p in files:
            groups.setdefault(str(p.parent), []).append(p)
    else:
        groups = {"__all__": files}

    kept: list[Path] = []
    dups: list[DupItem] = []

    for base, paths in groups.items():
        if len(paths) <= 1:
            kept.extend(paths)
            continue

        dh: list[int] = []
        for p in paths:
            bits, _ = _dhash64(p)
            dh.append(bits)

        tree = _BKTree()
        dsu = _DSU(len(paths))
        for i, k in enumerate(dh):
            if tree.root is None:
                tree.add(k, i)
                continue
            hits = tree.search(k, threshold)
            for j in hits:
                if j != i and _hamming(k, dh[j]) <= threshold:
                    dsu.union(i, j)
            tree.add(k, i)

        clusters: dict[int, list[int]] = {}
        for i in range(len(paths)):
            clusters.setdefault(dsu.find(i), []).append(i)

        for root_idx, members in clusters.items():
            if len(members) == 1:
                kept.append(paths[members[0]])
                continue

            member_paths = [paths[i] for i in members]
            member_paths = sorted(member_paths, key=lambda x: _keep_key(x, keep))
            keep_path = member_paths[0]
            kept.append(keep_path)

            gid = f"dhash:{threshold}:{root_idx}:{os.path.basename(base)}"
            for r in member_paths[1:]:
                dups.append(
                    DupItem(group_id=gid, kept=keep_path, removed=r, reason="dhash")
                )

    return kept, dups


def _write_report(report_path: Path, items: list[DupItem]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "keep", "remove", "reason"])
        for it in items:
            w.writerow([it.group_id, str(it.kept), str(it.removed), it.reason])


def run_dedup(
    root: str | Path,
    mode: str,
    action: str,
    keep: str,
    threshold: int,
    cross_dir: bool,
    trash_dir: str | Path | None,
    report_path: str | Path | None,
) -> dict[str, int]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root dir not found: {root}")

    files = _iter_images(root)
    if not files:
        raise RuntimeError(f"No images found under: {root}")

    if mode == "exact":
        _, dup_items = _dedup_exact(files, same_dir_only=not cross_dir, keep=keep)
    elif mode == "dhash":
        _, dup_items = _dedup_dhash(
            files,
            threshold=threshold,
            same_dir_only=not cross_dir,
            keep=keep,
        )
    else:
        raise ValueError("mode must be one of: exact, dhash")

    if report_path is None:
        report_path = root / "dedup_report.csv"
    _write_report(Path(report_path), dup_items)

    removed_count = 0
    failed_count = 0
    if action == "report":
        removed_count = 0
    elif action == "delete":
        for it in dup_items:
            try:
                it.removed.unlink(missing_ok=True)
                removed_count += 1
            except Exception:
                failed_count += 1
    elif action == "move":
        if trash_dir is None:
            trash_dir = root / "_dedup_trash"
        trash_dir = Path(trash_dir)
        for it in dup_items:
            if not it.removed.exists():
                continue
            rel = it.removed.relative_to(root)
            dst = trash_dir / rel
            try:
                _safe_move(it.removed, dst)
                removed_count += 1
            except Exception:
                failed_count += 1
    else:
        raise ValueError("action must be one of: report, delete, move")

    return {
        "scanned": len(files),
        "duplicate_files": len(dup_items),
        "removed": removed_count,
        "failed": failed_count,
        "report": 1,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deduplicate images (exact by sha256, or similar by dHash)."
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to scan, e.g. ./results/car-view-cls",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="exact",
        choices=["exact", "dhash"],
        help="Dedup mode: exact (sha256) or dhash (perceptual)",
    )
    p.add_argument(
        "--action",
        type=str,
        default="report",
        choices=["report", "delete", "move"],
        help="What to do with duplicates",
    )
    p.add_argument(
        "--keep",
        type=str,
        default="first",
        choices=[
            "first",
            "largest",
            "smallest",
            "newest",
            "oldest",
            "shortest_name",
            "longest_name",
        ],
        help="Which file to keep within each duplicate group",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for dhash mode (0-64)",
    )
    p.add_argument(
        "--cross_dir",
        action="store_true",
        help="Dedup across all sub-directories under root (default: within each dir)",
    )
    p.add_argument(
        "--trash_dir",
        type=str,
        default=None,
        help="Trash directory for move action (default: <root>/_dedup_trash)",
    )
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Report CSV path (default: <root>/dedup_report.csv)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stats = run_dedup(
        root=args.root,
        mode=args.mode,
        action=args.action,
        keep=args.keep,
        threshold=args.threshold,
        cross_dir=bool(args.cross_dir),
        trash_dir=args.trash_dir,
        report_path=args.report,
    )
    print(
        " ".join(
            [
                f"scanned={stats['scanned']}",
                f"duplicates={stats['duplicate_files']}",
                f"removed={stats['removed']}",
                f"failed={stats['failed']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
