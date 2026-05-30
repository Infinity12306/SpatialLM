#!/usr/bin/env python3
"""Copy point clouds listed by scene_id in worst_test.csv."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


DEFAULT_CSV = Path("/data2/chenjq24/SpatialLM/spatiallm-dataset/worst_test.csv")
DEFAULT_SOURCE_DIR = Path("/nas1/chenjunqing2024/spatiallm-dataset/pcd")
DEFAULT_OUTPUT_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset/worst_pred_pcd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy .ply point clouds for every scene_id in a CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--source-pcd-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source <scene_id>.ply files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to copy .ply files into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser.parse_args()


def read_scene_ids(csv_path: Path) -> list[str]:
    scene_ids: list[str] = []
    seen: set[str] = set()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "scene_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a 'scene_id' column")

        for row in reader:
            scene_id = row["scene_id"].strip()
            if scene_id and scene_id not in seen:
                scene_ids.append(scene_id)
                seen.add(scene_id)

    return scene_ids


def main() -> int:
    args = parse_args()

    scene_ids = read_scene_ids(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    missing: list[Path] = []

    for scene_id in scene_ids:
        src = args.source_pcd_dir / f"{scene_id}.ply"
        dst = args.output_dir / src.name

        if not src.is_file():
            missing.append(src)
            print(f"missing: {src}", file=sys.stderr)
            continue

        if dst.exists() and not args.overwrite:
            skipped += 1
            print(f"skip existing: {dst}")
            continue

        print(f"copy: {src} -> {dst}")
        if not args.dry_run:
            shutil.copy2(src, dst)
        copied += 1

    print(
        f"done: {len(scene_ids)} scenes, {copied} copied, "
        f"{skipped} skipped, {len(missing)} missing"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
