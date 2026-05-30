#!/usr/bin/env python3
"""Gather point cloud, GT layout, and region renderings for scene examples."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DATA_ROOT = Path("/data2/chenjq24/SpatialLM")
DEFAULT_PCD_DIR = DATA_ROOT / "spatiallm-dataset-link" / "pcd"
DEFAULT_LAYOUT_DIR = DATA_ROOT / "spatiallm-dataset-link" / "layout"
DEFAULT_RAW_DIR = DATA_ROOT / "spatiallm-dataset" / "region_bbox" / "raw"
DEFAULT_EXPANDED_DIR = DATA_ROOT / "spatiallm-dataset" / "region_bbox" / "expanded"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "spatiallm-dataset" / "region_bbox" / "example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one or more scenes into region_bbox/example/{scene_id}/. "
            "Each scene receives pcd.ply, gt_layout.txt, raw_render.png, "
            "and expanded_render.png."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="+",
        help=(
            "Scene id(s), or path(s) to txt files containing one scene id per line."
        ),
    )
    parser.add_argument("--pcd-dir", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if an output file already exists.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue with exit code 0 even when a scene is missing files.",
    )
    return parser.parse_args()


def read_scene_ids(inputs: list[str]) -> list[str]:
    scene_ids: list[str] = []
    seen: set[str] = set()

    for value in inputs:
        path = Path(value)
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines()
            candidates = [line.strip() for line in lines]
        else:
            if path.suffix == ".txt":
                raise FileNotFoundError(f"Scene id file does not exist: {path}")
            candidates = [value.strip()]

        for scene_id in candidates:
            if not scene_id or scene_id.startswith("#"):
                continue
            if scene_id not in seen:
                scene_ids.append(scene_id)
                seen.add(scene_id)

    return scene_ids


def expected_files(args: argparse.Namespace, scene_id: str) -> dict[str, Path]:
    return {
        "pcd": args.pcd_dir / f"{scene_id}.ply",
        "gt_layout": args.layout_dir / f"{scene_id}.txt",
        "raw_render": args.raw_dir / "render" / f"{scene_id}.png",
        "expanded_render": args.expanded_dir / "render" / f"{scene_id}.png",
    }


def output_files(args: argparse.Namespace, scene_id: str) -> dict[str, Path]:
    out_dir = args.output_root / scene_id
    return {
        "pcd": out_dir / "pcd.ply",
        "gt_layout": out_dir / "gt_layout.txt",
        "raw_render": out_dir / "raw_render.png",
        "expanded_render": out_dir / "expanded_render.png",
    }


def gather_scene(args: argparse.Namespace, scene_id: str) -> list[str]:
    sources = expected_files(args, scene_id)
    destinations = output_files(args, scene_id)

    missing = [f"{name}: {path}" for name, path in sources.items() if not path.is_file()]
    if missing:
        return [f"missing {item}" for item in missing]

    existing = [
        f"{name}: {path}"
        for name, path in destinations.items()
        if args.no_overwrite and path.exists()
    ]
    if existing:
        return [f"output exists {item}" for item in existing]

    out_dir = args.output_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, src in sources.items():
        shutil.copy2(src, destinations[name])

    return []


def main() -> int:
    args = parse_args()
    scene_ids = read_scene_ids(args.scenes)
    if not scene_ids:
        print("No scene ids found.", file=sys.stderr)
        return 1

    failures: dict[str, list[str]] = {}
    for scene_id in scene_ids:
        errors = gather_scene(args, scene_id)
        if errors:
            failures[scene_id] = errors
            print(f"[FAIL] {scene_id}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            continue
        print(f"[OK] {scene_id} -> {args.output_root / scene_id}")

    if failures:
        print(
            f"Gathered {len(scene_ids) - len(failures)}/{len(scene_ids)} scenes; "
            f"{len(failures)} failed.",
            file=sys.stderr,
        )
        return 0 if args.allow_missing else 1

    print(f"Gathered {len(scene_ids)} scene(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
