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
DEFAULT_RENDER_DIR = DATA_ROOT / "spatiallm-dataset" / "region_bbox"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "spatiallm-dataset" / "region_bbox" / "example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one or more scenes into region_bbox/example/{scene_id}/. "
            "Each scene receives pcd.ply and gt_layout.txt. If available, "
            "raw_render.png and expanded_render.png are copied too."
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
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=DEFAULT_RENDER_DIR,
        help=(
            "Render root containing raw and expanded subdirs. The script accepts "
            "either raw/render/{scene_id}.png or raw/{scene_id}.png, and the "
            "same layout under expanded."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Deprecated: raw render directory. Overrides --render-dir/raw.",
    )
    parser.add_argument(
        "--expanded-dir",
        type=Path,
        default=None,
        help="Deprecated: expanded render directory. Overrides --render-dir/expanded.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--require-renders",
        action="store_true",
        help="Fail scenes that do not have both raw and expanded render images.",
    )
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
    }


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def render_files(args: argparse.Namespace, scene_id: str) -> dict[str, Path | None]:
    raw_dir = args.raw_dir or args.render_dir / "raw"
    expanded_dir = args.expanded_dir or args.render_dir / "expanded"
    return {
        "raw_render": first_existing_path(
            [
                raw_dir / "render" / f"{scene_id}.png",
                raw_dir / f"{scene_id}.png",
            ]
        ),
        "expanded_render": first_existing_path(
            [
                expanded_dir / "render" / f"{scene_id}.png",
                expanded_dir / f"{scene_id}.png",
            ]
        ),
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
    renders = render_files(args, scene_id)
    destinations = output_files(args, scene_id)

    missing = [f"{name}: {path}" for name, path in sources.items() if not path.is_file()]
    if args.require_renders:
        missing.extend(
            f"{name}: not found under {args.render_dir}"
            for name, path in renders.items()
            if path is None
        )
    if missing:
        return [f"missing {item}" for item in missing]

    copy_sources = {
        **sources,
        **{name: path for name, path in renders.items() if path is not None},
    }
    existing = [
        f"{name}: {path}"
        for name, path in destinations.items()
        if name in copy_sources and args.no_overwrite and path.exists()
    ]
    if existing:
        return [f"output exists {item}" for item in existing]

    out_dir = args.output_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, src in copy_sources.items():
        dst = destinations[name]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copyfile(src, dst)

    missing_renders = [name for name, path in renders.items() if path is None]
    if missing_renders:
        return [f"optional render missing: {', '.join(missing_renders)}"]

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
            optional_only = all(error.startswith("optional render missing") for error in errors)
            if optional_only:
                print(f"[OK] {scene_id} -> {args.output_root / scene_id}")
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
            else:
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
