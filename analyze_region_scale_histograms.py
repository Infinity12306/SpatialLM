#!/usr/bin/env python3
"""Analyze expanded region scale distributions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REGION_PATTERN = re.compile(r"^\s*([^=\s]+)\s*=\s*Region\((.*?)\)\s*$")


@dataclass(frozen=True)
class RegionScale:
    source_dir: str
    file_path: str
    scene_id: str
    region_name: str
    scale_x: float
    scale_y: float
    scale_z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Create scale_x/scale_y/scale_z histograms from expanded region bbox files."
    )
    parser.add_argument(
        "--region_dirs",
        "--region-dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Region directories. Each can be a region root containing an expanded/ "
            "subdirectory, or the expanded directory itself."
        ),
    )
    parser.add_argument(
        "--expanded_subdir",
        "--expanded-subdir",
        default="expanded",
        help="Expanded-region subdirectory name under each region root.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for histogram PNG/CSV/JSON outputs.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--range",
        dest="hist_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional shared histogram range for all scale axes.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Write density histograms instead of raw counts.",
    )
    return parser.parse_args()


def resolve_expanded_dir(region_dir: Path, expanded_subdir: str) -> Path:
    expanded_dir = region_dir / expanded_subdir
    if expanded_dir.is_dir():
        return expanded_dir
    return region_dir


def iter_region_files(region_dirs: Iterable[Path], expanded_subdir: str) -> Iterable[tuple[Path, Path]]:
    for region_dir in region_dirs:
        expanded_dir = resolve_expanded_dir(region_dir, expanded_subdir)
        if not expanded_dir.is_dir():
            raise FileNotFoundError(f"Region directory not found: {expanded_dir}")
        for path in sorted(expanded_dir.glob("*.txt")):
            yield region_dir, path


def parse_region_line(line: str) -> tuple[str, tuple[float, float, float]] | None:
    match = REGION_PATTERN.match(line)
    if match is None:
        return None

    region_name, body = match.groups()
    values = [float(part.strip()) for part in body.split(",")]
    if len(values) == 6:
        scale_x, scale_y, scale_z = values[3:6]
    elif len(values) == 7:
        scale_x, scale_y, scale_z = values[4:7]
    else:
        raise ValueError(f"Expected 6 or 7 Region values, got {len(values)}: {line}")

    return region_name, (abs(scale_x), abs(scale_y), abs(scale_z))


def load_region_scales(region_dirs: list[Path], expanded_subdir: str) -> list[RegionScale]:
    rows: list[RegionScale] = []
    for source_dir, path in iter_region_files(region_dirs, expanded_subdir):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parsed = parse_region_line(line)
            if parsed is None:
                continue
            region_name, scales = parsed
            rows.append(
                RegionScale(
                    source_dir=str(source_dir),
                    file_path=str(path),
                    scene_id=path.stem,
                    region_name=region_name,
                    scale_x=scales[0],
                    scale_y=scales[1],
                    scale_z=scales[2],
                )
            )
    return rows


def histogram(values: np.ndarray, bins: int, hist_range: tuple[float, float] | None, density: bool) -> dict:
    counts, edges = np.histogram(values, bins=bins, range=hist_range, density=density)
    return {
        "bin_edges": edges.tolist(),
        "values": counts.tolist(),
    }


def stats(values: np.ndarray) -> dict:
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {
            str(p): float(np.percentile(values, p)) for p in percentiles
        },
    }


def write_region_scale_csv(path: Path, rows: list[RegionScale]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_dir",
                "file_path",
                "scene_id",
                "region_name",
                "scale_x",
                "scale_y",
                "scale_z",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_histogram_csv(path: Path, histograms: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["histogram", "bin_index", "bin_left", "bin_right", "value"],
        )
        writer.writeheader()
        for name, hist in histograms.items():
            edges = hist["bin_edges"]
            values = hist["values"]
            for index, value in enumerate(values):
                writer.writerow(
                    {
                        "histogram": name,
                        "bin_index": index,
                        "bin_left": edges[index],
                        "bin_right": edges[index + 1],
                        "value": value,
                    }
                )


def write_histogram_png(path: Path, arrays: dict[str, np.ndarray], bins: int, hist_range, density: bool) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (name, values) in zip(axes.ravel(), arrays.items()):
        axis.hist(values, bins=bins, range=hist_range, density=density)
        axis.set_title(name)
        axis.set_xlabel("Region scale")
        axis.set_ylabel("Density" if density else "Count")
        axis.grid(True, alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_region_scales(args.region_dirs, args.expanded_subdir)
    if not rows:
        raise ValueError("No Region entries found in the provided region directories.")

    scale_x = np.array([row.scale_x for row in rows], dtype=np.float64)
    scale_y = np.array([row.scale_y for row in rows], dtype=np.float64)
    scale_z = np.array([row.scale_z for row in rows], dtype=np.float64)
    arrays = {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scale_z": scale_z,
        "all_scales": np.concatenate([scale_x, scale_y, scale_z]),
    }
    hist_range = tuple(args.hist_range) if args.hist_range is not None else None

    histograms = {
        name: histogram(values, args.bins, hist_range, args.density)
        for name, values in arrays.items()
    }
    summary = {
        "region_dirs": [str(path) for path in args.region_dirs],
        "expanded_subdir": args.expanded_subdir,
        "region_count": len(rows),
        "scene_count": len({row.scene_id for row in rows}),
        "bins": args.bins,
        "range": list(hist_range) if hist_range is not None else None,
        "density": args.density,
        "stats": {name: stats(values) for name, values in arrays.items()},
        "histograms": histograms,
    }

    (args.output_dir / "region_scale_histograms.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_region_scale_csv(args.output_dir / "region_scales.csv", rows)
    write_histogram_csv(args.output_dir / "region_scale_histograms.csv", histograms)
    write_histogram_png(
        args.output_dir / "region_scale_histograms.png",
        arrays,
        args.bins,
        hist_range,
        args.density,
    )

    print(f"Parsed {len(rows)} regions from {summary['scene_count']} scenes.")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
