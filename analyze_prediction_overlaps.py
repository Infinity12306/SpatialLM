#!/usr/bin/env python3
"""Analyze overlapping predicted object bboxes within each scene."""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from spatiallm import Layout
from spatiallm.layout.entity import Bbox


DEFAULT_PRED_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset/pred_hier/test/ckpt_12000/final"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset/eval/test/overlap"
)


@dataclass
class OverlapPair:
    scene_id: str
    bbox_i: int
    bbox_j: int
    class_i: str
    class_j: str
    iou_3d: float
    volume_i: float
    volume_j: float
    intersection_volume: float
    intersection_bev_area: float
    z_overlap: float


@dataclass
class SceneSummary:
    scene_id: str
    pred_path: Path
    pred_count: int
    overlap_pair_count: int
    overlap_ratio: float
    same_class_overlap_pair_count: int
    max_iou_3d: float
    mean_iou_3d: float
    max_intersection_volume: float
    mean_intersection_volume: float


def load_bboxes(path: Path, minimum_scale: float) -> list[Bbox]:
    layout = Layout(path.read_text(encoding="utf-8"))
    for bbox in layout.bboxes:
        bbox.scale_x = max(abs(bbox.scale_x), minimum_scale)
        bbox.scale_y = max(abs(bbox.scale_y), minimum_scale)
        bbox.scale_z = max(abs(bbox.scale_z), minimum_scale)
    return layout.bboxes


def footprint_polygon(bbox: Bbox) -> Polygon:
    hx = bbox.scale_x * 0.5
    hy = bbox.scale_y * 0.5
    local = np.array(
        [
            [-hx, -hy],
            [hx, -hy],
            [hx, hy],
            [-hx, hy],
        ],
        dtype=np.float64,
    )
    cos_angle = np.cos(bbox.angle_z)
    sin_angle = np.sin(bbox.angle_z)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        dtype=np.float64,
    )
    center = np.array([bbox.position_x, bbox.position_y], dtype=np.float64)
    corners = local @ rotation.T + center
    return Polygon(corners)


def z_interval(bbox: Bbox) -> tuple[float, float]:
    half_z = bbox.scale_z * 0.5
    return bbox.position_z - half_z, bbox.position_z + half_z


def volume(bbox: Bbox) -> float:
    return float(bbox.scale_x * bbox.scale_y * bbox.scale_z)


def compute_pair_overlap(
    scene_id: str,
    index_i: int,
    bbox_i: Bbox,
    index_j: int,
    bbox_j: Bbox,
) -> OverlapPair | None:
    polygon_i = footprint_polygon(bbox_i)
    polygon_j = footprint_polygon(bbox_j)
    if not polygon_i.is_valid or not polygon_j.is_valid:
        return None

    intersection_bev_area = float(polygon_i.intersection(polygon_j).area)
    z_min_i, z_max_i = z_interval(bbox_i)
    z_min_j, z_max_j = z_interval(bbox_j)
    z_overlap = max(0.0, min(z_max_i, z_max_j) - max(z_min_i, z_min_j))
    intersection_volume = intersection_bev_area * z_overlap

    volume_i = volume(bbox_i)
    volume_j = volume(bbox_j)
    intersection_volume = min(intersection_volume, volume_i, volume_j)
    union_volume = volume_i + volume_j - intersection_volume
    iou_3d = intersection_volume / union_volume if union_volume > 0 else 0.0
    iou_3d = min(max(iou_3d, 0.0), 1.0)

    return OverlapPair(
        scene_id=scene_id,
        bbox_i=index_i,
        bbox_j=index_j,
        class_i=bbox_i.class_name,
        class_j=bbox_j.class_name,
        iou_3d=float(iou_3d),
        volume_i=volume_i,
        volume_j=volume_j,
        intersection_volume=float(intersection_volume),
        intersection_bev_area=intersection_bev_area,
        z_overlap=float(z_overlap),
    )


def analyze_scene(
    pred_path: Path,
    output_dir: Path,
    minimum_scale: float,
    min_iou: float,
    min_intersection_volume: float,
    same_class_only: bool,
) -> tuple[SceneSummary, list[float]]:
    scene_id = pred_path.stem
    bboxes = load_bboxes(pred_path, minimum_scale)

    overlap_pairs: list[OverlapPair] = []
    for index_i, index_j in itertools.combinations(range(len(bboxes)), 2):
        bbox_i = bboxes[index_i]
        bbox_j = bboxes[index_j]
        if same_class_only and bbox_i.class_name != bbox_j.class_name:
            continue

        overlap = compute_pair_overlap(scene_id, index_i, bbox_i, index_j, bbox_j)
        if overlap is None:
            continue
        if overlap.iou_3d <= min_iou:
            continue
        if overlap.intersection_volume <= min_intersection_volume:
            continue
        overlap_pairs.append(overlap)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_overlap_pairs(output_dir / f"{scene_id}.csv", overlap_pairs)

    pair_count = len(overlap_pairs)
    pred_count = len(bboxes)
    ious = [pair.iou_3d for pair in overlap_pairs]
    same_class_ious = [
        pair.iou_3d for pair in overlap_pairs if pair.class_i == pair.class_j
    ]
    intersection_volumes = [pair.intersection_volume for pair in overlap_pairs]
    return (
        SceneSummary(
            scene_id=scene_id,
            pred_path=pred_path,
            pred_count=pred_count,
            overlap_pair_count=pair_count,
            overlap_ratio=pair_count / (pred_count**2) if pred_count > 0 else 0.0,
            same_class_overlap_pair_count=len(same_class_ious),
            max_iou_3d=max(ious) if ious else 0.0,
            mean_iou_3d=float(np.mean(ious)) if ious else 0.0,
            max_intersection_volume=(
                max(intersection_volumes) if intersection_volumes else 0.0
            ),
            mean_intersection_volume=(
                float(np.mean(intersection_volumes)) if intersection_volumes else 0.0
            ),
        ),
        same_class_ious,
    )


def write_overlap_pairs(path: Path, pairs: list[OverlapPair]) -> None:
    fieldnames = list(OverlapPair.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in sorted(
            pairs,
            key=lambda item: (
                -item.iou_3d,
                -item.intersection_volume,
                item.bbox_i,
                item.bbox_j,
            ),
        ):
            writer.writerow(pair.__dict__)


def write_summary(path: Path, summaries: list[SceneSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SceneSummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = summary.__dict__.copy()
            row["pred_path"] = str(row["pred_path"])
            writer.writerow(row)


def write_histogram_csv(
    path: Path,
    counts: np.ndarray,
    bin_edges: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        for left, right, count in zip(bin_edges[:-1], bin_edges[1:], counts):
            writer.writerow(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": int(count),
                }
            )


def write_histogram_png(
    path: Path,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    total_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1000, 620
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 70, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)

    max_count = int(counts.max()) if counts.size and counts.max() > 0 else 1
    bar_width = plot_width / max(len(counts), 1)
    for index, count in enumerate(counts):
        left = x0 + index * bar_width
        right = x0 + (index + 1) * bar_width
        top = y0 - (float(count) / max_count) * plot_height
        draw.rectangle(
            (left + 1, top, right - 1, y0),
            fill=(65, 118, 192),
            outline=(45, 80, 140),
        )

    for tick_value in np.linspace(0.0, 1.0, 6):
        x = x0 + tick_value * plot_width
        draw.line((x, y0, x, y0 + 6), fill="black", width=1)
        draw.text((x - 16, y0 + 12), f"{tick_value:.1f}", fill="black")

    for tick_value in np.linspace(0, max_count, 5):
        y = y0 - (tick_value / max_count) * plot_height
        draw.line((x0 - 6, y, x0, y), fill="black", width=1)
        draw.text((8, y - 8), str(int(round(tick_value))), fill="black")

    title = "Same-class Overlapping Prediction Pair IoU Histogram"
    subtitle = f"pairs={total_count}, bins={len(counts)}, range=[0, 1]"
    draw.text((margin_left, 20), title, fill="black")
    draw.text((margin_left, 42), subtitle, fill="black")
    draw.text((margin_left + plot_width // 2 - 30, height - 35), "3D IoU", fill="black")
    draw.text((10, margin_top - 25), "Count", fill="black")
    image.save(path)


def write_same_class_iou_histogram(
    ious: list[float],
    bins: int,
    csv_path: Path | None,
    png_path: Path | None,
) -> None:
    values = np.asarray(ious, dtype=np.float64)
    counts, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    if csv_path is not None:
        write_histogram_csv(csv_path, counts, bin_edges)
    if png_path is not None:
        write_histogram_png(png_path, counts, bin_edges, total_count=len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Analyze overlapping predicted object bboxes within each scene."
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Directory containing scene-level prediction txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-scene overlap CSV files.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "summary.csv",
        help="Path for the ranked summary CSV.",
    )
    parser.add_argument(
        "--minimum_scale",
        type=float,
        default=1e-6,
        help="Clamp each bbox dimension to at least this value before overlap.",
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.0,
        help="Only record pairs with 3D IoU strictly greater than this value.",
    )
    parser.add_argument(
        "--min_intersection_volume",
        type=float,
        default=1e-8,
        help="Only record pairs with intersection volume strictly greater than this value.",
    )
    parser.add_argument(
        "--same_class_only",
        action="store_true",
        help="Only consider pairs whose predicted category names are identical.",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=50,
        help="Number of bins for the same-class overlap IoU histogram.",
    )
    parser.add_argument(
        "--same_class_iou_histogram_png",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "same_class_iou_histogram.png",
        help="Path for the same-class overlap IoU histogram PNG.",
    )
    parser.add_argument(
        "--same_class_iou_histogram_csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "same_class_iou_histogram.csv",
        help="Path for the same-class overlap IoU histogram CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_paths = sorted(args.pred_dir.glob("*.txt"))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction txt files found in {args.pred_dir}")

    summaries: list[SceneSummary] = []
    same_class_ious: list[float] = []
    for pred_path in pred_paths:
        summary, scene_same_class_ious = analyze_scene(
            pred_path,
            args.output_dir,
            args.minimum_scale,
            args.min_iou,
            args.min_intersection_volume,
            args.same_class_only,
        )
        summaries.append(summary)
        same_class_ious.extend(scene_same_class_ious)

    summaries.sort(
        key=lambda item: (
            -item.overlap_ratio,
            -item.overlap_pair_count,
            -item.max_iou_3d,
            item.scene_id,
        )
    )
    write_summary(args.summary_csv, summaries)
    write_same_class_iou_histogram(
        same_class_ious,
        args.hist_bins,
        args.same_class_iou_histogram_csv,
        args.same_class_iou_histogram_png,
    )
    print(f"Wrote per-scene overlap CSVs to {args.output_dir}")
    print(f"Wrote summary CSV to {args.summary_csv}")
    print(
        "Wrote same-class IoU histogram to "
        f"{args.same_class_iou_histogram_png} and {args.same_class_iou_histogram_csv}"
    )


if __name__ == "__main__":
    main()
