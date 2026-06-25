#!/usr/bin/env python3
"""Create PLY point-cloud visualizations with layout, region, and bbox overlays."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_topdown_arkitscenes import load_points_and_colors
from spatiallm import Layout
from spatiallm.layout.entity import Bbox, Door, Region, Wall, Window


DEFAULT_PCD_DIR = Path("/data2/chenjq24/SpatialLM/spatiallm-dataset-link/pcd_test")
DEFAULT_GT_DIR = Path("/data2/chenjq24/SpatialLM/spatiallm-dataset-link/layout")
DEFAULT_GT_REGION_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/region_test/expanded"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset/visualization/ply"
)

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
RECT_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))

LAYOUT_COLORS = {
    "wall": (35, 35, 35),
    "door": (230, 170, 20),
    "window": (35, 160, 230),
    "region": (220, 30, 200),
}

HIGH_CONTRAST_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
    (86, 180, 233),
    (213, 94, 0),
    (240, 228, 66),
    (0, 0, 0),
    (178, 24, 43),
    (33, 102, 172),
    (102, 189, 99),
    (166, 118, 29),
    (117, 112, 179),
    (217, 95, 2),
]


@dataclass
class SourceSpec:
    label: str
    path: Path | None
    directory: Path | None
    elements: set[str]
    is_gt: bool = False


@dataclass
class LegendEntry:
    color: tuple[int, int, int]
    pred_count: int = 0
    gt_count: int = 0


def parse_elements(value: str | None, default: str) -> set[str]:
    text = value or default
    elements = {item.strip().lower() for item in text.split(",") if item.strip()}
    valid = {"layout", "bbox", "region"}
    invalid = elements - valid
    if invalid:
        raise ValueError(f"Invalid render element(s): {sorted(invalid)}")
    return elements


def read_scene_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def deterministic_category_color(category: str) -> tuple[int, int, int]:
    digest = hashlib.md5(category.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "little") / 65535.0
    saturation = 0.72
    value = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def scene_color_map(labels: set[str]) -> dict[str, tuple[int, int, int]]:
    color_map: dict[str, tuple[int, int, int]] = {}
    for index, label in enumerate(sorted(labels)):
        if index < len(HIGH_CONTRAST_COLORS):
            color_map[label] = HIGH_CONTRAST_COLORS[index]
            continue
        hue = ((index - len(HIGH_CONTRAST_COLORS)) * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
        color_map[label] = int(r * 255), int(g * 255), int(b * 255)
    return color_map


def collect_bbox_labels(layout: Layout, elements: set[str]) -> set[str]:
    if "bbox" not in elements:
        return set()
    return {f"bbox: {bbox.class_name}" for bbox in layout.bboxes}


def add_legend_count(
    legend: dict[str, LegendEntry],
    label: str,
    color: tuple[int, int, int],
    is_gt: bool,
) -> None:
    entry = legend.setdefault(label, LegendEntry(color=color))
    if is_gt:
        entry.gt_count += 1
    else:
        entry.pred_count += 1


def bbox_corners(bbox: Bbox) -> np.ndarray:
    hx = bbox.scale_x * 0.5
    hy = bbox.scale_y * 0.5
    hz = bbox.scale_z * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    cos_angle = np.cos(bbox.angle_z)
    sin_angle = np.sin(bbox.angle_z)
    rotation = np.array(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
    return local @ rotation.T + center


def region_corners(region: Region) -> np.ndarray:
    hx = region.scale_x * 0.5
    hy = region.scale_y * 0.5
    hz = region.scale_z * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    center = np.array([region.position_x, region.position_y, region.position_z])
    return local + center


def wall_corners(wall: Wall) -> np.ndarray:
    return np.array(
        [
            [wall.ax, wall.ay, wall.az],
            [wall.bx, wall.by, wall.bz],
            [wall.bx, wall.by, wall.bz + wall.height],
            [wall.ax, wall.ay, wall.az + wall.height],
        ],
        dtype=np.float64,
    )


def fixture_corners(fixture: Door | Window, wall_lookup: dict[int, Wall]) -> np.ndarray | None:
    wall = wall_lookup.get(fixture.wall_id)
    if wall is None:
        return None
    wall_start = np.array([wall.ax, wall.ay], dtype=np.float64)
    wall_end = np.array([wall.bx, wall.by], dtype=np.float64)
    wall_vec = wall_end - wall_start
    wall_length = np.linalg.norm(wall_vec)
    if wall_length < 1e-9:
        return None
    wall_xy_unit = wall_vec / wall_length
    center = np.array(
        [fixture.position_x, fixture.position_y, fixture.position_z],
        dtype=np.float64,
    )
    offset = 0.5 * np.concatenate(
        [wall_xy_unit * fixture.width, np.array([fixture.height])]
    )
    start = center - offset
    end = center + offset
    return np.array(
        [
            [start[0], start[1], start[2]],
            [end[0], end[1], start[2]],
            [end[0], end[1], end[2]],
            [start[0], start[1], end[2]],
        ],
        dtype=np.float64,
    )


def sample_segment(
    start: np.ndarray,
    end: np.ndarray,
    spacing: float,
    dashed: bool,
    dash_length: float,
) -> np.ndarray:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return start.reshape(1, 3)
    point_count = max(2, int(np.ceil(length / spacing)) + 1)
    distances = np.linspace(0.0, length, point_count)
    if dashed and dash_length > 0:
        keep_mask = (np.floor(distances / dash_length).astype(np.int64) % 2) == 0
        distances = distances[keep_mask]
        if distances.size == 0:
            distances = np.array([0.0])
    t = distances / length
    return start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]


def thicken_points(points: np.ndarray, thickness: float) -> np.ndarray:
    if thickness <= 0 or points.size == 0:
        return points
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [thickness, 0.0, 0.0],
            [-thickness, 0.0, 0.0],
            [0.0, thickness, 0.0],
            [0.0, -thickness, 0.0],
            [0.0, 0.0, thickness],
            [0.0, 0.0, -thickness],
        ],
        dtype=np.float64,
    )
    return (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def edge_points(
    corners: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    spacing: float,
    dashed: bool,
    dash_length: float,
    thickness: float,
) -> np.ndarray:
    segments = [
        sample_segment(corners[start], corners[end], spacing, dashed, dash_length)
        for start, end in edges
    ]
    if not segments:
        return np.empty((0, 3), dtype=np.float64)
    return thicken_points(np.concatenate(segments, axis=0), thickness)


def append_colored_points(
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    if points.size == 0:
        return
    point_chunks.append(points.astype(np.float64))
    color_chunks.append(
        np.repeat(np.array(color, dtype=np.uint8)[None, :], points.shape[0], axis=0)
    )


def add_layout_overlay(
    layout: Layout,
    elements: set[str],
    is_gt: bool,
    color_map: dict[str, tuple[int, int, int]],
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    legend: dict[str, LegendEntry],
    edge_spacing: float,
    dash_length: float,
    edge_thickness: float,
) -> None:
    dashed = is_gt
    if "bbox" in elements:
        for bbox in layout.bboxes:
            legend_label = f"bbox: {bbox.class_name}"
            color = color_map.get(legend_label, deterministic_category_color(bbox.class_name))
            points = edge_points(
                bbox_corners(bbox),
                BOX_EDGES,
                edge_spacing,
                dashed,
                dash_length,
                edge_thickness,
            )
            append_colored_points(point_chunks, color_chunks, points, color)
            add_legend_count(legend, legend_label, color, is_gt)

    if "region" in elements:
        for region in layout.regions:
            color = LAYOUT_COLORS["region"]
            points = edge_points(
                region_corners(region),
                BOX_EDGES,
                edge_spacing,
                dashed,
                dash_length,
                edge_thickness,
            )
            append_colored_points(point_chunks, color_chunks, points, color)
            add_legend_count(legend, "region", color, is_gt)

    if "layout" in elements:
        wall_lookup = {wall.id: wall for wall in layout.walls}
        for wall in layout.walls:
            color = LAYOUT_COLORS["wall"]
            points = edge_points(
                wall_corners(wall),
                RECT_EDGES,
                edge_spacing,
                dashed,
                dash_length,
                edge_thickness,
            )
            append_colored_points(point_chunks, color_chunks, points, color)
            add_legend_count(legend, "wall", color, is_gt)

        for fixture in layout.doors + layout.windows:
            corners = fixture_corners(fixture, wall_lookup)
            if corners is None:
                continue
            label = fixture.entity_label
            color = LAYOUT_COLORS[label]
            points = edge_points(
                corners,
                RECT_EDGES,
                edge_spacing,
                dashed,
                dash_length,
                edge_thickness,
            )
            append_colored_points(point_chunks, color_chunks, points, color)
            add_legend_count(legend, label, color, is_gt)


def resolve_txt_path(scene_id: str, path: Path | None, directory: Path | None) -> Path | None:
    if path is not None:
        return path
    if directory is None:
        return None
    candidate = directory / f"{scene_id}.txt"
    return candidate if candidate.exists() else None


def resolve_pcd_path(scene_id: str, pcd_path: Path | None, pcd_dir: Path) -> Path:
    if pcd_path is not None:
        return pcd_path
    return pcd_dir / f"{scene_id}.ply"


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    seed = int.from_bytes(hashlib.md5(seed_text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    comments: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0].astype(np.float32)
    vertex_data["y"] = points[:, 1].astype(np.float32)
    vertex_data["z"] = points[:, 2].astype(np.float32)
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["alpha"] = 255

    comment_text = "".join(f"comment {comment}\n" for comment in comments)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by visualize_scene_ply.py\n"
        f"{comment_text}"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii", errors="replace"))
        vertex_data.tofile(f)


def write_legend_png(
    path: Path,
    scene_id: str,
    legend: dict[str, LegendEntry],
    has_pred: bool,
    has_gt: bool,
) -> None:
    rows = sorted(legend.items(), key=lambda item: item[0])
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = text_font = small_font = ImageFont.load_default()

    row_height = 38
    width = 920
    height = max(180, 130 + row_height * len(rows))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), f"Scene: {scene_id}", fill="black", font=title_font)
    draw.text((24, 54), "PLY overlay legend", fill="black", font=text_font)
    style_parts = []
    if has_pred:
        style_parts.append("solid = prediction")
    if has_gt:
        style_parts.append("dashed = GT")
    style = ", ".join(style_parts) if style_parts else "no overlay entities"
    draw.text((24, 82), style, fill="black", font=small_font)

    y = 118
    draw.text((24, y), "Color", fill="black", font=small_font)
    draw.text((120, y), "Entity/category", fill="black", font=small_font)
    draw.text((650, y), "Pred", fill="black", font=small_font)
    draw.text((735, y), "GT", fill="black", font=small_font)
    y += 28

    for label, entry in rows:
        color = entry.color
        draw.rectangle((26, y + 4, 90, y + 30), fill=color, outline="black")
        draw.text((120, y + 6), label, fill="black", font=text_font)
        draw.text((660, y + 6), str(entry.pred_count), fill="black", font=text_font)
        draw.text((745, y + 6), str(entry.gt_count), fill="black", font=text_font)
        y += row_height

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_layout(path: Path | None) -> Layout | None:
    if path is None:
        return None
    return Layout(path.read_text(encoding="utf-8"))


LoadedSource = tuple[SourceSpec, Path, Layout]


def write_scene_view(
    scene_id: str,
    view_name: str,
    scene_dir: Path,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    loaded_sources: list[LoadedSource],
    color_map: dict[str, tuple[int, int, int]],
    args: argparse.Namespace,
    pcd_path: Path,
) -> None:
    point_chunks = [scene_points.astype(np.float64)]
    color_chunks = [scene_colors.astype(np.uint8)]
    legend: dict[str, LegendEntry] = {}
    comments = [
        f"scene_id {scene_id}",
        f"view {view_name}",
        f"point_cloud {pcd_path}",
    ]

    has_pred = any(not source.is_gt for source, _, _ in loaded_sources)
    has_gt = any(source.is_gt for source, _, _ in loaded_sources)
    for source, source_path, layout in loaded_sources:
        comments.append(
            f"{'gt' if source.is_gt else 'pred'}_{source.label}_txt {source_path}"
        )
        add_layout_overlay(
            layout,
            source.elements,
            source.is_gt,
            color_map,
            point_chunks,
            color_chunks,
            legend,
            args.edge_spacing,
            args.dash_length,
            args.edge_thickness,
        )

    if len(point_chunks) == 1:
        comments.append("warning no overlay txt file was found")

    all_points = np.concatenate(point_chunks, axis=0)
    all_colors = np.concatenate(color_chunks, axis=0)
    for label, entry in sorted(legend.items()):
        comments.append(
            f"legend {label} color {entry.color[0]} {entry.color[1]} {entry.color[2]}"
        )

    output_ply = scene_dir / f"{view_name}.ply"
    output_legend = scene_dir / f"{view_name}_legend.png"
    write_binary_ply(output_ply, all_points, all_colors, comments)
    write_legend_png(output_legend, scene_id, legend, has_pred=has_pred, has_gt=has_gt)


def visualize_scene(
    scene_id: str,
    args: argparse.Namespace,
    sources: list[SourceSpec],
) -> None:
    pcd_path = resolve_pcd_path(scene_id, args.pcd_path, args.pcd_dir)
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    scene_points, scene_colors = load_points_and_colors(pcd_path)
    scene_points, scene_colors = downsample_points(
        scene_points,
        scene_colors,
        args.max_scene_points,
        scene_id,
    )

    loaded_sources: list[LoadedSource] = []
    bbox_labels: set[str] = set()

    for source in sources:
        source_path = resolve_txt_path(scene_id, source.path, source.directory)
        if source_path is None:
            continue
        layout = load_layout(source_path)
        if layout is None:
            continue
        loaded_sources.append((source, source_path, layout))
        bbox_labels.update(collect_bbox_labels(layout, source.elements))

    color_map = scene_color_map(bbox_labels)
    pred_sources = [item for item in loaded_sources if not item[0].is_gt]
    gt_sources = [item for item in loaded_sources if item[0].is_gt]
    scene_dir = args.output_dir / scene_id

    write_scene_view(
        scene_id,
        "pred",
        scene_dir,
        scene_points,
        scene_colors,
        pred_sources if args.render_gt else loaded_sources,
        color_map,
        args,
        pcd_path,
    )
    written_views = ["pred"]

    if args.render_gt:
        write_scene_view(
            scene_id,
            "gt",
            scene_dir,
            scene_points,
            scene_colors,
            gt_sources,
            color_map,
            args,
            pcd_path,
        )
        write_scene_view(
            scene_id,
            "pred_with_GT",
            scene_dir,
            scene_points,
            scene_colors,
            loaded_sources,
            color_map,
            args,
            pcd_path,
        )
        written_views.extend(["gt", "pred_with_GT"])

    print(f"{scene_id}: wrote {', '.join(written_views)} to {scene_dir}")


def build_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    txt_elements = parse_elements(args.txt_elements, "bbox")
    stage1_elements = parse_elements(args.stage1_elements, "region")
    stage2_elements = parse_elements(args.stage2_elements, "bbox")
    gt_elements = parse_elements(args.gt_elements, "bbox")

    if args.txt_path is not None or args.txt_dir is not None:
        sources.append(
            SourceSpec("txt", args.txt_path, args.txt_dir, txt_elements, is_gt=False)
        )
    if args.stage1_path is not None or args.stage1_dir is not None:
        sources.append(
            SourceSpec(
                "stage1",
                args.stage1_path,
                args.stage1_dir,
                stage1_elements,
                is_gt=False,
            )
        )
    if args.stage2_path is not None or args.stage2_dir is not None:
        sources.append(
            SourceSpec(
                "stage2",
                args.stage2_path,
                args.stage2_dir,
                stage2_elements,
                is_gt=False,
            )
        )
    if args.render_gt:
        if "region" in gt_elements and args.gt_region_dir is not None:
            sources.append(
                SourceSpec("gt_region", None, args.gt_region_dir, {"region"}, is_gt=True)
            )
            gt_elements = set(gt_elements)
            gt_elements.discard("region")
        if gt_elements:
            sources.append(SourceSpec("gt", None, args.gt_dir, gt_elements, is_gt=True))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Create combined PLY files with scene points and layout/region/bbox overlays."
    )
    parser.add_argument("--scene_id", action="append", default=[])
    parser.add_argument("--scene_ids_file", type=Path)
    parser.add_argument("--pcd_path", type=Path)
    parser.add_argument("--pcd_dir", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--txt_path", type=Path, help="Generic single layout txt file.")
    parser.add_argument("--txt_dir", type=Path, help="Generic per-scene layout txt dir.")
    parser.add_argument(
        "--txt_elements",
        help="Comma-separated subset of layout,bbox,region for --txt_*; default: bbox.",
    )

    parser.add_argument("--stage1_path", type=Path)
    parser.add_argument("--stage1_dir", type=Path)
    parser.add_argument(
        "--stage1_elements",
        help=(
            "Comma-separated subset of layout,bbox,region for stage1 txt. "
            "Default: region."
        ),
    )

    parser.add_argument("--stage2_path", type=Path)
    parser.add_argument("--stage2_dir", type=Path)
    parser.add_argument(
        "--stage2_elements",
        help=(
            "Comma-separated subset of layout,bbox,region for stage2/final txt. "
            "Default: bbox."
        ),
    )

    parser.add_argument("--render_gt", action="store_true")
    parser.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--gt_region_dir", type=Path, default=DEFAULT_GT_REGION_DIR)
    parser.add_argument(
        "--gt_elements",
        help="Comma-separated subset of layout,bbox,region for GT; default: bbox.",
    )

    parser.add_argument("--edge_spacing", type=float, default=0.03)
    parser.add_argument("--edge_thickness", type=float, default=0.015)
    parser.add_argument("--dash_length", type=float, default=0.18)
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=0,
        help="Optional random downsample limit for scene points. 0 keeps all points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_ids = list(dict.fromkeys(args.scene_id + read_scene_ids(args.scene_ids_file)))
    if not scene_ids:
        if args.pcd_path is None:
            raise ValueError("Provide --scene_id/--scene_ids_file or --pcd_path.")
        scene_ids = [args.pcd_path.stem]
    if args.pcd_path is not None and len(scene_ids) != 1:
        raise ValueError("--pcd_path can only be used with one scene.")

    sources = build_sources(args)
    if not sources:
        raise ValueError("Provide at least one txt source or enable --render_gt.")

    for scene_id in scene_ids:
        visualize_scene(scene_id, args, sources)


if __name__ == "__main__":
    main()
