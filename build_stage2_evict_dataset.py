#!/usr/bin/env python3
"""Build stage-2 point-cloud-evicted hierarchical datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from render_topdown_arkitscenes import load_points_and_colors, save_points_and_colors
from spatiallm import Layout


DEFAULT_DATASET_ROOT = Path("/data2/chenjq24/SpatialLM/spatiallm-dataset-link")
DEFAULT_NUM_BINS = 1280
DEFAULT_ENCODER_STRIDES = (2, 2, 2, 2)
DEFAULT_TARGET_PT_NUM = 1536
DEFAULT_SPLITS = ("train_20000", "val", "test")
LAYOUT_LABELS = {"wall", "door", "window"}


@dataclass(frozen=True)
class SplitSpec:
    key: str
    input_json: Path
    input_region_dir: Path
    output_json: Path
    output_region_dir: Path
    dataset_name: str


@dataclass
class EvictStats:
    split: str
    sample_index: int
    scene_id: str
    input_point_cloud: str
    output_point_cloud: str
    raw_point_count: int
    kept_point_count: int
    raw_voxel_count: int
    kept_voxel_count: int
    target_pt_num: int
    object_box_count: int
    layout_box_count: int
    object_overlap_voxel_count: int
    layout_overlap_voxel_count: int
    protected_object_voxel_count: int
    initial_layout_discarded_voxel_count: int
    score_discarded_voxel_count: int
    fallback_used: bool
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evict low-utility raw points from hierarchical stage-2 region PCDs "
            "using GT object/layout boxes, then write _evict stage-2 JSON files."
        )
    )
    parser.add_argument(
        "--dataset_root",
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--layout_dir",
        "--layout-dir",
        type=Path,
        default=None,
        help="GT scene layout directory. Defaults to <dataset_root>/layout.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
        help="Stage-2 datasets to process.",
    )
    parser.add_argument(
        "--target_pt_num",
        "--target-pt-num",
        type=int,
        default=DEFAULT_TARGET_PT_NUM,
        help="Target number of final point tokens/coarse voxels to keep.",
    )
    parser.add_argument("--num_bins", "--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument(
        "--world_size",
        "--world-size",
        type=float,
        default=32.0,
        help=(
            "World size used by point-token voxelization. Use 16 for res-16 "
            "stage-2 training and 32 for the original setting."
        ),
    )
    parser.add_argument(
        "--encoder_strides",
        "--encoder-strides",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENCODER_STRIDES),
        help="Point-encoder pooling strides. SpatialLM1.1 uses: 2 2 2 2.",
    )
    parser.add_argument(
        "--object_min_overlap_scale",
        "--object-min-overlap-scale",
        type=float,
        default=None,
        help=(
            "Minimum scale in meters for object-overlap protection. Defaults "
            "to one final point-token cell size."
        ),
    )
    parser.add_argument(
        "--layout_min_overlap_scale",
        "--layout-min-overlap-scale",
        type=float,
        default=None,
        help=(
            "Minimum scale in meters for layout-overlap discard. Defaults to "
            "one final point-token cell size so zero-thickness walls are handled."
        ),
    )
    parser.add_argument(
        "--distance_chunk_size",
        "--distance-chunk-size",
        type=int,
        default=8192,
        help="Number of voxel centers scored per torch chunk.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for distance scoring. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split sample limit for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing evicted PCD files.",
    )
    parser.add_argument(
        "--copy_auxiliary",
        "--copy-auxiliary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy raw/expanded region txts and scene id files to *_evict dirs.",
    )
    args = parser.parse_args()

    if args.target_pt_num <= 0:
        parser.error("--target_pt_num must be positive.")
    if args.num_bins <= 0:
        parser.error("--num_bins must be positive.")
    if args.world_size <= 0:
        parser.error("--world_size must be positive.")
    if args.distance_chunk_size <= 0:
        parser.error("--distance_chunk_size must be positive.")
    if any(stride <= 0 for stride in args.encoder_strides):
        parser.error("--encoder_strides must be positive integers.")
    return args


def split_specs(dataset_root: Path, selected_splits: list[str]) -> list[SplitSpec]:
    mapping = {
        "train_20000": (
            "spatiallm_stage2_bbox_train_20000.json",
            "region_20000",
            "spatiallm_stage2_bbox_train_20000_evict.json",
            "region_20000_evict",
            "spatiallm_stage2_bbox_train_20000_evict",
        ),
        "val": (
            "spatiallm_stage2_bbox_val.json",
            "region_val",
            "spatiallm_stage2_bbox_val_evict.json",
            "region_val_evict",
            "spatiallm_stage2_bbox_val_evict",
        ),
        "test": (
            "spatiallm_stage2_bbox_test.json",
            "region_test",
            "spatiallm_stage2_bbox_test_evict.json",
            "region_test_evict",
            "spatiallm_stage2_bbox_test_evict",
        ),
    }

    specs = []
    for key in selected_splits:
        input_json, input_region, output_json, output_region, dataset_name = mapping[key]
        specs.append(
            SplitSpec(
                key=key,
                input_json=dataset_root / input_json,
                input_region_dir=dataset_root / input_region,
                output_json=dataset_root / output_json,
                output_region_dir=dataset_root / output_region,
                dataset_name=dataset_name,
            )
        )
    return specs


def clean_layout_text(text: str) -> str:
    return text.replace("<|layout_s|>", "").replace("<|layout_e|>", "").strip()


def assistant_text(item: dict[str, Any]) -> str:
    for message in item.get("conversations", []):
        if message.get("from") == "gpt":
            return str(message.get("value", ""))
    return ""


def point_cloud_path(item: dict[str, Any]) -> str:
    point_clouds = item.get("point_clouds")
    if isinstance(point_clouds, str):
        return point_clouds
    if isinstance(point_clouds, list) and point_clouds:
        return str(point_clouds[0])
    raise ValueError("sample has no point_clouds entry")


def base_scene_id_from_region_path(point_cloud: str) -> str:
    stem = Path(point_cloud).stem
    if "_region_" in stem:
        return stem.rsplit("_region_", 1)[0]
    return stem


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def relative_dataset_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return os.path.relpath(path, dataset_root)


def update_item_point_cloud(item: dict[str, Any], new_point_cloud: str) -> dict[str, Any]:
    new_item = deepcopy(item)
    point_clouds = new_item.get("point_clouds")
    if isinstance(point_clouds, str):
        new_item["point_clouds"] = new_point_cloud
    else:
        new_item["point_clouds"] = [new_point_cloud]
    return new_item


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary_path.replace(path)


def copy_auxiliary_region_files(input_region_dir: Path, output_region_dir: Path) -> None:
    for child_name in ("raw", "expanded"):
        source_dir = input_region_dir / child_name
        target_dir = output_region_dir / child_name
        if not source_dir.is_dir() or target_dir.exists():
            continue
        shutil.copytree(source_dir, target_dir)

    for file_name in ("selected_scene_ids.txt", "combined_scene_ids.txt"):
        source_path = input_region_dir / file_name
        target_path = output_region_dir / file_name
        if source_path.is_file() and not target_path.exists():
            shutil.copy2(source_path, target_path)


def boxes_from_layout(
    layout_text: str,
    labels: set[str] | None = None,
) -> list[dict[str, np.ndarray | str]]:
    layout = Layout(clean_layout_text(layout_text))
    boxes = []
    for box in layout.to_boxes():
        label = str(box["class"])
        if labels is not None and label not in labels:
            continue
        scale = np.asarray(box["scale"], dtype=np.float32)
        if np.any(scale < 0):
            scale = np.abs(scale)
        boxes.append(
            {
                "label": label,
                "center": np.asarray(box["center"], dtype=np.float32),
                "rotation": np.asarray(box["rotation"], dtype=np.float32),
                "scale": scale,
            }
        )
    return boxes


def scene_layout_boxes(scene_id: str, layout_dir: Path) -> list[dict[str, Any]]:
    layout_path = layout_dir / f"{scene_id}.txt"
    if not layout_path.is_file():
        return []
    return boxes_from_layout(layout_path.read_text(encoding="utf-8"), LAYOUT_LABELS)


def object_boxes_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    return boxes_from_layout(assistant_text(item), {"bbox"})


def voxelize_points(
    points: np.ndarray,
    world_size: float,
    num_bins: int,
    pooling_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    grid_size = world_size / num_bins
    point_token_cell_size = grid_size * pooling_factor
    min_bound = points.min(axis=0)
    shifted_points = points - min_bound
    fine_grid = np.floor(shifted_points / grid_size).astype(np.int64)
    fine_grid = np.clip(fine_grid, 0, num_bins - 1)
    coarse_grid = fine_grid // pooling_factor
    unique_coarse_grid, point_to_voxel = np.unique(
        coarse_grid,
        axis=0,
        return_inverse=True,
    )
    voxel_centers = (
        min_bound[None, :]
        + (unique_coarse_grid.astype(np.float32) + 0.5) * point_token_cell_size
    )
    return unique_coarse_grid, point_to_voxel, voxel_centers, point_token_cell_size


def box_tensors(
    boxes: list[dict[str, Any]],
    device: torch.device,
    min_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not boxes:
        empty = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_rotation = torch.empty((0, 3, 3), dtype=torch.float32, device=device)
        return empty, empty_rotation, empty

    centers = np.stack([box["center"] for box in boxes], axis=0).astype(np.float32)
    rotations = np.stack([box["rotation"] for box in boxes], axis=0).astype(np.float32)
    scales = np.stack([box["scale"] for box in boxes], axis=0).astype(np.float32)
    scales = np.abs(scales)
    if min_scale is not None and min_scale > 0:
        scales = np.maximum(scales, float(min_scale))
    half_sizes = scales * 0.5

    return (
        torch.from_numpy(centers).to(device=device),
        torch.from_numpy(rotations).to(device=device),
        torch.from_numpy(half_sizes).to(device=device),
    )


def outside_line_lengths(
    voxel_centers: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    half_sizes: torch.Tensor,
) -> torch.Tensor:
    if centers.shape[0] == 0:
        return torch.empty(
            (voxel_centers.shape[0], 0),
            dtype=voxel_centers.dtype,
            device=voxel_centers.device,
        )

    rel_world = voxel_centers[:, None, :] - centers[None, :, :]
    local = torch.einsum("vbi,bij->vbj", rel_world, rotations)
    abs_local = local.abs()
    denom = abs_local.clamp_min(1e-6)
    t_axis = torch.where(
        abs_local > half_sizes[None, :, :],
        1.0 - half_sizes[None, :, :] / denom,
        torch.zeros_like(abs_local),
    )
    t_enter = t_axis.max(dim=-1).values.clamp(0.0, 1.0)
    center_dist = torch.linalg.norm(rel_world, dim=-1)
    return t_enter * center_dist


def compute_voxel_scores(
    voxel_centers_np: np.ndarray,
    object_boxes: list[dict[str, Any]],
    layout_boxes: list[dict[str, Any]],
    object_min_overlap_scale: float,
    layout_min_overlap_scale: float,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voxel_count = voxel_centers_np.shape[0]
    object_distance_sum = np.zeros(voxel_count, dtype=np.float32)
    layout_distance_sum = np.zeros(voxel_count, dtype=np.float32)
    object_overlap = np.zeros(voxel_count, dtype=bool)
    layout_overlap = np.zeros(voxel_count, dtype=bool)

    obj_centers, obj_rot, obj_half = box_tensors(object_boxes, device, min_scale=None)
    obj_centers_ov, obj_rot_ov, obj_half_ov = box_tensors(
        object_boxes,
        device,
        min_scale=object_min_overlap_scale,
    )
    layout_centers, layout_rot, layout_half = box_tensors(
        layout_boxes,
        device,
        min_scale=None,
    )
    layout_centers_ov, layout_rot_ov, layout_half_ov = box_tensors(
        layout_boxes,
        device,
        min_scale=layout_min_overlap_scale,
    )

    with torch.no_grad():
        for start in range(0, voxel_count, chunk_size):
            stop = min(start + chunk_size, voxel_count)
            voxel_chunk = torch.from_numpy(
                voxel_centers_np[start:stop].astype(np.float32, copy=False)
            ).to(device=device)

            if obj_centers.shape[0] > 0:
                obj_dist = outside_line_lengths(
                    voxel_chunk,
                    obj_centers,
                    obj_rot,
                    obj_half,
                )
                object_distance_sum[start:stop] = (
                    obj_dist.sum(dim=1).detach().cpu().numpy()
                )
                obj_overlap_dist = outside_line_lengths(
                    voxel_chunk,
                    obj_centers_ov,
                    obj_rot_ov,
                    obj_half_ov,
                )
                object_overlap[start:stop] = (
                    obj_overlap_dist.min(dim=1).values <= 1e-6
                ).detach().cpu().numpy()

            if layout_centers.shape[0] > 0:
                layout_dist = outside_line_lengths(
                    voxel_chunk,
                    layout_centers,
                    layout_rot,
                    layout_half,
                )
                layout_distance_sum[start:stop] = (
                    layout_dist.sum(dim=1).detach().cpu().numpy()
                )
                layout_overlap_dist = outside_line_lengths(
                    voxel_chunk,
                    layout_centers_ov,
                    layout_rot_ov,
                    layout_half_ov,
                )
                layout_overlap[start:stop] = (
                    layout_overlap_dist.min(dim=1).values <= 1e-6
                ).detach().cpu().numpy()

    return object_distance_sum, layout_distance_sum, object_overlap, layout_overlap


def select_voxels(
    object_distance_sum: np.ndarray,
    layout_distance_sum: np.ndarray,
    object_overlap: np.ndarray,
    layout_overlap: np.ndarray,
    target_pt_num: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    voxel_count = object_distance_sum.shape[0]
    protected = object_overlap.copy()
    keep = np.ones(voxel_count, dtype=bool)

    layout_discard = layout_overlap & ~protected
    keep[layout_discard] = False

    score_discard_count = 0
    fallback_used = False
    if int(keep.sum()) > target_pt_num:
        score = object_distance_sum - layout_distance_sum
        candidates = np.flatnonzero(keep & ~protected)
        protected_kept_count = int((keep & protected).sum())
        keep_candidate_count = max(0, target_pt_num - protected_kept_count)

        if candidates.size > keep_candidate_count:
            if keep_candidate_count == 0:
                selected_candidates = np.empty((0,), dtype=np.int64)
            else:
                order = np.argsort(score[candidates], kind="stable")
                selected_candidates = candidates[order[:keep_candidate_count]]

            new_keep = protected & keep
            new_keep[selected_candidates] = True
            score_discard_count = int(keep.sum() - new_keep.sum())
            keep = new_keep

    if not np.any(keep) and voxel_count > 0:
        score = object_distance_sum - layout_distance_sum
        keep[int(np.argmin(score))] = True
        fallback_used = True

    stats = {
        "protected_object_voxel_count": int(protected.sum()),
        "initial_layout_discarded_voxel_count": int(layout_discard.sum()),
        "score_discarded_voxel_count": score_discard_count,
        "fallback_used": fallback_used,
    }
    return keep, stats


def evict_one_sample(
    item: dict[str, Any],
    split: str,
    sample_index: int,
    dataset_root: Path,
    layout_dir: Path,
    output_region_dir: Path,
    target_pt_num: int,
    world_size: float,
    num_bins: int,
    pooling_factor: int,
    object_min_overlap_scale: float,
    layout_min_overlap_scale: float,
    device: torch.device,
    distance_chunk_size: int,
    overwrite: bool,
) -> tuple[dict[str, Any], EvictStats]:
    input_point_cloud = point_cloud_path(item)
    input_pcd_path = resolve_path(input_point_cloud, dataset_root)
    scene_id = base_scene_id_from_region_path(input_point_cloud)
    output_pcd_path = output_region_dir / "pcd" / input_pcd_path.name
    output_point_cloud = relative_dataset_path(output_pcd_path, dataset_root)
    output_item = update_item_point_cloud(item, output_point_cloud)

    common = {
        "split": split,
        "sample_index": sample_index,
        "scene_id": scene_id,
        "input_point_cloud": input_point_cloud,
        "output_point_cloud": output_point_cloud,
        "target_pt_num": target_pt_num,
    }

    try:
        if output_pcd_path.exists() and not overwrite:
            points, _ = load_points_and_colors(output_pcd_path)
            stat = EvictStats(
                **common,
                raw_point_count=-1,
                kept_point_count=int(points.shape[0]),
                raw_voxel_count=-1,
                kept_voxel_count=-1,
                object_box_count=-1,
                layout_box_count=-1,
                object_overlap_voxel_count=-1,
                layout_overlap_voxel_count=-1,
                protected_object_voxel_count=-1,
                initial_layout_discarded_voxel_count=-1,
                score_discarded_voxel_count=-1,
                fallback_used=False,
                status="exists",
            )
            return output_item, stat

        points, colors = load_points_and_colors(input_pcd_path)
        if points.shape[0] == 0:
            raise ValueError(f"empty point cloud: {input_pcd_path}")

        object_boxes = object_boxes_from_item(item)
        layout_boxes = scene_layout_boxes(scene_id, layout_dir)

        _, point_to_voxel, voxel_centers, _ = voxelize_points(
            points,
            world_size,
            num_bins,
            pooling_factor,
        )
        raw_voxel_count = int(voxel_centers.shape[0])

        (
            object_distance_sum,
            layout_distance_sum,
            object_overlap,
            layout_overlap,
        ) = compute_voxel_scores(
            voxel_centers,
            object_boxes,
            layout_boxes,
            object_min_overlap_scale,
            layout_min_overlap_scale,
            device,
            distance_chunk_size,
        )
        keep_voxels, selection_stats = select_voxels(
            object_distance_sum,
            layout_distance_sum,
            object_overlap,
            layout_overlap,
            target_pt_num,
        )
        keep_points = keep_voxels[point_to_voxel]
        kept_points = points[keep_points]
        kept_colors = colors[keep_points]

        output_pcd_path.parent.mkdir(parents=True, exist_ok=True)
        save_points_and_colors(output_pcd_path, kept_points, kept_colors)

        stat = EvictStats(
            **common,
            raw_point_count=int(points.shape[0]),
            kept_point_count=int(kept_points.shape[0]),
            raw_voxel_count=raw_voxel_count,
            kept_voxel_count=int(keep_voxels.sum()),
            object_box_count=len(object_boxes),
            layout_box_count=len(layout_boxes),
            object_overlap_voxel_count=int(object_overlap.sum()),
            layout_overlap_voxel_count=int(layout_overlap.sum()),
            protected_object_voxel_count=int(
                selection_stats["protected_object_voxel_count"]
            ),
            initial_layout_discarded_voxel_count=int(
                selection_stats["initial_layout_discarded_voxel_count"]
            ),
            score_discarded_voxel_count=int(
                selection_stats["score_discarded_voxel_count"]
            ),
            fallback_used=bool(selection_stats["fallback_used"]),
            status="ok",
        )
        return output_item, stat
    except Exception as exc:
        stat = EvictStats(
            **common,
            raw_point_count=-1,
            kept_point_count=-1,
            raw_voxel_count=-1,
            kept_voxel_count=-1,
            object_box_count=-1,
            layout_box_count=-1,
            object_overlap_voxel_count=-1,
            layout_overlap_voxel_count=-1,
            protected_object_voxel_count=-1,
            initial_layout_discarded_voxel_count=-1,
            score_discarded_voxel_count=-1,
            fallback_used=False,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        return output_item, stat


def write_stats_csv(path: Path, rows: list[EvictStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(EvictStats.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    temporary_path.replace(path)


def update_dataset_info(dataset_root: Path, specs: list[SplitSpec]) -> None:
    info_path = dataset_root / "dataset_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    for spec in specs:
        info[spec.dataset_name] = {
            "file_name": spec.output_json.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        }

    write_json(info_path, info)


def process_split(
    spec: SplitSpec,
    args: argparse.Namespace,
    layout_dir: Path,
    device: torch.device,
    pooling_factor: int,
    point_token_cell_size: float,
) -> None:
    if not spec.input_json.is_file():
        raise FileNotFoundError(spec.input_json)
    if not spec.input_region_dir.is_dir():
        raise NotADirectoryError(spec.input_region_dir)

    spec.output_region_dir.mkdir(parents=True, exist_ok=True)
    (spec.output_region_dir / "pcd").mkdir(parents=True, exist_ok=True)
    if args.copy_auxiliary:
        copy_auxiliary_region_files(spec.input_region_dir, spec.output_region_dir)

    items = load_json_list(spec.input_json)
    if args.limit is not None:
        items = items[: args.limit]

    object_min_overlap_scale = (
        point_token_cell_size
        if args.object_min_overlap_scale is None
        else args.object_min_overlap_scale
    )
    layout_min_overlap_scale = (
        point_token_cell_size
        if args.layout_min_overlap_scale is None
        else args.layout_min_overlap_scale
    )

    output_items: list[dict[str, Any]] = []
    stats: list[EvictStats] = []
    for sample_index, item in enumerate(items):
        output_item, stat = evict_one_sample(
            item=item,
            split=spec.key,
            sample_index=sample_index,
            dataset_root=args.dataset_root,
            layout_dir=layout_dir,
            output_region_dir=spec.output_region_dir,
            target_pt_num=args.target_pt_num,
            world_size=args.world_size,
            num_bins=args.num_bins,
            pooling_factor=pooling_factor,
            object_min_overlap_scale=object_min_overlap_scale,
            layout_min_overlap_scale=layout_min_overlap_scale,
            device=device,
            distance_chunk_size=args.distance_chunk_size,
            overwrite=args.overwrite,
        )
        output_items.append(output_item)
        stats.append(stat)

        if (sample_index + 1) % 100 == 0 or sample_index + 1 == len(items):
            ok_count = sum(row.status in {"ok", "exists"} for row in stats)
            error_count = sum(row.status == "error" for row in stats)
            print(
                f"[{spec.key}] {sample_index + 1}/{len(items)} "
                f"processed, ok_or_exists={ok_count}, errors={error_count}"
            )

    write_json(spec.output_json, output_items)
    stats_path = spec.output_region_dir / "eviction_stats.csv"
    write_stats_csv(stats_path, stats)
    metadata = {
        "split": spec.key,
        "input_json": str(spec.input_json),
        "output_json": str(spec.output_json),
        "input_region_dir": str(spec.input_region_dir),
        "output_region_dir": str(spec.output_region_dir),
        "target_pt_num": args.target_pt_num,
        "world_size": args.world_size,
        "num_bins": args.num_bins,
        "encoder_strides": args.encoder_strides,
        "pooling_factor": pooling_factor,
        "point_token_cell_size": point_token_cell_size,
        "object_min_overlap_scale": object_min_overlap_scale,
        "layout_min_overlap_scale": layout_min_overlap_scale,
        "device": str(device),
        "sample_count": len(output_items),
        "status_counts": {
            "ok": sum(row.status == "ok" for row in stats),
            "exists": sum(row.status == "exists" for row in stats),
            "error": sum(row.status == "error" for row in stats),
        },
        "stats_csv": str(stats_path),
    }
    write_json(spec.output_region_dir / "eviction_metadata.json", metadata)

    error_count = metadata["status_counts"]["error"]
    if error_count:
        raise RuntimeError(
            f"{spec.key} completed with {error_count} error(s). "
            f"See {stats_path}."
        )

    print(f"[{spec.key}] wrote evicted PCDs to {spec.output_region_dir / 'pcd'}")
    print(f"[{spec.key}] wrote evicted JSON to {spec.output_json}")
    print(f"[{spec.key}] wrote stats to {stats_path}")


def main() -> None:
    args = parse_args()
    layout_dir = args.layout_dir or args.dataset_root / "layout"
    if not layout_dir.is_dir():
        raise NotADirectoryError(layout_dir)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pooling_factor = math.prod(args.encoder_strides)
    point_token_cell_size = args.world_size / args.num_bins * pooling_factor

    print(f"dataset_root={args.dataset_root}")
    print(f"layout_dir={layout_dir}")
    print(f"splits={','.join(args.splits)}")
    print(f"target_pt_num={args.target_pt_num}")
    print(f"world_size={args.world_size}")
    print(f"num_bins={args.num_bins}")
    print(f"pooling_factor={pooling_factor}")
    print(f"point_token_cell_size={point_token_cell_size}")
    print(f"device={device}")

    specs = split_specs(args.dataset_root, args.splits)
    for spec in specs:
        process_split(
            spec=spec,
            args=args,
            layout_dir=layout_dir,
            device=device,
            pooling_factor=pooling_factor,
            point_token_cell_size=point_token_cell_size,
        )

    update_dataset_info(args.dataset_root, specs)
    print(f"Updated {args.dataset_root / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
