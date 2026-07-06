#!/usr/bin/env python3
"""Hierarchical inference with predicted regions and GT-assisted stage-2 filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from build_hierarchical_region_dataset import STAGE1_PROMPT, STAGE2_PROMPT
from build_stage2_evict_dataset import (
    boxes_from_layout,
    compute_voxel_scores,
    scene_layout_boxes,
    select_voxels,
    voxelize_points,
)
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    POINT_PROMPT,
    HierarchicalPrediction,
    RegionPrediction,
    apply_subset_args,
    center_crop_point_arrays,
    classwise_nms,
    clean_generated_text,
    decode_generated_layout,
    final_layout_from_parts,
    load_model_and_tokenizer,
    make_conversation,
    model_world_size,
    points_in_region,
    prepare_point_arrays,
    prepare_scene_point_cloud,
    scenes_from_json,
    write_prediction_outputs,
    write_region_debug_pcds,
)
from spatiallm import Layout
from spatiallm.layout.entity import Bbox, Region


LAYOUT_LABELS = {"wall", "door", "window"}
DEFAULT_ENCODER_STRIDES = (2, 2, 2, 2)


def prompt_with_point_token(prompt: str) -> str:
    return prompt.replace("<point_cloud>", POINT_PROMPT)


def generate_layout_text(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
    point_token_keep_bboxes: torch.Tensor | None = None,
) -> str:
    if args.seed >= 0:
        set_seed(args.seed)

    conversation = make_conversation(model, prompt)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": point_cloud,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": not args.greedy,
        "use_cache": True,
    }
    if point_token_keep_bboxes is not None:
        generate_kwargs["point_token_keep_bboxes"] = point_token_keep_bboxes
    if not args.greedy:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1] :]
    return clean_generated_text(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    )


def read_gt_layout_text(scene_id: str, layout_dir: Path) -> str:
    layout_path = layout_dir / f"{scene_id}.txt"
    if not layout_path.is_file():
        return ""
    return layout_path.read_text(encoding="utf-8")


def bbox_center_in_region(bbox: Bbox, region: Region) -> bool:
    center = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
    region_center = np.array([region.position_x, region.position_y, region.position_z])
    region_scale = np.array([region.scale_x, region.scale_y, region.scale_z])
    if np.any(region_scale <= 0):
        return False
    half_size = region_scale * 0.5
    return bool(np.all((center >= region_center - half_size) & (center <= region_center + half_size)))


def box_dict_center_in_region(box: dict, region: Region) -> bool:
    center = np.asarray(box["center"], dtype=np.float32)
    region_center = np.array([region.position_x, region.position_y, region.position_z])
    region_scale = np.array([region.scale_x, region.scale_y, region.scale_z])
    if np.any(region_scale <= 0):
        return False
    half_size = region_scale * 0.5
    return bool(np.all((center >= region_center - half_size) & (center <= region_center + half_size)))


def gt_bboxes_for_region(gt_layout: Layout, region: Region) -> list[Bbox]:
    return [bbox for bbox in gt_layout.bboxes if bbox_center_in_region(bbox, region)]


def gt_box_dicts_for_region(gt_layout_text: str, region: Region) -> list[dict]:
    return [
        box
        for box in boxes_from_layout(gt_layout_text, {"bbox"})
        if box_dict_center_in_region(box, region)
    ]


def keep_bboxes_from_gt_bboxes(
    bboxes: list[Bbox],
    min_extent: np.ndarray,
    expand_ratio: float,
) -> torch.Tensor:
    if not bboxes:
        return torch.empty((1, 0, 7), dtype=torch.float32)

    scale_multiplier = 1.0 + 2.0 * expand_ratio
    rows = []
    for bbox in bboxes:
        rows.append(
            [
                bbox.position_x - min_extent[0],
                bbox.position_y - min_extent[1],
                bbox.position_z - min_extent[2],
                abs(bbox.scale_x) * scale_multiplier,
                abs(bbox.scale_y) * scale_multiplier,
                abs(bbox.scale_z) * scale_multiplier,
                bbox.angle_z,
            ]
        )
    return torch.tensor([rows], dtype=torch.float32)


def evict_region_points(
    points: np.ndarray,
    colors: np.ndarray,
    object_boxes: list[dict],
    layout_boxes: list[dict],
    world_size: float,
    num_bins: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors

    pooling_factor = int(np.prod(args.encoder_strides))
    _, point_to_voxel, voxel_centers, point_token_cell_size = voxelize_points(
        points,
        world_size,
        num_bins,
        pooling_factor,
    )
    if voxel_centers.shape[0] <= args.target_pt_num:
        return points, colors

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
    device = torch.device(args.evict_device or args.device)

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
        args.distance_chunk_size,
    )
    keep_voxels, _ = select_voxels(
        object_distance_sum,
        layout_distance_sum,
        object_overlap,
        layout_overlap,
        args.target_pt_num,
    )
    keep_points = keep_voxels[point_to_voxel]
    if not np.any(keep_points):
        return points, colors
    return points[keep_points], colors[keep_points]


def predict_hierarchical_scene(
    scene,
    stage1_model,
    stage1_tokenizer,
    stage2_model,
    stage2_tokenizer,
    args: argparse.Namespace,
) -> HierarchicalPrediction:
    stage1_prompt = prompt_with_point_token(STAGE1_PROMPT)
    stage2_prompt = prompt_with_point_token(STAGE2_PROMPT)
    num_bins_stage1 = stage1_model.config.point_config["num_bins"]
    num_bins_stage2 = stage2_model.config.point_config["num_bins"]
    world_size_stage1 = model_world_size(stage1_model)
    world_size_stage2 = model_world_size(stage2_model)

    scene_pcd = prepare_scene_point_cloud(
        scene.pcd_path,
        num_bins_stage1,
        args.no_cleanup,
        world_size=world_size_stage1,
    )
    stage1_generated = generate_layout_text(
        stage1_model,
        stage1_tokenizer,
        stage1_prompt,
        scene_pcd.input_tensor,
        args,
    )
    stage1_layout = decode_generated_layout(
        stage1_generated,
        scene_pcd.min_extent,
        num_bins_stage1,
        world_size=world_size_stage1,
    )

    gt_layout_text = read_gt_layout_text(scene.scene_id, args.gt_layout_dir)
    gt_layout = Layout(gt_layout_text)
    layout_boxes = scene_layout_boxes(scene.scene_id, args.gt_layout_dir)

    region_predictions: list[RegionPrediction] = []
    all_bboxes: list[Bbox] = []
    for region_index, region in enumerate(stage1_layout.regions):
        mask = points_in_region(scene_pcd.points, region)
        region_points = scene_pcd.points[mask]
        region_colors = scene_pcd.colors[mask]

        if region_points.shape[0] < args.min_region_points:
            region_predictions.append(
                RegionPrediction(
                    index=region_index,
                    region=region,
                    point_count=int(region_points.shape[0]),
                    bboxes=[],
                    skipped=True,
                    skip_reason="too_few_points",
                    points=region_points,
                    colors=region_colors,
                )
            )
            continue

        region_points, region_colors = center_crop_point_arrays(
            region_points,
            region_colors,
            world_size_stage2,
        )

        if region_points.shape[0] < args.min_region_points:
            region_predictions.append(
                RegionPrediction(
                    index=region_index,
                    region=region,
                    point_count=int(region_points.shape[0]),
                    bboxes=[],
                    skipped=True,
                    skip_reason="too_few_points_after_center_crop",
                    points=region_points,
                    colors=region_colors,
                )
            )
            continue

        region_gt_bboxes = gt_bboxes_for_region(gt_layout, region)
        if args.evict_points:
            object_boxes = gt_box_dicts_for_region(gt_layout_text, region)
            region_points, region_colors = evict_region_points(
                region_points,
                region_colors,
                object_boxes,
                layout_boxes,
                world_size_stage2,
                num_bins_stage2,
                args,
            )

        region_pcd = prepare_point_arrays(
            region_points,
            region_colors,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        keep_bboxes = None
        if args.use_gt_bbox_mask:
            keep_bboxes = keep_bboxes_from_gt_bboxes(
                region_gt_bboxes,
                region_pcd.min_extent,
                args.bbox_mask_expand_ratio,
            )

        stage2_generated = generate_layout_text(
            stage2_model,
            stage2_tokenizer,
            stage2_prompt,
            region_pcd.input_tensor,
            args,
            point_token_keep_bboxes=keep_bboxes,
        )
        region_layout = decode_generated_layout(
            stage2_generated,
            region_pcd.min_extent,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        all_bboxes.extend(region_layout.bboxes)
        region_predictions.append(
            RegionPrediction(
                index=region_index,
                region=region,
                point_count=int(region_points.shape[0]),
                bboxes=region_layout.bboxes,
                skipped=False,
                points=region_points,
                colors=region_colors,
            )
        )

    all_bboxes = classwise_nms(all_bboxes, args.bbox_nms_iou)
    final_layout = final_layout_from_parts(stage1_layout, all_bboxes)
    return HierarchicalPrediction(
        scene_id=scene.scene_id,
        stage1_text=stage1_layout.to_language_string(),
        final_text=final_layout.to_language_string(),
        stage1_layout=stage1_layout,
        final_layout=final_layout,
        region_predictions=region_predictions,
    )


def parse_args(
    default_evict_points: bool = True,
    default_use_gt_bbox_mask: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hierarchical inference with predicted regions and GT-assisted stage-2 filtering"
    )
    parser.add_argument(
        "-i",
        "--data_json",
        type=Path,
        required=True,
        help="Stage1-style JSON dataset, e.g. spatiallm_stage1_region_test.json.",
    )
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt_layout_dir", type=Path, default=None)
    parser.add_argument(
        "--stage1_model_path",
        default="saves/hierarchical/stage1_regions",
    )
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes",
    )
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--min_region_points", type=int, default=1)
    parser.add_argument("--bbox_nms_iou", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--save_region_pcds", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--evict_points",
        action=argparse.BooleanOptionalAction,
        default=default_evict_points,
        help="Evict low-utility raw points from each predicted region before stage2.",
    )
    parser.add_argument(
        "--use_gt_bbox_mask",
        action=argparse.BooleanOptionalAction,
        default=default_use_gt_bbox_mask,
        help="Use GT object boxes to mask final point tokens before LLM generation.",
    )
    parser.add_argument("--bbox_mask_expand_ratio", type=float, default=0.1)
    parser.add_argument("--target_pt_num", type=int, default=1536)
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument(
        "--encoder_strides",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENCODER_STRIDES),
    )
    parser.add_argument("--object_min_overlap_scale", type=float, default=None)
    parser.add_argument("--layout_min_overlap_scale", type=float, default=None)
    parser.add_argument("--distance_chunk_size", type=int, default=8192)
    parser.add_argument("--evict_device", default=None)
    args = parser.parse_args()

    if args.gt_layout_dir is None:
        args.gt_layout_dir = args.dataset_root / "layout"
    if args.target_pt_num <= 0:
        parser.error("--target_pt_num must be positive.")
    if args.distance_chunk_size <= 0:
        parser.error("--distance_chunk_size must be positive.")
    if any(stride <= 0 for stride in args.encoder_strides):
        parser.error("--encoder_strides must be positive integers.")
    return args


def main(
    default_evict_points: bool = True,
    default_use_gt_bbox_mask: bool = False,
) -> int:
    args = parse_args(
        default_evict_points=default_evict_points,
        default_use_gt_bbox_mask=default_use_gt_bbox_mask,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_from_json(args.data_json, args.dataset_root)
    scenes = apply_subset_args(
        scenes,
        args.start_index,
        args.end_index,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    if not scenes:
        raise ValueError("No scenes found for inference.")
    if not args.gt_layout_dir.is_dir():
        raise NotADirectoryError(args.gt_layout_dir)

    stage1_model, stage1_tokenizer = load_model_and_tokenizer(
        args.stage1_model_path,
        args.inference_dtype,
        args.device,
    )
    stage2_model, stage2_tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Hierarchical pred-region inference"):
        final_path = args.output_dir / "final" / f"{scene.scene_id}.txt"
        stage1_path = args.output_dir / "stage1" / f"{scene.scene_id}.txt"
        if args.skip_existing and final_path.exists() and stage1_path.exists():
            continue

        try:
            prediction = predict_hierarchical_scene(
                scene,
                stage1_model,
                stage1_tokenizer,
                stage2_model,
                stage2_tokenizer,
                args,
            )
            write_prediction_outputs(prediction, args.output_dir)
            if args.save_region_pcds:
                write_region_debug_pcds(prediction, args.output_dir)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene.scene_id, str(exc)))
            error_dir = args.output_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / f"{scene.scene_id}.txt").write_text(str(exc), encoding="utf-8")

    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(f"Wrote hierarchical predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(default_evict_points=True, default_use_gt_bbox_mask=False))
