#!/usr/bin/env python3
"""Oracle stage-2 hierarchical inference on pre-filtered region PCDs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from build_hierarchical_region_dataset import STAGE2_PROMPT
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    POINT_PROMPT,
    clean_generated_text,
    classwise_nms,
    decode_generated_layout,
    load_gt_region_layout,
    load_model_and_tokenizer,
    make_conversation,
    model_world_size,
    prepare_point_arrays,
    resolve_point_cloud_path,
)
from spatiallm import Layout
from spatiallm.layout.entity import Bbox
from spatiallm.pcd import get_points_and_colors, load_o3d_pcd


REGION_STEM_RE = re.compile(r"^(?P<scene_id>.+)_region_(?P<region_index>\d+)$")


@dataclass
class RegionSample:
    scene_id: str
    region_index: int
    pcd_path: Path
    label_text: str


@dataclass
class SceneGroup:
    scene_id: str
    regions: list[RegionSample]


def prompt_with_point_token(prompt: str) -> str:
    return prompt.replace("<point_cloud>", POINT_PROMPT)


def extract_label_text(item: dict) -> str:
    conversations = item.get("conversations") or []
    for turn in conversations:
        if turn.get("from") == "gpt":
            return clean_generated_text(turn.get("value", ""))
    raise ValueError("Stage-2 sample has no gpt label text.")


def parse_region_stem(point_cloud: str) -> tuple[str, int]:
    match = REGION_STEM_RE.match(Path(point_cloud).stem)
    if match is None:
        raise ValueError(f"Cannot parse region point cloud name: {point_cloud}")
    return match.group("scene_id"), int(match.group("region_index"))


def load_scene_groups(data_json: Path, dataset_root: Path) -> list[SceneGroup]:
    data = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {data_json}.")

    grouped: dict[str, list[RegionSample]] = {}
    for item in data:
        point_clouds = item.get("point_clouds") or []
        if not point_clouds:
            continue
        raw_point_cloud = str(point_clouds[0])
        scene_id, region_index = parse_region_stem(raw_point_cloud)
        pcd_path = resolve_point_cloud_path(raw_point_cloud, dataset_root, data_json.parent)
        grouped.setdefault(scene_id, []).append(
            RegionSample(
                scene_id=scene_id,
                region_index=region_index,
                pcd_path=pcd_path,
                label_text=extract_label_text(item),
            )
        )

    groups = [
        SceneGroup(
            scene_id=scene_id,
            regions=sorted(regions, key=lambda region: region.region_index),
        )
        for scene_id, regions in grouped.items()
    ]
    return sorted(groups, key=lambda group: group.scene_id)


def apply_subset_args(
    groups: list[SceneGroup],
    start_index: int,
    end_index: int | None,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> list[SceneGroup]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < --num_shards")

    groups = groups[start_index:end_index]
    if num_shards > 1:
        groups = [
            group for index, group in enumerate(groups)
            if index % num_shards == shard_index
        ]
    if limit is not None:
        groups = groups[:limit]
    return groups


def prepare_region_point_cloud(pcd_path: Path, num_bins: int, world_size: float):
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    pcd = load_o3d_pcd(str(pcd_path))
    points, colors = get_points_and_colors(pcd)
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud has no points: {pcd_path}")
    return prepare_point_arrays(points, colors, num_bins, world_size=world_size)


def keep_bboxes_from_label(
    label_text: str,
    min_extent: np.ndarray,
    expand_ratio: float,
) -> torch.Tensor:
    layout = Layout(label_text)
    if not layout.bboxes:
        return torch.empty((1, 0, 7), dtype=torch.float32)

    scale_multiplier = 1.0 + 2.0 * expand_ratio
    rows = []
    for bbox in layout.bboxes:
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
        conversation, add_generation_prompt=True, return_tensors="pt"
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


def predict_scene_group(
    group: SceneGroup,
    model,
    tokenizer,
    args: argparse.Namespace,
) -> Layout:
    prompt = prompt_with_point_token(STAGE2_PROMPT)
    num_bins = model.config.point_config["num_bins"]
    world_size = model_world_size(model)

    all_bboxes: list[Bbox] = []
    for region in group.regions:
        prepared = prepare_region_point_cloud(region.pcd_path, num_bins, world_size)
        keep_bboxes = None
        if args.use_gt_bbox_mask:
            keep_bboxes = keep_bboxes_from_label(
                region.label_text,
                prepared.min_extent,
                args.bbox_mask_expand_ratio,
            )
        generated = generate_layout_text(
            model,
            tokenizer,
            prompt,
            prepared.input_tensor,
            args,
            point_token_keep_bboxes=keep_bboxes,
        )
        region_layout = decode_generated_layout(
            generated,
            prepared.min_extent,
            num_bins,
            world_size=world_size,
        )
        all_bboxes.extend(region_layout.bboxes)

    all_bboxes = classwise_nms(all_bboxes, args.bbox_nms_iou)
    final_layout = Layout()
    final_layout.bboxes = all_bboxes
    for bbox_id, bbox in enumerate(final_layout.bboxes):
        bbox.id = bbox_id
    return final_layout


def write_outputs(scene_id: str, final_layout: Layout, output_dir: Path, gt_region_dir: Path | None) -> None:
    final_dir = output_dir / "final"
    stage1_dir = output_dir / "stage1"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{scene_id}.txt"
    final_path.write_text(final_layout.to_language_string(), encoding="utf-8")

    if gt_region_dir is not None:
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage1_layout = load_gt_region_layout(scene_id, gt_region_dir)
        (stage1_dir / f"{scene_id}.txt").write_text(
            stage1_layout.to_language_string(),
            encoding="utf-8",
        )


def parse_args(default_use_gt_bbox_mask: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Oracle hierarchical stage-2 inference")
    parser.add_argument(
        "-i",
        "--data_json",
        type=Path,
        required=True,
        help="Stage-2 region JSON, e.g. spatiallm_stage2_bbox_test_evict.json.",
    )
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes",
    )
    parser.add_argument(
        "--gt_region_dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "region_test" / "expanded",
        help="Optional GT Region txt dir copied to output_dir/stage1 for evaluation/debug.",
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
    parser.add_argument("--bbox_nms_iou", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--use_gt_bbox_mask",
        action=argparse.BooleanOptionalAction,
        default=default_use_gt_bbox_mask,
        help="Use GT object boxes to mask final point tokens before LLM generation.",
    )
    parser.add_argument("--bbox_mask_expand_ratio", type=float, default=0.1)
    return parser.parse_args()


def main(default_use_gt_bbox_mask: bool = False) -> int:
    args = parse_args(default_use_gt_bbox_mask=default_use_gt_bbox_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_scene_groups(args.data_json, args.dataset_root)
    groups = apply_subset_args(
        groups,
        args.start_index,
        args.end_index,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    if not groups:
        raise ValueError("No scene groups found for inference.")

    if args.gt_region_dir is not None and not args.gt_region_dir.is_dir():
        raise NotADirectoryError(args.gt_region_dir)

    model, tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )

    failures: list[tuple[str, str]] = []
    for group in tqdm(groups, desc="Oracle stage-2 inference"):
        final_path = args.output_dir / "final" / f"{group.scene_id}.txt"
        if args.skip_existing and final_path.exists():
            continue
        try:
            final_layout = predict_scene_group(group, model, tokenizer, args)
            write_outputs(group.scene_id, final_layout, args.output_dir, args.gt_region_dir)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((group.scene_id, str(exc)))
            error_dir = args.output_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / f"{group.scene_id}.txt").write_text(str(exc), encoding="utf-8")

    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(f"Wrote oracle stage-2 predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(default_use_gt_bbox_mask=False))
