#!/usr/bin/env python3
"""Hierarchical inference with predicted regions and scorer-selected point tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tqdm import tqdm

from build_hierarchical_region_dataset import STAGE1_PROMPT, STAGE2_PROMPT
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    HierarchicalPrediction,
    RegionPrediction,
    center_crop_point_arrays,
    classwise_nms,
    decode_generated_layout,
    final_layout_from_parts,
    generate_layout_text,
    load_model_and_tokenizer,
    load_scenes,
    model_world_size,
    points_in_region,
    prepare_point_arrays,
    prepare_scene_point_cloud,
    prompt_with_point_token,
    write_prediction_outputs,
    write_region_debug_pcds,
)
from spatiallm.layout.entity import Bbox
from spatiallm.model import PointBackboneType
from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig


DEFAULT_SCORER_PATH = Path(
    "/data2/chenjq24/SpatialLM/saves/point_token_scorer/"
    "bboxmask_ckpt14392_context_bf16"
)


def latest_scorer_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "scorer.pt").is_file():
        return path / "scorer.pt"

    checkpoints = []
    for candidate in path.glob("checkpoint-*"):
        if not candidate.is_dir() or not (candidate / "scorer.pt").is_file():
            continue
        suffix = candidate.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), candidate / "scorer.pt"))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]

    raise FileNotFoundError(f"No scorer.pt found under {path}")


def load_point_token_scorer(path: Path, device: str) -> PointTokenScorer:
    scorer_path = latest_scorer_checkpoint(path)
    ckpt = torch.load(scorer_path, map_location="cpu")
    config = ScorerConfig(**ckpt["config"])
    scorer = PointTokenScorer(config)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()
    scorer.requires_grad_(False)
    scorer.to(device=device, dtype=torch.float32)
    print(f"Loaded point-token scorer: {scorer_path}")
    return scorer


def scorer_keep_indices(
    scores: torch.Tensor,
    threshold: float,
    min_keep: int,
    max_keep: int,
) -> torch.Tensor:
    num_tokens = int(scores.numel())
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if max_keep <= 0:
        raise ValueError("--scorer_max_keep must be positive.")
    if min_keep < 0:
        raise ValueError("--scorer_min_keep must be non-negative.")

    min_keep = min(min_keep, num_tokens)
    max_keep = min(max_keep, num_tokens)
    if min_keep > max_keep:
        min_keep = max_keep

    keep_indices = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        selected_scores = scores[keep_indices]
        top_local = torch.topk(selected_scores, k=max_keep).indices
        keep_indices = keep_indices[top_local]

    return keep_indices.sort().values


def install_scorer_point_filter(
    stage2_model,
    scorer: PointTokenScorer,
    args: argparse.Namespace,
) -> None:
    scorer_device = torch.device(args.device)

    def scorer_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
    ):
        if self.point_backbone_type != PointBackboneType.SONATA:
            raise NotImplementedError(
                "Point-token scorer inference currently supports Sonata only."
            )

        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        if point_cloud.shape[0] == 0:
            raise ValueError("Point cloud has no valid points after removing NaNs.")

        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        input_dict = {
            "coord": feats[:, :3].to(device),
            "grid_coord": coords.to(device),
            "feat": feats.to(device),
            "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
            "return_grid_coord": True,
        }

        with torch.inference_mode():
            encoded = self.point_backbone(input_dict)
            context = encoded["context"]
            grid_coord = encoded["grid_coord"].to(torch.int32)

            if context.shape[0] == 0:
                raise ValueError("Point encoder produced zero point tokens.")

            projector_dtype = next(self.point_proj.parameters()).dtype
            point_tokens = self.point_proj(context.to(projector_dtype))

            center = (grid_coord.float().amin(dim=0) + grid_coord.float().amax(dim=0)) * 0.5
            attention_mask = torch.ones(
                1,
                grid_coord.shape[0],
                dtype=torch.bool,
                device=scorer_device,
            )

            fastpath_enabled = None
            if args.scorer_disable_mha_fastpath and hasattr(torch.backends, "mha"):
                fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
                torch.backends.mha.set_fastpath_enabled(False)
            try:
                logits = scorer(
                    point_tokens.float().unsqueeze(0).to(scorer_device),
                    grid_coord.float().unsqueeze(0).to(scorer_device),
                    center.float().unsqueeze(0).to(scorer_device),
                    attention_mask,
                ).squeeze(0)
            finally:
                if fastpath_enabled is not None:
                    torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

            scores = torch.sigmoid(logits)
            keep_indices = scorer_keep_indices(
                scores,
                args.scorer_threshold,
                args.scorer_min_keep,
                args.scorer_max_keep,
            ).to(point_tokens.device)

            if args.scorer_debug:
                print(
                    "scorer tokens: "
                    f"raw={point_tokens.shape[0]}, keep={keep_indices.numel()}, "
                    f"threshold={args.scorer_threshold}"
                )

            selected_tokens = point_tokens[keep_indices].to(dtype)
            return selected_tokens.unsqueeze(0)

    stage2_model.forward_point_cloud = MethodType(scorer_forward_point_cloud, stage2_model)


def predict_hierarchical_scene_with_scorer(
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

        region_pcd = prepare_point_arrays(
            region_points,
            region_colors,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        stage2_generated = generate_layout_text(
            stage2_model,
            stage2_tokenizer,
            stage2_prompt,
            region_pcd.input_tensor,
            args,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hierarchical inference with predicted regions and scorer-selected point tokens"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--data_json",
        type=Path,
        help="Stage1-style JSON dataset, e.g. spatiallm_stage1_region_test.json.",
    )
    input_group.add_argument(
        "-p",
        "--point_cloud",
        type=Path,
        help="PLY file or directory of PLY files.",
    )
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--stage1_model_path",
        default="saves/hierarchical/stage1_regions",
        help="Stage-1 checkpoint used to predict regions.",
    )
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes",
        help="Stage-2 checkpoint used to predict object bboxes.",
    )
    parser.add_argument(
        "--scorer_path",
        type=Path,
        default=DEFAULT_SCORER_PATH,
        help="Scorer checkpoint dir, scorer.pt file, or dir containing checkpoint-*.",
    )
    parser.add_argument("--scorer_threshold", type=float, default=0.5)
    parser.add_argument("--scorer_max_keep", type=int, default=4096)
    parser.add_argument("--scorer_min_keep", type=int, default=1)
    parser.add_argument(
        "--scorer_disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scorer_debug", action="store_true")
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
    args = parser.parse_args()

    if args.scorer_max_keep <= 0:
        parser.error("--scorer_max_keep must be positive.")
    if args.scorer_min_keep < 0:
        parser.error("--scorer_min_keep must be non-negative.")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args)
    if not scenes:
        raise ValueError("No scenes found for inference.")

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
    scorer = load_point_token_scorer(args.scorer_path, args.device)
    install_scorer_point_filter(stage2_model, scorer, args)

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Hierarchical scorer inference"):
        final_path = args.output_dir / "final" / f"{scene.scene_id}.txt"
        stage1_path = args.output_dir / "stage1" / f"{scene.scene_id}.txt"
        if args.skip_existing and final_path.exists() and stage1_path.exists():
            continue

        try:
            prediction = predict_hierarchical_scene_with_scorer(
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

    print(f"Wrote hierarchical scorer predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
