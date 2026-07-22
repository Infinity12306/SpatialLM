#!/usr/bin/env python3
"""Hierarchical inference with an end-to-end learned point-token attention bias."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    load_model_and_tokenizer,
    load_scenes,
    write_prediction_outputs,
    write_region_debug_pcds,
)
from inference_hierarchical_scorer import (
    latest_scorer_checkpoint,
    predict_hierarchical_scene_with_scorer,
)
from spatiallm.model import PointBackboneType
from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig


DEFAULT_SCORER_PATH = Path(
    "/data2/chenjq24/SpatialLM/saves/point_token_attention_scorer/"
    "train_scorer_atten_endToEnd/checkpoint-40000"
)


def load_attention_scorer(
    path: Path,
    device: str,
) -> tuple[PointTokenScorer, dict]:
    scorer_path = latest_scorer_checkpoint(path)
    checkpoint = torch.load(scorer_path, map_location="cpu")
    scorer = PointTokenScorer(ScorerConfig(**checkpoint["config"]))
    scorer.load_state_dict(checkpoint["model"])
    scorer.eval().requires_grad_(False)
    scorer.to(device=device, dtype=torch.float32)
    print(f"Loaded end-to-end attention scorer: {scorer_path}")
    return scorer, checkpoint


def center_crop_pair(
    context: torch.Tensor,
    grid_coord: torch.Tensor,
    max_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_tokens is None or context.shape[0] <= max_tokens:
        return context, grid_coord
    start = (context.shape[0] - max_tokens) // 2
    end = start + max_tokens
    return context[start:end], grid_coord[start:end]


def build_causal_mask(
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    key_length: int,
) -> torch.Tensor:
    key_positions = torch.arange(key_length, device=input_tensor.device)
    blocked = key_positions[None, :] > cache_position[:, None]
    mask = torch.zeros(
        input_tensor.shape[0],
        1,
        input_tensor.shape[1],
        key_length,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return mask.masked_fill(blocked[None, None, :, :], torch.finfo(input_tensor.dtype).min)


def install_attention_scorer_bias(
    stage2_model,
    scorer: PointTokenScorer,
    args: argparse.Namespace,
) -> None:
    """Apply the scorer's learned log-sigmoid bias during every generation step."""
    if stage2_model.point_backbone_type != PointBackboneType.SONATA:
        raise NotImplementedError("Attention scorer inference currently supports Sonata only.")
    if hasattr(stage2_model, "_attention_scorer_original_forward_point_cloud"):
        raise RuntimeError("Attention scorer patch is already installed.")

    scorer_device = torch.device(args.device)
    decoder = stage2_model.model
    decoder._attention_scorer_point_start = None
    decoder._attention_scorer_key_bias = None

    original_forward_point_cloud = stage2_model.forward_point_cloud
    original_update_causal_mask = decoder._update_causal_mask
    original_generate = stage2_model.generate
    stage2_model._attention_scorer_original_forward_point_cloud = original_forward_point_cloud

    def scored_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
    ):
        if point_token_keep_bboxes is not None:
            raise ValueError(
                "End-to-end attention scorer inference cannot be combined with oracle bbox filtering."
            )

        self.point_backbone.to(torch.float32)
        valid = ~torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[valid]
        if point_cloud.shape[0] == 0:
            raise ValueError("Point cloud has no valid points after removing NaNs.")

        coords = point_cloud[:, :3].to(device=device, dtype=torch.int64)
        feats = point_cloud[:, 3:].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            encoded = self.point_backbone(
                {
                    "coord": feats[:, :3],
                    "grid_coord": coords,
                    "feat": feats,
                    "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
                    "return_grid_coord": True,
                }
            )
            context = encoded["context"]
            grid_coord = encoded["grid_coord"].to(torch.int32)
            if context.shape[0] == 0:
                raise ValueError("Point encoder produced zero point tokens.")

            raw_token_count = int(context.shape[0])
            full_center = (
                grid_coord.float().amin(dim=0) + grid_coord.float().amax(dim=0)
            ) * 0.5
            context, grid_coord = center_crop_pair(
                context,
                grid_coord,
                args.attention_max_point_tokens,
            )
            projector_dtype = next(self.point_proj.parameters()).dtype
            point_tokens = self.point_proj(context.to(projector_dtype))
            attention_mask = torch.ones(
                1,
                point_tokens.shape[0],
                dtype=torch.bool,
                device=scorer_device,
            )

            fastpath_enabled = None
            if args.attention_disable_mha_fastpath and hasattr(torch.backends, "mha"):
                fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
                torch.backends.mha.set_fastpath_enabled(False)
            try:
                logits = scorer(
                    point_tokens.float().unsqueeze(0).to(scorer_device),
                    grid_coord.float().unsqueeze(0).to(scorer_device),
                    full_center.float().unsqueeze(0).to(scorer_device),
                    attention_mask,
                ).squeeze(0)
            finally:
                if fastpath_enabled is not None:
                    torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

            key_bias = F.logsigmoid(logits).detach()
            self.model._attention_scorer_key_bias = key_bias
            if args.attention_debug:
                scores = torch.sigmoid(logits)
                print(
                    "attention scorer tokens: "
                    f"raw={raw_token_count}, scored={point_tokens.shape[0]}, "
                    f"score_mean={scores.mean().item():.6f}, "
                    f"score_ge_0.5={(scores >= 0.5).float().mean().item():.6f}, "
                    f"soft_keep={scores.sum().item():.2f}, "
                    f"training_budget={args.attention_budget}"
                )
            return point_tokens.to(dtype).unsqueeze(0)

    def capture_point_start(self, forward_args, forward_kwargs):
        input_ids = forward_kwargs.get("input_ids")
        if input_ids is None and forward_args:
            input_ids = forward_args[0]
        if input_ids is not None and input_ids.ndim == 2 and input_ids.shape[1] > 1:
            positions = torch.where(input_ids[0] == self.config.point_start_token_id)[0]
            if positions.numel() == 1:
                self.model._attention_scorer_point_start = int(positions[0].item()) + 1
                self.model._attention_scorer_key_bias = None

    def update_causal_mask_with_point_bias(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions,
    ):
        causal_mask = original_update_causal_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        )
        key_bias = self._attention_scorer_key_bias
        point_start = self._attention_scorer_point_start
        if key_bias is None or point_start is None:
            return causal_mask

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        key_length = past_length + input_tensor.shape[1]
        if (
            causal_mask is None
            or causal_mask.shape[-2] != input_tensor.shape[1]
            or causal_mask.shape[-1] < key_length
        ):
            causal_mask = build_causal_mask(input_tensor, cache_position, key_length)
        else:
            causal_mask = causal_mask[..., : input_tensor.shape[1], :key_length]

        full_key_bias = torch.zeros(
            key_length,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        available = max(0, min(int(key_bias.numel()), key_length - point_start))
        if available > 0:
            full_key_bias[point_start : point_start + available] = key_bias[:available].to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        return causal_mask + full_key_bias[None, None, None, :]

    def generate_with_math_sdpa(self, *generate_args, **generate_kwargs):
        with sdpa_kernel([SDPBackend.MATH]):
            return original_generate(*generate_args, **generate_kwargs)

    stage2_model.forward_point_cloud = MethodType(scored_forward_point_cloud, stage2_model)
    stage2_model._attention_scorer_pre_hook = stage2_model.register_forward_pre_hook(
        capture_point_start,
        with_kwargs=True,
    )
    decoder._update_causal_mask = MethodType(update_causal_mask_with_point_bias, decoder)
    stage2_model.generate = MethodType(generate_with_math_sdpa, stage2_model)


def install_attention_scorer_topk(
    stage2_model,
    scorer: PointTokenScorer,
    args: argparse.Namespace,
) -> None:
    """Hard-select the highest-scoring tokens and preserve their Sonata order."""
    if stage2_model.point_backbone_type != PointBackboneType.SONATA:
        raise NotImplementedError("Attention scorer inference currently supports Sonata only.")
    if hasattr(stage2_model, "_attention_scorer_original_forward_point_cloud"):
        raise RuntimeError("Attention scorer patch is already installed.")

    scorer_device = torch.device(args.device)
    stage2_model._attention_scorer_original_forward_point_cloud = (
        stage2_model.forward_point_cloud
    )

    def topk_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
    ):
        if point_token_keep_bboxes is not None:
            raise ValueError(
                "End-to-end attention scorer inference cannot be combined with oracle bbox filtering."
            )

        self.point_backbone.to(torch.float32)
        valid = ~torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[valid]
        if point_cloud.shape[0] == 0:
            raise ValueError("Point cloud has no valid points after removing NaNs.")

        coords = point_cloud[:, :3].to(device=device, dtype=torch.int64)
        feats = point_cloud[:, 3:].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            encoded = self.point_backbone(
                {
                    "coord": feats[:, :3],
                    "grid_coord": coords,
                    "feat": feats,
                    "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
                    "return_grid_coord": True,
                }
            )
            context = encoded["context"]
            grid_coord = encoded["grid_coord"].to(torch.int32)
            if context.shape[0] == 0:
                raise ValueError("Point encoder produced zero point tokens.")

            raw_token_count = int(context.shape[0])
            full_center = (
                grid_coord.float().amin(dim=0) + grid_coord.float().amax(dim=0)
            ) * 0.5
            context, grid_coord = center_crop_pair(
                context,
                grid_coord,
                args.attention_max_point_tokens,
            )
            projector_dtype = next(self.point_proj.parameters()).dtype
            point_tokens = self.point_proj(context.to(projector_dtype))
            attention_mask = torch.ones(
                1,
                point_tokens.shape[0],
                dtype=torch.bool,
                device=scorer_device,
            )

            fastpath_enabled = None
            if args.attention_disable_mha_fastpath and hasattr(torch.backends, "mha"):
                fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
                torch.backends.mha.set_fastpath_enabled(False)
            try:
                logits = scorer(
                    point_tokens.float().unsqueeze(0).to(scorer_device),
                    grid_coord.float().unsqueeze(0).to(scorer_device),
                    full_center.float().unsqueeze(0).to(scorer_device),
                    attention_mask,
                ).squeeze(0)
            finally:
                if fastpath_enabled is not None:
                    torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

            keep_count = min(args.attention_top_k, int(logits.numel()))
            keep_indices = torch.topk(logits, k=keep_count).indices.sort().values
            selected_tokens = point_tokens[keep_indices.to(point_tokens.device)]
            if args.attention_debug:
                scores = torch.sigmoid(logits)
                print(
                    "attention scorer hard top-k: "
                    f"raw={raw_token_count}, scored={point_tokens.shape[0]}, "
                    f"keep={selected_tokens.shape[0]}, k={args.attention_top_k}, "
                    f"score_mean={scores.mean().item():.6f}"
                )
            return selected_tokens.to(dtype).unsqueeze(0)

    stage2_model.forward_point_cloud = MethodType(topk_forward_point_cloud, stage2_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hierarchical inference with an end-to-end point-token attention scorer"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--data_json", type=Path)
    input_group.add_argument("-p", "--point_cloud", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage1_model_path", required=True)
    parser.add_argument("--stage2_model_path", required=True)
    parser.add_argument(
        "--attention_scorer_path",
        type=Path,
        default=DEFAULT_SCORER_PATH,
    )
    parser.add_argument(
        "--attention_selection",
        choices=("soft_bias", "hard_topk"),
        default="soft_bias",
    )
    parser.add_argument("--attention_max_point_tokens", type=int, default=3200)
    parser.add_argument("--attention_top_k", type=int, default=1536)
    parser.add_argument(
        "--attention_budget",
        type=int,
        default=1536,
        help="Training capacity budget, used for checkpoint validation/debugging only.",
    )
    parser.add_argument(
        "--attention_disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attention_debug", action="store_true")
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
    if args.attention_max_point_tokens <= 0:
        parser.error("--attention_max_point_tokens must be positive.")
    if args.attention_budget <= 0:
        parser.error("--attention_budget must be positive.")
    if args.attention_top_k <= 0:
        parser.error("--attention_top_k must be positive.")
    return args


def validate_checkpoint_args(checkpoint: dict, args: argparse.Namespace) -> None:
    training_args = checkpoint.get("args", {})
    expected_max = training_args.get("max_point_tokens")
    expected_budget = training_args.get("budget")
    if expected_max is not None and int(expected_max) != args.attention_max_point_tokens:
        raise ValueError(
            "Attention scorer max_point_tokens mismatch: "
            f"checkpoint={expected_max}, inference={args.attention_max_point_tokens}."
        )
    if expected_budget is not None and int(expected_budget) != args.attention_budget:
        raise ValueError(
            "Attention scorer budget mismatch: "
            f"checkpoint={expected_budget}, inference={args.attention_budget}."
        )


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
    scorer, checkpoint = load_attention_scorer(args.attention_scorer_path, args.device)
    validate_checkpoint_args(checkpoint, args)
    if args.attention_selection == "hard_topk":
        install_attention_scorer_topk(stage2_model, scorer, args)
    else:
        install_attention_scorer_bias(stage2_model, scorer, args)

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Hierarchical end-to-end attention scorer inference"):
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
    print(f"Wrote attention-scorer predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
