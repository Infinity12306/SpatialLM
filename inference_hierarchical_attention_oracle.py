#!/usr/bin/env python3
"""Oracle stage-2 inference with GT-label attention-based point-token filtering."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import MethodType

import torch
from tqdm import tqdm
from transformers import set_seed
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from build_hierarchical_region_dataset import STAGE2_PROMPT
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    LAYOUT_END,
    LAYOUT_START,
    clean_generated_text,
    classwise_nms,
    decode_generated_layout,
    load_model_and_tokenizer,
    make_conversation,
    model_world_size,
)
from inference_hierarchical_evict import (
    apply_subset_args,
    load_scene_groups,
    prepare_region_point_cloud,
    prompt_with_point_token,
    write_outputs,
)
from spatiallm import Layout
from spatiallm.layout.entity import Bbox


def install_attention_keep_patch(model) -> None:
    if hasattr(model, "_attention_oracle_original_forward_point_cloud"):
        return

    model._attention_oracle_original_forward_point_cloud = model.forward_point_cloud
    model._attention_oracle_keep_indices = None
    model._attention_oracle_last_point_token_count = None

    def patched_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
    ):
        point_features = self._attention_oracle_original_forward_point_cloud(
            point_cloud,
            device,
            dtype,
            point_token_keep_bboxes,
        )
        self._attention_oracle_last_point_token_count = int(point_features.shape[1])
        keep_indices = getattr(self, "_attention_oracle_keep_indices", None)
        if keep_indices is not None:
            keep_indices = keep_indices.to(point_features.device)
            point_features = point_features[:, keep_indices, :]
        return point_features

    model.forward_point_cloud = MethodType(patched_forward_point_cloud, model)


def build_prompt_and_label_ids(model, tokenizer, prompt: str, label_text: str):
    if LAYOUT_START not in label_text:
        label_text = f"{LAYOUT_START}{label_text}{LAYOUT_END}"
    conversation = make_conversation(model, prompt)
    prompt_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    label_ids = tokenizer(
        label_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    input_ids = torch.cat([prompt_ids, label_ids], dim=1)
    return prompt_ids, label_ids, input_ids


def compute_last_layer_point_attention_scores(
    model,
    hidden_states: torch.Tensor,
    point_start: int,
    num_point_tokens: int,
    label_query_positions: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if num_point_tokens <= 0:
        return torch.empty(0, dtype=torch.float32, device=hidden_states.device)
    if label_query_positions.numel() == 0:
        return torch.ones(num_point_tokens, dtype=torch.float32, device=hidden_states.device)

    decoder_layer = model.model.layers[-1]
    attn = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)
    seq_len = int(normed.shape[0])
    position_ids = torch.arange(seq_len, device=normed.device).unsqueeze(0)

    query_states = attn.q_proj(normed).view(
        1,
        seq_len,
        attn.num_heads,
        attn.head_dim,
    ).transpose(1, 2)
    key_states = attn.k_proj(normed).view(
        1,
        seq_len,
        attn.num_key_value_heads,
        attn.head_dim,
    ).transpose(1, 2)

    cos, sin = model.model.rotary_emb(query_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, attn.num_key_value_groups)

    query_states = query_states[0]
    key_states = key_states[0]
    key_positions = torch.arange(seq_len, device=normed.device)
    point_slice = slice(point_start, point_start + num_point_tokens)

    point_score_sum = torch.zeros(num_point_tokens, dtype=torch.float32, device=normed.device)
    total_query_heads = 0
    scale = 1.0 / math.sqrt(attn.head_dim)

    for start in range(0, int(label_query_positions.numel()), chunk_size):
        query_pos = label_query_positions[start : start + chunk_size]
        q = query_states[:, query_pos, :]
        scores = torch.einsum("hqd,hkd->hqk", q, key_states) * scale
        future_mask = key_positions[None, None, :] > query_pos[None, :, None]
        scores = scores.masked_fill(future_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        point_score_sum += probs[:, :, point_slice].sum(dim=(0, 1))
        total_query_heads += probs.shape[0] * probs.shape[1]

    return point_score_sum / max(total_query_heads, 1)


def select_top_attention_tokens(
    scores: torch.Tensor,
    keep_ratio: float,
    min_keep: int,
) -> torch.Tensor:
    num_tokens = int(scores.numel())
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if not 0 < keep_ratio <= 1:
        raise ValueError("--attention_keep_ratio must satisfy 0 < ratio <= 1.")

    keep_count = int(math.ceil(num_tokens * keep_ratio))
    keep_count = max(min_keep, keep_count)
    keep_count = min(num_tokens, keep_count)
    if keep_count >= num_tokens:
        return torch.arange(num_tokens, dtype=torch.long, device=scores.device)

    keep_indices = torch.topk(scores, k=keep_count).indices
    return keep_indices.sort().values


@torch.no_grad()
def compute_attention_keep_indices(
    model,
    tokenizer,
    prompt: str,
    label_text: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    prompt_ids, label_ids, input_ids = build_prompt_and_label_ids(
        model,
        tokenizer,
        prompt,
        label_text,
    )
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    model._attention_oracle_keep_indices = None
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        point_clouds=point_cloud,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
        num_logits_to_keep=1,
    )
    num_point_tokens = int(model._attention_oracle_last_point_token_count)
    if num_point_tokens <= 0:
        return torch.empty(0, dtype=torch.long, device=model.device)

    point_start_original = int(
        torch.where(input_ids[0] == model.config.point_start_token_id)[0][0].item()
    )
    point_start_expanded = point_start_original + 1
    label_start_original = int(prompt_ids.shape[1])
    label_len = int(label_ids.shape[1])

    query_start = label_start_original + num_point_tokens - 2
    label_query_positions = torch.arange(
        query_start,
        query_start + label_len,
        device=model.device,
        dtype=torch.long,
    )
    label_query_positions = label_query_positions[label_query_positions >= 0]

    hidden_before_last = outputs.hidden_states[-2][0]
    scores = compute_last_layer_point_attention_scores(
        model,
        hidden_before_last,
        point_start_expanded,
        num_point_tokens,
        label_query_positions,
        args.attention_chunk_size,
    )
    keep_indices = select_top_attention_tokens(
        scores,
        args.attention_keep_ratio,
        args.attention_min_keep,
    )
    if args.attention_debug:
        print(
            "attention oracle: "
            f"tokens={num_point_tokens}, keep={keep_indices.numel()}, "
            f"ratio={args.attention_keep_ratio}, "
            f"score_mean={scores.mean().item():.6g}, score_max={scores.max().item():.6g}"
        )
    return keep_indices


def generate_layout_text_with_current_keep(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
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

    output_ids = model.generate(**generate_kwargs)
    generated_ids = output_ids[0, input_ids.shape[1] :]
    return clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))


def predict_scene_group(
    group,
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
        keep_indices = compute_attention_keep_indices(
            model,
            tokenizer,
            prompt,
            region.label_text,
            prepared.input_tensor,
            args,
        )
        model._attention_oracle_keep_indices = keep_indices
        generated = generate_layout_text_with_current_keep(
            model,
            tokenizer,
            prompt,
            prepared.input_tensor,
            args,
        )
        model._attention_oracle_keep_indices = None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Oracle stage-2 inference with GT-label attention point-token filtering"
    )
    parser.add_argument(
        "-i",
        "--data_json",
        type=Path,
        required=True,
        help="Stage-2 region JSON, e.g. spatiallm_stage2_bbox_test.json.",
    )
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes_20000_res_16_max_4096/checkpoint-14392",
    )
    parser.add_argument(
        "--gt_region_dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "region_test" / "expanded",
    )
    parser.add_argument("--attention_keep_ratio", type=float, default=0.75)
    parser.add_argument("--attention_min_keep", type=int, default=1)
    parser.add_argument("--attention_chunk_size", type=int, default=64)
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
    parser.add_argument("--bbox_nms_iou", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.attention_keep_ratio <= 1:
        parser.error("--attention_keep_ratio must satisfy 0 < ratio <= 1.")
    if args.attention_min_keep < 0:
        parser.error("--attention_min_keep must be non-negative.")
    if args.attention_chunk_size <= 0:
        parser.error("--attention_chunk_size must be positive.")
    return args


def main() -> int:
    args = parse_args()
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
    install_attention_keep_patch(model)

    failures: list[tuple[str, str]] = []
    for group in tqdm(groups, desc="GT-attention oracle stage-2 inference"):
        final_path = args.output_dir / "final" / f"{group.scene_id}.txt"
        if args.skip_existing and final_path.exists():
            continue
        try:
            final_layout = predict_scene_group(group, model, tokenizer, args)
            write_outputs(group.scene_id, final_layout, args.output_dir, args.gt_region_dir)
        except Exception as exc:
            model._attention_oracle_keep_indices = None
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

    print(f"Wrote GT-attention oracle predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
