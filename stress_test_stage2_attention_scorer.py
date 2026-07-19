#!/usr/bin/env python3
"""Stress-test stage-2 attention-scorer memory on the longest cached samples."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from train_point_token_scorer import PointTokenScorer, ScorerConfig
from train_stage2_attention_scorer import (
    DEFAULT_MODEL_PATH,
    AttentionScorerCacheDataset,
    AttentionScorerCollator,
    build_tokenizer_and_template,
    forward_loss,
    load_frozen_stage2_model,
)


DEFAULT_FEATURE_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/bboxmask_ckpt14392_context_bf16"
)
DEFAULT_MESSAGE_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_cache_messages/bboxmask_ckpt14392_context_bf16"
)
DEFAULT_TOPK_JSON = DEFAULT_FEATURE_CACHE_ROOT / "top100_attention_scorer_stress_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the longest cached train/val samples and repeatedly run the "
            "same train/eval forward paths used by train_stage2_attention_scorer.py."
        )
    )
    parser.add_argument("--topk_json", type=Path, default=DEFAULT_TOPK_JSON)
    parser.add_argument("--feature_cache_root", type=Path, default=DEFAULT_FEATURE_CACHE_ROOT)
    parser.add_argument("--message_cache_root", type=Path, default=DEFAULT_MESSAGE_CACHE_ROOT)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument(
        "--sdpa_backend",
        choices=["default", "math", "flash", "efficient", "cudnn"],
        default="default",
        help=(
            "Optional PyTorch SDPA backend override around the frozen LLM forward. "
            "Use math if the default fused backend hits dense-mask stride/alignment errors."
        ),
    )
    parser.add_argument("--template", default="spatiallm_qwen")
    parser.add_argument("--cutoff_len", type=int, default=8192)
    parser.add_argument("--world_size", type=float, default=16.0)
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument("--max_point_tokens", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=1536)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing on the frozen stage-2 model during the train stress loop.",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shard_cache_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keep_train_batch_for_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the last train batch tensors alive while eval starts. This "
            "matches the normal training loop more closely."
        ),
    )
    args = parser.parse_args()
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive.")
    if args.per_device_eval_batch_size <= 0:
        parser.error("--per_device_eval_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    if args.train_steps <= 0:
        parser.error("--train_steps must be positive.")
    if args.eval_steps <= 0:
        parser.error("--eval_steps must be positive.")
    return args


def load_top_entries(path: Path, split: str, count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run build_attention_scorer_stress_topk_cache.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("splits", {}).get(split)
    if entries is None:
        raise KeyError(f"Split {split!r} not found in {path}.")
    if len(entries) < count:
        raise ValueError(f"Need {count} entries for split {split}, but only found {len(entries)}.")
    return entries[:count]


def make_subset_dataset(
    cache_root: Path,
    message_root: Path,
    split: str,
    entries: list[dict[str, Any]],
    max_point_tokens: int,
    shard_cache_size: int,
) -> AttentionScorerCacheDataset:
    dataset = AttentionScorerCacheDataset(
        cache_root / split,
        message_root / split,
        max_samples=None,
        max_point_tokens=max_point_tokens,
        shard_cache_size=shard_cache_size,
    )
    dataset.samples = entries
    return dataset


def build_fixed_batch(
    dataset: AttentionScorerCacheDataset,
    collator: AttentionScorerCollator,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    samples = [dataset[index] for index in range(batch_size)]
    return collator(samples)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_report(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(
        f"{prefix}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak_allocated={peak_allocated:.2f}GiB peak_reserved={peak_reserved:.2f}GiB",
        flush=True,
    )


def set_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    else:
        raise AttributeError("Loaded model does not expose gradient_checkpointing_enable().")
    if hasattr(model, "config"):
        model.config.use_cache = False


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    tokenizer, template = build_tokenizer_and_template(args)
    collator = AttentionScorerCollator(
        tokenizer=tokenizer,
        template=template,
        cutoff_len=args.cutoff_len,
        pad_to_multiple_of=8,
    )

    train_entries = load_top_entries(args.topk_json, args.train_split, args.per_device_train_batch_size)
    eval_entries = load_top_entries(args.topk_json, args.eval_split, args.per_device_eval_batch_size)
    train_dataset = make_subset_dataset(
        args.feature_cache_root,
        args.message_cache_root,
        args.train_split,
        train_entries,
        args.max_point_tokens,
        args.shard_cache_size,
    )
    eval_dataset = make_subset_dataset(
        args.feature_cache_root,
        args.message_cache_root,
        args.eval_split,
        eval_entries,
        args.max_point_tokens,
        args.shard_cache_size,
    )

    print("Selected train samples:")
    for entry in train_entries:
        print(
            f"  rank={entry.get('rank')} seq={entry.get('inserted_seq_len')} "
            f"point={entry.get('effective_point_token_count')} text={entry.get('text_token_count')} "
            f"scene={entry.get('scene_id')}"
        )
    print("Selected eval samples:")
    for entry in eval_entries:
        print(
            f"  rank={entry.get('rank')} seq={entry.get('inserted_seq_len')} "
            f"point={entry.get('effective_point_token_count')} text={entry.get('text_token_count')} "
            f"scene={entry.get('scene_id')}"
        )

    print("Building fixed CPU batches...", flush=True)
    train_batch_cpu = build_fixed_batch(train_dataset, collator, args.per_device_train_batch_size)
    eval_batch_cpu = build_fixed_batch(eval_dataset, collator, args.per_device_eval_batch_size)

    total_start = time.perf_counter()
    frozen_model = load_frozen_stage2_model(args, device)
    set_gradient_checkpointing(frozen_model, args.gradient_checkpointing)
    with torch.no_grad():
        dummy_context = torch.zeros(
            1,
            1,
            train_dataset.feature_dim,
            dtype=next(frozen_model.point_proj.parameters()).dtype,
            device=device,
        )
        dummy_point_tokens = frozen_model.point_proj(dummy_context)
        point_token_dim = int(dummy_point_tokens.shape[-1])

    scorer_config = ScorerConfig(
        encoder_feature_dim=train_dataset.feature_dim,
        point_token_dim=point_token_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        coord_scale=train_dataset.coord_scale,
    )
    scorer = PointTokenScorer(scorer_config).to(device=device, dtype=torch.float32)
    scorer.train()
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_batch = move_batch_to_device(train_batch_cpu, device)
    eval_batch = move_batch_to_device(eval_batch_cpu, device)
    print(
        "Loaded model/scorer and moved batches: "
        f"train_bs={args.per_device_train_batch_size}, eval_bs={args.per_device_eval_batch_size}, "
        f"train_point_shape={tuple(train_batch['features'].shape)}, "
        f"eval_point_shape={tuple(eval_batch['features'].shape)}, "
        f"attn_impl={args.attn_implementation}, sdpa_backend={args.sdpa_backend}, "
        f"gradient_checkpointing={args.gradient_checkpointing}",
        flush=True,
    )
    memory_report("after setup", device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    cuda_sync(device)
    train_start = time.perf_counter()
    last_train_stats: dict[str, float] = {}
    for step in range(1, args.train_steps + 1):
        for _ in range(args.gradient_accumulation_steps):
            loss, stats = forward_loss(
                frozen_model,
                scorer,
                train_batch,
                args.budget,
                args.sdpa_backend,
            )
            (loss / args.gradient_accumulation_steps).backward()
            last_train_stats = stats
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step == args.train_steps:
            print(
                f"train step={step} loss={last_train_stats.get('loss', 0.0):.6f} "
                f"lm={last_train_stats.get('lm_loss', 0.0):.6f} "
                f"cap={last_train_stats.get('capacity_loss', 0.0):.6f} "
                f"seq_len={last_train_stats.get('seq_len', 0.0):.0f}",
                flush=True,
            )
            memory_report("train", device)
    cuda_sync(device)
    train_elapsed = time.perf_counter() - train_start
    train_micro_steps = args.train_steps * args.gradient_accumulation_steps
    print(
        f"train elapsed={train_elapsed:.2f}s "
        f"seconds_per_optimizer_step={train_elapsed / args.train_steps:.4f} "
        f"seconds_per_micro_step={train_elapsed / train_micro_steps:.4f} "
        f"optimizer_steps_per_sec={args.train_steps / train_elapsed:.4f} "
        f"micro_steps_per_sec={train_micro_steps / train_elapsed:.4f}",
        flush=True,
    )

    if not args.keep_train_batch_for_eval:
        train_batch = {}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    memory_report("before eval", device)

    scorer.eval()
    cuda_sync(device)
    eval_start = time.perf_counter()
    last_eval_stats: dict[str, float] = {}
    with torch.no_grad():
        for step in range(1, args.eval_steps + 1):
            loss, stats = forward_loss(
                frozen_model,
                scorer,
                eval_batch,
                args.budget,
                args.sdpa_backend,
            )
            last_eval_stats = stats
            if step == 1 or step == args.eval_steps:
                print(
                    f"eval step={step} loss={last_eval_stats.get('loss', 0.0):.6f} "
                    f"lm={last_eval_stats.get('lm_loss', 0.0):.6f} "
                    f"cap={last_eval_stats.get('capacity_loss', 0.0):.6f} "
                    f"seq_len={last_eval_stats.get('seq_len', 0.0):.0f}",
                    flush=True,
                )
                memory_report("eval", device)
    cuda_sync(device)
    eval_elapsed = time.perf_counter() - eval_start
    print(
        f"eval elapsed={eval_elapsed:.2f}s steps_per_sec={args.eval_steps / eval_elapsed:.4f}",
        flush=True,
    )
    total_elapsed = time.perf_counter() - total_start
    print(f"total elapsed={total_elapsed:.2f}s", flush=True)
    memory_report("final", device)


if __name__ == "__main__":
    main()
