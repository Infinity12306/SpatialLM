#!/usr/bin/env python3
"""Build a small index of the longest cached samples for attention-scorer stress tests."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.tuner.data.template import IGNORE_INDEX
from train_stage2_attention_scorer import (
    DEFAULT_MODEL_PATH,
    AttentionScorerCollator,
    build_tokenizer_and_template,
)


DEFAULT_FEATURE_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/bboxmask_ckpt14392_context_bf16"
)
DEFAULT_MESSAGE_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_cache_messages/bboxmask_ckpt14392_context_bf16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan point-token feature cache and aligned message cache, then save "
            "the top-k samples with the longest inserted sequence length."
        )
    )
    parser.add_argument("--feature_cache_root", type=Path, default=DEFAULT_FEATURE_CACHE_ROOT)
    parser.add_argument("--message_cache_root", type=Path, default=DEFAULT_MESSAGE_CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output_json", type=Path, default=DEFAULT_FEATURE_CACHE_ROOT / "top100_attention_scorer_stress_samples.json")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--template", default="spatiallm_qwen")
    parser.add_argument("--cutoff_len", type=int, default=8192)
    parser.add_argument("--world_size", type=float, default=16.0)
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument("--max_point_tokens", type=int, default=4096)
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be positive.")
    if args.max_point_tokens <= 0:
        parser.error("--max_point_tokens must be positive.")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_single_position(input_ids: list[int], token_id: int, token_name: str) -> int:
    positions = [index for index, value in enumerate(input_ids) if value == token_id]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one {token_name} token, got {len(positions)}.")
    return positions[0]


def scan_split(
    split: str,
    args: argparse.Namespace,
    collator: AttentionScorerCollator,
    point_start_token_id: int,
    point_end_token_id: int,
) -> list[dict[str, Any]]:
    feature_split_dir = args.feature_cache_root / split
    message_split_dir = args.message_cache_root / split
    index_path = feature_split_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not message_split_dir.is_dir():
        raise FileNotFoundError(message_split_dir)

    index = load_json(index_path)
    samples = index["samples"]
    if args.max_scan_samples is not None:
        samples = samples[: args.max_scan_samples]

    heap: list[tuple[int, int, dict[str, Any]]] = []
    loaded_message_shard_rel: str | None = None
    loaded_message_shard: dict[str, Any] | None = None

    for ordinal, entry in enumerate(tqdm(samples, desc=f"scan {split}")):
        shard_rel = str(entry["shard"])
        if shard_rel != loaded_message_shard_rel:
            shard_path = message_split_dir / shard_rel
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            loaded_message_shard = torch.load(shard_path, map_location="cpu")
            loaded_message_shard_rel = shard_rel
        assert loaded_message_shard is not None

        item_index = int(entry["item_index"])
        message_item = loaded_message_shard["items"][item_index]
        input_ids, labels = collator._encode_messages(message_item["messages"])

        point_start = find_single_position(input_ids, point_start_token_id, "point_start")
        point_end = find_single_position(input_ids, point_end_token_id, "point_end")
        removed_placeholder_count = max(point_end - point_start - 1, 0)

        raw_point_tokens = int(entry.get("token_count", 0))
        effective_point_tokens = min(raw_point_tokens, args.max_point_tokens)
        text_token_count = len(input_ids)
        label_token_count = sum(1 for label in labels if label != IGNORE_INDEX)
        inserted_seq_len = text_token_count - removed_placeholder_count + effective_point_tokens

        record = {
            "split": split,
            "rank": 0,
            "shard": shard_rel,
            "item_index": item_index,
            "epoch": int(entry.get("epoch", message_item.get("epoch", -1))),
            "sample_index": int(entry.get("sample_index", message_item.get("sample_index", -1))),
            "point_cloud_index": int(entry.get("point_cloud_index", message_item.get("point_cloud_index", -1))),
            "scene_id": str(entry.get("scene_id", message_item.get("scene_id", ""))),
            "point_cloud": str(entry.get("point_cloud", message_item.get("point_cloud", ""))),
            "raw_point_token_count": raw_point_tokens,
            "effective_point_token_count": effective_point_tokens,
            "text_token_count": text_token_count,
            "label_token_count": label_token_count,
            "prompt_token_count": text_token_count - label_token_count,
            "removed_placeholder_count": removed_placeholder_count,
            "inserted_seq_len": inserted_seq_len,
            "point_start_index": point_start,
            "point_end_index": point_end,
        }

        heap_key = (inserted_seq_len, ordinal)
        if len(heap) < args.top_k:
            heapq.heappush(heap, (heap_key[0], heap_key[1], record))
        elif heap_key > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, (heap_key[0], heap_key[1], record))

    top_records = [item[2] for item in sorted(heap, key=lambda value: (value[0], value[1]), reverse=True)]
    for rank, record in enumerate(top_records, start=1):
        record["rank"] = rank
    return top_records


def main() -> None:
    args = parse_args()
    if args.output_json.exists() and not args.overwrite:
        print(f"Output exists, skip: {args.output_json}")
        return

    tokenizer_args = SimpleNamespace(
        model_path=args.model_path,
        template=args.template,
        cutoff_len=args.cutoff_len,
        num_bins=args.num_bins,
        world_size=args.world_size,
    )
    tokenizer, template = build_tokenizer_and_template(tokenizer_args)
    collator = AttentionScorerCollator(
        tokenizer=tokenizer,
        template=template,
        cutoff_len=args.cutoff_len,
        pad_to_multiple_of=8,
    )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    point_start_token_id = int(config.point_start_token_id)
    point_end_token_id = int(config.point_end_token_id)

    result = {
        "metadata": {
            "feature_cache_root": str(args.feature_cache_root),
            "message_cache_root": str(args.message_cache_root),
            "model_path": str(args.model_path),
            "top_k": args.top_k,
            "cutoff_len": args.cutoff_len,
            "world_size": args.world_size,
            "num_bins": args.num_bins,
            "max_point_tokens": args.max_point_tokens,
            "point_start_token_id": point_start_token_id,
            "point_end_token_id": point_end_token_id,
        },
        "splits": {},
    }
    for split in args.splits:
        result["splits"][split] = scan_split(
            split,
            args,
            collator,
            point_start_token_id,
            point_end_token_id,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
