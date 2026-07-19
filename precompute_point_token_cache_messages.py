#!/usr/bin/env python3
"""Precompute processed training messages for point-token cache items."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from filter_point_tokens_with_scorer import (
    load_index,
    parse_dataset_json_overrides,
    processed_messages_for_item,
    resolve_dataset_json,
)
from precompute_point_token_scorer_data import load_samples
from spatiallm.tuner.data.mm_plugin import SpatialLMPlugin


DEFAULT_INPUT_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/bboxmask_ckpt14392_context_bf16"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_cache_messages/bboxmask_ckpt14392_context_bf16"
)
DEFAULT_DATASET_ROOT = Path("/data2/chenjq24/SpatialLM/spatiallm-dataset-link")


@dataclass
class MessageShardMeta:
    split: str
    shard: str
    item_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct processed messages for existing point-token scorer cache "
            "items. Output shards are aligned with the source cache shards, so "
            "later filtering can load messages by the same shard/item_index."
        )
    )
    parser.add_argument("--input_cache_root", type=Path, default=DEFAULT_INPUT_CACHE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset_json",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="Override source JSON for a split. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed used to reconstruct processed messages. If omitted, use "
            "source cache metadata['seed'] when present, otherwise fall back to 0."
        ),
    )
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument("--world_size", type=float, default=16.0)
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite message shard files owned by this shard_index.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        parser.error("--num_shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards.")
    if args.num_bins <= 0:
        parser.error("--num_bins must be positive.")
    if args.world_size <= 0:
        parser.error("--world_size must be positive.")
    if args.bbox_expand_ratio < 0:
        parser.error("--bbox_expand_ratio must be non-negative.")
    return args


def process_split(
    split: str,
    args: argparse.Namespace,
    dataset_overrides: dict[str, Path],
) -> list[MessageShardMeta]:
    input_dir = args.input_cache_root / split
    output_dir = args.output_root / split
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(input_dir)
    source_metadata = index["metadata"]
    samples_meta = index["samples"]
    base_seed = int(source_metadata.get("seed", 0) if args.seed is None else args.seed)

    dataset_json = resolve_dataset_json(
        split,
        source_metadata,
        dataset_overrides,
        args.dataset_root,
    )
    raw_samples = load_samples(dataset_json, max_samples=None)
    plugin = SpatialLMPlugin(
        num_bins=args.num_bins,
        world_size=args.world_size,
        do_augmentation=bool(source_metadata.get("do_augmentation", False)),
        random_rotation=bool(source_metadata.get("random_rotation", True)),
        point_token_bbox_mask=True,
        point_token_bbox_expand_ratio=args.bbox_expand_ratio,
    )

    by_shard: dict[str, list[dict[str, Any]]] = {}
    for meta in samples_meta:
        by_shard.setdefault(str(meta["shard"]), []).append(meta)

    shard_rels = list(by_shard)
    assigned_shards = [
        shard_rel
        for ordinal, shard_rel in enumerate(shard_rels)
        if ordinal % args.num_shards == args.shard_index
    ]

    metadata = {
        "format": "point_token_cache_messages_v1",
        "source_cache_dir": str(input_dir.resolve()),
        "source_cache_format": source_metadata.get("format"),
        "dataset_json": str(dataset_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "world_size": args.world_size,
        "num_bins": args.num_bins,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "seed": base_seed,
        "source_seed_present": "seed" in source_metadata,
        "random_rotation": bool(source_metadata.get("random_rotation", True)),
        "do_augmentation": bool(source_metadata.get("do_augmentation", False)),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }

    processed: list[MessageShardMeta] = []
    progress = tqdm(assigned_shards, desc=f"messages {split} shard {args.shard_index}")
    for shard_rel in progress:
        output_path = output_dir / shard_rel
        if output_path.exists() and not args.overwrite:
            processed.append(
                MessageShardMeta(
                    split=split,
                    shard=shard_rel,
                    item_count=len(by_shard[shard_rel]),
                )
            )
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shard_metas = by_shard[shard_rel]
        max_item_index = max(int(meta["item_index"]) for meta in shard_metas)
        items: list[dict[str, Any] | None] = [None] * (max_item_index + 1)

        for meta in shard_metas:
            item_index = int(meta["item_index"])
            messages = processed_messages_for_item(
                raw_samples[int(meta["sample_index"])],
                meta,
                plugin,
                args.dataset_root,
                base_seed,
            )
            items[item_index] = {
                "messages": messages,
                "epoch": int(meta["epoch"]),
                "sample_index": int(meta["sample_index"]),
                "point_cloud_index": int(meta["point_cloud_index"]),
                "scene_id": str(meta["scene_id"]),
                "point_cloud": str(meta["point_cloud"]),
            }

        if any(item is None for item in items):
            missing = [idx for idx, item in enumerate(items) if item is None]
            raise RuntimeError(f"Missing local item indices for {shard_rel}: {missing[:10]}")

        torch.save({"metadata": metadata, "items": items}, output_path)
        processed.append(
            MessageShardMeta(split=split, shard=shard_rel, item_count=len(items))
        )

    return processed


def main() -> None:
    args = parse_args()
    dataset_overrides = parse_dataset_json_overrides(args.dataset_json)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_processed: list[MessageShardMeta] = []
    for split in args.splits:
        all_processed.extend(process_split(split, args, dataset_overrides))

    index_path = (
        args.output_root
        / f"index_shard_{args.shard_index:05d}_of_{args.num_shards:05d}.json"
    )
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "processed_shards": [asdict(item) for item in all_processed],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote message-cache shard index: {index_path}")


if __name__ == "__main__":
    main()
