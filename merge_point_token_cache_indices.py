#!/usr/bin/env python3
"""Merge partial epoch-shard point-token cache indices into index.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge index_epoch_shard_*.json files under cache dirs."
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--message_cache_dir", type=Path, default=None)
    parser.add_argument("--pattern", default="index_epoch_shard_*.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "metadata" not in data or "samples" not in data:
        raise ValueError(f"Invalid cache index: {path}")
    return data


def sample_sort_key(sample: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(sample.get("epoch", 0)),
        int(sample.get("sample_index", 0)),
        int(sample.get("point_cloud_index", 0)),
    )


def merge_one_dir(cache_dir: Path, pattern: str, overwrite: bool) -> Path:
    paths = sorted(cache_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No partial indices found under {cache_dir} matching {pattern!r}")

    merged_samples: list[dict[str, Any]] = []
    merged_from: list[str] = []
    selected_epochs: set[int] = set()
    total_tokens = 0
    total_positive = 0
    has_token_totals = False

    first = load_json(paths[0])
    metadata = dict(first["metadata"])

    for path in paths:
        data = load_json(path)
        part_metadata = data["metadata"]
        samples = data["samples"]
        merged_samples.extend(samples)
        merged_from.append(path.name)

        for epoch in part_metadata.get("selected_epochs", []):
            selected_epochs.add(int(epoch))

        if "total_tokens" in part_metadata or any("token_count" in sample for sample in samples):
            has_token_totals = True
            total_tokens += int(
                part_metadata.get(
                    "total_tokens",
                    sum(int(sample.get("token_count", 0)) for sample in samples),
                )
            )
            total_positive += int(
                part_metadata.get(
                    "total_positive_tokens",
                    sum(int(sample.get("positive_count", 0)) for sample in samples),
                )
            )

    merged_samples.sort(key=sample_sort_key)
    metadata["selected_epochs"] = sorted(selected_epochs)
    metadata["partial_epoch_run"] = False
    metadata["merged_index"] = True
    metadata["merged_from"] = merged_from
    metadata["num_items"] = len(merged_samples)
    if has_token_totals:
        metadata["total_tokens"] = total_tokens
        metadata["total_positive_tokens"] = total_positive
        metadata["positive_ratio"] = (
            float(total_positive / total_tokens) if total_tokens > 0 else 0.0
        )

    output_path = cache_dir / "index.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Index exists: {output_path}. Use --overwrite to replace it.")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "samples": merged_samples}, handle, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    cache_index = merge_one_dir(args.cache_dir, args.pattern, args.overwrite)
    print(f"Wrote merged cache index: {cache_index}")
    if args.message_cache_dir is not None:
        message_index = merge_one_dir(args.message_cache_dir, args.pattern, args.overwrite)
        print(f"Wrote merged message-cache index: {message_index}")


if __name__ == "__main__":
    main()
