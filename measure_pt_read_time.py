#!/usr/bin/env python3
"""Measure sequential torch.load time for .pt cache shards."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_CACHE_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/stage2_res16_ckpt14392_context_bf16/train"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure sequential .pt read time.")
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pattern", default="**/*.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--touch_tensors",
        action="store_true",
        help="Touch loaded tensor data by summing first tensor values to force materialization.",
    )
    return parser.parse_args()


def count_items(obj: Any) -> int:
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return len(obj["items"])
    return 0


def touch_first_tensor(obj: Any) -> float:
    if isinstance(obj, torch.Tensor):
        return float(obj.flatten()[:1].float().sum().item()) if obj.numel() else 0.0
    if isinstance(obj, dict):
        for value in obj.values():
            result = touch_first_tensor(value)
            if result != 0.0:
                return result
    if isinstance(obj, list):
        for value in obj:
            result = touch_first_tensor(value)
            if result != 0.0:
                return result
    return 0.0


def main() -> None:
    args = parse_args()
    if not args.cache_dir.is_dir():
        raise FileNotFoundError(args.cache_dir)

    paths = sorted(args.cache_dir.glob(args.pattern))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt files found under {args.cache_dir} with pattern {args.pattern!r}")

    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        size_mb = path.stat().st_size / 1024**2
        start = time.perf_counter()
        obj = torch.load(path, map_location="cpu")
        if args.touch_tensors:
            _ = touch_first_tensor(obj)
        elapsed = time.perf_counter() - start
        item_count = count_items(obj)
        row = {
            "index": index,
            "path": str(path),
            "size_mb": size_mb,
            "item_count": item_count,
            "read_seconds": elapsed,
            "mb_per_sec": size_mb / elapsed if elapsed > 0 else 0.0,
        }
        rows.append(row)
        print(
            f"[{index}/{len(paths)}] {elapsed:.3f}s "
            f"{size_mb:.1f}MB {row['mb_per_sec']:.1f}MB/s items={item_count} {path}",
            flush=True,
        )
        del obj

    total_elapsed = time.perf_counter() - total_start
    read_times = [float(row["read_seconds"]) for row in rows]
    sizes = [float(row["size_mb"]) for row in rows]
    print("\nSummary")
    print(f"files={len(rows)}")
    print(f"total_time={total_elapsed:.3f}s")
    print(f"sum_read_time={sum(read_times):.3f}s")
    print(f"avg_read_time={sum(read_times) / len(read_times):.3f}s")
    print(f"min_read_time={min(read_times):.3f}s")
    print(f"max_read_time={max(read_times):.3f}s")
    print(f"total_size={sum(sizes):.1f}MB")
    print(f"avg_throughput={sum(sizes) / sum(read_times):.1f}MB/s")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "path", "size_mb", "item_count", "read_seconds", "mb_per_sec"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
