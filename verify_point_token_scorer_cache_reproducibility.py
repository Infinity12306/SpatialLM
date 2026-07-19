#!/usr/bin/env python3
"""Verify that point-token scorer cache items are reproducible from metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import spatiallm  # noqa: F401 - registers custom AutoConfig/AutoModel classes.
from precompute_point_token_scorer_data import (
    encode_one_point_cloud,
    load_samples,
    load_spatiallm_model,
    messages_from_sharegpt,
    resolve_path,
    sample_seed,
    storage_dtype,
)
from spatiallm.tuner.data.mm_plugin import SpatialLMPlugin


DEFAULT_CACHE_DIR = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/bboxmask_ckpt14392_context_bf16/train"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute selected scorer-cache items from dataset_json + PCD + "
            "epoch/sample_index/point_cloud_index and compare with cached tensors."
        )
    )
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset_json", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed used by precompute_point_token_scorer_data.py. If omitted, "
            "the script uses cache metadata['seed'] when present, otherwise 0."
        ),
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        default=None,
        help="Global indices in cache index.json to verify. Defaults to the first N.",
    )
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--random_selection", action="store_true")
    parser.add_argument("--selection_seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--storage_dtype",
        choices=["metadata", "float32", "float16", "bfloat16"],
        default="metadata",
        help="Dtype to cast recomputed encoder context before comparison.",
    )
    parser.add_argument(
        "--set_torch_seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also set torch/random seeds from the recovered sample seed. This is "
            "needed because Sonata may shuffle serialization orders with torch.randperm."
        ),
    )
    parser.add_argument("--feature_atol", type=float, default=0.0)
    parser.add_argument("--feature_rtol", type=float, default=0.0)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument(
        "--no_fail",
        action="store_true",
        help="Print mismatch report but return exit code 0.",
    )
    return parser.parse_args()


def load_index(cache_dir: Path) -> dict[str, Any]:
    index_path = cache_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing cache index: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(path_text: str | None, fallback_root: Path | None = None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_file() or path.is_dir():
        return path
    if fallback_root is not None:
        fallback = fallback_root / path.name
        if fallback.exists():
            return fallback
    return path


def choose_indices(args: argparse.Namespace, total: int) -> list[int]:
    if args.sample_indices:
        indices = args.sample_indices
    elif args.random_selection:
        rng = random.Random(args.selection_seed)
        indices = rng.sample(range(total), k=min(args.num_samples, total))
    else:
        indices = list(range(min(args.num_samples, total)))

    bad = [idx for idx in indices if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"Cache sample indices out of range: {bad}; total={total}")
    return indices


def load_cached_item(
    cache_dir: Path,
    item_meta: dict[str, Any],
    shard_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shard_rel = str(item_meta["shard"])
    if shard_rel not in shard_cache:
        shard_cache[shard_rel] = torch.load(cache_dir / shard_rel, map_location="cpu")
    shard = shard_cache[shard_rel]
    return shard["items"][int(item_meta["item_index"])]


def seed_all(sample_seed_value: int, use_torch_seed: bool) -> None:
    np.random.seed(sample_seed_value)
    if use_torch_seed:
        random.seed(sample_seed_value)
        torch.manual_seed(sample_seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed_value)


def tensor_stats(
    recomputed: torch.Tensor,
    cached: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    same_shape = tuple(recomputed.shape) == tuple(cached.shape)
    if not same_shape:
        return {
            "same_shape": False,
            "shape_recomputed": list(recomputed.shape),
            "shape_cached": list(cached.shape),
            "exact_equal": False,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    exact_equal = torch.equal(recomputed, cached)
    lhs = recomputed.float()
    rhs = cached.float()
    diff = (lhs - rhs).abs()
    return {
        "same_shape": True,
        "shape": list(recomputed.shape),
        "dtype_recomputed": str(recomputed.dtype),
        "dtype_cached": str(cached.dtype),
        "exact_equal": bool(exact_equal),
        "allclose": bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def main() -> None:
    args = parse_args()
    index = load_index(args.cache_dir)
    metadata = index["metadata"]
    cache_samples = index["samples"]
    base_seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)

    dataset_root = args.dataset_root or resolve_existing_path(
        metadata.get("dataset_root"), Path("/data2/chenjq24/SpatialLM/spatiallm-dataset-link")
    )
    dataset_json = args.dataset_json or resolve_existing_path(
        metadata.get("dataset_json"), dataset_root
    )
    model_path = args.model_path or Path(str(metadata["model_path"]))
    if dataset_root is None or dataset_json is None:
        raise ValueError("Cannot resolve dataset_root/dataset_json from args or metadata.")

    samples = load_samples(dataset_json, max_samples=None)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model_args = SimpleNamespace(
        model_path=model_path,
        num_bins=int(metadata["num_bins"]),
        world_size=float(metadata["world_size"]),
        torch_dtype=args.torch_dtype,
        device=str(device),
    )
    model = load_spatiallm_model(model_args)
    out_dtype_name = (
        str(metadata.get("storage_dtype", "bfloat16"))
        if args.storage_dtype == "metadata"
        else args.storage_dtype
    )
    out_dtype = storage_dtype(out_dtype_name)
    plugin = SpatialLMPlugin(
        num_bins=int(metadata["num_bins"]),
        world_size=float(metadata["world_size"]),
        do_augmentation=bool(metadata.get("do_augmentation", False)),
        random_rotation=bool(metadata.get("random_rotation", True)),
        point_token_bbox_mask=True,
        point_token_bbox_expand_ratio=float(metadata.get("bbox_expand_ratio", 0.1)),
    )

    selected_indices = choose_indices(args, len(cache_samples))
    shard_cache: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    ok = True

    print(f"cache_dir={args.cache_dir}")
    print(f"dataset_json={dataset_json}")
    print(f"dataset_root={dataset_root}")
    print(f"model_path={model_path}")
    print(f"base_seed={base_seed} metadata_seed_present={'seed' in metadata}")
    print(f"selected_cache_indices={selected_indices}")

    for cache_index in selected_indices:
        item_meta = cache_samples[cache_index]
        cached = load_cached_item(args.cache_dir, item_meta, shard_cache)
        recovered_seed = sample_seed(
            base_seed,
            int(item_meta["epoch"]),
            int(item_meta["sample_index"]),
            int(item_meta["point_cloud_index"]),
        )
        seed_all(recovered_seed, args.set_torch_seed)

        sample = samples[int(item_meta["sample_index"])]
        messages = messages_from_sharegpt(sample)
        pcd_path = resolve_path(str(item_meta["point_cloud"]), dataset_root)
        mm_inputs = plugin._get_mm_inputs([messages], [str(pcd_path)])
        point_cloud = mm_inputs["point_clouds"][0]
        keep_bboxes = mm_inputs["point_token_keep_bboxes"][0]
        features, grid_coord, center, labels = encode_one_point_cloud(
            model,
            point_cloud,
            keep_bboxes,
            device,
            out_dtype,
        )
        labels = labels.to(cached["labels"].dtype)

        feature_report = tensor_stats(
            features,
            cached["features"],
            args.feature_atol,
            args.feature_rtol,
        )
        grid_equal = torch.equal(grid_coord, cached["grid_coord"])
        label_equal = torch.equal(labels, cached["labels"])
        center_diff = (
            (center.float() - cached["region_center_grid_coord"].float()).abs().max().item()
            if center.shape == cached["region_center_grid_coord"].shape
            else None
        )
        item_ok = (
            feature_report["same_shape"]
            and feature_report["allclose"]
            and grid_equal
            and label_equal
            and center_diff == 0.0
        )
        ok = ok and item_ok

        report = {
            "cache_index": cache_index,
            "epoch": int(item_meta["epoch"]),
            "sample_index": int(item_meta["sample_index"]),
            "point_cloud_index": int(item_meta["point_cloud_index"]),
            "scene_id": item_meta.get("scene_id"),
            "point_cloud": item_meta.get("point_cloud"),
            "recovered_seed": int(recovered_seed),
            "grid_equal": bool(grid_equal),
            "label_equal": bool(label_equal),
            "center_max_abs_diff": None if center_diff is None else float(center_diff),
            "features": feature_report,
            "ok": bool(item_ok),
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    summary = {
        "ok": bool(ok),
        "cache_dir": str(args.cache_dir),
        "dataset_json": str(dataset_json),
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "base_seed": base_seed,
        "metadata_seed_present": "seed" in metadata,
        "set_torch_seed": bool(args.set_torch_seed),
        "feature_atol": args.feature_atol,
        "feature_rtol": args.feature_rtol,
        "num_checked": len(reports),
        "items": reports,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print("summary=" + json.dumps(summary, ensure_ascii=False))

    if (not ok) and (not args.no_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
