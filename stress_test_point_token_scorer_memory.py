#!/usr/bin/env python3
"""Stress test point-token scorer train/eval memory on longest cached samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from spatiallm.model.point_token_scorer import (
    DEFAULT_PROJECTOR_MODEL_PATH,
    PointTokenCacheDataset,
    PointTokenScorer,
    ScorerConfig,
    collate_batch,
    load_frozen_point_projector,
    masked_bce_loss,
    project_encoder_features,
)


DEFAULT_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/bboxmask_ckpt14392_context_bf16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build worst-case train/eval batches from precomputed top-token samples "
            "and repeat scorer train/eval loops to test peak GPU memory."
        )
    )
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--train_cache_dir", type=Path, default=None)
    parser.add_argument("--eval_cache_dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--prepare_topk_only", action="store_true")
    parser.add_argument(
        "--release_train_tensors_before_eval",
        action="store_true",
        help="Delete the last train batch tensors before entering eval.",
    )
    parser.add_argument(
        "--debug_memory",
        action="store_true",
        help="Print detailed GPU memory usage around the first eval batch.",
    )
    parser.add_argument(
        "--disable_eval_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable PyTorch Transformer eval fastpath during the eval stress loop.",
    )
    parser.add_argument("--projector_model_path", type=Path, default=DEFAULT_PROJECTOR_MODEL_PATH)
    parser.add_argument(
        "--projector_torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--pos_weight", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.train_cache_dir is None:
        args.train_cache_dir = args.cache_root / "train"
    if args.eval_cache_dir is None:
        args.eval_cache_dir = args.cache_root / "val"
    if args.train_bs <= 0 or args.eval_bs <= 0:
        parser.error("--train_bs and --eval_bs must be positive.")
    if args.train_bs > args.topk or args.eval_bs > args.topk:
        parser.error("--train_bs and --eval_bs must be <= --topk.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    return args


def topk_path(cache_root: Path, split: str, topk: int) -> Path:
    return cache_root / f"top{topk}_{split}_longest_samples.json"


def load_index(cache_dir: Path) -> dict[str, Any]:
    index_path = cache_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_topk(cache_dir: Path, output_path: Path, split: str, topk: int) -> None:
    index = load_index(cache_dir)
    samples = index["samples"]
    ranked = sorted(
        enumerate(samples),
        key=lambda item: int(item[1].get("token_count", 0)),
        reverse=True,
    )[:topk]
    top_samples = []
    for rank, (sample_index, sample) in enumerate(ranked):
        item = dict(sample)
        item["rank"] = rank
        item["dataset_index"] = sample_index
        top_samples.append(item)

    payload = {
        "split": split,
        "cache_dir": str(cache_dir),
        "topk": topk,
        "metadata": index.get("metadata", {}),
        "samples": top_samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {split} top{topk}: {output_path}")
    if top_samples:
        counts = [int(sample["token_count"]) for sample in top_samples]
        print(
            f"{split}: max={max(counts)}, min_top{topk}={min(counts)}, "
            f"mean_top{topk}={sum(counts) / len(counts):.1f}"
        )


def ensure_topk_files(args: argparse.Namespace) -> tuple[Path, Path]:
    train_topk_path = topk_path(args.cache_root, "train", args.topk)
    eval_topk_path = topk_path(args.cache_root, "val", args.topk)
    if not train_topk_path.is_file():
        write_topk(args.train_cache_dir, train_topk_path, "train", args.topk)
    if not eval_topk_path.is_file():
        write_topk(args.eval_cache_dir, eval_topk_path, "val", args.topk)
    return train_topk_path, eval_topk_path


def load_topk_entries(path: Path, batch_size: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload["samples"][:batch_size]
    if len(samples) < batch_size:
        raise ValueError(f"{path} has only {len(samples)} samples, need {batch_size}.")
    return samples


def load_items_from_entries(
    cache_dir: Path,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shard_cache: dict[str, dict[str, Any]] = {}
    items = []
    for entry in entries:
        shard_rel = str(entry["shard"])
        if shard_rel not in shard_cache:
            shard_cache[shard_rel] = torch.load(cache_dir / shard_rel, map_location="cpu")
        item = shard_cache[shard_rel]["items"][int(entry["item_index"])]
        items.append(
            {
                "features": item["features"],
                "grid_coord": item["grid_coord"],
                "region_center_grid_coord": item["region_center_grid_coord"],
                "labels": item["labels"].float(),
                "scene_id": item["scene_id"],
                "point_cloud": item["point_cloud"],
            }
        )
    return items


def make_pos_weight(args: argparse.Namespace, train_dataset: PointTokenCacheDataset, device: torch.device):
    if args.pos_weight == "auto":
        positive = train_dataset.positive_count
        negative = train_dataset.token_count - positive
        value = negative / max(positive, 1)
        print(f"Using auto pos_weight={value:.4f}")
        return torch.tensor(value, dtype=torch.float32, device=device)
    if str(args.pos_weight).lower() in {"none", "0"}:
        return None
    return torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)


def report_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(
        f"{prefix}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak_allocated={peak_allocated:.2f}GiB peak_reserved={peak_reserved:.2f}GiB"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_topk_path, eval_topk_path = ensure_topk_files(args)
    if args.prepare_topk_only:
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    train_dataset = PointTokenCacheDataset(args.train_cache_dir, shard_cache_size=1)
    point_projector = load_frozen_point_projector(
        args.projector_model_path,
        args.projector_torch_dtype,
        device,
    )
    with torch.no_grad():
        dummy_context = torch.zeros(
            1,
            1,
            train_dataset.feature_dim,
            dtype=next(point_projector.parameters()).dtype,
            device=device,
        )
        point_token_dim = int(point_projector(dummy_context).shape[-1])

    config = ScorerConfig(
        encoder_feature_dim=train_dataset.feature_dim,
        point_token_dim=point_token_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        coord_scale=train_dataset.coord_scale,
    )
    model = PointTokenScorer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = make_pos_weight(args, train_dataset, device)

    train_entries = load_topk_entries(train_topk_path, args.train_bs)
    eval_entries = load_topk_entries(eval_topk_path, args.eval_bs)
    print(
        f"train_bs={args.train_bs}, train token range="
        f"{train_entries[-1]['token_count']}..{train_entries[0]['token_count']}"
    )
    print(
        f"eval_bs={args.eval_bs}, eval token range="
        f"{eval_entries[-1]['token_count']}..{eval_entries[0]['token_count']}"
    )

    train_cpu_batch = collate_batch(load_items_from_entries(args.train_cache_dir, train_entries))
    eval_cpu_batch = collate_batch(load_items_from_entries(args.eval_cache_dir, eval_entries))
    print(
        f"train batch features={tuple(train_cpu_batch['features'].shape)}, "
        f"eval batch features={tuple(eval_cpu_batch['features'].shape)}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.iterations + 1):
        train_batch_gpu = {
            key: value.to(device, non_blocking=True)
            for key, value in train_cpu_batch.items()
        }
        point_tokens = project_encoder_features(point_projector, train_batch_gpu["features"])
        logits = model(
            point_tokens,
            train_batch_gpu["grid_coord"],
            train_batch_gpu["region_center_grid_coord"],
            train_batch_gpu["attention_mask"],
        )
        loss = masked_bce_loss(
            logits,
            train_batch_gpu["labels"],
            train_batch_gpu["attention_mask"],
            pos_weight,
        )
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            print(f"train step={step} loss={loss.item():.6f}")
            report_memory("train", device)

    model.eval()
    if args.release_train_tensors_before_eval:
        del train_batch_gpu, point_tokens, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report_memory("after releasing train tensors", device)

    fastpath_enabled = None
    if args.disable_eval_mha_fastpath and hasattr(torch.backends, "mha"):
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        print("Disabled torch.backends.mha fastpath for eval stress loop.")

    with torch.no_grad():
        try:
            for step in range(1, args.iterations + 1):
                if args.debug_memory and step == 1:
                    report_memory("eval before batch to device", device)
                eval_batch_gpu = {
                    key: value.to(device, non_blocking=True)
                    for key, value in eval_cpu_batch.items()
                }
                if args.debug_memory and step == 1:
                    print(f"eval batch features shape={tuple(eval_batch_gpu['features'].shape)}")
                    report_memory("eval after batch to device", device)
                eval_point_tokens = project_encoder_features(point_projector, eval_batch_gpu["features"])
                if args.debug_memory and step == 1:
                    print(f"eval point_tokens shape={tuple(eval_point_tokens.shape)}")
                    report_memory("eval after projector", device)
                eval_logits = model(
                    eval_point_tokens,
                    eval_batch_gpu["grid_coord"],
                    eval_batch_gpu["region_center_grid_coord"],
                    eval_batch_gpu["attention_mask"],
                )
                if args.debug_memory and step == 1:
                    report_memory("eval after scorer forward", device)
                eval_loss = masked_bce_loss(
                    eval_logits,
                    eval_batch_gpu["labels"],
                    eval_batch_gpu["attention_mask"],
                    pos_weight,
                )
                if step == 1 or step % 10 == 0:
                    print(f"eval step={step} loss={eval_loss.item():.6f}")
                    report_memory("eval", device)
        finally:
            if fastpath_enabled is not None:
                torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    report_memory("final", device)


if __name__ == "__main__":
    main()
