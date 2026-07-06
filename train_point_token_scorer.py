#!/usr/bin/env python3
"""Train a transformer scorer for GT-BBoxMask point-token prediction."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses


DEFAULT_PROJECTOR_MODEL_PATH = Path(
    "/data2/chenjq24/SpatialLM/saves/hierarchical/"
    "stage2_bboxes_20000_res16_max4096_bbox_mask/checkpoint-14392"
)


@dataclass
class ScorerConfig:
    encoder_feature_dim: int
    point_token_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    coord_scale: float


class PointTokenScorer(nn.Module):
    def __init__(self, config: ScorerConfig):
        super().__init__()
        self.config = config
        self.coord_scale = float(config.coord_scale)
        input_dim = config.point_token_dim + 6
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.score_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        point_tokens: torch.Tensor,
        grid_coord: torch.Tensor,
        region_center_grid_coord: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        denom = max(self.coord_scale - 1.0, 1.0)
        grid_norm = grid_coord.float() / denom
        center_norm = region_center_grid_coord.float()[:, None, :] / denom
        center_norm = center_norm.expand(-1, point_tokens.shape[1], -1)
        x = torch.cat([point_tokens.float(), grid_norm, center_norm], dim=-1)
        x = self.input_mlp(x) + self.position_mlp(grid_norm)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        return self.score_head(x).squeeze(-1)


class PointTokenCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_samples: int | None = None,
        shard_cache_size: int = 2,
    ):
        self.cache_dir = cache_dir
        index_path = cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.metadata = self.index["metadata"]
        self.samples = self.index["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.shard_cache_size = max(1, shard_cache_size)
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_shard(self, shard_rel: str) -> dict[str, Any]:
        if shard_rel in self._shard_cache:
            shard = self._shard_cache.pop(shard_rel)
            self._shard_cache[shard_rel] = shard
            return shard

        shard = torch.load(self.cache_dir / shard_rel, map_location="cpu")
        self._shard_cache[shard_rel] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index]
        shard = self._load_shard(entry["shard"])
        item = shard["items"][entry["item_index"]]
        return {
            "features": item["features"],
            "grid_coord": item["grid_coord"],
            "region_center_grid_coord": item["region_center_grid_coord"],
            "labels": item["labels"].float(),
            "scene_id": item["scene_id"],
            "point_cloud": item["point_cloud"],
        }

    @property
    def feature_dim(self) -> int:
        feature_dim = self.metadata.get("feature_dim")
        if feature_dim is not None:
            return int(feature_dim)
        sample = self[0]
        return int(sample["features"].shape[-1])

    @property
    def coord_scale(self) -> float:
        return float(self.metadata.get("reduced_grid_size", 80))

    @property
    def positive_count(self) -> int:
        return int(sum(int(sample["positive_count"]) for sample in self.samples))

    @property
    def token_count(self) -> int:
        return int(sum(int(sample["token_count"]) for sample in self.samples))


class ShardLocalBatchSampler(BatchSampler):
    """Build batches from one shard at a time to improve shard cache locality."""

    def __init__(
        self,
        dataset: PointTokenCacheDataset,
        batch_size: int,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        self.shard_to_indices: dict[str, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            self.shard_to_indices.setdefault(str(sample["shard"]), []).append(index)
        self.shards = list(self.shard_to_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        shard_order = list(range(len(self.shards)))
        if self.shuffle_shards:
            shard_order = torch.randperm(len(self.shards), generator=generator).tolist()

        for shard_idx in shard_order:
            indices = list(self.shard_to_indices[self.shards[shard_idx]])
            if self.shuffle_within_shard:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.shard_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_len = max(sample["features"].shape[0] for sample in samples)
    feature_dim = samples[0]["features"].shape[-1]
    feature_dtype = samples[0]["features"].dtype

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=feature_dtype)
    grid_coord = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    centers = torch.zeros(batch_size, 3, dtype=torch.float32)

    for i, sample in enumerate(samples):
        length = sample["features"].shape[0]
        features[i, :length] = sample["features"].to(feature_dtype)
        grid_coord[i, :length] = sample["grid_coord"].float()
        labels[i, :length] = sample["labels"].float()
        attention_mask[i, :length] = True
        centers[i] = sample["region_center_grid_coord"].float()

    return {
        "features": features,
        "grid_coord": grid_coord,
        "region_center_grid_coord": centers,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-token mask scorer.")
    parser.add_argument("--train_cache_dir", type=Path, required=True)
    parser.add_argument("--eval_cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--projector_model_path",
        type=Path,
        default=DEFAULT_PROJECTOR_MODEL_PATH,
        help="SpatialLM checkpoint used to load the frozen point projector.",
    )
    parser.add_argument(
        "--projector_torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Dtype used when loading the frozen point projector.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_weight", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_cache_size", type=int, default=1)
    parser.add_argument(
        "--shard_local_shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shuffle shard order and shuffle samples within each shard, then build "
            "each train batch from a single shard."
        ),
    )
    parser.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop incomplete train batches.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--debug_memory",
        action="store_true",
        help="Print detailed GPU memory usage around the first eval batch.",
    )
    parser.add_argument(
        "--disable_eval_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable PyTorch MHA/Transformer eval fastpath during evaluation. "
            "This avoids very large explicit attention buffers for long sequences."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb_project", default="spatiallm-point-token-scorer")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    if args.num_train_epochs <= 0:
        parser.error("--num_train_epochs must be positive.")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    return args


def load_frozen_point_projector(
    model_path: Path,
    dtype_name: str,
    device: torch.device,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(dtype_name),
    )
    point_projector = model.point_proj
    point_projector.requires_grad_(False)
    point_projector.eval()
    point_projector.to(device)
    return point_projector


@torch.no_grad()
def project_encoder_features(
    point_projector: nn.Module,
    encoder_features: torch.Tensor,
) -> torch.Tensor:
    projector_dtype = next(point_projector.parameters()).dtype
    point_tokens = point_projector(encoder_features.to(projector_dtype))
    return point_tokens.float()


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    mask = attention_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def report_gpu_memory(prefix: str, device: torch.device) -> None:
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    point_projector: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    threshold: float,
    debug_memory: bool = False,
    disable_mha_fastpath: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    tp = fp = fn = tn = 0
    fastpath_enabled = None
    if disable_mha_fastpath and hasattr(torch.backends, "mha"):
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval", leave=False)):
            if debug_memory and batch_idx == 0:
                report_gpu_memory("eval before batch to device", device)
            batch = {key: value.to(device) for key, value in batch.items()}
            if debug_memory and batch_idx == 0:
                print(f"eval batch features shape={tuple(batch['features'].shape)}")
                report_gpu_memory("eval after batch to device", device)
            point_tokens = project_encoder_features(point_projector, batch["features"])
            if debug_memory and batch_idx == 0:
                print(f"eval point_tokens shape={tuple(point_tokens.shape)}")
                report_gpu_memory("eval after projector", device)
            logits = model(
                point_tokens,
                batch["grid_coord"],
                batch["region_center_grid_coord"],
                batch["attention_mask"],
            )
            if debug_memory and batch_idx == 0:
                report_gpu_memory("eval after scorer forward", device)
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["attention_mask"],
                pos_weight,
            )
            valid = batch["attention_mask"]
            pred = (torch.sigmoid(logits) >= threshold) & valid
            label = (batch["labels"] >= 0.5) & valid
            tp += int((pred & label).sum().item())
            fp += int((pred & ~label & valid).sum().item())
            fn += int((~pred & label & valid).sum().item())
            tn += int((~pred & ~label & valid).sum().item())
            tokens = int(valid.sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
    finally:
        if fastpath_enabled is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    model.train()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ScorerConfig,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
            "args": vars(args),
        },
        ckpt_dir / "scorer.pt",
    )


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    return wandb


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir.exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"Output directory exists: {args.output_dir}. "
                "Use --overwrite_output_dir to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_dataset = PointTokenCacheDataset(
        args.train_cache_dir,
        max_samples=args.max_train_samples,
        shard_cache_size=args.shard_cache_size,
    )
    eval_dataset = (
        PointTokenCacheDataset(
            args.eval_cache_dir,
            max_samples=args.max_eval_samples,
            shard_cache_size=args.shard_cache_size,
        )
        if args.eval_cache_dir is not None
        else None
    )

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
        dummy_point_tokens = point_projector(dummy_context)
        point_token_dim = int(dummy_point_tokens.shape[-1])
    print(
        "Loaded frozen point projector: "
        f"context_dim={train_dataset.feature_dim}, "
        f"point_token_dim={point_token_dim}, "
        f"dtype={next(point_projector.parameters()).dtype}"
    )

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

    if args.pos_weight == "auto":
        positive = train_dataset.positive_count
        negative = train_dataset.token_count - positive
        value = negative / max(positive, 1)
        pos_weight = torch.tensor(value, dtype=torch.float32, device=device)
        print(f"Using auto pos_weight={value:.4f}")
    elif args.pos_weight.lower() in {"none", "0"}:
        pos_weight = None
    else:
        pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)

    if args.shard_local_shuffle:
        train_batch_sampler = ShardLocalBatchSampler(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
            drop_last=args.drop_last,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )
        print(
            "Using shard-local train batching: "
            f"num_shards={len(train_batch_sampler.shards)}, "
            f"batch_size={args.per_device_train_batch_size}"
        )
    else:
        train_batch_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            drop_last=args.drop_last,
        )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )
        if eval_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    wandb = maybe_init_wandb(args)
    global_step = 0
    running_loss = 0.0
    running_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_train_epochs):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            point_tokens = project_encoder_features(point_projector, batch["features"])
            logits = model(
                point_tokens,
                batch["grid_coord"],
                batch["region_center_grid_coord"],
                batch["attention_mask"],
            )
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["attention_mask"],
                pos_weight,
            )
            (loss / args.gradient_accumulation_steps).backward()
            tokens = int(batch["attention_mask"].sum().item())
            running_loss += float(loss.item()) * tokens
            running_tokens += tokens

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                train_loss = running_loss / max(running_tokens, 1)
                log = {"train/loss": train_loss, "step": global_step, "epoch": epoch}
                print(f"step={global_step} epoch={epoch} train_loss={train_loss:.6f}")
                if wandb is not None:
                    wandb.log(log, step=global_step)
                running_loss = 0.0
                running_tokens = 0

            if eval_loader is not None and global_step % args.eval_steps == 0:
                # The current train batch is no longer needed after backward,
                # optimizer.step(), and logging. Releasing these references before
                # eval avoids overlapping the largest train and eval batches.
                del batch, point_tokens, logits, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics = evaluate(
                    model,
                    point_projector,
                    eval_loader,
                    device,
                    pos_weight,
                    args.threshold,
                    args.debug_memory,
                    args.disable_eval_mha_fastpath,
                )
                print(
                    "eval "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                )
                if wandb is not None:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)

            if global_step % args.save_steps == 0:
                save_checkpoint(args.output_dir, model, optimizer, global_step, config, args)

        if micro_step % args.gradient_accumulation_steps != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    save_checkpoint(args.output_dir, model, optimizer, global_step, config, args)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
