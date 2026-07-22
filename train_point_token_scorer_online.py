#!/usr/bin/env python3
"""Train a point-token scorer with online frozen SpatialLM point encoding."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.model.point_token_scorer import (
    PointTokenScorer,
    ScorerConfig,
    batched_point_token_bbox_overlap_labels,
    masked_bce_with_logits,
    packed_point_tokens_to_padded,
)
from spatiallm.tuner.data.mm_plugin import SpatialLMPlugin


DEFAULT_CONFIG = Path(
    "/data2/chenjq24/SpatialLM/configs/scorer/"
    "point_token_scorer_spatiallm_original.yaml"
)


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "train_dataset_json",
        "dataset_root",
        "model_name_or_path",
        "output_dir",
        "wandb_project",
        "wandb_run_name",
    }
    missing = sorted(key for key in required if not config.get(key))
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if config.get("lr_scheduler_type", "cosine") != "cosine":
        raise ValueError("Online scorer initialization requires cosine scheduling.")
    warmup_ratio = float(config.get("warmup_ratio", 0.03))
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1].")
    if float(config.get("point_token_bbox_expand_ratio", 0.1)) < 0:
        raise ValueError("point_token_bbox_expand_ratio must be non-negative.")
    for key in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "prefetch_factor",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(config.get("num_workers", 8)) < 0:
        raise ValueError("num_workers must be non-negative.")


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError(f"Unsupported model_torch_dtype: {name}")
    return choices[name]


def messages_from_sharegpt(sample: dict[str, Any]) -> list[dict[str, str]]:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    messages = []
    for message in sample.get("conversations", []):
        role = role_map.get(str(message.get("from", "")))
        if role is None:
            raise ValueError(f"Unsupported conversation role: {message!r}")
        messages.append({"role": role, "content": str(message.get("value", ""))})
    return messages


def sample_seed(
    base_seed: int,
    epoch: int,
    sample_index: int,
    point_cloud_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, epoch, sample_index, point_cloud_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class OnlinePointCloudDataset(Dataset):
    """Raw stage-2 examples flattened to one point cloud per item."""

    def __init__(
        self,
        dataset_json: Path,
        dataset_root: Path,
        seed: int,
        max_samples: int | None = None,
    ):
        with dataset_json.open("r", encoding="utf-8") as handle:
            samples = json.load(handle)
        if not isinstance(samples, list):
            raise ValueError(f"Expected JSON list: {dataset_json}")
        if max_samples is not None:
            samples = samples[: int(max_samples)]

        self.dataset_root = dataset_root
        self.seed = int(seed)
        self.epoch = 0
        self.items: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(samples):
            point_clouds = sample.get("point_clouds", [])
            if isinstance(point_clouds, str):
                point_clouds = [point_clouds]
            messages = messages_from_sharegpt(sample)
            for point_cloud_index, point_cloud_text in enumerate(point_clouds):
                point_cloud_path = Path(str(point_cloud_text))
                if not point_cloud_path.is_absolute():
                    point_cloud_path = dataset_root / point_cloud_path
                self.items.append(
                    {
                        "messages": messages,
                        "point_cloud": str(point_cloud_path),
                        "sample_index": sample_index,
                        "point_cloud_index": point_cloud_index,
                    }
                )
        if not self.items:
            raise ValueError(f"No point-cloud examples found in {dataset_json}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            **item,
            "augmentation_seed": sample_seed(
                self.seed,
                self.epoch,
                int(item["sample_index"]),
                int(item["point_cloud_index"]),
            ),
        }


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int, shuffle: bool):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class OnlinePointCloudCollator:
    """Apply synchronized augmentation and pack raw point clouds in workers."""

    def __init__(
        self,
        num_bins: int,
        world_size: float,
        do_augmentation: bool,
        random_rotation: bool,
        bbox_expand_ratio: float,
    ):
        self.plugin = SpatialLMPlugin(
            num_bins=num_bins,
            world_size=world_size,
            do_augmentation=do_augmentation,
            random_rotation=random_rotation,
            point_token_bbox_mask=False,
            point_token_bbox_expand_ratio=bbox_expand_ratio,
            point_cloud_batch_encoding=True,
            point_token_scorer_gt_mask=True,
        )

    @staticmethod
    def _seed_worker_transform(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        packed_clouds = []
        offsets = []
        bbox_list = []
        total_points = 0
        for sample in samples:
            self._seed_worker_transform(int(sample["augmentation_seed"]))
            mm_inputs = self.plugin._get_mm_inputs(
                [sample["messages"]],
                [sample["point_cloud"]],
            )
            point_cloud = mm_inputs["point_clouds"]
            if point_cloud is None or point_cloud.shape[0] == 0:
                raise ValueError(f"Empty point cloud: {sample['point_cloud']}")
            packed_clouds.append(point_cloud)
            total_points += int(point_cloud.shape[0])
            offsets.append(total_points)
            bbox_list.append(mm_inputs["point_token_scorer_gt_bboxes"][0])

        max_boxes = max((boxes.shape[0] for boxes in bbox_list), default=0)
        gt_bboxes = torch.full(
            (len(samples), max_boxes, 7),
            torch.nan,
            dtype=torch.float32,
        )
        for index, boxes in enumerate(bbox_list):
            if boxes.numel() > 0:
                gt_bboxes[index, : boxes.shape[0]] = boxes.float()

        return {
            "point_clouds": torch.cat(packed_clouds, dim=0).float(),
            "point_cloud_offsets": torch.tensor(offsets, dtype=torch.long),
            "point_token_scorer_gt_bboxes": gt_bboxes,
        }


def build_dataloader(
    dataset: OnlinePointCloudDataset,
    sampler: EpochShuffleSampler,
    collator: OnlinePointCloudCollator,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    drop_last: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": True,
        "drop_last": drop_last,
        # Workers are recreated after dataset.set_epoch(), making per-epoch
        # augmentation seeds visible without shared multiprocessing state.
        "persistent_workers": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError("Point projector contains no Linear layer.")


def last_linear(module: nn.Module) -> nn.Linear:
    layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not layers:
        raise ValueError("Point projector contains no Linear layer.")
    return layers[-1]


def load_frozen_point_modules(
    model_path: str | Path,
    model_dtype: str,
    num_bins: int,
    world_size: float,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.point_config["num_bins"] = int(num_bins)
    config.point_config["world_size"] = float(world_size)
    config.point_config["max_point_tokens"] = None
    spatial_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype(model_dtype),
    )
    spatial_model.set_point_backbone_dtype(torch.float32)
    spatial_model.requires_grad_(False)
    spatial_model.eval()

    point_encoder = spatial_model.point_backbone
    point_projector = spatial_model.point_proj
    spatial_model.point_backbone = None
    spatial_model.point_proj = None
    del spatial_model
    gc.collect()

    point_encoder.to(device=device, dtype=torch.float32)
    point_projector.to(device=device)
    point_encoder.requires_grad_(False).eval()
    point_projector.requires_grad_(False).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return point_encoder, point_projector


@torch.inference_mode()
def encode_packed_point_clouds(
    point_encoder: nn.Module,
    point_projector: nn.Module,
    point_clouds: torch.Tensor,
    offsets: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    coords = point_clouds[:, :3].to(
        device=device,
        dtype=torch.int64,
        non_blocking=True,
    )
    features = point_clouds[:, 3:].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    encoded = point_encoder(
        {
            "coord": features[:, :3],
            "grid_coord": coords,
            "feat": features,
            "offset": offsets.to(device=device, dtype=torch.long, non_blocking=True),
            "return_grid_coord": True,
        }
    )
    projector_dtype = next(point_projector.parameters()).dtype
    encoded["point_tokens"] = point_projector(
        encoded["context"].to(projector_dtype)
    ).float()
    return encoded


@torch.no_grad()
def prepare_scorer_batch(
    point_encoder: nn.Module,
    point_projector: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = encode_packed_point_clouds(
        point_encoder,
        point_projector,
        batch["point_clouds"],
        batch["point_cloud_offsets"],
        device,
    )
    batch_size = int(batch["point_cloud_offsets"].numel())
    point_tokens, grid_coord, attention_mask, centers = (
        packed_point_tokens_to_padded(
            encoded["point_tokens"],
            encoded["grid_coord"],
            encoded["batch"],
            batch_size,
        )
    )
    labels = batched_point_token_bbox_overlap_labels(
        grid_coord,
        attention_mask,
        batch["point_token_scorer_gt_bboxes"],
        float(point_encoder.final_voxel_size),
    )
    return point_tokens, grid_coord, attention_mask, centers, labels


def resolve_pos_weight(
    configured: Any,
    positive_tokens: int,
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if configured is None:
        return None
    if isinstance(configured, str):
        lowered = configured.lower()
        if lowered in {"none", "null", "0"}:
            return None
        if lowered == "auto":
            value = (total_tokens - positive_tokens) / max(positive_tokens, 1)
            return torch.tensor(value, dtype=torch.float32, device=device)
    value = float(configured)
    if value <= 0:
        raise ValueError("pos_weight must be positive, 'auto', or null.")
    return torch.tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(
    scorer: PointTokenScorer,
    point_encoder: nn.Module,
    point_projector: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    threshold: float,
) -> dict[str, float]:
    scorer.eval()
    total_loss = 0.0
    total_tokens = 0
    total_points = 0
    tp = fp = fn = tn = 0
    fastpath_enabled = None
    if hasattr(torch.backends, "mha"):
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        for batch in tqdm(dataloader, desc="eval", leave=False):
            total_points += int(batch["point_clouds"].shape[0])
            point_tokens, grid_coord, mask, centers, labels = prepare_scorer_batch(
                point_encoder,
                point_projector,
                batch,
                device,
            )
            logits = scorer(point_tokens, grid_coord, centers, mask)
            loss = masked_bce_with_logits(logits, labels, mask, pos_weight)
            predictions = (torch.sigmoid(logits) >= threshold) & mask
            targets = labels & mask
            tp += int((predictions & targets).sum().item())
            fp += int((predictions & ~targets & mask).sum().item())
            fn += int((~predictions & targets).sum().item())
            tn += int((~predictions & ~targets & mask).sum().item())
            tokens = int(mask.sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
    finally:
        if fastpath_enabled is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    scorer.train()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "positive_ratio": (tp + fn) / max(total_tokens, 1),
        "point_tokens": float(total_tokens),
        "input_points": float(total_points),
    }


def checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    if output_dir.is_dir():
        for child in output_dir.glob("checkpoint-*"):
            step = checkpoint_step(child)
            if step is not None and (child / "scorer.pt").is_file():
                candidates.append((step, child))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def serializable_args(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }


def save_checkpoint(
    output_dir: Path,
    scorer: PointTokenScorer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: dict[str, Any],
    label_stats: dict[str, int],
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": asdict(scorer.config),
            "args": serializable_args(config),
            "label_stats": dict(label_stats),
            "training_mode": "online_frozen_spatiallm_encoder",
        },
        checkpoint_dir / "scorer.pt",
    )


def load_checkpoint(
    checkpoint_dir: Path,
    scorer: PointTokenScorer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_dir / "scorer.pt", map_location=device)
    if checkpoint.get("training_mode") != "online_frozen_spatiallm_encoder":
        raise ValueError(
            f"Refusing to online-resume an incompatible scorer: {checkpoint_dir}"
        )
    scorer.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    label_stats = checkpoint.get("label_stats", {})
    stats = {
        "positive_tokens": int(label_stats.get("positive_tokens", 0)),
        "total_tokens": int(label_stats.get("total_tokens", 0)),
    }
    step = int(checkpoint["step"])
    print(f"Resumed online scorer from {checkpoint_dir} at step {step}.")
    return step, stats, checkpoint


def configure_wandb(config: dict[str, Any]) -> None:
    os.environ["WANDB_PROJECT"] = str(config["wandb_project"])
    os.environ["WANDB_NAME"] = str(config["wandb_run_name"])
    if config.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(config["wandb_entity"])
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)


def copy_wandb_history(
    wandb_module,
    config: dict[str, Any],
    max_step: int,
) -> None:
    source_id = config.get("wandb_copy_from_run_id")
    if not source_id:
        return
    entity = config.get("wandb_copy_from_entity") or config.get("wandb_entity")
    project = config.get("wandb_copy_from_project") or config["wandb_project"]
    if not entity:
        raise ValueError("W&B clean-copy requires a source entity.")
    source = wandb_module.Api().run(f"{entity}/{project}/{source_id}")
    copied = 0
    for row in source.scan_history(
        page_size=int(config.get("wandb_copy_page_size", 1000))
    ):
        step = row.get("_step")
        if step is None or int(step) > max_step:
            continue
        payload = {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and value is not None
        }
        if payload:
            wandb_module.log(payload, step=int(step))
            copied += 1
    print(f"Clean-copied W&B history through step {max_step}: rows={copied}")


def init_wandb(config: dict[str, Any], resume_step: int | None):
    if not bool(config.get("use_wandb", True)):
        return None
    import wandb

    wandb_config = {
        **serializable_args(config),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    init_kwargs = {
        "project": config["wandb_project"],
        "name": config["wandb_run_name"],
        "config": wandb_config,
    }
    if config.get("wandb_entity"):
        init_kwargs["entity"] = config["wandb_entity"]
    wandb.init(**init_kwargs)
    if config.get("wandb_copy_from_run_id"):
        if resume_step is None:
            raise RuntimeError(
                "wandb_copy_from_run_id requires a resumed local checkpoint."
            )
        copy_wandb_history(wandb, config, resume_step)
    return wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    cli_args = parser.parse_args()
    config = read_config(cli_args.config)
    validate_config(config)
    configure_wandb(config)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(config.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    requested_device = str(config.get("device", "cuda"))
    device = torch.device(
        requested_device
        if torch.cuda.is_available() or requested_device == "cpu"
        else "cpu"
    )
    output_dir = Path(config["output_dir"])
    resume_checkpoint = None
    if output_dir.exists() and bool(config.get("overwrite_output_dir", False)):
        shutil.rmtree(output_dir)
    elif output_dir.exists() and bool(
        config.get("auto_resume_from_latest_checkpoint", True)
    ):
        resume_checkpoint = latest_checkpoint(output_dir)
        if resume_checkpoint is None and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty but has no online scorer checkpoint: "
                f"{output_dir}"
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    point_encoder, point_projector = load_frozen_point_modules(
        config["model_name_or_path"],
        str(config.get("model_torch_dtype", "bfloat16")),
        int(config.get("num_bins", 1280)),
        float(config.get("world_size", 16.0)),
        device,
    )
    scorer_config = ScorerConfig(
        encoder_feature_dim=int(first_linear(point_projector).in_features),
        point_token_dim=int(last_linear(point_projector).out_features),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        coord_scale=float(point_encoder.reduced_grid_size),
    )
    scorer = PointTokenScorer(scorer_config).to(device)
    print(
        "Loaded frozen online point modules: "
        f"encoder_feature_dim={scorer_config.encoder_feature_dim}, "
        f"point_token_dim={scorer_config.point_token_dim}, "
        f"coord_scale={scorer_config.coord_scale}, "
        f"final_voxel_size={point_encoder.final_voxel_size}"
    )

    dataset_root = Path(config["dataset_root"])
    train_dataset = OnlinePointCloudDataset(
        Path(config["train_dataset_json"]),
        dataset_root,
        seed,
        config.get("max_train_samples"),
    )
    eval_dataset = None
    if config.get("eval_dataset_json"):
        eval_dataset = OnlinePointCloudDataset(
            Path(config["eval_dataset_json"]),
            dataset_root,
            seed + 1_000_000,
            config.get("max_eval_samples"),
        )

    collator_args = {
        "num_bins": int(config.get("num_bins", 1280)),
        "world_size": float(config.get("world_size", 16.0)),
        "do_augmentation": bool(config.get("do_augmentation", False)),
        "random_rotation": bool(config.get("random_rotation", True)),
        "bbox_expand_ratio": float(
            config.get("point_token_bbox_expand_ratio", 0.1)
        ),
    }
    train_sampler = EpochShuffleSampler(train_dataset, seed, shuffle=True)
    train_loader = build_dataloader(
        train_dataset,
        train_sampler,
        OnlinePointCloudCollator(**collator_args),
        int(config.get("per_device_train_batch_size", 4)),
        int(config.get("num_workers", 8)),
        int(config.get("prefetch_factor", 2)),
        bool(config.get("drop_last", False)),
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_sampler = EpochShuffleSampler(eval_dataset, seed, shuffle=False)
        eval_loader = build_dataloader(
            eval_dataset,
            eval_sampler,
            OnlinePointCloudCollator(**collator_args),
            int(config.get("per_device_eval_batch_size", 4)),
            int(config.get("num_workers", 8)),
            int(config.get("prefetch_factor", 2)),
            False,
        )

    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    global_step = 0
    label_stats = {"positive_tokens": 0, "total_tokens": 0}
    resumed_state = None
    if resume_checkpoint is not None:
        global_step, label_stats, resumed_state = load_checkpoint(
            resume_checkpoint,
            scorer,
            optimizer,
            device,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(config.get("learning_rate", 1e-4))
            group["initial_lr"] = float(config.get("learning_rate", 1e-4))

    gradient_accumulation_steps = int(
        config.get("gradient_accumulation_steps", 1)
    )
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )
    total_training_steps = steps_per_epoch * int(config.get("num_train_epochs", 4))
    remaining_steps = max(total_training_steps - global_step, 0)
    warmup_steps = int(remaining_steps * float(config.get("warmup_ratio", 0.03)))
    scheduler = get_scheduler(
        str(config.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(remaining_steps, 1),
    )
    if (
        resumed_state is not None
        and not bool(config.get("reset_scheduler_on_resume", True))
        and resumed_state.get("scheduler") is not None
    ):
        scheduler.load_state_dict(resumed_state["scheduler"])

    print(
        "Online scorer schedule: "
        f"examples={len(train_dataset)}, micro_batches_per_epoch={micro_batches_per_epoch}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_training_steps}, "
        f"start_step={global_step}, remaining_steps={remaining_steps}, "
        f"warmup_steps={warmup_steps}"
    )
    wandb = init_wandb(
        config,
        global_step if resume_checkpoint is not None else None,
    )
    if global_step >= total_training_steps:
        print("No remaining online scorer training steps.")
        if wandb is not None:
            wandb.finish()
        return

    start_epoch = min(
        global_step // steps_per_epoch,
        int(config.get("num_train_epochs", 4)),
    )
    completed_steps_in_epoch = global_step % steps_per_epoch
    skip_micro_batches = min(
        completed_steps_in_epoch * gradient_accumulation_steps,
        micro_batches_per_epoch,
    )
    logging_steps = int(config.get("logging_steps", 20))
    eval_steps = int(config.get("eval_steps", 500))
    save_steps = int(config.get("save_steps", 1000))
    threshold = float(config.get("threshold", 0.5))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    running_loss = 0.0
    running_tokens = 0
    running_examples = 0
    running_input_points = 0
    running_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, int(config.get("num_train_epochs", 4))):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        progress = tqdm(
            train_loader,
            desc=f"online scorer epoch {epoch}",
            total=micro_batches_per_epoch,
            initial=skip_micro_batches if epoch == start_epoch else 0,
        )
        trained_micro_steps = 0
        for raw_micro_step, batch in enumerate(progress, start=1):
            if epoch == start_epoch and raw_micro_step <= skip_micro_batches:
                continue
            if global_step >= total_training_steps:
                break

            batch_examples = int(batch["point_cloud_offsets"].numel())
            batch_input_points = int(batch["point_clouds"].shape[0])
            point_tokens, grid_coord, mask, centers, labels = prepare_scorer_batch(
                point_encoder,
                point_projector,
                batch,
                device,
            )
            positive_tokens = int(labels.sum().item())
            valid_tokens = int(mask.sum().item())
            label_stats["positive_tokens"] += positive_tokens
            label_stats["total_tokens"] += valid_tokens
            pos_weight = resolve_pos_weight(
                config.get("pos_weight", "auto"),
                label_stats["positive_tokens"],
                label_stats["total_tokens"],
                device,
            )
            logits = scorer(point_tokens, grid_coord, centers, mask)
            loss = masked_bce_with_logits(logits, labels, mask, pos_weight)
            (loss / gradient_accumulation_steps).backward()

            running_loss += float(loss.item()) * valid_tokens
            running_tokens += valid_tokens
            running_examples += batch_examples
            running_input_points += batch_input_points
            trained_micro_steps += 1
            if trained_micro_steps % gradient_accumulation_steps != 0:
                continue

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed = max(time.perf_counter() - running_started, 1e-6)
                log = {
                    "train/loss": running_loss / max(running_tokens, 1),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/positive_ratio": label_stats["positive_tokens"]
                    / max(label_stats["total_tokens"], 1),
                    "train/pos_weight": (
                        float(pos_weight.item()) if pos_weight is not None else 1.0
                    ),
                    "train/point_tokens_per_second": running_tokens / elapsed,
                    "train/input_points_per_second": running_input_points / elapsed,
                    "train/examples_per_second": running_examples / elapsed,
                    "epoch": epoch,
                }
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={log['train/loss']:.6f} "
                    f"positive_ratio={log['train/positive_ratio']:.6f} "
                    f"examples/s={log['train/examples_per_second']:.3f}"
                )
                if wandb is not None:
                    wandb.log(log, step=global_step)
                running_loss = 0.0
                running_tokens = 0
                running_examples = 0
                running_input_points = 0
                running_started = time.perf_counter()

            if eval_loader is not None and global_step % eval_steps == 0:
                del batch, point_tokens, grid_coord, mask, centers, labels, logits, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                eval_pos_weight = resolve_pos_weight(
                    config.get("pos_weight", "auto"),
                    label_stats["positive_tokens"],
                    label_stats["total_tokens"],
                    device,
                )
                metrics = evaluate(
                    scorer,
                    point_encoder,
                    point_projector,
                    eval_loader,
                    device,
                    eval_pos_weight,
                    threshold,
                )
                print(
                    "eval "
                    + " ".join(
                        f"{key}={value:.6f}" for key, value in metrics.items()
                    )
                )
                if wandb is not None:
                    wandb.log(
                        {f"eval/{key}": value for key, value in metrics.items()},
                        step=global_step,
                    )

            if global_step % save_steps == 0:
                save_checkpoint(
                    output_dir,
                    scorer,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                    label_stats,
                )

        if (
            trained_micro_steps > 0
            and trained_micro_steps % gradient_accumulation_steps != 0
            and global_step < total_training_steps
        ):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        skip_micro_batches = 0

    save_checkpoint(
        output_dir,
        scorer,
        optimizer,
        scheduler,
        global_step,
        config,
        label_stats,
    )
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
