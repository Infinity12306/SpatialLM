#!/usr/bin/env python3
"""Train an end-to-end point-token attention scorer with frozen stage-2 SpatialLM."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.tuner.data.collator import _encode_messages_example
from spatiallm.tuner.data.template import (
    IGNORE_INDEX,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
)
from spatiallm.tuner.hparams.data_args import DataArguments
from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig


DEFAULT_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_scorer_cache/stage2_res16_ckpt14392_context_bf16"
)
DEFAULT_MESSAGE_CACHE_ROOT = Path(
    "/data2/chenjq24/SpatialLM/spatiallm-dataset-link/"
    "point_token_cache_messages/stage2_res16_ckpt14392_context_bf16"
)
DEFAULT_MODEL_PATH = Path(
    "/data2/chenjq24/SpatialLM/saves/hierarchical/"
    "stage2_bboxes_20000_res_16_max_4096/checkpoint-14392"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data2/chenjq24/SpatialLM/saves/point_token_attention_scorer/"
    "stage2_res16_ckpt14392_budget1536"
)
DEFAULT_CONFIG = Path(
    "/data2/chenjq24/SpatialLM/configs/scorer/stage2_attention_scorer.yaml"
)


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sdpa_backend_context(name: str):
    if name == "default":
        return nullcontext()
    backends = {
        "math": [SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
    }
    return sdpa_kernel(backends[name])


class AttentionScorerCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        message_cache_dir: Path,
        max_samples: int | None = None,
        max_point_tokens: int | None = 4096,
        shard_cache_size: int = 1,
    ):
        self.cache_dir = cache_dir
        self.message_cache_dir = message_cache_dir
        index_path = cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.metadata = self.index["metadata"]
        self.samples = self.index["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.max_point_tokens = max_point_tokens
        self.shard_cache_size = max(1, shard_cache_size)
        self._feature_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._message_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_cached(
        root: Path,
        shard_rel: str,
        cache: OrderedDict[str, dict[str, Any]],
        shard_cache_size: int,
    ) -> dict[str, Any]:
        if shard_rel in cache:
            shard = cache.pop(shard_rel)
            cache[shard_rel] = shard
            return shard
        path = root / shard_rel
        if not path.is_file():
            raise FileNotFoundError(path)
        shard = torch.load(path, map_location="cpu")
        cache[shard_rel] = shard
        while len(cache) > shard_cache_size:
            cache.popitem(last=False)
        return shard

    @staticmethod
    def _center_crop(
        tensor: torch.Tensor,
        max_point_tokens: int | None,
    ) -> torch.Tensor:
        if max_point_tokens is None or tensor.shape[0] <= max_point_tokens:
            return tensor
        excess = tensor.shape[0] - max_point_tokens
        start = excess // 2
        return tensor[start : start + max_point_tokens]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index]
        shard_rel = str(entry["shard"])
        item_index = int(entry["item_index"])

        feature_shard = self._load_cached(
            self.cache_dir,
            shard_rel,
            self._feature_cache,
            self.shard_cache_size,
        )
        message_shard = self._load_cached(
            self.message_cache_dir,
            shard_rel,
            self._message_cache,
            self.shard_cache_size,
        )

        item = feature_shard["items"][item_index]
        message_item = message_shard["items"][item_index]
        features = self._center_crop(item["features"], self.max_point_tokens)
        grid_coord = self._center_crop(item["grid_coord"], self.max_point_tokens)

        return {
            "features": features,
            "grid_coord": grid_coord,
            "region_center_grid_coord": item["region_center_grid_coord"],
            "messages": message_item["messages"],
            "scene_id": str(item.get("scene_id", entry.get("scene_id", ""))),
            "point_cloud": str(item.get("point_cloud", entry.get("point_cloud", ""))),
            "raw_token_count": int(item["features"].shape[0]),
        }

    @property
    def feature_dim(self) -> int:
        feature_dim = self.metadata.get("feature_dim")
        if feature_dim is not None:
            return int(feature_dim)
        return int(self[0]["features"].shape[-1])

    @property
    def coord_scale(self) -> float:
        return float(self.metadata.get("reduced_grid_size", 80))


class ShardLocalBatchSampler(BatchSampler):
    """Build batches from one shard at a time to avoid repeated large shard loads."""

    def __init__(
        self,
        dataset: AttentionScorerCacheDataset,
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
        self.skip_batches_once = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def skip_first_batches(self, num_batches: int) -> None:
        self.skip_batches_once = max(0, int(num_batches))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        skip_remaining = self.skip_batches_once
        self.skip_batches_once = 0

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
                if skip_remaining > 0:
                    skip_remaining -= 1
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


class AttentionScorerCollator:
    def __init__(
        self,
        tokenizer,
        template,
        cutoff_len: int,
        pad_to_multiple_of: int | None = 8,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.cutoff_len = int(cutoff_len)
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        if messages and messages[0].get("role") == "system":
            return str(messages[0].get("content", "")), messages[1:]
        return "", messages

    def _encode_messages(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        system, turns = self._split_system(messages)
        return _encode_messages_example(
            messages=turns,
            system=system,
            point_clouds=[],
            template=self.template,
            tokenizer=self.tokenizer,
            cutoff_len=self.cutoff_len,
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_messages(sample["messages"]) for sample in samples]
        max_text_len = max(len(input_ids) for input_ids, _ in encoded)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_text_len = ((max_text_len + multiple - 1) // multiple) * multiple

        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_text_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_text_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        text_attention_mask = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        for index, (cur_input_ids, cur_labels) in enumerate(encoded):
            length = len(cur_input_ids)
            input_ids[index, :length] = torch.tensor(cur_input_ids, dtype=torch.long)
            labels[index, :length] = torch.tensor(cur_labels, dtype=torch.long)
            text_attention_mask[index, :length] = 1

        max_point_len = max(sample["features"].shape[0] for sample in samples)
        feature_dim = samples[0]["features"].shape[-1]
        feature_dtype = samples[0]["features"].dtype
        features = torch.zeros(batch_size, max_point_len, feature_dim, dtype=feature_dtype)
        grid_coord = torch.zeros(batch_size, max_point_len, 3, dtype=torch.float32)
        point_attention_mask = torch.zeros(batch_size, max_point_len, dtype=torch.bool)
        centers = torch.zeros(batch_size, 3, dtype=torch.float32)
        raw_token_counts = torch.zeros(batch_size, dtype=torch.long)

        for index, sample in enumerate(samples):
            length = sample["features"].shape[0]
            features[index, :length] = sample["features"].to(feature_dtype)
            grid_coord[index, :length] = sample["grid_coord"].float()
            point_attention_mask[index, :length] = True
            centers[index] = sample["region_center_grid_coord"].float()
            raw_token_counts[index] = int(sample["raw_token_count"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "text_attention_mask": text_attention_mask,
            "features": features,
            "grid_coord": grid_coord,
            "region_center_grid_coord": centers,
            "point_attention_mask": point_attention_mask,
            "raw_token_counts": raw_token_counts,
        }


def build_tokenizer_and_template(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    data_args = DataArguments(
        template=args.template,
        cutoff_len=args.cutoff_len,
        num_bins=args.num_bins,
        world_size=args.world_size,
    )
    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        world_size=data_args.world_size,
        do_augmentation=False,
        random_rotation=False,
        point_token_bbox_mask=False,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template


def load_frozen_stage2_model(args: argparse.Namespace, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_name(args.model_torch_dtype),
        attn_implementation=args.attn_implementation,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "set_point_backbone_dtype") and getattr(model, "point_backbone", None) is not None:
        model.set_point_backbone_dtype(torch.float32)
    model.to(device)
    return model


@torch.no_grad()
def project_encoder_features(
    model: nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    projector = model.point_proj
    projector_dtype = next(projector.parameters()).dtype
    return projector(features.to(projector_dtype)).float()


def build_inserted_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    labels: torch.Tensor,
    point_tokens: torch.Tensor,
    point_attention_mask: torch.Tensor,
    point_log_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_embeds = model.model.embed_tokens(input_ids)
    new_input_embeds: list[torch.Tensor] = []
    new_attention_masks: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    new_key_biases: list[torch.Tensor] = []
    max_len = 0

    point_start_token_id = model.config.point_start_token_id
    point_end_token_id = model.config.point_end_token_id

    for batch_idx in range(input_ids.shape[0]):
        cur_input_ids = input_ids[batch_idx]
        cur_embeds = token_embeds[batch_idx]
        cur_text_mask = text_attention_mask[batch_idx]
        cur_labels = labels[batch_idx]
        cur_point_tokens = point_tokens[batch_idx][point_attention_mask[batch_idx]]
        cur_point_log_bias = point_log_bias[batch_idx][point_attention_mask[batch_idx]]

        point_start_positions = torch.where(cur_input_ids == point_start_token_id)[0]
        point_end_positions = torch.where(cur_input_ids == point_end_token_id)[0]
        if point_start_positions.numel() != 1 or point_end_positions.numel() != 1:
            raise ValueError(
                "Each sample must contain exactly one point start and one point end token, "
                f"got {point_start_positions.numel()} and {point_end_positions.numel()}."
            )
        point_start = int(point_start_positions[0].item())
        point_end = int(point_end_positions[0].item())

        inserted_embeds = torch.cat(
            (
                cur_embeds[: point_start + 1],
                cur_point_tokens.to(device=cur_embeds.device, dtype=cur_embeds.dtype),
                cur_embeds[point_end:],
            ),
            dim=0,
        )
        inserted_attention_mask = torch.cat(
            (
                cur_text_mask[: point_start + 1],
                torch.ones(
                    cur_point_tokens.shape[0],
                    dtype=cur_text_mask.dtype,
                    device=cur_text_mask.device,
                ),
                cur_text_mask[point_end:],
            ),
            dim=0,
        )
        inserted_labels = torch.cat(
            (
                cur_labels[: point_start + 1],
                torch.full(
                    (cur_point_tokens.shape[0],),
                    IGNORE_INDEX,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device,
                ),
                cur_labels[point_end:],
            ),
            dim=0,
        )
        inserted_key_bias = torch.cat(
            (
                torch.zeros(point_start + 1, dtype=point_log_bias.dtype, device=point_log_bias.device),
                cur_point_log_bias,
                torch.zeros(
                    cur_input_ids.shape[0] - point_end,
                    dtype=point_log_bias.dtype,
                    device=point_log_bias.device,
                ),
            ),
            dim=0,
        )

        new_input_embeds.append(inserted_embeds)
        new_attention_masks.append(inserted_attention_mask)
        new_labels.append(inserted_labels)
        new_key_biases.append(inserted_key_bias)
        max_len = max(max_len, inserted_embeds.shape[0])

    padded_embeds: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    padded_labels: list[torch.Tensor] = []
    padded_biases: list[torch.Tensor] = []
    for embeds, mask, cur_labels, key_bias in zip(
        new_input_embeds,
        new_attention_masks,
        new_labels,
        new_key_biases,
    ):
        pad_len = max_len - embeds.shape[0]
        if pad_len > 0:
            embeds = torch.cat(
                [
                    embeds,
                    torch.zeros(
                        pad_len,
                        embeds.shape[-1],
                        dtype=embeds.dtype,
                        device=embeds.device,
                    ),
                ],
                dim=0,
            )
            mask = F.pad(mask, (0, pad_len), value=0)
            cur_labels = F.pad(cur_labels, (0, pad_len), value=IGNORE_INDEX)
            key_bias = F.pad(key_bias, (0, pad_len), value=0.0)
        padded_embeds.append(embeds)
        padded_masks.append(mask)
        padded_labels.append(cur_labels)
        padded_biases.append(key_bias)

    return (
        torch.stack(padded_embeds, dim=0),
        torch.stack(padded_masks, dim=0),
        torch.stack(padded_labels, dim=0),
        torch.stack(padded_biases, dim=0),
    )


def build_4d_attention_mask(
    attention_mask: torch.Tensor,
    key_bias: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    min_dtype = torch.finfo(dtype).min

    blocked_future = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    causal_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    causal_mask = causal_mask.masked_fill(blocked_future, min_dtype)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len).clone()

    key_padding_mask = attention_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(key_padding_mask, min_dtype)
    return causal_mask + key_bias.to(dtype)[:, None, None, :]


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def capacity_loss(
    score_logits: torch.Tensor,
    point_attention_mask: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores = torch.sigmoid(score_logits).masked_fill(~point_attention_mask, 0.0)
    token_counts = point_attention_mask.sum(dim=1).float()
    sum_scores = scores.sum(dim=1)
    active = token_counts > float(budget)
    denom = (token_counts - float(budget)).clamp_min(1.0)
    per_sample = torch.where(
        active,
        F.relu((sum_scores - float(budget)) / denom),
        torch.zeros_like(sum_scores),
    )
    valid_scores = scores[point_attention_mask]
    hard_keep = (valid_scores >= 0.5).float().mean() if valid_scores.numel() else scores.new_tensor(0.0)
    stats = {
        "score_mean": float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
        "score_hard_keep_ratio": float(hard_keep.item()),
        "soft_keep_mean": float(sum_scores.mean().item()),
        "token_count_mean": float(token_counts.mean().item()),
    }
    return per_sample.mean(), stats


def forward_loss(
    frozen_model: nn.Module,
    scorer: PointTokenScorer,
    batch: dict[str, torch.Tensor],
    budget: int,
    sdpa_backend: str = "default",
) -> tuple[torch.Tensor, dict[str, float]]:
    point_tokens = project_encoder_features(frozen_model, batch["features"])
    score_logits = scorer(
        point_tokens,
        batch["grid_coord"],
        batch["region_center_grid_coord"],
        batch["point_attention_mask"],
    )
    point_log_bias = F.logsigmoid(score_logits).masked_fill(~batch["point_attention_mask"], 0.0)

    inputs_embeds, inserted_attention_mask, inserted_labels, key_bias = build_inserted_inputs(
        frozen_model,
        batch["input_ids"],
        batch["text_attention_mask"],
        batch["labels"],
        point_tokens,
        batch["point_attention_mask"],
        point_log_bias,
    )
    attention_mask_4d = build_4d_attention_mask(
        inserted_attention_mask,
        key_bias,
        inputs_embeds.dtype,
    )
    with sdpa_backend_context(sdpa_backend):
        outputs = frozen_model.model(
            input_ids=None,
            attention_mask=attention_mask_4d,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    logits = frozen_model.lm_head(outputs.last_hidden_state)
    lm_loss = causal_lm_loss(logits, inserted_labels)
    cap_loss, cap_stats = capacity_loss(
        score_logits,
        batch["point_attention_mask"],
        budget,
    )
    loss = lm_loss + cap_loss
    stats = {
        "loss": float(loss.detach().item()),
        "lm_loss": float(lm_loss.detach().item()),
        "capacity_loss": float(cap_loss.detach().item()),
        "seq_len": float(inserted_attention_mask.shape[1]),
        **cap_stats,
    }
    return loss, stats


def build_dataloader(
    dataset: AttentionScorerCacheDataset,
    collator: AttentionScorerCollator,
    batch_size: int,
    num_workers: int,
    shard_local_shuffle: bool,
    seed: int,
    drop_last: bool = False,
    train: bool = True,
):
    if train and shard_local_shuffle:
        sampler = ShardLocalBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
            drop_last=drop_last,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, sampler
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last if train else False,
    )
    return loader, None


@torch.no_grad()
def evaluate(
    frozen_model: nn.Module,
    scorer: PointTokenScorer,
    dataloader: DataLoader,
    device: torch.device,
    budget: int,
    sdpa_backend: str = "default",
    max_batches: int | None = None,
) -> dict[str, float]:
    scorer.eval()
    totals: dict[str, float] = {}
    total_batches = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        loss, stats = forward_loss(frozen_model, scorer, batch, budget, sdpa_backend)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        total_batches += 1
        if max_batches is not None and total_batches >= max_batches:
            break
    scorer.train()
    if total_batches == 0:
        return {}
    return {key: value / total_batches for key, value in totals.items()}


def save_checkpoint(
    output_dir: Path,
    scorer: PointTokenScorer,
    optimizer: torch.optim.Optimizer,
    step: int,
    scorer_config: ScorerConfig,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(scorer_config),
            "args": vars(args),
        },
        checkpoint_dir / "scorer.pt",
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        scorer_path = candidate / "scorer.pt"
        if not candidate.is_dir() or not scorer_path.is_file():
            continue
        suffix = candidate.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), scorer_path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def load_checkpoint(
    checkpoint_path: Path,
    scorer: PointTokenScorer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    scorer.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    print(f"Resumed scorer/optimizer from {checkpoint_path} at step {step}.")
    return step


def optimizer_steps_per_epoch(num_micro_batches: int, gradient_accumulation_steps: int) -> int:
    return math.ceil(num_micro_batches / gradient_accumulation_steps)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    remaining_steps: int,
):
    warmup_steps = math.ceil(remaining_steps * args.warmup_ratio) if remaining_steps > 0 else 0
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(remaining_steps, 1),
    )
    print(
        "LR scheduler: "
        f"type={args.lr_scheduler_type}, remaining_steps={remaining_steps}, "
        f"warmup_ratio={args.warmup_ratio}, warmup_steps={warmup_steps}"
    )
    return scheduler


def _is_wandb_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return not (math.isnan(value) or math.isinf(value))
    return False


def copy_wandb_history_until_step(wandb_module, args: argparse.Namespace, max_step: int) -> None:
    api = wandb_module.Api()
    entity = args.wandb_copy_from_entity or args.wandb_entity or api.default_entity
    project = args.wandb_copy_from_project or args.wandb_project
    source_path = f"{entity}/{project}/{args.wandb_copy_from_run_id}"
    source_run = api.run(source_path)

    rows: list[tuple[int, dict[str, Any]]] = []
    for row in source_run.scan_history(page_size=args.wandb_copy_page_size):
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        if step > max_step:
            continue

        metrics = {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and _is_wandb_scalar(value)
        }
        if metrics:
            rows.append((step, metrics))

    rows.sort(key=lambda item: item[0])
    for step, metrics in rows:
        wandb_module.log(metrics, step=step)

    if wandb_module.run is not None:
        wandb_module.run.summary["clean_copy_from_run_id"] = args.wandb_copy_from_run_id
        wandb_module.run.summary["clean_copy_source_path"] = source_path
        wandb_module.run.summary["clean_copy_max_step"] = max_step

    print(
        "Copied W&B history to clean run: "
        f"source={source_path}, max_step={max_step}, rows={len(rows)}"
    )


def maybe_init_wandb(args: argparse.Namespace, resume_checkpoint_step: int | None):
    if not args.use_wandb:
        return None
    import wandb

    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": wandb_config,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    wandb.init(**init_kwargs)
    if args.wandb_copy_from_run_id:
        if resume_checkpoint_step is None:
            raise RuntimeError(
                "wandb_copy_from_run_id requires local checkpoint resume so the copy cutoff step is known."
            )
        copy_wandb_history_until_step(wandb, args, resume_checkpoint_step)
    return wandb


PATH_KEYS = {
    "train_cache_dir",
    "eval_cache_dir",
    "train_message_cache_dir",
    "eval_message_cache_dir",
    "model_path",
    "output_dir",
}
INT_KEYS = {
    "num_bins",
    "cutoff_len",
    "max_point_tokens",
    "budget",
    "num_train_epochs",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "ffn_dim",
    "num_workers",
    "shard_cache_size",
    "max_train_samples",
    "max_eval_samples",
    "max_eval_batches",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "wandb_copy_page_size",
    "seed",
}
FLOAT_KEYS = {
    "world_size",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "dropout",
}
BOOL_KEYS = {
    "shard_local_shuffle",
    "drop_last",
    "tf32",
    "overwrite_output_dir",
    "use_wandb",
}


def default_config() -> dict[str, Any]:
    return {
        "train_cache_dir": DEFAULT_CACHE_ROOT / "train",
        "eval_cache_dir": DEFAULT_CACHE_ROOT / "val",
        "train_message_cache_dir": DEFAULT_MESSAGE_CACHE_ROOT / "train",
        "eval_message_cache_dir": DEFAULT_MESSAGE_CACHE_ROOT / "val",
        "model_path": DEFAULT_MODEL_PATH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "model_torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "sdpa_backend": "math",
        "template": "spatiallm_qwen",
        "cutoff_len": 8192,
        "world_size": 16.0,
        "num_bins": 1280,
        "max_point_tokens": 3584,
        "budget": 1536,
        "num_train_epochs": 4,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "hidden_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
        "num_workers": 8,
        "shard_cache_size": 1,
        "shard_local_shuffle": True,
        "drop_last": False,
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_eval_batches": None,
        "logging_steps": 20,
        "eval_steps": 500,
        "save_steps": 1000,
        "seed": 0,
        "device": "cuda",
        "tf32": True,
        "overwrite_output_dir": False,
        "use_wandb": True,
        "wandb_entity": None,
        "wandb_project": "spatiallm-point-token-scorer",
        "wandb_run_name": "stage2_res16_ckpt14392_budget1536_sdpa",
        "wandb_copy_from_run_id": None,
        "wandb_copy_from_project": None,
        "wandb_copy_from_entity": None,
        "wandb_copy_page_size": 1000,
    }


def _read_yaml_config(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        parser.error(f"Config must be a YAML mapping: {config_path}")
    return loaded


def _coerce_optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _coerce_optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _normalize_config(config: dict[str, Any], parser: argparse.ArgumentParser) -> dict[str, Any]:
    for key in PATH_KEYS:
        config[key] = _coerce_optional_path(config.get(key))
    for key in INT_KEYS:
        config[key] = _coerce_optional_int(config.get(key))
    for key in FLOAT_KEYS:
        config[key] = float(config[key])
    for key in BOOL_KEYS:
        value = config[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "y"}:
                config[key] = True
                continue
            if lowered in {"false", "0", "no", "n"}:
                config[key] = False
                continue
        parser.error(f"Config key {key!r} must be boolean, got {value!r}.")
    return config


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    dtype_choices = {"auto", "float32", "float16", "bfloat16"}
    if args.model_torch_dtype not in dtype_choices:
        parser.error(f"model_torch_dtype must be one of {sorted(dtype_choices)}.")
    if args.attn_implementation not in {"eager", "sdpa"}:
        parser.error("attn_implementation must be one of ['eager', 'sdpa'].")
    if args.sdpa_backend not in {"default", "math", "flash", "efficient", "cudnn"}:
        parser.error("sdpa_backend must be one of ['default', 'math', 'flash', 'efficient', 'cudnn'].")
    if args.lr_scheduler_type not in {"linear", "cosine", "constant", "constant_with_warmup"}:
        parser.error("lr_scheduler_type must be one of ['linear', 'cosine', 'constant', 'constant_with_warmup'].")
    if args.train_cache_dir is None or args.train_message_cache_dir is None:
        parser.error("train_cache_dir and train_message_cache_dir are required.")
    if (args.eval_cache_dir is None) != (args.eval_message_cache_dir is None):
        parser.error("eval_cache_dir and eval_message_cache_dir must be both set or both null.")
    if args.model_path is None or args.output_dir is None:
        parser.error("model_path and output_dir are required.")
    if args.max_point_tokens is not None and args.max_point_tokens <= 0:
        parser.error("max_point_tokens must be positive or null.")
    if args.budget <= 0:
        parser.error("budget must be positive.")
    if args.per_device_train_batch_size <= 0:
        parser.error("per_device_train_batch_size must be positive.")
    if args.per_device_eval_batch_size <= 0:
        parser.error("per_device_eval_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient_accumulation_steps must be positive.")
    if args.num_train_epochs <= 0:
        parser.error("num_train_epochs must be positive.")
    if args.num_workers < 0:
        parser.error("num_workers must be non-negative.")
    if args.shard_cache_size <= 0:
        parser.error("shard_cache_size must be positive.")
    if args.logging_steps <= 0 or args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error("logging_steps, eval_steps, and save_steps must be positive.")
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        parser.error("warmup_ratio must be in [0, 1].")
    if args.use_wandb and (not args.wandb_project or not args.wandb_run_name):
        parser.error("use_wandb=true requires non-empty wandb_project and wandb_run_name.")
    if args.wandb_copy_from_run_id and not args.use_wandb:
        parser.error("wandb_copy_from_run_id requires use_wandb=true.")
    if args.wandb_copy_page_size <= 0:
        parser.error("wandb_copy_page_size must be positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only a point-token scorer by adding logsigmoid scorer outputs "
            "as key-side attention bias to a frozen stage-2 SpatialLM."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG}.",
    )
    cli_args = parser.parse_args()
    config = default_config()
    yaml_config = _read_yaml_config(cli_args.config, parser)
    unknown_keys = sorted(set(yaml_config) - set(config))
    if unknown_keys:
        parser.error(f"Unknown config keys in {cli_args.config}: {unknown_keys}")
    config.update(yaml_config)
    config["config"] = cli_args.config
    args = argparse.Namespace(**_normalize_config(config, parser))
    validate_args(args, parser)
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_checkpoint_path: Path | None = None
    if args.output_dir.exists() and args.overwrite_output_dir:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        resume_checkpoint_path = latest_checkpoint(args.output_dir)
        if resume_checkpoint_path is None:
            print(
                f"Output directory exists but no checkpoint-*/scorer.pt was found: {args.output_dir}. "
                "Starting from scratch and keeping existing files."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer, template = build_tokenizer_and_template(args)

    train_dataset = AttentionScorerCacheDataset(
        args.train_cache_dir,
        args.train_message_cache_dir,
        max_samples=args.max_train_samples,
        max_point_tokens=args.max_point_tokens,
        shard_cache_size=args.shard_cache_size,
    )
    eval_dataset = None
    if args.eval_cache_dir is not None and args.eval_message_cache_dir is not None:
        eval_dataset = AttentionScorerCacheDataset(
            args.eval_cache_dir,
            args.eval_message_cache_dir,
            max_samples=args.max_eval_samples,
            max_point_tokens=args.max_point_tokens,
            shard_cache_size=args.shard_cache_size,
        )

    collator = AttentionScorerCollator(
        tokenizer=tokenizer,
        template=template,
        cutoff_len=args.cutoff_len,
        pad_to_multiple_of=8,
    )
    train_loader, train_sampler = build_dataloader(
        train_dataset,
        collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        shard_local_shuffle=args.shard_local_shuffle,
        seed=args.seed,
        drop_last=args.drop_last,
        train=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader, _ = build_dataloader(
            eval_dataset,
            collator,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_workers,
            shard_local_shuffle=False,
            seed=args.seed,
            train=False,
        )

    frozen_model = load_frozen_stage2_model(args, device)
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

    print(
        "Loaded frozen stage2 model and scorer: "
        f"context_dim={train_dataset.feature_dim}, point_token_dim={point_token_dim}, "
        f"train_items={len(train_dataset)}, eval_items={len(eval_dataset) if eval_dataset else 0}, "
        f"budget={args.budget}, max_point_tokens={args.max_point_tokens}"
    )

    micro_batches_per_epoch = len(train_loader)
    if micro_batches_per_epoch <= 0:
        raise RuntimeError("Train dataloader is empty; check train cache and batch settings.")
    steps_per_epoch = optimizer_steps_per_epoch(
        micro_batches_per_epoch,
        args.gradient_accumulation_steps,
    )
    total_training_steps = steps_per_epoch * args.num_train_epochs

    global_step = 0
    resume_checkpoint_step: int | None = None
    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, scorer, optimizer, device)
        resume_checkpoint_step = global_step
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["initial_lr"] = args.learning_rate

    wandb = maybe_init_wandb(args, resume_checkpoint_step)

    remaining_steps = max(total_training_steps - global_step, 0)
    scheduler = build_lr_scheduler(optimizer, args, remaining_steps)
    if global_step >= total_training_steps:
        print(
            f"No remaining training steps: global_step={global_step}, "
            f"total_training_steps={total_training_steps}."
        )
        if wandb is not None:
            wandb.finish()
        return

    start_epoch = min(global_step // steps_per_epoch, args.num_train_epochs)
    completed_steps_in_start_epoch = global_step % steps_per_epoch
    skip_micro_batches = min(
        completed_steps_in_start_epoch * args.gradient_accumulation_steps,
        micro_batches_per_epoch,
    )
    print(
        "Training schedule: "
        f"micro_batches_per_epoch={micro_batches_per_epoch}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_training_steps}, "
        f"start_step={global_step}, start_epoch={start_epoch}, "
        f"skip_micro_batches={skip_micro_batches}, remaining_steps={remaining_steps}"
    )

    running: dict[str, float] = {}
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_train_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            if epoch == start_epoch and skip_micro_batches > 0:
                train_sampler.skip_first_batches(skip_micro_batches)
        progress_initial = (
            skip_micro_batches
            if train_sampler is not None and epoch == start_epoch
            else 0
        )
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch}",
            total=micro_batches_per_epoch,
            initial=progress_initial,
        )
        trained_micro_steps = 0
        skip_in_loop = skip_micro_batches if train_sampler is None and epoch == start_epoch else 0

        for raw_micro_step, batch in enumerate(progress, start=1):
            if skip_in_loop > 0 and raw_micro_step <= skip_in_loop:
                continue
            if global_step >= total_training_steps:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss, stats = forward_loss(
                frozen_model,
                scorer,
                batch,
                args.budget,
                args.sdpa_backend,
            )
            (loss / args.gradient_accumulation_steps).backward()
            trained_micro_steps += 1

            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value)
            running_count += 1

            if trained_micro_steps % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                log = {
                    f"train/{key}": value / max(running_count, 1)
                    for key, value in running.items()
                }
                log["train/lr"] = scheduler.get_last_lr()[0]
                log["step"] = global_step
                log["epoch"] = epoch
                print(
                    f"step={global_step} epoch={epoch} "
                    + " ".join(
                        f"{key}={value:.6f}" for key, value in log.items() if key.startswith("train/")
                    )
                )
                if wandb is not None:
                    wandb.log(log, step=global_step)
                running = {}
                running_count = 0

            if eval_loader is not None and global_step % args.eval_steps == 0:
                del batch, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics = evaluate(
                    frozen_model,
                    scorer,
                    eval_loader,
                    device,
                    args.budget,
                    args.sdpa_backend,
                    max_batches=args.max_eval_batches,
                )
                print(
                    "eval "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                )
                if wandb is not None:
                    wandb.log({f"eval/{key}": value for key, value in metrics.items()}, step=global_step)

            if global_step % args.save_steps == 0:
                save_checkpoint(args.output_dir, scorer, optimizer, global_step, scorer_config, args)

        if (
            trained_micro_steps > 0
            and trained_micro_steps % args.gradient_accumulation_steps != 0
            and global_step < total_training_steps
        ):
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    save_checkpoint(args.output_dir, scorer, optimizer, global_step, scorer_config, args)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
