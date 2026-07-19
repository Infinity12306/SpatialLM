#!/usr/bin/env python3
"""Train stage-2 SpatialLM from offline scorer-filtered point tokens."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_scheduler,
)

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.tuner.data.collator import _encode_messages_example
from spatiallm.tuner.data.template import (
    IGNORE_INDEX,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
)
from spatiallm.tuner.hparams.data_args import DataArguments
from spatiallm.tuner.framework.utils import count_parameters


DEFAULT_CONFIG = Path(
    "/data2/chenjq24/SpatialLM/configs/"
    "spatiallm_stage2_filtered_point_token_scorer.yaml"
)


def torch_dtype_from_name(name: str) -> torch.dtype | str | None:
    if name in {None, "auto"}:
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is not None:
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(config: dict[str, Any]) -> str | None:
    explicit = config.get("resume_from_checkpoint")
    output_dir = Path(config["output_dir"])

    if explicit not in (None, "", "auto"):
        if isinstance(explicit, bool):
            if not explicit:
                return None
            latest = find_latest_checkpoint(output_dir)
            if latest is None:
                print(f"No checkpoint found in {output_dir}; starting from scratch.")
                return None
            print(f"Resuming from latest checkpoint: {latest}")
            return str(latest)
        checkpoint = Path(explicit)
        print(f"Resuming from configured checkpoint: {checkpoint}")
        return str(checkpoint)

    if config.get("overwrite_output_dir", False):
        return None

    if not config.get("auto_resume_from_latest_checkpoint", True):
        return None

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        if output_dir.exists():
            print(f"Output directory exists but no checkpoint was found: {output_dir}")
        return None

    print(f"Auto-resuming from latest checkpoint: {latest}")
    return str(latest)


def configure_wandb_env(config: dict[str, Any]) -> None:
    report_to = config.get("report_to", "none")
    if isinstance(report_to, str):
        reports = {report_to.lower()}
    else:
        reports = {str(item).lower() for item in report_to}
    if "wandb" not in reports:
        return

    wandb_project = config.get("wandb_project")
    if wandb_project:
        os.environ["WANDB_PROJECT"] = str(wandb_project)

    wandb_entity = config.get("wandb_entity")
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = str(wandb_entity)

    wandb_run_name = config.get("wandb_run_name") or config.get("run_name")
    if wandb_run_name:
        os.environ["WANDB_NAME"] = str(wandb_run_name)


class FilteredPointTokenDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_samples: int | None = None,
        max_point_tokens: int | None = None,
        shard_cache_size: int = 1,
    ):
        self.cache_dir = cache_dir
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

    @staticmethod
    def _center_crop_tokens(
        point_tokens: torch.Tensor,
        max_point_tokens: int | None,
    ) -> torch.Tensor:
        if max_point_tokens is None or point_tokens.shape[0] <= max_point_tokens:
            return point_tokens
        excess = point_tokens.shape[0] - max_point_tokens
        trim_start = excess // 2
        return point_tokens[trim_start : trim_start + max_point_tokens]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index]
        shard = self._load_shard(entry["shard"])
        item = shard["items"][entry["item_index"]]
        point_tokens = self._center_crop_tokens(
            item["point_tokens"],
            self.max_point_tokens,
        )
        if "messages" not in item:
            raise KeyError(
                f"Filtered cache item has no processed messages: "
                f"{entry['shard']}:{entry['item_index']}"
            )
        return {
            "point_tokens": point_tokens,
            "messages": item["messages"],
            "scene_id": item.get("scene_id", ""),
            "point_cloud": item.get("point_cloud", ""),
        }

    @property
    def point_token_dim(self) -> int:
        dim = self.metadata.get("point_token_dim")
        if dim is not None:
            return int(dim)
        return int(self[0]["point_tokens"].shape[-1])


class ShardLocalBatchSampler(BatchSampler):
    """Build batches from one shard at a time to avoid random 1GB shard loads."""

    def __init__(
        self,
        dataset: FilteredPointTokenDataset,
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


class FilteredPointTokenCollator:
    def __init__(
        self,
        tokenizer,
        template,
        cutoff_len: int,
        compute_dtype: torch.dtype,
        pad_to_multiple_of: int | None = 8,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.cutoff_len = int(cutoff_len)
        self.compute_dtype = compute_dtype
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
        max_len = max(len(input_ids) for input_ids, _ in encoded)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for index, (cur_input_ids, cur_labels) in enumerate(encoded):
            length = len(cur_input_ids)
            input_ids[index, :length] = torch.tensor(cur_input_ids, dtype=torch.long)
            labels[index, :length] = torch.tensor(cur_labels, dtype=torch.long)
            attention_mask[index, :length] = 1

        max_point_len = max(sample["point_tokens"].shape[0] for sample in samples)
        point_dim = samples[0]["point_tokens"].shape[-1]
        point_token_features = torch.full(
            (batch_size, max_point_len, point_dim),
            fill_value=float("nan"),
            dtype=self.compute_dtype,
        )
        for index, sample in enumerate(samples):
            point_tokens = sample["point_tokens"].to(self.compute_dtype)
            point_token_features[index, : point_tokens.shape[0]] = point_tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "point_token_features": point_token_features,
        }


class FilteredPointTokenTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        shard_local_shuffle: bool = True,
        reset_scheduler_on_resume: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shard_local_shuffle = shard_local_shuffle
        self.reset_scheduler_on_resume = reset_scheduler_on_resume
        self._num_training_steps: int | None = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self._num_training_steps = int(num_training_steps)
        return super().create_scheduler(num_training_steps, optimizer)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None or not self.reset_scheduler_on_resume:
            return

        step = checkpoint_step(Path(checkpoint))
        if step is None:
            return

        total_steps = self._num_training_steps
        if total_steps is None or total_steps <= step:
            return

        remaining_steps = total_steps - step
        if self.args.warmup_steps > 0:
            warmup_steps = min(int(self.args.warmup_steps), remaining_steps)
        else:
            warmup_steps = int(remaining_steps * float(self.args.warmup_ratio))

        for group in self.optimizer.param_groups:
            group["lr"] = self.args.learning_rate
            group["initial_lr"] = self.args.learning_rate

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=remaining_steps,
        )
        print(
            "Reset LR scheduler after resume: "
            f"checkpoint_step={step}, total_steps={total_steps}, "
            f"remaining_steps={remaining_steps}, warmup_steps={warmup_steps}, "
            f"scheduler={self.args.lr_scheduler_type}, base_lr={self.args.learning_rate}",
            flush=True,
        )

    def get_train_dataloader(self) -> DataLoader:
        if (
            not self.shard_local_shuffle
            or self.train_dataset is None
            or self.args.world_size != 1
        ):
            return super().get_train_dataloader()

        sampler = ShardLocalBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_training_args(config: dict[str, Any], has_eval: bool) -> Seq2SeqTrainingArguments:
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    report_to = config.get("report_to", "none")
    if isinstance(report_to, str) and report_to.lower() == "none":
        report_to = []

    kwargs: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "do_train": True,
        "do_eval": has_eval,
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "learning_rate": config.get("learning_rate", 1e-4),
        "weight_decay": config.get("weight_decay", 0.0),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": config.get("warmup_ratio", 0.0),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 1000),
        "save_total_limit": config.get("save_total_limit", None),
        "bf16": config.get("bf16", False),
        "fp16": config.get("fp16", False),
        "tf32": config.get("tf32", None),
        "overwrite_output_dir": config.get("overwrite_output_dir", False),
        "remove_unused_columns": False,
        "dataloader_num_workers": config.get("dataloader_num_workers", 0),
        "dataloader_pin_memory": config.get("dataloader_pin_memory", True),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "ddp_timeout": config.get("ddp_timeout", 180000000),
        "report_to": report_to,
        "run_name": config.get("wandb_run_name", config.get("run_name", None)),
        "ignore_data_skip": config.get("ignore_data_skip", False),
        "seed": config.get("seed", 42),
    }
    kwargs[eval_key] = config.get("eval_strategy", "steps" if has_eval else "no")
    if has_eval:
        kwargs["eval_steps"] = config.get("eval_steps", 500)

    filtered = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return Seq2SeqTrainingArguments(**filtered)


def build_tokenizer_and_template(config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", True),
        use_fast=config.get("use_fast_tokenizer", True),
        padding_side="right",
    )
    data_args = DataArguments(
        template=config.get("template", "spatiallm_qwen"),
        cutoff_len=config.get("cutoff_len", 8192),
        num_bins=config.get("num_bins", 1280),
        world_size=config.get("world_size", 16.0),
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


def load_filtered_point_token_model(config: dict[str, Any]):
    model_path = config["model_name_or_path"]
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=config.get("trust_remote_code", True),
    )
    model_config.point_config["num_bins"] = config.get("num_bins", 1280)
    model_config.point_config["world_size"] = config.get("world_size", 16.0)
    model_config.point_config["max_point_tokens"] = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=config.get("trust_remote_code", True),
        torch_dtype=torch_dtype_from_name(config.get("model_torch_dtype", "auto")),
    )

    if config.get("freeze_point_backbone", True):
        model.point_backbone.requires_grad_(False)
        model.set_point_backbone_dtype(torch.float32)

    if not config.get("pure_bf16", False):
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.train()
    trainable, total = count_parameters(model)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || "
        f"trainable%: {100 * trainable / total:.4f}"
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SpatialLM stage-2 from scorer-filtered point-token cache."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    configure_wandb_env(config)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    if config.get("tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer, template = build_tokenizer_and_template(config)
    train_dataset = FilteredPointTokenDataset(
        Path(config["filtered_point_token_dir"]),
        max_samples=args.max_train_samples or config.get("max_train_samples"),
        max_point_tokens=config.get("max_point_tokens"),
        shard_cache_size=config.get("shard_cache_size", 1),
    )
    eval_dir = config.get("eval_filtered_point_token_dir")
    eval_dataset = (
        FilteredPointTokenDataset(
            Path(eval_dir),
            max_samples=args.max_eval_samples or config.get("max_eval_samples"),
            max_point_tokens=config.get("max_point_tokens"),
            shard_cache_size=config.get("shard_cache_size", 1),
        )
        if eval_dir
        else None
    )

    compute_dtype = (
        torch.bfloat16
        if config.get("bf16", False) or config.get("pure_bf16", False)
        else torch.float16
        if config.get("fp16", False)
        else torch.float32
    )
    collator = FilteredPointTokenCollator(
        tokenizer=tokenizer,
        template=template,
        cutoff_len=config.get("cutoff_len", 8192),
        compute_dtype=compute_dtype,
        pad_to_multiple_of=8 if config.get("pad_to_multiple_of_8", True) else None,
    )
    model = load_filtered_point_token_model(config)
    training_args = build_training_args(config, has_eval=eval_dataset is not None)
    trainer = FilteredPointTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        shard_local_shuffle=config.get("shard_local_shuffle", True),
        reset_scheduler_on_resume=config.get("reset_scheduler_on_resume", True),
    )

    resume_from_checkpoint = resolve_resume_checkpoint(config)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
