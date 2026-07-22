#!/usr/bin/env python3
"""Jointly train stage-2 SpatialLM and a hard point-token scorer online."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import TrainerCallback, get_scheduler

from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig
from spatiallm.tuner.data import (
    IGNORE_INDEX,
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
)
from spatiallm.tuner.framework.loader import load_model, load_tokenizer
from spatiallm.tuner.hparams import get_train_args
from spatiallm.tuner.trainer import CustomSeq2SeqTrainer


DEFAULT_CONFIG = Path(
    "/data2/chenjq24/SpatialLM/configs/spatiallm_stage2_joint_scorer.yaml"
)

JOINT_CONFIG_KEYS = {
    "scorer_init_path",
    "scorer_threshold",
    "scorer_min_keep",
    "scorer_max_keep",
    "scorer_loss_weight",
    "scorer_pos_weight",
    "scorer_detach_input",
    "auto_resume_from_latest_checkpoint",
    "reset_scheduler_on_resume",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_copy_from_run_id",
    "wandb_copy_from_project",
    "wandb_copy_from_entity",
    "wandb_copy_page_size",
}


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def checkpoint_step(path: Path | None) -> int | None:
    if path is None or not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def latest_scorer_file(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "scorer.pt"
    if direct.is_file():
        return direct
    candidates: list[tuple[int, Path]] = []
    if path.is_dir():
        for child in path.glob("checkpoint-*"):
            scorer_file = child / "scorer.pt"
            step = checkpoint_step(child)
            if step is not None and scorer_file.is_file():
                candidates.append((step, scorer_file))
    if not candidates:
        raise FileNotFoundError(f"No scorer.pt found under {path}")
    return max(candidates, key=lambda item: item[0])[1]


def load_initialized_scorer(path: Path) -> tuple[PointTokenScorer, dict[str, Any]]:
    scorer_file = latest_scorer_file(path)
    checkpoint = torch.load(scorer_file, map_location="cpu")
    scorer_config = ScorerConfig(**checkpoint["config"])
    scorer = PointTokenScorer(scorer_config)
    scorer.load_state_dict(checkpoint["model"])
    print(f"Loaded scorer initialization: {scorer_file}", flush=True)
    return scorer, checkpoint


def auto_pos_weight_from_scorer_checkpoint(checkpoint: dict[str, Any]) -> float:
    label_stats = checkpoint.get("label_stats", {})
    if label_stats:
        positive = int(label_stats["positive_tokens"])
        total = int(label_stats["total_tokens"])
        value = (total - positive) / max(positive, 1)
        print(
            "Resolved scorer_pos_weight="
            f"{value:.6f} from online scorer checkpoint label_stats",
            flush=True,
        )
        return value

    # Backward compatibility for offline cache-trained scorer checkpoints.
    scorer_args = checkpoint.get("args", {})
    cache_dir = scorer_args.get("train_cache_dir")
    if cache_dir is None:
        raise ValueError(
            "scorer_pos_weight=auto requires label_stats or train_cache_dir in "
            "the scorer checkpoint."
        )
    index_path = Path(cache_dir) / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)["metadata"]
    positive = int(metadata["total_positive_tokens"])
    total = int(metadata["total_tokens"])
    value = (total - positive) / max(positive, 1)
    print(
        f"Resolved scorer_pos_weight={value:.6f} from {index_path}", flush=True
    )
    return value


def resolve_pos_weight(value: Any, checkpoint: dict[str, Any]) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "auto":
            return auto_pos_weight_from_scorer_checkpoint(checkpoint)
        if lowered in {"none", "null", "0"}:
            return None
    numeric = float(value)
    if numeric <= 0:
        raise ValueError("scorer_pos_weight must be positive, 'auto', or null.")
    return numeric


def configure_wandb(config: dict[str, Any]) -> None:
    project = config.get("wandb_project")
    run_name = config.get("wandb_run_name") or config.get("run_name")
    if not project or not run_name:
        raise ValueError("wandb_project and wandb_run_name must both be configured.")
    os.environ["WANDB_PROJECT"] = str(project)
    os.environ["WANDB_NAME"] = str(run_name)
    if config.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(config["wandb_entity"])
    # Explicitly avoid W&B native resume; local Trainer checkpoints own resume state.
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
    source_entity = config.get("wandb_copy_from_entity") or config.get("wandb_entity")
    source_project = config.get("wandb_copy_from_project") or config["wandb_project"]
    if not source_entity:
        raise ValueError("W&B clean-copy requires a source entity.")
    source = wandb_module.Api().run(f"{source_entity}/{source_project}/{source_id}")
    page_size = int(config.get("wandb_copy_page_size", 1000))
    copied = 0
    for row in source.scan_history(page_size=page_size):
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
    print(
        f"Clean-copied W&B history through step {max_step}: rows={copied}",
        flush=True,
    )


class JointWandbCallback(TrainerCallback):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.initialized:
            return
        import wandb

        if wandb.run is None:
            return
        wandb.config.update(
            {
                **self.config,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            allow_val_change=True,
        )
        if self.config.get("wandb_copy_from_run_id"):
            if state.global_step <= 0:
                raise RuntimeError(
                    "wandb_copy_from_run_id requires a resumed local checkpoint."
                )
            copy_wandb_history(wandb, self.config, int(state.global_step))
        self.initialized = True


class JointPointTokenTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, reset_scheduler_on_resume: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_scheduler_on_resume = reset_scheduler_on_resume
        self._num_training_steps: int | None = None
        self._joint_metric_sums: dict[str, dict[str, torch.Tensor]] = {
            "train": {},
            "eval": {},
        }
        self._joint_metric_counts = {"train": 0, "eval": 0}

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        self._num_training_steps = int(num_training_steps)
        return super().create_scheduler(num_training_steps, optimizer)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None or not self.reset_scheduler_on_resume:
            return
        step = checkpoint_step(Path(checkpoint))
        total_steps = self._num_training_steps
        if step is None or total_steps is None or total_steps <= step:
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
            "Reset joint-training scheduler after resume: "
            f"checkpoint_step={step}, remaining_steps={remaining_steps}, "
            f"warmup_steps={warmup_steps}",
            flush=True,
        )

    @staticmethod
    def _unwrap(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        metrics = getattr(self._unwrap(model), "_last_joint_metrics", {})
        if metrics:
            split = "train" if model.training else "eval"
            sums = self._joint_metric_sums[split]
            for key, value in metrics.items():
                detached = value.detach()
                sums[key] = sums.get(key, detached.new_zeros(())) + detached
            self._joint_metric_counts[split] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        split = "eval" if any(key.startswith("eval") for key in logs) else "train"
        count = self._joint_metric_counts[split]
        if count > 0:
            prefix = "eval_joint" if split == "eval" else "joint"
            count_tensor = next(iter(self._joint_metric_sums[split].values())).new_tensor(
                float(count)
            )
            if dist.is_initialized():
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            for key, value in self._joint_metric_sums[split].items():
                reduced = value.clone()
                if dist.is_initialized():
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                logs[f"{prefix}/{key}"] = float((reduced / count_tensor).item())
            self._joint_metric_sums[split] = {}
            self._joint_metric_counts[split] = 0
        super().log(logs)

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super()._save(output_dir, state_dict)
        save_dir = Path(output_dir or self.args.output_dir)
        model = self._unwrap(self.model)
        scorer = getattr(model, "point_token_scorer", None)
        if scorer is None:
            return
        torch.save(
            {
                "model": scorer.state_dict(),
                "config": asdict(scorer.config),
                "joint_training": {
                    "threshold": getattr(model, "point_token_scorer_threshold", 0.5),
                    "min_keep": getattr(model, "point_token_scorer_min_keep", 1),
                    "max_keep": getattr(model, "point_token_scorer_max_keep", None),
                    "loss_weight": getattr(model, "point_token_scorer_loss_weight", 1.0),
                    "pos_weight": getattr(model, "point_token_scorer_pos_weight", None),
                },
            },
            save_dir / "scorer.pt",
        )


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model_name_or_path",
        "scorer_init_path",
        "output_dir",
        "wandb_project",
        "wandb_run_name",
    }
    missing = sorted(key for key in required if not config.get(key))
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if not config.get("point_cloud_batch_encoding", False):
        raise ValueError("Joint training requires point_cloud_batch_encoding: true")
    if not config.get("point_token_scorer_gt_mask", False):
        raise ValueError("Joint training requires point_token_scorer_gt_mask: true")
    if config.get("point_token_bbox_mask", False):
        raise ValueError("GT oracle filtering must be disabled during joint training.")
    if config.get("lr_scheduler_type", "cosine") != "cosine":
        raise ValueError("Joint training currently requires cosine LR scheduling.")
    if float(config.get("warmup_ratio", 0.03)) < 0:
        raise ValueError("warmup_ratio must be non-negative.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    cli_args = parser.parse_args()
    config = read_config(cli_args.config)
    validate_config(config)
    configure_wandb(config)

    trainer_config = {
        key: value for key, value in config.items() if key not in JOINT_CONFIG_KEYS
    }
    trainer_config["run_name"] = config["wandb_run_name"]
    trainer_config["report_to"] = "wandb"
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(trainer_config)
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        world_size=data_args.world_size,
        do_augmentation=data_args.do_augmentation,
        random_rotation=data_args.random_rotation,
        point_token_bbox_mask=False,
        point_token_bbox_expand_ratio=data_args.point_token_bbox_expand_ratio,
        point_cloud_batch_encoding=True,
        point_token_scorer_gt_mask=True,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(model_args, data_args, training_args)
    model = load_model(
        tokenizer, data_args, model_args, finetuning_args, is_trainable=True
    )

    scorer, scorer_checkpoint = load_initialized_scorer(
        Path(config["scorer_init_path"])
    )
    if scorer.config.point_token_dim != model.config.hidden_size:
        raise ValueError(
            "Scorer/model point-token dimension mismatch: "
            f"{scorer.config.point_token_dim} != {model.config.hidden_size}"
        )
    reduced_grid_size = float(model.point_backbone.reduced_grid_size)
    if float(scorer.config.coord_scale) != reduced_grid_size:
        raise ValueError(
            "Scorer/model coordinate scale mismatch: "
            f"{scorer.config.coord_scale} != {reduced_grid_size}"
        )

    model.point_token_scorer = scorer.to(
        device=next(model.parameters()).device, dtype=torch.float32
    )
    model.point_token_scorer.train()
    model.point_token_scorer_threshold = float(config.get("scorer_threshold", 0.5))
    model.point_token_scorer_min_keep = int(config.get("scorer_min_keep", 1))
    model.point_token_scorer_max_keep = int(
        config.get("scorer_max_keep") or data_args.max_point_tokens
    )
    model.point_token_scorer_loss_weight = float(
        config.get("scorer_loss_weight", 1.0)
    )
    model.point_token_scorer_pos_weight = resolve_pos_weight(
        config.get("scorer_pos_weight", "auto"), scorer_checkpoint
    )
    model.point_token_scorer_detach_input = bool(
        config.get("scorer_detach_input", True)
    )

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    callbacks = [JointWandbCallback(config)]
    trainer = JointPointTokenTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=generating_args.to_dict(obey_generation_config=True),
        reset_scheduler_on_resume=bool(
            config.get("reset_scheduler_on_resume", True)
        ),
        **dataset_module,
        **tokenizer_module,
    )

    resume_checkpoint = training_args.resume_from_checkpoint
    if not config.get("auto_resume_from_latest_checkpoint", True):
        resume_checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
