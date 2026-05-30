#!/usr/bin/env python3
"""Run SpatialLM hierarchical two-stage SFT from a single config file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"base", "stage1", "stage2"}}


def latest_checkpoint(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Stage output directory does not exist: {output_dir}")

    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))

    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]

    model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    if any((output_dir / name).exists() for name in model_files):
        return output_dir

    raise FileNotFoundError(f"No checkpoint or model file found in {output_dir}")


def write_stage_config(stage_config: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stage_config, f, sort_keys=False)


def run_stage(train_script: Path, stage_name: str, stage_config: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{stage_name}.yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        config_path = Path(f.name)
        yaml.safe_dump(stage_config, f, sort_keys=False)

    print(f"[{stage_name}] config: {config_path}")
    print(f"[{stage_name}] dataset: {stage_config.get('dataset')}")
    print(f"[{stage_name}] output_dir: {stage_config.get('output_dir')}")
    subprocess.run(
        [sys.executable, str(train_script), str(config_path)],
        check=True,
    )


def set_stage_run_name(
    stage_config: dict[str, Any],
    stage_name: str,
    wandb_run_name: str | None,
) -> None:
    stage_wandb_run_name = stage_config.pop("wandb_run_name", None)
    if stage_config.get("run_name") is not None:
        return
    if stage_wandb_run_name:
        stage_config["run_name"] = stage_wandb_run_name
    elif wandb_run_name:
        stage_config["run_name"] = f"{wandb_run_name}_{stage_name}"


def build_stage_configs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    wandb_run_name = config.get("wandb_run_name")
    base = config.get("base", {})
    stage1 = deep_merge(base, config.get("stage1", {}))
    stage2 = deep_merge(base, config.get("stage2", {}))

    if "dataset" not in stage1:
        raise ValueError("stage1.dataset must be configured.")
    if "dataset" not in stage2:
        raise ValueError("stage2.dataset must be configured.")
    if "output_dir" not in stage1:
        raise ValueError("stage1.output_dir must be configured.")
    if "output_dir" not in stage2:
        raise ValueError("stage2.output_dir must be configured.")

    set_stage_run_name(stage1, "stage1", wandb_run_name)
    set_stage_run_name(stage2, "stage2", wandb_run_name)

    return flatten_config(stage1), flatten_config(stage2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-stage region SFT followed by second-stage bbox SFT."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--train_script",
        "--train-script",
        type=Path,
        default=Path("/data2/chenjq24/SpatialLM/train.py"),
    )
    parser.add_argument(
        "--skip_stage1",
        "--skip-stage1",
        action="store_true",
        help="Use the latest checkpoint in stage1.output_dir and only run stage2.",
    )
    parser.add_argument(
        "--skip_stage2",
        "--skip-stage2",
        action="store_true",
        help="Only run stage1.",
    )
    parser.add_argument(
        "--print_configs",
        "--print-configs",
        action="store_true",
        help="Print resolved stage configs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stage1_config, stage2_config = build_stage_configs(config)

    if args.print_configs:
        print(
            json.dumps(
                {
                    "stage1": stage1_config,
                    "stage2": stage2_config,
                },
                indent=2,
            )
        )
        return

    if not args.skip_stage1:
        run_stage(args.train_script, "stage1", stage1_config)

    if args.skip_stage2:
        return

    if stage2_config.get("model_name_or_path") in {None, "AUTO_STAGE1_LATEST"}:
        stage2_config["model_name_or_path"] = str(
            latest_checkpoint(Path(stage1_config["output_dir"]))
        )

    run_stage(args.train_script, "stage2", stage2_config)


if __name__ == "__main__":
    main()
