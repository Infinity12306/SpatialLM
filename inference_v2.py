#!/usr/bin/env python3
"""SpatialLM inference by scene id with GT/prediction render artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from generate_region_bboxes import (
    expand_region,
    load_points_and_colors,
    make_regions,
    parse_bboxes,
    remove_top_point_fraction,
    render_scene,
)
from inference import DETECT_TYPE_PROMPT, preprocess_point_cloud
from spatiallm import Layout
from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd


DATA_ROOT = Path("/data2/chenjq24/SpatialLM")
DEFAULT_PCD_DIR = DATA_ROOT / "spatiallm-dataset-link" / "pcd"
DEFAULT_LAYOUT_DIR = DATA_ROOT / "spatiallm-dataset-link" / "layout"
POINT_PROMPT = "<|point_start|><|point_pad|><|point_end|>"
LAYOUT_START = "<|layout_s|>"
LAYOUT_END = "<|layout_e|>"


OBJECT_CATEGORIES = [
    "sofa",
    "chair",
    "dining_chair",
    "bar_chair",
    "stool",
    "bed",
    "pillow",
    "wardrobe",
    "nightstand",
    "tv_cabinet",
    "wine_cabinet",
    "bathroom_cabinet",
    "shoe_cabinet",
    "entrance_cabinet",
    "decorative_cabinet",
    "washing_cabinet",
    "wall_cabinet",
    "sideboard",
    "cupboard",
    "coffee_table",
    "dining_table",
    "side_table",
    "dressing_table",
    "desk",
    "integrated_stove",
    "gas_stove",
    "range_hood",
    "micro-wave_oven",
    "sink",
    "stove",
    "refrigerator",
    "hand_sink",
    "shower",
    "shower_room",
    "toilet",
    "tub",
    "illumination",
    "chandelier",
    "floor-standing_lamp",
    "wall_decoration",
    "painting",
    "curtain",
    "carpet",
    "plants",
    "potted_bonsai",
    "tv",
    "computer",
    "air_conditioner",
    "washing_machine",
    "clothes_rack",
    "mirror",
    "bookcase",
    "cushion",
    "bar",
    "screen",
    "combination_sofa",
    "dining_table_combination",
    "leisure_table_and_chair_combination",
    "multifunctional_combination_bed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SpatialLM scene-id inference with per-scene GT/prediction render "
            "outputs."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="+",
        help="Scene id(s), or txt file(s) containing one scene id per line.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        required=True,
        help="Output root. Artifacts are written to out_dir/{scene_id}/.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model_path",
        dest="model_path",
        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument("--pcd_dir", "--pcd-dir", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument(
        "--layout_dir", "--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        choices=["all", "arch", "object"],
        default="all",
        help="Elements to detect.",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        choices=OBJECT_CATEGORIES,
        help="Optional object categories for category-conditioned detection.",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=Path,
        default=Path("code_template.txt"),
        help="Path to the code template file.",
    )
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic decoding instead of the original sampling defaults.",
    )
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--k", type=int, default=3, help="Number of regions per layout.")
    parser.add_argument(
        "--expand_fraction",
        "--expand-fraction",
        type=float,
        default=0.25,
        help="Fraction of original region scale added on each side.",
    )
    parser.add_argument(
        "--top_point_fraction",
        "--top-point-fraction",
        type=float,
        default=0.2,
        help="Highest-z point fraction removed before rendering.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--point_size", "--point-size", type=int, default=1)
    return parser.parse_args()


def read_scene_ids(inputs: list[str]) -> list[str]:
    scene_ids: list[str] = []
    seen: set[str] = set()

    for value in inputs:
        path = Path(value)
        if path.is_file():
            candidates = path.read_text(encoding="utf-8").splitlines()
        else:
            if path.suffix.lower() == ".txt":
                raise FileNotFoundError(f"Scene id file does not exist: {path}")
            candidates = [value]

        for candidate in candidates:
            scene_id = candidate.strip()
            if not scene_id or scene_id.startswith("#"):
                continue
            scene_id = Path(scene_id).stem
            if scene_id not in seen:
                scene_ids.append(scene_id)
                seen.add(scene_id)

    return scene_ids


def load_code_template(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")

    script_relative_path = Path(__file__).resolve().parent / path
    if script_relative_path.exists():
        return script_relative_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Code template file not found: {path}")


def build_prompt(code_template: str, detect_type: str, categories: list[str]) -> str:
    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    return (
        f"{POINT_PROMPT}{task_prompt} "
        f"The reference code is as followed: {code_template}"
    )


def make_conversation(model, prompt: str) -> list[dict[str, str]]:
    if model.config.model_type == "spatiallm_qwen":
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def clean_generated_text(text: str) -> str:
    return text.replace(LAYOUT_START, "").replace(LAYOUT_END, "").strip()


def prepare_point_cloud_for_model(
    pcd_path: Path, num_bins: int, no_cleanup: bool
) -> tuple[torch.Tensor, np.ndarray]:
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")

    point_cloud = load_o3d_pcd(str(pcd_path))
    grid_size = Layout.get_grid_size(num_bins)
    if not no_cleanup:
        point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

    points, colors = get_points_and_colors(point_cloud)
    min_extent = np.min(points, axis=0)
    input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
    return input_pcd, min_extent


def generate_layout_text(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
) -> str:
    if args.seed >= 0:
        set_seed(args.seed)

    conversation = make_conversation(model, prompt)
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": point_cloud,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": not args.greedy,
        "use_cache": True,
    }
    if not args.greedy:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1] :]
    return clean_generated_text(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    )


def predict_layout(
    model,
    tokenizer,
    prompt: str,
    pcd_path: Path,
    args: argparse.Namespace,
) -> str:
    input_pcd, min_extent = prepare_point_cloud_for_model(
        pcd_path, model.config.point_config["num_bins"], args.no_cleanup
    )
    generated_text = generate_layout_text(model, tokenizer, prompt, input_pcd, args)

    layout = Layout(generated_text)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    layout.translate(min_extent)
    return layout.to_language_string()


def render_layout_regions(
    layout_text: str,
    points: np.ndarray,
    colors: np.ndarray,
    output_prefix: Path,
    args: argparse.Namespace,
) -> None:
    object_bboxes = parse_bboxes(layout_text)
    raw_regions = make_regions(object_bboxes, args.k)
    expanded_regions = [
        expand_region(region, args.expand_fraction) for region in raw_regions
    ]

    render_scene(
        points,
        colors,
        object_bboxes,
        raw_regions,
        output_prefix.with_name(f"{output_prefix.name}_raw_render.png"),
        args.resolution,
        args.point_size,
        region_color=(220, 40, 40),
    )
    render_scene(
        points,
        colors,
        object_bboxes,
        expanded_regions,
        output_prefix.with_name(f"{output_prefix.name}_expanded_render.png"),
        args.resolution,
        args.point_size,
        region_color=(40, 80, 230),
    )


def expected_output_paths(scene_dir: Path) -> list[Path]:
    return [
        scene_dir / "pcd.ply",
        scene_dir / "gt_layout.txt",
        scene_dir / "pred_layout.txt",
        scene_dir / "gt_raw_render.png",
        scene_dir / "gt_expanded_render.png",
        scene_dir / "pred_raw_render.png",
        scene_dir / "pred_expanded_render.png",
    ]


def process_scene(
    scene_id: str,
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> None:
    scene_dir = args.out_dir / scene_id
    if args.skip_existing and all(path.exists() for path in expected_output_paths(scene_dir)):
        print(f"[SKIP] {scene_id}: outputs already exist")
        return

    pcd_path = args.pcd_dir / f"{scene_id}.ply"
    gt_layout_path = args.layout_dir / f"{scene_id}.txt"
    if not pcd_path.exists():
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    if not gt_layout_path.exists():
        raise FileNotFoundError(f"GT layout file not found: {gt_layout_path}")

    scene_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pcd_path, scene_dir / "pcd.ply")
    shutil.copy2(gt_layout_path, scene_dir / "gt_layout.txt")

    pred_layout_text = predict_layout(model, tokenizer, prompt, pcd_path, args)
    (scene_dir / "pred_layout.txt").write_text(pred_layout_text, encoding="utf-8")

    gt_layout_text = gt_layout_path.read_text(encoding="utf-8")
    points, colors = load_points_and_colors(pcd_path)
    points, colors, _ = remove_top_point_fraction(
        points, colors, args.top_point_fraction
    )

    render_layout_regions(gt_layout_text, points, colors, scene_dir / "gt", args)
    render_layout_regions(pred_layout_text, points, colors, scene_dir / "pred", args)


def main() -> int:
    args = parse_args()
    scene_ids = read_scene_ids(args.scenes)
    if not scene_ids:
        print("No scene ids found.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    code_template = load_code_template(args.code_template_file)
    prompt = build_prompt(code_template, args.detect_type, args.category)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=getattr(torch, args.inference_dtype)
    )
    model.to(args.device)
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    failures: list[tuple[str, str]] = []
    for scene_id in tqdm(scene_ids, desc="Inference"):
        try:
            process_scene(scene_id, model, tokenizer, prompt, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene_id, str(exc)))
            scene_dir = args.out_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "error.txt").write_text(str(exc), encoding="utf-8")

    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(f"Wrote outputs for {len(scene_ids)} scene(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
