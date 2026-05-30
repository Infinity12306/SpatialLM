import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from inference import DETECT_TYPE_PROMPT, preprocess_point_cloud
from spatiallm import Layout
from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd


POINT_PLACEHOLDER = "<point_cloud>"
POINT_PROMPT = "<|point_start|><|point_pad|><|point_end|>"
LAYOUT_START = "<|layout_s|>"
LAYOUT_END = "<|layout_e|>"


@dataclass
class TestExample:
    scene_id: str
    point_cloud_path: Path
    prompt: Optional[str] = None


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def load_code_template(path: Path) -> str:
    if path.exists():
        return path.read_text()

    script_relative_path = Path(__file__).resolve().parent / path
    if script_relative_path.exists():
        return script_relative_path.read_text()

    raise FileNotFoundError(f"Code template file not found: {path}")


def build_template_prompt(
    code_template: str,
    detect_type: str,
    categories: List[str],
) -> str:
    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    return (
        f"{POINT_PROMPT}{task_prompt} "
        f"The reference code is as followed: {code_template}"
    )


def normalize_json_prompt(prompt: str) -> str:
    prompt = prompt.replace(POINT_PLACEHOLDER, POINT_PROMPT)
    if POINT_PROMPT not in prompt:
        prompt = POINT_PROMPT + prompt
    return prompt


def first_user_prompt(example: dict[str, Any]) -> Optional[str]:
    conversations = example.get("conversations") or example.get("messages") or []
    for message in conversations:
        role = message.get("from") or message.get("role")
        if role in {"human", "user"}:
            content = message.get("value") or message.get("content")
            if content:
                return normalize_json_prompt(content)
    return None


def resolve_point_cloud_path(raw_path: str, media_root: Path, json_dir: Path) -> Path:
    point_cloud_path = Path(raw_path)
    if point_cloud_path.is_absolute():
        return point_cloud_path

    media_candidate = media_root / point_cloud_path
    if media_candidate.exists():
        return media_candidate

    json_candidate = json_dir / point_cloud_path
    if json_candidate.exists():
        return json_candidate

    return media_candidate


def load_json_examples(path: Path, media_root: Path, use_json_prompt: bool) -> List[TestExample]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("items") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {path}")

    examples = []
    seen_scene_ids = set()
    for index, item in enumerate(data):
        point_clouds = item.get("point_clouds") or item.get("point_cloud")
        if isinstance(point_clouds, list):
            if not point_clouds:
                continue
            raw_point_cloud = point_clouds[0]
        else:
            raw_point_cloud = point_clouds
        if not raw_point_cloud:
            continue

        point_cloud_path = resolve_point_cloud_path(
            str(raw_point_cloud), media_root, path.parent
        )
        scene_id = str(item.get("id") or point_cloud_path.stem)
        if scene_id in seen_scene_ids:
            scene_id = f"{scene_id}_{index:06d}"
        seen_scene_ids.add(scene_id)

        examples.append(
            TestExample(
                scene_id=scene_id,
                point_cloud_path=point_cloud_path,
                prompt=first_user_prompt(item) if use_json_prompt else None,
            )
        )
    return examples


def load_test_examples(
    test_data: Path,
    media_root: Optional[Path],
    use_json_prompt: bool,
) -> List[TestExample]:
    if test_data.is_dir():
        point_cloud_files = sorted(test_data.glob("*.ply"))
        return [
            TestExample(scene_id=point_cloud_path.stem, point_cloud_path=point_cloud_path)
            for point_cloud_path in point_cloud_files
        ]

    if test_data.suffix.lower() == ".json":
        return load_json_examples(
            test_data,
            media_root if media_root is not None else test_data.parent,
            use_json_prompt,
        )

    if test_data.suffix.lower() == ".ply":
        return [TestExample(scene_id=test_data.stem, point_cloud_path=test_data)]

    raise ValueError(
        f"Unsupported test data path: {test_data}. Expected a PLY file, JSON file, "
        "or a directory containing PLY files."
    )


def apply_subset_args(
    examples: List[TestExample],
    start_index: int,
    end_index: Optional[int],
    limit: Optional[int],
    num_shards: int,
    shard_index: int,
) -> List[TestExample]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < --num_shards")

    examples = examples[start_index:end_index]
    if num_shards > 1:
        examples = [
            example
            for index, example in enumerate(examples)
            if index % num_shards == shard_index
        ]
    if limit is not None:
        examples = examples[:limit]
    return examples


def make_conversation(model, prompt: str) -> list[dict[str, str]]:
    if model.config.model_type == "spatiallm_qwen":
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def clean_generated_text(text: str) -> str:
    return text.replace(LAYOUT_START, "").replace(LAYOUT_END, "").strip()


def generate_layout_text(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    num_beams: int,
    seed: int,
    greedy: bool,
) -> str:
    if seed >= 0:
        set_seed(seed)

    conversation = make_conversation(model, prompt)
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": point_cloud,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": not greedy,
        "use_cache": True,
    }
    if not greedy:
        generate_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
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


def prepare_point_cloud(point_cloud_path: Path, num_bins: int, no_cleanup: bool):
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")

    point_cloud = load_o3d_pcd(str(point_cloud_path))
    grid_size = Layout.get_grid_size(num_bins)
    if not no_cleanup:
        point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

    points, colors = get_points_and_colors(point_cloud)
    min_extent = np.min(points, axis=0)
    return preprocess_point_cloud(points, colors, grid_size, num_bins), min_extent


def predict_example(
    example: TestExample,
    model,
    tokenizer,
    template_prompt: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    prompt = example.prompt or template_prompt
    input_pcd, min_extent = prepare_point_cloud(
        example.point_cloud_path,
        model.config.point_config["num_bins"],
        args.no_cleanup,
    )
    generated_text = generate_layout_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        point_cloud=input_pcd,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams,
        seed=args.seed,
        greedy=args.greedy,
    )

    layout = Layout(generated_text)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    layout.translate(min_extent)
    output_path = output_dir / f"{example.scene_id}.txt"
    output_path.write_text(layout.to_language_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run SpatialLM checkpoint inference on a PLY directory/file or a processed JSON dataset file."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=Path,
        required=True,
        help="Path or Hugging Face ID for the model checkpoint.",
    )
    parser.add_argument(
        "-i",
        "--test_data",
        type=Path,
        required=True,
        help="Path to a PLY file, a directory of PLY files, or a processed JSON dataset file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where prediction .txt files will be written.",
    )
    parser.add_argument(
        "--media_root",
        type=Path,
        help="Base directory for relative point_clouds paths in JSON files. Defaults to the JSON parent directory.",
    )
    parser.add_argument(
        "--ignore_json_prompt",
        action="store_true",
        help="For JSON test data, ignore the stored user prompt and build one from --detect_type instead.",
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        choices=["all", "arch", "object"],
        default="all",
        help="Prompt type used for PLY inputs, or JSON inputs with --ignore_json_prompt.",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        help="Optional object categories for category-conditioned SpatialLM1.1 detection.",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=Path,
        default=Path("code_template.txt"),
        help="Path to the code template used when constructing prompts.",
    )
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_test_examples(
        args.test_data,
        args.media_root,
        use_json_prompt=not args.ignore_json_prompt,
    )
    examples = apply_subset_args(
        examples,
        args.start_index,
        args.end_index,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    if not examples:
        raise ValueError(f"No test examples found in {args.test_data}")

    code_template = load_code_template(args.code_template_file)
    template_prompt = build_template_prompt(
        code_template, args.detect_type, args.category
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=getattr(torch, args.inference_dtype)
    )
    model.to(args.device)
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    failures = []
    for example in tqdm(examples, desc="Testing checkpoint"):
        output_path = args.output_dir / f"{example.scene_id}.txt"
        if args.skip_existing and output_path.exists():
            continue

        try:
            predict_example(example, model, tokenizer, template_prompt, args.output_dir, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((example.scene_id, str(exc)))
            (args.output_dir / f"{example.scene_id}.error.txt").write_text(str(exc))

    if failures:
        print(f"Completed with {len(failures)} failure(s).")
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}")


if __name__ == "__main__":
    main()
