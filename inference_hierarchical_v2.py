#!/usr/bin/env python3
"""Inspection-oriented two-stage hierarchical SpatialLM inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from generate_region_bboxes import (
    expand_region,
    load_points_and_colors,
    make_regions,
    parse_bboxes,
    remove_top_point_fraction,
    render_scene,
    write_regions,
)
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    HierarchicalPrediction,
    SceneInput,
    load_model_and_tokenizer,
    predict_hierarchical_scene,
    write_prediction_outputs,
)
from render_topdown_arkitscenes import save_points_and_colors


DEFAULT_PCD_DIR = DEFAULT_DATASET_ROOT / "pcd"
DEFAULT_LAYOUT_DIR = DEFAULT_DATASET_ROOT / "layout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Two-stage hierarchical SpatialLM inference with per-scene artifacts"
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
        help="Output root. Per-scene artifacts are written to out_dir/{scene_id}/.",
    )
    parser.add_argument("--pcd_dir", "--pcd-dir", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument(
        "--layout_dir", "--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR
    )
    parser.add_argument(
        "--stage1_model_path",
        default="saves/hierarchical/stage1_regions",
    )
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes",
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
    parser.add_argument("--min_region_points", type=int, default=1)
    parser.add_argument(
        "--bbox_nms_iou",
        type=float,
        default=0.0,
        help="Class-wise bbox NMS IoU threshold. <=0 disables NMS.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--expand_fraction", type=float, default=0.25)
    parser.add_argument("--top_point_fraction", type=float, default=0.2)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--point_size", type=int, default=2)
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


def gt_regions_from_layout(layout_text: str, k: int, expand_fraction: float):
    raw_regions = make_regions(parse_bboxes(layout_text), k)
    expanded_regions = [
        expand_region(region, expand_fraction) for region in raw_regions
    ]
    return raw_regions, expanded_regions


def render_overlay(
    layout_text: str,
    regions,
    points,
    colors,
    output_path: Path,
    args: argparse.Namespace,
    region_color: tuple[int, int, int],
) -> None:
    render_scene(
        points,
        colors,
        parse_bboxes(layout_text),
        regions,
        output_path,
        args.resolution,
        args.point_size,
        region_color=region_color,
    )


def expected_scene_outputs(scene_dir: Path) -> list[Path]:
    return [
        scene_dir / "pcd.ply",
        scene_dir / "gt_layout.txt",
        scene_dir / "stage1_pred_layout.txt",
        scene_dir / "pred_layout.txt",
        scene_dir / "gt_render.png",
        scene_dir / "stage1_region_render.png",
        scene_dir / "pred_render.png",
        scene_dir / "metadata.json",
    ]


def write_region_artifacts(
    prediction: HierarchicalPrediction,
    scene_dir: Path,
) -> list[dict]:
    region_dir = scene_dir / "regions"
    region_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for region_prediction in prediction.region_predictions:
        prefix = f"region_{region_prediction.index}"
        region_text = "\n".join(
            bbox.to_language_string() for bbox in region_prediction.bboxes
        )
        (region_dir / f"{prefix}_pred_bboxes.txt").write_text(
            region_text,
            encoding="utf-8",
        )
        if (
            region_prediction.points is not None
            and region_prediction.colors is not None
            and region_prediction.points.shape[0] > 0
        ):
            save_points_and_colors(
                region_dir / f"{prefix}.ply",
                region_prediction.points,
                region_prediction.colors,
            )
        metadata.append(
            {
                "region_index": region_prediction.index,
                "point_count": region_prediction.point_count,
                "bbox_count": len(region_prediction.bboxes),
                "skipped": region_prediction.skipped,
                "skip_reason": region_prediction.skip_reason,
            }
        )
    return metadata


def process_scene(
    scene_id: str,
    stage1_model,
    stage1_tokenizer,
    stage2_model,
    stage2_tokenizer,
    args: argparse.Namespace,
) -> None:
    scene_dir = args.out_dir / scene_id
    if args.skip_existing and all(path.exists() for path in expected_scene_outputs(scene_dir)):
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

    prediction = predict_hierarchical_scene(
        SceneInput(scene_id=scene_id, pcd_path=pcd_path),
        stage1_model,
        stage1_tokenizer,
        stage2_model,
        stage2_tokenizer,
        args,
    )
    write_prediction_outputs(prediction, args.out_dir)
    (scene_dir / "stage1_pred_layout.txt").write_text(
        prediction.stage1_text,
        encoding="utf-8",
    )
    (scene_dir / "pred_layout.txt").write_text(
        prediction.final_text,
        encoding="utf-8",
    )

    gt_layout_text = gt_layout_path.read_text(encoding="utf-8")
    raw_gt_regions, expanded_gt_regions = gt_regions_from_layout(
        gt_layout_text,
        args.k,
        args.expand_fraction,
    )
    write_regions(scene_dir / "gt_raw_regions.txt", raw_gt_regions)
    write_regions(scene_dir / "gt_expanded_regions.txt", expanded_gt_regions)

    points, colors = load_points_and_colors(pcd_path)
    render_points, render_colors, render_info = remove_top_point_fraction(
        points,
        colors,
        args.top_point_fraction,
    )
    render_overlay(
        gt_layout_text,
        expanded_gt_regions,
        render_points,
        render_colors,
        scene_dir / "gt_render.png",
        args,
        region_color=(40, 80, 230),
    )
    render_overlay(
        prediction.stage1_text,
        prediction.stage1_layout.regions,
        render_points,
        render_colors,
        scene_dir / "stage1_region_render.png",
        args,
        region_color=(220, 40, 40),
    )
    render_overlay(
        prediction.final_text,
        prediction.stage1_layout.regions,
        render_points,
        render_colors,
        scene_dir / "pred_render.png",
        args,
        region_color=(220, 40, 40),
    )

    region_metadata = write_region_artifacts(prediction, scene_dir)
    metadata = {
        "scene_id": scene_id,
        "pcd_path": str(pcd_path),
        "gt_layout_path": str(gt_layout_path),
        "stage1_region_count": len(prediction.stage1_layout.regions),
        "final_bbox_count": len(prediction.final_layout.bboxes),
        "render": render_info,
        "regions": region_metadata,
    }
    (scene_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = read_scene_ids(args.scenes)
    if not scene_ids:
        print("No scene ids found.", file=sys.stderr)
        return 1

    stage1_model, stage1_tokenizer = load_model_and_tokenizer(
        args.stage1_model_path,
        args.inference_dtype,
        args.device,
    )
    stage2_model, stage2_tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )

    failures: list[tuple[str, str]] = []
    for scene_id in tqdm(scene_ids, desc="Hierarchical inference v2"):
        try:
            process_scene(
                scene_id,
                stage1_model,
                stage1_tokenizer,
                stage2_model,
                stage2_tokenizer,
                args,
            )
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
