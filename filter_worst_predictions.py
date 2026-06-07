import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from bbox import BBox3D
from bbox.metrics import iou_3d
from scipy.optimize import linear_sum_assignment

from generate_region_bboxes import expand_region, make_regions, parse_bboxes
from spatiallm import Layout
from spatiallm.layout.entity import Bbox, Region


LARGE_COST_VALUE = 1e6


@dataclass
class Match:
    pred_index: int
    gt_index: int
    iou: float


@dataclass
class SceneScore:
    scene_id: str
    pred_path: Path
    gt_path: Path
    pred_count: int
    gt_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_tp_iou: float
    mean_gt_best_iou: float
    mean_pred_best_iou: float
    matched_pairs: str
    unmatched_pred: str
    unmatched_gt: str


def canonical_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace("_", " ")


def read_label_mapping(path: Path, label_from: str, label_to: str) -> Dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        mapping = {}
        for row in reader:
            source = canonical_class_name(row[label_from])
            target = canonical_class_name(row[label_to])
            if source and target:
                mapping[source] = target
    return mapping


def load_bboxes(
    path: Path,
    class_map: Optional[Dict[str, str]],
    drop_unmapped: bool,
    minimum_scale: float,
) -> List[Bbox]:
    if not path.exists():
        return []

    layout = Layout(path.read_text())
    bboxes = []
    for bbox in layout.bboxes:
        class_name = canonical_class_name(bbox.class_name)
        if class_map is not None:
            mapped_class_name = class_map.get(class_name)
            if mapped_class_name is None:
                if drop_unmapped:
                    continue
            else:
                class_name = mapped_class_name
        bbox.class_name = class_name
        bbox.scale_x = max(bbox.scale_x, minimum_scale)
        bbox.scale_y = max(bbox.scale_y, minimum_scale)
        bbox.scale_z = max(bbox.scale_z, minimum_scale)
        bboxes.append(bbox)
    return bboxes


def clamp_region_scale(region: Region, minimum_scale: float) -> Region:
    region.scale_x = max(region.scale_x, minimum_scale)
    region.scale_y = max(region.scale_y, minimum_scale)
    region.scale_z = max(region.scale_z, minimum_scale)
    return region


def load_regions(path: Path, minimum_scale: float) -> List[Region]:
    if not path.exists():
        return []

    layout = Layout(path.read_text())
    return [clamp_region_scale(region, minimum_scale) for region in layout.regions]


def generated_region_to_layout_region(index: int, region: Any) -> Region:
    return Region(
        id=index,
        position_x=region.position_x,
        position_y=region.position_y,
        position_z=region.position_z,
        scale_x=region.scale_x,
        scale_y=region.scale_y,
        scale_z=region.scale_z,
    )


def derive_gt_regions(
    gt_path: Path,
    k: int,
    expand_fraction: float,
    minimum_scale: float,
) -> List[Region]:
    if not gt_path.exists():
        return []

    raw_regions = make_regions(parse_bboxes(gt_path.read_text()), k)
    expanded_regions = [expand_region(region, expand_fraction) for region in raw_regions]
    return [
        clamp_region_scale(generated_region_to_layout_region(index, region), minimum_scale)
        for index, region in enumerate(expanded_regions)
    ]


def load_gt_regions(
    scene_id: str,
    gt_dir: Path,
    gt_region_dir: Optional[Path],
    k: int,
    expand_fraction: float,
    minimum_scale: float,
) -> Tuple[List[Region], Path]:
    if gt_region_dir is not None:
        region_path = gt_region_dir / f"{scene_id}.txt"
        if region_path.exists():
            return load_regions(region_path, minimum_scale), region_path

    gt_path = gt_dir / f"{scene_id}.txt"
    return derive_gt_regions(gt_path, k, expand_fraction, minimum_scale), gt_path


def bbox_to_bbox3d(bbox: Any) -> BBox3D:
    return BBox3D(
        bbox.position_x,
        bbox.position_y,
        bbox.position_z,
        bbox.scale_x,
        bbox.scale_y,
        bbox.scale_z,
        euler_angles=[0, 0, getattr(bbox, "angle_z", 0.0)],
        is_center=True,
    )


def compute_iou_matrix(pred_bboxes: List[Bbox], gt_bboxes: List[Bbox]) -> np.ndarray:
    if not pred_bboxes or not gt_bboxes:
        return np.zeros((len(pred_bboxes), len(gt_bboxes)), dtype=np.float32)

    pred_boxes = [bbox_to_bbox3d(bbox) for bbox in pred_bboxes]
    gt_boxes = [bbox_to_bbox3d(bbox) for bbox in gt_bboxes]
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            try:
                iou_matrix[pred_index, gt_index] = iou_3d(pred_box, gt_box)
            except Exception:
                iou_matrix[pred_index, gt_index] = 0.0
    return iou_matrix


def grouped_indices(bboxes: List[Any]) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for index, bbox in enumerate(bboxes):
        groups[getattr(bbox, "class_name", getattr(bbox, "entity_label", "entity"))].append(index)
    return groups


def match_bboxes(
    pred_bboxes: List[Bbox],
    gt_bboxes: List[Bbox],
    iou_matrix: np.ndarray,
    iou_threshold: float,
    class_aware: bool,
) -> List[Match]:
    if not pred_bboxes or not gt_bboxes:
        return []

    if class_aware:
        pred_groups = grouped_indices(pred_bboxes)
        gt_groups = grouped_indices(gt_bboxes)
        class_groups = [
            (pred_groups[class_name], gt_groups[class_name])
            for class_name in sorted(set(pred_groups) & set(gt_groups))
        ]
    else:
        class_groups = [(list(range(len(pred_bboxes))), list(range(len(gt_bboxes))))]

    matches = []
    for pred_indices, gt_indices in class_groups:
        sub_iou = iou_matrix[np.ix_(pred_indices, gt_indices)]
        cost_matrix = np.full(sub_iou.shape, LARGE_COST_VALUE, dtype=np.float32)
        cost_matrix[sub_iou >= iou_threshold] = 1.0 - sub_iou[sub_iou >= iou_threshold]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row, col in zip(row_indices, col_indices):
            iou = float(sub_iou[row, col])
            if iou >= iou_threshold:
                matches.append(Match(pred_indices[row], gt_indices[col], iou))
    return matches


def best_iou_means(
    pred_bboxes: List[Bbox],
    gt_bboxes: List[Bbox],
    iou_matrix: np.ndarray,
    class_aware: bool,
) -> Tuple[float, float]:
    if not pred_bboxes and not gt_bboxes:
        return 1.0, 1.0
    if not pred_bboxes or not gt_bboxes:
        return 0.0, 0.0

    valid_iou = iou_matrix.copy()
    if class_aware:
        for pred_index, pred_bbox in enumerate(pred_bboxes):
            for gt_index, gt_bbox in enumerate(gt_bboxes):
                if pred_bbox.class_name != gt_bbox.class_name:
                    valid_iou[pred_index, gt_index] = 0.0

    mean_gt_best_iou = float(valid_iou.max(axis=0).mean()) if gt_bboxes else 1.0
    mean_pred_best_iou = float(valid_iou.max(axis=1).mean()) if pred_bboxes else 1.0
    return mean_gt_best_iou, mean_pred_best_iou


def format_bbox_ref(prefix: str, bbox: Any) -> str:
    class_name = getattr(bbox, "class_name", getattr(bbox, "entity_label", "entity"))
    return f"{prefix}{bbox.id}:{class_name}"


def score_entities(
    scene_id: str,
    pred_path: Path,
    gt_path: Path,
    pred_bboxes: List[Any],
    gt_bboxes: List[Any],
    iou_threshold: float,
    class_aware: bool,
) -> SceneScore:
    iou_matrix = compute_iou_matrix(pred_bboxes, gt_bboxes)
    matches = match_bboxes(
        pred_bboxes, gt_bboxes, iou_matrix, iou_threshold, class_aware
    )

    matched_pred = {match.pred_index for match in matches}
    matched_gt = {match.gt_index for match in matches}
    tp = len(matches)
    pred_count = len(pred_bboxes)
    gt_count = len(gt_bboxes)
    fp = pred_count - tp
    fn = gt_count - tp

    if pred_count == 0 and gt_count == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    mean_tp_iou = float(np.mean([match.iou for match in matches])) if matches else 0.0
    mean_gt_best_iou, mean_pred_best_iou = best_iou_means(
        pred_bboxes, gt_bboxes, iou_matrix, class_aware
    )

    matched_pairs = ";".join(
        [
            f"{format_bbox_ref('p', pred_bboxes[m.pred_index])}"
            f"->{format_bbox_ref('g', gt_bboxes[m.gt_index])}"
            f":{m.iou:.3f}"
            for m in sorted(matches, key=lambda item: item.iou)
        ]
    )
    unmatched_pred = ";".join(
        format_bbox_ref("p", pred_bboxes[index])
        for index in range(pred_count)
        if index not in matched_pred
    )
    unmatched_gt = ";".join(
        format_bbox_ref("g", gt_bboxes[index])
        for index in range(gt_count)
        if index not in matched_gt
    )

    return SceneScore(
        scene_id=scene_id,
        pred_path=pred_path,
        gt_path=gt_path,
        pred_count=pred_count,
        gt_count=gt_count,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_tp_iou=mean_tp_iou,
        mean_gt_best_iou=mean_gt_best_iou,
        mean_pred_best_iou=mean_pred_best_iou,
        matched_pairs=matched_pairs,
        unmatched_pred=unmatched_pred,
        unmatched_gt=unmatched_gt,
    )


def score_scene(
    scene_id: str,
    pred_path: Path,
    gt_path: Path,
    class_map: Optional[Dict[str, str]],
    drop_unmapped: bool,
    minimum_scale: float,
    iou_threshold: float,
    class_aware: bool,
) -> SceneScore:
    pred_bboxes = load_bboxes(pred_path, class_map, drop_unmapped, minimum_scale)
    gt_bboxes = load_bboxes(gt_path, class_map, drop_unmapped, minimum_scale)
    return score_entities(
        scene_id,
        pred_path,
        gt_path,
        pred_bboxes,
        gt_bboxes,
        iou_threshold,
        class_aware,
    )


def score_region_scene(
    scene_id: str,
    pred_path: Path,
    gt_dir: Path,
    gt_region_dir: Optional[Path],
    minimum_scale: float,
    iou_threshold: float,
    k: int,
    expand_fraction: float,
) -> SceneScore:
    pred_regions = load_regions(pred_path, minimum_scale)
    gt_regions, gt_path = load_gt_regions(
        scene_id,
        gt_dir,
        gt_region_dir,
        k,
        expand_fraction,
        minimum_scale,
    )
    return score_entities(
        scene_id,
        pred_path,
        gt_path,
        pred_regions,
        gt_regions,
        iou_threshold,
        class_aware=False,
    )


def read_scene_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def collect_scene_ids(
    pred_dir: Path,
    gt_dir: Path,
    scene_ids_file: Optional[Path],
    include_missing_pred: bool,
) -> List[str]:
    requested_scene_ids = read_scene_ids(scene_ids_file)
    if requested_scene_ids is not None:
        return requested_scene_ids

    pred_ids = {path.stem for path in pred_dir.glob("*.txt")}
    gt_ids = {path.stem for path in gt_dir.glob("*.txt")}
    if include_missing_pred:
        return sorted(gt_ids)
    return sorted(pred_ids & gt_ids)


def write_csv(path: Path, scores: Iterable[SceneScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [score.__dict__ for score in scores]
    fieldnames = list(SceneScore.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scene_ids(path: Path, scores: Iterable[SceneScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(score.scene_id for score in scores) + "\n")


def export_pairs(path: Path, scores: Iterable[SceneScore]) -> None:
    pred_export_dir = path / "pred"
    gt_export_dir = path / "gt"
    pred_export_dir.mkdir(parents=True, exist_ok=True)
    gt_export_dir.mkdir(parents=True, exist_ok=True)
    for score in scores:
        if score.pred_path.exists():
            shutil.copy2(score.pred_path, pred_export_dir / score.pred_path.name)
        if score.gt_path.exists():
            shutil.copy2(score.gt_path, gt_export_dir / score.gt_path.name)


def print_summary(scores: List[SceneScore]) -> None:
    headers = [
        "rank",
        "scene_id",
        "f1",
        "recall",
        "precision",
        "mean_gt_best_iou",
        "tp",
        "fp",
        "fn",
        "pred",
        "gt",
    ]
    print(",".join(headers))
    for rank, score in enumerate(scores, start=1):
        print(
            ",".join(
                [
                    str(rank),
                    score.scene_id,
                    f"{score.f1:.4f}",
                    f"{score.recall:.4f}",
                    f"{score.precision:.4f}",
                    f"{score.mean_gt_best_iou:.4f}",
                    str(score.tp),
                    str(score.fp),
                    str(score.fn),
                    str(score.pred_count),
                    str(score.gt_count),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank the worst prediction files by 3D IoU matching against ground-truth "
            "layout or region files."
        )
    )
    parser.add_argument(
        "--target",
        choices=["bbox", "region"],
        default="bbox",
        help="Score object Bbox lines or first-stage Region lines.",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("/data2/chenjq24/SpatialLM/arkitscenes-spatiallm/pred/ckpt-5620"),
        help="Directory containing predicted layout .txt files.",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=Path("/data2/chenjq24/SpatialLM/arkitscenes-spatiallm/layout"),
        help="Directory containing ground-truth layout .txt files.",
    )
    parser.add_argument(
        "--gt_region_dir",
        type=Path,
        help=(
            "Optional directory containing GT Region .txt files. Only used with "
            "--target region. If omitted or a scene file is missing, GT regions are "
            "derived from GT Bbox lines using --k and --expand_fraction."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of worst scenes to print/export.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.25,
        help="IoU threshold used to count true-positive bbox matches.",
    )
    parser.add_argument(
        "--minimum_scale",
        type=float,
        default=0.1,
        help="Clamp each bbox dimension to at least this value before IoU.",
    )
    parser.add_argument(
        "--sort_by",
        choices=[
            "f1",
            "recall",
            "precision",
            "mean_gt_best_iou",
            "mean_tp_iou",
            "mean_pred_best_iou",
        ],
        default="f1",
        help="Metric used for ascending worst-first sorting.",
    )
    parser.add_argument(
        "--class_agnostic",
        action="store_true",
        help="Ignore class labels during bbox matching. Region matching is always class-agnostic.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="K used when deriving GT regions from GT object bboxes.",
    )
    parser.add_argument(
        "--expand_fraction",
        type=float,
        default=0.25,
        help="Per-side expansion fraction used when deriving GT regions.",
    )
    parser.add_argument(
        "--scene_ids",
        type=Path,
        help="Optional newline-separated list of scene IDs to score.",
    )
    parser.add_argument(
        "--include_missing_pred",
        action="store_true",
        help=(
            "Score every GT file, treating missing prediction files as empty. "
            "By default only scene IDs present in both directories are scored."
        ),
    )
    parser.add_argument(
        "--label_mapping",
        type=Path,
        help="Optional TSV label mapping, such as spatiallm-testset/benchmark_categories.tsv.",
    )
    parser.add_argument("--label_from", default="spatiallm59")
    parser.add_argument("--label_to", default="spatiallm20")
    parser.add_argument(
        "--drop_unmapped",
        action="store_true",
        help="When --label_mapping is used, drop boxes with no target mapping.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="Optional path for the selected worst-scene CSV.",
    )
    parser.add_argument(
        "--output_all_csv",
        type=Path,
        help=(
            "Optional path for a CSV containing every scored scene, sorted with "
            "the same worst-first ordering as the selected CSV."
        ),
    )
    parser.add_argument(
        "--output_scene_ids",
        type=Path,
        help="Optional path for a newline-separated selected worst-scene ID list.",
    )
    parser.add_argument(
        "--export_dir",
        type=Path,
        help="Optional directory where selected pred/gt txt pairs are copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = (
        read_label_mapping(args.label_mapping, args.label_from, args.label_to)
        if args.label_mapping
        else None
    )
    class_aware = not args.class_agnostic
    scene_ids = collect_scene_ids(
        args.pred_dir, args.gt_dir, args.scene_ids, args.include_missing_pred
    )

    scores = []
    missing_pred_count = 0
    missing_gt_count = 0
    for scene_id in scene_ids:
        pred_path = args.pred_dir / f"{scene_id}.txt"
        gt_path = args.gt_dir / f"{scene_id}.txt"
        if not pred_path.exists():
            missing_pred_count += 1
            if not args.include_missing_pred:
                continue
        if not gt_path.exists():
            missing_gt_count += 1
            continue
        if args.target == "region":
            scores.append(
                score_region_scene(
                    scene_id,
                    pred_path,
                    args.gt_dir,
                    args.gt_region_dir,
                    args.minimum_scale,
                    args.iou_threshold,
                    args.k,
                    args.expand_fraction,
                )
            )
        else:
            scores.append(
                score_scene(
                    scene_id,
                    pred_path,
                    gt_path,
                    class_map,
                    args.drop_unmapped,
                    args.minimum_scale,
                    args.iou_threshold,
                    class_aware,
                )
            )

    scores.sort(
        key=lambda score: (
            getattr(score, args.sort_by),
            score.f1,
            score.mean_gt_best_iou,
            -score.fn,
            -score.fp,
            score.scene_id,
        )
    )
    selected_scores = scores[: args.top_k]
    print_summary(selected_scores)

    if args.output_csv:
        write_csv(args.output_csv, selected_scores)
    if args.output_all_csv:
        write_csv(args.output_all_csv, scores)
    if args.output_scene_ids:
        write_scene_ids(args.output_scene_ids, selected_scores)
    if args.export_dir:
        export_pairs(args.export_dir, selected_scores)

    if missing_pred_count or missing_gt_count:
        print(
            f"# skipped_or_empty_missing_pred={missing_pred_count} "
            f"skipped_missing_gt={missing_gt_count}"
        )


if __name__ == "__main__":
    main()
