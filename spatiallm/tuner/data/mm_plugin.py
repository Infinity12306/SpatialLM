import os
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Union, Sequence, Optional, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import get_world_preset
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors
from spatiallm.pcd.transform import Compose

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer

    PointCloudInput = Union[str, dict, NDArray]

LAYOUT_S_PLACEHOLDER = os.environ.get("LAYOUT_S_PLACEHOLDER", "<|layout_s|>")
LAYOUT_E_PLACEHOLDER = os.environ.get("LAYOUT_E_PLACEHOLDER", "<|layout_e|>")
POINT_S_TOKEN = os.environ.get("POINT_S_TOKEN", "<|point_start|>")
POINT_E_TOKEN = os.environ.get("POINT_E_TOKEN", "<|point_end|>")
POINT_CLOUD_PLACEHOLDER = os.environ.get("POINT_CLOUD_PLACEHOLDER", "<point_cloud>")


class SpatialLMPlugin:
    def __init__(
        self,
        point_token: str = "<|point_pad|>",
        num_bins: int = 1280,
        world_size: float = 32.0,
        do_augmentation: bool = False,
        random_rotation: bool = False,
        point_token_bbox_mask: bool = False,
        point_token_bbox_expand_ratio: float = 0.1,
        point_cloud_batch_encoding: bool = False,
        point_token_scorer_gt_mask: bool = False,
    ):
        self.point_token = point_token
        self.point_token_bbox_mask = point_token_bbox_mask
        self.point_token_bbox_expand_ratio = point_token_bbox_expand_ratio
        self.point_cloud_batch_encoding = point_cloud_batch_encoding
        self.point_token_scorer_gt_mask = point_token_scorer_gt_mask

        default_world_extent = get_world_preset()
        global_extent = get_world_preset(world_size)
        self.num_bins = num_bins
        self.world_size = float(global_extent[1] - global_extent[0])
        self.center_crop_enabled = self.world_size < (
            default_world_extent[1] - default_world_extent[0]
        )
        self.grid_size = (global_extent[1] - global_extent[0]) / self.num_bins
        self.do_augmentation = do_augmentation
        self.random_rotation = random_rotation
        self.augmentation = Compose(
            [
                dict(type="RandomColorGrayScale", p=0.05),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.75, ratio=0.1),
                dict(type="ChromaticJitter", p=0.8, std=0.05),
                dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                dict(type="RandomColorDrop", p=0.1, color_augment=0.0),
                dict(type="RandomJitter", sigma=0.025, clip=0.05, ratio=0.8, p=0.9),
                dict(type="RandomJitter", sigma=0.2, clip=0.2, ratio=0.05, p=0.85),
                dict(type="RandomJitter", sigma=0.4, clip=1.0, ratio=0.001, p=0.75),
                dict(type="RandomJitter", sigma=0.5, clip=4.0, ratio=0.0005, p=0.7),
                dict(
                    type="ElasticDistortion",
                    distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    p=[0.85, 0.5],
                ),
            ]
        )

        self.transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=self.grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=self.num_bins,
                ),
            ]
        )

    def _center_crop_to_world_size(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        extent = max_bound - min_bound
        if (not self.center_crop_enabled) or np.all(extent <= self.world_size):
            return points, colors, {
                "center_cropped": False,
                "crop_min_bound": min_bound,
                "crop_max_bound": max_bound,
            }

        crop_center = (min_bound + max_bound) * 0.5
        crop_half_size = np.full(3, self.world_size * 0.5, dtype=np.float64)
        crop_min_bound = crop_center - crop_half_size
        crop_max_bound = crop_center + crop_half_size
        crop_mask = np.all(
            (points >= crop_min_bound) & (points <= crop_max_bound),
            axis=1,
        )
        if not np.any(crop_mask):
            nearest_index = int(np.argmin(np.sum((points - crop_center) ** 2, axis=1)))
            crop_mask[nearest_index] = True

        return points[crop_mask], colors[crop_mask], {
            "center_cropped": True,
            "crop_min_bound": crop_min_bound,
            "crop_max_bound": crop_max_bound,
        }

    def _preprocess_point_cloud(self, point_cloud: dict) -> np.ndarray:
        r"""
        Pre-processes a single point cloud.
        """
        point_cloud = self.transform(point_cloud)
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        color = point_cloud["color"]
        assert len(coord) == len(xyz) == len(color)
        return np.concatenate([coord, xyz, color], axis=1)

    def _regularize_point_clouds(
        self, point_clouds: Sequence["PointCloudInput"], **kwargs
    ) -> torch.Tensor:
        points_list = []
        max_len = 0
        for point_cloud in point_clouds:
            if not isinstance(point_cloud, dict):
                raise ValueError(
                    "Point cloud input must be a dictionary with 'name' and 'coord' keys."
                )
            point_feats = self._preprocess_point_cloud(point_cloud, **kwargs)
            max_len = max(max_len, len(point_feats))
            points_list.append(point_feats)

        for i in range(len(points_list)):
            points_list[i] = np.pad(
                points_list[i],
                ((0, max_len - len(points_list[i])), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

        # convert list of point clouds to batch with shape (batch_size, max_len, 3)
        return torch.as_tensor(np.stack(points_list, axis=0))

    def _pack_point_clouds(
        self, point_clouds: Sequence["PointCloudInput"], **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-size point clouds and return cumulative point offsets."""
        points_list = []
        offsets = []
        total_points = 0
        for point_cloud in point_clouds:
            if not isinstance(point_cloud, dict):
                raise ValueError(
                    "Point cloud input must be a dictionary with 'name' and 'coord' keys."
                )
            point_feats = self._preprocess_point_cloud(point_cloud, **kwargs)
            if len(point_feats) == 0:
                raise ValueError("Point cloud is empty after preprocessing.")
            points_list.append(point_feats)
            total_points += len(point_feats)
            offsets.append(total_points)

        if not points_list:
            return (
                torch.empty((0, 9), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        return (
            torch.as_tensor(
                np.concatenate(points_list, axis=0),
                dtype=torch.float32,
            ),
            torch.as_tensor(offsets, dtype=torch.long),
        )

    def _regularize_point_token_keep_bboxes(
        self,
        point_token_keep_bboxes: Sequence[np.ndarray],
    ) -> torch.Tensor:
        batch_size = len(point_token_keep_bboxes)
        max_boxes = max((bboxes.shape[0] for bboxes in point_token_keep_bboxes), default=0)
        if max_boxes == 0:
            return torch.empty((batch_size, 0, 7), dtype=torch.float32)

        batched = np.full((batch_size, max_boxes, 7), np.nan, dtype=np.float32)
        for index, bboxes in enumerate(point_token_keep_bboxes):
            if bboxes.size == 0:
                continue
            batched[index, : bboxes.shape[0], :] = bboxes.astype(np.float32)
        return torch.as_tensor(batched)

    def _bboxes_to_point_token_keep_array(self, layout: Layout) -> np.ndarray:
        if not layout.bboxes:
            return np.empty((0, 7), dtype=np.float32)

        scale_multiplier = 1.0 + 2.0 * self.point_token_bbox_expand_ratio
        values = []
        for bbox in layout.bboxes:
            values.append(
                [
                    bbox.position_x,
                    bbox.position_y,
                    bbox.position_z,
                    bbox.scale_x * scale_multiplier,
                    bbox.scale_y * scale_multiplier,
                    bbox.scale_z * scale_multiplier,
                    bbox.angle_z,
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _get_mm_inputs(
        self,
        batched_messages: Sequence[Dict[str, str]],
        point_clouds: Sequence["PointCloudInput"],
    ) -> dict:
        input_dict = {"point_clouds": None}  # default key

        point_clouds_data = []
        transformations = []
        for pcd_path in point_clouds:
            pcd = load_o3d_pcd(pcd_path)
            points, colors = get_points_and_colors(pcd)

            if self.do_augmentation:
                data_aug = {"name": "pcd", "coord": points, "color": colors}
                data_aug = self.augmentation(data_aug)
                points = data_aug["coord"]
                colors = data_aug["color"]

            # randomly apply scale and rotation transformation to the point cloud
            if self.random_rotation:
                angle_z = np.random.random() * 2 * np.pi
            else:
                angle_z = np.random.choice(np.array([0, 0.5, 1.0, 1.5]) * np.pi)

            scaling = np.random.uniform(0.75, 1.25)
            rotmat = R.from_rotvec(np.array([0, 0, angle_z])).as_matrix()
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            center_pt = (min_bound + max_bound) / 2
            scaled_points = (points - center_pt) * scaling
            transformed_points = (rotmat @ scaled_points.T).T + center_pt
            transformed_points, colors, crop_info = self._center_crop_to_world_size(
                transformed_points,
                colors,
            )
            # store transformation parameters for sync the augmentation to the layout
            transformations.append(
                {
                    "angle_z": angle_z,
                    "center_pt": center_pt,
                    "scaling": scaling,
                    "min_bound": np.min(transformed_points, axis=0),
                    "transformed_points": transformed_points,
                    **crop_info,
                }
            )

            point_cloud = {"name": "pcd", "coord": transformed_points, "color": colors}
            point_clouds_data.append(point_cloud)

        # Here we assume each conversation has exactly one point cloud
        assert len(batched_messages) == len(point_clouds_data)
        processed_messages = []
        point_token_keep_bboxes = []
        for mi, messages in enumerate(batched_messages):
            processed, keep_bboxes = self.process_messages(
                messages,
                [transformations[mi]],
                return_point_token_keep_bboxes=True,
            )
            processed_messages.append(processed)
            point_token_keep_bboxes.append(keep_bboxes)

        if len(processed_messages) != 0:
            input_dict["messages"] = processed_messages
        if len(point_clouds_data) != 0:
            if self.point_cloud_batch_encoding:
                packed_points, point_offsets = self._pack_point_clouds(
                    point_clouds_data
                )
                input_dict["point_clouds"] = packed_points
                input_dict["point_cloud_offsets"] = point_offsets
            else:
                # Legacy NaN-padded tensor with shape (batch_size, max_len, 9).
                input_dict["point_clouds"] = self._regularize_point_clouds(
                    point_clouds_data
                )
        if self.point_token_bbox_mask:
            input_dict["point_token_keep_bboxes"] = self._regularize_point_token_keep_bboxes(
                point_token_keep_bboxes
            )
        if self.point_token_scorer_gt_mask:
            input_dict["point_token_scorer_gt_bboxes"] = (
                self._regularize_point_token_keep_bboxes(point_token_keep_bboxes)
            )
        return input_dict

    def _validate_input(
        self,
        point_clouds: Sequence["PointCloudInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(point_clouds) != 0 and self.point_token is None:
            raise ValueError(
                "This model does not support point cloud input. Please check whether the correct `template` is used."
            )

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        point_clouds: Sequence["PointCloudInput"],
        tokenizer: "PreTrainedTokenizer",
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(point_clouds)
        return input_ids, labels

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        transformations: Sequence[dict],
        return_point_token_keep_bboxes: bool = False,
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], np.ndarray]]:
        r"""
        Pre-processes input messages to sync the transformation between point cloud and layout.
        """
        self._validate_input(transformations)
        messages = deepcopy(messages)
        num_point_tokens = 0
        point_token_keep_bboxes = np.empty((0, 7), dtype=np.float32)

        for message in messages:
            content = message["content"]
            if LAYOUT_S_PLACEHOLDER in content and LAYOUT_E_PLACEHOLDER in content:
                transformation = transformations[num_point_tokens - 1]
                min_bound = transformation["min_bound"]
                center_pt = transformation["center_pt"]
                scaling = transformation["scaling"]
                transformed_points = transformation["transformed_points"]
                center_cropped = transformation["center_cropped"]
                crop_min_bound = transformation["crop_min_bound"]
                crop_max_bound = transformation["crop_max_bound"]
                layout_start_pos = content.index(LAYOUT_S_PLACEHOLDER)
                layout_end_pos = content.index(LAYOUT_E_PLACEHOLDER)
                layout_content = content[
                    layout_start_pos + len(LAYOUT_S_PLACEHOLDER) : layout_end_pos
                ]
                # parse layout_content
                layout = Layout(layout_content)
                # transformation augmentation
                layout.translate(-center_pt)
                layout.scale(scaling)
                layout.rotate(transformation["angle_z"])
                layout.translate(center_pt)
                if center_cropped:
                    layout.bboxes = [
                        bbox for bbox in layout.bboxes
                        if (
                            crop_min_bound[0] <= bbox.position_x <= crop_max_bound[0]
                            and crop_min_bound[1] <= bbox.position_y <= crop_max_bound[1]
                            and crop_min_bound[2] <= bbox.position_z <= crop_max_bound[2]
                        )
                    ]
                layout.filter_empty_bboxes(transformed_points, num_points=100)
                layout.reorder_entities()
                layout.translate(-min_bound)
                if self.point_token_bbox_mask or self.point_token_scorer_gt_mask:
                    point_token_keep_bboxes = self._bboxes_to_point_token_keep_array(layout)
                layout.normalize_and_discretize(
                    self.num_bins,
                    world_size=self.world_size,
                )
                new_layout_content = layout.to_language_string()
                content = content.replace(
                    f"{LAYOUT_S_PLACEHOLDER}{layout_content}{LAYOUT_E_PLACEHOLDER}",
                    new_layout_content,
                )
                message["content"] = content

            if POINT_CLOUD_PLACEHOLDER in content:
                content = content.replace(
                    POINT_CLOUD_PLACEHOLDER,
                    f"{POINT_S_TOKEN}{self.point_token}{POINT_E_TOKEN}",
                    1,
                )
                num_point_tokens += 1
                message["content"] = content

        if len(transformations) != num_point_tokens:
            raise ValueError(
                f"The number of point clouds does not match the number of {POINT_CLOUD_PLACEHOLDER} tokens."
            )
        if return_point_token_keep_bboxes:
            return messages, point_token_keep_bboxes
        return messages

    def get_mm_inputs(
        self,
        point_clouds: Sequence["PointCloudInput"],
        batch_prompts: Sequence[List[int]],
    ) -> Dict[str, Union[List[dict]]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            point_clouds: a list of point cloud inputs, shape (num_point_clouds,)
            pointlens: number of point clouds in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(point_clouds)
        return self._get_mm_inputs(batch_prompts, point_clouds)


def get_mm_plugin(
    point_token: str = "<|point_pad|>",
    **kwargs,
) -> "SpatialLMPlugin":
    return SpatialLMPlugin(point_token, **kwargs)
