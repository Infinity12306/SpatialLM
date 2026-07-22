# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        },
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."
        },
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."
        },
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."
        },
    )
    cutoff_len: int = field(
        default=8192,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."
        },
    )
    val_size: float = field(
        default=0.0,
        metadata={
            "help": "Size of the validation set, should be an integer or a float in range `[0,1)`."
        },
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the preprocessed datasets. "
                "If save_dir not exists, it will save the preprocessed datasets. "
                "If save_dir exists, it will load the preprocessed datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use a shared file system for the datasets."
        },
    )
    num_bins: int = field(
        default=1280,
        metadata={"help": "The number of bins for point cloud quantization."},
    )
    world_size: float = field(
        default=32.0,
        metadata={
            "help": (
                "World extent in meters used for point cloud quantization and "
                "layout position discretization. Region point clouds larger than "
                "this extent on any axis are center-cropped before encoding."
            )
        },
    )
    max_point_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of encoded point tokens inserted into the "
                "language model. Longer point-token sequences are center-cropped "
                "by removing tokens from both ends."
            )
        },
    )
    point_token_bbox_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep only final point tokens whose voxels overlap "
                "GT object bboxes. This is intended for hierarchical stage-2 "
                "experiments."
            )
        },
    )
    point_token_bbox_expand_ratio: float = field(
        default=0.1,
        metadata={
            "help": (
                "Per-side expansion ratio for GT object bboxes used by "
                "point-token masking. 0.1 makes each dimension 1.2x."
            )
        },
    )
    point_cloud_batch_encoding: bool = field(
        default=False,
        metadata={
            "help": (
                "Pack all point clouds in a local batch and run the Sonata "
                "encoder once using offsets instead of per-sample encoding."
            )
        },
    )
    point_token_scorer_gt_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Return augmented GT object bboxes for online point-token "
                "scorer supervision without applying GT filtering."
            )
        },
    )
    do_augmentation: bool = field(
        default=False,
        metadata={"help": "Whether or not to do data augmentation."},
    )
    random_rotation: bool = field(
        default=False,
        metadata={"help": "Whether or not to do non axis-aligned random rotation."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")
        if self.world_size <= 0:
            raise ValueError("`world_size` must be positive.")
        if self.max_point_tokens is not None and self.max_point_tokens <= 0:
            raise ValueError("`max_point_tokens` must be positive when configured.")
        if self.point_token_bbox_expand_ratio < 0:
            raise ValueError("`point_token_bbox_expand_ratio` must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
