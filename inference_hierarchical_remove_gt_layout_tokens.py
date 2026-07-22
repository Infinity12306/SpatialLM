#!/usr/bin/env python3
"""Predicted-region inference after removing GT layout-overlap point tokens."""

from inference_hierarchical_pred_region_evict import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_evict_points=False,
            default_use_gt_bbox_mask=False,
            default_remove_gt_layout_tokens=True,
        )
    )
