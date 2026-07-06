#!/usr/bin/env python3
"""Hierarchical inference with predicted regions and GT bbox token masking."""

from inference_hierarchical_pred_region_evict import main


if __name__ == "__main__":
    raise SystemExit(main(default_evict_points=False, default_use_gt_bbox_mask=True))
