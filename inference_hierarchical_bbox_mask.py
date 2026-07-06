#!/usr/bin/env python3
"""Oracle stage-2 hierarchical inference with GT bbox token masking."""

from inference_hierarchical_evict import main


if __name__ == "__main__":
    raise SystemExit(main(default_use_gt_bbox_mask=True))
