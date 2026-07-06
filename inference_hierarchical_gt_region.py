#!/usr/bin/env python3
"""Stage-2 inference directly on GT-region point-cloud JSON."""

from inference_hierarchical_evict import main


if __name__ == "__main__":
    raise SystemExit(main(default_use_gt_bbox_mask=False))
