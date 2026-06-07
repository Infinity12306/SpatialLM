# Changelog

## f6b175e03d03336108ae837afe82de094dde3f66
- Summary: Add hierarchical region training and inference tooling.
- Author: Codex <codex@openai.com>
- Date: 2026-05-30 13:04:43 +0800
- Details: Add Region layout support plus scripts for hierarchical dataset generation, inference, evaluation, rendering, and checkpoint testing.
- Details: Add hierarchical training configuration and orchestration helpers with updated SFT batch, bf16, and learning-rate defaults.
- Details: Set repo-root PYTHONPATH for subprocess training and ignore local datasets, outputs, saves, and experiment artifacts.

### Files
- `.gitignore`: Adds ignored local dataset, output, save, render, WandB, temporary, and script artifact paths.
- `build_hierarchical_region_dataset.py`: New script that samples scenes, builds stage-1 region and stage-2 bbox ShareGPT datasets, writes region PCDs and metadata, and updates dataset info.
- `configs/spatiallm_hierarchical_20000.yaml`: New two-stage hierarchical training config for 20k samples with base SFT settings and stage-specific datasets and output directories.
- `configs/spatiallm_sft.yaml`: Increases gradient accumulation and learning rate, and enables bf16 training.
- `copy_worst_test_pcd.py`: New utility that copies PLY point clouds for scene IDs listed in a CSV.
- `eval_hierarchical.py`: New evaluation script for hierarchical predictions across layout, object, and region metrics with optional JSON output.
- `filter_worst_predictions.py`: New scoring and export utility that ranks weak prediction scenes by class-aware bbox matching metrics.
- `gather_scene.py`: New utility that collects PCD, GT layout, and region render artifacts for selected scenes into example folders.
- `generate_region_bboxes.py`: New utility that clusters object boxes into region boxes, writes raw and expanded regions, renders overlays, and records a manifest.
- `inference_hierarchical.py`: New two-stage inference script that predicts regions, runs per-region bbox detection, applies optional NMS, and writes stage and final outputs.
- `inference_hierarchical_v2.py`: New per-scene artifact wrapper for hierarchical inference that copies inputs, writes predictions, renders overlays, and saves metadata.
- `inference_v2.py`: New single-model scene inference tool that writes predictions and top-down GT and prediction region renders.
- `poetry.toml`: New Poetry setting that disables virtualenv creation for this checkout.
- `render_topdown_arkitscenes.py`: New renderer for top-down ARKitScenes point-cloud images after top-height clipping, with optional debug artifacts.
- `spatiallm/layout/entity.py`: Adds a Region entity with transform, normalization, serialization, and sorting support.
- `spatiallm/layout/layout.py`: Parses, stores, exports, sorts, and serializes Region entities alongside walls, doors, windows, and bboxes.
- `test_model_checkpoint.py`: New checkpoint inference utility for PLY files, PLY directories, or JSON datasets with subsetting and sharding controls.
- `train.py`: Adds the repo root to PYTHONPATH for subprocess launches and passes the prepared environment to subprocess.run.
- `train_hierarchical_spatiallm.py`: New orchestrator that resolves stage configs, launches stage-1 training, finds the latest checkpoint, and launches stage-2 training.
- `wait_for_gpu_and_train.sh`: New GPU polling launcher that waits for low-memory GPUs, dispatches queued commands, writes logs, and retries likely failed launches.
