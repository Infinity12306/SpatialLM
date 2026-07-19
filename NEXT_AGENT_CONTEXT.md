# Context For Next Coding Agent

This file is a handoff document for a new Coding Agent working in
`/data2/chenjq24/SpatialLM`. It is intentionally written for agent-to-agent
continuity, not for user-facing explanation.

## Working Rules

- Primary language with the user is Chinese. Keep English terms when they are
  more natural, e.g. `stage2`, `point token`, `bbox`, `wandb`, `shard`.
- Do not delete existing files or directories unless the user explicitly asks.
- The worktree is dirty and has many intentional untracked files. Do not clean,
  reset, checkout, or revert user changes.
- Use `rg` / `find` / `sed` for inspection, and `apply_patch` for manual edits.
- Training code should use YAML configs by default, not many CLI arguments.
- All training tasks should use wandb with explicit project and run names.
- Training code should record `CUDA_VISIBLE_DEVICES` into wandb config when
  possible.
- Default training scheduler policy: cosine LR scheduler, warmup ratio `0.03`.
- Training code should normally support auto-resume from latest checkpoint in
  output dir, but W&B native resume is not preferred. If continuing a curve is
  needed, use clean-copy semantics: copy old W&B history up to checkpoint step
  into a clean run, then continue logging.
- Data processing should support shards where practical. Training DataLoader
  `num_workers` should usually default to `8`. Keep efficiency improvements
  simple before adding complex async queues.

## Repository Overview

Important top-level files:

- `train.py`: original SpatialLM training entry.
- `inference.py`, `inference_v2.py`: original one-stage inference variants.
- `eval.py`: original eval script.
- `train_hierarchical_spatiallm.py`: custom two-stage hierarchical training.
- `inference_hierarchical.py`, `inference_hierarchical_v2.py`: custom two-stage
  inference.
- `eval_hierarchical.py`: custom eval with layout/object/region summaries.
- `generate_region_bboxes.py`: generate GT region bboxes by K-Means.
- `build_hierarchical_region_dataset.py`: construct stage1/stage2 datasets and
  region point clouds.
- `apply_bbox_nms.py`: object bbox NMS post-processing.
- `filter_worst_predictions.py`: worst prediction analysis, extended for
  region/full-test CSV use cases.
- `analyze_prediction_overlaps.py`: duplicate/overlap analysis and histogram.
- `visualize_scene_ply.py`: 3D visualization helper.
- `eval_results.md`: current comparison table; read this for latest metrics
  instead of relying on memory.

Important directories:

- `spatiallm/`: SpatialLM package and modified model/data code.
- `configs/`: YAML configs for original, hierarchical, and filtered-token
  training.
- `configs/scorer/`: YAML configs for end-to-end attention scorer training.
- `run_scripts/`: most current run scripts; prefer these over old root scripts.
- `run_scripts/inference_and_eval/`: inference + NMS + eval scripts.
- `run_scripts/stress_tests/`: memory stress scripts.
- `spatiallm-dataset-link`: symlink to `/nas1/chenjunqing2024/spatiallm-dataset`.
  NAS access can be slow.
- `spatiallm-dataset`: local prediction/eval/debug outputs.
- `saves`: checkpoints, likely symlinked or large. Do not delete.
- `data`: local copied caches, used to avoid NAS bottlenecks.
- `logs`: run logs. Evaluation logs are usually in `logs/eval_results`.
- `reports`: weekly reports and images.

## Data Layout

Core data root:

- `/data2/chenjq24/SpatialLM/spatiallm-dataset-link`
- This is a symlink to NAS. Be careful when copying symlinks. If the user wants
  real files, use dereference-aware commands and verify with `du`/`file`.

Common files:

- `split.csv`: train/val/test split source.
- `pcd/`: full-scene point clouds.
- `layout/`: GT txt files containing layout and object bboxes.
- `spatiallm_train.json`, `spatiallm_val.json`, `spatiallm_test.json`: original
  one-stage data.
- `spatiallm_stage1_region_*`: hierarchical stage1 data.
- `spatiallm_stage2_bbox_*`: hierarchical stage2 data.
- `region_20000`, `region_20000_2`, `region_val`, `region_test`: region bbox
  and region PC datasets.
- `point_token_scorer_cache/...`: cached point encoder context/features and
  GT mask labels for scorer training.
- `point_token_cache_messages/...`: cached tokenized prompt/label messages.
- `filtered_point_token`: scorer-filtered point token cache on NAS.

Local cache currently used for filtered-token LLM training:

- `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token/train`
- `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token/val`
- Both have `index.json` and worker shard subdirs.

## Hierarchical Two-Stage Method

The hierarchical idea:

1. Stage 1 input: whole scene point cloud.
2. Stage 1 output: layout (`Wall`, `Door`, `Window`) and `Region`.
3. Stage 2 input: per-region point cloud.
4. Stage 2 output: object `Bbox` inside that region.
5. Final output: combine all region object predictions back into scene coords.

Prompts:

- Stage1: detect walls, doors, windows, regions.
- Stage2: detect object bboxes.

Region generation:

- `generate_region_bboxes.py`
- K-Means with `k=3` over GT 3D object bboxes only.
- Layout bboxes (`wall`, `window`, `door`) are not used for K-Means.
- Raw region bbox is minimal bbox around clustered 3D objects.
- Expanded region bbox expands dimensions by 25%.
- Scenes with no object bbox can have no regions; stage2 data skips them.

Important edge case:

- Some validation/test scenes have only layout and no object bbox. Stage1 should
  output layout only; stage2 should not run. Dataset builders and inference
  scripts should not assert that every scene has stage2 regions.

## Hierarchical Training Configs

Main hierarchical configs:

- `configs/spatiallm_hierarchical_20000.yaml`: original two-stage 20k.
- `configs/spatiallm_hierarchical_40000.yaml`: 40k training.
- `configs/spatiallm_hierarchical_40000_resume_3600.yaml`: resume stage1 from
  `checkpoint-3600`.
- `configs/spatiallm_hierarchical_40000_resume_3600_stage2.yaml`: stage2-only
  support.
- `configs/spatiallm_hierarchical_20000_res_16_max_4096.yaml`: stage2 world
  size 16m, max 4096 point tokens.
- `configs/spatiallm_hierarchical_20000_res_16_max_7200.yaml`: stage2 max 7200.
- `configs/spatiallm_hierarchical_20000_res_16_max_4096_bbox_mask.yaml`:
  stage2 bbox-mask training.
- `configs/spatiallm_hierarchical_20000_res_16_max_4096_evict.yaml`:
  stage2 evict-data training.

Checkpoint landmarks:

- Baseline HF/local model: `manycore-research/SpatialLM1.1-Qwen-0.5B`.
- Local HF cache path:
  `/data2/chenjq24/huggingface/hub/models--manycore-research--SpatialLM1.1-Qwen-0.5B`
- Hier-20k stage1: `saves/hierarchical/stage1_regions_20000_res_16/checkpoint-9996`
- Hier-20k res16 stage2:
  `saves/hierarchical/stage2_bboxes_20000_res_16_max_4096/checkpoint-14392`
- BBoxMask stage2:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_bbox_mask/checkpoint-14392`
- Evict stage2:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_evict/checkpoint-14392`
- Hier-40k stage1:
  `saves/hierarchical/stage1_regions_40000/checkpoint-5000`
- Hier-40k stage2:
  `saves/hierarchical/stage2_bboxes_40000/checkpoint-14408`

## Point Encoding Details

SpatialLM uses a point backbone (`Sonata`) followed by `point_proj`.

Original code pads variable-size point clouds with NaN in batch. Model removes
NaN rows before point encoding. This is because each sample has a different
number of raw points and generated point tokens.

Important dtypes observed:

- LLM `inputs_embeds` dtype in bf16 training/inference is `torch.bfloat16`.
- `point_backbone` is loaded then set to fp32 by `set_point_backbone_dtype`.
- `point_proj` parameters are bf16 when model is loaded in bf16.
- Point backbone context is effectively fed to projector in bf16 in normal
  flow. Some cache pipelines intentionally save context as bf16 to match this.

World size / binning:

- Baseline scene world size is 32m.
- Res16 stage2 uses 16m world size because region PCs are smaller.
- `num_bins` is generally 1280.
- Stage2 `max_point_tokens` center-crops point token sequences if too long.
- `cutoff_len` is text token length; it does not directly cap point tokens.

Point token order:

- Do not assume simple x-y-z scan order.
- Sonata/GridSample/GridPooling use grid coords, hashing, sorting, unique, and
  pooling/serialization. Internal serialized order can affect token sequence.
- When creating masks, use the actual `grid_coord` associated with final point
  tokens, not an independently generated dense voxel traversal.

## Model-Side Modifications

Modified files:

- `spatiallm/model/spatiallm_qwen.py`
- `spatiallm/model/spatiallm_llama.py`

Current notable modifications:

- Forward supports `point_token_keep_bboxes` for bbox-mask in online point
  encoding.
- Forward supports `point_token_features`: precomputed point tokens can be
  passed directly instead of raw `point_clouds`.
- `point_token_features` are padded with NaN; forward removes rows containing
  NaN per sample.
- Label reconstruction now handles the case where no point input was inserted.
- `prepare_inputs_for_generation` forwards `point_token_features`.

Do not casually revert these files. They are required by filtered-token LLM
training and related inference.

## BBoxMask Method

Implemented in:

- `train_hierarchical_spatiallm_mask.py`
- `spatiallm/model/sonata_encoder.py` modifications
- `inference_hierarchical_bbox_mask.py`
- `inference_hierarchical_pred_region_bbox_mask.py`
- `analyze_bbox_mask_point_tokens.py`

Idea:

- In stage2, keep only point tokens whose final voxel overlaps a GT object bbox
  expanded by 10% on each side.
- Any voxel overlapping expanded object bbox is kept.
- This is a GT oracle-style method and performs very well, but it is not
  directly usable at test time without GT object bboxes.

Important distinction:

- BBoxMask happens after point encoding in the current training path, so the
  point encoder still pays full raw-res16 cost.
- User later discussed moving mask before encoder for efficiency, but current
  implemented mask training mostly filters point tokens after encoding.

## Evict Method

Implemented in:

- `build_stage2_evict_dataset.py`
- `analyze_evict_point_tokens.py`
- `inference_hierarchical_evict.py`
- `inference_hierarchical_pred_region_evict.py`

Idea:

- Preprocess stage2 region point clouds using GT layout/object bboxes.
- First discard empty final voxels.
- Protect voxels overlapping object bbox.
- Prefer voxels near objects and far from layout.
- Remove raw points in unselected voxels until target point-token budget.

Caveats:

- Evict uses GT during dataset/test construction in the current experiments.
- Analysis showed evict may still produce more tokens than target in some cases;
  verify actual counting logic before assuming strict `<= target_pt_num`.
- Evict performed much worse than GT BBoxMask in eval; likely because it can
  remove useful context/boundary points and preserves object regions more tightly
  than BBoxMask's 10% expansion.

## Point Token Scorer: Supervised BBoxMask Replacement

Purpose:

- Train a scorer to imitate GT BBoxMask labels, so inference can filter point
  tokens without GT object bboxes.

Precompute:

- `precompute_point_token_scorer_data.py`
- Produces cached point backbone context features, grid coords, center coords,
  and GT BBoxMask labels.
- Features are point encoder context, not already bbox-masked.
- Cache format includes labels generated from GT expanded object bboxes.
- The cache was built with seed logic based on base seed, epoch, sample index,
  and point cloud index. `verify_point_token_scorer_cache_reproducibility.py`
  checks reproducibility for labels/grid transforms.

Training:

- `train_point_token_scorer.py`
- Scorer architecture:
  - Frozen `point_proj` maps cached 512D point encoder context to point token
    embedding dimension.
  - Point token embedding is cast to fp32.
  - Concatenate token feature, normalized `grid_coord`, and normalized region
    center grid coord.
  - MLP input projection.
  - Transformer Encoder with position encoding.
  - Linear head outputs logits.
  - Loss: `BCEWithLogitsLoss`, with `pos_weight auto` support.
- `threshold` is for eval metrics/inference decision, not for BCE itself.
- Scorer checkpoint of interest:
  `saves/point_token_scorer/bboxmask_ckpt14392_context_bf16/checkpoint-29488`

Inference:

- `inference_hierarchical_scorer.py`
- Uses stage1 predicted regions, stage2 BBoxMask-trained ckpt, and scorer.
- Default test threshold was `0.5`, also tested `0.25`.
- Uses `scorer_max_keep 4096`, `scorer_min_keep 1`.

Observed result:

- Scorer eval during scorer training had high F1 vs GT BBoxMask labels.
- Downstream object detection still dropped significantly compared with GT
  BBoxMask. Hypothesis: GT BBoxMask creates unrealistically clean object-only
  point clouds; scorer retains noise, so the LLM trained only on clean BBoxMask
  inputs fails under noisy filtered inputs.

## Filtered Point Token LLM Training

Purpose:

- Continue training the LLM/projector on point tokens filtered by the scorer,
  to adapt the model to noisy scorer-filtered inputs.
- Point encoder is not trained in this setup because point tokens are already
  offline filtered and cached.

Scripts:

- `filter_point_tokens_with_scorer.py`: filters cached features with scorer and
  creates filtered point token cache.
- `merge_filtered_point_token_cache_indices.py`: merges shard indices.
- `train_stage2_filtered_point_tokens.py`: trains from filtered point-token
  cache using `point_token_features` model input.

Current local filtered cache:

- `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token/train`
- `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token/val`

Current resume config:

- `configs/spatiallm_stage2_filtered_point_token_scorer.yaml`
- Starts from BBoxMask stage2 ckpt:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_bbox_mask/checkpoint-14392`
- Output:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_scorer_filtered_point_token`
- `auto_resume_from_latest_checkpoint: true`
- `num_train_epochs: 4`
- `learning_rate: 1e-5`
- `warmup_ratio: 0.00`
- batch: per-device 2, GAS 8, effective 16
- run script:
  `run_scripts/run_train_stage2_filtered_point_tokens.sh`

Current restart config:

- `configs/spatiallm_stage2_filtered_point_token_scorer_restart_epoch2.yaml`
- Does not resume:
  `auto_resume_from_latest_checkpoint: false`
- Output:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_scorer_filtered_point_token_restart_epoch2`
- `num_train_epochs: 2`
- `learning_rate: 1e-4`
- `warmup_ratio: 0.03`
- run name currently in file: `G1_hier_res16_filtered_point_token_restart_epoch2`
- run script:
  `run_scripts/run_train_stage2_filtered_point_tokens_restart_epoch2.sh`

Important resume behavior in `train_stage2_filtered_point_tokens.py`:

- Finds latest `checkpoint-*` in output dir when `auto_resume...` is true.
- Can reset LR scheduler after resume while keeping optimizer state.
- Previous native HF scheduler loading caused confusing LR behavior after
  resuming from a finished/near-finished one-epoch run. The current preferred
  logic allows user to set LR and warmup explicitly for remaining training.

## End-to-End Attention Scorer

Goal:

- Inspired by "Make Each Token Count: Towards Improving Long-Context Performance
  with KV Cache Eviction".
- Train a scorer without GT BBoxMask labels.
- Freeze stage2 SpatialLM, train only scorer.
- Scorer outputs log-scores added to attention logits before softmax.
- Loss is next-token prediction loss plus capacity loss.

Current implementation:

- `train_stage2_attention_scorer.py`
- Config: `configs/scorer/stage2_attention_scorer.yaml`
- Run script: `run_scripts/run_train_stage2_attention_scorer.sh`
- Stress scripts:
  - `build_attention_scorer_stress_topk_cache.py`
  - `stress_test_stage2_attention_scorer.py`
  - `run_scripts/stress_tests/stress_test_attention_scorer.sh`

Current config highlights:

- cache root:
  `spatiallm-dataset-link/point_token_scorer_cache/stage2_res16_ckpt14392_context_bf16`
- message cache:
  `spatiallm-dataset-link/point_token_cache_messages/stage2_res16_ckpt14392_context_bf16`
- model:
  `saves/hierarchical/stage2_bboxes_20000_res_16_max_4096/checkpoint-14392`
- `attn_implementation: sdpa`
- `sdpa_backend: math`
- `max_point_tokens: 3200`
- `budget: 1536`
- per-device train BS 2, eval BS 4, GAS 8
- scorer output dir:
  `saves/point_token_attention_scorer/train_scorer_atten_endToEnd`

Attention implementation caveats:

- Eager attention may materialize large `B x H x L x L` attention matrices and
  can OOM.
- SDPA with additive bias has backend limitations; math backend was used after
  encountering an SDPA alignment error (`LSE is not correctly aligned
  (strideH)`).
- There was concern that full attention matrices explode memory. Stress-test
  before increasing `max_point_tokens` or batch size.

W&B / resume for attention scorer:

- W&B native resume was removed/avoided.
- There is code to clean-copy old W&B history up to checkpoint step into a clean
  run before continuing.
- `wandb_copy_page_size` controls how many history rows are fetched per API
  page.
- Check `train_stage2_attention_scorer.py` before changing this; it has custom
  checkpoint, scheduler, and W&B logic.

## Inference And Evaluation

Prefer scripts under `run_scripts/inference_and_eval/`.

Representative scripts:

- `run_inference_hierarchical_res16_4shards.sh`: res16 hierarchical inference.
- `run_inference_hierarchical_bbox_mask_2shards.sh`: pred-region BBoxMask.
- `run_inference_hierarchical_evict_2shards.sh`: pred-region evict.
- `run_inference_hierarchical_gt_region_bbox_mask_2shards.sh`: GT-region
  BBoxMask diagnostic.
- `run_inference_hierarchical_gt_region_evict_2shards.sh`: GT-region evict.
- `run_inference_hierarchical_gt_region_res16_2shards.sh`: GT-region res16
  baseline diagnostic.
- `run_inference_hierarchical_scorer_2shards.sh`: supervised scorer threshold
  0.5.
- `run_inference_hierarchical_scorer_2shards_threshold_0.25.sh`: supervised
  scorer threshold 0.25.
- `run_inference_hierarchical_scorer_filtered_point_token_2shards.sh`: stage2
  ckpt trained further on scorer-filtered point tokens.
- `run_inference_hierarchical_attention_oracle_4shards.sh`: GT-region attention
  oracle, keep top 75% by GT-label attention.

NMS:

- `apply_bbox_nms.py` should apply NMS only to object bbox predictions.
- It preserves/copies stage1 layout/region dirs into output so eval can still
  read them. Do not assume layout/region were NMS-processed just because they
  appear in NMS output dir.

Eval:

- `eval_hierarchical.py` reports layout, region, and object metrics.
- Object/layout comparisons usually use `micro` summary rows.
- Object metrics usually use label mapping:
  `--label_mapping spatiallm-testset/benchmark_categories.tsv --label_from spatiallm59 --label_to spatiallm20`
- `missing_pred empty` means missing prediction files are treated as empty
  predictions rather than hard failure.
- Baseline original eval with `eval.py` and label mapping only evaluates mapped
  SpatialLM20 categories; unmapped GT bboxes are ignored.

Greedy inference:

- Passing `--greedy` disables sampling. In normal deterministic kernels this
  should be deterministic across runs, but GPU kernel nondeterminism can still
  exist in a strict bitwise sense.

## Current Result Tracking

- Main table: `eval_results.md`
- Older working table: `tmp.md`
- Weekly reports:
  - `reports/0629_report.md`
  - `reports/0706_report.md`
  - `reports/0713_report.md`

High-level experimental findings:

- Baseline layout is strong.
- Hierarchical stage1 region recall matters; pred-region recall gap vs GT-region
  diagnostics indicates missed regions hurt object results.
- Res16 increases point token counts and training time drastically.
- GT BBoxMask gives very strong object results but is an oracle and may make
  the task artificially easy.
- Evict did not significantly help and often hurt.
- Supervised scorer has high mask-label F1 during scorer eval but downstream
  object results are far worse than GT BBoxMask.
- Training LLM further on scorer-filtered point tokens is an active attempt to
  adapt to scorer noise.
- Attention-oracle keep75 is a GT diagnostic using GT labels to select tokens by
  attention. The existing run script uses GT regions.

## Common Performance / IO Issues

- NAS-backed `spatiallm-dataset-link` can be very slow and can make GPU util
  oscillate from 0 to high values.
- For heavy repeated training, local copies under `/data2/chenjq24/SpatialLM/data`
  are preferred.
- `num_workers=8` often helps training scorer/filtered-token data loading.
- `shard_size=1` in cache can still cause stalls if shard files are on NAS.
- Avoid `find -L` over large NAS dirs unless necessary; it can hang/slow.
- `nvitop` memory can exceed PyTorch allocated memory because of reserved CUDA
  memory, CUDA context, cuBLAS/cuDNN workspaces, fragmentation, and non-PyTorch
  allocations.
- In stress tests, reset peak memory before eval if you need true eval peak.

## Useful Commands

Run filtered-token resume training:

```bash
CUDA_VISIBLE_DEVICES=7 bash /data2/chenjq24/SpatialLM/run_scripts/run_train_stage2_filtered_point_tokens.sh
```

Run filtered-token restart epoch=2:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> bash /data2/chenjq24/SpatialLM/run_scripts/run_train_stage2_filtered_point_tokens_restart_epoch2.sh
```

Run end-to-end attention scorer:

```bash
CUDA_VISIBLE_DEVICES=5 bash /data2/chenjq24/SpatialLM/run_scripts/run_train_stage2_attention_scorer.sh
```

Search for ScanNet under `/nas1`:

```bash
find /nas1 -type d -name 'ScanNet' 2>/dev/null
```

Check local disk usage depth 2:

```bash
du -h --max-depth=2 /data2/chenjq24 2>/dev/null | sort -h
```

## Files Recently Created Or Modified By This Work

Many of these are untracked according to git; treat them as intended work:

- `train_hierarchical_spatiallm.py`
- `train_hierarchical_spatiallm_mask.py`
- `train_stage2_filtered_point_tokens.py`
- `train_stage2_attention_scorer.py`
- `train_point_token_scorer.py`
- `precompute_point_token_scorer_data.py`
- `precompute_point_token_cache_messages.py`
- `filter_point_tokens_with_scorer.py`
- `merge_filtered_point_token_cache_indices.py`
- `merge_point_token_cache_indices.py`
- `verify_point_token_scorer_cache_reproducibility.py`
- `stress_test_stage2_attention_scorer.py`
- `stress_test_point_token_scorer_memory.py`
- `build_attention_scorer_stress_topk_cache.py`
- `measure_pt_read_time.py`
- `code_grab.py`
- `inference_hierarchical_*.py` variants listed above.
- `configs/spatiallm_*hierarchical*.yaml`
- `configs/spatiallm_stage2_filtered_point_token_scorer*.yaml`
- `configs/scorer/stage2_attention_scorer.yaml`
- `run_scripts/**`
- `eval_results.md`
- `reports/*.md`

Before editing any of these, inspect the current file because the user may have
made additional manual edits after this handoff was written.

## Open Threads / Likely Next Requests

The user may continue any of these:

1. Run/compare filtered-token resume epoch=4 vs restart epoch=2.
2. Run inference/eval for the latest filtered-token checkpoints and update
   `eval_results.md`.
3. Continue debugging low GPU util caused by shard reads or NAS IO.
4. Tune end-to-end attention scorer memory/throughput.
5. Implement pre-encoder filtering so point encoder does not process full raw
   res16 point clouds.
6. Improve region prediction recall, e.g. always supplement to 3 regions using
   K-Means over dense points with fixed mean GT region size.
7. Improve region generation with size-regularized K-Means to avoid one huge
   region plus two small regions.
8. Prepare thesis/report material based on two-stage, point-token gating, and
   differentiable bbox/IoU-loss ideas.

## Last Known Immediate State

As of this handoff:

- Resume filtered-token config is:
  `configs/spatiallm_stage2_filtered_point_token_scorer.yaml`
- Restart filtered-token epoch=2 config is:
  `configs/spatiallm_stage2_filtered_point_token_scorer_restart_epoch2.yaml`
- Resume run script is:
  `run_scripts/run_train_stage2_filtered_point_tokens.sh`
- Restart run script is:
  `run_scripts/run_train_stage2_filtered_point_tokens_restart_epoch2.sh`
- Local filtered-token cache exists under:
  `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token`
- Current comparison results should be read from:
  `eval_results.md`
