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
- After changing the repository, synchronize this handoff document with the
  changed files, behavior, experiment state, results, and follow-up work. Fix
  or remove stale statements instead of only appending new notes.

## Repository Overview

Important top-level files:

- `train.py`: original SpatialLM training entry.
- `inference.py`, `inference_v2.py`: original one-stage inference variants.
- `eval.py`: original eval script.
- `train_hierarchical_spatiallm.py`: custom two-stage hierarchical training.
- `train_point_token_scorer_online.py`: online scorer initialization from raw
  point clouds using the original SpatialLM encoder/projector.
- `train_stage2_joint_scorer.py`: joint SpatialLM + hard point-token scorer
  stage2 training.
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
- `configs/scorer/`: scorer-only training configs; any config that also trains
  the SpatialLM LLM belongs directly under `configs/` with a `spatiallm_`
  prefix.
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

ScanNet comparison data root:

- `/nas1/chenjunqing2024/scannet`
- Canonical raw ScanNet source: `/nas1/huyueyang23/dataset/scannet/scannet`.
  It contains 1,513 scene directories, the four raw files required by the
  ScanNet/V-DETR exporter for every scene, and
  `scannetv2-labels.combined.tsv`. All derived files must remain under the
  comparison data root above.
- `scannet18_stage1_region_test_half.json`: 156-scene test-half inference input.
- `baseline/spatiallm/scannet18_spatiallm_bbox_test_half.json`: 156-scene
  one-stage SpatialLM baseline inference input with the fine-tuning prompt.
- `metadata/scannetv2_test_half.txt`: one scene id per line; evaluator accepts
  this txt format directly.
- `layout/<scene_id>.txt`: ScanNet18 object bbox GT.
- `region/val/expanded/<scene_id>.txt`: expanded Region GT.
- `predictions/ours/<run_tag>/`: raw and NMS prediction outputs.
- `predictions/baselines/spatiallm/<run_tag>/`: raw and NMS SpatialLM baseline
  predictions.
- `evaluation/ours/`: final object/region metric JSON files.
- `evaluation/baselines/spatiallm/`: SpatialLM baseline object metric JSONs.
- `vdetr/scannet_test_half_detection_data/`: evaluation-only V-DETR NPY input
  arrays for the 156-scene test half.
- `vdetr/scannet_full_val_detection_data/`: evaluation-only V-DETR NPY input
  arrays for the complete official 312-scene validation split.
- `vdetr/meta_data/test_half/scannetv2_val.txt`: V-DETR's required `val` alias
  containing the test-half scene ids.
- `predictions/baselines/vdetr/`: V-DETR score-thresholded, post-NMS `Bbox`
  prediction txt files.
- `evaluation/baselines/vdetr/`: V-DETR common P/R/F1 logs and JSON metrics.

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
- Original-SpatialLM online scorer initialization:
  `saves/point_token_scorer/stage2_res16_spatiallm_original_online/checkpoint-7196/scorer.pt`
- Joint hard-scorer stage2 checkpoint evaluated at step 12000:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_joint_scorer/checkpoint-12000`
- Joint hard-scorer final model/scorer pair is stored at:
  `saves/hierarchical/stage2_bboxes_20000_res16_max4096_joint_scorer`
  with completed training state `global_step=14392`, `epoch=4.0`; the matching
  `checkpoint-14392` also exists.

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

## Remove GT Layout Tokens Ablation

Purpose:

- Test whether point-token filtering can help without turning each Stage-2
  input into the unusually clean object-only representation created by GT
  BBoxMask.
- Keep unrelated objects and noise, but remove tokens associated with GT
  `Wall`, `Door`, and `Window` geometry.

Implementation:

- Entry: `inference_hierarchical_remove_gt_layout_tokens.py`.
- Pipeline:
  `run_scripts/inference_and_eval/run_inference_hierarchical_remove_gt_layout_tokens.sh`.
- The Stage-2 point encoder first produces its normal token sequence and applies
  the checkpoint's `max_point_tokens=4096` center crop. Layout-overlap tokens
  are removed afterwards, so retained tokens are a strict subsequence of the
  baseline LLM input and cannot be replenished from outside the crop window.
- GT layout boxes are converted into the region point cloud's shifted local
  coordinates. A final token is removed when its voxel intersects a GT
  `Wall`, `Door`, or `Window` yaw box. The default expansion ratio is `0.0`.
- This is an oracle diagnostic because it reads GT layout files. Stage 1 still
  predicts layout and regions normally; only Stage 2 token filtering uses GT.
- Each inference shard reports aggregate cropped/removed/kept token counts in
  its log. The default comparison uses stage1 checkpoint 9996 and stage2
  checkpoint 14000, matching the main `Hier-20k Res16 Max4096` table entry.

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

Attention-scorer inference:

- `inference_hierarchical_attention_scorer.py` evaluates the learned scorer on
  predicted regions without GT labels. It reproduces training by center-cropping
  to `max_point_tokens`, adding `logsigmoid(scorer_logits)` as a point-token key
  bias in every LLM attention layer, and retaining that bias during KV-cache
  autoregressive decoding.
- `budget` remains the differentiable training capacity constraint; soft-bias
  inference does not reinterpret it as hard top-k. Math SDPA is used, matching
  training and avoiding the custom-mask backend alignment issue.
- The inference entry also supports deployment-style `hard_topk`: scorer logits
  select the highest-scoring k tokens, then selected indices are sorted to
  preserve Sonata order. The checkpoint-40000 runner currently uses hard top-k
  with `k=1536`, as requested for the main evaluation.
- The checkpoint-40000 SpatialLM20 inference + NMS + eval runner is
  `run_scripts/inference_and_eval/run_inference_hierarchical_attention_scorer_e2e_40000_2shards.sh`.

## Online Joint Hard-Scorer Training

Goal:

- Avoid the distribution mismatch caused by separately training a GT-BBoxMask
  LLM and a scorer, then composing them only at inference time.
- The first implementation intentionally keeps hard selection non-differentiable:
  scorer BCE is supervised by the online GT object mask, while next-token loss
  passes through selected tokens to the LLM, point projector, and point encoder.
  LLM loss does not train the scorer in this version.

Scorer initialization:

- Use the original `manycore-research/SpatialLM1.1-Qwen-0.5B`
  encoder/projector, not a BBoxMask or already stage2-trained checkpoint.
- Initialization is online: raw PCDs receive deterministic per-epoch
  rotation/scale augmentation, a local batch is packed and encoded with one
  frozen Sonata forward, and GT labels are built on GPU. There is no feature
  cache prerequisite.
- Online trainer: `train_point_token_scorer_online.py`. The language model is
  released after loading; only the frozen point encoder/projector are moved to
  GPU, and only scorer BCE is backpropagated.
- Scorer config:
  `configs/scorer/point_token_scorer_spatiallm_original.yaml`
- Scorer runner:
  `run_scripts/run_train_point_token_scorer_spatiallm_original.sh`

Joint training:

- Entry: `train_stage2_joint_scorer.py`
- Config: `configs/spatiallm_stage2_joint_scorer.yaml`
- Runner: `run_scripts/run_train_stage2_joint_scorer.sh`
- Inference + NMS + eval runner:
  `run_scripts/inference_and_eval/run_inference_hierarchical_joint_scorer.sh`.
  It resolves a matching `model.safetensors` and `scorer.pt` from the same
  checkpoint (or the final output root) and supports one shard per
  comma-separated `CUDA_VISIBLE_DEVICES` entry.
- Local point clouds are packed as `[sum(N_i), 9]` plus cumulative offsets and
  encoded with one Sonata forward per local batch.
- Sonata returns packed final `context`, `grid_coord`, `batch`, and `offset`.
- Scorer inputs are padded only after point encoding. Hard-selected indices are
  sorted to preserve original Sonata token order before LLM insertion.
- GT object bboxes are expanded by 10% on every side, i.e. dimensions are
  multiplied by 1.2 while center/yaw remain unchanged.
- Final-voxel overlap labels are built on GPU in a batched tensor operation.
- Default loss is `next_token_loss + scorer_loss_weight * scorer_bce_loss`.
- `scorer_detach_input: true` means scorer BCE updates only the scorer; point
  encoder/projector are updated through next-token loss on selected tokens.
- Checkpoints contain the full SpatialLM+scorer state and an additional
  standalone `scorer.pt` for inference convenience.
- Auto-resume resets cosine scheduling over remaining optimizer steps with the
  configured warmup ratio. W&B native resume is not used; optional history
  continuation uses clean-copy.

Validation completed:

- A small Sonata GPU test showed packed two-sample encoding exactly matched
  per-sample encoding (`grid_coord` identical, max context error `0.0`).
- A real two-sample stage2 smoke train using the original SpatialLM completed
  forward/backward and checkpoint save. It encoded 1585 final tokens, kept 64,
  logged separate LM/scorer losses, and produced gradients for both point
  encoder and scorer.
- Online scorer initialization passed raw-data collate and real GPU tests. A
  sample with 22,750 input points produced 321 final tokens; encoder/projector
  stayed frozen and every scorer parameter received a gradient. The full
  one-step trainer saved `label_stats`, and joint training resolved its automatic
  positive-class weight from those checkpoint statistics.

Completed runs and downstream evaluation:

- Online scorer initialization completed and produced
  `checkpoint-7196/scorer.pt`.
- Joint training completed four epochs / 14,392 optimizer steps. Full
  model/scorer pairs exist at checkpoints 8000, 10000, 12000, 14000, 14392 and
  at the final output root.
- The 500-scene predicted-region SpatialLM20 evaluation uses greedy inference,
  scorer threshold `0.5`, keep range 1..4096, and class-wise NMS@0.1.
- Step 12000 object micro P/R/F1 is `0.7246/0.6243/0.6707` at IoU 0.25 and
  `0.6048/0.5211/0.5598` at IoU 0.50.
- Final object micro P/R/F1 is `0.7089/0.6346/0.6697` at IoU 0.25 and
  `0.5918/0.5297/0.5590` at IoU 0.50. Step 12000 is slightly better than final
  by F1, but both are substantially below the non-scorer Hier-20k Res16
  Max4096 baseline (`0.7491/0.6656` F1).
- The two runs regenerated stage1 predictions, so their layout/region metrics
  differ slightly even though both use the same stage1 checkpoint 9996. Do not
  interpret those small differences as a stage2 scorer effect.

## Inference And Evaluation

Prefer scripts under `run_scripts/inference_and_eval/`.

Single-scene visualization:

- `render_scene_multiview.py` accepts one PLY/PCD, a required top-z crop
  percentage, an optional SpatialLM prediction txt, and optional per-axis
  quantile trimming via `--normalization-trim-percent` (default 0.5 percent
  from each tail; set 0 to disable).
- It ignores layout entities and renders only `Bbox` objects. The cropped point
  cloud's quantile-trimmed subject is centered and uniformly scaled by its
  longest AABB edge. The longest normalized edge is exactly 1 while the
  original x/y/z aspect ratio is preserved; object bboxes use the same affine
  transform.
- Outputs include the full original-coordinate z-cropped PLY, normalized subject
  PLY, four diagonal and four horizontal-edge-midpoint 45-degree elevated
  views, one top-down view, a spatially arranged 3x3 contact sheet, and JSON
  metadata containing crop, subject selection, normalization, camera,
  class-color, and output information.
- Each view defaults to 2048x2048. Use `--width` and `--height` to override the
  render resolution independently; the contact sheet follows those dimensions.
- Example:
  `python render_scene_multiview.py INPUT.ply --top-crop-percent 10 --normalization-trim-percent 0.5 --prediction-txt PRED.txt --output-dir OUTPUT`

Report-scene selection:

- `select_hierarchical_improved_scenes.py` compares the NMS@0.1 SpatialLM
  baseline against Hier-20k Res16 Max4096 stage1-9996/stage2-14000 per scene.
- It uses the same SpatialLM20 class mapping, 0.1 minimum box scale, and
  class-wise Hungarian matching as `eval_hierarchical.py` at IoU 0.25/0.50.
- The default report ranking is the mean of the two per-scene micro-F1 deltas;
  candidates need at least three GT objects and cannot regress at either
  threshold. Full CSV/JSON results go under
  `spatiallm-dataset/eval/scene_comparison/hier_20000_res16_max4096_vs_spatiallm/`.
- Run with `python select_hierarchical_improved_scenes.py`; all comparison paths
  and selection thresholds remain configurable CLI arguments.
- `render_top5_scene_comparisons.py` renders the default top five report scenes
  for GT, SpatialLM, and Hier with object bboxes only. Point rasterization is
  shared by all three panels for each camera; wall/door/window/region entities
  are ignored. GT and both predictions share the benchmark label mapping and
  deterministic per-class colors. It writes individual 2048x2048 views,
  full-resolution three-column views, and a 3x3 comparison contact sheet under
  `spatiallm-dataset/visualization/hier_20000_res16_max4096_vs_spatiallm_top5/`.

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
- `run_inference_hierarchical_attention_scorer_e2e_40000_2shards.sh`: predicted-
  region checkpoint-40000 end-to-end attention scorer with hard top-k 1536.
- `run_inference_hierarchical_joint_scorer.sh`: matching joint SpatialLM/scorer
  checkpoint inference, NMS, and eval.
- `run_inference_hierarchical_remove_gt_layout_tokens.sh`: oracle diagnostic
  that removes only GT layout-overlap tokens after the normal center crop.

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
- Online joint hard-scorer training also failed to recover the non-scorer
  baseline: step 12000/final object F1 are `0.6707/0.5598` and
  `0.6697/0.5590`, respectively, versus baseline `0.7491/0.6656`. The result
  suggests the issue is not only the earlier separately trained
  scorer/BBoxMask-LLM distribution mismatch.

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

Re-run only NMS=0.25 and eval on the existing SpatialLM baseline and
hierarchical raw ScanNet predictions:

```bash
bash /data2/chenjq24/SpatialLM/run_scripts/inference_and_eval/run_scannet18_spatiallm_ours_nms025_eval.sh
```

This command does not load a model or rerun inference. It writes separate
`nms_iou_0.25` prediction directories and metric JSON files for both methods.

Prepare and evaluate released V-DETR on the ScanNet18 test half:

```bash
cd /data2/chenjq24/SpatialLM/V-DETR
conda run --no-capture-output -n vdetr python prepare_scannet18_eval_data.py --num_workers 8
CUDA_VISIBLE_DEVICES=6 bash run_scannet18_inference_nms_eval.sh
```

See `V-DETR/SCANNET18_EVAL.md` for sharding, verification, multi-GPU, paths,
and threshold policy. The common comparison first applies objectness and
maximum semantic probability thresholds of 0.5, then V-DETR's empty-box filter
with at least 5 points, and finally its default confidence-aware class-wise NMS
IoU 0.25. It keeps one argmax class per query and uses the same
`eval_hierarchical.py` P/R/F1 at IoU 0.25/0.50 as SpatialLM.

Prepare and evaluate V-DETR on the complete 312-scene official validation
split using its YAML config:

```bash
cd /data2/chenjq24/SpatialLM/V-DETR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_scannet18_full_val_inference_nms_eval.sh configs/scannet18_full_val_eval.yaml
```

This runner incrementally prepares missing arrays and rejects any scene list
whose count is not exactly 312. `DRY_RUN=1` only validates configuration;
`PREPARE_ONLY=1` prepares and verifies data without GPU inference.

Report both V-DETR native mAP and common P/R/F1 on full val:

```bash
cd /data2/chenjq24/SpatialLM/V-DETR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_scannet18_full_val_map_prf_eval.sh configs/scannet18_full_val_eval.yaml
```

This runs two forward passes by design: native mAP uses V-DETR's existing
per-class-proposal AP logic, while common P/R/F1 exports one argmax class per
query. Their post-processing settings must not be shared: native mAP passes no
common overrides and follows the README/main.py defaults (confidence 0, no
semantic threshold, no empty-box filtering, NMS 0.25), while P/R/F1 uses
objectness/semantic 0.5, empty-point threshold 5, and NMS 0.25.

Run the no-empty-filter common P/R/F1 ablation on full val:

```bash
cd /data2/chenjq24/SpatialLM/V-DETR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_scannet18_full_val_prf_no_empty_eval.sh configs/scannet18_full_val_eval.yaml
```

This keeps the common objectness/semantic thresholds at 0.5, argmax class and
NMS 0.25, but does not pass `--remove_empty_box` and does not compute mAP. Its
prediction/evaluation names end in `no_empty`, separate from the original run.

Run the one-stage SpatialLM ScanNet18 baseline inference, NMS, and evaluation:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash /data2/chenjq24/SpatialLM/run_scripts/inference_and_eval/run_scannet18_spatiallm_baseline_inference_nms_eval.sh
```

Its default model directory is
`saves/baselines/spatiallm/scannet18_bbox_ft`; the script selects the latest
valid `checkpoint-*`. `MODEL_DIR` may point either to that parent directory or
directly to a checkpoint. Inference uses the prompt stored in the processed
test JSON and evaluates only the explicit ScanNet18 object classes.

Run ScanNet18 hierarchical inference, NMS, and evaluation (one shard per GPU):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash /data2/chenjq24/SpatialLM/run_scripts/inference_and_eval/run_scannet18_ours_inference_nms_eval.sh
```

The default script expects fine-tuned checkpoints under
`saves/hierarchical/scannet18/`. Override `STAGE1_MODEL_DIR`,
`STAGE2_MODEL_DIR`, `RUN_TAG`, or `NMS_IOU` through environment variables when
needed. ScanNet18 evaluation must use `--no_label_mapping` with the explicit
18-class list; otherwise the evaluator retains its backward-compatible
SpatialLM20 mapping behavior.

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

Evaluate a matching joint SpatialLM/scorer pair (the final output root is used
by default; set `JOINT_CKPT` to evaluate a specific checkpoint):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash /data2/chenjq24/SpatialLM/run_scripts/inference_and_eval/run_inference_hierarchical_joint_scorer.sh
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
- `train_point_token_scorer_online.py`
- `train_stage2_joint_scorer.py`
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
- `inference_hierarchical_*.py` variants listed above.
- `configs/spatiallm_*hierarchical*.yaml`
- `configs/spatiallm_stage2_filtered_point_token_scorer*.yaml`
- `configs/spatiallm_stage2_joint_scorer.yaml`
- `configs/scorer/point_token_scorer_spatiallm_original.yaml`
- `configs/scorer/stage2_attention_scorer.yaml`
- `run_scripts/**`
- `eval_results.md`
- `reports/*.md`

Before editing any of these, inspect the current file because the user may have
made additional manual edits after this handoff was written.

## Open Threads / Likely Next Requests

The user may continue any of these:

1. Diagnose why online joint hard-scorer training remains far below the
   non-scorer baseline; inspect keep ratios/mask quality and consider evaluating
   the remaining 8000/10000/14000 checkpoints before launching a new run.
2. Finish the remove-GT-layout-token oracle ablation. Its code and runner exist,
   but no completed eval log is present under `logs/eval_results` yet.
3. Run/compare filtered-token resume epoch=4 vs restart epoch=2 or evaluate any
   newer filtered-token checkpoints.
4. Continue debugging low GPU utilization caused by shard reads or NAS I/O.
5. Tune end-to-end attention scorer memory/throughput or implement pre-encoder
   filtering so the point encoder does not process full raw res16 point clouds.
6. Improve region prediction recall, e.g. supplement to three regions using
   K-Means over dense points with fixed mean GT region size.
7. Improve region generation with size-regularized K-Means to avoid one huge
   region plus two small regions.
8. Prepare thesis/report material based on two-stage, point-token gating, and
   differentiable bbox/IoU-loss ideas.

## Last Known Immediate State

As of 2026-07-22:

- The worktree is intentionally dirty and `main` is five commits ahead of
  `origin/main`. Do not clean, reset, or discard untracked experiment files.
- ScanNet18 fine-tuning and the 156-scene test-half comparison have completed.
  The one-stage SpatialLM baseline has final weights and `checkpoint-1200` under
  `saves/baselines/spatiallm/scannet18_bbox_ft`. The hierarchical checkpoints
  are stage1 `checkpoint-1200` and stage2 `checkpoint-1904` under
  `saves/hierarchical/scannet18/`.
- On the common 2,174-object test-half evaluator, SpatialLM baseline object F1
  is `0.6838/0.5200`, hierarchical is `0.7176/0.5796`, and V-DETR is
  `0.7986/0.7264` at IoU 0.25/0.50. V-DETR uses NMS@0.25 while both SpatialLM
  rows in the main comparison use NMS@0.1. Separate NMS@0.25 SpatialLM outputs
  also exist for ablation.
- V-DETR full-val common P/R/F1 inference completed on all 312 scenes. The
  corrected README-default native evaluation also completed without the common
  0.5 thresholds or empty-box filter and reports mAP25/mAP50 `77.53/65.93`.
  Ignore the earlier `69.03/59.72` run that incorrectly inherited common-eval
  post-processing.
- Original-SpatialLM online scorer initialization completed at
  `saves/point_token_scorer/stage2_res16_spatiallm_original_online/checkpoint-7196/scorer.pt`.
  It consumes raw PCDs directly and does not require a feature-cache precompute.
- Joint hard-scorer training completed four epochs / 14,392 steps. Matching
  `model.safetensors` and `scorer.pt` pairs exist at the final output root and
  checkpoints 8000, 10000, 12000, 14000, and 14392. Step 12000 and final have
  completed 500-scene NMS@0.1 evaluations; their object F1 values are
  `0.6707/0.5598` and `0.6697/0.5590`, respectively.
- The GT-layout-token-removal ablation entry and runner are implemented, but no
  completed evaluation log is present yet. Its current output directory only
  contains setup metadata, so do not treat it as a finished result.
- Filtered-token configs and runners remain available for resume and restart;
  the local cache is under
  `/data2/chenjq24/SpatialLM/data/bbox_mask_scorer_filtered_point_token`.
- Current comparison results should always be read from `eval_results.md`.
