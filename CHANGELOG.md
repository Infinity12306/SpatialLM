# 更新日志

## 未提交：添加仅移除 GT layout point token 的分层推理消融
- 摘要：新增最小 oracle ablation，在 Hier-20k Res16 Max4096 的 Stage 2 中保留原始中心裁剪后的 token 顺序，仅删除 final voxel 与 GT `Wall`、`Door`、`Window` 相交的 point token，再复用现有 predicted-region inference、NMS 和 SpatialLM20 eval。
- 详情：过滤发生在原有 `max_point_tokens=4096` 中心裁剪之后，因此保留 token 是 baseline LLM 输入的严格子序列，不会从裁剪窗口外补入 token；默认 layout bbox 不额外扩张，并在每个 inference shard 的日志末尾汇总裁剪、删除和保留 token 数。

### 文件
- `inference_hierarchical_pred_region_evict.py`：复用既有 GT-assisted predicted-region 推理，添加局部坐标 layout bbox 构造和仅作用于 Stage 2 的 token 删除 hook。
- `inference_hierarchical_remove_gt_layout_tokens.py`：新增默认关闭 evict/GT object mask、启用 GT layout token 删除的轻量入口。
- `run_scripts/inference_and_eval/run_inference_hierarchical_remove_gt_layout_tokens.sh`：新增多 GPU 分片 inference + NMS@0.1 + layout/region/object eval 一键脚本，默认与主 Hier-20k Res16 Max4096 的 stage1-9996/stage2-14000 结果对齐。

## 未提交：添加 joint scorer 分层推理与评测入口
- 摘要：新增面向 online joint LLM + hard scorer 训练产物的多分片 inference + NMS + eval 脚本，确保 stage-2 model 和 `scorer.pt` 始终来自同一 checkpoint。

### 文件
- `run_scripts/inference_and_eval/run_inference_hierarchical_joint_scorer.sh`：自动选择最新的完整 model/scorer 配对，在 500-scene SpatialLM test split 上运行 predicted-region scorer inference、NMS@0.1 和 layout/region/object 评估。

## 未提交：添加分层模型相对 SpatialLM baseline 的逐场景改进筛选
- 摘要：新增 SpatialLM20 object bbox 逐场景对比工具，复用正式 evaluator 的类别映射、minimum scale 和 Hungarian matching，在相同 NMS@0.1 结果上按 IoU 0.25/0.50 micro F1 提升筛选适合报告可视化的场景。
- 详情：默认比较 SpatialLM baseline 与 Hier-20k Res16 Max4096 stage1-9996/stage2-14000 的 500 个 test scenes；排序分数为两个 IoU 阈值 F1 增量均值，报告候选默认要求至少 3 个 GT objects 且两个阈值均不退化，同时输出完整 CSV/JSON、类别级变化和点云/GT/预测路径。
- 详情：新增前五名场景的批量对比渲染入口，每个相机只栅格化一次点云，再分别叠加 GT、SpatialLM 与分层方法的 object bbox；三路类别统一映射到相同类别名并使用固定颜色，显式忽略 wall、door、window 和 region，输出三方单图、全分辨率三列对比图及 3x3 对比总览。

### 文件
- `select_hierarchical_improved_scenes.py`：新增逐场景 object metric 计算、聚合指标复核、候选过滤、排序和报告产物生成 CLI。
- `render_top5_scene_comparisons.py`：新增五场景、九视角的 GT/SpatialLM/Hier object-bbox 高效对比渲染脚本，并固定三路类别颜色。

## 未提交：添加 end-to-end attention scorer 的 SpatialLM20 评测入口
- 摘要：新增固定参数的两分片评测脚本，使用 `checkpoint-40000` 的 end-to-end attention scorer 对预测 Region 的 stage2 point tokens 做 hard top-k 筛选，并依次执行 object bbox NMS 和 SpatialLM20 layout/region/object 评测。
- 详情：先将 encoder token 中心裁到训练时的 `max_point_tokens=3200`，再按 scorer logits 选取得分最高的 `min(1536, token_count)` 个 tokens；选中索引恢复为 Sonata 原始顺序后输入 LLM。推理入口同时保留 soft attention-bias 模式用于后续 ablation，但当前一键评测固定使用 hard top-k 1536。

### 文件
- `inference_hierarchical_attention_scorer.py`：新增预测 Region 的 end-to-end attention-scorer 推理入口，支持 hard top-k 和训练同构的 soft attention key bias 两种模式。
- `run_scripts/inference_and_eval/run_inference_hierarchical_attention_scorer_e2e_40000_2shards.sh`：新增 checkpoint-40000 attention scorer 的 inference + NMS + eval 一键入口，所有参数直接保存在 shell 脚本中。

## 未提交：支持在现有 ScanNet raw 预测上对比 NMS IoU 0.25
- 摘要：新增不重跑 inference 的 SpatialLM baseline 与分层方法联合后处理脚本，分别对现有 test-half raw 预测执行 class-wise 3D bbox NMS 0.25 和 ScanNet18 eval。

### 文件
- `run_scripts/inference_and_eval/run_scannet18_spatiallm_ours_nms025_eval.sh`：校验两种方法各 156 个场景的 raw 预测，产出独立的 `nms_iou_0.25` 目录、日志和评估 JSON。

## 未提交：添加 filtered-token restart epoch2 的 SpatialLM20 评测入口
- 摘要：新增固定参数的两分片评测脚本，对 scorer-filtered point-token restart-epoch2 stage2 checkpoint 依次执行预测 Region 分层推理、object bbox NMS 和 SpatialLM20 layout/region/object 评测。
- 详情：stage1 使用 `checkpoint-9996`，stage2 使用 `checkpoint-28784`，scorer 使用 `checkpoint-29488`；推理采用 greedy decoding、scorer threshold 0.5、最多保留 4096 个 point tokens，并执行 class-wise 3D NMS@0.1。

### 文件
- `run_scripts/inference_and_eval/run_inference_hierarchical_scorer_filtered_point_token_restart_epoch2_2shards.sh`：新增 inference + NMS + eval 一键入口，所有非训练参数直接保存在 shell 脚本中。

## 未提交：添加点云九视角 object-bbox 可视化
- 摘要：新增单场景无窗口软件渲染器，支持按 z 轴顶部百分比裁剪、以主体包围盒最长边统一缩放并保持长宽高比例、可选 SpatialLM object bbox overlay，以及四个角点方向、四条水平边中点方向的 45°斜上方视角和 top-down 视角输出。
- 详情：可选按 x/y/z 每轴两端分位数快速剔除离群点，默认每端 0.5%，仅让主体参与归一化、渲染和 normalized PLY；原坐标 z 裁剪 PLY 仍保留全部点，参数设为 0 时以全部 z 裁剪点的 AABB 最长边归一化。
- 详情：每张视角图片的宽高可通过 `--width`、`--height` 独立指定，默认分辨率为 2048x2048；3x3 contact sheet 随单张图片尺寸自动调整。
- 详情：预测 txt 仅解析 `Bbox` 实体，显式忽略 wall、door、window 和 region；同时保存原坐标裁剪 PLY、归一化主体 PLY、九张 PNG、按方位排列的 3x3 contact sheet 和完整变换/camera metadata。

### 文件
- `render_scene_multiview.py`：新增 raw PCD 裁剪、object bbox 同步变换、headless pinhole rasterization、z-buffer point splatting 和多视角输出 CLI。

## 未提交：添加 V-DETR ScanNet18 test-half 数据适配与评估流程
- 摘要：在 `V-DETR/` 内新增 evaluation-only ScanNet18 数据适配器和 inference + NMS + common P/R/F1 eval 入口，使用与 SpatialLM 对比一致的 156 场景 test-half。
- 摘要：新增 YAML 驱动的官方完整 validation split 入口，增量准备并强制校验 312 个场景后复用同一后处理和 P/R/F1 evaluator。
- 摘要：新增 full-val native mAP + common P/R/F1 combined runner；native mAP 保留 V-DETR 原生 per-class proposal 逻辑，P/R/F1 保留每个 query 一个 argmax class 的通用导出逻辑。
- 修正：native mAP 阶段不再传入 common P/R/F1 的 0.5 objectness/semantic threshold 和空框过滤；现在严格调用 README 的 `main.py --test_only --auto_test` 默认评估路径（confidence 0、NMS 0.25、`exact_eval=False`、per-class proposals）。
- 摘要：新增 full-val no-empty P/R/F1 ablation，仅关闭 predicted empty-box filtering，保留 objectness/semantic 0.5、argmax class、class-wise NMS@0.25 和通用 evaluator，不计算 mAP。
- 数据源：确认原始 ScanNet 位于 `/nas1/huyueyang23/dataset/scannet/scannet`；1,513 个场景均具备 V-DETR/ScanNet exporter 所需的 mesh、aggregation、segmentation 和 axis-alignment metadata，派生数据仍统一写入 `/nas1/chenjunqing2024/scannet`。
- 详情：适配器从已有 axis-aligned PLY 和 ScanNet18 layout GT 生成 V-DETR loader 需要的 `vert/bbox/sem_label/ins_label` NPY；semantic/instance 是仅用于测试 loader 兼容的等长零占位，不可用于训练。
- 详情：评估入口调用 V-DETR 原生 `main.py --test_only --auto_test`，每个 query 只保留 argmax class；依次执行 `p_obj >= 0.5`、`max p_cls >= 0.5`、默认空框过滤（至少 5 点）和 V-DETR 默认 class-wise 3D NMS@0.25，导出通用 `Bbox` txt 后交给 `eval_hierarchical.py` 报告 IoU 0.25/0.50 的 P/R/F1，不计算原生 AP。
- 详情：修复当前代码与发布 checkpoint 间缺失模型参数的 `auto_reload` 兼容，以及 ScanNet tensor-list point cloud 在单/多 GPU gather 中的错误，使原生 CUDA 空框过滤可以接收 `[B,N,C]` 点云。

### 文件
- `V-DETR/prepare_scannet18_eval_data.py`：新增支持 process workers、shards、断点跳过、manifest 和 verify-only 的测试数据适配器。
- `V-DETR/main.py`：新增按 scene index 导出 NMS 后 ScanNet18 bbox、类别别名归一化与跳过 native AP 开关。
- `V-DETR/engine.py`：按 inference 参数显式启用 V-DETR 空框过滤。
- `V-DETR/utils/ap_calculator.py`：累积 NMS 后预测时使用 dataset `scan_idx`，保证单/多 GPU 下预测与 scene id 一致。
- `V-DETR/utils/dist.py`：正确 stack/gather ScanNet collate 返回的 point-cloud tensor list。
- `V-DETR/run_scannet18_inference_nms_eval.sh`：新增 `vdetr` conda 环境下的数据校验、单/多 GPU 预测导出和通用 P/R/F1 评估入口。
- `V-DETR/configs/scannet18_full_val_eval.yaml`：完整 312-scene val 的数据、checkpoint、后处理与输出配置。
- `V-DETR/run_scannet18_full_val_inference_nms_eval.sh`：读取 YAML、增量准备/校验 full-val 数据并调用通用 V-DETR pipeline。
- `V-DETR/run_scannet18_full_val_map_prf_eval.sh`：顺序执行 V-DETR native mAP 和 common P/R/F1 两个 full-val 阶段，避免混用不同 proposal 定义。
- `V-DETR/run_scannet18_full_val_prf_no_empty_eval.sh`：在独立输出目录评估不做预测空框过滤的 full-val P/R/F1。
- `V-DETR/SCANNET18_EVAL.md`：新增数据处理、完整性检查、推理和指标解释命令。

## 未提交：添加 SpatialLM ScanNet18 baseline 推理、NMS 和评估流程
- 摘要：复用 `test_model_checkpoint.py`、`apply_bbox_nms.py` 和 `eval_hierarchical.py`，新增面向 ScanNet test-half 的 SpatialLM 单阶段 bbox baseline 多 GPU 分片推理、NMS 和 object-only 评估入口。
- 详情：运行脚本使用数据 JSON 中与微调一致的 `Detect boxes.` prompt，支持自动选择输出目录中最新的有效 checkpoint，并在推理完成后校验 156 个 test-half 预测文件。

### 文件
- `run_scripts/inference_and_eval/run_scannet18_spatiallm_baseline_inference_nms_eval.sh`：新增 SpatialLM baseline 一键 pipeline，预测和评估产物默认写入 `/nas1/chenjunqing2024/scannet`。

## 未提交：添加 ScanNet18 分层推理、NMS 和评估流程
- 摘要：复用现有分层推理和按类别 3D bbox NMS，新增面向 ScanNet test-half 的多 GPU 分片推理、结果完整性校验与评估编排脚本。
- 详情：扩展分层 evaluator，支持每行一个 scene id 的 txt split、显式 object class 列表和禁用 label mapping，从而直接评估 ScanNet18（含 door/window）。

### 文件
- `eval_hierarchical.py`：新增 txt metadata、`--object_classes` 和 `--no_label_mapping`，同时保留原有 SpatialLM20 默认行为。
- `run_scripts/inference_and_eval/run_scannet18_ours_inference_nms_eval.sh`：新增 ScanNet18 test-half 一键 inference + NMS + object/region eval 入口，一张 GPU 对应一个 shard，所有预测和指标产物默认写入 `/nas1/chenjunqing2024/scannet`。

## 未提交：添加在线 point-token scorer 联合训练
- 摘要：使用原始 SpatialLM 初始化 scorer，并在线联合训练 point encoder、hard scorer 和 LLM。
- 详情：为 Sonata 添加 packed-batch 编码路径，用单次 encoder forward 处理 local batch，并返回 final token 的 batch/offset 元数据。
- 详情：在 GPU 上按 batch 计算 final voxel 与 1.2 倍 GT object bbox 的重叠标签；scorer 使用 BCE loss，hard-filtered token 的 next-token loss 回传到 LLM、point projector 和 point encoder。
- 详情：原始 SpatialLM scorer 初始化直接读取 raw PCD，在线执行 packed-batch encoder/projector forward 和 GPU GT mask 构建，不再预计算 feature cache；支持 cosine warmup、最新 checkpoint 恢复、剩余 step scheduler 重建、WandB clean-copy 和 CUDA_VISIBLE_DEVICES 记录。

### 文件
- `spatiallm/model/point_token_scorer.py`：抽取共享 scorer，并新增 packed-to-padded、batched GPU bbox label、BCE 和顺序保持的 hard selection。
- `spatiallm/model/sonata_encoder.py`：返回 final token 的 `batch` 和 `offset`。
- `spatiallm/model/spatiallm_llama.py`、`spatiallm/model/spatiallm_qwen.py`：添加 packed point-cloud batch encoding 和联合 scorer loss/selection 路径。
- `spatiallm/tuner/data/mm_plugin.py`：支持 packed point-cloud collate，并单独输出 scorer GT bboxes，不启用 GT oracle filtering。
- `spatiallm/tuner/data/template.py`、`spatiallm/tuner/hparams/data_args.py`、`spatiallm/tuner/trainer.py`：接线 batch encoding 和 online scorer GT mask 数据参数。
- `train_point_token_scorer.py`：保留 cache-based scorer trainer，供已有离线缓存实验复现。
- `train_point_token_scorer_online.py`：新增原始 SpatialLM scorer 在线初始化训练器；只将 frozen point encoder/projector 搬到 GPU，raw PCD augmentation、packed encoding、GPU GT mask 和 scorer BCE 在同一训练流程完成，并在 checkpoint 保存累计标签统计。
- `train_stage2_joint_scorer.py`：新增 online hard-selection 联合训练器、scorer 初始化加载、独立 scorer checkpoint 和联合指标记录。
- `configs/scorer/point_token_scorer_spatiallm_original.yaml`：原始 SpatialLM encoder/projector 的 scorer 初始化训练配置。
- `configs/spatiallm_stage2_joint_scorer.yaml`：stage2 online LLM + scorer 联合训练配置；从 `configs/scorer/` 移出并添加 `spatiallm_` 前缀，使该子目录只保留纯 scorer 训练配置。
- `run_scripts/run_train_point_token_scorer_spatiallm_original.sh`、`run_scripts/run_train_stage2_joint_scorer.sh`：在线 scorer 初始化和联合训练入口；原始 SpatialLM scorer 不再需要独立 cache 预计算入口。

## 5e8212c08e82f1335ea75dfdd2239fddd3465fce
- 摘要：Add cached point-token stage-2 scorer and filtering workflows
- 作者：Junqing Chen <junqingchen03@gmail.com>
- 日期：2026-07-19 12:35:55 +0800
- 详情：扩展 point-token scorer 缓存预计算，支持确定性 epoch 分片、对齐的消息缓存及分片索引合并。
- 详情：为 SpatialLM Llama/Qwen 添加预计算 point-token feature 输入，并新增 attention scorer、过滤 token 训练和 oracle 推理流程。
- 详情：补充缓存过滤、复现验证、I/O 测量、压力测试、评估结果和后续开发上下文，同时更新实验规范与启动配置。

### 文件
- `.gitignore`：扩展本地产物忽略规则，覆盖 `logs/`、整个 `configs/` 目录、`data/` 和 `run_scripts/`。
- `AGENTS.md`：补充训练任务的 WandB 记录、cosine scheduler、恢复训练、YAML 配置、shard 数据处理和 DataLoader 效率规范。
- `CHANGELOG.md`：添加上一条 point-token bbox mask、evict 和 scorer 工具提交的元数据及逐文件摘要。
- `NEXT_AGENT_CONTEXT.md`：新增开发交接文档，汇总仓库结构、两阶段方法、缓存与 scorer 工作流、实验结果、常用命令和待办事项。
- `build_attention_scorer_stress_topk_cache.py`：新增 attention scorer 压力样本构建器，按插入后的序列长度从 feature/message 缓存中选取 top-k 样本。
- `eval_results.md`：新增评估对比文档，以表格汇总不同分层、mask、evict、scorer 和 attention-oracle 方法的 region、layout 与 object 指标。
- `filter_point_tokens_with_scorer.py`：新增 scorer 驱动的 point-token 缓存过滤器，支持批量打分、阈值与数量约束、并行分片和对齐消息缓存。
- `inference_hierarchical_attention_oracle.py`：新增 attention-oracle 分层推理脚本，使用第 2 阶段模型的末层注意力分数筛选 point token。
- `measure_pt_read_time.py`：新增 `.pt` 缓存读取基准工具，可统计文件、item 和字节吞吐并导出逐文件 CSV。
- `merge_filtered_point_token_cache_indices.py`：新增过滤缓存的并行分片索引合并工具，负责校验元数据、排序样本并汇总 token 统计。
- `merge_point_token_cache_indices.py`：新增 scorer feature 缓存及可选 message 缓存的 epoch 分片索引合并工具。
- `precompute_point_token_cache_messages.py`：新增与现有 point-token feature shard 对齐的预处理消息缓存生成工具。
- `precompute_point_token_scorer_data.py`：添加全随机源确定性播种、显式 epoch 选择与分片、同步消息缓存输出、覆盖保护和部分索引写入。
- `spatiallm/model/spatiallm_llama.py`：支持跳过 point backbone、直接插入带 NaN padding 的预计算 point-token features，并将其贯穿生成输入和标签对齐逻辑。
- `spatiallm/model/spatiallm_qwen.py`：支持跳过 point backbone、直接插入带 NaN padding 的预计算 point-token features，并将其贯穿生成输入和标签对齐逻辑。
- `stress_test.py`：新增通用 GPU 显存与矩阵乘压力工具，可按目标显存、持续时间和 dtype 运行负载。
- `stress_test_stage2_attention_scorer.py`：新增第 2 阶段 attention scorer 压力测试，使用最长缓存样本测量固定 batch 的训练、评估、吞吐和峰值显存。
- `train_stage2_attention_scorer.py`：新增端到端 attention scorer 训练器，通过冻结的第 2 阶段 SpatialLM 优化 attention bias，并支持 YAML、WandB clean-copy、cosine warmup、分片 DataLoader 和 checkpoint 恢复。
- `train_stage2_filtered_point_tokens.py`：新增从 scorer 过滤后的投影 point-token 缓存训练第 2 阶段 SpatialLM 的流程，支持 shard-local batch、YAML 配置、WandB 和按剩余步数重建 scheduler 的自动恢复。
- `verify_point_token_scorer_cache_reproducibility.py`：新增 scorer 缓存复现检查器，从元数据恢复样本并比较 feature、grid、标签和区域中心。
- `wait_for_gpu_and_train.sh`：将 GPU 排队启动目标切换为过滤 point-token 的第 2 阶段训练脚本。

## 02c6f623920e0b3f2b90ebef99e929214794a68c
- 摘要：Add point-token bbox masking, evict workflows, and scorer tooling
- 作者：Codex <codex@openai.com>
- 日期：2026-07-06 16:43:52 +0800
- 详情：为 Sonata 点编码器添加基于 bbox 重叠的点 token 过滤，并将 keep bbox 参数贯穿 Llama/Qwen 生成流程。
- 详情：为第 2 阶段训练添加 GT bbox 点 token mask 的数据参数、模板注册和多模态插件接线。
- 详情：添加 evict 数据集构建、oracle/预测区域推理变体，以及点 token scorer 的缓存预计算、训练、压力测试和推理工具。
- 详情：翻译并扩展更新日志，更新本地忽略规则、AGENTS 指南和 GPU 启动脚本默认配置。

### 文件
- `.gitignore`：添加 `proposal/`、`reports/` 和 `.vscode/` 本地产物忽略规则。
- `AGENTS.md`：新增仓库协作说明，声明主要使用中文交流并保留必要英文术语。
- `CHANGELOG.md`：将既有更新日志翻译为中文，并保留上一条世界尺寸点 token 控制提交的记录。
- `build_stage2_evict_dataset.py`：新增第 2 阶段 evict 数据集构建脚本，用 GT 对象和布局框筛选区域点云并写出 `_evict` 数据集与统计信息。
- `inference_hierarchical_bbox_mask.py`：新增 oracle 第 2 阶段推理入口，默认启用 GT bbox 点 token mask。
- `inference_hierarchical_evict.py`：新增基于预过滤区域 PCD 的 oracle 第 2 阶段分层推理脚本，并支持可选 GT bbox mask。
- `inference_hierarchical_gt_region.py`：新增直接在 GT-region 点云 JSON 上执行第 2 阶段推理的入口。
- `inference_hierarchical_pred_region_bbox_mask.py`：新增预测区域分层推理入口，默认关闭点云 evict 并启用 GT bbox 点 token mask。
- `inference_hierarchical_pred_region_evict.py`：新增预测区域推理脚本，支持按 GT 辅助筛选区域点、GT bbox token mask、NMS、分片和错误续跑。
- `inference_hierarchical_scorer.py`：新增 scorer 驱动的分层推理脚本，用训练好的点 token scorer 选择第 2 阶段 token。
- `precompute_point_token_scorer_data.py`：新增 scorer 数据预计算脚本，缓存点编码器 context 特征、grid 坐标和 GT bbox mask 标签。
- `spatiallm/model/sonata_encoder.py`：添加点 token 与旋转 bbox 的重叠 mask 计算，支持按 bbox 保留 token，并可返回编码后的 grid 坐标。
- `spatiallm/model/spatiallm_llama.py`：将 world_size 传给 Sonata，并在前向和生成输入中透传 `point_token_keep_bboxes`。
- `spatiallm/model/spatiallm_qwen.py`：将 world_size 传给 Sonata，并在前向和生成输入中透传 `point_token_keep_bboxes`。
- `spatiallm/tuner/data/mm_plugin.py`：添加 GT bbox 点 token mask 开关、bbox 扩展比例、batched keep bbox 构造，以及训练数据预处理中的 keep bbox 输出。
- `spatiallm/tuner/data/template.py`：在 SpatialLM Llama/Qwen 模板注册时传入 bbox mask 相关配置。
- `spatiallm/tuner/hparams/data_args.py`：新增 `point_token_bbox_mask` 和 `point_token_bbox_expand_ratio` 数据参数，并校验扩展比例非负。
- `spatiallm/tuner/trainer.py`：训练注册模板时传入 bbox mask 相关数据参数。
- `stress_test_point_token_scorer_memory.py`：新增 scorer 显存压力测试脚本，用最长缓存样本重复执行训练和评估循环。
- `train_hierarchical_spatiallm_mask.py`：新增分层训练编排脚本，强制第 2 阶段启用 GT bbox 点 token mask 并支持阶段级覆盖参数。
- `train_point_token_scorer.py`：新增点 token scorer 训练脚本，包含 transformer scorer、缓存数据集、分 shard batch、评估、checkpoint 和 WandB 支持。
- `wait_for_gpu_and_train.sh`：更新默认排队命令为 point-token scorer 训练脚本，并支持未配置日志文件时启动和跳过 OOM 日志检查。

## 8015cbe0000d60700a1ec847dc4b9a405ecba6a6
- 摘要：添加世界尺寸点 token 控制和第 2 阶段诊断功能。
- 作者：Codex <codex@openai.com>
- 日期：2026-06-25 11:09:23 +0800
- 详情：添加可配置的布局世界范围、点 token 中心裁剪，以及训练和推理所需的对应数据、模型、训练器接线。
- 详情：扩展分层推理和训练辅助工具，加入 GT 区域第 2 阶段评估、世界尺寸和 token 覆盖项，并更新默认启动配置。
- 详情：添加最坏情况第 2 阶段压力训练和场景 PLY 可视化工具，刷新忽略规则，并包含上一条提交的更新日志条目。

### 文件
- `.gitignore`：更新本地产物忽略规则，覆盖保存目录、运行脚本、分析脚本和 nohup 输出。
- `CHANGELOG.md`：添加上一条预测诊断提交的元数据和文件级更新日志条目。
- `configs/spatiallm_hierarchical_20000.yaml`：更新 GPU05 分层实验的 WandB run 名称。
- `inference_hierarchical.py`：添加可配置的世界尺寸预处理、可选的 GT 区域第 1 阶段旁路、第 2 阶段区域中心裁剪，以及感知世界尺寸的解码。
- `spatiallm/layout/entity.py`：添加世界尺寸预设，并对墙、门、bbox 和区域应用感知世界尺寸的归一化与反归一化。
- `spatiallm/layout/layout.py`：将可选世界尺寸贯穿到网格尺寸计算、归一化和反归一化流程中。
- `spatiallm/model/__init__.py`：添加共享辅助函数，用于对编码后的点 token 序列进行中心裁剪。
- `spatiallm/model/spatiallm_llama.py`：在投影前，对 SceneScript 和 Sonata 点特征应用可选的最大点 token 裁剪。
- `spatiallm/model/spatiallm_qwen.py`：在投影前，对 SceneScript 和 Sonata 点特征应用可选的最大点 token 裁剪。
- `spatiallm/tuner/data/mm_plugin.py`：添加感知世界尺寸的量化、点云中心裁剪、裁剪后的 bbox 过滤，以及裁剪布局离散化。
- `spatiallm/tuner/data/template.py`：将世界尺寸传入 SpatialLM 多模态插件注册流程。
- `spatiallm/tuner/framework/loader.py`：在模型点配置中保存世界尺寸和最大点 token 限制。
- `spatiallm/tuner/hparams/data_args.py`：添加经过校验的 world_size 和 max_point_tokens 数据参数。
- `spatiallm/tuner/trainer.py`：注册 SpatialLM 模板时传入世界尺寸。
- `train_hierarchical_spatiallm.py`：添加按阶段区分的世界尺寸、最大 token 和输出目录覆盖项，并记录这些解析后的设置。
- `train_stage2_longest_token_batch.py`：新增压力测试工具，它会根据点 token 统计构建固定的最坏情况第 2 阶段 batch，并启动受控训练。
- `visualize_scene_ply.py`：新增可视化工具，为场景点、预测结果、GT 布局、区域和 bbox 写出 PLY 叠加层及图例。
- `wait_for_gpu_and_train.sh`：更新分层训练的默认排队命令和日志路径。

## 46e64d02417fb6e0c0da0b8e78819c62f88fa12c
- 摘要：添加预测诊断功能和灵活的分层工作流控制。
- 作者：Codex <codex@openai.com>
- 日期：2026-06-07 11:48:38 +0800
- 详情：添加预测重叠和区域尺度分析器，并加入独立的按类别 bbox NMS 后处理。
- 详情：扩展最差预测排序，加入区域评分和完整 CSV 导出，并让场景渲染收集支持可选执行和灵活路径。
- 详情：添加仅第 2 阶段训练、checkpoint 恢复校验、按阶段覆盖 step、更新后的忽略规则，以及上一条提交的更新日志。

### 文件
- `.gitignore`：忽略生成的 YAML 配置和本地运行脚本，同时保留主训练配置继续纳入版本管理。
- `CHANGELOG.md`：添加分层工具提交的仓库提交元数据和逐文件变更摘要。
- `analyze_prediction_overlaps.py`：新增工具，用于测量预测 bbox 的两两重叠，写出逐场景和排序后的 CSV 报告，并生成同类别 IoU 直方图。
- `analyze_region_scale_histograms.py`：新增工具，用于分析扩展区域尺寸，并导出尺度统计、CSV 数据和直方图。
- `apply_bbox_nms.py`：新增工具，对扁平或分层预测目录应用基于顺序、按类别执行的 3D bbox NMS。
- `filter_worst_predictions.py`：添加区域预测评分、派生或显式 GT 区域支持、可复用实体匹配，以及完整排序 CSV 导出。
- `gather_scene.py`：让渲染产物变为可选，支持多种渲染目录布局，并一致地替换已有的复制输出。
- `train_hierarchical_spatiallm.py`：添加仅第 2 阶段执行、第 1 阶段 checkpoint 恢复校验、配置别名归一化、模型覆盖项，以及按阶段设置保存和评估 step 的控制。

## f6b175e03d03336108ae837afe82de094dde3f66
- 摘要：添加分层区域训练和推理工具。
- 作者：Codex <codex@openai.com>
- 日期：2026-05-30 13:04:43 +0800
- 详情：添加 Region 布局支持，以及用于分层数据集生成、推理、评估、渲染和 checkpoint 测试的脚本。
- 详情：添加分层训练配置和编排辅助工具，并更新 SFT batch、bf16 和学习率默认值。
- 详情：为子进程训练设置仓库根目录 PYTHONPATH，并忽略本地数据集、输出、保存目录和实验产物。

### 文件
- `.gitignore`：添加被忽略的本地数据集、输出、保存、渲染、WandB、临时文件和脚本产物路径。
- `build_hierarchical_region_dataset.py`：新增脚本，用于采样场景、构建第 1 阶段区域和第 2 阶段 bbox 的 ShareGPT 数据集、写出区域 PCD 和元数据，并更新数据集信息。
- `configs/spatiallm_hierarchical_20000.yaml`：新增 20k 样本两阶段分层训练配置，包含基础 SFT 设置以及阶段专用数据集和输出目录。
- `configs/spatiallm_sft.yaml`：增大梯度累积和学习率，并启用 bf16 训练。
- `copy_worst_test_pcd.py`：新增工具，用于复制 CSV 中列出的场景 ID 对应的 PLY 点云。
- `eval_hierarchical.py`：新增评估脚本，用于基于布局、对象和区域指标评估分层预测，并支持可选 JSON 输出。
- `filter_worst_predictions.py`：新增评分和导出工具，按感知类别的 bbox 匹配指标对低质量预测场景进行排序。
- `gather_scene.py`：新增工具，用于把选中场景的 PCD、GT 布局和区域渲染产物收集到示例文件夹。
- `generate_region_bboxes.py`：新增工具，用于将对象框聚类成区域框，写出原始和扩展区域，渲染叠加层，并记录 manifest。
- `inference_hierarchical.py`：新增两阶段推理脚本，用于预测区域、执行逐区域 bbox 检测、应用可选 NMS，并写出阶段输出和最终输出。
- `inference_hierarchical_v2.py`：新增逐场景产物封装器，用于分层推理，负责复制输入、写出预测、渲染叠加层并保存元数据。
- `inference_v2.py`：新增单模型场景推理工具，用于写出预测，并生成俯视 GT 和预测区域渲染图。
- `poetry.toml`：新增 Poetry 设置，为当前 checkout 禁用虚拟环境创建。
- `render_topdown_arkitscenes.py`：新增渲染器，用于在按顶部高度裁剪后生成 ARKitScenes 点云的俯视图，并支持可选调试产物。
- `spatiallm/layout/entity.py`：添加 Region 实体，支持变换、归一化、序列化和排序。
- `spatiallm/layout/layout.py`：在墙、门、窗和 bbox 之外，解析、存储、导出、排序并序列化 Region 实体。
- `test_model_checkpoint.py`：新增 checkpoint 推理工具，支持 PLY 文件、PLY 目录或 JSON 数据集，并提供子集和分片控制。
- `train.py`：把仓库根目录添加到 PYTHONPATH，并将准备好的环境传给 subprocess.run。
- `train_hierarchical_spatiallm.py`：新增编排器，用于解析阶段配置、启动第 1 阶段训练、查找最新 checkpoint，并启动第 2 阶段训练。
- `wait_for_gpu_and_train.sh`：新增 GPU 轮询启动器，会等待低显存占用的 GPU、分发排队命令、写入日志，并重试疑似失败的启动。
