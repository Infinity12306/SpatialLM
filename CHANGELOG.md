# 更新日志

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
