我们将主要使用中文交流。如果某些术语用英文表达更自然，可以继续保留英文。用户有时也可能在中文中夹杂少量英文。

所有训练任务默认都必须使用 wandb 记录实验，并且必须显式指定 wandb project name 和 wandb run name。训练代码默认将 `CUDA_VISIBLE_DEVICES` 写入 wandb config，方便之后追踪实验实际使用的 GPU。训练代码默认不使用 W&B 原生 resume；如需接续已有曲线，默认采用 clean-copy 方式：从旧 run 复制到 resumed checkpoint step，再在新的 clean run 中继续记录。

撰写训练代码时，默认使用 cosine lr scheduler，默认 warmup ratio 为 0.03。训练代码默认支持从 output directory 中最新的 checkpoint 恢复训练；恢复训练时仍使用 warmup ratio，并按照剩余 optimizer steps 计算 warmup steps。训练参数默认通过 YAML config 文件传入，而不是通过命令行参数逐项传入。

训练代码和数据处理代码应尽量高效：数据处理优先支持 shards；训练 DataLoader 的 `num_workers` 通常默认设为 8；涉及 GPU 的流程应避免 GPU 长时间等待 CPU，例如用 DataLoader 预取数据，而不是每步同步等待 CPU 读文件。效率优化以代码简洁为前提，先采用简单直接的方法，避免过早引入手写任务队列或复杂异步 CPU/GPU 流水线；如果仍不够快，再和用户讨论更复杂的方案。

完成对代码库的修改时，默认应同步检查并更新 `NEXT_AGENT_CONTEXT.md`：记录新增或修改的核心文件、行为、实验状态、评测结果和后续事项，同时修正或删除已经过时的交接信息，使下一位 agent 可以直接依据该文档继续工作。
