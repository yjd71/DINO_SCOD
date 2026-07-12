# RSBL DINO PC-HBM

本页记录 DINO-PC-HBM 的独立入口。原有 `train_base_model.py`、
`train_ts_model.py`、`train_ts_model_pseudo.py`、SAM 训练器与不带 memory 的
`inference.py --checkpoint ...` 调用仍然保留。

## 前置条件

- 所有命令均在仓库根目录、Conda `yjd` 环境执行。
- DINOv2 代码位于 `./dinov2/`，权重位于
  `./weight/dinov2_vitb14_pretrain.pth`。
- 数据目录与 `configs/base_model_config.py`、`configs/ts_model_config.py`
  中的路径一致。
- PC-HBM memory 只由无增强 labeled 数据重建，固定保存为 CPU FP16；
  unlabeled pseudo 数据不会写入 memory。

## Base PC-HBM

单卡训练：

```powershell
conda run -n yjd python train_base_model_pc_hbm.py `
  --output-dir .\results\base_pc_hbm `
  --epochs 30
```

两卡 DDP：

```powershell
conda run -n yjd python -m torch.distributed.run --standalone --nproc_per_node=2 `
  train_base_model_pc_hbm.py --output-dir .\results\base_pc_hbm --epochs 30
```

Base 使用 1-based 调度：epoch 1–5 为 `off`，6–10 为
`parent_only`，11–30 为 `full`。恢复训练使用精确 resume artifact：

```powershell
conda run -n yjd python train_base_model_pc_hbm.py `
  --output-dir .\results\base_pc_hbm `
  --resume .\results\base_pc_hbm\training_resume.pth `
  --epochs 30
```

每个导出 epoch 会生成独立 Decoder 与 memory，例如
`base_pc_hbm_decoder_epoch_30.pth` 和
`base_pc_hbm_memory_epoch_30.pth`；`training_resume.pth` 还包含 optimizer、
scheduler、GradScaler、EMA memory producer、配置和 RNG 状态。

## 在线 pseudo Teacher/Student PC-HBM

TS 默认拒绝不完整的 legacy Base 权重，并要求完整 Base PC-HBM checkpoint：

```powershell
conda run -n yjd python train_ts_model_pseudo_pc_hbm.py `
  --base-pc-checkpoint .\results\base_pc_hbm\base_pc_hbm_decoder_epoch_30.pth `
  --output-dir .\results\pc_hbm\ts_model `
  --epochs 15
```

恢复 TS 时仍应提供同一个 Base 初始化 artifact，并同时指定 resume：

```powershell
conda run -n yjd python train_ts_model_pseudo_pc_hbm.py `
  --base-pc-checkpoint .\results\base_pc_hbm\base_pc_hbm_decoder_epoch_30.pth `
  --resume .\results\pc_hbm\ts_model\ts_pc_hbm_resume_latest.pth `
  --output-dir .\results\pc_hbm\ts_model `
  --epochs 15
```

训练结束后会用最终 Student 再建一次 memory，并导出匹配的一对
`ts_pc_hbm_student_final.pth` 与 `ts_pc_hbm_memory_final.pth`。这两个文件是
推荐的推理 artifact。

`--allow-legacy-pc-init` 仅用于显式迁移实验：它会随机初始化旧 checkpoint
中不存在的 PC-HBM 参数，不应用于正式 TS pseudo 训练，也不能据此宣称获得
PC-HBM 训练结果。

## 推理

推荐用最终 Student/Memory 配对，并校验 producer fingerprint：

```powershell
conda run -n yjd python inference.py `
  --decoder-checkpoint .\results\pc_hbm\ts_model\ts_pc_hbm_student_final.pth `
  --memory-checkpoint .\results\pc_hbm\ts_model\ts_pc_hbm_memory_final.pth `
  --require-producer-match `
  --pred-root .\results\pc_hbm\predictions
```

memory 通过架构、schema、输入/输出尺寸、DINO 层索引、维度、dtype 和 source
校验时，模型返回 `z_final` logits；memory 缺失、未就绪或不兼容时会发出
`RuntimeWarning` 并回退到 `z_main` logits。`inference.py` 只在保存预测图前
执行一次 sigmoid，checkpoint 内的 `p_final` 不会被再次当作 logits 使用。

旧调用仍可用：

```powershell
conda run -n yjd python inference.py --checkpoint .\path\to\legacy_decoder.pth
```

旧 raw state dict 可以在无 memory 的 baseline 路径加载；当提供 memory 时，
Decoder checkpoint 必须包含完整 PC-HBM 参数，任何非 PC missing key、
unexpected key 或部分 PC state 都会报错。

## 验收

```powershell
conda run -n yjd python -m pytest -q
conda run -n yjd python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu
conda run -n yjd python tests/cuda_smoke_pc_hbm.py
```

若 Windows 的 PyTorch 构建不含 libuv，可在 torchrun 前让仓库内的严格守卫
`sitecustomize.py` 对父启动器生效（普通 Python/pytest 不会 patch）：

```powershell
$env:PYTHONPATH=(Get-Location).Path
conda run -n yjd python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu
```

CUDA smoke 不降低物理 batch，依次覆盖 Base batch 16 full backward、Teacher
batch 32 inference、Student labeled batch 32 full backward、Student unlabeled
batch 32 core backward。若默认配置在约 12 GiB GPU 上 OOM，脚本只按锁定顺序
尝试：chunk 512→256、P1 token 384→256、P3/P2 token 128→96、top-K
16→12；全部失败时以非零状态退出并明确报告未通过。

这些 smoke 只验证正确性、显存和训练链路，不执行完整 30+15 epoch，也不代表
COD 指标提升。SAM/SAM2 checkpoint 与原 SAM trainer 不属于此 PC-HBM 入口。
