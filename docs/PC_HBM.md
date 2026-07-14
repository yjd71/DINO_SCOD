# RSBL DINO PC-HBM

`train_base_model_pc_hbm.py` 默认使用 `two_stage` 完成整个预热流程；后续
Teacher–Student 入口仍使用 `teacher_only` 协议。原始 Base、TS、pseudo、SAM trainer
均不受影响。

## Two-stage Base 与 Teacher-only TS 数据流

- Base 固定冻结 DINO，但不冻结 Decoder。epoch 1–5 为 `off`，只训练 legacy Decoder；
  epoch 6–10 为 `parent_only`，epoch 11–30 为 `full`，后两阶段联合训练 legacy Decoder
  与 `pc_hbm.*`。full correction 在 epoch 11/12/13 的强度为 1/3、2/3、1。
- Base 可以从随机 Decoder 开始完成全部预热，不要求外部 baseline checkpoint；也可以用
  `--baseline-checkpoint` 选择性 warm-start legacy Decoder。
- TS 标签分支和 Student 无标签分支都走原始 `off` 路径。只有 Teacher 无标签分支
  使用 PC-HBM，并用纠正后的概率、P3 和 P2 特征蒸馏 raw Student。
- 无标签训练同时使用置信度加权 soft pseudo loss 与 `L_u_hard`：`p_final>=0.5` 二值化，仅保留 `p_final>=0.70` 的可靠前景和 `p_final<=0.30` 的可靠背景；hard loss 默认权重为 2.0，并在 TS 前 3 个 epoch 线性升温。P3/P2 特征蒸馏保持启用。
- TS 最终导出 `student_raw.pth`，其中没有 `pc_hbm.*`，推理不需要 memory。

标签数据仍用于 PC-HBM 监督和 labeled-only memory，但标签样本的纠正特征和纠正预测
不会进入 Student 或最终输出。Base 与 TS 必须使用同一个 labeled split：省略
`--labeled-indices-pt` 时，两者都读取 `./Dataset/COD/sampled_images.txt`；显式传入 `.pt`
时，该文件覆盖 txt 选择。checkpoint 会验证 stable sample-key split fingerprint。

## Base enhancer

PowerShell 单卡：

```powershell
conda run -n yjd python train_base_model_pc_hbm.py `
  --output-dir ./results/base_teacher_enhancer `
  --batch-size 16 `
  --epochs 30
```

Linux/Bash 双卡：

```bash
conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train_base_model_pc_hbm.py \
  --output-dir ./results/base_teacher_enhancer \
  --batch-size 16 \
  --epochs 30
```

`--batch-size` 表示每个 rank 的物理 batch；双卡且每卡 16 时 global batch 为 32。
上述命令不需要 baseline。若已有可靠的 legacy Decoder，可选择性追加：

```bash
--baseline-checkpoint ./results/baseline/base_decoder.pth
```

该参数只做 warm-start，不会改变 `off → parent_only → full` 的 two-stage 调度，也不会
在后两阶段冻结 legacy Decoder。

如需用 `.pt` 覆盖默认的 `sampled_images.txt`，请在 Base、TS 和 resume 命令中都追加
同一个 `--labeled-indices-pt ./data/labeled_indices.pt`。

训练结束后主要产物为：

- `teacher_enhancer.pth`：完整 Teacher Decoder。
- `teacher_enhancer_memory.pth`：由最终 Teacher producer 重建的 CPU FP16 memory。
- `training_resume.pth`：optimizer、scheduler、GradScaler、EMA producer、配置和 RNG。

恢复时提供同一个 labeled split，不要求再次提供 baseline warm-start：

```bash
conda run -n yjd python train_base_model_pc_hbm.py \
  --resume ./results/base_teacher_enhancer/training_resume.pth \
  --output-dir ./results/base_teacher_enhancer \
  --batch-size 16 \
  --epochs 30
```

## Teacher–Student 蒸馏

Linux/Bash 双卡：

```bash
conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train_ts_model_pseudo_pc_hbm.py \
  --training-design teacher_only \
  --teacher-pc-checkpoint ./results/base_teacher_enhancer/teacher_enhancer.pth \
  --output-dir ./results/teacher_only_ts \
  --epochs 15
```

`--base-pc-checkpoint` 是 `--teacher-pc-checkpoint` 的兼容别名。未指定
`--student-checkpoint` 时，raw Student 从 enhancer checkpoint 的非 PC 权重初始化。
Teacher legacy 参数按名称跟随 Student EMA，Teacher `pc_hbm.*` 始终冻结。

TS resume：

```bash
conda run -n yjd python train_ts_model_pseudo_pc_hbm.py \
  --training-design teacher_only \
  --teacher-pc-checkpoint ./results/base_teacher_enhancer/teacher_enhancer.pth \
  --resume ./results/teacher_only_ts/ts_pc_hbm_resume_latest.pth \
  --output-dir ./results/teacher_only_ts \
  --epochs 15
```

## 推理

Teacher-only 最终 Student 只走 baseline/off 路径：

```powershell
conda run -n yjd python inference.py `
  --decoder-checkpoint ./results/teacher_only_ts/student_raw.pth `
  --pred-root ./results/teacher_only_ts/predictions
```

不要给 `student_raw.pth` 配 memory。TS 必须继续使用
`--training-design teacher_only`；Base 的 two-stage 预热 checkpoint 同时提供其 legacy
初始化和冻结的 Teacher PC-HBM enhancer。

## 验收

```powershell
conda run -n yjd python -m pytest -q
conda run -n yjd python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu
conda run -n yjd python tests/cuda_smoke_pc_hbm.py
```

Windows 的非 libuv PyTorch 需要让仓库内的 `sitecustomize.py` 在父进程启动前可见：

```powershell
$env:PYTHONPATH=(Get-Location).Path
conda run -n yjd python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu
```

部分 Windows PyTorch 构建即使完成 rendezvous 仍不提供可用 Gloo device；这种情况下
CPU/Gloo smoke 必须在 Linux 训练服务器执行。CUDA smoke 固定覆盖 Base batch 16
真实 joint labeled full backward（同时检查 legacy 与 PC-HBM finite gradients）、
Teacher batch 32 inference、Student labeled batch 32 raw backward，
以及 Student unlabeled batch 32 raw soft/feature-distillation backward；OOM 时只按脚本中锁定的
chunk/token/top-K 顺序降容量，不降低 batch size。
