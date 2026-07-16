# DualUCOD BGFBR × PC-HBM 工程与实验手册

本扩展保留冻结的 DINOv2-B/14 编码器以及 `392 → 28×28 → 98×98` 合同，主实验使用
`bgfbr_pc_v1`，并通过显式 profile 保留 legacy Transformer 对照。完整训练结果不随工程提交
预填；每次运行应复制
[`experiments/BGFBR_PC_HBM_REPORT_TEMPLATE.md`](../experiments/BGFBR_PC_HBM_REPORT_TEMPLATE.md)
并记录真实指标。

## 1. Profile 合同

| Profile | Decoder | Base 模式/消融 | TS 用法 |
|---|---|---|---|
| `bgfbr_pc` / `default` | BGFBR | 1–5 off、6–10 parent、11–30 full | 主实验 |
| `bgfbr_off` | BGFBR | 全程 off | Base-only 对照；不会改写 TS 固定 pseudo/core 模式 |
| `parent_only` | BGFBR | 全程 parent_only | Base-only 对照 |
| `no_gbe` | BGFBR | GBE 输出置零 | 与同 profile Teacher、memory 配套 |
| `no_ode` | BGFBR | ODE identity bypass | 同上 |
| `no_rcab` | BGFBR | RCAB identity bypass | 同上 |
| `no_pc_boundary_context` | BGFBR | PC boundary context 置零，通道仍为 7/10/10/16 | 同上 |
| `legacy_off` | legacy Transformer | 全程 off | legacy parity 对照 |

每个 profile 使用独立输出目录。组件开关属于 memory schema v2 的兼容合同，禁止跨 profile
复用 `teacher_enhancer_memory.pth`。

## 2. Windows / yjd

PowerShell 直接使用 `yjd` 解释器，避免 `conda run` 在部分中文 Windows 上触发 GBK 输出错误：

```powershell
$python = 'C:\Users\UserY\.conda\envs\yjd\python.exe'

# 主 Base 实验：每卡/进程 batch 16
& $python train_base_model_pc_hbm.py `
  --experiment-profile bgfbr_pc `
  --training-design two_stage `
  --output-dir .\results\bgfbr_pc\base `
  --batch-size 16 --epochs 30 --seed 2025 --deterministic

# Teacher-only TS：CLI 内锁定 labeled/unlabeled batch 32
& $python train_ts_model_pseudo_pc_hbm.py `
  --experiment-profile bgfbr_pc `
  --training-design teacher_only `
  --teacher-pc-checkpoint .\results\bgfbr_pc\base\teacher_enhancer.pth `
  --output-dir .\results\bgfbr_pc\ts `
  --epochs 15 --seed 2025 --deterministic

& $python -m pytest -q
& $python tests\cuda_smoke_pc_hbm.py
```

Windows 的部分 PyTorch 构建没有可用 Gloo device。此时不要把双进程失败归因于模型，按下一节
在 Linux 训练机执行正式 DDP 验收。

## 3. Linux 双进程 DDP

```bash
conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train_base_model_pc_hbm.py \
  --experiment-profile bgfbr_pc \
  --training-design two_stage \
  --output-dir ./results/bgfbr_pc/base \
  --batch-size 16 --epochs 30 --seed 2025 --deterministic

conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train_ts_model_pseudo_pc_hbm.py \
  --experiment-profile bgfbr_pc \
  --training-design teacher_only \
  --teacher-pc-checkpoint ./results/bgfbr_pc/base/teacher_enhancer.pth \
  --output-dir ./results/bgfbr_pc/ts \
  --epochs 15 --seed 2025 --deterministic

conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu
```

`--batch-size 16` 是每 rank batch，双卡 global batch 为 32。TS 的两个 batch 均固定为 32，
不得以降低 batch size 作为通过验收的手段。

## 4. 初始化、迁移与恢复

四种初始化参数互斥：

- `--baseline-checkpoint`：同架构 normal load；BGFBR 目标只接受 BGFBR checkpoint，legacy 目标只接受 legacy checkpoint。
- `--legacy-warm-start`：仅用于 `two_stage + BGFBR`，显式执行
  `load_legacy_into_bgfbr(..., reuse_projectors=True, reuse_pc_core=False)`。
- `--decoder-checkpoint`：仅用于 joint compatibility mode。
- `--resume`：只恢复同 profile、同 split、同训练设计的 format-v2 resume。

```powershell
# legacy → BGFBR 显式迁移；不会导入 Transformer head、旧 PC 或 memory
& $python train_base_model_pc_hbm.py `
  --experiment-profile bgfbr_pc --training-design two_stage `
  --legacy-warm-start .\results\legacy\base_decoder.pth `
  --output-dir .\results\bgfbr_pc_migrated\base --batch-size 16 --epochs 30

# BGFBR 同架构 warm start
& $python train_base_model_pc_hbm.py `
  --experiment-profile bgfbr_pc --training-design two_stage `
  --baseline-checkpoint .\results\bgfbr_pretrain\decoder.pth `
  --output-dir .\results\bgfbr_pc_warm\base --batch-size 16 --epochs 30

# 精确恢复；不要再传 warm-start 参数
& $python train_base_model_pc_hbm.py `
  --experiment-profile bgfbr_pc --training-design two_stage `
  --resume .\results\bgfbr_pc\base\training_resume.pth `
  --output-dir .\results\bgfbr_pc\base --batch-size 16 --epochs 30
```

## 5. Memory 重建

Memory 只能从完整 labeled split 构建，以 CPU FP16 保存，伪标签不得更新它。Base trainer 会按 epoch
重建，并在训练结束由最终 Teacher producer 生成 `teacher_enhancer_memory.pth`。需要重建时：

1. 固定 `--labeled-indices-pt`（或固定 `Dataset/COD/sampled_images.txt`）与 seed；
2. 为新 profile 使用空的独立 `--output-dir`，不要拷贝旧 memory；
3. 正常运行 Base，让 trainer 原子写出 schema-v2 memory；
4. TS 同时传入该目录的 `teacher_enhancer.pth`，保持 profile 与 split 完全一致。

schema v1、producer fingerprint 不一致、GBE/GPM/ODE/RCAB 或 boundary-context 合同不一致都会被拒绝。

## 6. 训练、推理与评估矩阵

对 manifest 中每个 profile 至少完成 Base；`bgfbr_pc` 与四个组件消融继续完成 TS。`bgfbr_off`、
`parent_only`、`legacy_off` 作为 Base-only 架构/阶段对照。

```powershell
# Legacy、BGFBR-off、BGFBR+PC、GBE、ODE、RCAB、edge-context 使用同一入口。
$profiles = @(
  'legacy_off', 'bgfbr_off', 'bgfbr_pc',
  'no_gbe', 'no_ode', 'no_rcab', 'no_pc_boundary_context'
)
foreach ($profile in $profiles) {
  & $python train_base_model_pc_hbm.py `
    --experiment-profile $profile --training-design two_stage `
    --output-dir ".\results\$profile\base" `
    --batch-size 16 --epochs 30 --seed 2025 --deterministic
}

# 主实验与组件消融继续跑 TS；每个 profile 使用自身 Base artifact/memory。
foreach ($profile in @('bgfbr_pc', 'no_gbe', 'no_ode', 'no_rcab', 'no_pc_boundary_context')) {
  & $python train_ts_model_pseudo_pc_hbm.py `
    --experiment-profile $profile --training-design teacher_only `
    --teacher-pc-checkpoint ".\results\$profile\base\teacher_enhancer.pth" `
    --output-dir ".\results\$profile\ts" `
    --epochs 15 --seed 2025 --deterministic
}
```

```powershell
# 推理：component profile 必须与训练 artifact 一致
& $python inference.py `
  --experiment-profile bgfbr_pc `
  --decoder-checkpoint .\results\bgfbr_pc\base\teacher_enhancer.pth `
  --memory-checkpoint .\results\bgfbr_pc\base\teacher_enhancer_memory.pth `
  --pred-root .\results\bgfbr_pc\predictions `
  --datasets CHAMELEON CAMO COD10K NC4K --batch-size 16 --num-workers 4 --amp

& $python evaluate.py `
  --pred-path .\results\bgfbr_pc\predictions `
  --datasets CHAMELEON CAMO COD10K NC4K --workers 4 --prefetch 8
```

每个数据集报告 MAE、S-measure、mean/max/adaptive E-measure、mean/max/adaptive F-measure 与 weighted F；
同时记录参数量、单图延迟、峰值显存、commit、profile、seed 和 split fingerprint。实验定义见
[`experiments/bgfbr_pc_hbm_manifest.json`](../experiments/bgfbr_pc_hbm_manifest.json)。

## 7. 来源与许可证

- BGFBR 的双前景/背景逐级细化设计参考 [DualUCOD](https://github.com/LPZliu/DualUCOD)。
- GPM、PAM 与 ODE 的模块结构参考 [FEDER](https://github.com/ChunmingHe/FEDER)。

上述两个上游仓库均以 MIT License 发布。本仓库实现按当前 tensor/API 合同重新组织；发布衍生代码时
应保留本说明、上游链接及相应 MIT copyright/license notice。
