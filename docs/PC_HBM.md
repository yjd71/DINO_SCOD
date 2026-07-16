# RSBL DINO PC-HBM

当前主配置默认使用 `decoder_arch="bgfbr_pc_v1"`，冻结 DINOv2-B/14，并保持
`392 → 28×28 → 98×98` 空间合同。完整 profile、四数据集实验矩阵、报告模板和命令见
[DualUCOD BGFBR × PC-HBM 工程与实验手册](BGFBR_PC_HBM.md)。

历史 Transformer decoder 没有删除；只有显式选择
`--experiment-profile legacy_off` 或 `decoder_arch="legacy_transformer"` 时才启用。
旧 selector/BACS/BPUS 配置也显式锁定 legacy，不能消费 BGFBR memory schema v2。

## 模式与训练合同

- Base 默认 two-stage：epoch 1–5 `off`、6–10 `parent_only`、11–30 `full`；
  full correction 在 11/12/13 epoch 依次使用 1/3、2/3、1 强度。
- `parent_only` 只运行 P3 route/parent/B3，不注入 correction，预测仍为 `z_main`。
- `teacher_pseudo` 运行完整 P3/P2/P1/mixture，并导出 P3、P2 和五组 P1 蒸馏目标。
- `student_core` 保留 P1-PRA 与五组 P1 蒸馏，只跳过 mixture，因而
  `z_final/p_final=None`。
- teacher-only TS 的 Teacher 是 BGFBR+PC；raw Student 是相同 BGFBR、但构造时
  `attach_pc=False`。最终 `student_raw.pth` 不含任何 `pc_hbm.*`。
- joint TS 的 labeled Student 使用 `full`，unlabeled Student 使用 `student_core`。

训练态请求非 `off` 模式时，缺失、不完整或 schema 不兼容的 memory 会立即报错；
eval/inference 会发出 warning 并回退 `z_main`。有效 memory 没有合格邻居时，P3/P2/P1
及 mixture correction 保持零，基础预测不变。

## Memory 与 checkpoint

BGFBR memory 合同为：

```text
memory_architecture = DINO_SCOD_BGFBR_PC_HBM
memory_schema_version = 2
decoder_architecture = bgfbr_pc_v1
decoder_contract_version = 1
```

Memory 只能由完整 labeled split 构建，以 CPU FP16 保存；伪标签不能更新 memory。
schema v1、split/producer fingerprint 不一致，以及 GBE/GPM/ODE/RCAB/boundary-context
语义不一致都会被拒绝并要求重建。

checkpoint 文件格式仍为 v2，但会额外校验 decoder architecture/contract。普通加载禁止
跨架构静默恢复。legacy → BGFBR 必须使用显式迁移：

```python
load_legacy_into_bgfbr(
    decoder,
    checkpoint,
    reuse_projectors=True,
    reuse_pc_core=False,
)
```

CLI 中对应 `--legacy-warm-start`，仅用于 `two_stage + BGFBR`；同架构初始化使用
`--baseline-checkpoint`。默认迁移只复用四个 `768→128` projector。若程序化开启
`reuse_pc_core=True`，同形 PC 权重会复用，扩展输入前部复制、新通道清零；显式复用的
`pc_hbm.*` 参数使用 0.5× base LR，其余参数使用 base LR。

## Windows 单卡（yjd）

为规避 `conda run` 的 GBK 输出问题，直接使用解释器：

```powershell
$python = 'C:\Users\UserY\.conda\envs\yjd\python.exe'

& $python train_base_model_pc_hbm.py `
  --experiment-profile bgfbr_pc --training-design two_stage `
  --output-dir .\results\bgfbr_pc\base --batch-size 16 --epochs 30

& $python train_ts_model_pseudo_pc_hbm.py `
  --experiment-profile bgfbr_pc --training-design teacher_only `
  --teacher-pc-checkpoint .\results\bgfbr_pc\base\teacher_enhancer.pth `
  --output-dir .\results\bgfbr_pc\ts --epochs 15

& $python -m pytest -q
& $python tests\cuda_smoke_pc_hbm.py
```

CUDA smoke 固定覆盖 Base batch 16，以及 Teacher/raw/joint Student batch 32 的 AMP
forward/backward；不得通过降低 batch size 过关。

## Linux 双进程 DDP

```bash
conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ddp_smoke.py --cpu

conda run -n yjd python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_base_model_pc_hbm.py \
  --experiment-profile bgfbr_pc --training-design two_stage \
  --output-dir ./results/bgfbr_pc/base --batch-size 16 --epochs 30
```

`tests/ddp_smoke.py` 使用真实 BGFBR+PC 图，覆盖 Base 的 off/parent/full、raw TS 和
joint TS 的 full/student_core+P1 双 backward。部分 Windows PyTorch 构建没有可用
Gloo device；这种失败不判为模型失败，双 rank 验收在 Linux 执行。

## 推理

component profile 必须与训练 checkpoint 和 memory 一致：

```powershell
& $python inference.py `
  --experiment-profile bgfbr_pc `
  --decoder-checkpoint .\results\bgfbr_pc\base\teacher_enhancer.pth `
  --memory-checkpoint .\results\bgfbr_pc\base\teacher_enhancer_memory.pth `
  --pred-root .\results\bgfbr_pc\predictions `
  --datasets CHAMELEON CAMO COD10K NC4K --batch-size 16 --amp
```

teacher-only 的 `student_raw.pth` 只运行 `off`，不要为它附加 memory。

## 结构来源与许可证

- 四级前景/背景细化结构参考 [DualUCOD](https://github.com/LPZliu/DualUCOD)。
- GPM、PAM 与 ODE 模块结构参考 [FEDER](https://github.com/ChunmingHe/FEDER)。

两个上游仓库均采用 MIT License。本实现按本仓库 DINO/PC-HBM tensor 合同重新组织；
分发衍生代码时应保留上游链接及相应 MIT copyright/license notice。
