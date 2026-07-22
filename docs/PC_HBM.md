# DINO PC-HBM 训练与评估

仓库只保留一个 canonical Decoder：`Model.decoder.Decoder`。它接收四层
`[B,784,768]` DINO patch tokens，并输出五个 `[B,1,98,98]` logits；其中
`outputs[3]` 是训练与正式推理使用的 `z_core`。

## Profile

| Profile | 原型库位置 | 用途 |
|---|---|---|
| `encoder_pc` / `default` | Decoder 前 | 默认完整 Encoder PC-HBM v3 |
| `encoder_pc_f4_f3` | Decoder 前 | 关闭 F2/F1 propagation |
| `encoder_pc_no_route_loss` | Decoder 前 | route InfoNCE 权重为 0 |
| `legacy_pc` | Decoder 内 | 保留的 decoder-side PC-HBM/schema v2 |
| `legacy_off` | 关闭 | 原始 Decoder 对照 |
| `parent_only` | Decoder 内 | 只运行 parent retrieval |

`EncoderPCHBMConfig.enabled=False` 是严格的无原型库 Base 对照：执行
Frozen DINOv2 → 原始 Decoder，不构造 Adapter、Refiner 或 memory，也不允许
正式推理传入 memory。

## Encoder PC-HBM v3 数据流

训练主链路为 Frozen DINOv2-B/14 → Encoder Adapter → 原始 Decoder。Adapter
输出四层同形 tokens；Decoder 内部 PC 永久关闭。memory 仅由 labeled 数据、Frozen
DINO 和 EMA Adapter 重建，不调用 Decoder。Teacher 可使用 Refiner 产生 pseudo label，
Student unlabeled 分支和正式推理始终跳过 Refiner并输出 `outputs[3]`。

Encoder memory 只接受 schema v3、CPU FP16、labeled-only。模型和 memory 的
producer/split fingerprint 必须匹配；旧格式不会静默降级。`legacy_pc` 使用独立的
decoder-side schema v2，两种 memory 不能互换。

## 从 Base 训练到评估

以下命令直接在仓库根目录运行，所有输出保存到
`./results/pc_hbm_bacs/pc_hbm_bacs_5`。

运行前请确认 `./data/cache/labeled_indices/pc_bacs_0202_keys.pt` 已准备好；
当前 checkout 未包含该文件。仓库中的其他 labeled-index 文件代表不同采样划分，
因此不会在命令中静默替换它。

Base（每个进程 batch 16）：

```bash
python -B -m torch.distributed.run --standalone --nproc_per_node=2 train_base_model_pc_hbm.py --experiment-profile encoder_pc --output-dir ./results/pc_hbm_bacs/pc_hbm_bacs_5/base --labeled-indices-pt ./data/cache/labeled_indices/pc_bacs_0202_keys.pt --batch-size 16 --epochs 30
```

Teacher/Student（labeled 与 unlabeled batch 均由入口固定为 32）：

```bash
python -B -m torch.distributed.run --standalone --nproc_per_node=2 train_ts_model_pseudo_pc_hbm.py --experiment-profile encoder_pc --teacher-pc-checkpoint ./results/pc_hbm_bacs/pc_hbm_bacs_5/base/encoder_pc_base_v3.pth --output-dir ./results/pc_hbm_bacs/pc_hbm_bacs_5/ts --labeled-indices-pt ./data/cache/labeled_indices/pc_bacs_0202_keys.pt --epochs 15
```

正式推理使用最终 Student v3 artifact 和与其匹配的 v3 memory：

```bash
python -B inference.py --experiment-profile encoder_pc --model-checkpoint ./results/pc_hbm_bacs/pc_hbm_bacs_5/ts/encoder_pc_ts_student_v3.pth --memory-checkpoint ./results/pc_hbm_bacs/pc_hbm_bacs_5/ts/encoder_pc_ts_memory_v3.pth --pred-root ./results/pc_hbm_bacs/pc_hbm_bacs_5/predictions --amp
```

评估四个 COD 数据集：

```bash
python -B evaluate.py --pred-path ./results/pc_hbm_bacs/pc_hbm_bacs_5/predictions --datasets CHAMELEON CAMO COD10K NC4K
```

结果表写入
`./results/pc_hbm_bacs/pc_hbm_bacs_5/predictions/evaluate_results.txt`，各数据集目录
同时保存 `evaluate.pkl`。

## 回归验收

```bash
python -B -m pytest -q
python -B -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_smoke_encoder_pc.py --cpu
python -B tests/cuda_smoke_encoder_pc.py
```

CPU DDP 命令若被当前 Windows PyTorch wheel 的 Gloo/libuv 构建能力阻塞，应记录
原始环境错误；不得通过跳过测试或降低 CUDA smoke batch size 规避。
