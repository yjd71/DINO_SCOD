# BGFBR × PC-HBM 实验报告

> 只填写实际运行结果；未完成项保留 `TBD`，不要用 smoke-test 数值代替正式实验。

## 运行身份

| 字段 | 值 |
|---|---|
| Git commit | TBD |
| Profile | TBD |
| Seed | 2025 |
| Labeled split path | TBD |
| Labeled split fingerprint | TBD |
| Base/TS training design | TBD |
| GPU / CUDA / PyTorch | TBD |
| 命令与日志路径 | TBD |

## 训练与资源

| Profile | Base 完成 epoch | TS 完成 epoch | 参数量 | 延迟 ms/image | 峰值显存 MiB | 状态/异常 |
|---|---:|---:|---:|---:|---:|---|
| legacy_off | TBD | N/A | TBD | TBD | TBD | TBD |
| bgfbr_off | TBD | N/A | TBD | TBD | TBD | TBD |
| parent_only | TBD | N/A | TBD | TBD | TBD | TBD |
| bgfbr_pc | TBD | TBD | TBD | TBD | TBD | TBD |
| no_gbe | TBD | TBD | TBD | TBD | TBD | TBD |
| no_ode | TBD | TBD | TBD | TBD | TBD | TBD |
| no_rcab | TBD | TBD | TBD | TBD | TBD | TBD |
| no_pc_boundary_context | TBD | TBD | TBD | TBD | TBD | TBD |

## 四数据集指标

| Profile / artifact | Dataset | Smeasure ↑ | meanE ↑ | adpE ↑ | maxE ↑ | meanF ↑ | adpF ↑ | maxF ↑ | wF ↑ | MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TBD | CHAMELEON | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | CAMO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | COD10K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | NC4K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 消融结论

- BGFBR 相对 legacy：TBD
- PC-HBM full 相对 BGFBR off / parent-only：TBD
- GBE：TBD
- ODE：TBD
- RCAB：TBD
- PC boundary context：TBD

## 完整性检查

- [ ] Base batch 16 与 TS batch 32 未降级
- [ ] Linux 2-rank DDP 通过
- [ ] CUDA AMP smoke 通过且 gradients finite
- [ ] profile、Teacher checkpoint、memory schema 与 split fingerprint 一致
- [ ] 四数据集预测数量与 GT 数量一致
- [ ] 日志中没有 inference memory fallback warning
- [ ] 指标来自 `evaluate.py` 生成的 `evaluate.pkl`
