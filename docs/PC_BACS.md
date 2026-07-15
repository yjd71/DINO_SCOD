# PC-BACS 离线选样

PC-BACS（PC-HBM-oriented Boundary-Aware Coverage Sampling）是正式训练前运行的
离线选样工具。它不向模型增加可学习参数，也不修改 Decoder、PC-HBM、Base/TS
loss、trainer 或推理路径。选样完成后，正式 Base 与 Teacher–Student 训练只消费同一份
稳定 sample-key `.pt`；最终推理不依赖任何 PC-BACS 文件或额外前向。

本次流程只负责生成 `41 / 202 / 404` 三个嵌套划分。30 epoch Base 和 15 epoch
Teacher–Student 正式训练不属于本次运行范围。

## 方法与数据约束

完整流程为：

1. 对 `TR-CAMO + TR-COD10K` 的 RGB 图像提取冻结的 DINOv2-ViT-B/14 全局特征。
2. 使用固定的 40 簇 KMeans，每簇选择离中心最近的一张图，得到 40 张 seed。
3. 仅用这 40 张 seed 的 GT 训练 5 epoch 临时 selector；selector 始终使用
   `pc_mode="off"` 的 legacy Decoder。
4. 用 selector 比较原图与水平翻转图在 98×98 logits 上的预测，计算 PC-BACS 分数。
5. 在 DINO 簇约束下依次补齐 `41 ⊂ 202 ⊂ 404`，并输出稳定 sample key。

候选池 Dataset 只接受 RGB 路径，不接受也不读取 GT root、`SAMLabel`、SAM checkpoint
或伪标签。GT 只在 40 张 seed 已确定之后用于 selector 的监督训练，绝不参与候选打分、
聚类或排序。

KMeans 固定为 `n_clusters=40`、`random_state=2025`、`n_init=10` 和
`algorithm="lloyd"`。每轮新增预算按剩余簇容量的平方根和最大余数法分配；簇内按
`(-score, sample_key)` 稳定排序。默认在同簇内进行 DINO 余弦去重，阈值为 `0.98`，
并依次使用簇内继续扫描、全局保持去重回填、全局取消去重回填，确保目标数量精确满足。

### 修正后的边界分数

令原图与翻转还原后的概率图分别为 `P` 和 `P_T`。Sobel 在平均概率图上计算，边界
使用 replicate padding，梯度幅值使用 `torch.hypot(gx, gy)`：

```text
P_bar = (P + P_T) / 2
G     = hypot(Sobel_x(P_bar), Sobel_y(P_bar))
D_bd  = sum(G * abs(P - P_T)) / (sum(G) + eps)
D_all = mean(abs(P - P_T))
V     = D_bd * (1 - D_all)
```

`eps` 只出现在加权分母中，不加进梯度幅值。因此常数预测的 `G` 和 `D_bd` 严格为
零，replicate padding 也不会人为产生图像外框边界。manifest 会记录这一修正版 score
formula version。

## 准备

所有命令都从仓库根目录运行，并统一使用本机 `yjd` conda 环境。DINOv2 权重必须位于：

```text
weight/dinov2_vitb14_pretrain.pth
```

默认数据布局为：

```text
Dataset/COD/
├── TR-CAMO/im/
└── TR-COD10K/im/
```

`SelectionPoolDataset` 会将 RGB 图像固定缩放为 392×392，使用 bilinear/antialias 和
ImageNet normalize，不执行随机增强。

先做只读校验而不运行全池前向、也不写文件：

```powershell
conda run -n yjd python select_pc_bacs.py `
  --build-seed-only `
  --dry-run `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --device cuda
```

## 完整运行

### 1. 构建 DINO cache 和 40 张 seed

```powershell
conda run -n yjd python select_pc_bacs.py `
  --build-seed-only `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --features-path ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --n-clusters 40 `
  --feature-batch-size 16 `
  --num-workers 8 `
  --device cuda `
  --amp `
  --seed 2025 `
  --output-dir ./Dataset/COD/splits/pc_bacs
```

该命令生成 `kmeans_0040_seed_keys.pt`。文件内容是排序后的 `list[str]`，key 形如
`TR-CAMO/<stem>` 或 `TR-COD10K/<stem>`。

### 2. 训练 5 epoch selector

```powershell
conda run -n yjd python train_base_model_pc_hbm.py `
  --training-design two_stage `
  --labeled-indices-pt ./Dataset/COD/splits/pc_bacs/kmeans_0040_seed_keys.pt `
  --output-dir ./results/pc_bacs_selector_0040 `
  --batch-size 16 `
  --epochs 5 `
  --seed 2025 `
  --deterministic
```

这个 checkpoint 只用于离线打分。正式 Base 训练必须从统一初始化重新开始，不能接着
selector 的优化器或训练状态继续训练。

### 3. 全池打分并生成 41/202/404

```powershell
conda run -n yjd python select_pc_bacs.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/pc_bacs/kmeans_0040_seed_keys.pt `
  --selector-checkpoint ./results/pc_bacs_selector_0040/teacher_enhancer.pth `
  --features-path ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --target-counts 41 202 404 `
  --n-clusters 40 `
  --feature-batch-size 16 `
  --score-batch-size 16 `
  --num-workers 8 `
  --dedup-threshold 0.98 `
  --device cuda `
  --amp `
  --seed 2025 `
  --output-dir ./Dataset/COD/splits/pc_bacs
```

如真实图像触发 OOM，只依次将 `--score-batch-size` / `--feature-batch-size` 降为
`8`、再降为 `4`；不要改变图像尺寸、算法、簇数或目标样本数。实际 batch 会写入
manifest。

score cache 已存在时，默认直接拒绝继续运行，避免静默覆盖或误用旧分数。第二次运行
需要严格复用时，在完全相同的命令中追加 `--reuse-scores`；只有 catalog、图像内容、
key 顺序、DINO 权重、预处理、selector 非 PC 权重和评分版本的 fingerprint 全部一致时
才允许命中。确认需要重新打分并替换现有 cache 时，改用 `--rebuild-scores`，工具会在
新分数通过校验后原子替换；不要同时传入 `--reuse-scores` 和 `--rebuild-scores`。

feature cache 失配同样会硬失败；确认数据或权重确实改变后，显式使用
`--rebuild-features` 原子重建。`--scores-path` 可指定 score cache 路径，省略时由工具按
selector fingerprint 自动命名。

`--max-samples` 只用于专门的小规模 smoke，禁止用于正式 4040 图选样。关闭余弦去重可
显式设置 `--dedup-threshold -1`，但正式默认结果使用 `0.98`。

## 旧 split 转换

旧 txt/pt 能无歧义解析为稳定 key 时，可以只做格式转换：

```powershell
conda run -n yjd python select_pc_bacs.py `
  --convert-split-only `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/legacy_seed.txt `
  --output-dir ./Dataset/COD/splits/pc_bacs
```

转换结果按样本数使用规范名称保存。无法区分跨数据集同名 stem 时直接报错；没有 key
sidecar 的旧 `.npy` feature cache 也不得复用。

## 产物与下游契约

默认输出目录包含：

```text
Dataset/COD/splits/pc_bacs/
├── kmeans_0040_seed_keys.pt
├── pc_bacs_0041_keys.pt
├── pc_bacs_0202_keys.pt
├── pc_bacs_0404_keys.pt
├── pc_bacs_scores.csv
└── pc_bacs_manifest.json
```

- split `.pt` 只保存排序后的稳定 `list[str]`。
- CSV 记录 key、cluster、`D_bd`、`D_all`、PC-BACS score、选择轮次/顺序和去重状态。
- manifest 记录配置、环境版本、cache 校验与去重统计，以及 catalog、图像、DINO、
  selector、score 和 split fingerprint。
- 已有正式产物内容相同时可复用；内容不同时拒绝覆盖。

正式 Base 与 TS 必须传入同一个 split，例如都使用
`pc_bacs_0202_keys.pt` 的 `--labeled-indices-pt`，由现有 fingerprint 契约检查一致性。
本次交付不运行 30 epoch Base、15 epoch TS，也不改变其训练和推理命令。

## 验证

```powershell
conda run -n yjd python -m pytest -q `
  tests/test_pc_bacs_score.py `
  tests/test_pc_bacs_selection.py `
  tests/test_pc_bacs_dataset.py `
  tests/test_pc_bacs_checkpoint_cli.py

conda run -n yjd python tests/pc_bacs_smoke.py

conda run -n yjd python -m pytest -q
```

全量运行后，再以 `--reuse-scores` 严格复用一次，并确认三份 split、cluster id、CSV
选择标志以及 manifest fingerprint 完全一致。只有确实要重新执行 selector 前向并替换
score cache 时才使用 `--rebuild-scores`。
