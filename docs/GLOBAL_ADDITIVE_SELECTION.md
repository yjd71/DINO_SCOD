# Global-Additive 无 KMeans 离线选样

Global-Additive 是与 PC-BACS 并列的独立消融协议。它从外部给定的 40 张有标签 seed
开始，复用与该 seed 严格匹配的 5-epoch selector 及其全精度 score cache，对其余候选
进行纯全局排序，最终生成 `41 ⊂ 202 ⊂ 404 ⊂ 808` 四个有标签划分，分别对应
4040 张训练图上的 1%、5%、10% 和 20%。

该流程始终不运行 KMeans、也不使用簇配额。默认的 `none` 模式保持原始纯全局 Top-K；
可选的 `dino-cosine` 模式在同一全局排名上增加全局 DINO 余弦去重。两种模式均不会
修改或覆盖原有 PC-BACS 的方法、缓存和产物。候选图像的 GT、SAM 标签和伪标签均不
参与排序。

## 评分与选择

令原图与水平翻转还原后的 selector 概率图分别为 `P` 和 `P_T`。源 score cache 中保存
由修正版 Sobel 计算得到的全精度 `D_bd` 和 `D_all`：

```text
P_bar = (P + P_T) / 2
G     = hypot(Sobel_x(P_bar), Sobel_y(P_bar))
D_bd  = sum(G * abs(P - P_T)) / (sum(G) + eps)
D_all = mean(abs(P - P_T))
S     = D_bd + (1 - D_all)
```

`S` 保持在 `[0, 2]`，不会截断到 1；按 `(-S, sample_key)` 对所有非 seed 样本稳定排序。
加法公式等价于按 `D_bd - D_all` 排序，但 manifest 和 CSV 始终记录原始公式
`D_bd + (1 - D_all)`，便于审计。

### 选择模式

`--dedup-mode none` 是默认且向后兼容的无去重协议。每个目标划分完整继承 40 张 seed，
再截取同一全局排名的前缀：

```text
41  = 40 seed + global top 1
202 = 40 seed + global top 162
404 = 40 seed + global top 364
808 = 40 seed + global top 768
```

因此四个正式划分天然严格嵌套。

`--dedup-mode dino-cosine` 启用可选的全局去重，默认
`--dedup-threshold 0.95`。每轮仍按同一个 `(-S, sample_key)` 排名扫描候选，但每个候选
都会与**全部当前已选样本**比较 L2 归一化 DINO 特征，包括外部 40 张 seed、较小 split
继承的样本以及本轮刚选中的样本。仅当最大余弦相似度严格满足
`similarity > threshold` 时跳过；恰好等于 `0.95` 的候选允许进入。

若严格去重扫描无法凑满当前目标，则从截至当前累计跳过的全部候选中按原始全局排名
执行 relaxed backfill，忽略相似度限制直到精确达到目标数量。回填样本会成为后续轮次
的已选参考集，所以该模式同样保证 seed 全继承、数量精确以及
`41 ⊂ 202 ⊂ 404 ⊂ 808`。该策略仍是全局选择，不引入簇、簇内范围或簇配额。

## 输入与严格校验

两种模式共用此前外部 1% seed 流程的三项输入：

```text
Dataset/COD/splits/pc_bacs_split0.01_seed/bootstrap_0040_keys.pt
results/pc_bacs_selector_split0.01_0040/teacher_enhancer.pth
Dataset/COD/cache/pc_bacs_scores_split0.01.pt
```

- seed split 必须是 40 个唯一、合法的稳定 key，形如 `TR-CAMO/<stem>` 或
  `TR-COD10K/<stem>`。
- selector 必须是匹配该 seed fingerprint 的 5-epoch、two-stage、
  teacher-enhancer checkpoint，且 PC 已冻结。
- 源 cache 必须包含 4040 张图的 float32 原始 `D_bd`/`D_all`，并严格匹配 catalog、
  key 顺序、图像内容、预处理、DINO 权重、selector 和修正版 Sobel 评分版本。
- 任一 fingerprint、shape、范围或原乘法分数一致性检查失败都会硬失败；工具不会退回
  CSV 舍入值，也不会自动重新执行全池前向。

`dino-cosine` 模式还必须通过 `--dino-feature-cache` 提供 keyed DINO feature cache，
正式使用：

```text
Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt
```

该 cache 必须覆盖同一 4040 图 catalog，并严格匹配 key 顺序、图像 fingerprint、DINO
权重和 392×392 预处理 fingerprint；feature 必须为有限的二维浮点张量。失配时直接
失败，禁止退回按 stem 猜测、重新提取特征或放宽校验。`none` 模式不需要该参数。

## 运行

所有命令均从仓库根目录执行，并使用本机 `yjd` conda 环境。先执行只读校验：

```powershell
conda run -n yjd python select_global_additive.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/pc_bacs_split0.01_seed/bootstrap_0040_keys.pt `
  --selector-checkpoint ./results/pc_bacs_selector_split0.01_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_split0.01.pt `
  --target-counts 41 202 404 808 `
  --dedup-mode none `
  --output-dir ./Dataset/COD/splits/global_additive_split0.01_seed `
  --dry-run
```

`--dry-run` 只验证输入与契约，不写入派生 cache、split、CSV 或 manifest。正式生成时移除
`--dry-run`：

```powershell
conda run -n yjd python select_global_additive.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/pc_bacs_split0.01_seed/bootstrap_0040_keys.pt `
  --selector-checkpoint ./results/pc_bacs_selector_split0.01_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_split0.01.pt `
  --target-counts 41 202 404 808 `
  --dedup-mode none `
  --output-dir ./Dataset/COD/splits/global_additive_split0.01_seed
```

上述命令保留原始无去重协议、输出目录、cache 和文件名。启用全局 DINO 去重时使用
独立正式命令：

```powershell
conda run -n yjd python select_global_additive.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/pc_bacs_split0.01_seed/bootstrap_0040_keys.pt `
  --selector-checkpoint ./results/pc_bacs_selector_split0.01_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_split0.01.pt `
  --target-counts 41 202 404 808 `
  --dedup-mode dino-cosine `
  --dino-feature-cache ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --dedup-threshold 0.95 `
  --output-dir ./Dataset/COD/splits/global_additive_dedup_split0.01_seed
```

可在该命令末尾追加 `--dry-run`，只验证 score cache、feature cache、selector 与 seed
契约。两种模式都直接复用已有全精度缓存，不重新运行 4040 张图的 GPU 前向，也不执行
Base 或 Teacher–Student 正式训练。

## 产物与格式

无去重模式继续使用原有独立输出：

```text
Dataset/COD/splits/global_additive_split0.01_seed/
├── global_additive_0041_keys.pt
├── global_additive_0041_labeled_names.txt
├── global_additive_0202_keys.pt
├── global_additive_0202_labeled_names.txt
├── global_additive_0404_keys.pt
├── global_additive_0404_labeled_names.txt
├── global_additive_0808_keys.pt
├── global_additive_0808_labeled_names.txt
├── global_additive_scores.csv
└── global_additive_manifest.json

Dataset/COD/cache/
└── global_additive_scores_split0.01.pt
```

全局 DINO 去重模式使用另一套目录、派生 cache 和文件名前缀，禁止与无去重产物混用：

```text
Dataset/COD/splits/global_additive_dedup_split0.01_seed/
├── global_additive_dedup_0041_keys.pt
├── global_additive_dedup_0041_labeled_names.txt
├── global_additive_dedup_0202_keys.pt
├── global_additive_dedup_0202_labeled_names.txt
├── global_additive_dedup_0404_keys.pt
├── global_additive_dedup_0404_labeled_names.txt
├── global_additive_dedup_0808_keys.pt
├── global_additive_dedup_0808_labeled_names.txt
├── global_additive_dedup_scores.csv
└── global_additive_dedup_manifest.json

Dataset/COD/cache/
└── global_additive_dedup_scores_split0.01.pt
```

- 每个 `.pt` 保存按 stable key 排序的 `list[str]`。
- 对应 `.txt` 使用 UTF-8/LF，每行一个有标签图像的 stem，不含数据集前缀和扩展名；
  行序与对应 `.pt` 逐项转换后的顺序一致。
- 若跨数据集出现重复 stem，工具拒绝生成存在歧义的 TXT。
- CSV 记录 key、`D_bd`、`D_all`、加法分数、全局排名以及四个 split 的选择标志；去重
  版本还记录严格选择、相似度跳过和 relaxed backfill 决策及其参考 key。
- manifest 记录公式版本、`uses_kmeans=false`、去重 mode/阈值、严格 `>` 比较语义、
  每轮严格选择/跳过/回填统计、输入/输出 fingerprint、tie-break、目标数量、运行环境与
  源码提交。无去重 manifest 保持 `dedup=false` 的既有语义。

所有文件均采用原子写入。正式产物已存在且内容完全相同时允许幂等复用；同名文件内容
不同时拒绝覆盖。

## 验证

```powershell
conda run -n yjd python -m pytest -q `
  tests/test_global_additive_selection.py `
  tests/test_global_additive_cli.py

conda run -n yjd python -m pytest -q
```

生成后需对所选模式确认每对 PT/TXT 数量分别为 41、202、404、808，seed 无缺失，并
满足 `41 ⊂ 202 ⊂ 404 ⊂ 808`。去重模式还需核对严格通过、相似度跳过及 relaxed
backfill 计数与 CSV 决策一致。正式 Base 与 TS 训练应传入同一模式下的同一份目标
`.pt`，TXT 仅用于人工检查或兼容只接受图像名称的外部工具。
