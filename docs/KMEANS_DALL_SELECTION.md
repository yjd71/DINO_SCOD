# KMeans + 高/低 D_all + 簇内去重选样

## 方法

该离线协议一次生成高、低 `D_all` 两套 `41 ⊂ 202 ⊂ 404 ⊂ 808` 有标签
划分。两套结果共享同一40张 KMeans 中心 seed、逐轮簇配额和去重设置，唯一
变化是原始 float32 `global_disagreement` 的排序方向：

- 高 `D_all`：`(-global_disagreement, sample_key)`；
- 低 `D_all`：`(global_disagreement, sample_key)`。

低值模式直接比较原始 `D_all`，不会构造 `1-D_all`，避免相邻 float32 数值在
接近1的位置发生舍入合并。KMeans 固定使用40簇、`random_state=2025`、
`n_init=10`、`algorithm="lloyd"`；中心距离只用于确定每簇一张 seed。

每轮新增预算按剩余簇容量的平方根和最大余数法分配。候选与同簇全部已选样本
进行 DINO 余弦比较，只有 `similarity > 0.95` 才跳过；严格配额不足时依次执行
保持簇内去重和取消去重的方向性 `D_all` 全局排名回填。

源 score cache 中的 `D_bd` 与 `scores=D_bd*(1-D_all)` 仅用于验证缓存完整性，
不会进入排序、配额、去重或回填。该流程不执行 GPU 前向、训练或 GT 访问。

## 正式命令

```powershell
conda run -n yjd python select_kmeans_dall.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --dino-feature-cache ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --selector-checkpoint ./results/pc_bacs_selector_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_eab7755af67e_aca91fa264a1.pt `
  --directions high low `
  --target-counts 41 202 404 808 `
  --dedup-threshold 0.95 `
  --output-root ./Dataset/COD/splits/kmeans_dall_dedup
```

追加 `--dry-run` 会完成 catalog、缓存、selector、KMeans 和两种方向的选样校验，
但不会创建输出目录或文件。

## 产物

```text
Dataset/COD/splits/kmeans_dall_dedup/
├── high/
│   ├── kmeans_dall_high_dedup_0040_seed_keys.pt
│   ├── kmeans_dall_high_dedup_0040_seed_labeled_names.txt
│   ├── kmeans_dall_high_dedup_{0041,0202,0404,0808}_keys.pt
│   ├── kmeans_dall_high_dedup_{0041,0202,0404,0808}_labeled_names.txt
│   ├── kmeans_dall_high_dedup_assignments.csv
│   └── kmeans_dall_high_dedup_manifest.json
└── low/
    └── 对应的 kmeans_dall_low_dedup_* 全套文件
```

PT 保存排序后的稳定 `list[str]`。TXT 使用 UTF-8/LF，每行一个无数据集前缀、
无扩展名的图像 stem，顺序与 PT 一致；重复 stem 会导致整个发布失败。CSV 记录
cluster、中心距离、原始 `D_all`、方向性簇内排名、去重参照和回填状态。
Manifest 记录输入、算法、环境、逐轮统计和输出指纹。两套文件一起暂存并发布；
相同内容允许幂等复用，已有不同内容时拒绝覆盖。

## 当前正式缓存的预期结果

两套结果共享40-seed指纹：
`7a5064395bf40d0a1be3826c7166a49566c8d4cfe9c81c02b613699de9bb75bb`。

| 方向 | 41 | 202 | 404 | 808 |
|---|---|---|---|---|
| 高 D_all | `6d1dd3c8d5cdde9d2852fbdbfccd942e3ee125586f97a304bfdf3bfcc63d01d0` | `c76d067bd66c91ab80787a818e29c9ae2c1d3e7d6926ba5f477fed5808dfda1b` | `ade3457e497938eb72f35da557363abacd116985f32743fb9fe9f35c47e45281` | `046dab78eb054babbc2b224c46b5e8f0f9bbcb650b2b112289c06a8b01879e34` |
| 低 D_all | `393c39c8915cd35d65bc6906ba40ecc45754a11bf303ea4f674bfdb3e1db881f` | `34e83c8dd2a8976bb0fd5138b79b5a57ef1075cb67bffd44cae912934a35d663` | `d7592f2987f9415048d3afc2b05fbdf4165d8f3696919f5fc0675f364e0731d2` | `873118bd275ce6e1f2ac8dc5dd8f242a74d1f09564fd8335bf3ddd3807e534f3` |

高值首个新增样本为
`TR-COD10K/COD10K-CAM-2-Terrestrial-29-Dog-1845`，四轮去重跳过事件为
`0/0/2/4`。低值首个新增样本为
`TR-COD10K/COD10K-CAM-2-Terrestrial-29-Dog-1824`，去重跳过事件为
`0/1/3/10`。当前缓存下两套结果均无需回填。
