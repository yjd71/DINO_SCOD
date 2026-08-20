# KMeans + D_bd + 簇内去重选样

## 方法

该离线消融协议生成 `41 ⊂ 202 ⊂ 404 ⊂ 808` 四档有标签划分，对应约
1%、5%、10%、20%。它不会重新执行模型前向、训练或访问 GT。

1. 严格复用 `4040×768` float32 keyed DINO 特征缓存，L2 归一化后执行
   `n_clusters=40`、`random_state=2025`、`n_init=10`、
   `algorithm="lloyd"` 的 KMeans。
2. 每簇以 `(center_distance, sample_key)` 选出一张中心 seed，共40张。
   selector checkpoint 必须是以这40张 seed 训练完成的 epoch-5、two-stage、
   teacher-enhancer、PC-frozen 产物。
3. 每轮新增预算按剩余簇容量的平方根和最大余数法分配。seed 之后，簇内候选
   只按 `(-D_bd, sample_key)` 排序，中心距离不再参与排名。
4. 候选与同簇全部已选样本比较 DINO 余弦相似度，包括 seed、较小划分和本轮
   新增样本。只有 `similarity > 0.95` 才跳过，等于阈值时允许选择。
5. 若严格簇配额无法填满，则依次执行全局 D_bd 排名且保持簇内去重的回填，
   以及取消去重的最终回填，保证每档数量精确。

源 score cache 中的 `D_all` 和 `scores=D_bd*(1-D_all)` 仅用于校验缓存没有损坏；
纯算法接口只接收 `boundary_disagreement`，二者不会进入排序、配额、去重或回填。

## 正式命令

```powershell
conda run -n yjd python select_kmeans_dbd.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --dino-feature-cache ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --selector-checkpoint ./results/pc_bacs_selector_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_eab7755af67e_aca91fa264a1.pt `
  --target-counts 41 202 404 808 `
  --dedup-threshold 0.95 `
  --output-dir ./Dataset/COD/splits/kmeans_dbd_dedup
```

追加 `--dry-run` 会完成 catalog、缓存、selector、KMeans 和选样校验，但不会
创建输出目录或文件。

## 产物

```text
Dataset/COD/splits/kmeans_dbd_dedup/
├── kmeans_dbd_dedup_0040_seed_keys.pt
├── kmeans_dbd_dedup_0040_seed_labeled_names.txt
├── kmeans_dbd_dedup_0041_keys.pt
├── kmeans_dbd_dedup_0041_labeled_names.txt
├── kmeans_dbd_dedup_0202_keys.pt
├── kmeans_dbd_dedup_0202_labeled_names.txt
├── kmeans_dbd_dedup_0404_keys.pt
├── kmeans_dbd_dedup_0404_labeled_names.txt
├── kmeans_dbd_dedup_0808_keys.pt
├── kmeans_dbd_dedup_0808_labeled_names.txt
├── kmeans_dbd_dedup_assignments.csv
└── kmeans_dbd_dedup_manifest.json
```

PT 保存排序后的稳定 `list[str]`。TXT 使用 UTF-8/LF，每行一个无数据集前缀、
无扩展名的图像 stem，顺序与 PT 一致；跨数据集出现重复 stem 时拒绝发布。
CSV 记录 cluster、中心距离、D_bd、选择档位、去重参照和回填状态。Manifest
记录输入、算法、环境、逐轮统计和全部输出指纹。相同内容允许幂等复用，已有
不同内容时拒绝覆盖。

## 当前正式缓存的预期结果

- 40 seed：`7a5064395bf40d0a1be3826c7166a49566c8d4cfe9c81c02b613699de9bb75bb`
- 41：`6d1dd3c8d5cdde9d2852fbdbfccd942e3ee125586f97a304bfdf3bfcc63d01d0`
- 202：`dd943b0e7824e22fff8328e65d642ccf35f60af6090582c67c9ec19783a79dbd`
- 404：`6cdb7c7f71ed79711f6c352c996d1a6eabf2474a2b26ae93ec0021b314c8b500`
- 808：`40ba922b1d3dd2bed36b57cf97db8923dd6e33198726f0937985245a6c2e099a`

首个新增样本为
`TR-COD10K/COD10K-CAM-2-Terrestrial-29-Dog-1845`。四轮去重跳过事件预期为
`0/2/2/4`，保持去重和取消去重的回填数均为0。
