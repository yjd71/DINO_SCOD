# KMeans-only 有标签采样

## 方法

该消融流程只使用冻结 DINOv2-ViT-B/14 全局特征上的 KMeans 聚类结果，不加载
selector checkpoint、score cache、GT、`D_bd`、`D_all` 或 PC-BACS 分数，也不执行
余弦去重、模型前向或训练。

固定协议如下：

1. 严格复用 keyed `4040×768` float32 DINO cache，并验证 catalog、逐图内容、
   sample-key 顺序、DINO 权重和 392×392 预处理指纹。
2. 特征 L2 归一化后执行 `n_clusters=40`、`random_state=2025`、`n_init=10`、
   `algorithm="lloyd"` 的 KMeans。
3. 每簇离中心最近的一张组成 40 张 seed；距离相同时按 sample key。
4. 依次补齐 `41 ⊂ 202 ⊂ 404 ⊂ 808`。每轮新增预算按剩余簇容量的平方根和
   最大余数法分配，余数相同时按 cluster id；簇内只按
   `(center_distance, sample_key)` 选择，中心不重新拟合。

## 正式命令

```powershell
conda run -n yjd python select_kmeans_only.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --features-path ./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt `
  --n-clusters 40 `
  --seed 2025 `
  --target-counts 41 202 404 808 `
  --output-dir ./Dataset/COD/splits/kmeans_only
```

追加 `--dry-run` 时执行完整缓存校验、KMeans 和划分模拟，但不创建输出目录或文件。

## 产物

```text
Dataset/COD/splits/kmeans_only/
├── kmeans_only_0040_seed_keys.pt
├── kmeans_only_0040_labeled_names.txt
├── kmeans_only_0041_keys.pt
├── kmeans_only_0041_labeled_names.txt
├── kmeans_only_0202_keys.pt
├── kmeans_only_0202_labeled_names.txt
├── kmeans_only_0404_keys.pt
├── kmeans_only_0404_labeled_names.txt
├── kmeans_only_0808_keys.pt
├── kmeans_only_0808_labeled_names.txt
├── kmeans_only_assignments.csv
└── kmeans_only_manifest.json
```

PT 保存排序后的稳定 `list[str]`；TXT 使用 UTF-8/LF，每行一个无数据集前缀、无扩展名
的图像 stem，顺序与对应 PT 一致。CSV 记录 cluster id、cluster size、中心距离、簇内
中心距离排名、seed 标志和各档选择状态。manifest 记录完整 KMeans 参数、逐轮配额、
输入及输出指纹和运行环境。相同内容允许幂等复用，不同内容拒绝覆盖。

当前正式缓存的预期 split fingerprint 为：

- 40 seed：`7a5064395bf40d0a1be3826c7166a49566c8d4cfe9c81c02b613699de9bb75bb`
- 41：`f6a7ea43eb2d249f7950e40c989b8bcb2bd5e86b283da673bd36173ba8bac6f6`
- 202：`a2f918fe17d43288eda8818672e4c59a39795b5933d04c4498a3395c009eddb6`
- 404：`6b9d194c7371fc357e5e657cb6c1dafbf41ca29dbfed3618dd57194bf2449181`
- 808：`b37286b75052e597b4454d21ebfe21c6022c5a7ab04399104ed83e64e7f6a1d8`

41 档首个新增样本应为
`TR-COD10K/COD10K-CAM-2-Terrestrial-26-Chameleon-1672`。
