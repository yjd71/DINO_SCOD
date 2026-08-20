# Global multiplicative-V selection

`select_global_multiplicative.py` is an offline ablation that ranks the full
training pool by

\[
V = D_{bd}(1-D_{all}).
\]

The selection stage does not fit or call KMeans, allocate cluster quotas, run
DINO or the selector, access GT, or apply deduplication. It deliberately reuses
the existing deterministic 40-image KMeans-center seed, so its manifest records
`seed_origin=kmeans_center_40` and
`uses_kmeans_during_selection=false` separately.

## Selection protocol

- The active catalog is the stable-key ordered 4040-image union of `TR-CAMO`
  and `TR-COD10K`.
- The seed must contain exactly the fixed 40 KMeans-center keys and must match
  the labeled-split fingerprint stored in the epoch-5 selector checkpoint.
- The source cache must provide aligned float32 `boundary_disagreement`,
  `global_disagreement`, and `scores` tensors. The CLI recomputes
  `D_bd * (1 - D_all)` and requires exact equality with `scores`.
- Non-seed candidates are sorted once by `(-V, sample_key)`. Each target is the
  complete seed plus the required prefix of that one global order.
- The resulting `41/202/404/808` splits are exact and strictly nested. There is
  no DINO cosine comparison or fallback path.

## Formal command

All Python execution uses the `yjd` environment. A dry run performs all CPU
identity checks and the complete selection trial without creating the output
directory:

```powershell
conda run -n yjd python select_global_multiplicative.py `
  --data-root ./Dataset/COD `
  --train-sets TR-CAMO TR-COD10K `
  --seed-split ./Dataset/COD/splits/pc_bacs/kmeans_0040_seed_keys.pt `
  --selector-checkpoint ./results/pc_bacs_selector_0040/teacher_enhancer.pth `
  --source-score-cache ./Dataset/COD/cache/pc_bacs_scores_eab7755af67e_aca91fa264a1.pt `
  --target-counts 41 202 404 808 `
  --output-dir ./Dataset/COD/splits/global_multiplicative_kmeans_seed `
  --dry-run
```

Remove `--dry-run` for formal publication. Existing identical artifacts are
accepted; any existing file with different content is refused.

## Outputs and audit

The output directory contains four
`global_multiplicative_{0041,0202,0404,0808}_keys.pt` files, their matching
`_labeled_names.txt` files, `global_multiplicative_scores.csv`, and
`global_multiplicative_manifest.json`. PT files contain sorted stable keys.
Each UTF-8/LF TXT contains the corresponding bare image stem in the same order;
duplicate stems across datasets are rejected.

The CSV records both raw disagreement components, full-precision-derived
ranking score, global candidate rank, seed status, and per-target selection
status/rank. The manifest records the formula and protocol versions, selector,
seed, source-cache, DINO-weight, preprocessing, catalog and output
fingerprints, environment versions, and explicitly states that selection-time
KMeans and deduplication are disabled. Split fingerprints use the ordered
stable-key JSON scheme recorded as `sha256_json_ordered_sample_keys_v1`; the
selector seed identity separately retains the repository's labeled-split
fingerprint scheme.

With the current formal inputs, the first added key is
`TR-COD10K/COD10K-CAM-2-Terrestrial-29-Dog-1845`; expected split fingerprints
are:

| Count | Fingerprint |
|---:|---|
| 41 | `00b41abefea81f2aa229eb4aa1ee1dcbe33c56f04a801234c0df4aedb5dd97e4` |
| 202 | `15390bdd4f5cb8f5765b65b51983618cf987c624cb08cc79a458de7babd6ea63` |
| 404 | `55ef3ac4d66bed6a98c05919761db689dbc1b12e8c039af9d3432a4b03bd377f` |
| 808 | `e9888da81ca8b69bfc0c4d9a3bdf45e2b9103d016f0002cc0e8514a1164fb9ca` |
