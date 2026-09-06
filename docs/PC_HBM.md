# DINO_SCOD PC-HBM-Lite

PC-HBM-Lite is the only supported PC memory architecture in this repository:

```text
global/environment route
  -> fg_boundary and bg_near balanced Top-K
  -> Parent-conditioned fixed P2 Child matching
  -> binary region evidence
  -> dominant foreground/background region context
  -> one projected P3 residual
  -> unchanged legacy Decoder
```

These are defaults, not equality locks: DINO input `392`, DINO layers
`(2, 5, 8, 11)`, `28x28` tokens, Decoder/Memory width `128`, and `98x98`
logits. Runtime hyperparameters are defined by `DinoPCHBMConfig`; changing
them changes the corresponding Router, query selector, region builder,
retriever, verifier, loss, schedule, or diagnostic component.

The current Child verifier contract is V3. Its default matcher is
`parent_conditioned`; the original `weighted_sum` implementation remains
available as a strict numerical baseline.

For example:

```python
pc_cfg = DinoPCHBMConfig(
    p3_top_ratio=0.20,
    p3_min_tokens=8,
    p3_max_tokens=128,
    route_top_img_k=6,
    route_global_weight=0.7,
    route_environment_weight=0.3,
    parent_topk_per_region=6,
    child_window_size=5,
    tau_parent=0.12,
    tau_child=0.18,
)
```

The formal experiment defaults keep `route_top_img_k=12`,
`parent_topk_per_region=64`, `child_window_size=3`, region maximum/minimum
quotas `(784, 784)/(0, 0)`, and region sampling ratios `(1.0, 1.0)`.

Validation now checks types, finite values, ranges, and relationships instead
of comparing tunable fields with their defaults. The remaining structural
relationships are:

- DINO stays frozen as ViT-B/14, so `encoder_dim=768`,
  `input_size=14*token_size`, and four unique layers must be selected from
  `[0, 11]`.
- `output_size=(14*token_size)//4` follows the unchanged Decoder scale.
- `memory_dim=decoder_dim` because the parameter-free Router has no channel
  projection.
- Evidence remains binary, so `region_names` contains exactly two names.
- Memory remains labeled-only, CPU-resident, Schema/Format V2, and architecture
  `DINO_SCOD_PC_HBM_LITE`. These are protocol identifiers rather than
  optimization hyperparameters.

## Memory V2

Memory is rebuilt from labeled samples at the start of every epoch. It is
detached, contiguous, and CPU-resident. `memory_storage_dtype` supports
`float16`, `bfloat16`, and `float32` (default `float16`). Formal Base and
Teacher/Student training accepts any non-empty labeled stable-key file with
unique normalized keys, including the generated 202-key and 404-key splits:

```text
data/cache/labeled_indices/pc_bacs_0202_keys.pt
data/cache/labeled_indices/pc_bacs_0404_keys.pt
```

Every Base run, Teacher/Student run, and memory rebuild verifies both the
run-specific labeled count and fingerprint. A Teacher checkpoint from one split
cannot be used with another split. The CUDA smoke and profiler intentionally
retain the fixed 202-key benchmark protocol. Pseudo labels never enter memory.

For formal training, `--labeled-indices-pt` is optional. When supplied, the PT
stable-key split overrides `Dataset/COD/sampled_images.txt`. When omitted, Base
training, TS labeled/unlabeled partitioning, and memory rebuild all use
`sampled_images.txt` and its fingerprint consistently.

The serialized state has this shape:

```python
{
    "format_version": 2,
    "schema_version": 2,
    "compat_meta": {
        "architecture": "DINO_SCOD_PC_HBM_LITE",
        "schema_version": 2,
        "input_size": 392,
        "token_hw": (28, 28),
        "output_hw": (98, 98),
        "dino_layer_indices": (2, 5, 8, 11),
        "encoder_dim": 768,
        "decoder_dim": 128,
        "memory_dim": 128,
        "child_window_size": 3,
        "region_names": ("fg_boundary", "bg_near"),
        "storage_dtype": "float16",
        "source": "labeled_only",
        "route_environment_min_mass": 0.001,
        "fg_boundary_kernel": 3,
        "bg_near_kernel": 7,
        "gt_binary_threshold": 0.5,
        "region_max_quota": (784, 784),
        "region_min_quota": (0, 0),
        "region_sampling_ratio": (1.0, 1.0),
        "producer_fingerprint": "...",
    },
    "memory_dim": 128,
    "storage_dtype": "float16",
    "route": {
        "global_keys": "... [N_img,128]",
        "environment_keys": "... [N_img,128]",
        "img_ids": ["..."],
    },
    "pairs": {
        "p3_keys": "... [N_pair,128]",
        "p2_keys": "... [N_pair,128]",
        "region_ids": "... [N_pair]",
        "pair_meta": [{"...": "..."}],
    },
    "finalized": True,
}
```

Any pre-V2 memory, resume, partial PC state, wrong architecture, or incomplete
metadata is rejected before live state is changed. To reuse only the baseline
Decoder tensors from an older checkpoint, export them explicitly:

```powershell
python tools/export_non_pc_decoder.py old.pth baseline_only.pth
```

Decoder and resume artifacts serialize the complete normalized
`DinoPCHBMConfig`. Training, TS, and inference reconstruct that configuration
before building the model, so non-default runtime values are not silently
replaced by source defaults.

## Child verifier V3

For each Parent-selected candidate, V3 computes

```text
c_abs = cos(q_child, k2)
c_rel = cos(q_child - W32 q3, k2 - W32 k3)
v = 0.5 * (c_abs + c_rel)
s_verified = s_parent + sigmoid(raw_eta) * v
```

`W32` is a learnable bias-free `128x128` projection initialized to the
identity. Parent `q3` and `k3` conditions are detached on the Child auxiliary
path. When either relation residual is too small, `c_rel` is exactly zero, so
the match is `0.5*c_abs`. When `v` is zero, the verified score is selected from
the original Parent score, preserving bitwise equality.

The matcher has 16,385 parameters (`128x128 + raw_eta`). The legacy
`weighted_sum` verifier has one parameter and retains its original score formula,
operation order, and masks. Both modes use the dominant-region context selection
described below. `child_match_logits` and
`child_match_strength` are the canonical V3 fields;
`child_verify_logits` and `verification_strength` remain deprecated aliases so
existing downstream consumers do not need to change. Existing
`pair_scores`, `pair_logits`, and `region_prob` continue to mean the final
verified result.

The matching objective is the binary region classification loss itself:

```text
L_match = L_reg
L_enh = L_seg + L_match
```

`L_reg` is cross-entropy on the verified binary region logits and therefore
trains the Child query path, `W32`, and `raw_eta` through the verified score.
There is no candidate BCE and no candidate-loss weight. `L_match` is added to
the segmentation objective directly; the former `lambda_pair` control has
been removed. Repair, harm, net-gain, margin-gain, and candidate AUROC remain
no-gradient diagnostics.

Decoder and resume checkpoints now carry `child_verifier_version=3` while the
Memory format/schema remains V2. Strict load, resume, TS teacher loading, and
inference reject V2 verifier checkpoints. Base Decoder initialization may
explicitly migrate a complete V2 parent-conditioned checkpoint:

```powershell
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' train_base_model_pc_hbm.py `
  --decoder-checkpoint .\checkpoints\decoder_v2.pth `
  --init-pcv-from-v2
```

This copies all compatible non-verifier tensors plus `W32` and `raw_eta`, and
drops the three removed learned scalars. It is an initialization path, not a
numerically equivalent conversion. `--init-pcv-from-legacy` remains available
for complete legacy `weighted_sum` Decoder checkpoints; the two migration
flags are mutually exclusive. Because `child_window_size` is compatibility
metadata, a Memory built with a different window (for example 7) must be
rebuilt even though its schema version is still 2.

New checkpoints omit `lambda_candidate_verify`, `lambda_pair`, and
`feature_distill_p3_weight`. Strict loading rejects checkpoints whose `pc_cfg`
still contains these obsolete objective controls rather than silently changing
their training semantics. V2 Decoder initialization drops them through the
explicit migration path above.

## Memory context selection and injection

For each boundary query, the final verified region probabilities decide whether
the retrieved evidence favors foreground (`fg_boundary`, index 0) or background
(`bg_near`, index 1). Candidate Parent keys are still softmax-weighted within
each region, but only the winning region's context is written back:

```text
alpha[q,r,:] = masked_softmax(s_verified[q,r,:])
context[q,r] = sum_j alpha[q,r,j] * parent_key[q,r,j]
p[q,:] = softmax_r(log_mean_exp_j(s_verified[q,r,j]))
r_star[q] = argmax_r p[q,r]
memory_context[q] = context[q,r_star[q]]
P3_corrected[q] = P3[q] + query_valid[q] * (W_out memory_context[q] + b_out)
```

The selected context is used at full strength: it is neither mixed with the
other region nor multiplied by the winning region probability. Exact ties
select foreground deterministically. Both regions must have at least one valid
candidate; otherwise the context and write-back are zero. Non-query P3 tokens
are unchanged. The output projection's weight and bias are zero-initialized.

The full region probabilities remain available for the matching objective and
memory confidence. Region selection is a hard `argmax`; the context path carries
gradients through the selected region's candidate weights and values, while
region classification remains supervised through the region logits. Injection
does not multiply the context by the diagnostic confidence gate or a ramp.

## Modes and training

Decoder modes are only:

- `off`: exact legacy path.
- `verify_only`: run retrieval and pair verification; P3 remains bitwise
  unchanged.
- `full`: apply the projected residual from the selected region context.
- `teacher_pseudo`: apply the same selected-context correction as `full`.

Base `two_stage` uses epochs 1-5 `off`, 6-10 `verify_only`, and 11 onward
`full`. Base `teacher_only` uses epochs 1-5 `verify_only` and 6 onward `full`.
There is no residual scale ramp when entering `full`.

Formal semi-supervised training is Teacher-only:

- Teacher: frozen PC enhancer, always `teacher_pseudo`.
- Student: raw Decoder, always `off`.
- Confidence:
  `2 * abs(p_final - 0.5) * (1 - Q + Q * C_mem)`.
- Targets: soft probability, confidence, and corrected P3 only.
- Student loss: labeled legacy structure loss, confidence-weighted main and
  side losses, plus P3 cosine distillation added directly with coefficient 1.
  The former `feature_distill_p3_weight` control has been removed.
- EMA updates only names shared by the raw Student and legacy Teacher Decoder.

## Commands

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'

& 'C:\Users\UserY\.conda\envs\yjd\python.exe' train_base_model_pc_hbm.py `
  --training-design two_stage `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt

& 'C:\Users\UserY\.conda\envs\yjd\python.exe' train_ts_model_pseudo_pc_hbm.py `
  --teacher-checkpoint .\results\base_pc_hbm_lite\teacher_enhancer.pth `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt

& 'C:\Users\UserY\.conda\envs\yjd\python.exe' -m pytest -q
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tests\ddp_smoke_child_verifier.py `
  --backend gloo --spawn-processes 2
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tests\cuda_smoke_pc_hbm_lite.py `
  --seed 2025 `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tools\evaluate_child_verification.py `
  --decoder-checkpoint .\checkpoints\decoder_v3.pth `
  --memory-checkpoint .\checkpoints\memory_v2.pth `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt `
  --output-json .\results\child_verification_eval.json
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tools\profile_pc_hbm_lite.py `
  --seed 2025 `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt
```

The full CUDA smoke and profiler are single-GPU. The verifier DDP smoke covers
both matching modes with `find_unused_parameters=False`; use `--backend nccl`
under `torchrun --nproc-per-node=2` for the real two-GPU variant.
