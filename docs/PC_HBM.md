# DINO_SCOD PC-HBM-Lite

PC-HBM-Lite is the only supported PC memory architecture in this repository:

```text
global/environment route
  -> fg_boundary and bg_near balanced Top-K
  -> aligned P2 cosine verification
  -> binary region evidence
  -> one gated P3 residual
  -> unchanged legacy Decoder
```

These are defaults, not equality locks: DINO input `392`, DINO layers
`(2, 5, 8, 11)`, `28x28` tokens, Decoder/Memory width `128`, and `98x98`
logits. Runtime hyperparameters are defined by `DinoPCHBMConfig`; changing
them changes the corresponding Router, query selector, region builder,
retriever, verifier, loss, schedule, or diagnostic component.

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
`float16`, `bfloat16`, and `float32` (default `float16`). The canonical
labeled-key file is:

```text
data/cache/labeled_indices/pc_bacs_0202_keys.pt
```

Every Base run, Teacher/Student run, memory artifact, and profiler run verifies
the labeled split fingerprint. Pseudo labels never enter memory.

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
        "region_max_quota": (48, 48),
        "region_min_quota": (8, 8),
        "region_sampling_ratio": (0.5, 0.5),
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

## Modes and training

Decoder modes are only:

- `off`: exact legacy path.
- `verify_only`: run retrieval and pair verification; P3 remains bitwise
  unchanged.
- `full`: apply the single gated P3 residual.
- `teacher_pseudo`: same correction as terminal `full`, with scale fixed to 1.

Base `two_stage` uses epochs 1-5 `off`, 6-10 `verify_only`, and 11 onward
`full`. Base `teacher_only` uses epochs 1-5 `verify_only` and 6 onward `full`.
The residual scale is `1/3`, `2/3`, then `1` over the first three full epochs.

Formal semi-supervised training is Teacher-only:

- Teacher: frozen PC enhancer, always `teacher_pseudo`.
- Student: raw Decoder, always `off`.
- Confidence:
  `2 * abs(p_final - 0.5) * (1 - Q + Q * C_mem)`.
- Targets: soft probability, confidence, and corrected P3 only.
- Student loss: labeled legacy structure loss, confidence-weighted main and
  side losses, plus P3 cosine distillation with weight `0.05`.
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
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tests\cuda_smoke_pc_hbm_lite.py `
  --seed 2025 `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt
& 'C:\Users\UserY\.conda\envs\yjd\python.exe' tools\profile_pc_hbm_lite.py `
  --seed 2025 `
  --labeled-indices-pt .\data\cache\labeled_indices\pc_bacs_0202_keys.pt
```

The smoke and profiler are intentionally single-GPU. No multi-process smoke is
part of the Lite acceptance protocol.
