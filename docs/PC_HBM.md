# DINO_SCOD PC-HBM-Lite

PC-HBM-Lite is the only supported PC memory architecture in this repository:

```text
global/environment route
  -> fg_boundary and bg_near Top-K (four from each region)
  -> aligned P2 cosine verification
  -> binary region evidence
  -> one gated P3 residual
  -> unchanged legacy Decoder
```

The fixed contract is DINO input `392`, DINO layers `(2, 5, 8, 11)`,
`28x28` tokens, Decoder width `128`, and `98x98` logits. DINO remains frozen.

## Memory V2

Memory is rebuilt from labeled samples at the start of every epoch. It is
detached, contiguous, CPU-resident, and FP16. The canonical labeled-key file
is:

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
