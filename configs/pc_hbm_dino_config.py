"""Single source of truth for the DINO PC-HBM experiment.

The defaults in this module intentionally mirror the implementation plan.  In
particular, memory is labelled-only, CPU resident and stored in FP16; the
original DINO/decoder spatial contract is not configurable from a training
entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DinoPCHBMConfig:
    """Configuration shared by the decoder, memory and both trainers."""

    enabled: bool = True

    # Fixed RSBL/DINO contract.
    input_size: int = 392
    encoder_dim: int = 768
    decoder_dim: int = 128
    token_size: int = 28
    output_size: int = 98
    dino_layer_indices: Tuple[int, int, int, int] = (2, 5, 8, 11)

    # PC-HBM dimensions.
    memory_dim: int = 128
    value_dim: int = 8
    geometry_dim: int = 6
    attn_num_heads: int = 8
    attn_head_dim: int = 16

    # Memory protocol.
    memory_source: str = "labeled_only"
    memory_rebuild_interval: int = 1
    use_unlabeled_memory_update: bool = False
    memory_storage_dtype: str = "float16"
    memory_device: str = "cpu"
    memory_gpu_cache: bool = False
    memory_format_version: int = 1
    memory_schema_version: int = 1
    memory_architecture: str = "DINO_SCOD_PC_HBM"
    exclude_self_match: bool = True

    route_top_img_k: int = 8
    parent_topk: int = 16
    query_chunk_size: int = 512

    # Boundary token selection.
    p3_top_ratio: float = 0.20
    p3_min_tokens: int = 32
    p3_max_tokens: int = 128
    p2_top_ratio: float = 0.20
    p2_min_tokens: int = 32
    p2_max_tokens: int = 128
    p1_top_ratio: float = 0.05
    p1_min_tokens: int = 96
    p1_max_tokens: int = 384

    # Label-region construction and deterministic sampling.
    fg_boundary_kernel: int = 3
    bg_near_kernel: int = 7
    gt_binary_threshold: float = 0.5
    sdf_reliability_scale: float = 0.15
    region_names: Tuple[str, str, str, str] = (
        "fg_core",
        "fg_boundary",
        "bg_near",
        "bg_far",
    )
    region_max_quota: Tuple[int, int, int, int] = (32, 64, 64, 32)
    region_min_quota: Tuple[int, int, int, int] = (4, 8, 8, 4)
    region_sampling_ratio: Tuple[float, float, float, float] = (
        0.20,
        0.50,
        0.50,
        0.20,
    )

    # Local attention and retrieval temperatures.
    child_window_size: int = 5
    p2_local_window: int = 3
    p1_local_window: int = 3
    tau_parent: float = 0.07
    tau_child: float = 0.10
    tau_hca: float = 0.10
    tau_bra: float = 0.10
    tau_pra: float = 0.10
    detach_p3_refs_for_p2: bool = True
    detach_p2_refs_for_p1: bool = True

    # Adaptive mixture (always at 98 x 98).
    r_max: float = 2.0
    max_offset: float = 1.5
    mask_corr_epsilon: float = 0.10
    mixture_init_bias: Tuple[float, float, float, float] = (
        1.0,
        -0.5,
        -0.5,
        -0.5,
    )
    mixture_eps_start: float = 0.10
    mixture_eps_end: float = 0.00
    mixture_temperature_start: float = 1.50
    mixture_temperature_end: float = 0.80
    mixture_schedule_start_epoch: int = 11
    mixture_schedule_end_epoch: int = 30
    ts_use_terminal_mixture_schedule: bool = True

    # Base-model stages.
    parent_start_epoch: int = 6
    full_pc_start_epoch: int = 11
    teacher_only_full_start_epoch: int = 6
    pc_injection_ramp_epochs: int = 3

    # Labeled loss.
    lambda_final: float = 1.0
    lambda_mem: float = 0.20
    lambda_boundary: float = 0.10
    lambda_mix_oracle: float = 0.10
    lambda_branch: float = 0.10
    lambda_quality: float = 0.025
    lambda_usage: float = 0.01
    lambda_reg: float = 0.02

    # Unlabeled branch.
    lambda_u: float = 1.0
    use_hard_pseudo: bool = False
    hard_loss_weight: float = 2.0
    pseudo_fg_threshold: float = 0.70
    pseudo_bg_threshold: float = 0.30
    pseudo_hard_ramp_epochs: int = 3
    feature_distill_p3_weight: float = 0.05
    feature_distill_p2_weight: float = 0.10

    # Optimization.
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    ema_momentum: float = 0.995

    # Diagnostics use a three-epoch persistence window unless overridden.
    diagnostic_window_epochs: int = 3
    warn_keep_collapse_threshold: float = 0.95
    warn_dead_branch_threshold: float = 0.01
    warn_gate_inactive_threshold: float = 0.03
    warn_child_auc_distance_from_half: float = 0.05
    warn_high_contradiction_threshold: float = 0.50
    warn_high_gate_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.memory_source != "labeled_only":
            raise ValueError("DINO PC-HBM memory must be labeled_only.")
        if self.use_unlabeled_memory_update:
            raise ValueError("Unlabeled pseudo-labels must never update PC-HBM memory.")
        if self.memory_device != "cpu" or self.memory_storage_dtype != "float16":
            raise ValueError("PC-HBM storage must be CPU float16.")
        if self.attn_num_heads * self.attn_head_dim != self.memory_dim:
            raise ValueError("attn_num_heads * attn_head_dim must equal memory_dim.")
        if len(self.region_names) != 4:
            raise ValueError("Exactly four mutually exclusive memory regions are required.")
        if not (
            len(self.region_max_quota)
            == len(self.region_min_quota)
            == len(self.region_sampling_ratio)
            == len(self.region_names)
        ):
            raise ValueError("Region sampling settings must have one value per region.")
        if self.parent_start_epoch < 1 or self.full_pc_start_epoch < self.parent_start_epoch:
            raise ValueError("Invalid parent/full PC epoch schedule.")
        if self.teacher_only_full_start_epoch < 2:
            raise ValueError("teacher_only_full_start_epoch must leave a parent-only warmup.")
        if self.feature_distill_p3_weight < 0 or self.feature_distill_p2_weight < 0:
            raise ValueError("Feature distillation weights must be non-negative.")
        if self.mixture_schedule_end_epoch < self.mixture_schedule_start_epoch:
            raise ValueError("Invalid mixture annealing interval.")

    def configure_training_design(self, training_design: str) -> None:
        """Apply the stage schedule for the selected trainer without changing schemas."""

        design = str(training_design)
        if design == "joint":
            return
        if design == "two_stage":
            # Complete Base preheating: legacy-only -> parent-only -> full PC-HBM.
            self.parent_start_epoch = 6
            self.full_pc_start_epoch = 11
            self.mixture_schedule_start_epoch = 11
            self.mixture_schedule_end_epoch = 30
            return
        if design != "teacher_only":
            raise ValueError(f"Unsupported PC-HBM training design: {design}")
        self.parent_start_epoch = 1
        self.full_pc_start_epoch = int(self.teacher_only_full_start_epoch)
        self.mixture_schedule_start_epoch = int(self.teacher_only_full_start_epoch)
        self.mixture_schedule_end_epoch = 30

    def pc_mode_for_epoch(self, epoch: int) -> str:
        """Return the 1-based Base-training mode for ``epoch``."""

        epoch = int(epoch)
        if epoch < self.parent_start_epoch:
            return "off"
        if epoch < self.full_pc_start_epoch:
            return "parent_only"
        return "full"

    def injection_scale(self, epoch: int) -> float:
        """Linear full-PC ramp: epochs 11/12/13 become 1/3, 2/3 and 1."""

        if int(epoch) < self.full_pc_start_epoch:
            return 0.0
        progress = int(epoch) - self.full_pc_start_epoch + 1
        return min(1.0, max(0.0, progress / max(1, self.pc_injection_ramp_epochs)))

    def mixture_schedule(self, epoch: int | None, *, ts_continuation: bool = False) -> tuple[float, float]:
        """Return ``(temperature, epsilon)`` for the relative mixture schedule."""

        if ts_continuation and self.ts_use_terminal_mixture_schedule:
            return self.mixture_temperature_end, self.mixture_eps_end
        current = self.mixture_schedule_start_epoch if epoch is None else int(epoch)
        span = max(1, self.mixture_schedule_end_epoch - self.mixture_schedule_start_epoch)
        alpha = min(1.0, max(0.0, (current - self.mixture_schedule_start_epoch) / span))
        temperature = self.mixture_temperature_start + alpha * (
            self.mixture_temperature_end - self.mixture_temperature_start
        )
        epsilon = self.mixture_eps_start + alpha * (
            self.mixture_eps_end - self.mixture_eps_start
        )
        return float(temperature), float(epsilon)

    def expected_memory_meta(self, *, producer_fingerprint: str | None = None) -> dict:
        """Build the stable compatibility contract stored beside memory tensors."""

        meta = {
            "architecture": self.memory_architecture,
            "schema_version": self.memory_schema_version,
            "input_size": self.input_size,
            "token_hw": (self.token_size, self.token_size),
            "output_hw": (self.output_size, self.output_size),
            "dino_layer_indices": tuple(self.dino_layer_indices),
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "memory_dim": self.memory_dim,
            "value_dim": self.value_dim,
            "geometry_dim": self.geometry_dim,
            "storage_dtype": self.memory_storage_dtype,
            "source": self.memory_source,
        }
        if producer_fingerprint is not None:
            meta["producer_fingerprint"] = str(producer_fingerprint)
        return meta


DEFAULT_PC_HBM_CONFIG = DinoPCHBMConfig()
