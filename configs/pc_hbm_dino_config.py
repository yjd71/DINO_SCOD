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

    # Decoder identity.  This contract is independent of checkpoint format v2.
    experiment_profile: str = "bgfbr_pc"
    experiment_base_mode: str = "scheduled"
    decoder_arch: str = "bgfbr_pc_v1"
    decoder_contract_version: int = 1

    # BGFBR semantics that affect memory feature interpretation.
    gbe_version: str = "sobel_rgb_v1"
    gbe_normalization: str = "per_sample_max"
    gbe_padding_mode: str = "replicate"
    gpm_dilations: Tuple[int, int, int] = (1, 3, 5)
    gpm_contract: str = "five_branch_pam_v1"
    f4_adapter_contract: str = "128_32_128_zero_init_gamma_v1"
    ode_contract: str = "dual_path_scalar_alpha_v1"
    rcab_reduction: int = 16
    bgfbr_stage_count: int = 4
    fg_bg_contract: str = "independent_fg_bg_logits_v1"
    boundary_feature_channels: Tuple[int, int, int, int] = (7, 10, 10, 16)
    use_gbe: bool = True
    use_ode: bool = True
    use_rcab: bool = True
    use_pc_boundary_context: bool = True
    sync_bn: bool = False

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
    memory_schema_version: int = 2
    memory_architecture: str = "DINO_SCOD_BGFBR_PC_HBM"
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
    lambda_bgfbr_fg: float = 1.0
    lambda_bgfbr_bg: float = 1.0
    lambda_bgfbr_final: float = 1.0
    lambda_bgfbr_global: float = 1.0
    lambda_mem: float = 0.20
    lambda_boundary: float = 0.10
    lambda_mix_oracle: float = 0.10
    lambda_branch: float = 0.10
    lambda_quality: float = 0.025
    lambda_usage: float = 0.01
    lambda_reg: float = 0.02

    # Unlabeled branch.
    lambda_u: float = 1.0
    lambda_u_bg: float = 1.0
    use_hard_pseudo: bool = True
    hard_loss_weight: float = 2.0
    pseudo_fg_threshold: float = 0.70
    pseudo_bg_threshold: float = 0.30
    pseudo_hard_ramp_epochs: int = 3
    feature_distill_p3_weight: float = 0.05
    feature_distill_p2_weight: float = 0.10
    feature_distill_p1_weight: float = 0.05

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
        if self.decoder_arch not in {"bgfbr_pc_v1", "legacy_transformer"}:
            raise ValueError(f"Unsupported decoder architecture: {self.decoder_arch!r}")
        if self.decoder_contract_version != 1:
            raise ValueError("decoder_contract_version must be 1")
        if self.memory_schema_version != 2:
            raise ValueError("PC-HBM memory schema v1 is obsolete; rebuild memory with schema v2")
        if self.memory_architecture != "DINO_SCOD_BGFBR_PC_HBM":
            raise ValueError("memory_architecture must be DINO_SCOD_BGFBR_PC_HBM")
        if tuple(self.gpm_dilations) != (1, 3, 5):
            raise ValueError("BGFBR GPM dilations are fixed to (1, 3, 5)")
        if tuple(self.boundary_feature_channels) != (7, 10, 10, 16):
            raise ValueError("PC boundary channels are fixed to P3/P2/P1/mixture = 7/10/10/16")
        if self.rcab_reduction != 16 or self.bgfbr_stage_count != 4:
            raise ValueError("BGFBR requires four stages and RCAB reduction 16")
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
        if (
            self.feature_distill_p3_weight < 0
            or self.feature_distill_p2_weight < 0
            or self.feature_distill_p1_weight < 0
        ):
            raise ValueError("Feature distillation weights must be non-negative.")
        if self.hard_loss_weight < 0:
            raise ValueError("hard_loss_weight must be non-negative.")
        if min(
            self.lambda_bgfbr_fg,
            self.lambda_bgfbr_bg,
            self.lambda_bgfbr_final,
            self.lambda_bgfbr_global,
            self.lambda_u_bg,
        ) < 0:
            raise ValueError("BGFBR labeled/unlabeled loss weights must be non-negative.")
        if self.pseudo_hard_ramp_epochs < 1:
            raise ValueError("pseudo_hard_ramp_epochs must be at least one.")
        if not 0.0 <= self.pseudo_bg_threshold < 0.5 < self.pseudo_fg_threshold <= 1.0:
            raise ValueError("Pseudo thresholds must satisfy 0 <= bg < 0.5 < fg <= 1.")
        if self.mixture_schedule_end_epoch < self.mixture_schedule_start_epoch:
            raise ValueError("Invalid mixture annealing interval.")

    def configure_training_design(self, training_design: str) -> None:
        """Apply the stage schedule for the selected trainer without changing schemas."""

        design = str(training_design)
        if design == "joint":
            self.parent_start_epoch = 6
            self.full_pc_start_epoch = 11
            self.mixture_schedule_start_epoch = 11
            self.mixture_schedule_end_epoch = 30
        elif design == "two_stage":
            # Complete Base preheating: legacy-only -> parent-only -> full PC-HBM.
            self.parent_start_epoch = 6
            self.full_pc_start_epoch = 11
            self.mixture_schedule_start_epoch = 11
            self.mixture_schedule_end_epoch = 30
        elif design == "teacher_only":
            self.parent_start_epoch = 1
            self.full_pc_start_epoch = int(self.teacher_only_full_start_epoch)
            self.mixture_schedule_start_epoch = int(self.teacher_only_full_start_epoch)
            self.mixture_schedule_end_epoch = 30
        else:
            raise ValueError(f"Unsupported PC-HBM training design: {design}")

        # A selected experiment profile is authoritative even when trainers
        # re-apply the training-design lifecycle configuration.
        base_mode = str(getattr(self, "experiment_base_mode", "scheduled"))
        if base_mode == "off":
            self.parent_start_epoch = 1_000_000_000
            self.full_pc_start_epoch = 1_000_000_000
        elif base_mode == "parent_only":
            self.parent_start_epoch = 1
            self.full_pc_start_epoch = 1_000_000_000
        elif base_mode != "scheduled":
            raise ValueError(f"Unsupported experiment_base_mode: {base_mode!r}")

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
            "decoder_architecture": self.decoder_arch,
            "decoder_contract_version": self.decoder_contract_version,
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
            "gbe_version": self.gbe_version,
            "gbe_normalization": self.gbe_normalization,
            "gbe_padding_mode": self.gbe_padding_mode,
            "gpm_dilations": tuple(self.gpm_dilations),
            "gpm_contract": self.gpm_contract,
            "f4_adapter_contract": self.f4_adapter_contract,
            "ode_contract": self.ode_contract,
            "rcab_reduction": self.rcab_reduction,
            "bgfbr_stage_count": self.bgfbr_stage_count,
            "fg_bg_contract": self.fg_bg_contract,
            "boundary_feature_channels": tuple(self.boundary_feature_channels),
            "use_gbe": self.use_gbe,
            "use_ode": self.use_ode,
            "use_rcab": self.use_rcab,
            "use_pc_boundary_context": self.use_pc_boundary_context,
            "sync_bn": self.sync_bn,
        }
        if producer_fingerprint is not None:
            meta["producer_fingerprint"] = str(producer_fingerprint)
        return meta


DEFAULT_PC_HBM_CONFIG = DinoPCHBMConfig()


@dataclass
class EncoderPCHBMConfig:
    """Strict configuration for the encoder-side PC-HBM v3 profile."""

    enabled: bool = True
    experiment_profile: str = "encoder_pc"
    pc_placement: str = "encoder"
    decoder_arch: str = "bgfbr_pc_v1"
    architecture: str = "DINO_SCOD_ENCODER_PC_HBM"
    adapter_architecture: str = "encoder_pc_hbm_v1"
    feature_space: str = "frozen_dinov2_projected_encoder_v1"

    input_size: int = 392
    token_size: int = 28
    output_size: int = 98
    encoder_dim: int = 768
    memory_dim: int = 128
    value_dim: int = 8
    geometry_dim: int = 6
    dino_layer_indices: Tuple[int, int, int, int] = (2, 5, 8, 11)

    memory_format_version: int = 3
    memory_schema_version: int = 3
    memory_source: str = "labeled_only"
    memory_storage_dtype: str = "float16"
    memory_device: str = "cpu"
    use_unlabeled_memory_update: bool = False
    memory_rebuild_interval: int = 1

    boundary_token_ratio: float = 0.20
    boundary_min_tokens: int = 32
    boundary_max_tokens: int = 128
    route_top_img_k: int = 8
    parent_topk: int = 16
    query_chunk_size: int = 512
    tau_route: float = 0.07
    tau_parent: float = 0.07
    route_margin_temperature: float = 0.03
    route_confidence_floor: float = 0.20
    semantic_window_size: int = 5
    detail_window_size: int = 3
    propagation_window_size: int = 3
    attention_heads: int = 8

    bootstrap_end_epoch: int = 5
    parent_end_epoch: int = 10
    f4_f3_end_epoch: int = 15
    hierarchy_end_epoch: int = 20
    final_epoch: int = 30
    memory_adapter_ema_decay: float = 0.995

    max_f4_injection: float = 0.25
    max_f3_injection: float = 1.0
    max_f2_injection: float = 0.75
    max_f1_injection: float = 0.50
    injection_alpha_init: float = 1.0
    detach_f3_refs_for_f2: bool = True
    detach_f2_refs_for_f1: bool = True

    lambda_coarse: float = 0.30
    lambda_boundary: float = 0.10
    lambda_route: float = 0.05
    lambda_parent: float = 0.10
    lambda_child_semantic: float = 0.10
    lambda_child_detail: float = 0.05
    lambda_geometry: float = 0.05
    lambda_gate: float = 0.05
    lambda_injection: float = 0.01
    lambda_refiner_final: float = 1.0
    lambda_mix_oracle: float = 0.10
    lambda_branch: float = 0.10
    lambda_quality: float = 0.025
    lambda_usage: float = 0.01
    lambda_reg: float = 0.02

    pseudo_fg_threshold: float = 0.70
    pseudo_bg_threshold: float = 0.30
    pseudo_hard_ramp_epochs: int = 3
    hard_coverage_target: float = 0.20

    @property
    def lambda_refined_final(self) -> float:
        """Canonical refiner-final weight (legacy field kept in artifacts)."""

        return float(self.lambda_refiner_final)

    @property
    def lambda_refiner_reg(self) -> float:
        """Canonical refiner-regularization weight alias."""

        return float(self.lambda_reg)

    def __post_init__(self) -> None:
        fixed = {
            "pc_placement": (self.pc_placement, "encoder"),
            "decoder_arch": (self.decoder_arch, "bgfbr_pc_v1"),
            "architecture": (self.architecture, "DINO_SCOD_ENCODER_PC_HBM"),
            "input_size": (self.input_size, 392),
            "token_size": (self.token_size, 28),
            "output_size": (self.output_size, 98),
            "encoder_dim": (self.encoder_dim, 768),
            "memory_dim": (self.memory_dim, 128),
            "value_dim": (self.value_dim, 8),
            "geometry_dim": (self.geometry_dim, 6),
            "memory_format_version": (self.memory_format_version, 3),
            "memory_schema_version": (self.memory_schema_version, 3),
            "memory_source": (self.memory_source, "labeled_only"),
            "memory_storage_dtype": (self.memory_storage_dtype, "float16"),
            "memory_device": (self.memory_device, "cpu"),
        }
        invalid = [name for name, (actual, expected) in fixed.items() if actual != expected]
        if invalid:
            raise ValueError(f"Encoder PC-HBM fixed contract mismatch: {invalid}.")
        if tuple(self.dino_layer_indices) != (2, 5, 8, 11):
            raise ValueError("Encoder PC-HBM requires DINO layers (2, 5, 8, 11).")
        if self.use_unlabeled_memory_update:
            raise ValueError("Encoder PC-HBM memory is strictly labeled-only.")
        if not 0.0 < self.route_confidence_floor <= 1.0:
            raise ValueError("route_confidence_floor must be in (0, 1].")
        if self.attention_heads <= 0 or self.memory_dim % self.attention_heads:
            raise ValueError("attention_heads must divide memory_dim.")
        if self.attention_heads != 8:
            raise ValueError("Encoder PC-HBM uses exactly 8 attention heads.")
        if (
            self.semantic_window_size != 5
            or self.detail_window_size != 3
            or self.propagation_window_size != 3
        ):
            raise ValueError("Encoder verification/propagation windows are fixed to 5/3/3.")
        if not self.detach_f3_refs_for_f2 or not self.detach_f2_refs_for_f1:
            raise ValueError("Encoder hierarchy references must remain detached.")
        if not (
            0 < self.bootstrap_end_epoch < self.parent_end_epoch
            < self.f4_f3_end_epoch < self.hierarchy_end_epoch < self.final_epoch
        ):
            raise ValueError("Encoder PC-HBM stage boundaries must be strictly increasing.")

    def stage_for_epoch(self, epoch: int) -> str:
        if not 1 <= int(epoch) <= self.final_epoch:
            raise ValueError(f"epoch must be in [1, {self.final_epoch}], got {epoch}.")
        if epoch <= self.bootstrap_end_epoch:
            return "bootstrap"
        if epoch <= self.parent_end_epoch:
            return "parent_only"
        if epoch <= self.f4_f3_end_epoch:
            return "parent_child_f3"
        if epoch <= self.hierarchy_end_epoch:
            return "hierarchical_full"
        return "hierarchical_refiner"

    def stage_progress(self, epoch: int, *, level: str) -> float:
        if level == "f4_f3":
            start, end = self.parent_end_epoch + 1, self.f4_f3_end_epoch
        elif level == "f2_f1":
            start, end = self.f4_f3_end_epoch + 1, self.hierarchy_end_epoch
        else:
            raise ValueError("level must be 'f4_f3' or 'f2_f1'.")
        if epoch < start:
            return 0.0
        if epoch >= end:
            return 1.0
        return float(epoch - start + 1) / float(end - start + 1)

    def expected_memory_meta(
        self,
        *,
        producer_fingerprint: str,
        split_fingerprint: str,
    ) -> dict:
        if not producer_fingerprint or not split_fingerprint:
            raise ValueError("Memory compatibility requires producer and split fingerprints.")
        return {
            "format_version": self.memory_format_version,
            "schema_version": self.memory_schema_version,
            "architecture": self.architecture,
            "adapter_architecture": self.adapter_architecture,
            "feature_space": self.feature_space,
            "route_source": (
                "block11_cls_block11_global_block8_boundary_"
                "block8_uncertainty_block8_environment_v1"
            ),
            "input_size": self.input_size,
            "token_size": self.token_size,
            "dino_layer_indices": tuple(self.dino_layer_indices),
            "encoder_dim": self.encoder_dim,
            "memory_dim": self.memory_dim,
            "value_dim": self.value_dim,
            "geometry_dim": self.geometry_dim,
            "storage_dtype": self.memory_storage_dtype,
            "device": self.memory_device,
            "source": self.memory_source,
            "producer_fingerprint": str(producer_fingerprint),
            "split_fingerprint": str(split_fingerprint),
        }


DEFAULT_ENCODER_PC_HBM_CONFIG = EncoderPCHBMConfig()
