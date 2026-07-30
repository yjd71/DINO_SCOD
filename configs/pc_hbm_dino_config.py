"""Single source of truth for the DINO PC-HBM-Lite experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any


@dataclass
class DinoPCHBMConfig:
    """Configuration shared by the Lite decoder, memory, and trainers."""

    enabled: bool = True

    # Fixed DINO / legacy Decoder contract.
    input_size: int = 392
    encoder_dim: int = 768
    decoder_dim: int = 128
    token_size: int = 28
    output_size: int = 98
    dino_layer_indices: tuple[int, int, int, int] = (2, 5, 8, 11)

    # Labeled-only CPU-FP16 pair memory.
    memory_dim: int = 128
    memory_source: str = "labeled_only"
    use_unlabeled_memory_update: bool = False
    memory_storage_dtype: str = "float16"
    memory_device: str = "cpu"
    memory_format_version: int = 2
    memory_schema_version: int = 2
    memory_architecture: str = "DINO_SCOD_PC_HBM_LITE"
    exclude_self_match: bool = True

    # Parameter-free dual-context routing.
    route_top_img_k: int = 4
    route_global_weight: float = 0.5
    route_environment_weight: float = 0.5
    route_environment_min_mass: float = 1.0e-3

    # Parameter-free P3 query selection.
    p3_top_ratio: float = 0.10
    p3_min_tokens: int = 16
    p3_max_tokens: int = 64
    query_boundary_weight: float = 0.5
    query_uncertainty_weight: float = 0.5

    # Two-region pair memory.
    fg_boundary_kernel: int = 3
    bg_near_kernel: int = 7
    gt_binary_threshold: float = 0.5
    region_names: tuple[str, str] = ("fg_boundary", "bg_near")
    region_max_quota: tuple[int, int] = (48, 48)
    region_min_quota: tuple[int, int] = (8, 8)
    region_sampling_ratio: tuple[float, float] = (0.50, 0.50)

    # Balanced retrieval and local Child verification.
    parent_topk_per_region: int = 4
    query_chunk_size: int = 512
    child_window_size: int = 3
    tau_parent: float = 0.07
    tau_child: float = 0.10
    child_mix_init_logit: float = 0.0

    # Stage schedule.
    verify_start_epoch: int = 6
    full_pc_start_epoch: int = 11
    teacher_only_full_start_epoch: int = 6
    pc_injection_ramp_epochs: int = 3

    # Labeled and unlabeled objectives.
    lambda_pair: float = 0.20
    lambda_u: float = 1.0
    feature_distill_p3_weight: float = 0.05

    # Optimization.
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    ema_momentum: float = 0.995

    # Lite diagnostics.
    diagnostic_window_epochs: int = 3
    warn_low_pair_valid_ratio: float = 0.05
    warn_pair_acc_near_random: float = 0.05
    warn_gate_inactive_threshold: float = 0.02
    warn_delta_large_threshold: float = 1.0

    def __post_init__(self) -> None:
        fixed: tuple[tuple[str, Any, Any], ...] = (
            ("input_size", self.input_size, 392),
            ("encoder_dim", self.encoder_dim, 768),
            ("decoder_dim", self.decoder_dim, 128),
            ("token_size", self.token_size, 28),
            ("output_size", self.output_size, 98),
            ("dino_layer_indices", tuple(self.dino_layer_indices), (2, 5, 8, 11)),
            ("memory_dim", self.memory_dim, 128),
            ("memory_source", self.memory_source, "labeled_only"),
            ("use_unlabeled_memory_update", self.use_unlabeled_memory_update, False),
            ("memory_storage_dtype", self.memory_storage_dtype, "float16"),
            ("memory_device", self.memory_device, "cpu"),
            ("memory_format_version", self.memory_format_version, 2),
            ("memory_schema_version", self.memory_schema_version, 2),
            ("memory_architecture", self.memory_architecture, "DINO_SCOD_PC_HBM_LITE"),
            ("route_top_img_k", self.route_top_img_k, 4),
            ("route_global_weight", self.route_global_weight, 0.5),
            ("route_environment_weight", self.route_environment_weight, 0.5),
            ("p3_top_ratio", self.p3_top_ratio, 0.10),
            ("p3_min_tokens", self.p3_min_tokens, 16),
            ("p3_max_tokens", self.p3_max_tokens, 64),
            ("query_boundary_weight", self.query_boundary_weight, 0.5),
            ("query_uncertainty_weight", self.query_uncertainty_weight, 0.5),
            ("fg_boundary_kernel", self.fg_boundary_kernel, 3),
            ("bg_near_kernel", self.bg_near_kernel, 7),
            ("gt_binary_threshold", self.gt_binary_threshold, 0.5),
            ("region_names", tuple(self.region_names), ("fg_boundary", "bg_near")),
            ("region_max_quota", tuple(self.region_max_quota), (48, 48)),
            ("region_min_quota", tuple(self.region_min_quota), (8, 8)),
            ("region_sampling_ratio", tuple(self.region_sampling_ratio), (0.5, 0.5)),
            ("parent_topk_per_region", self.parent_topk_per_region, 4),
            ("child_window_size", self.child_window_size, 3),
            ("child_mix_init_logit", self.child_mix_init_logit, 0.0),
            ("verify_start_epoch", self.verify_start_epoch, 6),
            ("full_pc_start_epoch", self.full_pc_start_epoch, 11),
            (
                "teacher_only_full_start_epoch",
                self.teacher_only_full_start_epoch,
                6,
            ),
            ("pc_injection_ramp_epochs", self.pc_injection_ramp_epochs, 3),
        )
        for name, actual, expected in fixed:
            if actual != expected:
                raise ValueError(f"{name} is fixed to {expected!r}, got {actual!r}")

        if (
            not math.isfinite(float(self.route_environment_min_mass))
            or self.route_environment_min_mass <= 0
        ):
            raise ValueError("route_environment_min_mass must be positive")
        if (
            not isinstance(self.query_chunk_size, Integral)
            or isinstance(self.query_chunk_size, bool)
            or self.query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be positive")
        if (
            not math.isfinite(float(self.tau_parent))
            or not math.isfinite(float(self.tau_child))
            or self.tau_parent <= 0
            or self.tau_child <= 0
        ):
            raise ValueError("cosine temperatures must be positive")
        if self.verify_start_epoch < 1:
            raise ValueError("verify_start_epoch must be at least one")
        if self.full_pc_start_epoch < self.verify_start_epoch:
            raise ValueError("full_pc_start_epoch must not precede verification")
        if self.teacher_only_full_start_epoch < 2:
            raise ValueError("teacher_only_full_start_epoch must leave a verification warmup")
        loss_weights = (
            float(self.lambda_pair),
            float(self.lambda_u),
            float(self.feature_distill_p3_weight),
        )
        if (
            not all(math.isfinite(value) for value in loss_weights)
            or any(value < 0 for value in loss_weights)
        ):
            raise ValueError("loss weights must be non-negative")
        if (
            not math.isfinite(float(self.grad_clip_norm))
            or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0,1)")
        if (
            not isinstance(self.diagnostic_window_epochs, Integral)
            or isinstance(self.diagnostic_window_epochs, bool)
            or self.diagnostic_window_epochs < 1
        ):
            raise ValueError("diagnostic_window_epochs must be positive")
        for name in (
            "warn_low_pair_valid_ratio",
            "warn_pair_acc_near_random",
            "warn_gate_inactive_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            not math.isfinite(float(self.warn_delta_large_threshold))
            or self.warn_delta_large_threshold <= 0
        ):
            raise ValueError("warn_delta_large_threshold must be positive")

    def configure_training_design(self, training_design: str) -> None:
        """Configure one of the two supported stage schedules."""

        design = str(training_design)
        if design == "two_stage":
            self.verify_start_epoch = 6
            self.full_pc_start_epoch = 11
            return
        if design == "teacher_only":
            self.verify_start_epoch = 1
            self.full_pc_start_epoch = int(self.teacher_only_full_start_epoch)
            return
        raise ValueError(f"Unsupported PC-HBM-Lite training design: {design}")

    def pc_mode_for_epoch(self, epoch: int) -> str:
        """Return the Lite mode for a one-based training epoch."""

        current = int(epoch)
        if current < self.verify_start_epoch:
            return "off"
        if current < self.full_pc_start_epoch:
            return "verify_only"
        return "full"

    def injection_scale(self, epoch: int) -> float:
        """Ramp the only P3 residual over the first three full-mode epochs."""

        current = int(epoch)
        if current < self.full_pc_start_epoch:
            return 0.0
        progress = current - self.full_pc_start_epoch + 1
        return min(1.0, max(0.0, progress / self.pc_injection_ramp_epochs))

    def expected_memory_meta(
        self,
        *,
        producer_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Return the complete stable compatibility contract for Schema V2."""

        meta: dict[str, Any] = {
            "architecture": self.memory_architecture,
            "schema_version": self.memory_schema_version,
            "input_size": self.input_size,
            "token_hw": (self.token_size, self.token_size),
            "output_hw": (self.output_size, self.output_size),
            "dino_layer_indices": tuple(self.dino_layer_indices),
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "memory_dim": self.memory_dim,
            "child_window_size": self.child_window_size,
            "region_names": tuple(self.region_names),
            "storage_dtype": self.memory_storage_dtype,
            "source": self.memory_source,
        }
        if producer_fingerprint is not None:
            meta["producer_fingerprint"] = str(producer_fingerprint)
        return meta


DEFAULT_PC_HBM_CONFIG = DinoPCHBMConfig()


__all__ = ["DEFAULT_PC_HBM_CONFIG", "DinoPCHBMConfig"]
