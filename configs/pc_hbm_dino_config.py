"""Single source of truth for the DINO PC-HBM-Lite experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _as_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _as_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
    maximum_open: bool = False,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = (
            normalized <= minimum if minimum_open else normalized < minimum
        )
        if invalid:
            bracket = ">" if minimum_open else ">="
            raise ValueError(f"{name} must be {bracket} {minimum}")
    if maximum is not None:
        invalid = (
            normalized >= maximum if maximum_open else normalized > maximum
        )
        if invalid:
            bracket = "<" if maximum_open else "<="
            raise ValueError(f"{name} must be {bracket} {maximum}")
    return normalized


def _as_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _as_tuple(name: str, value: Any, *, length: int) -> tuple[Any, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(value)


@dataclass
class DinoPCHBMConfig:
    """Configuration shared by the Lite decoder, memory, and trainers."""

    enabled: bool = True

    # DINO / legacy Decoder shape contract.
    input_size: int = 392
    encoder_dim: int = 768
    decoder_dim: int = 128
    token_size: int = 28
    output_size: int = 98
    dino_layer_indices: tuple[int, int, int, int] = (2, 5, 8, 11)

    # Labeled-only CPU pair memory with configurable floating storage.
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
    route_top_img_k: int = 12
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
    region_max_quota: tuple[int, int] = (784, 784)
    region_min_quota: tuple[int, int] = (0, 0)
    region_sampling_ratio: tuple[float, float] = (1.0, 1.0)

    # Balanced retrieval and local Child verification.
    parent_topk_per_region: int = 64
    query_chunk_size: int = 512
    child_window_size: int = 3
    tau_parent: float = 0.07
    tau_child: float = 0.10
    child_mix_init_logit: float = 0.0
    child_verification_mode: str = "parent_conditioned"
    verification_strength_init: float = 0.25
    verification_logit_clip: float = 6.0
    relation_norm_eps: float = 1.0e-4

    # Direct fixed-Child matching objective.
    lambda_candidate_verify: float = 0.50

    # Stage schedule.
    verify_start_epoch: int = 6
    full_pc_start_epoch: int = 11
    teacher_only_full_start_epoch: int = 6
    pc_injection_ramp_epochs: int = 3

    # Labeled and unlabeled objectives.
    lambda_pair: float = 1.0
    lambda_u: float = 1.0
    feature_distill_p3_weight: float = 1.0

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
        self.child_verification_mode = _as_nonempty_str(
            "child_verification_mode", self.child_verification_mode
        ).lower()
        if self.child_verification_mode not in {
            "weighted_sum",
            "parent_conditioned",
        }:
            raise ValueError(
                "child_verification_mode must be 'weighted_sum' or "
                "'parent_conditioned'"
            )
        for name in ("verification_logit_clip", "relation_norm_eps"):
            setattr(
                self,
                name,
                _as_float(
                    name,
                    getattr(self, name),
                    minimum=0.0,
                    minimum_open=True,
                ),
            )
        self.verification_strength_init = _as_float(
            "verification_strength_init",
            self.verification_strength_init,
            minimum=0.0,
            maximum=1.0,
            minimum_open=True,
            maximum_open=True,
        )
        self.lambda_candidate_verify = _as_float(
            "lambda_candidate_verify",
            self.lambda_candidate_verify,
            minimum=0.0,
        )

        for name in (
            "enabled",
            "use_unlabeled_memory_update",
            "exclude_self_match",
            "use_amp",
        ):
            setattr(self, name, _as_bool(name, getattr(self, name)))

        for name in (
            "input_size",
            "encoder_dim",
            "decoder_dim",
            "token_size",
            "output_size",
            "memory_dim",
            "memory_format_version",
            "memory_schema_version",
            "route_top_img_k",
            "parent_topk_per_region",
            "query_chunk_size",
            "verify_start_epoch",
            "full_pc_start_epoch",
            "teacher_only_full_start_epoch",
            "pc_injection_ramp_epochs",
            "diagnostic_window_epochs",
        ):
            setattr(
                self,
                name,
                _as_int(name, getattr(self, name), minimum=1),
            )
        self.p3_min_tokens = _as_int(
            "p3_min_tokens", self.p3_min_tokens, minimum=0
        )
        self.p3_max_tokens = _as_int(
            "p3_max_tokens", self.p3_max_tokens, minimum=0
        )

        if self.input_size != self.token_size * 14:
            raise ValueError(
                "input_size must equal token_size * 14 for the frozen "
                "DINOv2 ViT-B/14 backbone"
            )
        expected_output_size = (self.token_size * 14) // 4
        if self.output_size != expected_output_size:
            raise ValueError(
                "output_size must equal (token_size * 14) // 4 for the "
                f"legacy Decoder, expected {expected_output_size}"
            )
        if self.encoder_dim != 768:
            raise ValueError(
                "the frozen DINOv2 ViT-B/14 backbone requires "
                "encoder_dim=768"
            )
        if self.memory_dim != self.decoder_dim:
            raise ValueError(
                "memory_dim must equal decoder_dim because the "
                "parameter-free Router pools decoder features directly"
            )

        dino_layers = _as_tuple(
            "dino_layer_indices",
            self.dino_layer_indices,
            length=4,
        )
        self.dino_layer_indices = tuple(
            _as_int(f"dino_layer_indices[{index}]", value, minimum=0)
            for index, value in enumerate(dino_layers)
        )
        if len(set(self.dino_layer_indices)) != 4:
            raise ValueError(
                "dino_layer_indices must contain four unique layer indices"
            )
        if max(self.dino_layer_indices) >= 12:
            raise ValueError(
                "dino_layer_indices must be in [0,11] for DINOv2 ViT-B/14"
            )
        self.dino_layer_indices = tuple(sorted(self.dino_layer_indices))

        for name in (
            "memory_source",
            "memory_device",
            "memory_architecture",
        ):
            setattr(
                self,
                name,
                _as_nonempty_str(name, getattr(self, name)),
            )
        self.memory_device = self.memory_device.lower()
        if self.memory_source != "labeled_only":
            raise ValueError(
                "PC-HBM-Lite protocol requires memory_source='labeled_only'"
            )
        if self.use_unlabeled_memory_update:
            raise ValueError(
                "use_unlabeled_memory_update must be False because Memory "
                "may only be rebuilt from labeled data"
            )
        if self.memory_device != "cpu":
            raise ValueError(
                "PC-HBM-Lite protocol requires memory_device='cpu'"
            )
        if self.memory_format_version != 2:
            raise ValueError(
                "PC-HBM-Lite protocol requires memory_format_version=2"
            )
        if self.memory_schema_version != 2:
            raise ValueError(
                "PC-HBM-Lite protocol requires memory_schema_version=2"
            )
        if self.memory_architecture != "DINO_SCOD_PC_HBM_LITE":
            raise ValueError(
                "PC-HBM-Lite protocol requires memory_architecture="
                "'DINO_SCOD_PC_HBM_LITE'"
            )
        self.memory_storage_dtype = _as_nonempty_str(
            "memory_storage_dtype",
            self.memory_storage_dtype,
        ).lower()
        dtype_aliases = {
            "float16": "float16",
            "fp16": "float16",
            "torch.float16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "torch.bfloat16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
            "torch.float32": "float32",
        }
        if self.memory_storage_dtype not in dtype_aliases:
            raise ValueError(
                "memory_storage_dtype must be float16, bfloat16, or float32"
            )
        self.memory_storage_dtype = dtype_aliases[self.memory_storage_dtype]

        for name in (
            "route_global_weight",
            "route_environment_weight",
            "query_boundary_weight",
            "query_uncertainty_weight",
        ):
            setattr(
                self,
                name,
                _as_float(name, getattr(self, name), minimum=0.0),
            )
        if self.route_global_weight + self.route_environment_weight <= 0.0:
            raise ValueError("route weights must not both be zero")
        if self.query_boundary_weight + self.query_uncertainty_weight <= 0.0:
            raise ValueError("query weights must not both be zero")
        self.route_environment_min_mass = _as_float(
            "route_environment_min_mass",
            self.route_environment_min_mass,
            minimum=0.0,
        )

        self.p3_top_ratio = _as_float(
            "p3_top_ratio",
            self.p3_top_ratio,
            minimum=0.0,
            maximum=1.0,
        )
        token_count = self.token_size * self.token_size
        if self.p3_min_tokens > self.p3_max_tokens:
            raise ValueError("p3_min_tokens must not exceed p3_max_tokens")
        if self.p3_max_tokens > token_count:
            raise ValueError(
                "p3_max_tokens must not exceed token_size squared "
                f"({token_count})"
            )

        for name in (
            "fg_boundary_kernel",
            "bg_near_kernel",
            "child_window_size",
        ):
            value = _as_int(name, getattr(self, name), minimum=1)
            if value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
            setattr(self, name, value)
        self.gt_binary_threshold = _as_float(
            "gt_binary_threshold",
            self.gt_binary_threshold,
            minimum=0.0,
            maximum=1.0,
        )

        region_names = _as_tuple(
            "region_names",
            self.region_names,
            length=2,
        )
        self.region_names = tuple(
            _as_nonempty_str(f"region_names[{index}]", value)
            for index, value in enumerate(region_names)
        )
        if len(set(self.region_names)) != 2:
            raise ValueError("region_names must contain two unique names")
        if any(
            name in {"pair_union", "pair_labels"}
            for name in self.region_names
        ):
            raise ValueError(
                "region_names must not use reserved names "
                "'pair_union' or 'pair_labels'"
            )

        minimum_quota = _as_tuple(
            "region_min_quota",
            self.region_min_quota,
            length=2,
        )
        maximum_quota = _as_tuple(
            "region_max_quota",
            self.region_max_quota,
            length=2,
        )
        sampling_ratio = _as_tuple(
            "region_sampling_ratio",
            self.region_sampling_ratio,
            length=2,
        )
        self.region_min_quota = tuple(
            _as_int(f"region_min_quota[{index}]", value, minimum=0)
            for index, value in enumerate(minimum_quota)
        )
        self.region_max_quota = tuple(
            _as_int(f"region_max_quota[{index}]", value, minimum=0)
            for index, value in enumerate(maximum_quota)
        )
        self.region_sampling_ratio = tuple(
            _as_float(
                f"region_sampling_ratio[{index}]",
                value,
                minimum=0.0,
                maximum=1.0,
            )
            for index, value in enumerate(sampling_ratio)
        )
        for index, (minimum, maximum) in enumerate(
            zip(self.region_min_quota, self.region_max_quota)
        ):
            if minimum > maximum:
                raise ValueError(
                    f"region_min_quota[{index}] must not exceed "
                    f"region_max_quota[{index}]"
                )
            if maximum > token_count:
                raise ValueError(
                    f"region_max_quota[{index}] must not exceed token_size "
                    f"squared ({token_count})"
                )

        for name in ("tau_parent", "tau_child"):
            setattr(
                self,
                name,
                _as_float(
                    name,
                    getattr(self, name),
                    minimum=0.0,
                    minimum_open=True,
                ),
            )
        self.child_mix_init_logit = _as_float(
            "child_mix_init_logit",
            self.child_mix_init_logit,
        )

        if self.full_pc_start_epoch < self.verify_start_epoch:
            raise ValueError("full_pc_start_epoch must not precede verification")

        for name in (
            "lambda_pair",
            "lambda_u",
            "feature_distill_p3_weight",
        ):
            setattr(
                self,
                name,
                _as_float(name, getattr(self, name), minimum=0.0),
            )
        self.grad_clip_norm = _as_float(
            "grad_clip_norm",
            self.grad_clip_norm,
            minimum=0.0,
        )
        self.ema_momentum = _as_float(
            "ema_momentum",
            self.ema_momentum,
            minimum=0.0,
            maximum=1.0,
        )
        for name in (
            "warn_low_pair_valid_ratio",
            "warn_pair_acc_near_random",
            "warn_gate_inactive_threshold",
        ):
            setattr(
                self,
                name,
                _as_float(
                    name,
                    getattr(self, name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        self.warn_delta_large_threshold = _as_float(
            "warn_delta_large_threshold",
            self.warn_delta_large_threshold,
            minimum=0.0,
        )

    def configure_training_design(self, training_design: str) -> None:
        """Configure one of the two supported stage schedules."""

        design = str(training_design)
        if design == "two_stage":
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
            "route_environment_min_mass": self.route_environment_min_mass,
            "fg_boundary_kernel": self.fg_boundary_kernel,
            "bg_near_kernel": self.bg_near_kernel,
            "gt_binary_threshold": self.gt_binary_threshold,
            "region_max_quota": tuple(self.region_max_quota),
            "region_min_quota": tuple(self.region_min_quota),
            "region_sampling_ratio": tuple(self.region_sampling_ratio),
        }
        if producer_fingerprint is not None:
            meta["producer_fingerprint"] = str(producer_fingerprint)
        return meta


DEFAULT_PC_HBM_CONFIG = DinoPCHBMConfig()


__all__ = ["DEFAULT_PC_HBM_CONFIG", "DinoPCHBMConfig"]
