"""Training-only pseudo-label refinement for encoder-side PC-HBM.

The refiner deliberately owns the detach boundary between the segmentation
core and pseudo-label generation.  It may be trained from a labeled Student
branch and copied to an EMA Teacher, but its predictions are never part of the
Student core or the formal inference path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.PC_HBM.refinement.boundary_deformation import deform_logits
from Model.PC_HBM.training.supervision import build_gt_boundary


_BRANCH_NAMES = ("keep", "residual", "deformation", "suppress")


@dataclass(frozen=True)
class EncoderRefinerEvidence:
    """Canonical 28x28 evidence consumed by the pseudo-label refiner."""

    verified_evidence: torch.Tensor
    boundary_probability: torch.Tensor
    pc_gate: torch.Tensor
    contradiction: torch.Tensor
    semantic_support: torch.Tensor
    detail_support: torch.Tensor
    valid_map: torch.Tensor
    route_confidence: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EncoderRefinerEvidence":
        """Create the strict evidence contract from an Adapter aux mapping."""

        nested = value.get("refiner_evidence")
        source = nested if isinstance(nested, Mapping) else value
        missing = [
            field
            for field in cls.__dataclass_fields__
            if field not in source or not torch.is_tensor(source[field])
        ]
        if missing:
            raise KeyError(f"encoder refiner evidence is missing tensors: {missing}")
        return cls(**{field: source[field] for field in cls.__dataclass_fields__})

    def validate(self, *, batch_size: int, memory_dim: int = 128) -> None:
        """Validate channels, batch/device placement, and spatial alignment."""

        verified = self.verified_evidence
        if verified.ndim != 4 or verified.shape[:2] != (batch_size, memory_dim):
            raise ValueError(
                "verified_evidence must be "
                f"[B,{memory_dim},H,W], got {tuple(verified.shape)}"
            )
        if not verified.is_floating_point():
            raise TypeError("verified_evidence must use a floating-point dtype")
        spatial_size = verified.shape[-2:]
        if spatial_size != (28, 28):
            raise ValueError(
                "encoder refiner evidence must use the fixed 28x28 token grid, "
                f"got {tuple(spatial_size)}"
            )
        device = verified.device
        for name in (
            "boundary_probability",
            "pc_gate",
            "contradiction",
            "semantic_support",
            "detail_support",
            "valid_map",
        ):
            tensor = getattr(self, name)
            if tensor.shape != (batch_size, 1, *spatial_size):
                raise ValueError(
                    f"{name} must be [B,1,{spatial_size[0]},{spatial_size[1]}], "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}, got {tensor.device}")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")

        route = self.route_confidence
        valid_route_shapes = {
            (batch_size,),
            (batch_size, 1),
            (batch_size, 1, 1, 1),
            (batch_size, 1, *spatial_size),
        }
        if tuple(route.shape) not in valid_route_shapes:
            raise ValueError(
                "route_confidence must be [B], [B,1], [B,1,1,1], or "
                f"[B,1,H,W], got {tuple(route.shape)}"
            )
        if route.device != device:
            raise ValueError(
                f"route_confidence must be on {device}, got {route.device}"
            )
        if not route.is_floating_point():
            raise TypeError("route_confidence must use a floating-point dtype")


class TeacherPseudoRefinerOutput(TypedDict):
    """Tensor contract returned by :class:`TeacherPseudoLabelRefiner`."""

    z_keep: torch.Tensor
    z_residual: torch.Tensor
    z_deformation: torch.Tensor
    z_suppress: torch.Tensor
    p_keep: torch.Tensor
    p_residual: torch.Tensor
    p_deformation: torch.Tensor
    p_suppress: torch.Tensor
    candidates: torch.Tensor
    candidate_probabilities: torch.Tensor
    mixture_logits: torch.Tensor
    pi: torch.Tensor
    mixture_entropy: torch.Tensor
    z_pseudo_refined: torch.Tensor
    p_pseudo_refined: torch.Tensor
    branch_quality: torch.Tensor
    correction_mask: torch.Tensor
    residual: torch.Tensor
    offset: torch.Tensor
    suppression: torch.Tensor
    encoder_evidence_98: torch.Tensor
    temperature: torch.Tensor
    epsilon_floor: torch.Tensor


@dataclass(frozen=True)
class RefinerLossWeights:
    """Starting loss weights from the encoder-side PC-HBM specification."""

    refined_final: float = 1.0
    mix_oracle: float = 0.10
    branch: float = 0.10
    quality: float = 0.025
    usage: float = 0.01
    regularization: float = 0.02

    @classmethod
    def from_config(cls, config: Any | None) -> "RefinerLossWeights":
        if config is None:
            return cls()

        def required_weight(canonical: str, *aliases: str) -> float:
            names = (canonical, *aliases)
            values = [
                (name, float(getattr(config, name)))
                for name in names
                if hasattr(config, name)
            ]
            if not values:
                raise AttributeError(
                    "refiner config is missing required weight "
                    f"{canonical!r} (accepted aliases: {aliases})"
                )
            reference = values[0][1]
            if any(value != reference for _, value in values[1:]):
                raise ValueError(
                    f"conflicting refiner weight aliases for {canonical!r}: {values}"
                )
            return reference

        return cls(
            refined_final=required_weight(
                "lambda_refined_final", "lambda_refiner_final"
            ),
            mix_oracle=required_weight("lambda_mix_oracle"),
            branch=required_weight("lambda_branch"),
            quality=required_weight("lambda_quality"),
            usage=required_weight("lambda_usage"),
            regularization=required_weight("lambda_refiner_reg", "lambda_reg"),
        )


def _gradient_strength(value: torch.Tensor) -> torch.Tensor:
    grad_x = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)


def _zero_init(module: nn.Conv2d) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class TeacherPseudoLabelRefiner(nn.Module):
    """Generate refined Teacher pseudo-labels from detached core evidence."""

    output_size = (98, 98)
    evidence_channels = 128 + 7

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.config = config
        self.memory_dim = int(getattr(config, "memory_dim", 128))
        self.decoder_channels = int(getattr(config, "decoder_dim", 128))
        if self.memory_dim != 128:
            raise ValueError("TeacherPseudoLabelRefiner requires memory_dim=128")
        if self.decoder_channels != 128:
            raise ValueError("TeacherPseudoLabelRefiner requires decoder_dim=128")

        self.residual_max = float(getattr(config, "refiner_residual_max", 2.0))
        self.offset_max = float(getattr(config, "refiner_offset_max", 1.5))
        if self.residual_max <= 0.0 or self.offset_max <= 0.0:
            raise ValueError("refiner residual and offset limits must be positive")

        self.evidence_projector = nn.Sequential(
            nn.Conv2d(self.evidence_channels, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128),
            nn.GELU(),
        )
        context_channels = 1 + self.decoder_channels + 128 + 1 + 1
        self.context_trunk = nn.Sequential(
            nn.Conv2d(context_channels, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.GELU(),
            nn.Conv2d(64, 64, 1),
            nn.GELU(),
        )
        self.correction_mask_head = nn.Conv2d(64, 1, 1)
        self.residual_head = nn.Conv2d(64, 1, 1)
        self.offset_head = nn.Conv2d(64, 2, 1)
        self.suppress_head = nn.Conv2d(64, 1, 1)
        self.mixture_head = nn.Conv2d(64, 4, 1)
        self.quality_head = nn.Conv2d(64, 4, 1)

        for head in (
            self.correction_mask_head,
            self.residual_head,
            self.offset_head,
            self.suppress_head,
            self.mixture_head,
            self.quality_head,
        ):
            _zero_init(head)
        with torch.no_grad():
            self.mixture_head.bias.copy_(
                torch.tensor(
                    (1.0, -0.5, -0.5, -0.5),
                    dtype=self.mixture_head.bias.dtype,
                )
            )

    def _schedule(
        self,
        epoch: int | None,
        *,
        ts_continuation: bool,
    ) -> tuple[float, float]:
        schedule = getattr(self.config, "mixture_schedule", None)
        if callable(schedule):
            try:
                temperature, epsilon = schedule(
                    epoch, ts_continuation=ts_continuation
                )
            except TypeError:
                temperature, epsilon = schedule(epoch)
            return float(temperature), float(epsilon)

        start_temperature = float(
            getattr(self.config, "mixture_temperature_start", 1.50)
        )
        end_temperature = float(
            getattr(self.config, "mixture_temperature_end", 0.80)
        )
        start_epsilon = float(getattr(self.config, "mixture_eps_start", 0.10))
        end_epsilon = float(getattr(self.config, "mixture_eps_end", 0.0))
        if ts_continuation:
            return end_temperature, end_epsilon
        start_epoch = int(getattr(self.config, "refiner_start_epoch", 21))
        end_epoch = int(getattr(self.config, "mixture_schedule_end_epoch", 30))
        current_epoch = start_epoch if epoch is None else int(epoch)
        progress = min(
            1.0,
            max(0.0, (current_epoch - start_epoch) / max(1, end_epoch - start_epoch)),
        )
        return (
            start_temperature + progress * (end_temperature - start_temperature),
            start_epsilon + progress * (end_epsilon - start_epsilon),
        )

    @staticmethod
    def _route_map(
        route_confidence: torch.Tensor,
        *,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        if route_confidence.ndim == 1:
            route_confidence = route_confidence[:, None, None, None]
        elif route_confidence.ndim == 2:
            route_confidence = route_confidence[:, :, None, None]
        if route_confidence.shape[-2:] != spatial_size:
            route_confidence = F.interpolate(
                route_confidence, size=spatial_size, mode="nearest"
            )
        return route_confidence

    def _project_evidence(
        self,
        evidence: EncoderRefinerEvidence,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_size = evidence.verified_evidence.shape[-2:]
        route_map = self._route_map(
            evidence.route_confidence, spatial_size=spatial_size
        )
        raw_evidence = torch.cat(
            (
                evidence.verified_evidence,
                evidence.boundary_probability,
                evidence.pc_gate,
                evidence.contradiction,
                evidence.semantic_support,
                evidence.detail_support,
                evidence.valid_map,
                route_map,
            ),
            dim=1,
        ).detach().to(dtype=dtype)
        projected = self.evidence_projector(raw_evidence)
        projected_98 = F.interpolate(
            projected, size=self.output_size, mode="bilinear", align_corners=False
        )
        valid_98 = F.interpolate(
            evidence.valid_map.detach().to(dtype=dtype),
            size=self.output_size,
            mode="nearest",
        ).clamp(0.0, 1.0)
        return projected_98, valid_98

    def forward(
        self,
        z_core: torch.Tensor,
        decoder_feature: torch.Tensor,
        encoder_aux: EncoderRefinerEvidence | Mapping[str, Any],
        epoch: int | None = None,
        *,
        temperature: float | None = None,
        epsilon_floor: float | None = None,
        ts_continuation: bool = False,
    ) -> TeacherPseudoRefinerOutput:
        if z_core.ndim != 4 or z_core.shape[1:] != (1, *self.output_size):
            raise ValueError(
                "z_core must be [B,1,98,98], "
                f"got {tuple(z_core.shape)}"
            )
        if not z_core.is_floating_point():
            raise TypeError("z_core must use a floating-point dtype")
        expected_decoder_shape = (
            z_core.size(0),
            self.decoder_channels,
            *self.output_size,
        )
        if tuple(decoder_feature.shape) != expected_decoder_shape:
            raise ValueError(
                f"decoder_feature must be {expected_decoder_shape}, "
                f"got {tuple(decoder_feature.shape)}"
            )
        if decoder_feature.device != z_core.device:
            raise ValueError("z_core and decoder_feature must share a device")
        if not decoder_feature.is_floating_point():
            raise TypeError("decoder_feature must use a floating-point dtype")

        evidence = (
            encoder_aux
            if isinstance(encoder_aux, EncoderRefinerEvidence)
            else EncoderRefinerEvidence.from_mapping(encoder_aux)
        )
        evidence.validate(batch_size=z_core.size(0), memory_dim=self.memory_dim)
        if evidence.verified_evidence.device != z_core.device:
            raise ValueError("z_core and encoder evidence must share a device")

        scheduled_temperature, scheduled_epsilon = self._schedule(
            epoch, ts_continuation=ts_continuation
        )
        temperature = (
            scheduled_temperature if temperature is None else float(temperature)
        )
        epsilon_floor = (
            scheduled_epsilon if epsilon_floor is None else float(epsilon_floor)
        )
        if temperature <= 0.0:
            raise ValueError("mixture temperature must be positive")
        if not 0.0 <= epsilon_floor < 1.0:
            raise ValueError("mixture epsilon_floor must be in [0,1)")

        # This internal detach is intentional defense-in-depth: even if a future
        # caller forgets the head-level detach, refiner losses cannot mutate the
        # Student core, Decoder, or Encoder Adapter.
        core = z_core.detach()
        decoder = decoder_feature.detach().to(dtype=core.dtype)
        encoder_evidence_98, valid_98 = self._project_evidence(
            evidence, dtype=core.dtype
        )
        core_probability = torch.sigmoid(core)
        uncertainty = 4.0 * core_probability * (1.0 - core_probability)
        core_gradient = _gradient_strength(core_probability)
        context = self.context_trunk(
            torch.cat(
                (
                    core_probability,
                    decoder,
                    encoder_evidence_98,
                    uncertainty,
                    core_gradient,
                ),
                dim=1,
            )
        )

        correction_mask = torch.sigmoid(self.correction_mask_head(context)) * valid_98
        residual = torch.tanh(self.residual_head(context)) * self.residual_max
        offset = torch.tanh(self.offset_head(context)) * self.offset_max
        suppress_raw = self.suppress_head(context)
        suppression_raw = F.softplus(suppress_raw) - math.log(2.0)
        suppression_forward = suppression_raw.clamp_min(0.0)
        # A straight-through clamp gives an exactly non-negative subtractive
        # correction in the forward pass while preserving the softplus slope
        # at the zero-initialized head (no first-step gradient deadlock).
        suppression = suppression_raw + (
            suppression_forward - suppression_raw
        ).detach()

        z_keep = core
        z_residual = core + correction_mask * residual
        sampled = deform_logits(core, offset, correction_mask)
        identity_sample = deform_logits(
            core, torch.zeros_like(offset), torch.zeros_like(correction_mask)
        )
        # Subtract the identical zero-offset sample first.  This is exactly
        # identity at initialization while preserving gradients to ``offset``.
        z_deformation = (sampled - identity_sample) + core
        z_suppress = core - correction_mask * suppression
        candidates = torch.cat(
            (z_keep, z_residual, z_deformation, z_suppress), dim=1
        )

        mixture_logits = self.mixture_head(context)
        pi = torch.softmax(mixture_logits / temperature, dim=1)
        if epsilon_floor > 0.0:
            pi = (1.0 - epsilon_floor) * pi + epsilon_floor / 4.0
        # Delta form is algebraically the specified weighted mixture and keeps
        # the zero-initialized refiner bitwise identical to ``z_core``.
        z_pseudo_refined = core + (pi * (candidates - core)).sum(
            dim=1, keepdim=True
        )
        # Evaluate each one-channel branch separately.  Besides making the
        # probability contract explicit, this preserves exact equality with
        # ``sigmoid(z_core)`` at zero initialization on vectorized CPU kernels.
        p_keep = torch.sigmoid(z_keep)
        p_residual = torch.sigmoid(z_residual)
        p_deformation = torch.sigmoid(z_deformation)
        p_suppress = torch.sigmoid(z_suppress)
        candidate_probabilities = torch.cat(
            (p_keep, p_residual, p_deformation, p_suppress), dim=1
        )
        p_pseudo_refined = torch.sigmoid(z_pseudo_refined)
        mixture_entropy = -(
            pi.clamp_min(1.0e-8) * pi.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True) / math.log(4.0)

        return {
            "z_keep": z_keep,
            "z_residual": z_residual,
            "z_deformation": z_deformation,
            "z_suppress": z_suppress,
            "p_keep": p_keep,
            "p_residual": p_residual,
            "p_deformation": p_deformation,
            "p_suppress": p_suppress,
            "candidates": candidates,
            "candidate_probabilities": candidate_probabilities,
            "mixture_logits": mixture_logits,
            "pi": pi,
            "mixture_entropy": mixture_entropy,
            "z_pseudo_refined": z_pseudo_refined,
            "p_pseudo_refined": p_pseudo_refined,
            "branch_quality": self.quality_head(context),
            "correction_mask": correction_mask,
            "residual": residual,
            "offset": offset,
            "suppression": suppression,
            "encoder_evidence_98": encoder_evidence_98,
            "temperature": core.new_tensor(temperature),
            "epsilon_floor": core.new_tensor(epsilon_floor),
        }


def _probability_bce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """AMP-safe BCE for tensors that are already sigmoid probabilities."""

    if prediction.shape != target.shape:
        raise ValueError(
            "probability prediction and target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    eps = 1.0e-4 if prediction.dtype in (torch.float16, torch.bfloat16) else 1.0e-6
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return F.binary_cross_entropy(
            prediction.float().clamp(eps, 1.0 - eps),
            target.detach().float().clamp(0.0, 1.0),
            reduction=reduction,
        )


def _candidate_errors(
    candidate_probabilities: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    eps = 1.0e-6
    probabilities = candidate_probabilities.float().clamp(eps, 1.0 - eps)
    expanded_target = target.float().expand_as(probabilities)
    bce = -(
        expanded_target * probabilities.log()
        + (1.0 - expanded_target) * (1.0 - probabilities).log()
    )
    return bce + (probabilities - expanded_target).abs()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _usage_loss(
    pi: torch.Tensor,
    target: torch.Tensor,
    keep_probability: torch.Tensor,
) -> torch.Tensor:
    false_negative = ((target > 0.5) & (keep_probability < 0.4)).to(pi.dtype)
    false_positive = ((target < 0.5) & (keep_probability > 0.6)).to(pi.dtype)
    boundary = build_gt_boundary(target, tuple(pi.shape[-2:])).to(pi.dtype)
    boundary = boundary * (1.0 - false_negative) * (1.0 - false_positive)
    stable = (
        1.0 - torch.maximum(torch.maximum(false_negative, false_positive), boundary)
    ).clamp(0.0, 1.0)
    targets = (
        (stable, (0.90, 0.03, 0.04, 0.03)),
        (false_negative, (0.20, 0.60, 0.15, 0.05)),
        (false_positive, (0.20, 0.05, 0.15, 0.60)),
        (boundary, (0.25, 0.15, 0.50, 0.10)),
    )
    total = pi.sum() * 0.0
    for mask, distribution in targets:
        desired = pi.new_tensor(distribution).view(1, 4, 1, 1)
        cross_entropy = -(desired * pi.clamp_min(1.0e-8).log()).sum(
            dim=1, keepdim=True
        )
        total = total + _masked_mean(cross_entropy, mask)
    return total


def teacher_pseudo_refiner_labeled_loss(
    refiner_output: Mapping[str, torch.Tensor],
    gt: torch.Tensor,
    config: Any | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only the pseudo refiner from a labeled target.

    ``p_pseudo_refined`` and all four candidate probabilities are already
    sigmoid probabilities.  They are therefore supervised with probability
    BCE and are never passed to ``binary_cross_entropy_with_logits``.
    """

    required = (
        "p_pseudo_refined",
        "candidate_probabilities",
        "pi",
        "branch_quality",
        "correction_mask",
        "offset",
        "z_pseudo_refined",
        "z_keep",
    )
    missing = [name for name in required if not torch.is_tensor(refiner_output.get(name))]
    if missing:
        raise KeyError(f"refiner output is missing tensors: {missing}")

    refined_probability = refiner_output["p_pseudo_refined"]
    if refined_probability.ndim != 4 or refined_probability.size(1) != 1:
        raise ValueError("p_pseudo_refined must be [B,1,H,W]")
    target = gt
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.size(1) != 1:
        raise ValueError("gt must be [B,H,W] or [B,1,H,W]")
    target = F.interpolate(
        target.detach().to(device=refined_probability.device, dtype=torch.float32),
        size=refined_probability.shape[-2:],
        mode="nearest",
    ).clamp(0.0, 1.0)

    candidate_probabilities = refiner_output["candidate_probabilities"]
    pi = refiner_output["pi"]
    quality = refiner_output["branch_quality"]
    expected_branch_shape = (
        refined_probability.size(0),
        4,
        *refined_probability.shape[-2:],
    )
    for name, tensor in (
        ("candidate_probabilities", candidate_probabilities),
        ("pi", pi),
        ("branch_quality", quality),
    ):
        if tuple(tensor.shape) != expected_branch_shape:
            raise ValueError(
                f"{name} must be {expected_branch_shape}, got {tuple(tensor.shape)}"
            )

    loss_refined_final = _probability_bce(refined_probability, target)
    branch_target = target.expand_as(candidate_probabilities)
    loss_branch = _probability_bce(candidate_probabilities, branch_target)

    errors = _candidate_errors(candidate_probabilities, target)
    oracle_temperature = float(getattr(config, "mix_oracle_temperature", 0.5))
    min_improvement = float(getattr(config, "mix_oracle_min_improvement", 0.03))
    target_mix = torch.softmax(
        -errors / max(oracle_temperature, 1.0e-6), dim=1
    ).detach()
    improvement = (errors[:, 0:1] - errors.min(dim=1, keepdim=True).values).detach()
    oracle_mask = (improvement > min_improvement).to(pi.dtype)
    kl = (
        target_mix
        * (target_mix.clamp_min(1.0e-8).log() - pi.clamp_min(1.0e-8).log())
    ).sum(dim=1, keepdim=True)
    loss_mix_oracle = _masked_mean(kl, oracle_mask)

    target_gain = (errors[:, 0:1] - errors).detach().to(dtype=quality.dtype)
    quality_weight = 0.25 + 0.75 * refiner_output["correction_mask"].detach()
    quality_weight = quality_weight.expand_as(quality)
    quality_error = F.smooth_l1_loss(
        quality, target_gain, reduction="none"
    )
    loss_quality = (quality_error * quality_weight).sum() / quality_weight.sum().clamp_min(1.0)
    loss_usage = _usage_loss(
        pi,
        target,
        candidate_probabilities[:, 0:1].detach(),
    )

    offset = refiner_output["offset"]
    correction_mask = refiner_output["correction_mask"]
    loss_regularization = offset.abs().mean()
    if offset.size(-2) > 1:
        loss_regularization = loss_regularization + (
            offset[..., 1:, :] - offset[..., :-1, :]
        ).abs().mean()
    if offset.size(-1) > 1:
        loss_regularization = loss_regularization + (
            offset[..., :, 1:] - offset[..., :, :-1]
        ).abs().mean()
    loss_regularization = loss_regularization + 0.1 * correction_mask.mean()
    loss_regularization = loss_regularization + 0.01 * (
        refiner_output["z_pseudo_refined"] - refiner_output["z_keep"]
    ).abs().mean()

    weights = RefinerLossWeights.from_config(config)
    total = (
        weights.refined_final * loss_refined_final
        + weights.mix_oracle * loss_mix_oracle
        + weights.branch * loss_branch
        + weights.quality * loss_quality
        + weights.usage * loss_usage
        + weights.regularization * loss_regularization
    )
    terms = {
        "L_refined_final": loss_refined_final.detach(),
        "L_mix_oracle": loss_mix_oracle.detach(),
        "L_branch": loss_branch.detach(),
        "L_quality": loss_quality.detach(),
        "L_usage": loss_usage.detach(),
        "L_refiner_reg": loss_regularization.detach(),
        "L_refiner_total": total.detach(),
    }
    return total, terms


__all__ = [
    "EncoderRefinerEvidence",
    "RefinerLossWeights",
    "TeacherPseudoLabelRefiner",
    "TeacherPseudoRefinerOutput",
    "teacher_pseudo_refiner_labeled_loss",
]
