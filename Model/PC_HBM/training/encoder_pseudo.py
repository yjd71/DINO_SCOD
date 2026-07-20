"""Pseudo targets and unlabeled losses for encoder-side PC-HBM v3.

This module is intentionally independent from :mod:`pseudo_label`, whose
contracts belong to the legacy decoder-side PC-HBM profile.  The encoder-side
Teacher may refine its core probability during training, but the unlabeled
Student is supervised through the unchanged Decoder core logit at
``outputs[3]`` only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def build_encoder_pc_confidence(
    teacher_payload: Mapping[str, Any],
    config: Any,
) -> torch.Tensor:
    """Build detached five-factor confidence for refined Teacher pseudo labels.

    The factors are refined-probability certainty, low PC-HCA contradiction,
    low refiner-mixture entropy, margin-derived route confidence with a floor,
    and a soft core/refined agreement term.  Route entropy is deliberately not
    read here: a diffuse route must not collapse otherwise useful pseudo
    labels to zero.
    """

    confidence, _ = _encoder_pc_confidence_with_components(teacher_payload, config)
    return confidence


def prepare_encoder_pc_pseudo_targets(
    teacher_payload: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Detach the Teacher refiner result and derive soft/hard pseudo targets."""

    refiner = _require_mapping(teacher_payload, "pseudo_refiner")
    p_soft = _require_probability(
        refiner,
        "p_pseudo_refined",
        expected_channels=1,
    ).detach().clone()
    confidence, components = _encoder_pc_confidence_with_components(
        teacher_payload, config
    )
    hard = _build_hard_targets(p_soft, confidence, config)
    hard_coverage = hard["hard_valid"].to(dtype=p_soft.dtype).mean()
    coverage_target = _positive_float(config, "hard_coverage_target", 0.20)
    hard_coverage_scale = (hard_coverage / coverage_target).clamp(max=1.0)
    return {
        "p_soft": p_soft,
        "confidence": confidence.detach().clone(),
        **hard,
        "hard_coverage": hard_coverage.detach(),
        "hard_coverage_scale": hard_coverage_scale.detach(),
        "confidence_components": {
            name: value.detach().clone() for name, value in components.items()
        },
    }


def confidence_weighted_logit_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    target_mode: str = "bilinear",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Confidence-weighted BCE **with logits** with a differentiable zero.

    ``target`` is a probability and is never treated as a logit.  Conversely,
    ``logits`` is passed directly to ``binary_cross_entropy_with_logits`` and
    is never sigmoid-transformed before BCE.
    """

    target, confidence = _prepare_weighted_target(
        logits,
        target,
        confidence,
        target_mode=target_mode,
    )
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_fp32 = logits.float()
        target_fp32 = target.float()
        confidence_fp32 = confidence.float()
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits_fp32, target_fp32, reduction="none"
        )
        weighted = pixel_loss * confidence_fp32
        return weighted.sum() / confidence_fp32.sum().clamp_min(float(eps))


def confidence_weighted_structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    target_mode: str = "bilinear",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """F3Net-style weighted BCE-with-logits plus weighted IoU.

    Empty confidence masks remain connected to ``logits`` and therefore
    produce a differentiable scalar zero rather than a detached constant.
    """

    target, confidence = _prepare_weighted_target(
        logits,
        target,
        confidence,
        target_mode=target_mode,
    )
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_fp32 = logits.float()
        target_fp32 = target.float()
        confidence_fp32 = confidence.float()
        structure_weight = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(target_fp32, kernel_size=31, stride=1, padding=15)
            - target_fp32
        )
        weight = structure_weight * confidence_fp32
        bce = F.binary_cross_entropy_with_logits(
            logits_fp32, target_fp32, reduction="none"
        )
        weighted_bce = (bce * weight).sum(dim=(1, 2, 3)) / weight.sum(
            dim=(1, 2, 3)
        ).clamp_min(float(eps))

        probability = torch.sigmoid(logits_fp32)
        intersection = (probability * target_fp32 * weight).sum(dim=(1, 2, 3))
        union = ((probability + target_fp32) * weight).sum(dim=(1, 2, 3))
        weighted_iou = 1.0 - (intersection + 1.0) / (
            union - intersection + 1.0
        )
        return (weighted_bce + weighted_iou).mean()


def encoder_pc_unlabeled_loss(
    outputs: Sequence[torch.Tensor],
    student_aux: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    config: Any,
    ts_epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only the unlabeled Student's Decoder core and side logits.

    The main tensor is unconditionally ``outputs[3]``.  ``student_aux`` is
    used only to verify the stable ``z_core`` metadata and that the
    training-only pseudo refiner was skipped.
    """

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError(
            "Student outputs must be (m4, m3, m2, z_core, global_logit)"
        )
    if not isinstance(student_aux, Mapping):
        raise TypeError("student_aux must be a mapping")
    if student_aux.get("pseudo_refiner") is not None:
        raise RuntimeError("Student unlabeled branch must skip pseudo refiner")
    if not isinstance(pseudo, Mapping):
        raise TypeError("pseudo must be a mapping")

    names = ("m4", "m3", "m2", "z_core", "global_logit")
    for name, value in zip(names, outputs):
        _validate_logit(value, name)
    m4, m3, m2, z_core, global_logit = outputs

    aux_z_core = student_aux.get("z_core")
    if not torch.is_tensor(aux_z_core):
        raise KeyError("student_aux['z_core'] is required")
    if (
        aux_z_core.shape != z_core.shape
        or aux_z_core.device != z_core.device
        or aux_z_core.dtype != z_core.dtype
    ):
        raise ValueError(
            "outputs[3] and student_aux['z_core'] must have matching shape, "
            "device, and dtype"
        )

    p_soft = _require_probability(pseudo, "p_soft", expected_channels=1)
    confidence = _require_probability(
        pseudo, "confidence", expected_channels=1
    )
    if p_soft.shape != confidence.shape:
        raise ValueError(
            "pseudo p_soft and confidence must have identical shapes, got "
            f"{tuple(p_soft.shape)} and {tuple(confidence.shape)}"
        )
    if p_soft.size(0) != z_core.size(0):
        raise ValueError("pseudo and Student batch sizes must match")
    p_soft = p_soft.detach()
    confidence = confidence.detach()
    hard = _build_hard_targets(p_soft, confidence, config)

    loss_soft = confidence_weighted_structure_loss(z_core, p_soft, confidence)
    loss_hard = confidence_weighted_structure_loss(
        z_core,
        hard["hard_target"],
        hard["hard_weight"],
        target_mode="nearest",
    )
    loss_side = (
        0.30 * confidence_weighted_structure_loss(m2, p_soft, confidence)
        + 0.20 * confidence_weighted_structure_loss(m3, p_soft, confidence)
        + 0.10 * confidence_weighted_structure_loss(m4, p_soft, confidence)
        + 0.10
        * confidence_weighted_structure_loss(global_logit, p_soft, confidence)
    )

    hard_coverage = hard["hard_valid"].to(
        device=z_core.device, dtype=z_core.dtype
    ).mean()
    coverage_target = _positive_float(config, "hard_coverage_target", 0.20)
    coverage_scale = (hard_coverage / coverage_target).clamp(max=1.0)
    ramp_epochs = _positive_int(config, "pseudo_hard_ramp_epochs", 3)
    if int(ts_epoch) < 0:
        raise ValueError("ts_epoch must be non-negative")
    hard_ramp = min(1.0, float(ts_epoch) / float(ramp_epochs))
    hard_loss_weight = _nonnegative_float(config, "hard_loss_weight", 2.0)
    loss_hard_scaled = (
        hard_loss_weight * hard_ramp * coverage_scale * loss_hard
    )
    total = loss_soft + loss_hard_scaled + loss_side

    positive_confidence = confidence > 0
    log = {
        "L_u_soft": loss_soft.detach(),
        "L_u_hard": loss_hard.detach(),
        "L_u_hard_scaled": loss_hard_scaled.detach(),
        "L_u_side": loss_side.detach(),
        "hard_ramp": z_core.new_tensor(hard_ramp).detach(),
        "hard_valid_ratio": hard_coverage.detach(),
        "hard_coverage_scale": coverage_scale.detach(),
        "pseudo_conf_mean": confidence.mean().detach(),
        "pseudo_conf_positive_fraction": positive_confidence
        .to(dtype=z_core.dtype)
        .mean()
        .detach(),
        "loss_unlabeled": total.detach(),
    }
    return total, log


def _encoder_pc_confidence_with_components(
    teacher_payload: Mapping[str, Any],
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not isinstance(teacher_payload, Mapping):
        raise TypeError("teacher_payload must be a mapping")
    refiner = _require_mapping(teacher_payload, "pseudo_refiner")
    encoder_aux = _require_mapping(teacher_payload, "encoder_pc_hbm")
    route = _require_mapping(encoder_aux, "route")
    p_refined = _require_probability(
        refiner,
        "p_pseudo_refined",
        expected_channels=1,
    )
    z_core = teacher_payload.get("z_core")
    if not torch.is_tensor(z_core):
        outputs = teacher_payload.get("outputs")
        if isinstance(outputs, (tuple, list)) and len(outputs) == 5:
            z_core = outputs[3]
    _validate_logit(z_core, "teacher z_core")
    if z_core.size(0) != p_refined.size(0):
        raise ValueError("Teacher z_core and refined probability batch sizes must match")

    p_refined = p_refined.detach()
    p_core = torch.sigmoid(
        z_core.detach().to(device=p_refined.device, dtype=p_refined.dtype)
    )
    if p_core.shape[-2:] != p_refined.shape[-2:]:
        p_core = F.interpolate(
            p_core,
            size=p_refined.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if p_core.shape != p_refined.shape:
        raise ValueError("Teacher core and refined probability shapes are incompatible")

    c23 = _require_probability(encoder_aux, "C23_map", expected_channels=1)
    c23 = _resize_probability_like(c23.detach(), p_refined, "C23_map")

    pi = _require_probability(refiner, "pi", expected_channels=4)
    if pi.size(0) != p_refined.size(0):
        raise ValueError("pseudo_refiner.pi batch size must match p_pseudo_refined")
    pi = pi.detach().to(device=p_refined.device, dtype=p_refined.dtype)
    if pi.shape[-2:] != p_refined.shape[-2:]:
        pi = F.interpolate(
            pi,
            size=p_refined.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pi_sum = pi.sum(dim=1, keepdim=True)
    tolerance = 5.0e-3 if pi.dtype in (torch.float16, torch.bfloat16) else 1.0e-4
    if not torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=tolerance, rtol=tolerance):
        raise ValueError("pseudo_refiner.pi must sum to one across four branches")
    pi = pi / pi_sum.clamp_min(torch.finfo(pi.dtype).eps)
    mixture_entropy = -(
        pi * pi.clamp_min(torch.finfo(pi.dtype).eps).log()
    ).sum(dim=1, keepdim=True) / math.log(4.0)

    route_confidence = route.get("route_confidence")
    if not torch.is_tensor(route_confidence):
        route_confidence = route.get("route_margin_confidence")
    if not torch.is_tensor(route_confidence):
        raise KeyError(
            "teacher_payload['encoder_pc_hbm']['route'] requires "
            "route_confidence or route_margin_confidence"
        )
    route_confidence = _broadcast_route_confidence(
        route_confidence.detach(), p_refined
    )
    route_floor = _unit_interval_float(config, "route_confidence_floor", 0.20)
    route_confidence = route_confidence.clamp_min(route_floor)

    probability_confidence = (2.0 * (p_refined - 0.5).abs()).clamp(0.0, 1.0)
    contradiction_confidence = (1.0 - c23).clamp(0.0, 1.0)
    mixture_confidence = (1.0 - mixture_entropy).clamp(0.0, 1.0)
    agreement_confidence = 0.5 + 0.5 * torch.exp(
        -(p_refined - p_core).abs() / 0.25
    )
    confidence = (
        probability_confidence
        * contradiction_confidence
        * mixture_confidence
        * route_confidence
        * agreement_confidence
    ).clamp(0.0, 1.0)
    components = {
        "probability_confidence": probability_confidence,
        "contradiction_confidence": contradiction_confidence,
        "mixture_confidence": mixture_confidence,
        "route_confidence": route_confidence,
        "agreement_confidence": agreement_confidence,
    }
    return confidence.detach(), components


def _build_hard_targets(
    p_soft: torch.Tensor,
    confidence: torch.Tensor,
    config: Any,
) -> dict[str, torch.Tensor]:
    fg_threshold = _unit_interval_float(config, "pseudo_fg_threshold", 0.70)
    bg_threshold = _unit_interval_float(config, "pseudo_bg_threshold", 0.30)
    if not bg_threshold < 0.5 < fg_threshold:
        raise ValueError("pseudo thresholds must satisfy bg < 0.5 < fg")
    hard_target = (p_soft >= 0.5).to(dtype=p_soft.dtype)
    hard_valid = (p_soft >= fg_threshold) | (p_soft <= bg_threshold)
    hard_weight = confidence * hard_valid.to(dtype=confidence.dtype)
    return {
        "hard_target": hard_target.detach().clone(),
        "hard_valid": hard_valid.detach().clone(),
        "hard_weight": hard_weight.detach().clone(),
    }


def _prepare_weighted_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logit(logits, "logits")
    if target_mode not in {"bilinear", "nearest"}:
        raise ValueError("target_mode must be 'bilinear' or 'nearest'")
    target = _canonical_probability_map(target, "target")
    confidence = _canonical_probability_map(confidence, "confidence")
    if target.size(0) != logits.size(0) or confidence.size(0) != logits.size(0):
        raise ValueError("target, confidence and logits batch sizes must match")
    if target.shape[-2:] != logits.shape[-2:]:
        interpolate_kwargs = {"mode": target_mode}
        if target_mode == "bilinear":
            interpolate_kwargs["align_corners"] = False
        target = F.interpolate(target.float(), size=logits.shape[-2:], **interpolate_kwargs)
    if confidence.shape[-2:] != logits.shape[-2:]:
        confidence = F.interpolate(
            confidence.float(),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    target = target.detach().to(device=logits.device, dtype=logits.dtype)
    confidence = confidence.detach().to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape or confidence.shape != logits.shape:
        raise ValueError(
            "logits, target and confidence must resolve to identical [B,1,H,W] shapes"
        )
    return target, confidence


def _resize_probability_like(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.size(0) != reference.size(0):
        raise ValueError(f"{name} batch size must match p_pseudo_refined")
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if value.shape != reference.shape:
        raise ValueError(f"{name} cannot be aligned with p_pseudo_refined")
    return value.clamp(0.0, 1.0)


def _broadcast_route_confidence(
    route_confidence: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if route_confidence.ndim == 0:
        route_confidence = route_confidence.expand(reference.size(0))
    if route_confidence.size(0) != reference.size(0):
        raise ValueError("route confidence batch size must match p_pseudo_refined")
    route_confidence = route_confidence.to(
        device=reference.device, dtype=reference.dtype
    )
    route_confidence = route_confidence.reshape(route_confidence.size(0), -1)
    if route_confidence.size(1) != 1:
        raise ValueError("route confidence must contain one scalar per batch item")
    if not torch.isfinite(route_confidence).all():
        raise ValueError("route confidence contains non-finite values")
    if bool(((route_confidence < 0.0) | (route_confidence > 1.0)).any()):
        raise ValueError("route confidence must be a probability in [0, 1]")
    return route_confidence.view(-1, 1, 1, 1)


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"{key!r} mapping is required")
    return value


def _require_probability(
    mapping: Mapping[str, Any],
    key: str,
    *,
    expected_channels: int,
) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"{key!r} probability tensor is required")
    value = _canonical_probability_map(value, key)
    if value.size(1) != expected_channels:
        raise ValueError(
            f"{key} must have {expected_channels} channels, got {value.size(1)}"
        )
    return value


def _canonical_probability_map(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or not value.is_floating_point() or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty floating [B,C,H,W] tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return value


def _validate_logit(value: Any, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 4 or value.size(1) != 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating [B,1,H,W] logit tensor")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be non-empty and finite")


def _positive_int(config: Any, name: str, default: int) -> int:
    value = int(getattr(config, name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _positive_float(config: Any, name: str, default: float) -> float:
    value = float(getattr(config, name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative_float(config: Any, name: str, default: float) -> float:
    value = float(getattr(config, name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return value


def _unit_interval_float(config: Any, name: str, default: float) -> float:
    value = float(getattr(config, name, default))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


__all__ = [
    "build_encoder_pc_confidence",
    "confidence_weighted_logit_bce",
    "confidence_weighted_structure_loss",
    "encoder_pc_unlabeled_loss",
    "prepare_encoder_pc_pseudo_targets",
]
