"""Soft pseudo-label and P3-only distillation losses for PC-HBM-Lite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .losses import zero_like_loss


def build_pc_confidence(
    aux: Mapping[str, Any],
) -> torch.Tensor:
    """Build ``2|p-0.5| * (1-Q+Q*C_mem)`` exactly once."""

    if not isinstance(aux, Mapping):
        raise TypeError("Teacher aux must be a mapping")
    probability = aux.get("p_final")
    pc = aux.get("pc_hbm")
    if not torch.is_tensor(probability):
        raise KeyError("Teacher aux['p_final'] probability is required")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("p_final must be [B,1,H,W]")
    if aux.get("forward_mode") != "teacher_pseudo":
        raise RuntimeError("pseudo confidence requires teacher_pseudo mode")
    if aux.get("pc_active") is not True:
        raise RuntimeError(
            "Teacher PC-HBM-Lite path is inactive: "
            f"{aux.get('fallback_reason')}"
        )
    if not isinstance(pc, Mapping):
        raise KeyError("Teacher aux['pc_hbm'] is required")
    query_map = pc.get("query_mask_map")
    memory_map = pc.get("memory_confidence_map")
    if not torch.is_tensor(query_map) or not torch.is_tensor(memory_map):
        raise KeyError(
            "query_mask_map and memory_confidence_map are required"
        )
    _validate_single_channel_map(query_map, "query_mask_map", probability.shape[0])
    _validate_single_channel_map(
        memory_map, "memory_confidence_map", probability.shape[0]
    )

    output_size = probability.shape[-2:]
    query = F.interpolate(
        query_map.detach().float(),
        size=output_size,
        mode="nearest",
    ).clamp(0.0, 1.0)
    memory_confidence = F.interpolate(
        memory_map.detach().float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    probability_fp32 = probability.detach().float().clamp(0.0, 1.0)
    prediction_confidence = 2.0 * torch.abs(probability_fp32 - 0.5)
    modifier = 1.0 - query + query * memory_confidence
    confidence = (prediction_confidence * modifier).clamp(0.0, 1.0)
    if not bool(torch.isfinite(confidence).all()):
        raise FloatingPointError("Teacher pseudo confidence contains NaN/Inf")
    return confidence.to(dtype=probability.dtype)


def prepare_pseudo_targets(
    teacher_aux: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Clone only the soft mask, confidence, and corrected P3 target."""

    probability = teacher_aux.get("p_final")
    if not torch.is_tensor(probability):
        raise KeyError("Teacher aux['p_final'] is required")
    confidence = build_pc_confidence(teacher_aux)
    distill = teacher_aux.get("distill_features")
    corrected_p3 = (
        distill.get("p3_corr") if isinstance(distill, Mapping) else None
    )
    if not torch.is_tensor(corrected_p3):
        raise KeyError(
            "Teacher distill_features['p3_corr'] is required"
        )
    return {
        "p_soft": probability.detach().clone(),
        "confidence": confidence.detach().clone(),
        "p3_corr": corrected_p3.detach().clone(),
    }


def weighted_structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Confidence-weighted F3Net loss over foreground and background."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must be [B,1,H,W]")
    target = _resize_single_channel(target, logits, "target")
    confidence = _resize_single_channel(
        confidence, logits, "confidence"
    ).clamp_min(0.0)
    target = target.detach()
    confidence = confidence.detach()
    if not bool((confidence > 0).any()):
        return zero_like_loss(logits)

    structure_weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    weight = structure_weight * confidence
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (bce * weight).sum(dim=(2, 3)) / (
        weight.sum(dim=(2, 3)) + eps
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target * weight).sum(dim=(2, 3))
    union = ((probability + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (
        union - intersection + 1.0
    )
    return (weighted_bce + weighted_iou).mean()


def confidence_weighted_feature_cosine_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Confidence-weighted per-pixel cosine distance in explicit FP32."""

    if student_feature.ndim != 4 or teacher_feature.ndim != 4:
        raise ValueError("P3 features must be [B,C,H,W]")
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "Student and Teacher P3 shapes must match, got "
            f"{tuple(student_feature.shape)} and {tuple(teacher_feature.shape)}"
        )
    weight = _resize_single_channel(
        confidence,
        student_feature[:, :1],
        "confidence",
    ).detach().float()
    if not bool((weight > 0).any()):
        return zero_like_loss(student_feature)
    student = F.normalize(student_feature.float(), dim=1, eps=eps)
    teacher = F.normalize(
        teacher_feature.detach().float(), dim=1, eps=eps
    )
    distance = 1.0 - (student * teacher).sum(dim=1, keepdim=True)
    return (distance * weight).sum() / (weight.sum() + eps)


def pc_unlabeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any],
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    config: Any,
    *,
    teacher_features: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the raw Student with soft masks and corrected P3 only."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError(
            "Student outputs must be (m4, m3, m2, z_main, global_logit)"
        )
    if not isinstance(aux, Mapping) or aux.get("forward_mode") != "off":
        raise RuntimeError("Teacher-only TS requires the Student in off mode")
    if aux.get("pc_active") is not False:
        raise RuntimeError("Raw Student must not activate PC-HBM-Lite")
    z_main = aux.get("z_main")
    if not torch.is_tensor(z_main):
        raise KeyError("Student aux['z_main'] is required")
    if (
        outputs[3].shape != z_main.shape
        or outputs[3].device != z_main.device
        or outputs[3].dtype != z_main.dtype
    ):
        raise ValueError("outputs[3] and aux['z_main'] must match")

    m4, m3, m2, _, global_logit = outputs
    main = weighted_structure_loss(z_main, pseudo, confidence)
    side = (
        0.30 * weighted_structure_loss(m2, pseudo, confidence)
        + 0.20 * weighted_structure_loss(m3, pseudo, confidence)
        + 0.10 * weighted_structure_loss(m4, pseudo, confidence)
        + 0.10 * weighted_structure_loss(global_logit, pseudo, confidence)
    )
    if not isinstance(teacher_features, Mapping):
        raise TypeError("teacher_features must be a mapping")
    student_features = aux.get("features")
    if not isinstance(student_features, Mapping):
        raise KeyError("Student aux['features'] is required")
    student_p3 = student_features.get("p3")
    teacher_p3 = teacher_features.get("p3_corr")
    if not torch.is_tensor(student_p3) or not torch.is_tensor(teacher_p3):
        raise KeyError("Student p3 and Teacher p3_corr are required")
    feature = confidence_weighted_feature_cosine_loss(
        student_p3, teacher_p3, confidence
    )

    feature_weight = float(
        getattr(config, "feature_distill_p3_weight", 0.05)
    )
    unlabeled_weight = float(getattr(config, "lambda_u", 1.0))
    if feature_weight < 0.0 or unlabeled_weight < 0.0:
        raise ValueError("unlabeled loss weights must be non-negative")
    feature_weighted = feature_weight * feature
    total = unlabeled_weight * (main + side + feature_weighted)
    positive = confidence.detach() > 0
    metrics = {
        "L_u_main": main.detach(),
        "L_u_side": side.detach(),
        "L_u_feat_p3": feature.detach(),
        "L_u_feat_p3_weighted": feature_weighted.detach(),
        "L_u_total": total.detach(),
        "pseudo_conf_mean": confidence.detach().float().mean(),
        "pseudo_coverage": positive.float().mean(),
    }
    return total, metrics


def _resize_single_channel(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must be [B,1,H,W]")
    if value.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} batch size must match reference")
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(
            value.float(),
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return value.to(device=reference.device, dtype=reference.dtype)


def _validate_single_channel_map(
    value: torch.Tensor,
    name: str,
    batch_size: int,
) -> None:
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must be [B,1,H,W]")
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match p_final")


__all__ = [
    "build_pc_confidence",
    "confidence_weighted_feature_cosine_loss",
    "pc_unlabeled_loss",
    "prepare_pseudo_targets",
    "weighted_structure_loss",
]
