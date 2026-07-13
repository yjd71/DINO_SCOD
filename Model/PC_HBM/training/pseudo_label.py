"""Online PC-HBM pseudo targets and Student-core weighted objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .losses import zero_like_loss


def build_pc_confidence(aux: Mapping[str, Any], *, strict: bool = True) -> torch.Tensor:
    """Multiply the five prescribed structural confidence factors.

    The input ``p_final`` is already a probability.  It is deliberately not
    passed through another sigmoid.
    """

    p_final = aux.get("p_final")
    z_main = aux.get("z_main")
    pc = aux.get("pc_hbm", {}) or {}
    mixture = aux.get("mixture", {}) or {}
    missing = []
    if not torch.is_tensor(p_final):
        missing.append("p_final")
    if not torch.is_tensor(z_main):
        missing.append("z_main")
    c23 = _nested_get(pc, "C23_map")
    pi = mixture.get("pi")
    route_entropy = _nested_get(pc, "route_entropy_norm")
    if not torch.is_tensor(c23):
        missing.append("pc_hbm.C23_map")
    if not torch.is_tensor(pi):
        missing.append("mixture.pi")
    if not torch.is_tensor(route_entropy):
        missing.append("pc_hbm.route_entropy_norm")
    if missing:
        if strict:
            raise KeyError(f"Teacher pseudo confidence is missing: {missing}")
        if not torch.is_tensor(p_final):
            raise KeyError("p_final is required even when strict=False")
        return torch.zeros_like(p_final).detach()

    p_final = p_final.detach().clamp(0.0, 1.0)
    p_main = torch.sigmoid(z_main.detach())
    if p_main.shape[-2:] != p_final.shape[-2:]:
        p_main = F.interpolate(p_main, size=p_final.shape[-2:], mode="bilinear", align_corners=False)

    probability_confidence = (2.0 * (p_final - 0.5).abs()).clamp(0.0, 1.0)
    main_final_agreement = (1.0 - (p_final - p_main).abs()).clamp(0.0, 1.0)
    c23 = F.interpolate(
        c23.detach().to(dtype=p_final.dtype),
        size=p_final.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    pi = pi.detach().to(dtype=p_final.dtype).clamp_min(1.0e-6)
    if pi.size(1) != 4:
        raise ValueError(f"mixture.pi must have four branches, got {tuple(pi.shape)}")
    mixture_entropy = -(pi * pi.log()).sum(dim=1, keepdim=True) / math.log(4.0)
    if mixture_entropy.shape[-2:] != p_final.shape[-2:]:
        mixture_entropy = F.interpolate(
            mixture_entropy,
            size=p_final.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    mixture_entropy = mixture_entropy.clamp(0.0, 1.0)

    route_entropy = route_entropy.detach().to(device=p_final.device, dtype=p_final.dtype)
    if route_entropy.ndim == 0:
        route_entropy = route_entropy.expand(p_final.size(0))
    if route_entropy.size(0) != p_final.size(0):
        raise ValueError("route_entropy_norm batch dimension must match p_final")
    route_entropy = route_entropy.reshape(route_entropy.size(0), -1).mean(dim=1)
    route_confidence = (1.0 - route_entropy).view(-1, 1, 1, 1).clamp(0.0, 1.0)

    confidence = (
        probability_confidence
        * main_final_agreement
        * (1.0 - c23)
        * (1.0 - mixture_entropy)
        * route_confidence
    )
    return confidence.clamp(0.0, 1.0).detach()


def prepare_pseudo_targets(
    teacher_aux: Mapping[str, Any],
    config: Any,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Clone soft probability, confidence and optional Teacher features."""

    p_final = teacher_aux.get("p_final")
    if not torch.is_tensor(p_final):
        raise KeyError("teacher_aux['p_final'] probability is required")
    p_soft = p_final.detach().clone()
    confidence = build_pc_confidence(teacher_aux, strict=strict).detach().clone()
    distill_features = teacher_aux.get("distill_features")
    cloned_features = None
    if distill_features is not None:
        if not isinstance(distill_features, Mapping):
            raise TypeError("teacher_aux['distill_features'] must be a mapping")
        cloned_features = {}
        for name in ("p3_corr", "p2_refined"):
            value = distill_features.get(name)
            if not torch.is_tensor(value):
                raise KeyError(f"teacher_aux['distill_features']['{name}'] is required")
            cloned_features[name] = value.detach().clone()

    return {
        "p_soft": p_soft,
        "confidence": confidence,
        "distill_features": cloned_features,
    }


def weighted_structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Confidence-weighted F3Net loss without dropping background samples."""

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if confidence.ndim == 3:
        confidence = confidence.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
    if confidence.shape[-2:] != logits.shape[-2:]:
        confidence = F.interpolate(
            confidence.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
    target = target.detach().to(device=logits.device, dtype=logits.dtype)
    confidence = confidence.detach().to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    if target.shape != logits.shape or confidence.shape != logits.shape:
        raise ValueError(
            "logits, target and confidence must resolve to the same [B,1,H,W] shape, "
            f"got {tuple(logits.shape)}, {tuple(target.shape)}, {tuple(confidence.shape)}"
        )

    structure_weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    weight = structure_weight * confidence
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (bce * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + eps)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * weight).sum(dim=(2, 3))
    union = ((probability + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def confidence_weighted_feature_cosine_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Confidence-weighted per-pixel normalized cosine distance."""

    if not all(torch.is_tensor(value) for value in (student_feature, teacher_feature, confidence)):
        raise TypeError("student_feature, teacher_feature and confidence must be tensors")
    if student_feature.ndim != 4 or teacher_feature.ndim != 4:
        raise ValueError("Student and Teacher features must be [B,C,H,W]")
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "Student/Teacher distillation feature shapes must match, got "
            f"{tuple(student_feature.shape)} and {tuple(teacher_feature.shape)}"
        )
    if confidence.ndim != 4 or confidence.shape[0] != student_feature.shape[0]:
        raise ValueError("confidence must be [B,1,H,W] with the same batch size")
    if confidence.shape[1] != 1:
        raise ValueError("confidence must have exactly one channel")

    target = teacher_feature.detach().to(device=student_feature.device)
    student_norm = F.normalize(student_feature.float(), dim=1, eps=eps)
    teacher_norm = F.normalize(target.float(), dim=1, eps=eps)
    distance = 1.0 - (student_norm * teacher_norm).sum(dim=1, keepdim=True)
    weight = F.interpolate(
        confidence.detach().float().to(student_feature.device),
        size=student_feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp_(0.0, 1.0)
    return (distance * weight).sum() / weight.sum().clamp_min(eps)


def pc_unlabeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any],
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    epoch: int,
    config: Any,
    *,
    teacher_features: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise raw Student logits and optional P3/P2 raw features."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Student outputs must be (m4, m3, m2, z_main, global_logit)")
    z_student = aux.get("z_main")
    if not torch.is_tensor(z_student):
        raise KeyError("Student aux['z_main'] logits are required")
    if aux.get("mixture_skipped") is False:
        raise RuntimeError("Unlabeled Student supervision requires off/student_core mode")

    m4, m3, m2, output_z_main, global_logit = outputs
    if not torch.is_tensor(output_z_main):
        raise TypeError("outputs[3] must be the Student main logit tensor")
    # ``find_unused_parameters=True`` routes every tensor in the DDP return
    # tree through _DDPSink.  When the same logical z_main occurs in both the
    # output tuple and aux mapping, PyTorch independently clones those two
    # occurrences, so storage identity/data_ptr is not a valid public
    # contract.  Keep aux['z_main'] as the supervised tensor and validate the
    # stable tensor metadata shared by wrapped and unwrapped forwards.
    if (
        output_z_main.shape != z_student.shape
        or output_z_main.device != z_student.device
        or output_z_main.dtype != z_student.dtype
    ):
        raise ValueError(
            "outputs[3] and aux['z_main'] must have matching shape, device, and dtype"
        )

    p_soft = pseudo.detach().clone()
    confidence = confidence.detach().clone()

    l_soft = weighted_structure_loss(z_student, p_soft, confidence)

    l_side = (
        0.30 * weighted_structure_loss(m2, p_soft, confidence)
        + 0.20 * weighted_structure_loss(m3, p_soft, confidence)
        + 0.10 * weighted_structure_loss(m4, p_soft, confidence)
        + 0.10 * weighted_structure_loss(global_logit, p_soft, confidence)
    )

    zero = zero_like_loss(z_student)
    l_feat_p3 = zero
    l_feat_p2 = zero
    if teacher_features is not None:
        if not isinstance(teacher_features, Mapping):
            raise TypeError("teacher_features must be a mapping")
        student_features = aux.get("features")
        if not isinstance(student_features, Mapping):
            raise KeyError("Student aux['features'] is required for feature distillation")
        p3_student = student_features.get("p3")
        p2_student = student_features.get("p2")
        p3_teacher = teacher_features.get("p3_corr")
        p2_teacher = teacher_features.get("p2_refined")
        if not all(torch.is_tensor(value) for value in (p3_student, p2_student, p3_teacher, p2_teacher)):
            raise KeyError("P3/P2 Student and corrected Teacher features are required")
        l_feat_p3 = confidence_weighted_feature_cosine_loss(
            p3_student, p3_teacher, confidence
        )
        l_feat_p2 = confidence_weighted_feature_cosine_loss(
            p2_student, p2_teacher, confidence
        )

    p3_weight = float(getattr(config, "feature_distill_p3_weight", 0.05))
    p2_weight = float(getattr(config, "feature_distill_p2_weight", 0.10))
    l_feature = p3_weight * l_feat_p3 + p2_weight * l_feat_p2
    unscaled = l_soft + l_side + l_feature
    total = float(getattr(config, "lambda_u", 1.0)) * unscaled
    positive_confidence = confidence > 0
    log = {
        "L_u_soft": l_soft.detach(),
        "L_u_side": l_side.detach(),
        "L_u_feat_p3": l_feat_p3.detach(),
        "L_u_feat_p2": l_feat_p2.detach(),
        "L_u_feature": l_feature.detach(),
        "pseudo_conf_mean": confidence.mean().detach(),
        "pseudo_conf_max": confidence.max().detach(),
        "pseudo_conf_positive_fraction": positive_confidence.to(z_student.dtype).mean().detach(),
        "pseudo_conf_positive_count": positive_confidence.sum().detach(),
        "loss_unlabeled": total.detach(),
    }
    return total, log


def compute_pc_hbm_unlabeled_loss(
    student_aux: Mapping[str, Any],
    pseudo_prob: torch.Tensor,
    confidence: torch.Tensor,
    config: Any,
    epoch: int,
    outputs: Sequence[torch.Tensor] | None = None,
    *,
    teacher_features: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reference-name wrapper; ``outputs`` is required for side supervision."""

    if outputs is None:
        z_main = student_aux.get("z_main")
        if not torch.is_tensor(z_main):
            raise KeyError("student_aux['z_main'] is required")
        zero = zero_like_loss(z_main).view(1, 1, 1, 1).expand_as(z_main)
        outputs = (zero, zero, zero, z_main, zero)
    return pc_unlabeled_loss(
        outputs,
        student_aux,
        pseudo_prob,
        confidence,
        epoch,
        config,
        teacher_features=teacher_features,
    )


structure_aware_confidence = build_pc_confidence


def _nested_get(mapping, key, default=None):
    if key in mapping:
        return mapping[key]
    for value in mapping.values():
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return default


__all__ = [
    "build_pc_confidence",
    "confidence_weighted_feature_cosine_loss",
    "compute_pc_hbm_unlabeled_loss",
    "pc_unlabeled_loss",
    "prepare_pseudo_targets",
    "structure_aware_confidence",
    "weighted_structure_loss",
]
