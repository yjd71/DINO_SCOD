"""Stage-aware labeled PC-HBM losses for the DINO decoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .branch_oracle import BRANCH_NAMES, oracle_distribution
from .supervision import (
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    build_geometry_target,
    build_gt_boundary,
    build_need_correction_map,
    build_region_label_map,
    gather_by_boundary_indices,
    normalize_boundary_indices,
)


def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero on ``reference``'s device/dtype."""

    return reference.sum() * 0.0


def probability_bce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute BCE on sigmoid probabilities without AMP's unsafe CUDA kernel.

    PC-HBM boundary and gate heads expose probabilities rather than raw logits.
    CUDA autocast therefore cannot use ``binary_cross_entropy`` directly.  Keep
    the probability contract, but perform the loss itself in explicit FP32.
    """

    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not prediction.is_floating_point():
        raise TypeError("prediction must use a floating-point dtype")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction!r}")

    eps = 1.0e-4 if prediction.dtype in (torch.float16, torch.bfloat16) else 1.0e-6
    device_type = prediction.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        probability_fp32 = prediction.float().clamp(eps, 1.0 - eps)
        target_fp32 = target.detach().to(device=prediction.device, dtype=torch.float32)
        target_fp32 = target_fp32.clamp(0.0, 1.0)
        return F.binary_cross_entropy(probability_fp32, target_fp32, reduction=reduction)


def structure_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """F3Net/RSBL weighted BCE plus weighted IoU for matching spatial sizes."""

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    target = target.to(device=logits.device, dtype=logits.dtype)
    structure_weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (structure_weight * bce).sum(dim=(2, 3)) / structure_weight.sum(
        dim=(2, 3)
    ).clamp_min(eps)
    probability = torch.sigmoid(logits)
    intersection = ((probability * target) * structure_weight).sum(dim=(2, 3))
    union = ((probability + target) * structure_weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def base_structure_loss(outputs: Sequence[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    """Preserve supervision of the original five Decoder logits."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Decoder outputs must be (m4, m3, m2, z_main, global_logit)")
    m4, m3, m2, z_main, global_logit = outputs
    total = zero_like_loss(z_main)
    for logit in (z_main, m2, m3, m4, global_logit):
        total = total + structure_loss(logit, gt)
    return total


def decoder_base_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Supervise the canonical original Decoder's five logits."""

    del aux, config
    return base_structure_loss(outputs, gt)


def pc_mode_for_epoch(epoch: int, config: Any) -> str:
    """Return the 1-based Base-training mode using the shared config."""

    if hasattr(config, "pc_mode_for_epoch"):
        return str(config.pc_mode_for_epoch(int(epoch)))
    if int(epoch) < int(getattr(config, "parent_start_epoch", 6)):
        return "off"
    if int(epoch) < int(getattr(config, "full_pc_start_epoch", 11)):
        return "parent_only"
    return "full"


def pc_injection_strength(epoch: int | None, config: Any) -> float:
    """Return 1/3, 2/3, 1 for the first three full-PC epochs."""

    if epoch is None:
        return 1.0
    if hasattr(config, "injection_scale"):
        return float(config.injection_scale(int(epoch)))
    start = int(getattr(config, "full_pc_start_epoch", 11))
    ramp_epochs = max(1, int(getattr(config, "pc_injection_ramp_epochs", 3)))
    return min(1.0, max(0.0, (int(epoch) - start + 1) / float(ramp_epochs)))


def pc_hbm_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    epoch: int | None,
    config: Any,
    *,
    pc_mode: str | None = None,
    strict: bool = True,
    _include_base: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the mode-specific labeled objective and detached log scalars.

    ``off`` uses only the original five-output loss. ``parent_only`` adds
    exactly parent-region and B3 supervision. ``full`` enables every PC-HBM
    term, with the memory group ramped over full epochs 11/12/13.
    """

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Decoder outputs must be (m4, m3, m2, z_main, global_logit)")
    reference = outputs[3]
    l_base = (
        decoder_base_loss(outputs, aux, gt, config)
        if _include_base
        else zero_like_loss(reference)
    )
    mode = _resolve_mode(pc_mode, aux, epoch, config)
    _validate_pc_activation(aux, mode, strict)
    terms = _zero_terms(reference)
    if _include_base:
        terms["L_base"] = l_base
    if mode == "off":
        if not _include_base:
            raise ValueError("PC-only labeled loss does not support pc_mode='off'")
        terms["loss_labeled"] = l_base
        return l_base, _detach_terms(terms)

    pc = dict((aux or {}).get("pc_hbm", {}) or {})
    p2 = dict((aux or {}).get("p2_bra", {}) or {})
    p1 = dict((aux or {}).get("p1_pra", {}) or {})
    mixture = dict((aux or {}).get("mixture", {}) or {})

    terms["L_parent"] = _parent_loss(pc, gt, reference)
    terms["L_B3"] = _single_boundary_loss(_nested_get(pc, "B3"), gt, reference)
    if mode == "parent_only":
        total = (
            l_base
            + float(getattr(config, "lambda_mem", 0.20)) * terms["L_parent"]
            + float(getattr(config, "lambda_boundary", 0.10)) * terms["L_B3"]
        )
        terms["L_mem"] = terms["L_parent"]
        terms["L_boundary"] = terms["L_B3"]
        terms["loss_labeled"] = total
        detached = _detach_terms(terms)
        if not _include_base:
            detached.pop("L_base", None)
        return total, detached

    z_final = (aux or {}).get("z_final")
    if z_final is None:
        if strict:
            raise RuntimeError("full PC-HBM loss requires aux['z_final'] logits")
        z_final = reference
    final_for_loss = z_final
    if final_for_loss.shape[-2:] != gt.shape[-2:]:
        final_for_loss = F.interpolate(
            final_for_loss, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    terms["L_final"] = structure_loss(final_for_loss, gt)
    terms["L_child"] = _child_loss(pc, gt, reference)
    terms["L_geometry"] = _geometry_loss(pc, gt, reference)
    terms["L_gate"] = _gate_loss(pc, aux or {}, gt, reference)
    terms["L_mem"] = (
        terms["L_parent"] + terms["L_child"] + terms["L_geometry"] + terms["L_gate"]
    )
    terms["L_boundary"] = _boundary_loss(pc, p2, p1, gt, reference)
    terms["L_mix_oracle"], oracle = _mix_oracle_loss(mixture, gt, config, reference)
    terms["L_branch"] = _branch_loss(mixture, gt, reference)
    terms["L_quality"] = _quality_loss(mixture, oracle, reference)
    terms["L_usage"] = _usage_loss(mixture, gt, reference)
    terms["L_reg"] = _regularization_loss(mixture, reference)
    strength = pc_injection_strength(epoch, config)
    terms["pc_strength"] = reference.new_tensor(strength)

    total = (
        l_base
        + float(getattr(config, "lambda_final", 1.0)) * terms["L_final"]
        + strength * float(getattr(config, "lambda_mem", 0.20)) * terms["L_mem"]
        + float(getattr(config, "lambda_boundary", 0.10)) * terms["L_boundary"]
        + float(getattr(config, "lambda_mix_oracle", 0.10)) * terms["L_mix_oracle"]
        + float(getattr(config, "lambda_branch", 0.10)) * terms["L_branch"]
        + float(getattr(config, "lambda_quality", 0.025)) * terms["L_quality"]
        + float(getattr(config, "lambda_usage", 0.01)) * terms["L_usage"]
        + float(getattr(config, "lambda_reg", 0.02)) * terms["L_reg"]
    )
    terms["loss_labeled"] = total
    _append_means(terms, pc, mixture, reference)
    detached = _detach_terms(terms)
    if not _include_base:
        detached.pop("L_base", None)
    return total, detached


def pc_hbm_pc_only_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    epoch: int | None,
    config: Any,
    *,
    pc_mode: str | None = None,
    strict: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only PC-HBM branches while keeping every baseline output unsupervised.

    The objective intentionally excludes ``L_base``.  ``parent_only`` therefore
    contains exactly the weighted parent and B3 terms, while ``full`` reuses the
    established PC-HBM objective starting at ``L_final``.
    """

    return pc_hbm_labeled_loss(
        outputs,
        aux,
        gt,
        epoch,
        config,
        pc_mode=pc_mode,
        strict=strict,
        _include_base=False,
    )


def compute_pc_hbm_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    config: Any,
    epoch: int | None = None,
    *,
    pc_mode: str | None = None,
    strict: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compatibility alias with the reference repository argument order."""

    return pc_hbm_labeled_loss(
        outputs, aux, gt, epoch, config, pc_mode=pc_mode, strict=strict
    )


def _resolve_mode(pc_mode, aux, epoch, config) -> str:
    mode = pc_mode
    if mode is None and aux is not None:
        mode = aux.get("forward_mode", aux.get("pc_mode"))
    if mode is None:
        mode = pc_mode_for_epoch(epoch, config) if epoch is not None else "full"
    mode = str(mode)
    if mode in {"teacher_pseudo", "student_core"}:
        mode = "full"
    if mode not in {"off", "parent_only", "full"}:
        raise ValueError(f"Unsupported labeled pc_mode: {mode}")
    return mode


def _validate_pc_activation(aux, mode: str, strict: bool) -> None:
    if mode == "off" or not strict:
        return
    if aux is None:
        raise RuntimeError(f"{mode} training requires PC-HBM auxiliary outputs")
    fallback = aux.get("fallback_reason")
    if fallback:
        raise RuntimeError(f"PC-HBM training cannot use baseline fallback: {fallback}")
    if aux.get("pc_active") is False:
        raise RuntimeError("PC-HBM training requested but aux['pc_active'] is false")


def _nested_get(mapping: Mapping[str, Any], key: str, default=None):
    if key in mapping:
        return mapping[key]
    for value in mapping.values():
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return default


def _boundary_indices(pc: Mapping[str, Any]) -> dict[str, torch.Tensor] | None:
    value = _nested_get(pc, "boundary_indices3")
    batch_ids = _nested_get(pc, "batch_ids3")
    flat_indices = _nested_get(pc, "flat_indices3")
    return normalize_boundary_indices(value, batch_ids=batch_ids, flat_indices=flat_indices)


def _pc_size(pc: Mapping[str, Any]) -> tuple[int, int]:
    for key in ("B3", "valid3_map", "M_pc_map", "C23_map", "gate_pc_map"):
        value = _nested_get(pc, key)
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[-2]), int(value.shape[-1])
    return 28, 28


def _query_valid(pc: Mapping[str, Any], count: int, reference: torch.Tensor) -> torch.Tensor:
    valid = _nested_get(pc, "query_valid", _nested_get(pc, "valid_token"))
    if valid is None:
        parent_valid = _nested_get(pc, "top_parent_valid")
        if torch.is_tensor(parent_valid) and parent_valid.ndim >= 2:
            valid = parent_valid.any(dim=-1)
    if valid is None:
        return torch.ones(count, device=reference.device, dtype=torch.bool)
    valid = torch.as_tensor(valid, device=reference.device, dtype=torch.bool).flatten()
    if valid.numel() != count:
        raise ValueError(f"query_valid has {valid.numel()} entries, expected {count}")
    return valid


def _parent_loss(pc, gt, reference):
    probabilities = _nested_get(pc, "P3_group")
    indices = _boundary_indices(pc)
    if not torch.is_tensor(probabilities) or indices is None or not probabilities.numel():
        return zero_like_loss(probabilities if torch.is_tensor(probabilities) else reference)
    if probabilities.ndim != 2 or probabilities.size(1) != 4:
        raise ValueError(f"P3_group must be [M,4], got {tuple(probabilities.shape)}")
    region_map = build_region_label_map(gt, _pc_size(pc))
    targets = gather_by_boundary_indices(region_map, indices).long().clamp(0, 3)
    if targets.numel() != probabilities.size(0):
        raise ValueError("P3_group and boundary token count differ")
    probabilities = probabilities.clamp_min(1.0e-6)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    per_token = F.nll_loss(probabilities.log(), targets, reduction="none")
    valid = _query_valid(pc, probabilities.size(0), probabilities)
    if not bool(valid.any()):
        return zero_like_loss(probabilities)
    return (per_token * valid.to(per_token.dtype)).sum() / valid.sum().clamp_min(1)


def _child_loss(pc, gt, reference):
    logits = _nested_get(pc, "S_child")
    parent_regions = _nested_get(pc, "top_parent_region_ids")
    indices = _boundary_indices(pc)
    if (
        not torch.is_tensor(logits)
        or not torch.is_tensor(parent_regions)
        or indices is None
        or not logits.numel()
    ):
        return zero_like_loss(logits if torch.is_tensor(logits) else reference)
    if logits.shape != parent_regions.shape:
        raise ValueError("S_child and top_parent_region_ids must have the same [M,K] shape")
    region_map = build_region_label_map(gt, _pc_size(pc))
    targets = gather_by_boundary_indices(region_map, indices).long()
    support = (parent_regions == targets[:, None]).to(logits.dtype)
    valid = parent_regions.ge(0)
    explicit_valid = _nested_get(pc, "top_parent_valid")
    if torch.is_tensor(explicit_valid):
        valid = valid & explicit_valid.to(device=valid.device, dtype=torch.bool)
    hard_negative = (
        ((targets[:, None] == REGION_FG_BOUNDARY) & (parent_regions == REGION_BG_NEAR))
        | ((targets[:, None] == REGION_BG_NEAR) & (parent_regions == REGION_FG_BOUNDARY))
    )
    weights = valid.to(logits.dtype) * (1.0 + hard_negative.to(logits.dtype))
    if not bool(valid.any()):
        return zero_like_loss(logits)
    loss = F.binary_cross_entropy_with_logits(logits, support, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _geometry_loss(pc, gt, reference):
    parent_geometry = _nested_get(pc, "G_attn")
    offset = _nested_get(pc, "O_pc_token")
    indices = _boundary_indices(pc)
    if not torch.is_tensor(parent_geometry) or indices is None or not parent_geometry.numel():
        return zero_like_loss(parent_geometry if torch.is_tensor(parent_geometry) else reference)
    if parent_geometry.ndim != 2 or parent_geometry.size(1) < 3:
        raise ValueError(f"G_attn must be [M,6], got {tuple(parent_geometry.shape)}")
    target_geometry = build_geometry_target(gt, _pc_size(pc))
    sdf_target = gather_by_boundary_indices(target_geometry["sdf"], indices).flatten()
    normal_target = gather_by_boundary_indices(target_geometry["normal"], indices)
    offset_target = gather_by_boundary_indices(target_geometry["offset"], indices)
    valid = _query_valid(pc, parent_geometry.size(0), parent_geometry).to(parent_geometry.dtype)
    denominator = valid.sum().clamp_min(1.0)
    sdf = ((parent_geometry[:, 0] - sdf_target).abs() * valid).sum() / denominator
    normal = (
        (1.0 - F.cosine_similarity(parent_geometry[:, 1:3], normal_target, dim=-1)) * valid
    ).sum() / denominator
    if torch.is_tensor(offset) and offset.shape == offset_target.shape:
        offset_error = (offset - offset_target).abs().mean(dim=-1)
        offset_loss = (offset_error * valid).sum() / denominator
    else:
        offset_loss = zero_like_loss(parent_geometry)
    return sdf + 0.5 * normal + 0.5 * offset_loss


def _gate_loss(pc, aux, gt, reference):
    gate = _nested_get(pc, "gate_pc_token")
    contradiction = _nested_get(pc, "C23_token")
    indices = _boundary_indices(pc)
    z_main = aux.get("z_main")
    if gate is None and indices is not None:
        gate_map = _nested_get(pc, "gate_pc_map")
        if torch.is_tensor(gate_map):
            gate = gather_by_boundary_indices(gate_map, indices)
    if contradiction is None and indices is not None:
        contradiction_map = _nested_get(pc, "C23_map")
        if torch.is_tensor(contradiction_map):
            contradiction = gather_by_boundary_indices(contradiction_map, indices)
    if (
        not torch.is_tensor(gate)
        or not torch.is_tensor(contradiction)
        or indices is None
        or not torch.is_tensor(z_main)
        or not gate.numel()
    ):
        return zero_like_loss(gate if torch.is_tensor(gate) else reference)
    target_map = build_need_correction_map(z_main, gt, _pc_size(pc), threshold=0.25)
    target = gather_by_boundary_indices(target_map, indices).reshape_as(gate)
    target = target * (1.0 - contradiction.detach()).reshape_as(target).clamp(0.0, 1.0)
    valid = _query_valid(pc, gate.size(0), gate).to(gate.dtype).view(-1, *([1] * (gate.ndim - 1)))
    per_element = probability_bce(gate, target, reduction="none")
    expanded_valid = valid.bool().expand_as(per_element)
    if not bool(expanded_valid.bool().any()):
        return zero_like_loss(per_element)
    expanded_valid = expanded_valid.to(per_element.dtype)
    return (per_element * expanded_valid).sum() / expanded_valid.sum().clamp_min(1.0)


def _single_boundary_loss(prediction, gt, reference, *, valid_mask=None):
    if not torch.is_tensor(prediction) or not prediction.numel():
        return zero_like_loss(prediction if torch.is_tensor(prediction) else reference)
    target = build_gt_boundary(gt, tuple(prediction.shape[-2:])).to(
        device=prediction.device, dtype=prediction.dtype
    )
    if target.shape != prediction.shape:
        if target.shape[0] != prediction.shape[0] or target.shape[1] != prediction.shape[1]:
            raise ValueError(
                f"boundary target shape {tuple(target.shape)} is incompatible with "
                f"prediction shape {tuple(prediction.shape)}"
            )

    per_element = probability_bce(prediction, target, reduction="none")
    if valid_mask is None:
        return per_element.mean()
    if not torch.is_tensor(valid_mask):
        raise TypeError("valid_mask must be a torch.Tensor or None")

    mask = valid_mask.detach()
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != prediction.ndim:
        raise ValueError(
            f"valid_mask must have {prediction.ndim - 1} or {prediction.ndim} dimensions, "
            f"got {mask.ndim}"
        )
    if mask.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"valid_mask batch size {mask.shape[0]} does not match prediction "
            f"batch size {prediction.shape[0]}"
        )
    mask = mask.to(device=prediction.device, dtype=torch.float32)
    if tuple(mask.shape[-2:]) != tuple(prediction.shape[-2:]):
        mask = F.interpolate(mask, size=prediction.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    elif mask.shape[1] != prediction.shape[1]:
        raise ValueError(
            f"valid_mask channels {mask.shape[1]} do not match prediction "
            f"channels {prediction.shape[1]}"
        )
    mask = mask.clamp(0.0, 1.0).to(per_element.dtype)
    denominator = mask.sum()
    if not bool(denominator > 0):
        return zero_like_loss(per_element)
    return (per_element * mask).sum() / denominator


def _boundary_loss(pc, p2, p1, gt, reference):
    predictions = (
        (_nested_get(pc, "B3"), None),
        (_nested_get(p2, "B2"), None),
        (_nested_get(p2, "B2_refined_map"), _nested_get(p2, "valid2_map")),
        (_nested_get(p1, "B1"), None),
    )
    total = zero_like_loss(reference)
    count = 0
    for prediction, valid_mask in predictions:
        if torch.is_tensor(prediction) and prediction.numel():
            total = total + _single_boundary_loss(
                prediction, gt, reference, valid_mask=valid_mask
            )
            count += 1
    return total if count else zero_like_loss(reference)


def _mix_oracle_loss(mixture, gt, config, reference):
    if "pi" not in mixture or any(name not in mixture for name in BRANCH_NAMES):
        return zero_like_loss(reference), {}
    oracle = oracle_distribution(
        mixture,
        gt,
        tau=float(getattr(config, "mix_oracle_temperature", 0.5)),
        min_improvement=float(getattr(config, "mix_oracle_min_improvement", 0.03)),
    )
    pi = mixture["pi"].clamp_min(1.0e-6)
    target = oracle["target_mix"].detach()
    mask = oracle["oracle_mask"].detach()
    kl = (target * (target.clamp_min(1.0e-6).log() - pi.log())).sum(dim=1, keepdim=True)
    return (kl * mask).sum() / mask.sum().clamp_min(1.0), oracle


def _branch_loss(mixture, gt, reference):
    if any(name not in mixture for name in BRANCH_NAMES):
        return zero_like_loss(reference)
    return sum(structure_loss(mixture[name], gt) for name in BRANCH_NAMES) / float(
        len(BRANCH_NAMES)
    )


def _quality_loss(mixture, oracle, reference):
    quality = mixture.get("branch_quality", mixture.get("pred_gain"))
    if not torch.is_tensor(quality) or not oracle:
        return zero_like_loss(reference)
    target_gain = (oracle["pixel_error"][:, 0:1] - oracle["pixel_error"]).detach()
    if quality.shape != target_gain.shape:
        raise ValueError(
            f"branch_quality must be {tuple(target_gain.shape)}, got {tuple(quality.shape)}"
        )
    weight = mixture.get("B_pix")
    if not torch.is_tensor(weight):
        weight = torch.ones_like(target_gain[:, :1])
    if weight.shape[-2:] != quality.shape[-2:]:
        weight = F.interpolate(weight, size=quality.shape[-2:], mode="bilinear", align_corners=False)
    weight = weight.detach().expand_as(quality)
    error = F.smooth_l1_loss(quality, target_gain, reduction="none")
    # The denominator explicitly covers all four branches, not only pixels.
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _usage_loss(mixture, gt, reference):
    pi = mixture.get("pi")
    if not torch.is_tensor(pi):
        return zero_like_loss(reference)
    target = F.interpolate(gt.float(), size=pi.shape[-2:], mode="nearest")
    z_keep = mixture.get("z_keep", reference)
    keep_probability = torch.sigmoid(z_keep)
    if keep_probability.shape[-2:] != pi.shape[-2:]:
        keep_probability = F.interpolate(
            keep_probability, size=pi.shape[-2:], mode="bilinear", align_corners=False
        )
    false_negative = ((target > 0.5) & (keep_probability < 0.4)).to(pi.dtype)
    false_positive = ((target < 0.5) & (keep_probability > 0.6)).to(pi.dtype)
    misalignment = build_gt_boundary(target, tuple(pi.shape[-2:]))
    misalignment = misalignment * (1.0 - false_negative) * (1.0 - false_positive)
    stable = (1.0 - torch.maximum(torch.maximum(false_negative, false_positive), misalignment)).clamp(
        0.0, 1.0
    )
    targets = (
        (stable, (0.90, 0.03, 0.04, 0.03)),
        (false_negative, (0.20, 0.60, 0.15, 0.05)),
        (false_positive, (0.20, 0.05, 0.15, 0.60)),
        (misalignment, (0.25, 0.15, 0.50, 0.10)),
    )
    total = zero_like_loss(pi)
    for mask, distribution in targets:
        target_pi = pi.new_tensor(distribution).view(1, 4, 1, 1)
        cross_entropy = -(target_pi * pi.clamp_min(1.0e-6).log()).sum(dim=1, keepdim=True)
        total = total + _masked_mean(cross_entropy, mask)
    return total


def _regularization_loss(mixture, reference):
    if not mixture:
        return zero_like_loss(reference)
    total = zero_like_loss(reference)
    offset = mixture.get("O_pix")
    if torch.is_tensor(offset):
        total = total + offset.abs().mean()
        if offset.size(-2) > 1:
            total = total + (offset[..., 1:, :] - offset[..., :-1, :]).abs().mean()
        if offset.size(-1) > 1:
            total = total + (offset[..., :, 1:] - offset[..., :, :-1]).abs().mean()
    mask_corr = mixture.get("Mask_corr")
    if torch.is_tensor(mask_corr):
        total = total + 0.1 * mask_corr.abs().mean()
    if torch.is_tensor(mixture.get("z_final")) and torch.is_tensor(mixture.get("z_keep")):
        total = total + 0.01 * (mixture["z_final"] - mixture["z_keep"]).abs().mean()
    return total


def _zero_terms(reference):
    names = (
        "L_base",
        "L_final",
        "L_parent",
        "L_child",
        "L_geometry",
        "L_gate",
        "L_mem",
        "L_B3",
        "L_boundary",
        "L_mix_oracle",
        "L_branch",
        "L_quality",
        "L_usage",
        "L_reg",
        "pc_strength",
        "loss_labeled",
    )
    return {name: zero_like_loss(reference) for name in names}


def _append_means(terms, pc, mixture, reference):
    for index, name in enumerate(("keep", "res", "def", "sup")):
        pi = mixture.get("pi")
        terms[f"pi_{name}_mean"] = (
            pi[:, index : index + 1].mean() if torch.is_tensor(pi) else zero_like_loss(reference)
        )
    for output_name, aux_name in (
        ("gate_pc_mean", "gate_pc_map"),
        ("C23_mean", "C23_map"),
        ("route_entropy_norm", "route_entropy_norm"),
        ("parent_entropy", "parent_entropy"),
    ):
        value = _nested_get(pc, aux_name)
        terms[output_name] = value.mean() if torch.is_tensor(value) and value.numel() else zero_like_loss(reference)


def _detach_terms(terms):
    return {name: value.detach() if torch.is_tensor(value) else value for name, value in terms.items()}


__all__ = [
    "base_structure_loss",
    "decoder_base_loss",
    "compute_pc_hbm_labeled_loss",
    "pc_hbm_labeled_loss",
    "pc_hbm_pc_only_labeled_loss",
    "pc_injection_strength",
    "pc_mode_for_epoch",
    "structure_loss",
    "zero_like_loss",
]
