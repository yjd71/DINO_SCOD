"""Losses and the locked epoch schedule for PC-HBM-Lite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .supervision import build_pair_label_map

VALID_PC_MODES = frozenset({"off", "verify_only", "full", "teacher_pseudo"})
VALID_TRAINING_DESIGNS = frozenset({"two_stage", "teacher_only"})


def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero on ``reference``'s device."""

    return reference.float().sum() * 0.0


def structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """F3Net/RSBL weighted BCE plus weighted IoU."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must be [B,1,H,W], got {tuple(logits.shape)}")
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"target must be [B,1,H,W], got {tuple(target.shape)}")
    target = F.interpolate(
        target.float(),
        size=logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(device=logits.device, dtype=logits.dtype)
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(2, 3)) / (
        weight.sum(dim=(2, 3)) + eps
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target * weight).sum(dim=(2, 3))
    union = ((probability + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (
        union - intersection + 1.0
    )
    return (weighted_bce + weighted_iou).mean()


def base_structure_loss(
    outputs: Sequence[torch.Tensor],
    gt: torch.Tensor,
) -> torch.Tensor:
    """Preserve supervision of all five legacy Decoder logits."""

    _validate_outputs(outputs)
    return sum(structure_loss(logit, gt) for logit in outputs)


def pc_mode_for_epoch(epoch: int, config: Any) -> str:
    """Return the configured one-based Base-training mode."""

    epoch = int(epoch)
    if epoch < 1:
        raise ValueError("epoch must be one-based and positive")
    resolver = getattr(config, "pc_mode_for_epoch", None)
    if callable(resolver):
        mode = str(resolver(epoch))
    else:
        verify_start = int(getattr(config, "verify_start_epoch", 6))
        full_start = int(getattr(config, "full_pc_start_epoch", 11))
        mode = "off" if epoch < verify_start else (
            "verify_only" if epoch < full_start else "full"
        )
    if mode not in {"off", "verify_only", "full"}:
        raise ValueError(f"Invalid scheduled PC-HBM-Lite mode: {mode!r}")
    return mode


def binary_pair_loss(
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    reference: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the two Lite evidence regions at selected P3 queries."""

    zero = zero_like_loss(reference)
    pc = aux.get("pc_hbm") if isinstance(aux, Mapping) else None
    if not isinstance(pc, Mapping):
        raise KeyError("active PC-HBM-Lite mode requires aux['pc_hbm']")
    logits = pc.get("pair_logits")
    valid = pc.get("query_valid")
    batch_ids = pc.get("query_batch_ids")
    flat_indices = pc.get("query_flat_indices")
    if not all(
        torch.is_tensor(value)
        for value in (logits, valid, batch_ids, flat_indices)
    ):
        raise KeyError(
            "pair supervision requires pair_logits, query_valid, "
            "query_batch_ids and query_flat_indices"
        )
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"pair_logits must be [M,2], got {tuple(logits.shape)}")
    count = logits.shape[0]
    for name, value in (
        ("query_valid", valid),
        ("query_batch_ids", batch_ids),
        ("query_flat_indices", flat_indices),
    ):
        if value.ndim != 1 or value.shape[0] != count:
            raise ValueError(f"{name} must be [M] with M={count}")
    valid = valid.to(device=logits.device, dtype=torch.bool)
    batch_ids = batch_ids.to(device=logits.device, dtype=torch.long)
    flat_indices = flat_indices.to(device=logits.device, dtype=torch.long)
    if count == 0 or not bool(valid.any()):
        return logits.float().sum() * 0.0, _empty_pair_metrics(zero)

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError(f"gt must be [B,1,H,W], got {tuple(gt.shape)}")
    token_hw = _token_hw(pc)
    target_map = build_pair_label_map(
        gt,
        token_hw,
        boundary_kernel=int(getattr(config, "fg_boundary_kernel", 3)),
        bg_near_kernel=int(getattr(config, "bg_near_kernel", 7)),
        threshold=float(getattr(config, "gt_binary_threshold", 0.5)),
    ).reshape(gt.shape[0], -1)
    selected_batch = batch_ids[valid]
    selected_flat = flat_indices[valid]
    if (
        bool((selected_batch < 0).any())
        or bool((selected_batch >= gt.shape[0]).any())
        or bool((selected_flat < 0).any())
        or bool((selected_flat >= target_map.shape[1]).any())
    ):
        raise IndexError("query indices fall outside the resized ground-truth map")

    selected_target = target_map[selected_batch, selected_flat]
    supervised = selected_target >= 0
    if not bool(supervised.any()):
        return logits.float().sum() * 0.0, _empty_pair_metrics(zero)
    effective_valid = valid.clone()
    effective_valid[valid] = supervised
    target = selected_target[supervised].long()
    selected_logits = logits[effective_valid].float()
    region_loss = F.cross_entropy(selected_logits, target)
    accuracy = (selected_logits.argmax(dim=1) == target).float().mean()
    return region_loss, {
        **_empty_pair_metrics(zero),
        "pair_valid_count": effective_valid.sum().detach().float(),
        "pair_accuracy": accuracy.detach(),
        "L_pair_region": region_loss.detach(),
    }


def pc_hbm_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    epoch: int | None,
    config: Any,
    *,
    pc_mode: str | None = None,
    training_design: str = "two_stage",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the complete Lite labeled objective."""

    _validate_outputs(outputs)
    mode = _resolve_mode(pc_mode, aux, epoch, config)
    design = str(training_design)
    if design not in VALID_TRAINING_DESIGNS:
        raise ValueError(
            f"Unsupported training_design={design!r}; "
            f"expected one of {sorted(VALID_TRAINING_DESIGNS)}"
        )
    reference = outputs[3]
    zero = zero_like_loss(reference)
    pair = zero
    pair_metrics = _empty_pair_metrics(zero)
    if mode != "off":
        _validate_active_aux(aux, mode)
        pair, pair_metrics = binary_pair_loss(
            aux, gt, reference, config
        )

    base = zero
    main = zero
    if design == "two_stage":
        base = base_structure_loss(outputs, gt)
    elif mode == "full":
        main = structure_loss(reference, gt)
    elif mode == "off":
        raise RuntimeError("teacher_only Base training does not support off mode")

    total = base + main + pair
    metrics = {
        "L_base": base.detach(),
        "L_main": main.detach(),
        "L_pair": pair.detach(),
        "L_total": total.detach(),
        **pair_metrics,
    }
    return total, metrics


def pc_hbm_pc_only_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | None,
    gt: torch.Tensor,
    epoch: int | None,
    config: Any,
    *,
    pc_mode: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-enhancer objective: pair CE, plus corrected main loss in full."""

    return pc_hbm_labeled_loss(
        outputs,
        aux,
        gt,
        epoch,
        config,
        pc_mode=pc_mode,
        training_design="teacher_only",
    )


def _validate_outputs(outputs: Sequence[torch.Tensor]) -> None:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError(
            "Decoder outputs must be (m4, m3, m2, z_main, global_logit)"
        )
    if not all(torch.is_tensor(value) for value in outputs):
        raise TypeError("all Decoder outputs must be tensors")


def _resolve_mode(pc_mode, aux, epoch, config) -> str:
    if pc_mode is not None:
        mode = str(pc_mode)
    elif isinstance(aux, Mapping) and aux.get("forward_mode") is not None:
        mode = str(aux["forward_mode"])
    elif epoch is not None:
        mode = pc_mode_for_epoch(int(epoch), config)
    else:
        mode = "off"
    if mode not in VALID_PC_MODES:
        raise ValueError(f"Unsupported PC-HBM-Lite mode: {mode!r}")
    if mode == "teacher_pseudo":
        raise ValueError("teacher_pseudo is inference-only and cannot be labeled mode")
    return mode


def _validate_active_aux(aux, mode: str) -> None:
    if not isinstance(aux, Mapping):
        raise TypeError(f"{mode} mode requires an aux mapping")
    if aux.get("pc_active") is not True:
        raise RuntimeError(
            f"{mode} PC-HBM-Lite path is inactive: {aux.get('fallback_reason')}"
        )


def _token_hw(
    pc: Mapping[str, Any],
) -> tuple[int, int]:
    query_map = pc.get("query_mask_map")
    if torch.is_tensor(query_map):
        if query_map.ndim != 4 or query_map.shape[1] != 1:
            raise ValueError("query_mask_map must be [B,1,H,W]")
        return int(query_map.shape[-2]), int(query_map.shape[-1])
    raise KeyError("query_mask_map is required to establish the query grid")


def _empty_pair_metrics(zero: torch.Tensor) -> dict[str, torch.Tensor]:
    detached = zero.detach()
    return {
        "pair_valid_count": detached,
        "pair_accuracy": detached,
        "L_pair_region": detached,
    }


__all__ = [
    "VALID_PC_MODES",
    "VALID_TRAINING_DESIGNS",
    "base_structure_loss",
    "binary_pair_loss",
    "pc_hbm_labeled_loss",
    "pc_hbm_pc_only_labeled_loss",
    "pc_mode_for_epoch",
    "structure_loss",
    "zero_like_loss",
]
