"""Complete diagnostics and persistence warnings for PC-HBM-Lite."""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .losses import structure_loss
from .supervision import build_pair_label_map


DIAGNOSTIC_NAMES = (
    "route_entropy_norm",
    "pair_query_valid_ratio",
    "pair_supervised_ratio",
    "pair_cls_acc",
    "parent_fg_top1_similarity",
    "parent_bg_top1_similarity",
    "child_fg_top1_similarity",
    "child_bg_top1_similarity",
    "candidate_entropy_mean",
    "region_margin_mean",
    "memory_confidence_mean",
    "gate_mean",
    "gate_on_error",
    "gate_on_correct",
    "p3_delta_l1",
    "child_mix_beta",
    "z_main_loss",
    "pseudo_conf_mean",
    "pseudo_conf_boundary_mean",
)


def collect_pc_diagnostics(
    aux: Mapping[str, Any],
    gt: torch.Tensor | None = None,
    *,
    pseudo_confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the fixed Lite diagnostic schema for every mode."""

    if not isinstance(aux, Mapping):
        raise TypeError("aux must be a mapping")
    z_main = aux.get("z_main")
    if not torch.is_tensor(z_main):
        raise KeyError("aux['z_main'] is required")
    zero = z_main.detach().float().sum() * 0.0
    metrics = {name: zero for name in DIAGNOSTIC_NAMES}
    if gt is not None:
        metrics["z_main_loss"] = structure_loss(z_main, gt).detach()
    if pseudo_confidence is not None:
        if not torch.is_tensor(pseudo_confidence):
            raise TypeError("pseudo_confidence must be a tensor")
        metrics["pseudo_conf_mean"] = (
            pseudo_confidence.detach().float().mean()
        )

    pc = aux.get("pc_hbm")
    if not isinstance(pc, Mapping):
        return metrics
    valid = pc.get("query_valid")
    if not torch.is_tensor(valid) or valid.ndim != 1:
        raise ValueError("query_valid must be [M]")
    valid = valid.detach().bool()
    count = valid.numel()
    metrics["pair_query_valid_ratio"] = (
        valid.float().mean() if count else zero
    )

    route = pc.get("route")
    if isinstance(route, Mapping):
        entropy = route.get("route_entropy_norm")
        metrics["route_entropy_norm"] = _mean_tensor(entropy, zero)

    candidate_valid = pc.get("retrieval_valid")
    if torch.is_tensor(candidate_valid):
        if candidate_valid.ndim != 3 or candidate_valid.shape[:2] != (
            count,
            2,
        ):
            raise ValueError("retrieval_valid must be [M,2,K]")
        candidate_valid = candidate_valid.detach().bool()
    else:
        candidate_valid = None
    parent_scores = pc.get("parent_cosine")
    if not torch.is_tensor(parent_scores):
        parent_scores = pc.get("retrieval_scores")
    child_scores = pc.get("child_cosine")
    metrics["parent_fg_top1_similarity"] = _region_top1(
        parent_scores, candidate_valid, valid, 0, zero
    )
    metrics["parent_bg_top1_similarity"] = _region_top1(
        parent_scores, candidate_valid, valid, 1, zero
    )
    metrics["child_fg_top1_similarity"] = _region_top1(
        child_scores, candidate_valid, valid, 0, zero
    )
    metrics["child_bg_top1_similarity"] = _region_top1(
        child_scores, candidate_valid, valid, 1, zero
    )

    candidate_entropy = pc.get("candidate_entropy")
    metrics["candidate_entropy_mean"] = _masked_query_mean(
        candidate_entropy, valid, zero
    )
    region_probability = pc.get("region_prob")
    if torch.is_tensor(region_probability):
        if region_probability.shape != (count, 2):
            raise ValueError("region_prob must be [M,2]")
        margin = (
            region_probability.detach().float()[:, 0]
            - region_probability.detach().float()[:, 1]
        ).abs()
        metrics["region_margin_mean"] = _masked_query_mean(
            margin, valid, zero
        )
    confidence = pc.get("memory_confidence")
    gate = pc.get("gate")
    metrics["memory_confidence_mean"] = _masked_query_mean(
        confidence, valid, zero
    )
    metrics["gate_mean"] = _masked_query_mean(gate, valid, zero)
    metrics["p3_delta_l1"] = _mean_abs(pc.get("p3_delta"), zero)
    metrics["child_mix_beta"] = _mean_tensor(pc.get("beta"), zero)

    query_map = pc.get("query_mask_map")
    if pseudo_confidence is not None and torch.is_tensor(query_map):
        query_at_output = F.interpolate(
            query_map.detach().float(),
            size=pseudo_confidence.shape[-2:],
            mode="nearest",
        )
        boundary = query_at_output > 0.5
        if bool(boundary.any()):
            expanded = boundary.expand_as(pseudo_confidence)
            metrics["pseudo_conf_boundary_mean"] = (
                pseudo_confidence.detach().float()[expanded].mean()
            )

    if gt is not None:
        supervised, labels = _query_supervision(pc, gt, valid)
        metrics["pair_supervised_ratio"] = (
            supervised.float().mean() if count else zero
        )
        logits = pc.get("pair_logits")
        if torch.is_tensor(logits) and bool(supervised.any()):
            if logits.shape != (count, 2):
                raise ValueError("pair_logits must be [M,2]")
            metrics["pair_cls_acc"] = (
                logits.detach()[supervised].argmax(dim=1)
                == labels[supervised]
            ).float().mean()
        correct = _query_prediction_correct(pc, z_main, gt, valid)
        gate_vector = _query_vector(gate, count, "gate")
        metrics["gate_on_error"] = _masked_query_mean(
            gate_vector, valid & ~correct, zero
        )
        metrics["gate_on_correct"] = _masked_query_mean(
            gate_vector, valid & correct, zero
        )
    return metrics


class DiagnosticWarningTracker:
    """Emit warnings only after a Lite failure persists for a full window."""

    def __init__(self, config: Any):
        self.window = max(
            1, int(getattr(config, "diagnostic_window_epochs", 3))
        )
        self.low_valid = float(
            getattr(config, "warn_low_pair_valid_ratio", 0.05)
        )
        self.random_band = float(
            getattr(config, "warn_pair_acc_near_random", 0.05)
        )
        self.inactive_gate = float(
            getattr(config, "warn_gate_inactive_threshold", 0.02)
        )
        self.large_delta = float(
            getattr(config, "warn_delta_large_threshold", 1.0)
        )
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def update(
        self,
        metrics: Mapping[str, Any],
        *,
        emit: bool = True,
    ) -> list[str]:
        for name in DIAGNOSTIC_NAMES:
            self.history[name].append(_as_float(metrics.get(name, 0.0)))
        messages = []
        checks = (
            (
                "pair_query_valid_ratio",
                lambda value: value < self.low_valid,
                "pair query valid ratio stayed below its threshold",
            ),
            (
                "pair_cls_acc",
                lambda value: abs(value - 0.5) <= self.random_band,
                "pair classification accuracy stayed near random",
            ),
            (
                "gate_mean",
                lambda value: value < self.inactive_gate,
                "P3 gate stayed inactive",
            ),
            (
                "p3_delta_l1",
                lambda value: value > self.large_delta,
                "P3 residual magnitude stayed abnormally large",
            ),
            (
                "child_mix_beta",
                lambda value: min(value, 1.0 - value) < 0.02,
                "child mixing beta stayed saturated",
            ),
        )
        for name, predicate, message in checks:
            values = self.history[name]
            if len(values) == self.window and all(
                predicate(value) for value in values
            ):
                messages.append(message)
        if emit:
            for message in messages:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        return messages


def _query_supervision(pc, gt, valid):
    count = valid.numel()
    if count == 0:
        return valid, torch.empty(0, dtype=torch.long, device=valid.device)
    query_map = pc.get("query_mask_map")
    if not torch.is_tensor(query_map):
        raise KeyError("query_mask_map is required for diagnostics")
    labels_map = build_pair_label_map(gt, query_map.shape[-2:])
    batch_ids = _query_vector(
        pc.get("query_batch_ids"), count, "query_batch_ids"
    ).long()
    flat_indices = _query_vector(
        pc.get("query_flat_indices"), count, "query_flat_indices"
    ).long()
    flat_labels = labels_map.reshape(labels_map.shape[0], -1)
    if (
        bool((batch_ids < 0).any())
        or bool((batch_ids >= flat_labels.shape[0]).any())
        or bool((flat_indices < 0).any())
        or bool((flat_indices >= flat_labels.shape[1]).any())
    ):
        raise IndexError("Diagnostic query index is out of bounds")
    labels = flat_labels[batch_ids, flat_indices]
    return valid & (labels >= 0), labels


def _query_prediction_correct(pc, z_main, gt, valid):
    count = valid.numel()
    if count == 0:
        return valid
    query_map = pc["query_mask_map"]
    size = query_map.shape[-2:]
    prediction = F.interpolate(
        torch.sigmoid(z_main.detach().float()),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0].reshape(z_main.shape[0], -1)
    target = F.interpolate(
        gt.detach().float(),
        size=size,
        mode="nearest",
    )[:, 0].reshape(gt.shape[0], -1)
    batch_ids = _query_vector(
        pc.get("query_batch_ids"), count, "query_batch_ids"
    ).long()
    flat_indices = _query_vector(
        pc.get("query_flat_indices"), count, "query_flat_indices"
    ).long()
    return (
        prediction[batch_ids, flat_indices] >= 0.5
    ) == (target[batch_ids, flat_indices] >= 0.5)


def _query_vector(value, count: int, name: str):
    if not torch.is_tensor(value):
        raise KeyError(f"{name} is required")
    value = value.detach().reshape(-1)
    if value.numel() != count:
        raise ValueError(f"{name} must have M={count} entries")
    return value


def _region_top1(scores, candidate_valid, query_valid, region, zero):
    if not torch.is_tensor(scores) or candidate_valid is None:
        return zero
    if scores.shape != candidate_valid.shape:
        raise ValueError("candidate score and valid shapes must match")
    region_valid = candidate_valid[:, region]
    has_candidate = region_valid.any(dim=1) & query_valid
    if not bool(has_candidate.any()):
        return zero
    top1 = scores.detach().float()[:, region].masked_fill(
        ~region_valid, -1.0e4
    ).max(dim=1).values
    return top1[has_candidate].mean()


def _masked_query_mean(value, valid, zero):
    if not torch.is_tensor(value) or value.numel() == 0:
        return zero
    if value.shape[0] != valid.numel():
        raise ValueError("Query metric first dimension must equal M")
    selected = value.detach().float()[valid]
    return selected.mean() if selected.numel() else zero


def _mean_tensor(value, zero):
    if not torch.is_tensor(value) or value.numel() == 0:
        return zero
    return value.detach().float().mean()


def _mean_abs(value, zero):
    if not torch.is_tensor(value) or value.numel() == 0:
        return zero
    return value.detach().float().abs().mean()


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


__all__ = [
    "DIAGNOSTIC_NAMES",
    "DiagnosticWarningTracker",
    "collect_pc_diagnostics",
]
