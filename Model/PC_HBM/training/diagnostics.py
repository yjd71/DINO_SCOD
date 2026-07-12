"""PC-HBM diagnostic metrics and persistent-collapse warnings."""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .losses import structure_loss
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


DIAGNOSTIC_NAMES = (
    "parent_top1_region_acc",
    "parent_topk_region_acc",
    "parent_entropy",
    "route_entropy_norm",
    "child_verify_auc",
    "child_positive_score",
    "child_hard_negative_score",
    "geometry_sdf_l1",
    "geometry_normal_cos",
    "geometry_offset_l1",
    "C23_mean",
    "C23_boundary_mean",
    "gate_pc_mean",
    "gate_pc_on_error",
    "gate_pc_on_correct",
    "pi_keep_mean",
    "pi_res_mean",
    "pi_def_mean",
    "pi_sup_mean",
    "pi_res_on_fn",
    "pi_sup_on_fp",
    "pi_def_on_misalignment",
    "z_main_loss",
    "z_final_loss",
    "pseudo_conf_mean",
    "pseudo_conf_boundary_mean",
)


@torch.no_grad()
def collect_pc_diagnostics(
    aux: Mapping[str, Any],
    gt: torch.Tensor | None = None,
    *,
    pseudo_confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Collect the complete stable metric schema from a full-mode aux dict."""

    reference = aux.get("z_main")
    if not torch.is_tensor(reference):
        reference = _first_tensor(aux)
    if reference is None:
        reference = torch.tensor(0.0)
    zero = reference.new_zeros(())
    metrics = {name: zero.clone() for name in DIAGNOSTIC_NAMES}
    pc = aux.get("pc_hbm", {}) or {}
    mixture = aux.get("mixture", {}) or {}
    indices = _boundary_indices(pc)

    parent_regions = _nested_get(pc, "top_parent_region_ids")
    if gt is not None and indices is not None and torch.is_tensor(parent_regions) and parent_regions.numel():
        target_regions = gather_by_boundary_indices(build_region_label_map(gt, _pc_size(pc)), indices)
        valid = parent_regions.ge(0)
        explicit = _nested_get(pc, "top_parent_valid")
        if torch.is_tensor(explicit):
            valid &= explicit.to(dtype=torch.bool, device=valid.device)
        valid_query = valid.any(dim=1)
        if bool(valid_query.any()):
            top1_valid = valid[:, 0] & valid_query
            if bool(top1_valid.any()):
                metrics["parent_top1_region_acc"] = (
                    (parent_regions[:, 0] == target_regions) & top1_valid
                ).float().sum() / top1_valid.float().sum()
            topk_correct = ((parent_regions == target_regions[:, None]) & valid).any(dim=1)
            metrics["parent_topk_region_acc"] = topk_correct[valid_query].float().mean()

        child_logits = _nested_get(pc, "S_child")
        if torch.is_tensor(child_logits) and child_logits.shape == parent_regions.shape:
            support = parent_regions == target_regions[:, None]
            score = torch.sigmoid(child_logits)
            flat_valid = valid.flatten()
            metrics["child_verify_auc"] = _binary_auc(
                score.flatten()[flat_valid], support.flatten()[flat_valid]
            )
            positives = valid & support
            if bool(positives.any()):
                metrics["child_positive_score"] = score[positives].mean()
            hard_negative = valid & (
                ((target_regions[:, None] == REGION_FG_BOUNDARY) & (parent_regions == REGION_BG_NEAR))
                | ((target_regions[:, None] == REGION_BG_NEAR) & (parent_regions == REGION_FG_BOUNDARY))
            )
            if bool(hard_negative.any()):
                metrics["child_hard_negative_score"] = score[hard_negative].mean()

        parent_geometry = _nested_get(pc, "G_attn")
        if torch.is_tensor(parent_geometry) and parent_geometry.numel():
            geometry = build_geometry_target(gt, _pc_size(pc))
            sdf = gather_by_boundary_indices(geometry["sdf"], indices).flatten()
            normal = gather_by_boundary_indices(geometry["normal"], indices)
            offset = gather_by_boundary_indices(geometry["offset"], indices)
            query_valid = valid.any(dim=1) if valid.ndim == 2 else torch.ones_like(sdf, dtype=torch.bool)
            if bool(query_valid.any()):
                metrics["geometry_sdf_l1"] = (
                    parent_geometry[:, 0][query_valid] - sdf[query_valid]
                ).abs().mean()
                metrics["geometry_normal_cos"] = F.cosine_similarity(
                    parent_geometry[:, 1:3][query_valid], normal[query_valid], dim=-1
                ).mean()
                predicted_offset = _nested_get(pc, "O_pc_token")
                if torch.is_tensor(predicted_offset) and predicted_offset.shape == offset.shape:
                    metrics["geometry_offset_l1"] = (
                        predicted_offset[query_valid] - offset[query_valid]
                    ).abs().mean()

    for metric_name, aux_name in (
        ("parent_entropy", "parent_entropy"),
        ("route_entropy_norm", "route_entropy_norm"),
        ("C23_mean", "C23_map"),
        ("gate_pc_mean", "gate_pc_map"),
    ):
        value = _nested_get(pc, aux_name)
        if torch.is_tensor(value) and value.numel():
            metrics[metric_name] = value.float().mean()

    c23 = _nested_get(pc, "C23_map")
    gate = _nested_get(pc, "gate_pc_map")
    if gt is not None and torch.is_tensor(c23):
        boundary = build_gt_boundary(gt, tuple(c23.shape[-2:])).bool()
        if bool(boundary.any()):
            metrics["C23_boundary_mean"] = c23[boundary].float().mean()
    if gt is not None and torch.is_tensor(gate) and torch.is_tensor(reference):
        need = build_need_correction_map(reference, gt, tuple(gate.shape[-2:])).bool()
        if bool(need.any()):
            metrics["gate_pc_on_error"] = gate[need].float().mean()
        correct = ~need
        if bool(correct.any()):
            metrics["gate_pc_on_correct"] = gate[correct].float().mean()

    pi = mixture.get("pi")
    if torch.is_tensor(pi) and pi.ndim == 4 and pi.size(1) == 4:
        for index, name in enumerate(("keep", "res", "def", "sup")):
            metrics[f"pi_{name}_mean"] = pi[:, index].float().mean()
        if gt is not None:
            target = F.interpolate(gt.float(), size=pi.shape[-2:], mode="nearest")
            keep = torch.sigmoid(mixture.get("z_keep", reference))
            if keep.shape[-2:] != pi.shape[-2:]:
                keep = F.interpolate(keep, size=pi.shape[-2:], mode="bilinear", align_corners=False)
            false_negative = (target > 0.5) & (keep < 0.4)
            false_positive = (target < 0.5) & (keep > 0.6)
            misalignment = build_gt_boundary(target, tuple(pi.shape[-2:])).bool()
            metrics["pi_res_on_fn"] = _conditional_mean(pi[:, 1:2], false_negative, zero)
            metrics["pi_sup_on_fp"] = _conditional_mean(pi[:, 3:4], false_positive, zero)
            metrics["pi_def_on_misalignment"] = _conditional_mean(
                pi[:, 2:3], misalignment, zero
            )

    if gt is not None and torch.is_tensor(reference):
        metrics["z_main_loss"] = structure_loss(reference, gt)
        z_final = aux.get("z_final")
        if torch.is_tensor(z_final):
            metrics["z_final_loss"] = structure_loss(z_final, gt)
    if torch.is_tensor(pseudo_confidence):
        metrics["pseudo_conf_mean"] = pseudo_confidence.float().mean()
        if gt is not None:
            boundary = build_gt_boundary(gt, tuple(pseudo_confidence.shape[-2:])).bool()
            metrics["pseudo_conf_boundary_mean"] = _conditional_mean(
                pseudo_confidence, boundary, zero
            )
    return {name: value.detach() for name, value in metrics.items()}


class DiagnosticWarningTracker:
    """Emit warnings only after a condition persists for a configured window."""

    def __init__(self, config: Any):
        self.config = config
        self.window = max(1, int(getattr(config, "diagnostic_window_epochs", 3)))
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, metrics: Mapping[str, Any], *, emit: bool = True) -> list[str]:
        for name, value in metrics.items():
            if torch.is_tensor(value):
                value = float(value.detach().cpu())
            if isinstance(value, (int, float)):
                self.history[name].append(float(value))
        messages = self._current_messages()
        if emit:
            for message in messages:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        return messages

    def _persistent(self, name: str, predicate) -> bool:
        values = self.history.get(name, ())
        return len(values) == self.window and all(predicate(value) for value in values)

    def _current_messages(self) -> list[str]:
        messages: list[str] = []
        keep_threshold = float(getattr(self.config, "warn_keep_collapse_threshold", 0.95))
        dead_threshold = float(getattr(self.config, "warn_dead_branch_threshold", 0.01))
        gate_threshold = float(getattr(self.config, "warn_gate_inactive_threshold", 0.03))
        auc_delta = float(getattr(self.config, "warn_child_auc_distance_from_half", 0.05))
        if self._persistent("pi_keep_mean", lambda value: value > keep_threshold):
            messages.append("PC-HBM mixture collapse: pi_keep_mean stayed above threshold")
        for branch in ("res", "def", "sup"):
            if self._persistent(f"pi_{branch}_mean", lambda value: value < dead_threshold):
                messages.append(f"PC-HBM dead mixture branch: pi_{branch}_mean stayed below threshold")
        if self._persistent("gate_pc_mean", lambda value: value < gate_threshold):
            messages.append("PC-HBM memory correction inactive: gate_pc_mean stayed too low")
        if self._persistent("child_verify_auc", lambda value: abs(value - 0.5) < auc_delta):
            messages.append("PC-HBM child verification remained close to random AUC")
        c23_threshold = float(getattr(self.config, "warn_high_contradiction_threshold", 0.50))
        high_gate = float(getattr(self.config, "warn_high_gate_threshold", 0.50))
        if self._persistent("C23_mean", lambda value: value > c23_threshold) and self._persistent(
            "gate_pc_mean", lambda value: value > high_gate
        ):
            messages.append("PC-HBM gate stayed high despite high parent-child contradiction")
        if len(self.history.get("z_final_loss", ())) == self.window and len(
            self.history.get("z_main_loss", ())
        ) == self.window:
            final = self.history["z_final_loss"]
            main = self.history["z_main_loss"]
            if final[-1] < final[0] - 1.0e-4 and main[-1] >= main[0] - 1.0e-4:
                messages.append("PC-HBM z_final improved while Student z_main did not improve")
        return messages


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.bool)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return scores.new_tensor(0.5)
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    rank_sum = ranks[labels].sum()
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)


def _conditional_mean(value, mask, zero):
    mask = mask.to(device=value.device, dtype=torch.bool)
    if not bool(mask.any()):
        return zero.clone()
    return value[mask.expand_as(value)].float().mean()


def _nested_get(mapping, key, default=None):
    if key in mapping:
        return mapping[key]
    for value in mapping.values():
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return default


def _boundary_indices(pc):
    return normalize_boundary_indices(
        _nested_get(pc, "boundary_indices3"),
        batch_ids=_nested_get(pc, "batch_ids3"),
        flat_indices=_nested_get(pc, "flat_indices3"),
    )


def _pc_size(pc):
    for key in ("B3", "valid3_map", "C23_map", "gate_pc_map"):
        value = _nested_get(pc, key)
        if torch.is_tensor(value):
            return int(value.shape[-2]), int(value.shape[-1])
    return 28, 28


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_tensor(child)
            if found is not None:
                return found
    return None


__all__ = ["DIAGNOSTIC_NAMES", "DiagnosticWarningTracker", "collect_pc_diagnostics"]
