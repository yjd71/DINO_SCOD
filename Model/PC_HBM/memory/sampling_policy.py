"""Deterministic, availability-aware sampling for labelled memory regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


MAX_QUOTA = {
    "fg_core": 32,
    "fg_boundary": 64,
    "bg_near": 64,
    "bg_far": 32,
}

MIN_QUOTA = {
    "fg_core": 4,
    "fg_boundary": 8,
    "bg_near": 8,
    "bg_far": 4,
}

SAMPLING_RATIO = {
    "fg_core": 0.20,
    "fg_boundary": 0.50,
    "bg_near": 0.50,
    "bg_far": 0.20,
}


@dataclass(frozen=True)
class RegionSamplingRule:
    max_count: int
    min_count: int
    ratio: float


DEFAULT_REGION_SAMPLING = {
    name: RegionSamplingRule(MAX_QUOTA[name], MIN_QUOTA[name], SAMPLING_RATIO[name])
    for name in MAX_QUOTA
}


def rules_from_config(config: object | None) -> dict[str, RegionSamplingRule]:
    if config is None:
        return dict(DEFAULT_REGION_SAMPLING)
    names = tuple(getattr(config, "region_names", tuple(MAX_QUOTA)))
    maximum = tuple(getattr(config, "region_max_quota", tuple(MAX_QUOTA[name] for name in names)))
    minimum = tuple(getattr(config, "region_min_quota", tuple(MIN_QUOTA[name] for name in names)))
    ratios = tuple(getattr(config, "region_sampling_ratio", tuple(SAMPLING_RATIO[name] for name in names)))
    if not (len(names) == len(maximum) == len(minimum) == len(ratios)):
        raise ValueError("Invalid region sampling configuration")
    return {
        str(name): RegionSamplingRule(int(max_count), int(min_count), float(ratio))
        for name, max_count, min_count, ratio in zip(names, maximum, minimum, ratios)
    }


def sample_region_indices(
    mask: torch.Tensor,
    score: torch.Tensor | None,
    region: str,
    *,
    rules: Mapping[str, RegionSamplingRule] | None = None,
) -> torch.Tensor:
    """Choose existing pixels only; missing regions are never synthesized."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")
    available = torch.nonzero(mask.flatten().bool(), as_tuple=False).flatten()
    if available.numel() == 0:
        return available
    policy = DEFAULT_REGION_SAMPLING if rules is None else rules
    if region not in policy:
        raise KeyError(f"Unknown PC-HBM region: {region}")
    rule = policy[region]
    count = int(available.numel())
    desired = max(int(rule.min_count), int(round(count * float(rule.ratio))))
    desired = min(count, int(rule.max_count), desired)
    if score is None:
        return available[:desired]
    if score.shape != mask.shape:
        raise ValueError(f"score shape {tuple(score.shape)} does not match mask {tuple(mask.shape)}")
    reliability = score.flatten().index_select(0, available)
    order = torch.argsort(reliability, descending=True, stable=True)
    return available.index_select(0, order[:desired])


__all__ = [
    "DEFAULT_REGION_SAMPLING",
    "MAX_QUOTA",
    "MIN_QUOTA",
    "RegionSamplingRule",
    "SAMPLING_RATIO",
    "rules_from_config",
    "sample_region_indices",
]

