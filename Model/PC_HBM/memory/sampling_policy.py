"""Deterministic uniform sampling for the two labeled pair regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


REGION_NAMES = ("fg_boundary", "bg_near")
MAX_QUOTA = {"fg_boundary": 48, "bg_near": 48}
MIN_QUOTA = {"fg_boundary": 8, "bg_near": 8}
SAMPLING_RATIO = {"fg_boundary": 0.5, "bg_near": 0.5}


@dataclass(frozen=True)
class RegionSamplingRule:
    max_count: int
    min_count: int
    ratio: float

    def __post_init__(self) -> None:
        if int(self.min_count) < 0:
            raise ValueError("min_count must be non-negative")
        if int(self.max_count) < int(self.min_count):
            raise ValueError("max_count must not be smaller than min_count")
        if not 0.0 <= float(self.ratio) <= 1.0:
            raise ValueError("ratio must be in [0,1]")


DEFAULT_REGION_SAMPLING = {
    name: RegionSamplingRule(
        max_count=MAX_QUOTA[name],
        min_count=MIN_QUOTA[name],
        ratio=SAMPLING_RATIO[name],
    )
    for name in REGION_NAMES
}


def rules_from_config(config: object | None) -> dict[str, RegionSamplingRule]:
    if config is None:
        return dict(DEFAULT_REGION_SAMPLING)
    names = tuple(getattr(config, "region_names", REGION_NAMES))
    maximum = tuple(getattr(config, "region_max_quota", (48, 48)))
    minimum = tuple(getattr(config, "region_min_quota", (8, 8)))
    ratios = tuple(getattr(config, "region_sampling_ratio", (0.5, 0.5)))
    if not (len(names) == len(maximum) == len(minimum) == len(ratios) == 2):
        raise ValueError("Two-region sampling configuration is incomplete")
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 2
    ):
        raise ValueError("region_names must contain two unique non-empty strings")
    return {
        name: RegionSamplingRule(int(max_count), int(min_count), float(ratio))
        for name, max_count, min_count, ratio in zip(
            names,
            maximum,
            minimum,
            ratios,
        )
    }


def sample_region_indices(
    mask: torch.Tensor,
    region: str,
    *,
    rules: Mapping[str, RegionSamplingRule] | None = None,
) -> torch.Tensor:
    """Uniformly cover ordered valid positions without randomness or repeats."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")
    policy = DEFAULT_REGION_SAMPLING if rules is None else rules
    if region not in policy:
        raise KeyError(f"Unknown PC-HBM-Lite region: {region}")
    available = torch.nonzero(mask.flatten().bool(), as_tuple=False).flatten()
    count = int(available.numel())
    if count == 0:
        return available

    rule = policy[region]
    desired = min(
        count,
        max(int(rule.min_count), int(round(count * float(rule.ratio)))),
        int(rule.max_count),
    )
    if desired <= 0:
        return available[:0]
    positions = torch.linspace(
        0,
        count - 1,
        steps=desired,
        device=available.device,
        dtype=torch.float32,
    ).round().long()
    selected = available.index_select(0, positions)
    if selected.unique().numel() != selected.numel():
        raise RuntimeError("Deterministic region sampling produced duplicate indices")
    return selected


__all__ = [
    "DEFAULT_REGION_SAMPLING",
    "MAX_QUOTA",
    "MIN_QUOTA",
    "REGION_NAMES",
    "RegionSamplingRule",
    "SAMPLING_RATIO",
    "rules_from_config",
    "sample_region_indices",
]
