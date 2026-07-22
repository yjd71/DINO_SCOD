"""Optimizer parameter groups for the canonical original Decoder."""

from __future__ import annotations

import torch.nn as nn


def trainable_parameter_groups(
    module: nn.Module,
    *,
    base_lr: float,
) -> list[dict]:
    """Return one explicit group containing every trainable Decoder parameter."""

    lr = float(base_lr)
    if lr <= 0.0:
        raise ValueError("base_lr must be positive")
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Decoder has no trainable parameters")
    return [{"params": parameters, "lr": lr, "group_name": "decoder"}]


__all__ = ["trainable_parameter_groups"]
