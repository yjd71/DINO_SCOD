"""Optimizer parameter groups for explicit legacy-to-BGFBR migration."""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def migration_aware_parameter_groups(
    module: nn.Module,
    *,
    base_lr: float,
    reused_parameter_names: Iterable[str] = (),
) -> list[dict]:
    """Keep ordinary parameters at ``base_lr`` and reused PC weights at half LR.

    Reused projectors remain ordinary decoder parameters. Only names inside
    ``pc_hbm.*`` receive the conservative learning rate.
    """

    lr = float(base_lr)
    if lr <= 0.0:
        raise ValueError("base_lr must be positive")
    reused_pc = {
        str(name)
        for name in reused_parameter_names
        if str(name).startswith("pc_hbm.")
    }
    named = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    unknown = reused_pc.difference(named)
    if unknown:
        raise RuntimeError(
            "Migrated PC optimizer names are absent or frozen: "
            + ", ".join(sorted(unknown))
        )

    regular = [parameter for name, parameter in named.items() if name not in reused_pc]
    conservative = [parameter for name, parameter in named.items() if name in reused_pc]
    groups: list[dict] = []
    if regular:
        groups.append({"params": regular, "lr": lr, "group_name": "base"})
    if conservative:
        groups.append(
            {
                "params": conservative,
                "lr": 0.5 * lr,
                "group_name": "migrated_pc",
            }
        )
    if not groups:
        raise RuntimeError("Decoder has no trainable parameters")
    return groups


__all__ = ["migration_aware_parameter_groups"]
