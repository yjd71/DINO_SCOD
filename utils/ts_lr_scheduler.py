"""Shared Teacher-Student cosine learning-rate schedule contract."""

from __future__ import annotations

import math
from typing import Any

from torch.optim.lr_scheduler import CosineAnnealingLR


TS_SCHEDULER_T_MAX = 30


def resolve_ts_scheduler_t_max(config: Any) -> int:
    """Return the fixed TS cosine period and reject accidental compression."""

    t_max = int(getattr(config, "scheduler_t_max", TS_SCHEDULER_T_MAX))
    if t_max != TS_SCHEDULER_T_MAX:
        raise ValueError(
            "TS cosine schedule must keep scheduler_t_max=30 so a 15-epoch run "
            "matches the first 15 epochs of the original 30-epoch experiment; "
            f"got {t_max}"
        )
    return t_max


def build_ts_cosine_scheduler(optimizer, config: Any) -> CosineAnnealingLR:
    """Build the fixed-period TS scheduler without coupling it to cfg.epochs."""

    return CosineAnnealingLR(
        optimizer,
        T_max=resolve_ts_scheduler_t_max(config),
        eta_min=float(getattr(config, "min_lr", 1.0e-7)),
    )


def validate_ts_scheduler_contract(scheduler, config: Any) -> None:
    """Reject injected or resumed schedulers from the old compressed curve."""

    expected = resolve_ts_scheduler_t_max(config)
    state = scheduler.state_dict()
    actual = state.get("T_max", getattr(scheduler, "T_max", None))
    if actual is None:
        raise RuntimeError("TS scheduler state does not record T_max")
    if int(actual) != expected:
        raise RuntimeError(
            "TS scheduler T_max mismatch: "
            f"expected {expected}, got {actual}. A checkpoint produced with "
            "T_max=15 cannot reproduce the original 30-epoch LR trajectory."
        )
    expected_eta_min = float(getattr(config, "min_lr", 1.0e-7))
    actual_eta_min = state.get("eta_min", getattr(scheduler, "eta_min", None))
    if actual_eta_min is None:
        raise RuntimeError("TS scheduler state does not record eta_min")
    if not math.isclose(
        float(actual_eta_min),
        expected_eta_min,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(
            "TS scheduler eta_min mismatch: "
            f"expected {expected_eta_min}, got {actual_eta_min}"
        )


__all__ = [
    "TS_SCHEDULER_T_MAX",
    "build_ts_cosine_scheduler",
    "resolve_ts_scheduler_t_max",
    "validate_ts_scheduler_contract",
]
