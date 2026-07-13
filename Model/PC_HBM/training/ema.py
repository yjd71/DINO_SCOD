"""Named, buffer-complete EMA helpers for Base and Teacher-Student training."""

from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema_copy(module: nn.Module) -> nn.Module:
    """Deep-copy a frozen evaluation module for memory production."""

    ema = copy.deepcopy(module).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema_module(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float = 0.995,
    *,
    shared_only: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    """EMA parameters by name and copy registered buffers.

    ``shared_only`` supports a raw Student paired with a PC-HBM Teacher.  In
    that mode names beginning with ``exclude_prefixes`` are deliberately left
    untouched on the Teacher, while every remaining Student/Teacher key must
    still match exactly.  The default retains the original strict contract.
    """

    momentum = float(momentum)
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"EMA momentum must be in [0,1], got {momentum}")
    exclude_prefixes = tuple(str(prefix) for prefix in exclude_prefixes)

    def selected(named_values):
        values = dict(named_values)
        if not shared_only:
            return values
        return {
            name: value
            for name, value in values.items()
            if not name.startswith(exclude_prefixes)
        }

    student_parameters = selected(student.named_parameters())
    teacher_parameters = selected(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        mismatch = sorted(student_parameters.keys() ^ teacher_parameters.keys())
        raise RuntimeError(f"EMA parameter key mismatch: {mismatch}")
    for name, student_value in student_parameters.items():
        teacher_value = teacher_parameters[name]
        if teacher_value.shape != student_value.shape:
            raise RuntimeError(f"EMA parameter shape mismatch for {name}")
        teacher_value.mul_(momentum).add_(
            student_value.detach().to(device=teacher_value.device, dtype=teacher_value.dtype),
            alpha=1.0 - momentum,
        )

    student_buffers = selected(student.named_buffers())
    teacher_buffers = selected(teacher.named_buffers())
    if student_buffers.keys() != teacher_buffers.keys():
        mismatch = sorted(student_buffers.keys() ^ teacher_buffers.keys())
        raise RuntimeError(f"EMA buffer key mismatch: {mismatch}")
    for name, student_value in student_buffers.items():
        teacher_value = teacher_buffers[name]
        if teacher_value.shape != student_value.shape:
            raise RuntimeError(f"EMA buffer shape mismatch for {name}")
        teacher_value.copy_(
            student_value.detach().to(device=teacher_value.device, dtype=teacher_value.dtype)
        )
    teacher.eval()
    teacher.requires_grad_(False)


__all__ = ["make_ema_copy", "update_ema_module"]
