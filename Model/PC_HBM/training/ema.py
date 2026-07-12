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
def update_ema_module(student: nn.Module, teacher: nn.Module, momentum: float = 0.995) -> None:
    """EMA parameters by exact name and copy every registered buffer."""

    momentum = float(momentum)
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"EMA momentum must be in [0,1], got {momentum}")
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
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

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
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
