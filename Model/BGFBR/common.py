"""Small reusable layers for BGFBR."""

from __future__ import annotations

import torch.nn as nn


def make_norm(norm: str, channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "sync_bn":
        return nn.SyncBatchNorm(channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported BGFBR norm: {norm!r}")


class ConvBNReLU(nn.Sequential):
    """Conv2d followed by the configured normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int | None = None,
        norm: str = "bn",
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size // 2)
        use_bias = str(norm).lower() in {"none", "identity"}
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
            ),
            make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )


__all__ = ["ConvBNReLU", "make_norm"]
