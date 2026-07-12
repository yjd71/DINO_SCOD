"""Fixed channel contract for the RSBL DINO PC-HBM adaptation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DinoPCHBMChannelSpec:
    x3: int = 128
    p3: int = 128
    p2: int = 128
    p1: int = 128
    pc_dim: int = 128
    value_dim: int = 8
    geometry_dim: int = 6


def build_dino_channel_spec() -> DinoPCHBMChannelSpec:
    """Return the immutable same-grid DINO channel specification."""

    return DinoPCHBMChannelSpec()


__all__ = ["DinoPCHBMChannelSpec", "build_dino_channel_spec"]

