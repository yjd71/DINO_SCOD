"""Encoder-side PC-HBM components.

This package is deliberately independent from the decoder-side PC-HBM engine so
that the legacy profiles and their checkpoints keep their original contracts.
"""

from .contracts import DinoFeatureBundle

__all__ = ["DinoFeatureBundle"]
