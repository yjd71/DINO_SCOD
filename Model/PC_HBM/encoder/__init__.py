"""Encoder-side PC-HBM components.

This package is deliberately independent from the decoder-side PC-HBM engine so
that the legacy profiles and their checkpoints keep their original contracts.
"""

from .contracts import DinoFeatureBundle
from .encoder_boundary_query import EncoderBoundaryOutput, EncoderBoundaryQuery
from .encoder_global_fusion import (
    EncoderBootstrap,
    EncoderBootstrapOutput,
    EncoderGlobalFusion,
    EncoderGlobalOutput,
)
from .feature_projector import DinoFeatureProjector, ProjectedDinoFeatures

__all__ = [
    "DinoFeatureBundle",
    "DinoFeatureProjector",
    "ProjectedDinoFeatures",
    "EncoderGlobalFusion",
    "EncoderGlobalOutput",
    "EncoderBoundaryQuery",
    "EncoderBoundaryOutput",
    "EncoderBootstrap",
    "EncoderBootstrapOutput",
]
