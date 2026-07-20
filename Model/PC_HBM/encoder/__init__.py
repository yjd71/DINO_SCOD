"""Encoder-side PC-HBM components, isolated from the legacy decoder engine."""

from .contracts import DinoFeatureBundle
from .encoder_boundary_query import EncoderBoundaryOutput, EncoderBoundaryQuery
from .encoder_global_fusion import (
    EncoderBootstrap,
    EncoderBootstrapOutput,
    EncoderGlobalFusion,
    EncoderGlobalOutput,
)
from .encoder_memory import (
    ENCODER_PC_MEMORY_ARCHITECTURE,
    ENCODER_PC_MEMORY_FORMAT_VERSION,
    ENCODER_PC_MEMORY_SCHEMA_VERSION,
    ENCODER_PC_REQUIRED_COMPAT_KEYS,
    ENCODER_PC_STATIC_COMPAT_KEYS,
    ENCODER_PC_STATIC_COMPAT_META,
    EncoderMemoryCompatibilityResult,
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)
from .feature_projector import DinoFeatureProjector, ProjectedDinoFeatures
from .encoder_memory_builder import EncoderMemoryBuilder
from .encoder_router import EncoderPCRouter, EncoderRouter

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
    "ENCODER_PC_MEMORY_ARCHITECTURE",
    "ENCODER_PC_MEMORY_FORMAT_VERSION",
    "ENCODER_PC_MEMORY_SCHEMA_VERSION",
    "ENCODER_PC_REQUIRED_COMPAT_KEYS",
    "ENCODER_PC_STATIC_COMPAT_KEYS",
    "ENCODER_PC_STATIC_COMPAT_META",
    "EncoderMemoryCompatibilityResult",
    "EncoderMemoryBuilder",
    "EncoderPCMemory",
    "EncoderPCRouter",
    "EncoderRouter",
    "build_encoder_memory_compat_meta",
]
