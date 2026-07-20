"""Encoder-side PC-HBM components, isolated from the legacy decoder engine."""

from .child_semantic_detail_verifier import (
    ChildSemanticDetailVerifier,
    EncoderParentChildDetailVerifier,
    EncoderParentRetriever,
    EncoderPCVerifier,
    EncoderSemanticDetailVerifier,
    NormalizedStructuredPrior,
    build_support_targets,
)
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
from .encoder_memory_builder import EncoderMemoryBuilder
from .encoder_router import EncoderPCRouter, EncoderRouter
from .encoder_feature_injector import (
    EncoderF4F3InjectionOutput,
    EncoderFeatureInjector,
    RouteTokenContextAdapter,
)
from .encoder_pc_adapter import (
    EncoderPCAdapter,
    EncoderPCAdapterOutput,
    EncoderPCHBMAdapter,
    EncoderPCStageFlags,
)
from .encoder_pc_segmentation_head import (
    ENCODER_PC_SEGMENTATION_ROLES,
    EncoderPCCoreResult,
    EncoderPCSegmentationHead,
)
from .encoder_level_propagation import (
    EncoderLevelPropagation,
    EncoderPropagationOutput,
    SameGridLocalCrossAttention,
)
from .feature_projector import DinoFeatureProjector, ProjectedDinoFeatures
from .route_context_adapter import (
    EncoderRouteContextAdapter,
    EncoderRouteContextOutput,
    EncoderStructuredGate,
    scatter_query_tokens,
)
from .teacher_pseudo_refiner import (
    EncoderRefinerEvidence,
    RefinerLossWeights,
    TeacherPseudoLabelRefiner,
    TeacherPseudoRefinerOutput,
    teacher_pseudo_refiner_labeled_loss,
)

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
    "EncoderF4F3InjectionOutput",
    "EncoderFeatureInjector",
    "RouteTokenContextAdapter",
    "EncoderPCAdapter",
    "EncoderPCAdapterOutput",
    "EncoderPCHBMAdapter",
    "EncoderPCStageFlags",
    "ENCODER_PC_SEGMENTATION_ROLES",
    "EncoderPCCoreResult",
    "EncoderPCSegmentationHead",
    "EncoderLevelPropagation",
    "EncoderPropagationOutput",
    "SameGridLocalCrossAttention",
    "EncoderRouteContextAdapter",
    "EncoderRouteContextOutput",
    "EncoderStructuredGate",
    "scatter_query_tokens",
    "ChildSemanticDetailVerifier",
    "EncoderParentChildDetailVerifier",
    "EncoderParentRetriever",
    "EncoderPCVerifier",
    "EncoderSemanticDetailVerifier",
    "NormalizedStructuredPrior",
    "build_encoder_memory_compat_meta",
    "build_support_targets",
    "EncoderRefinerEvidence",
    "RefinerLossWeights",
    "TeacherPseudoLabelRefiner",
    "TeacherPseudoRefinerOutput",
    "teacher_pseudo_refiner_labeled_loss",
]
