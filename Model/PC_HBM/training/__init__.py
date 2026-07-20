"""Training contracts for DINO PC-HBM."""

from .diagnostics import DIAGNOSTIC_NAMES, DiagnosticWarningTracker, collect_pc_diagnostics
from .ema import make_ema_copy, update_ema_module
from .encoder_pseudo import (
    build_encoder_pc_confidence,
    confidence_weighted_logit_bce,
    confidence_weighted_structure_loss,
    encoder_pc_unlabeled_loss,
    prepare_encoder_pc_pseudo_targets,
)
from .encoder_training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_labeled_loss,
    make_ema_encoder_adapter,
    update_ema_encoder_adapter,
)
from .losses import (
    base_structure_loss,
    decoder_base_loss,
    compute_pc_hbm_labeled_loss,
    pc_hbm_labeled_loss,
    pc_hbm_pc_only_labeled_loss,
    pc_injection_strength,
    pc_mode_for_epoch,
    structure_loss,
)
from .optimizer import migration_aware_parameter_groups
from .pseudo_label import (
    build_pc_confidence,
    confidence_weighted_feature_cosine_loss,
    compute_pc_hbm_unlabeled_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
    weighted_structure_loss,
)

__all__ = [
    "DIAGNOSTIC_NAMES",
    "DiagnosticWarningTracker",
    "EncoderPCStage",
    "base_structure_loss",
    "decoder_base_loss",
    "build_pc_confidence",
    "build_encoder_pc_optimizer",
    "build_encoder_pc_confidence",
    "collect_pc_diagnostics",
    "confidence_weighted_feature_cosine_loss",
    "confidence_weighted_logit_bce",
    "confidence_weighted_structure_loss",
    "compute_pc_hbm_labeled_loss",
    "compute_pc_hbm_unlabeled_loss",
    "configure_encoder_pc_stage",
    "encoder_pc_labeled_loss",
    "make_ema_copy",
    "make_ema_encoder_adapter",
    "migration_aware_parameter_groups",
    "pc_hbm_labeled_loss",
    "pc_hbm_pc_only_labeled_loss",
    "pc_injection_strength",
    "pc_mode_for_epoch",
    "pc_unlabeled_loss",
    "encoder_pc_unlabeled_loss",
    "prepare_encoder_pc_pseudo_targets",
    "prepare_pseudo_targets",
    "structure_loss",
    "update_ema_module",
    "update_ema_encoder_adapter",
    "weighted_structure_loss",
]
