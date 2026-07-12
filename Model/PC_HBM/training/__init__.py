"""Training contracts for DINO PC-HBM."""

from .diagnostics import DIAGNOSTIC_NAMES, DiagnosticWarningTracker, collect_pc_diagnostics
from .ema import make_ema_copy, update_ema_module
from .losses import (
    base_structure_loss,
    compute_pc_hbm_labeled_loss,
    pc_hbm_labeled_loss,
    pc_injection_strength,
    pc_mode_for_epoch,
    structure_loss,
)
from .pseudo_label import (
    build_pc_confidence,
    compute_pc_hbm_unlabeled_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
    weighted_structure_loss,
)

__all__ = [
    "DIAGNOSTIC_NAMES",
    "DiagnosticWarningTracker",
    "base_structure_loss",
    "build_pc_confidence",
    "collect_pc_diagnostics",
    "compute_pc_hbm_labeled_loss",
    "compute_pc_hbm_unlabeled_loss",
    "make_ema_copy",
    "pc_hbm_labeled_loss",
    "pc_injection_strength",
    "pc_mode_for_epoch",
    "pc_unlabeled_loss",
    "prepare_pseudo_targets",
    "structure_loss",
    "update_ema_module",
    "weighted_structure_loss",
]
