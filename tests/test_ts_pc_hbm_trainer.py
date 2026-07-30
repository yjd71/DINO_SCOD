from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from Model.PC_HBM.training.ema import update_ema_module
from Model.ts_model import TSModel
from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    build_artifact_metadata,
)
from utils.pc_memory_runner import module_fingerprint
from utils.trainer_ts_model_pseudo_pc_hbm import (
    PCHBMPseudoTrainer,
    validate_teacher_enhancer_checkpoint,
)


def _teacher_aux():
    p3 = torch.randn(1, 4, 2, 2)
    return {
        "p_final": torch.rand(1, 1, 4, 4),
        "pc_hbm": {
            "query_mask_map": torch.ones(1, 1, 2, 2),
            "memory_confidence_map": torch.rand(1, 1, 2, 2),
        },
        "distill_features": {"p3_corr": p3},
    }


def test_teacher_targets_clone_only_lite_contract():
    source = _teacher_aux()
    cloned = PCHBMPseudoTrainer._clone_teacher_target_aux(source)
    assert set(cloned["pc_hbm"]) == {
        "query_mask_map",
        "memory_confidence_map",
    }
    assert set(cloned["distill_features"]) == {"p3_corr"}
    source["p_final"].zero_()
    assert not torch.equal(source["p_final"], cloned["p_final"])


def test_teacher_pc_fingerprint_is_immutable():
    trainer = PCHBMPseudoTrainer.__new__(PCHBMPseudoTrainer)
    trainer.core_model = SimpleNamespace(
        teacher=SimpleNamespace(pc_hbm=nn.Linear(2, 2))
    )
    trainer._teacher_pc_fingerprint = module_fingerprint(
        trainer.core_model.teacher.pc_hbm
    )
    trainer._validate_teacher_pc_contract()
    with torch.no_grad():
        trainer.core_model.teacher.pc_hbm.weight.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        trainer._validate_teacher_pc_contract()


def test_ema_updates_shared_legacy_names_but_not_teacher_pc():
    student = nn.Module()
    student.legacy = nn.Linear(2, 2)
    teacher = nn.Module()
    teacher.legacy = nn.Linear(2, 2)
    teacher.pc_hbm = nn.Linear(2, 2)
    pc_before = {
        name: value.clone() for name, value in teacher.pc_hbm.state_dict().items()
    }
    update_ema_module(
        student,
        teacher,
        momentum=0.0,
        shared_only=True,
        exclude_prefixes=("pc_hbm.",),
    )
    for name, value in student.legacy.state_dict().items():
        assert torch.equal(value, teacher.legacy.state_dict()[name])
    for name, value in pc_before.items():
        assert torch.equal(value, teacher.pc_hbm.state_dict()[name])


def test_teacher_checkpoint_validation_accepts_only_frozen_v2_identity():
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint=CANONICAL_LABELED_SPLIT_FINGERPRINT,
        baseline_fingerprint="base",
        pc_frozen=True,
    )
    payload = {"artifact_meta": metadata}
    assert validate_teacher_enhancer_checkpoint(
        payload, CANONICAL_LABELED_SPLIT_FINGERPRINT
    ) == metadata
    bad = {"artifact_meta": {**metadata, "pc_frozen": False}}
    with pytest.raises(RuntimeError):
        validate_teacher_enhancer_checkpoint(
            bad, CANONICAL_LABELED_SPLIT_FINGERPRINT
        )


class RecordingStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = None

    def forward(self, features, **kwargs):
        self.mode = kwargs["pc_mode"]
        return ("outputs", {"forward_mode": self.mode})


def test_ts_student_paths_are_always_off():
    model = TSModel.__new__(TSModel)
    nn.Module.__init__(model)
    model.student = RecordingStudent()
    model.student_labeled([torch.empty(0)])
    assert model.student.mode == "off"
    model.student_unlabeled([torch.empty(0)])
    assert model.student.mode == "off"


def test_ts_forward_has_no_combined_legacy_branch():
    model = TSModel.__new__(TSModel)
    nn.Module.__init__(model)
    with pytest.raises(TypeError):
        model(torch.empty(0), torch.empty(0))
