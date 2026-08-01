from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from Model.ts_model import TSModel
from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    state_dict_fingerprint,
)
from utils.trainer_base_model_pc_hbm import (
    BasePCHBMTrainer,
    configure_teacher_only_trainability,
    configure_two_stage_trainability,
)
from utils.trainer_ts_model_pseudo_pc_hbm import (
    validate_teacher_enhancer_checkpoint,
)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.legacy = nn.Linear(2, 2)
        self.decoder.pc_hbm = nn.Linear(2, 2)


def test_teacher_only_trainability_contains_only_pc_parameters():
    model = TinyBase()
    names = configure_teacher_only_trainability(model)
    assert names
    assert all(name.startswith("pc_hbm.") for name in names)
    assert not any(
        parameter.requires_grad
        for parameter in model.decoder.legacy.parameters()
    )


def test_two_stage_trains_decoder_but_keeps_dino_frozen():
    model = TinyBase()
    names = configure_two_stage_trainability(model)
    assert any(name.startswith("legacy.") for name in names)
    assert any(name.startswith("pc_hbm.") for name in names)
    assert all(
        parameter.requires_grad for parameter in model.decoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.dino.parameters()
    )


def test_memory_rebuild_runs_even_during_off_epochs():
    trainer = BasePCHBMTrainer.__new__(BasePCHBMTrainer)
    calls = []
    trainer.memory_rebuild_fn = lambda **kwargs: calls.append(kwargs)
    trainer.model = object()
    trainer.memory_decoder = nn.Linear(2, 2)
    trainer.memory_loader = object()
    trainer.memory = object()
    trainer.device = "cpu"
    trainer.cfg = SimpleNamespace(
        labeled_split_count=404,
        labeled_split_fingerprint="0" * 64,
    )
    trainer.pc_cfg = SimpleNamespace()
    trainer.amp_enabled = False
    trainer._assert_memory_ready = (
        lambda epoch, producer: calls.append(epoch)
    )
    trainer._rebuild_epoch_memory(1)
    trainer._rebuild_epoch_memory(6)
    assert calls[1] == 1
    assert calls[3] == 6


def test_two_stage_final_teacher_initializes_matching_ts_student():
    base = TinyBase()
    initial_fingerprint = state_dict_fingerprint(
        {
            name: value
            for name, value in base.decoder.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
    )
    trainer = BasePCHBMTrainer.__new__(BasePCHBMTrainer)
    trainer.training_design = "two_stage"
    trainer.decoder = base.decoder
    trainer.checkpoint_metadata = {
        "labeled_split_fingerprint": (
            CANONICAL_LABELED_SPLIT_FINGERPRINT
        ),
        "baseline_fingerprint": initial_fingerprint,
    }
    trainer.resume_baseline_fingerprint = initial_fingerprint

    with torch.no_grad():
        trainer.decoder.legacy.weight.add_(1.0)
    final_fingerprint = trainer._current_legacy_fingerprint()
    assert final_fingerprint != initial_fingerprint

    teacher_metadata = trainer._artifact_metadata("teacher_enhancer")
    memory_metadata = trainer._artifact_metadata("teacher_memory")
    resume_metadata = trainer._artifact_metadata("resume")
    assert teacher_metadata["baseline_fingerprint"] == final_fingerprint
    assert memory_metadata["baseline_fingerprint"] == final_fingerprint
    assert resume_metadata["baseline_fingerprint"] == initial_fingerprint
    validate_teacher_enhancer_checkpoint(
        {"artifact_meta": teacher_metadata},
        CANONICAL_LABELED_SPLIT_FINGERPRINT,
    )

    ts_model = TSModel.__new__(TSModel)
    nn.Module.__init__(ts_model)
    ts_model.teacher = trainer.decoder
    ts_model.student = nn.Module()
    ts_model.student.legacy = nn.Linear(2, 2)
    ts_model._initialize_raw_student_from_teacher()
    assert (
        state_dict_fingerprint(ts_model.student.state_dict())
        == teacher_metadata["baseline_fingerprint"]
    )
