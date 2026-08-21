from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.checkpoint_pc_hbm import state_dict_fingerprint
from utils.trainer_base_model_pc_hbm import (
    BasePCHBMTrainer,
    configure_teacher_only_trainability,
    configure_two_stage_trainability,
)


class TinyPC(nn.Linear):
    def build_memory_entries(self, **kwargs):
        return kwargs


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.legacy = nn.Linear(2, 2)
        self.decoder.pc_hbm = TinyPC(2, 2)


def test_teacher_only_trainability_contains_only_pc_parameters():
    model = TinyBase()
    names = configure_teacher_only_trainability(model)
    assert names
    assert all(name.startswith("pc_hbm.") for name in names)
    assert not any(
        parameter.requires_grad for parameter in model.decoder.legacy.parameters()
    )


def test_two_stage_trains_decoder_but_keeps_dino_frozen():
    model = TinyBase()
    names = configure_two_stage_trainability(model)
    assert any(name.startswith("legacy.") for name in names)
    assert any(name.startswith("pc_hbm.") for name in names)
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.dino.parameters())


def test_epoch_memory_is_built_by_current_student_even_in_off_epoch():
    base = TinyBase()
    trainer = BasePCHBMTrainer.__new__(BasePCHBMTrainer)
    calls = []
    trainer.memory_rebuild_fn = lambda **kwargs: calls.append(kwargs)
    trainer.model = base
    trainer.decoder = base.decoder
    trainer.memory_loader = object()
    trainer.memory = object()
    trainer.device = "cpu"
    trainer.cfg = SimpleNamespace(
        labeled_split_count=404,
        labeled_split_fingerprint="0" * 64,
    )
    trainer.pc_cfg = SimpleNamespace(memory_producer_role="labeled_student")
    trainer.amp_enabled = False
    trainer._assert_memory_ready = lambda epoch, producer: calls.append(
        (epoch, producer)
    )

    trainer._rebuild_epoch_memory(1)
    assert calls[0]["memory_decoder"] is trainer.decoder
    assert calls[0]["entry_builder"].__self__ is trainer.decoder.pc_hbm
    assert calls[1] == (1, trainer.decoder)


def test_base_student_artifacts_track_final_non_pc_fingerprint():
    base = TinyBase()
    initial = state_dict_fingerprint(
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
        "labeled_split_fingerprint": "0" * 64,
        "baseline_fingerprint": initial,
    }
    trainer.resume_baseline_fingerprint = initial
    with torch.no_grad():
        trainer.decoder.legacy.weight.add_(1.0)
    final = trainer._current_legacy_fingerprint()

    student_meta = trainer._artifact_metadata("base_student")
    memory_meta = trainer._artifact_metadata("base_student_memory")
    resume_meta = trainer._artifact_metadata("resume")
    assert student_meta["baseline_fingerprint"] == final
    assert memory_meta["baseline_fingerprint"] == final
    assert resume_meta["baseline_fingerprint"] == initial
    assert student_meta["pc_frozen"] is False
