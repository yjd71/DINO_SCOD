from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.ema import update_ema_module
from Model.ts_model import TSModel
import train_ts_model_pseudo_pc_hbm as ts_entrypoint
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


def test_ts_trainer_does_not_remap_teacher_artifact_schedule():
    pc_cfg = DinoPCHBMConfig(
        verify_start_epoch=4,
        full_pc_start_epoch=9,
        teacher_only_full_start_epoch=5,
    )
    expected = dict(vars(pc_cfg))

    model = nn.Module()
    model.training_design = "teacher_only"
    model.teacher = nn.Module()
    model.teacher.pc_hbm = nn.Linear(2, 2)
    model.student = nn.Module()
    model.student.pc_hbm = None
    runtime_cfg = SimpleNamespace(
        pc_training_design="teacher_only",
        device="cpu",
        distributed=False,
        l_batch_size=31,
        u_batch_size=32,
    )

    # Stop before dataset/optimizer construction while still exercising the
    # complete TS config-contract prefix of __init__.
    with pytest.raises(ValueError, match="batches of 32"):
        PCHBMPseudoTrainer(model, runtime_cfg, pc_cfg)

    assert vars(pc_cfg) == expected


@pytest.mark.parametrize("resume_path", [None, "resume.pth"])
def test_ts_entrypoint_preserves_checkpoint_config(
    monkeypatch,
    resume_path,
):
    pc_cfg = DinoPCHBMConfig(
        verify_start_epoch=4,
        full_pc_start_epoch=9,
        teacher_only_full_start_epoch=5,
    )
    expected = dict(vars(pc_cfg))
    captured = {}

    args = SimpleNamespace(
        training_design="teacher_only",
        teacher_checkpoint="teacher.pth",
        student_checkpoint=None,
        resume=resume_path,
        labeled_indices_pt="labeled.pt",
        output_dir="results",
        epochs=30,
        seed=2025,
        deterministic=False,
        num_workers=None,
        learning_rate=None,
        no_amp=False,
    )
    runtime_cfg = SimpleNamespace()

    class FakeTSModel:
        def __init__(self, *, teacher_pth, student_pth, pc_cfg, training_design):
            captured["model_config"] = dict(vars(pc_cfg))
            captured["teacher_path"] = teacher_pth
            captured["training_design"] = training_design

        def to(self, device):
            captured["device"] = device
            return self

    class FakeTrainer:
        def __init__(self, model, cfg, pc_cfg, *, resume_path):
            captured["trainer_config"] = dict(vars(pc_cfg))
            captured["resume_path"] = resume_path

        def train(self):
            captured["trained"] = True

    def configure_distributed(cfg, _context, seed):
        cfg.seed = seed
        cfg.device = torch.device("cpu")

    monkeypatch.setattr(ts_entrypoint, "parse_args", lambda: args)
    monkeypatch.setattr(ts_entrypoint, "init_distributed", lambda: object())
    monkeypatch.setattr(ts_entrypoint, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(ts_entrypoint, "Config", lambda: runtime_cfg)
    monkeypatch.setattr(
        ts_entrypoint,
        "configure_distributed",
        configure_distributed,
    )
    monkeypatch.setattr(ts_entrypoint, "set_seed", lambda *_: None)
    monkeypatch.setattr(
        ts_entrypoint,
        "read_pc_config",
        lambda *_args, **_kwargs: pc_cfg,
    )
    monkeypatch.setattr(
        ts_entrypoint,
        "validate_canonical_labeled_indices_pt",
        lambda _path: "split",
    )
    monkeypatch.setattr(
        ts_entrypoint,
        "validate_teacher_enhancer_checkpoint",
        lambda *_args: {"baseline_fingerprint": "baseline"},
    )
    monkeypatch.setattr(ts_entrypoint, "TSModel", FakeTSModel)
    monkeypatch.setattr(
        ts_entrypoint,
        "wrap_distributed",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(ts_entrypoint, "PCHBMPseudoTrainer", FakeTrainer)

    ts_entrypoint.main()

    assert vars(pc_cfg) == expected
    assert captured["model_config"] == expected
    assert captured["trainer_config"] == expected
    assert captured["teacher_path"] == "teacher.pth"
    assert captured["training_design"] == "teacher_only"
    assert captured["resume_path"] == resume_path
    assert captured["trained"] is True
