from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from configs.ts_model_config import Config
import utils.distributed as distributed
import utils.trainer_ts_model_pseudo_pc_hbm as ts_trainer
from utils.trainer_ts_model_pseudo_pc_hbm import (
    PCHBMPseudoTrainer,
    validate_teacher_enhancer_checkpoint,
)
import Model.ts_model as ts_model_module
import train_ts_model_pseudo_pc_hbm as ts_entrypoint
from train_ts_model_pseudo_pc_hbm import parse_args, validate_training_args
from utils.ts_lr_scheduler import (
    build_ts_cosine_scheduler,
    validate_ts_scheduler_contract,
)


class _TinyTeacherStudent(nn.Module):
    def __init__(self, legacy_value, pc_value):
        super().__init__()
        self.legacy = nn.Linear(1, 1, bias=False)
        self.pc_hbm = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.legacy.weight.fill_(legacy_value)
            self.pc_hbm.weight.fill_(pc_value)


def _fingerprint_contract_trainer(training_design, teacher):
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.training_design = training_design
    trainer.core_model = SimpleNamespace(teacher=teacher)
    trainer._teacher_pc_fingerprint = None
    trainer._capture_teacher_pc_fingerprint()
    return trainer


def test_teacher_only_pc_fingerprint_rejects_real_state_change():
    teacher = _TinyTeacherStudent(legacy_value=1.0, pc_value=2.0)
    trainer = _fingerprint_contract_trainer("teacher_only", teacher)

    trainer._validate_teacher_pc_contract()
    with torch.no_grad():
        teacher.pc_hbm.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="teacher-only TS"):
        trainer._validate_teacher_pc_contract()


def test_joint_pc_fingerprint_allows_intentional_full_ema_update():
    student = _TinyTeacherStudent(legacy_value=3.0, pc_value=5.0)
    teacher = _TinyTeacherStudent(legacy_value=1.0, pc_value=2.0)
    trainer = _fingerprint_contract_trainer("joint", teacher)
    before = ts_trainer.module_fingerprint(teacher.pc_hbm)

    ts_trainer.update_ema_module(
        student,
        teacher,
        momentum=0.5,
        shared_only=False,
    )

    assert ts_trainer.module_fingerprint(teacher.pc_hbm) != before
    trainer._validate_teacher_pc_contract()


def test_teacher_only_fingerprint_uses_post_resume_teacher_state():
    teacher = _TinyTeacherStudent(legacy_value=1.0, pc_value=2.0)
    trainer = _fingerprint_contract_trainer("teacher_only", teacher)
    pre_resume_fingerprint = trainer._teacher_pc_fingerprint

    # Simulate load_training_resume restoring a different frozen EMA Teacher.
    with torch.no_grad():
        teacher.pc_hbm.weight.fill_(7.0)
    trainer._capture_teacher_pc_fingerprint()

    assert trainer._teacher_pc_fingerprint != pre_resume_fingerprint
    trainer._validate_teacher_pc_contract()


def _teacher_aux_in_inference_mode(*, include_p1=False):
    with torch.inference_mode():
        aux = {
            "p_final": torch.rand(2, 1, 98, 98),
            "z_main": torch.randn(2, 1, 98, 98),
            "pc_active": True,
            "fallback_reason": None,
            "pc_hbm": {
                "C23_map": torch.rand(2, 1, 28, 28),
                "route_entropy_norm": torch.rand(2),
            },
            "mixture": {"pi": torch.softmax(torch.randn(2, 4, 98, 98), dim=1)},
            "distill_features": {
                "p3_corr": torch.randn(2, 128, 28, 28),
                "p2_refined": torch.randn(2, 128, 28, 28),
            },
        }
        if include_p1:
            aux["p1_pra"] = {
                "B1": torch.rand(2, 1, 98, 98),
                "G1_raw_map": torch.randn(2, 1, 98, 98),
                "R1_map": torch.randn(2, 128, 98, 98),
                "O1_map": torch.randn(2, 2, 98, 98),
                "R_sup_map": torch.randn(2, 1, 98, 98),
                "valid1_map": torch.randint(0, 2, (2, 1, 98, 98)).float(),
            }
        return aux


def test_teacher_targets_are_cloned_out_of_inference_mode():
    aux = _teacher_aux_in_inference_mode()
    assert aux["p_final"].is_inference()

    cloned = PCHBMPseudoTrainer._clone_teacher_target_aux(aux)

    tensors = (
        cloned["p_final"],
        cloned["z_main"],
        cloned["pc_hbm"]["C23_map"],
        cloned["pc_hbm"]["route_entropy_norm"],
        cloned["mixture"]["pi"],
        cloned["distill_features"]["p3_corr"],
        cloned["distill_features"]["p2_refined"],
    )
    assert all(not tensor.is_inference() for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_joint_teacher_p1_targets_are_cloned_and_nested_for_distillation():
    aux = _teacher_aux_in_inference_mode(include_p1=True)

    cloned = PCHBMPseudoTrainer._clone_teacher_target_aux(
        aux,
        training_design="joint",
    )

    p1 = cloned["distill_features"]["p1"]
    assert set(p1) == {
        "B1",
        "G1_raw_map",
        "R1_map",
        "O1_map",
        "R_sup_map",
        "valid1_map",
    }
    assert all(not tensor.is_inference() for tensor in p1.values())
    assert all(tensor.grad_fn is None for tensor in p1.values())
    assert all(
        p1[name].data_ptr() != aux["p1_pra"][name].data_ptr()
        for name in p1
    )


def test_teacher_only_targets_do_not_require_or_export_p1():
    cloned = PCHBMPseudoTrainer._clone_teacher_target_aux(
        _teacher_aux_in_inference_mode(),
        training_design="teacher_only",
    )

    assert "p1" not in cloned["distill_features"]


def test_joint_teacher_targets_require_every_p1_tensor():
    aux = _teacher_aux_in_inference_mode(include_p1=True)
    del aux["p1_pra"]["O1_map"]

    with pytest.raises(KeyError, match="p1_pra.O1_map"):
        PCHBMPseudoTrainer._clone_teacher_target_aux(
            aux,
            training_design="joint",
        )


def _ts_model_for_teacher_aux(aux, training_design):
    class FakeTeacher(nn.Module):
        def forward(self, features, **kwargs):
            return (torch.zeros(1),) * 5, aux

    model = object.__new__(ts_model_module.TSModel)
    nn.Module.__init__(model)
    model.training_design = training_design
    model.teacher = FakeTeacher()
    return model


@pytest.mark.parametrize("training_design", ("teacher_only", "joint"))
def test_teacher_pseudo_always_requires_p3_p2_targets(training_design):
    aux = _teacher_aux_in_inference_mode(include_p1=training_design == "joint")
    del aux["distill_features"]["p3_corr"]
    model = _ts_model_for_teacher_aux(aux, training_design)

    with pytest.raises(RuntimeError, match="P3/P2"):
        model.teacher_pseudo([torch.zeros(1)], memory=object(), epoch=31)


def test_joint_teacher_pseudo_requires_complete_p1_targets():
    aux = _teacher_aux_in_inference_mode(include_p1=True)
    del aux["p1_pra"]["R_sup_map"]
    model = _ts_model_for_teacher_aux(aux, "joint")

    with pytest.raises(RuntimeError, match="complete P1"):
        model.teacher_pseudo([torch.zeros(1)], memory=object(), epoch=31)


def test_ts_decoder_epoch_continues_after_base_schedule():
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.pc_cfg = SimpleNamespace(mixture_schedule_end_epoch=30)
    assert trainer._decoder_epoch(1) == 31
    assert trainer._decoder_epoch(15) == 45


@pytest.mark.parametrize("producer_design", ("teacher_only", "two_stage"))
def test_ts_accepts_supported_frozen_teacher_enhancer_designs(producer_design):
    artifact = {
        "artifact_meta": {
            "training_design": producer_design,
            "artifact_role": "teacher_enhancer",
            "labeled_split_fingerprint": "split-sha256",
            "baseline_fingerprint": "baseline-sha256",
            "pc_frozen": True,
        }
    }

    metadata = validate_teacher_enhancer_checkpoint(artifact, "split-sha256")
    assert metadata["training_design"] == producer_design


def test_ts_rejects_joint_teacher_enhancer_design():
    artifact = {
        "artifact_meta": {
            "training_design": "joint",
            "artifact_role": "teacher_enhancer",
            "labeled_split_fingerprint": "split-sha256",
            "baseline_fingerprint": "baseline-sha256",
            "pc_frozen": True,
        }
    }

    with pytest.raises(RuntimeError, match="training_design"):
        validate_teacher_enhancer_checkpoint(artifact, "split-sha256")


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("artifact_role", "resume"),
        ("labeled_split_fingerprint", "wrong-split"),
        ("pc_frozen", False),
    ),
)
def test_ts_rejects_wrong_two_stage_teacher_enhancer_identity(field, wrong_value):
    artifact_meta = {
        "training_design": "two_stage",
        "artifact_role": "teacher_enhancer",
        "labeled_split_fingerprint": "split-sha256",
        "baseline_fingerprint": "baseline-sha256",
        "pc_frozen": True,
    }
    artifact_meta[field] = wrong_value

    with pytest.raises(RuntimeError, match=field):
        validate_teacher_enhancer_checkpoint(
            {"artifact_meta": artifact_meta}, "split-sha256"
        )


def test_ts_model_teacher_only_uses_raw_student_off_paths(monkeypatch):
    class FakeDino(nn.Module):
        def load_state_dict(self, state):
            return None

    class FakeDecoder(nn.Module):
        decoder_arch = "legacy_transformer"

        def __init__(self, pc_cfg=None):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.pc_hbm = nn.Linear(2, 2) if pc_cfg is not None else None
            self.modes = []

        def forward(self, features, **kwargs):
            self.modes.append(kwargs.get("pc_mode"))
            value = torch.zeros(1, 1, 2, 2)
            return (value,) * 5, {"z_main": value, "mixture_skipped": True}

    monkeypatch.setattr(ts_model_module.torch.hub, "load", lambda *args, **kwargs: FakeDino())
    monkeypatch.setattr(ts_model_module.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(ts_model_module, "Decoder", FakeDecoder)
    monkeypatch.setattr(
        ts_model_module,
        "load_decoder_compatible",
        lambda *args, **kwargs: None,
    )

    model = ts_model_module.TSModel(
        teacher_pth="teacher.pth",
        pc_cfg=SimpleNamespace(enabled=True, dino_layer_indices=(2, 5, 8, 11)),
        training_design="teacher_only",
    )
    assert model.teacher.pc_hbm is not None
    assert model.student.pc_hbm is None
    model.student_labeled([torch.zeros(1)], memory=object(), epoch=1)
    model.student_unlabeled([torch.zeros(1)], memory=object(), epoch=1)
    assert model.student.modes == ["off", "off"]


def test_train_prints_start_and_end_time_for_every_epoch(monkeypatch, capsys):
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.current_epoch = 1
    trainer.cfg = SimpleNamespace(epochs=2)
    trainer.scheduler = SimpleNamespace(step=lambda: None)
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-4}])
    trainer.train_epoch = lambda: {
        "loss": 1.25,
        "confidence": 0.125,
        "confidence_max": 0.25,
        "confidence_positive_fraction": 0.5,
        "L_u_feat_p1": 0.03125,
        "L_u_feat_p1_B1": 0.01,
        "L_u_feat_p1_G1": 0.02,
        "L_u_feat_p1_R1": 0.03,
        "L_u_feat_p1_O1": 0.04,
        "L_u_feat_p1_R_sup": 0.05,
    }
    trainer._save_epoch = lambda epoch, metrics: None
    trainer._export_final_memory = lambda: None

    timestamps = iter(
        (
            "07-13 01:00:00",
            "07-13 01:05:00",
            "07-13 01:05:01",
            "07-13 01:10:00",
        )
    )
    monkeypatch.setattr(ts_trainer, "_current_local_timestamp", lambda: next(timestamps))
    monkeypatch.setattr(ts_trainer, "is_main_process", lambda: True)
    monkeypatch.setattr(ts_trainer, "synchronize", lambda: None)

    trainer.train()

    output = capsys.readouterr().out
    assert "epoch 1/2: start_time=07-13 01:00:00" in output
    assert "epoch 1/2: loss=1.250000" in output
    assert "L_u_feat_p1=0.031250" in output
    assert "P1_B1=1.000e-02" in output
    assert "P1_G1=2.000e-02" in output
    assert "P1_R1=3.000e-02" in output
    assert "P1_O1=4.000e-02" in output
    assert "P1_R_sup=5.000e-02" in output
    assert "end_time=07-13 01:05:00" in output
    assert "epoch 2/2: start_time=07-13 01:05:01" in output
    assert "epoch 2/2: loss=1.250000" in output
    assert "end_time=07-13 01:10:00" in output


def test_ts_resume_rejects_missing_or_different_pc_config():
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.pc_cfg = DinoPCHBMConfig()
    saved = asdict(trainer.pc_cfg)

    trainer._validate_resume_config(saved)
    with pytest.raises(RuntimeError, match="has no pc_cfg"):
        trainer._validate_resume_config(None)

    incompatible = dict(saved)
    incompatible["memory_schema_version"] += 1
    with pytest.raises(RuntimeError, match="memory_schema_version"):
        trainer._validate_resume_config(incompatible)

    legacy = dict(saved)
    legacy.pop("feature_distill_p1_weight")
    with pytest.raises(RuntimeError, match="feature_distill_p1_weight"):
        trainer._validate_resume_config(legacy)


def test_wrap_distributed_forwards_optional_unused_parameter_flag(monkeypatch):
    captured = {}

    class FakeDDP:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(distributed, "DistributedDataParallel", FakeDDP)
    context = distributed.DistributedContext(
        distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    model = torch.nn.Linear(2, 1)
    wrapped = distributed.wrap_distributed(
        model,
        context,
        find_unused_parameters=True,
    )

    assert isinstance(wrapped, FakeDDP)
    assert captured["model"] is model
    assert captured["find_unused_parameters"] is True


def test_wrap_distributed_keeps_legacy_false_default(monkeypatch):
    captured = {}

    class FakeDDP:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(distributed, "DistributedDataParallel", FakeDDP)
    context = distributed.DistributedContext(
        distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    distributed.wrap_distributed(torch.nn.Linear(2, 1), context)
    assert captured["find_unused_parameters"] is False


def test_ts_cli_defaults_to_teacher_only_and_sampled_images_fallback(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ts_model_pseudo_pc_hbm.py",
            "--teacher-pc-checkpoint",
            "teacher.pth",
        ],
    )
    args = parse_args()
    validate_training_args(args)
    assert args.training_design == "teacher_only"
    assert args.teacher_pc_checkpoint == "teacher.pth"
    assert args.labeled_indices_pt is None


def test_ts_cli_keeps_base_pc_checkpoint_as_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ts_model_pseudo_pc_hbm.py",
            "--base-pc-checkpoint",
            "teacher.pth",
            "--labeled-indices-pt",
            "split.pt",
        ],
    )
    assert parse_args().teacher_pc_checkpoint == "teacher.pth"


@pytest.mark.parametrize(
    "args, message",
    [
        (
            SimpleNamespace(
                training_design="teacher_only",
                labeled_indices_pt="split.pt",
                allow_legacy_pc_init=True,
            ),
            "allow-legacy-pc-init",
        ),
    ],
)
def test_teacher_only_cli_rejects_unsafe_initialization(args, message):
    with pytest.raises(ValueError, match=message):
        validate_training_args(args)


def test_joint_cli_retains_legacy_optional_split():
    validate_training_args(
        SimpleNamespace(
            training_design="joint",
            experiment_profile="legacy_pc",
            labeled_indices_pt=None,
            allow_legacy_pc_init=True,
        )
    )


@pytest.mark.parametrize("training_design", ["teacher_only", "joint"])
def test_ts_cli_enables_unused_parameter_discovery_for_every_design(
    monkeypatch,
    training_design,
):
    args = SimpleNamespace(
        training_design=training_design,
        experiment_profile="legacy_pc",
        teacher_pc_checkpoint="teacher_enhancer.pth",
        student_checkpoint=None,
        output_dir="results/ts",
        resume=None,
        allow_legacy_pc_init=False,
        labeled_indices_pt=None,
        epochs=1,
        num_workers=0,
        memory_batch_size=1,
        memory_num_workers=0,
        seed=2027,
        deterministic=True,
    )
    context = SimpleNamespace(rank=0, device="cuda:0")
    cfg = SimpleNamespace()
    pc_cfg = SimpleNamespace(configure_training_design=lambda _design: None)
    model = SimpleNamespace(to=lambda _device: model)
    wrap_calls = []
    train_calls = []

    monkeypatch.setattr(ts_entrypoint, "parse_args", lambda: args)
    monkeypatch.setattr(ts_entrypoint, "init_distributed", lambda: context)
    monkeypatch.setattr(ts_entrypoint, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts_entrypoint, "Config", lambda: cfg)
    monkeypatch.setattr(
        ts_entrypoint,
        "configure_distributed",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ts_entrypoint, "DinoPCHBMConfig", lambda: pc_cfg)
    monkeypatch.setattr(
        ts_entrypoint,
        "apply_experiment_profile",
        lambda *_args, **_kwargs: SimpleNamespace(name="legacy_pc"),
    )
    monkeypatch.setattr(ts_entrypoint, "TSModel", lambda **_kwargs: model)

    def fake_wrap_distributed(
        wrapped_model,
        wrapped_context,
        *,
        find_unused_parameters=False,
    ):
        wrap_calls.append(
            (wrapped_model, wrapped_context, find_unused_parameters)
        )
        return wrapped_model

    monkeypatch.setattr(ts_entrypoint, "wrap_distributed", fake_wrap_distributed)
    monkeypatch.setattr(
        ts_entrypoint,
        "PCHBMPseudoTrainer",
        lambda *_args, **_kwargs: SimpleNamespace(
            train=lambda: train_calls.append(training_design)
        ),
    )
    monkeypatch.setattr(ts_entrypoint, "cleanup_distributed", lambda: None)

    ts_entrypoint.main()

    assert wrap_calls == [(model, context, True)]
    assert train_calls == [training_design]


def test_ts_config_keeps_15_epochs_on_fixed_30_epoch_cosine_period():
    cfg = Config()

    assert cfg.epochs == 15
    assert cfg.learning_rate == pytest.approx(1.0e-4)
    assert cfg.min_lr == pytest.approx(1.0e-7)
    assert cfg.scheduler_t_max == 30


def test_decoder_ts_default_scheduler_keeps_fixed_30_epoch_period(monkeypatch):
    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.pc_hbm = nn.Identity()

    class StopAfterScheduler(RuntimeError):
        pass

    model = SimpleNamespace(
        pc_cfg=object(),
        teacher=TinyDecoder(),
        student=TinyDecoder(),
    )
    cfg = SimpleNamespace(
        pc_training_design="joint",
        distributed=False,
        device="cpu",
        l_batch_size=32,
        u_batch_size=32,
        learning_rate=1.0e-4,
        weight_decay=0.0,
        epochs=15,
        min_lr=1.0e-7,
        scheduler_t_max=30,
    )
    captured = {}
    real_validate = validate_ts_scheduler_contract

    monkeypatch.setattr(
        ts_trainer,
        "trainable_parameter_groups",
        lambda decoder, **_kwargs: decoder.parameters(),
    )

    def capture_scheduler(scheduler, config):
        real_validate(scheduler, config)
        captured["scheduler"] = scheduler

    monkeypatch.setattr(
        ts_trainer,
        "validate_ts_scheduler_contract",
        capture_scheduler,
    )
    monkeypatch.setattr(
        ts_trainer.torch.amp,
        "GradScaler",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterScheduler()),
    )

    with pytest.raises(StopAfterScheduler):
        PCHBMPseudoTrainer(model, cfg, SimpleNamespace(use_amp=False))

    assert captured["scheduler"].T_max == 30
    assert captured["scheduler"].eta_min == pytest.approx(1.0e-7)


def test_ts_cosine_schedule_matches_original_epoch_13_trajectory():
    cfg = Config()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = build_ts_cosine_scheduler(optimizer, cfg)
    used_lrs = []
    logged_lrs = []

    for _epoch in range(1, cfg.epochs + 1):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        logged_lrs.append(optimizer.param_groups[0]["lr"])

    assert scheduler.T_max == 30
    assert logged_lrs[11] == pytest.approx(6.5485398869e-5)
    assert used_lrs[12] == pytest.approx(6.5485398869e-5)
    assert logged_lrs[12] == pytest.approx(6.04351889563e-5)
    assert logged_lrs[14] == pytest.approx(5.005e-5)


def test_ts_scheduler_contract_accepts_t_max_30():
    cfg = Config()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=cfg.min_lr,
    )

    validate_ts_scheduler_contract(scheduler, cfg)


def test_ts_scheduler_contract_rejects_old_t_max_15_resume():
    cfg = Config()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15,
        eta_min=cfg.min_lr,
    )

    with pytest.raises(RuntimeError, match="expected 30, got 15"):
        validate_ts_scheduler_contract(scheduler, cfg)


def test_ts_scheduler_contract_rejects_changed_min_lr_resume():
    cfg = Config()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=1.0e-6,
    )

    with pytest.raises(RuntimeError, match="eta_min mismatch"):
        validate_ts_scheduler_contract(scheduler, cfg)


def test_ts_scheduler_t_max_cannot_be_rebound_to_epochs():
    cfg = SimpleNamespace(epochs=15, scheduler_t_max=15, min_lr=1.0e-7)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)

    with pytest.raises(ValueError, match="scheduler_t_max=30"):
        build_ts_cosine_scheduler(optimizer, cfg)
