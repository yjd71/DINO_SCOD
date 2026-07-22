from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import EncoderPCCoreResult
import Model.ts_model as ts_model_module
import utils.trainer_ts_model_encoder_pc as trainer_module
import train_ts_model_pseudo_pc_hbm as ts_entrypoint
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.pc_memory_runner import module_fingerprint


class _TinyModule(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))
        self.register_buffer("running", torch.tensor(float(value)))


class _TinyDecoder(_TinyModule):
    decoder_arch = "legacy_transformer"
    decoder_architecture = "legacy_transformer"

    def __init__(self, value=0.0, pc_cfg=None):
        super().__init__(value)
        self.pc_hbm = None


class _TinyDino(_TinyModule):
    def load_state_dict(self, _state, strict=True):
        return None


class _FakeHead:
    def __init__(self, adapter, decoder, refiner):
        self.adapter = adapter
        self.decoder = decoder
        self.pseudo_refiner = refiner


def _patch_encoder_model_construction(monkeypatch, *, loader=None):
    monkeypatch.setattr(
        ts_model_module.torch.hub,
        "load",
        lambda *args, **kwargs: _TinyDino(),
    )
    monkeypatch.setattr(ts_model_module.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(ts_model_module, "Decoder", _TinyDecoder)
    monkeypatch.setattr(
        ts_model_module, "EncoderPCHBMAdapter", lambda _config: _TinyModule()
    )
    monkeypatch.setattr(
        ts_model_module, "TeacherPseudoLabelRefiner", lambda _config: _TinyModule()
    )
    monkeypatch.setattr(ts_model_module, "EncoderPCSegmentationHead", _FakeHead)

    calls = []

    def default_loader(_source, **kwargs):
        calls.append(kwargs)
        with torch.no_grad():
            kwargs["encoder_pc_hbm"].weight.fill_(2.0)
            kwargs["decoder"].weight.fill_(3.0)
            kwargs["pseudo_refiner"].weight.fill_(4.0)
        producer = module_fingerprint(kwargs["encoder_pc_hbm"])
        return {
            "epoch": 30,
            "artifact_meta": {
                "model_role": "base",
                "training_design": "two_stage",
                "split_fingerprint": "split",
                "producer_fingerprint": producer,
                "dino_weight_fingerprint": module_fingerprint(_TinyDino()),
            }
        }

    monkeypatch.setattr(
        ts_model_module,
        "load_encoder_pc_checkpoint",
        default_loader if loader is None else loader,
    )
    return calls


def test_encoder_ts_strict_base_v3_initializes_isomorphic_three_module_roles(monkeypatch):
    calls = _patch_encoder_model_construction(monkeypatch)
    config = EncoderPCHBMConfig()
    model = ts_model_module.TSModel(
        teacher_pth="base-v3.pth",
        pc_cfg=config,
        training_design="teacher_only",
    )

    assert len(calls) == 1
    assert calls[0]["expected_model_role"] == "base"
    assert calls[0]["expected_training_design"] == "two_stage"
    assert calls[0]["expected_config"] is config
    for student, teacher in (
        (model.student_encoder_pc_hbm, model.teacher_encoder_pc_hbm),
        (model.student, model.teacher),
        (model.student_pseudo_refiner, model.teacher_pseudo_refiner),
    ):
        assert tuple(dict(student.named_parameters())) == tuple(
            dict(teacher.named_parameters())
        )
        assert torch.equal(student.weight, teacher.weight)
        assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert model.teacher.pc_hbm is None
    assert model.student.pc_hbm is None
    assert all(not parameter.requires_grad for parameter in model.dino.parameters())


def test_encoder_ts_propagates_strict_v3_loader_rejection(monkeypatch):
    def reject(_source, **_kwargs):
        raise RuntimeError("requires checkpoint format v3")

    _patch_encoder_model_construction(monkeypatch, loader=reject)
    with pytest.raises(RuntimeError, match="format v3"):
        ts_model_module.TSModel(
            teacher_pth="legacy-v2.pth",
            pc_cfg=EncoderPCHBMConfig(),
            training_design="teacher_only",
        )


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    (
        (
            {
                "epoch": 29,
                "artifact_meta": {
                    "split_fingerprint": "split",
                    "producer_fingerprint": "producer",
                },
            },
            "final Base v3 artifact",
        ),
        (
            {"epoch": 30, "artifact_meta": {"split_fingerprint": "split"}},
            "producer_fingerprint",
        ),
        (
            {
                "epoch": 30,
                "artifact_meta": {
                    "split_fingerprint": "split",
                    "producer_fingerprint": "forged",
                    "dino_weight_fingerprint": "dino-test-fingerprint",
                },
            },
            "producer fingerprint",
        ),
        (
            {
                "epoch": 30,
                "artifact_meta": {
                    "split_fingerprint": "split",
                    "producer_fingerprint": module_fingerprint(_TinyModule()),
                    "dino_weight_fingerprint": "forged-dino",
                },
            },
            "DINO fingerprint",
        ),
    ),
)
def test_encoder_ts_rejects_incomplete_or_forged_base_artifact(
    monkeypatch, checkpoint, message
):
    _patch_encoder_model_construction(
        monkeypatch,
        loader=lambda _source, **_kwargs: checkpoint,
    )
    with pytest.raises(RuntimeError, match=message):
        ts_model_module.TSModel(
            teacher_pth="base-v3.pth",
            pc_cfg=EncoderPCHBMConfig(),
            training_design="teacher_only",
        )


class _RecordingRoleHead:
    def __init__(self):
        self.roles = []
        self.refiner_calls = 0
        self.outputs = tuple(torch.randn(1, 1, 2, 2) for _ in range(5))

    def __call__(self, *, role, **kwargs):
        self.roles.append((role, kwargs))
        if role == "teacher_pseudo":
            self.refiner_calls += 1
            return {
                "outputs": self.outputs,
                "z_core": self.outputs[3],
                "aux": {
                    "encoder_pc_hbm": MappingProxyType({"route": {}})
                },
                "pseudo_refiner": {"p_pseudo_refined": torch.rand(1, 1, 2, 2)},
            }
        core = EncoderPCCoreResult(
            self.outputs,
            {"encoder_pc_hbm": {"route": {}}, "features": {"p1": torch.rand(1)}},
        )
        if role == "labeled_refiner":
            self.refiner_calls += 1
            return {"p_pseudo_refined": torch.rand(1, 1, 2, 2)}
        if role == "inference":
            return self.outputs[3]
        return core


def _role_model():
    model = object.__new__(ts_model_module.TSModel)
    nn.Module.__init__(model)
    model.encoder_pc_profile_v3 = True
    model.teacher_encoder_pc_head = _RecordingRoleHead()
    model.student_encoder_pc_head = _RecordingRoleHead()
    return model


def test_encoder_ts_roles_execute_refiner_only_for_teacher_and_labeled_student(monkeypatch):
    model = _role_model()
    monkeypatch.setattr(ts_model_module, "DinoFeatureBundle", object)
    bundle = object()

    teacher = model.teacher_pseudo(bundle, object(), 31)
    labeled_outputs, labeled_aux = model.student_labeled(
        bundle,
        object(),
        31,
        query_image_ids=["image"],
    )
    unlabeled_outputs, unlabeled_aux = model.student_unlabeled(bundle, object(), 31)

    assert "encoder_pc_hbm" in teacher
    assert model.teacher_encoder_pc_head.refiner_calls == 1
    assert model.student_encoder_pc_head.refiner_calls == 1
    assert [role for role, _ in model.student_encoder_pc_head.roles] == [
        "labeled_core",
        "labeled_refiner",
        "student_core",
    ]
    assert labeled_aux["pseudo_refiner"] is not None
    assert unlabeled_aux["pseudo_refiner"] is None
    assert unlabeled_aux["z_core"] is unlabeled_outputs[3]
    assert labeled_outputs[3] is labeled_aux["z_core"]


def test_encoder_ts_inference_returns_student_z_core_without_refiner(monkeypatch):
    model = _role_model()
    bundle = object()
    model.extract_feature_bundle = lambda _x: bundle
    result = model.inference(torch.rand(1, 3, 2, 2), memory=object(), epoch=40)
    assert result is model.student_encoder_pc_head.outputs[3]
    assert model.student_encoder_pc_head.refiner_calls == 0
    assert [role for role, _ in model.student_encoder_pc_head.roles] == ["inference"]


def test_encoder_ts_rejects_legacy_combined_forward_api():
    model = _role_model()
    with pytest.raises(ValueError, match="explicit student_labeled"):
        model(l_x=torch.rand(1), u_x=torch.rand(1))


def test_encoder_ts_ema_updates_all_three_modules_by_name_and_copies_buffers():
    model = _role_model()
    for name in (
        "student_encoder_pc_hbm",
        "student",
        "student_pseudo_refiner",
    ):
        setattr(model, name, _TinyDecoder(2.0) if name == "student" else _TinyModule(2.0))
    for name in (
        "teacher_encoder_pc_hbm",
        "teacher",
        "teacher_pseudo_refiner",
    ):
        setattr(model, name, _TinyDecoder(0.0) if name == "teacher" else _TinyModule(0.0))

    model.update_teacher(momentum=0.5)

    for teacher in (
        model.teacher_encoder_pc_hbm,
        model.teacher,
        model.teacher_pseudo_refiner,
    ):
        assert teacher.weight.item() == pytest.approx(1.0)
        assert teacher.running.item() == pytest.approx(2.0)
        assert all(not parameter.requires_grad for parameter in teacher.parameters())


class _FakeMemory:
    def __init__(self):
        self.ready = False
        self.compat_meta = {}

    def is_ready(self):
        return self.ready

    def state_dict(self):
        return {"schema_version": 3, "compat_meta": dict(self.compat_meta)}


class _MemoryLoader:
    def __init__(self, keys=("image",)):
        self.dataset = SimpleNamespace(sample_keys=list(keys))

    def __len__(self):
        return len(self.dataset.sample_keys)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _Scheduler:
    def __init__(self):
        self.steps = 0
        self.T_max = 30
        self.eta_min = 1.0e-7

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {
            "steps": self.steps,
            "T_max": self.T_max,
            "eta_min": self.eta_min,
        }

    def load_state_dict(self, state):
        self.steps = state["steps"]
        self.T_max = state["T_max"]
        self.eta_min = state["eta_min"]


class _TinyTS(nn.Module):
    def __init__(self, split_fingerprint):
        super().__init__()
        self.encoder_pc_profile_v3 = True
        self.dino = _TinyModule()
        self.student_encoder_pc_hbm = _TinyModule(0.5)
        self.teacher_encoder_pc_hbm = _TinyModule(0.5)
        self.student = _TinyDecoder(0.5)
        self.teacher = _TinyDecoder(0.5)
        self.student_pseudo_refiner = _TinyModule(0.5)
        self.teacher_pseudo_refiner = _TinyModule(0.5)
        self.encoder_base_artifact_meta = {
            "split_fingerprint": split_fingerprint,
            "dino_weight_fingerprint": module_fingerprint(self.dino),
            "baseline_fingerprint": "baseline",
        }
        self.ema_calls = 0
        self.student_refiner_calls = 0
        self.teacher_refiner_calls = 0

    def extract_feature_bundle(self, images):
        return images.detach()

    def teacher_pseudo(self, *_args, **_kwargs):
        self.teacher_refiner_calls += 1
        return {"teacher": True}

    def update_teacher(self, momentum=0.995):
        self.ema_calls += 1

    def forward(self, *, branch, **_kwargs):
        base = self.student.weight * self.student_encoder_pc_hbm.weight
        outputs = tuple(base.reshape(1, 1, 1, 1) + index for index in range(5))
        if branch == "student_labeled":
            self.student_refiner_calls += 1
            return outputs, {
                "pseudo_refiner": {"marker": self.student_pseudo_refiner.weight},
                "z_core": outputs[3],
            }
        if branch == "student_unlabeled":
            return outputs, {"pseudo_refiner": None, "z_core": outputs[3]}
        raise AssertionError(branch)


def _trainer_fixture(
    monkeypatch,
    tmp_path,
    *,
    base_split=None,
    use_default_scheduler=False,
):
    keys = ("image",)
    split = compute_labeled_split_fingerprint(keys)
    core = _TinyTS(split if base_split is None else base_split)
    memory = _FakeMemory()
    rebuild_calls = []

    def rebuild(model, adapter, loader, target, device, **kwargs):
        rebuild_calls.append((model, adapter, loader, target, device, kwargs))
        target.ready = True
        target.compat_meta = {
            "producer_fingerprint": module_fingerprint(adapter),
            "producer_source": kwargs["producer_source"],
        }

    monkeypatch.setattr(trainer_module, "configure_encoder_pc_stage", lambda *args: None)
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    monkeypatch.setattr(trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module, "reduce_mean", lambda value, _device: value)
    monkeypatch.setattr(
        trainer_module,
        "encoder_pc_labeled_loss",
        lambda *_args, **_kwargs: (
            core.student.weight.square() + core.student_encoder_pc_hbm.weight.square(),
            {},
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "teacher_pseudo_refiner_labeled_loss",
        lambda *_args, **_kwargs: (core.student_pseudo_refiner.weight.square(), {}),
    )
    monkeypatch.setattr(
        trainer_module,
        "prepare_encoder_pc_pseudo_targets",
        lambda *_args, **_kwargs: {
            "p_soft": torch.ones(1, 1, 1, 1),
            "confidence": torch.ones(1, 1, 1, 1),
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "encoder_pc_unlabeled_loss",
        lambda outputs, aux, *_args, **_kwargs: (
            outputs[3].square().mean(),
            {"pseudo_conf_mean": outputs[3].new_tensor(1.0)},
        ),
    )
    labeled = [(None, torch.ones(1, 3, 2, 2), torch.ones(1, 1, 2, 2), ["image"])]
    unlabeled = [torch.ones(1, 3, 2, 2)]
    params = (
        list(core.student_encoder_pc_hbm.parameters())
        + list(core.student.parameters())
        + list(core.student_pseudo_refiner.parameters())
    )
    optimizer = _CountingSGD(params)
    cfg = SimpleNamespace(
        device="cpu",
        distributed=False,
        l_batch_size=32,
        u_batch_size=32,
        use_amp=False,
        save_dir=tmp_path,
        epochs=1,
        grad_clip_norm=5.0,
    )
    trainer = trainer_module.EncoderPCTSTrainer(
        core,
        cfg,
        EncoderPCHBMConfig(),
        memory=memory,
        labeled_loader=labeled,
        unlabeled_loader=unlabeled,
        memory_loader=_MemoryLoader(keys),
        optimizer=optimizer,
        scheduler=None if use_default_scheduler else _Scheduler(),
        memory_rebuild_fn=rebuild,
    )
    return trainer, core, memory, rebuild_calls


def test_encoder_ts_trainer_uses_sequential_backwards_one_step_and_teacher_memory(
    monkeypatch, tmp_path
):
    trainer, core, _memory, rebuild_calls = _trainer_fixture(monkeypatch, tmp_path)
    metrics = trainer.train_epoch(1)

    assert trainer.optimizer.step_calls == 1
    assert trainer.scheduler.steps == 1
    assert core.ema_calls == 1
    assert core.teacher_refiner_calls == 1
    assert core.student_refiner_calls == 1
    assert rebuild_calls[0][1] is core.teacher_encoder_pc_hbm
    assert rebuild_calls[0][5]["producer_source"] == "ema_teacher_adapter"
    assert "loss" in metrics
    assert all(parameter.grad is None for parameter in core.dino.parameters())


def test_encoder_ts_default_scheduler_keeps_fixed_30_epoch_period(
    monkeypatch, tmp_path
):
    trainer, _core, _memory, _calls = _trainer_fixture(
        monkeypatch,
        tmp_path,
        use_default_scheduler=True,
    )

    assert trainer.scheduler.T_max == 30
    assert trainer.scheduler.eta_min == pytest.approx(1.0e-7)


def test_encoder_ts_trainer_rejects_base_split_mismatch(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="split fingerprint"):
        _trainer_fixture(monkeypatch, tmp_path, base_split="wrong")


def test_encoder_ts_checkpoint_uses_student_and_all_three_teacher_ema_modules(
    monkeypatch, tmp_path
):
    trainer, core, _memory, _calls = _trainer_fixture(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(
        trainer_module,
        "save_encoder_pc_training_resume",
        lambda path, **kwargs: captured.update(path=path, **kwargs),
    )
    trainer._save_resume(1, {"loss": 1.0})

    assert captured["encoder_pc_hbm"] is core.student_encoder_pc_hbm
    assert captured["decoder"] is core.student
    assert captured["pseudo_refiner"] is core.student_pseudo_refiner
    assert captured["ema_adapter"] is core.teacher_encoder_pc_hbm
    assert captured["ema_decoder"] is core.teacher
    assert captured["ema_refiner"] is core.teacher_pseudo_refiner
    assert captured["model_role"] == "student"
    assert captured["training_design"] == "teacher_student"


def test_encoder_ts_resume_restores_student_and_all_three_teacher_modules(
    monkeypatch, tmp_path
):
    trainer, core, _memory, _calls = _trainer_fixture(monkeypatch, tmp_path)
    captured = {}

    def load(_path, **kwargs):
        captured.update(kwargs)
        return {"epoch": 4, "stage_state": {"name": "full", "ts_epoch": 4}}

    monkeypatch.setattr(trainer_module, "load_encoder_pc_training_resume", load)
    trainer.resume("resume-v3.pth", restore_rng=False)

    assert captured["encoder_pc_hbm"] is core.student_encoder_pc_hbm
    assert captured["decoder"] is core.student
    assert captured["pseudo_refiner"] is core.student_pseudo_refiner
    assert captured["ema_adapter"] is core.teacher_encoder_pc_hbm
    assert captured["ema_decoder"] is core.teacher
    assert captured["ema_refiner"] is core.teacher_pseudo_refiner
    assert captured["expected_model_role"] == "student"
    assert captured["expected_training_design"] == "teacher_student"
    assert trainer.current_epoch == 5


def test_encoder_ts_resume_rejects_old_t_max_15_scheduler(monkeypatch, tmp_path):
    trainer, _core, _memory, _calls = _trainer_fixture(monkeypatch, tmp_path)

    def load(_path, **kwargs):
        kwargs["scheduler"].T_max = 15
        return {"epoch": 4, "stage_state": {"name": "full", "ts_epoch": 4}}

    monkeypatch.setattr(trainer_module, "load_encoder_pc_training_resume", load)

    with pytest.raises(RuntimeError, match="expected 30, got 15"):
        trainer.resume("old-t-max-15-resume.pth", restore_rng=False)


def test_encoder_ts_final_memory_is_rebuilt_from_online_student_and_fingerprinted(
    monkeypatch, tmp_path
):
    trainer, core, memory, rebuild_calls = _trainer_fixture(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    monkeypatch.setattr(
        trainer_module,
        "save_encoder_pc_checkpoint",
        lambda path, **kwargs: captured.update(path=path, **kwargs),
    )

    model_path, memory_path = trainer._finalize_artifacts(1)

    assert rebuild_calls[-1][1] is core.student_encoder_pc_hbm
    assert rebuild_calls[-1][5]["producer_source"] == "student_final"
    fingerprint = module_fingerprint(core.student_encoder_pc_hbm)
    assert memory.compat_meta["producer_fingerprint"] == fingerprint
    assert captured["artifact_meta"]["producer_fingerprint"] == fingerprint
    assert captured["artifact_meta"]["dino_weight_fingerprint"] == module_fingerprint(
        core.dino
    )
    assert captured["encoder_pc_hbm"] is core.student_encoder_pc_hbm
    assert captured["model_role"] == "student"
    assert captured["training_design"] == "teacher_student"
    assert model_path.name == "encoder_pc_ts_student_v3.pth"
    assert memory_path.name == "encoder_pc_ts_memory_v3.pth"
    assert memory_path.exists()


def test_encoder_ts_cli_isolates_encoder_config_and_trainer(monkeypatch, tmp_path):
    args = SimpleNamespace(
        training_design="teacher_only",
        experiment_profile="encoder_pc",
        teacher_pc_checkpoint="base-v3.pth",
        student_checkpoint=None,
        output_dir=str(tmp_path),
        resume=None,
        allow_legacy_pc_init=False,
        labeled_indices_pt="split.pt",
        epochs=2,
        num_workers=0,
        memory_batch_size=1,
        memory_num_workers=0,
        seed=7,
        deterministic=True,
    )
    context = SimpleNamespace(rank=0, device=torch.device("cpu"))
    cfg = SimpleNamespace()
    model = SimpleNamespace(to=lambda _device: model)
    calls = {}

    monkeypatch.setattr(ts_entrypoint, "parse_args", lambda: args)
    monkeypatch.setattr(ts_entrypoint, "init_distributed", lambda: context)
    monkeypatch.setattr(ts_entrypoint, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(ts_entrypoint, "Config", lambda: cfg)
    monkeypatch.setattr(
        ts_entrypoint, "configure_distributed", lambda *args, **kwargs: None
    )
    def construct_model(**kwargs):
        calls["model_kwargs"] = kwargs
        return model

    monkeypatch.setattr(ts_entrypoint, "TSModel", construct_model)

    def wrap(wrapped, _context, **kwargs):
        calls["ddp"] = kwargs
        return wrapped

    monkeypatch.setattr(ts_entrypoint, "wrap_distributed", wrap)
    monkeypatch.setattr(
        ts_entrypoint,
        "PCHBMPseudoTrainer",
        lambda *args, **kwargs: pytest.fail("legacy TS trainer was selected"),
    )

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            calls["trainer"] = (args, kwargs)

        def train(self):
            calls["trained"] = True

    monkeypatch.setattr(ts_entrypoint, "EncoderPCTSTrainer", FakeTrainer)
    monkeypatch.setattr(ts_entrypoint, "cleanup_distributed", lambda: None)

    ts_entrypoint.main()

    assert isinstance(calls["model_kwargs"]["pc_cfg"], EncoderPCHBMConfig)
    assert calls["model_kwargs"]["teacher_pth"] == "base-v3.pth"
    assert calls["ddp"]["find_unused_parameters"] is True
    assert calls["trained"] is True
    assert cfg.l_batch_size == cfg.u_batch_size == 32


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"training_design": "joint"}, "fixed EMA"),
        ({"student_checkpoint": "student.pth"}, "Base v3 artifact"),
        ({"allow_legacy_pc_init": True}, "legacy"),
    ),
)
def test_encoder_ts_cli_rejects_noncanonical_initialization(overrides, message):
    values = {
        "training_design": "teacher_only",
        "experiment_profile": "encoder_pc",
        "allow_legacy_pc_init": False,
        "student_checkpoint": None,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        ts_entrypoint.validate_training_args(SimpleNamespace(**values))
