from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
import utils.distributed as distributed
import utils.trainer_ts_model_pseudo_pc_hbm as ts_trainer
from utils.trainer_ts_model_pseudo_pc_hbm import PCHBMPseudoTrainer


def _teacher_aux_in_inference_mode():
    with torch.inference_mode():
        return {
            "p_final": torch.rand(2, 1, 98, 98),
            "z_main": torch.randn(2, 1, 98, 98),
            "pc_hbm": {
                "C23_map": torch.rand(2, 1, 28, 28),
                "route_entropy_norm": torch.rand(2),
            },
            "mixture": {"pi": torch.softmax(torch.randn(2, 4, 98, 98), dim=1)},
        }


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
    )
    assert all(not tensor.is_inference() for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_ts_decoder_epoch_continues_after_base_schedule():
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.pc_cfg = SimpleNamespace(mixture_schedule_end_epoch=30)
    assert trainer._decoder_epoch(1) == 31
    assert trainer._decoder_epoch(15) == 45


def test_train_prints_start_and_end_time_for_every_epoch(monkeypatch, capsys):
    trainer = object.__new__(PCHBMPseudoTrainer)
    trainer.current_epoch = 1
    trainer.cfg = SimpleNamespace(epochs=2)
    trainer.scheduler = SimpleNamespace(step=lambda: None)
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-4}])
    trainer.train_epoch = lambda: {"loss": 1.25, "confidence": 0.125}
    trainer._save_epoch = lambda epoch, metrics: None
    trainer._export_final_memory = lambda: None

    timestamps = iter(
        (
            "2026-07-13T01:00:00+08:00",
            "2026-07-13T01:05:00+08:00",
            "2026-07-13T01:05:01+08:00",
            "2026-07-13T01:10:00+08:00",
        )
    )
    monkeypatch.setattr(ts_trainer, "_current_local_timestamp", lambda: next(timestamps))
    monkeypatch.setattr(ts_trainer, "is_main_process", lambda: True)
    monkeypatch.setattr(ts_trainer, "synchronize", lambda: None)

    trainer.train()

    output = capsys.readouterr().out
    assert "epoch 1/2: start_time=2026-07-13T01:00:00+08:00" in output
    assert "epoch 1/2: loss=1.250000" in output
    assert "end_time=2026-07-13T01:05:00+08:00" in output
    assert "epoch 2/2: start_time=2026-07-13T01:05:01+08:00" in output
    assert "epoch 2/2: loss=1.250000" in output
    assert "end_time=2026-07-13T01:10:00+08:00" in output


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
