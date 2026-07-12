from types import SimpleNamespace

import torch

import utils.distributed as distributed
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
