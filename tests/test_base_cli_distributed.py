import sys

import pytest

import train_base_model_pc_hbm as base_entrypoint
from utils import distributed


def _parse_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train_base_model_pc_hbm.py", *args])
    return base_entrypoint.parse_args()


def test_base_batch_size_defaults_to_config_value(monkeypatch):
    args = _parse_args(monkeypatch)

    assert args.batch_size is None


def test_base_batch_size_is_explicitly_per_rank(monkeypatch):
    args = _parse_args(monkeypatch, "--batch-size", "16")

    assert args.batch_size == 16


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_base_batch_size_rejects_non_positive_or_invalid_values(monkeypatch, value):
    with pytest.raises(SystemExit):
        _parse_args(monkeypatch, "--batch-size", value)


@pytest.mark.parametrize("backend", ["nccl", distributed.dist.Backend.NCCL])
def test_synchronize_binds_nccl_barrier_to_current_device(monkeypatch, backend):
    barrier_calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda **kwargs: barrier_calls.append(kwargs),
    )

    distributed.synchronize()

    assert barrier_calls == [{"device_ids": [3]}]


def test_synchronize_keeps_nccl_device_agnostic_without_cuda(monkeypatch):
    barrier_calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        distributed.torch.cuda,
        "current_device",
        lambda: pytest.fail("current_device must not run without CUDA"),
    )
    monkeypatch.setattr(distributed.dist, "barrier", lambda: barrier_calls.append(None))

    distributed.synchronize()

    assert barrier_calls == [None]


def test_synchronize_keeps_gloo_barrier_device_agnostic(monkeypatch):
    barrier_calls = []
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed.dist, "barrier", lambda: barrier_calls.append(None))

    distributed.synchronize()

    assert barrier_calls == [None]


def test_synchronize_is_a_noop_without_process_group(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "get_backend",
        lambda: pytest.fail("get_backend must not run without a process group"),
    )
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda: pytest.fail("barrier must not run without a process group"),
    )

    distributed.synchronize()
