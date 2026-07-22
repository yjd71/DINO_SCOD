import re
import sys

import pytest
import torch

import train_base_model_pc_hbm as base_entrypoint
from utils import distributed
from utils.dataloader import _load_sample_keys
from utils.logging_utils import current_time


def _parse_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train_base_model_pc_hbm.py", *args])
    return base_entrypoint.parse_args()


def test_log_time_uses_month_day_clock_format():
    assert re.fullmatch(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}", current_time())


def test_base_batch_size_defaults_to_config_value(monkeypatch):
    args = _parse_args(monkeypatch)

    assert args.batch_size is None


def test_base_batch_size_is_explicitly_per_rank(monkeypatch):
    args = _parse_args(monkeypatch, "--batch-size", "16")

    assert args.batch_size == 16


def test_two_stage_allows_sampled_images_fallback(monkeypatch):
    args = _parse_args(monkeypatch)

    assert args.training_design == "two_stage"
    assert args.labeled_indices_pt is None
    base_entrypoint.validate_training_args(args)


def test_teacher_only_allows_sampled_images_fallback_with_baseline(monkeypatch):
    args = _parse_args(
        monkeypatch,
        "--training-design",
        "teacher_only",
        "--experiment-profile",
        "legacy_pc",
        "--baseline-checkpoint",
        "baseline_decoder.pth",
    )

    assert args.labeled_indices_pt is None
    base_entrypoint.validate_training_args(args)


def test_teacher_only_still_requires_baseline_checkpoint(monkeypatch):
    args = _parse_args(monkeypatch, "--training-design", "teacher_only")

    with pytest.raises(ValueError, match="baseline-checkpoint"):
        base_entrypoint.validate_training_args(args)


def test_labeled_selection_falls_back_to_sampled_txt(tmp_path):
    items = [
        {"key": "TR-CAMO/camo_001", "stem": "camo_001"},
        {"key": "TR-COD10K/cod10k_001", "stem": "cod10k_001"},
    ]
    sampled_txt = tmp_path / "sampled_images.txt"
    sampled_txt.write_text("TR-CAMO/camo_001\n", encoding="utf-8")

    selected = _load_sample_keys(str(sampled_txt), None, items)

    assert selected == {"TR-CAMO/camo_001"}


def test_labeled_indices_pt_overrides_sampled_txt(tmp_path):
    items = [
        {"key": "TR-CAMO/camo_001", "stem": "camo_001"},
        {"key": "TR-COD10K/cod10k_001", "stem": "cod10k_001"},
    ]
    sampled_txt = tmp_path / "sampled_images.txt"
    sampled_txt.write_text("TR-CAMO/camo_001\n", encoding="utf-8")
    indices_pt = tmp_path / "labeled_indices.pt"
    torch.save(["TR-COD10K/cod10k_001"], indices_pt)

    selected = _load_sample_keys(str(sampled_txt), str(indices_pt), items)

    assert selected == {"TR-COD10K/cod10k_001"}


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
