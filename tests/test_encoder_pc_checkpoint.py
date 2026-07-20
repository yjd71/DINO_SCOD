from __future__ import annotations

from itertools import chain
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.checkpoint_pc_hbm import (
    ENCODER_PC_ARCHITECTURE,
    ENCODER_PC_ARTIFACT_KIND,
    ENCODER_PC_FORMAT_VERSION,
    ENCODER_PC_MODEL_ARCHITECTURE,
    ENCODER_PC_RESUME_KIND,
    load_bgfbr_decoder_warm_start,
    load_encoder_pc_checkpoint,
    load_encoder_pc_training_resume,
    save_encoder_pc_checkpoint,
    save_encoder_pc_training_resume,
)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 3)


class TinyDetachedBGFBR(nn.Module):
    decoder_architecture = "bgfbr_pc_v1"
    decoder_contract_version = 1

    def __init__(self, *, attached: bool = False):
        super().__init__()
        self.project = nn.Linear(3, 2)
        self.register_buffer("contract_buffer", torch.tensor([7.0]))
        if attached:
            self.pc_hbm = nn.Linear(2, 2)


class TinyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct = nn.Conv2d(2, 1, 1)


class TinyScaler:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = float(state["scale"])


def _config():
    return {
        "architecture": ENCODER_PC_ARCHITECTURE,
        "memory_schema_version": 3,
        "memory_dim": 128,
    }


def _zero(module: nn.Module) -> None:
    with torch.no_grad():
        for tensor in module.state_dict().values():
            tensor.zero_()


def _assert_same_state(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key]), key


def _parameters(*modules: nn.Module):
    return list(chain.from_iterable(module.parameters() for module in modules))


def test_encoder_pc_loader_rejects_live_config_drift(tmp_path: Path):
    adapter = TinyAdapter()
    decoder = TinyDetachedBGFBR()
    refiner = TinyRefiner()
    path = tmp_path / "strict_config_v3.pt"
    save_encoder_pc_checkpoint(
        path,
        epoch=30,
        encoder_pc_hbm=adapter,
        decoder=decoder,
        pseudo_refiner=refiner,
        config=_config(),
        model_role="base",
        training_design="encoder_pc_base",
    )
    drifted = {**_config(), "memory_dim": 64}

    with pytest.raises(RuntimeError, match="live contract"):
        load_encoder_pc_checkpoint(
            path,
            encoder_pc_hbm=TinyAdapter(),
            decoder=TinyDetachedBGFBR(),
            pseudo_refiner=TinyRefiner(),
            expected_model_role="base",
            expected_training_design="encoder_pc_base",
            expected_config=drifted,
        )


def test_encoder_pc_v3_artifact_round_trip_and_metadata(tmp_path: Path):
    torch.manual_seed(11)
    source_adapter = TinyAdapter()
    source_decoder = TinyDetachedBGFBR()
    source_refiner = TinyRefiner()
    checkpoint_path = tmp_path / "encoder_pc_v3.pt"

    payload = save_encoder_pc_checkpoint(
        checkpoint_path,
        epoch=30,
        encoder_pc_hbm=source_adapter,
        decoder=source_decoder,
        pseudo_refiner=source_refiner,
        config=_config(),
        model_role="base",
        training_design="encoder_pc_base",
        artifact_meta={"split_fingerprint": "split-a", "producer_fingerprint": "adapter-a"},
    )

    assert payload["format_version"] == ENCODER_PC_FORMAT_VERSION
    assert payload["architecture"] == ENCODER_PC_ARCHITECTURE
    assert payload["model_architecture"] == ENCODER_PC_MODEL_ARCHITECTURE
    assert payload["artifact_kind"] == ENCODER_PC_ARTIFACT_KIND
    assert payload["artifact_meta"]["model_role"] == "base"
    assert payload["artifact_meta"]["training_design"] == "encoder_pc_base"
    assert not any(key.startswith("pc_hbm.") for key in payload["decoder"])

    target_adapter = TinyAdapter()
    target_decoder = TinyDetachedBGFBR()
    target_refiner = TinyRefiner()
    for module in (target_adapter, target_decoder, target_refiner):
        _zero(module)
    loaded = load_encoder_pc_checkpoint(
        checkpoint_path,
        encoder_pc_hbm=target_adapter,
        decoder=target_decoder,
        pseudo_refiner=target_refiner,
        expected_model_role="base",
        expected_training_design="encoder_pc_base",
        expected_artifact_meta={
            "split_fingerprint": "split-a",
            "producer_fingerprint": "adapter-a",
        },
    )

    assert loaded["epoch"] == 30
    _assert_same_state(source_adapter, target_adapter)
    _assert_same_state(source_decoder, target_decoder)
    _assert_same_state(source_refiner, target_refiner)


def test_encoder_pc_loader_rejects_legacy_format_role_and_design(tmp_path: Path):
    adapter = TinyAdapter()
    decoder = TinyDetachedBGFBR()
    refiner = TinyRefiner()
    path = tmp_path / "artifact.pt"
    payload = save_encoder_pc_checkpoint(
        path,
        epoch=1,
        encoder_pc_hbm=adapter,
        decoder=decoder,
        pseudo_refiner=refiner,
        config=_config(),
        model_role="student",
        training_design="encoder_pc_teacher_student",
    )

    common = dict(
        encoder_pc_hbm=TinyAdapter(),
        decoder=TinyDetachedBGFBR(),
        pseudo_refiner=TinyRefiner(),
        expected_model_role="student",
        expected_training_design="encoder_pc_teacher_student",
    )
    with pytest.raises(RuntimeError, match="format v3"):
        load_encoder_pc_checkpoint({"format_version": 2}, **common)
    with pytest.raises(RuntimeError, match="model_role mismatch"):
        load_encoder_pc_checkpoint(
            payload,
            **{**common, "expected_model_role": "base"},
        )
    with pytest.raises(RuntimeError, match="training_design mismatch"):
        load_encoder_pc_checkpoint(
            payload,
            **{**common, "expected_training_design": "encoder_pc_base"},
        )


def test_encoder_pc_artifact_rejects_decoder_side_pc_state(tmp_path: Path):
    with pytest.raises(RuntimeError, match="attach_pc=False"):
        save_encoder_pc_checkpoint(
            tmp_path / "invalid.pt",
            epoch=1,
            encoder_pc_hbm=TinyAdapter(),
            decoder=TinyDetachedBGFBR(attached=True),
            pseudo_refiner=TinyRefiner(),
            config=_config(),
            model_role="base",
            training_design="encoder_pc_base",
        )


def test_encoder_pc_training_resume_restores_full_state_and_rng(tmp_path: Path):
    torch.manual_seed(21)
    source_adapter = TinyAdapter()
    source_decoder = TinyDetachedBGFBR()
    source_refiner = TinyRefiner()
    source_modules = (source_adapter, source_decoder, source_refiner)
    optimizer = torch.optim.Adam(_parameters(*source_modules), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for parameter in _parameters(*source_modules):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    scaler = TinyScaler(1024.0)
    ema_adapter = TinyAdapter()
    ema_decoder = TinyDetachedBGFBR()
    ema_refiner = TinyRefiner()
    stage_state = {"name": "hierarchical_refiner", "epoch": 23, "progress": 0.6}
    split_state = {"labeled_split_fingerprint": "split-23", "round": 2}
    memory_profile = {"schema_version": 3, "producer_fingerprint": "ema-23"}

    torch.manual_seed(12345)
    resume_path = tmp_path / "resume_v3.pt"
    payload = save_encoder_pc_training_resume(
        resume_path,
        epoch=23,
        encoder_pc_hbm=source_adapter,
        decoder=source_decoder,
        pseudo_refiner=source_refiner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema_adapter=ema_adapter,
        ema_decoder=ema_decoder,
        ema_refiner=ema_refiner,
        config=_config(),
        stage_state=stage_state,
        split_state=split_state,
        memory_profile=memory_profile,
        model_role="student",
        training_design="encoder_pc_teacher_student",
        artifact_meta={"run_id": "ts-a"},
    )
    expected_next_random = torch.rand(5)
    assert payload["artifact_kind"] == ENCODER_PC_RESUME_KIND
    assert payload["stage_state"] == stage_state
    assert payload["split_state"] == split_state
    assert payload["memory_profile"] == memory_profile

    target_adapter = TinyAdapter()
    target_decoder = TinyDetachedBGFBR()
    target_refiner = TinyRefiner()
    target_modules = (target_adapter, target_decoder, target_refiner)
    target_optimizer = torch.optim.Adam(_parameters(*target_modules), lr=9e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(
        target_optimizer, step_size=4, gamma=0.9
    )
    target_scaler = TinyScaler(1.0)
    target_ema_adapter = TinyAdapter()
    target_ema_decoder = TinyDetachedBGFBR()
    target_ema_refiner = TinyRefiner()
    for module in (
        *target_modules,
        target_ema_adapter,
        target_ema_decoder,
        target_ema_refiner,
    ):
        _zero(module)
    torch.manual_seed(999)

    loaded = load_encoder_pc_training_resume(
        resume_path,
        encoder_pc_hbm=target_adapter,
        decoder=target_decoder,
        pseudo_refiner=target_refiner,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        scaler=target_scaler,
        ema_adapter=target_ema_adapter,
        ema_decoder=target_ema_decoder,
        ema_refiner=target_ema_refiner,
        expected_model_role="student",
        expected_training_design="encoder_pc_teacher_student",
        expected_split_state=split_state,
        expected_memory_profile=memory_profile,
        expected_artifact_meta={"run_id": "ts-a"},
    )

    assert loaded["epoch"] == 23
    assert torch.equal(torch.rand(5), expected_next_random)
    assert target_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert target_scheduler.state_dict() == scheduler.state_dict()
    assert target_scaler.scale == scaler.scale
    _assert_same_state(source_adapter, target_adapter)
    _assert_same_state(source_decoder, target_decoder)
    _assert_same_state(source_refiner, target_refiner)
    _assert_same_state(ema_adapter, target_ema_adapter)
    _assert_same_state(ema_decoder, target_ema_decoder)
    _assert_same_state(ema_refiner, target_ema_refiner)


def test_encoder_pc_resume_is_strict_about_kind_and_ema(tmp_path: Path):
    adapter = TinyAdapter()
    decoder = TinyDetachedBGFBR()
    refiner = TinyRefiner()
    optimizer = torch.optim.SGD(_parameters(adapter, decoder, refiner), lr=0.1)
    path = tmp_path / "resume.pt"
    payload = save_encoder_pc_training_resume(
        path,
        epoch=1,
        encoder_pc_hbm=adapter,
        decoder=decoder,
        pseudo_refiner=refiner,
        optimizer=optimizer,
        config=_config(),
        stage_state={"name": "bootstrap"},
        split_state={"fingerprint": "split"},
        memory_profile={"schema_version": 3},
        model_role="base",
        training_design="encoder_pc_base",
    )
    common = dict(
        encoder_pc_hbm=TinyAdapter(),
        decoder=TinyDetachedBGFBR(),
        pseudo_refiner=TinyRefiner(),
        expected_model_role="base",
        expected_training_design="encoder_pc_base",
        restore_rng=False,
    )
    wrong_kind = dict(payload)
    wrong_kind["artifact_kind"] = ENCODER_PC_ARTIFACT_KIND
    with pytest.raises(RuntimeError, match="kind mismatch"):
        load_encoder_pc_training_resume(wrong_kind, **common)
    with pytest.raises(RuntimeError, match="ema_adapter"):
        load_encoder_pc_training_resume(payload, ema_adapter=TinyAdapter(), **common)


def test_bgfbr_warm_start_loads_all_non_pc_and_only_ignores_pc():
    torch.manual_seed(33)
    legacy = TinyDetachedBGFBR(attached=True)
    target = TinyDetachedBGFBR(attached=False)
    _zero(target)
    checkpoint = {
        "format_version": 2,
        "decoder_architecture": "bgfbr_pc_v1",
        "decoder_contract_version": 1,
        "decoder": legacy.state_dict(),
        "optimizer": {"must_not_migrate": True},
        "memory": {"must_not_migrate": True},
    }

    result = load_bgfbr_decoder_warm_start(
        target,
        checkpoint,
        drop_prefixes=("pc_hbm.",),
        strict_non_pc=True,
    )

    for key, value in target.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key]), key
    assert set(result["ignored_pc_keys"]) == {"pc_hbm.weight", "pc_hbm.bias"}
    assert set(result["loaded_keys"]) == set(target.state_dict())


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_bgfbr_warm_start_rejects_non_pc_key_drift(mutation: str):
    source = TinyDetachedBGFBR(attached=True).state_dict()
    source = dict(source)
    if mutation == "missing":
        source.pop("project.bias")
    else:
        source["unrelated.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="non-PC keys"):
        load_bgfbr_decoder_warm_start(
            TinyDetachedBGFBR(),
            {"decoder_architecture": "bgfbr_pc_v1", "decoder": source},
        )


def test_bgfbr_warm_start_rejects_attached_target():
    with pytest.raises(RuntimeError, match="detached BGFBR Decoder"):
        load_bgfbr_decoder_warm_start(
            TinyDetachedBGFBR(attached=True),
            {"decoder": TinyDetachedBGFBR(attached=True).state_dict()},
        )
