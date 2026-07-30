from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory import PCMemory
from tools.export_non_pc_decoder import export_non_pc_decoder
from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    build_artifact_metadata,
    extract_non_pc_decoder_state,
    load_decoder_compatible,
    load_memory_checkpoint,
    load_training_resume,
    read_artifact_metadata,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
    validate_canonical_labeled_indices_pt,
    validate_canonical_labeled_split_fingerprint,
)
from utils.pc_memory_runner import _validate_canonical_memory_split


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = nn.Linear(3, 2)
        self.pc_hbm = nn.Linear(2, 2)


def _metadata(role="resume"):
    return build_artifact_metadata(
        training_design="two_stage",
        artifact_role=role,
        labeled_split_fingerprint="split",
        baseline_fingerprint="baseline",
        pc_frozen=False,
    )


def _ready_memory(cfg):
    memory = PCMemory(config=cfg)
    state = {
        "format_version": 2,
        "schema_version": 2,
        "compat_meta": cfg.expected_memory_meta(
            producer_fingerprint="a" * 64
        ),
        "memory_dim": 128,
        "storage_dtype": "float16",
        "route": {
            "global_keys": torch.randn(1, 128).half(),
            "environment_keys": torch.randn(1, 128).half(),
            "img_ids": ["sample"],
        },
        "pairs": {
            "p3_keys": torch.randn(2, 128).half(),
            "p2_keys": torch.randn(2, 128).half(),
            "region_ids": torch.tensor([0, 1]),
            "pair_meta": [
                {"image_id": "sample", "region_id": 0},
                {"image_id": "sample", "region_id": 1},
            ],
        },
        "finalized": True,
    }
    memory.load_state_dict(state)
    return memory


def test_decoder_v2_round_trip_and_baseline_only_loading(tmp_path):
    cfg = DinoPCHBMConfig()
    source = TinyDecoder()
    path = tmp_path / "decoder.pth"
    save_decoder_checkpoint(
        path,
        source,
        cfg,
        3,
        artifact_meta=_metadata("decoder"),
    )
    target = TinyDecoder()
    load_decoder_compatible(
        target,
        path,
        require_pc_complete=True,
    )
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    assert read_artifact_metadata(path)["artifact_metadata_version"] == 2

    baseline = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("pc_hbm.")
    }
    other = TinyDecoder()
    before_pc = copy.deepcopy(other.pc_hbm.state_dict())
    load_decoder_compatible(other, baseline)
    for key, value in before_pc.items():
        assert torch.equal(value, other.pc_hbm.state_dict()[key])


def test_old_pc_state_is_rejected_before_decoder_mutation():
    decoder = TinyDecoder()
    before = copy.deepcopy(decoder.state_dict())
    old = {"decoder": copy.deepcopy(decoder.state_dict())}
    old["decoder"]["baseline.weight"].zero_()
    with pytest.raises(RuntimeError, match="format_version"):
        load_decoder_compatible(
            decoder,
            old,
            require_pc_complete=True,
        )
    for key, value in before.items():
        assert torch.equal(value, decoder.state_dict()[key])


def test_memory_v2_round_trip_and_old_state_rejection(tmp_path):
    cfg = DinoPCHBMConfig()
    source = _ready_memory(cfg)
    path = tmp_path / "memory.pth"
    save_memory_checkpoint(path, source)
    target = PCMemory(config=cfg)
    load_memory_checkpoint(
        path,
        target,
        expected_compat=cfg.expected_memory_meta(),
    )
    assert target.is_ready()
    before = target.state_dict()
    with pytest.raises(RuntimeError):
        load_memory_checkpoint(
            {"format_version": 1, "memory": {"format_version": 1}},
            target,
        )
    assert torch.equal(
        before["pairs"]["p3_keys"],
        target.state_dict()["pairs"]["p3_keys"],
    )


def test_resume_preflight_rejects_bad_model_without_mutation(tmp_path):
    cfg = DinoPCHBMConfig()
    model = TinyDecoder()
    ema = TinyDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    path = tmp_path / "resume.pth"
    payload = save_training_resume(
        path,
        epoch=2,
        model=model,
        optimizer=optimizer,
        ema_model=ema,
        pc_cfg=cfg,
        artifact_meta=_metadata(),
    )
    bad = copy.deepcopy(payload)
    bad["model"]["baseline.weight"] = torch.randn(9, 9)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="tensor mismatch"):
        load_training_resume(
            bad,
            model=model,
            optimizer=optimizer,
            ema_model=ema,
            restore_rng=True,
        )
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


def test_resume_preflights_rng_before_model_mutation(tmp_path):
    cfg = DinoPCHBMConfig()
    model = TinyDecoder()
    ema = TinyDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    payload = save_training_resume(
        tmp_path / "resume_rng.pth",
        epoch=2,
        model=model,
        optimizer=optimizer,
        ema_model=ema,
        pc_cfg=cfg,
        artifact_meta=_metadata(),
    )
    bad = copy.deepcopy(payload)
    bad["model"]["baseline.weight"].zero_()
    bad["rng_state"]["python"] = "invalid"
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="RNG"):
        load_training_resume(
            bad,
            model=model,
            optimizer=optimizer,
            ema_model=ema,
            restore_rng=True,
        )
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


def test_explicit_export_discards_all_pc_keys(tmp_path):
    source = tmp_path / "old.pth"
    output = tmp_path / "baseline.pth"
    torch.save({"decoder": TinyDecoder().state_dict()}, source)
    payload = export_non_pc_decoder(source, output)
    assert output.is_file()
    assert all(
        not name.startswith("pc_hbm.") for name in payload["decoder"]
    )
    assert extract_non_pc_decoder_state(output).keys() == payload[
        "decoder"
    ].keys()


def test_canonical_labeled_split_is_fail_fast(tmp_path):
    assert validate_canonical_labeled_split_fingerprint(
        CANONICAL_LABELED_SPLIT_FINGERPRINT
    ) == CANONICAL_LABELED_SPLIT_FINGERPRINT
    with pytest.raises(RuntimeError, match="canonical 202-key"):
        validate_canonical_labeled_split_fingerprint("wrong")

    wrong_path = tmp_path / "wrong_keys.pt"
    torch.save(["sample"], wrong_path)
    with pytest.raises(RuntimeError, match="exactly 202"):
        validate_canonical_labeled_indices_pt(wrong_path)

    wrong_loader = SimpleNamespace(
        dataset=SimpleNamespace(sample_keys=["sample"])
    )
    with pytest.raises(RuntimeError, match="exactly 202"):
        _validate_canonical_memory_split(wrong_loader)
