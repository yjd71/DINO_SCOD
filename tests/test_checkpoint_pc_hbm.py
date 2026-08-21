from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory
from tools.export_non_pc_decoder import export_non_pc_decoder
from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    LabeledSplitIdentity,
    build_artifact_metadata,
    extract_non_pc_decoder_state,
    load_decoder_compatible,
    load_memory_checkpoint,
    load_training_resume,
    read_artifact_metadata,
    read_pc_config,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
    state_dict_fingerprint,
    validate_canonical_labeled_indices_pt,
    validate_base_student_checkpoint,
    validate_canonical_labeled_split_fingerprint,
    validate_labeled_indices_pt,
    validate_labeled_sample_txt,
    validate_labeled_sample_keys,
    validate_labeled_split_source,
)
from utils.pc_memory_runner import (
    _validate_canonical_memory_split,
    _validate_memory_split,
)


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


def test_decoder_v3_round_trip_and_baseline_only_loading(tmp_path):
    cfg = DinoPCHBMConfig()
    source = TinyDecoder()
    path = tmp_path / "decoder.pth"
    payload = save_decoder_checkpoint(
        path,
        source,
        cfg,
        3,
        artifact_meta=_metadata("decoder"),
    )
    assert payload["child_verifier_version"] == 3
    assert payload["child_verification_mode"] == "parent_conditioned"
    assert payload["pc_cfg"]["verification_strength_init"] == pytest.approx(
        0.25
    )
    for removed_name in (
            "lambda_candidate_verify",
            "feature_distill_p3_weight",
    ):
        assert removed_name not in payload["pc_cfg"]
    target = TinyDecoder()
    load_decoder_compatible(
        target,
        path,
        require_pc_complete=True,
        expected_pc_cfg=cfg,
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


def test_decoder_v3_rejects_missing_child_verifier_metadata(tmp_path):
    cfg = DinoPCHBMConfig()
    source = TinyDecoder()
    path = tmp_path / "decoder.pth"
    payload = save_decoder_checkpoint(path, source, cfg, 1)
    payload.pop("child_verifier_version")
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="child_verifier_version=3"):
        load_decoder_compatible(
            TinyDecoder(),
            path,
            require_pc_complete=True,
            expected_pc_cfg=cfg,
        )


def test_decoder_rejects_removed_loss_controls_before_mutation(tmp_path):
    cfg = DinoPCHBMConfig()
    source = TinyDecoder()
    path = tmp_path / "decoder_with_obsolete_loss_controls.pth"
    payload = save_decoder_checkpoint(path, source, cfg, 1)
    payload["pc_cfg"].update(
        {
            "lambda_candidate_verify": 0.5,
            "lambda_pair": 1.0,
            "feature_distill_p3_weight": 1.0,
        }
    )
    torch.save(payload, path)

    target = TinyDecoder()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="invalid PC-HBM-Lite config"):
        load_decoder_compatible(
            target,
            path,
            require_pc_complete=True,
            expected_pc_cfg=cfg,
        )
    for name, value in before.items():
        assert torch.equal(value, target.state_dict()[name])


def test_decoder_config_is_reconstructed_and_mismatch_is_atomic(tmp_path):
    cfg = DinoPCHBMConfig(
        route_top_img_k=6,
        tau_parent=0.12,
        p3_top_ratio=0.2,
    )
    source = TinyDecoder()
    with torch.no_grad():
        source.baseline.weight.fill_(7.0)
    path = tmp_path / "custom_decoder.pth"
    save_decoder_checkpoint(
        path,
        source,
        cfg,
        2,
        artifact_meta=_metadata("decoder"),
    )

    restored_cfg = read_pc_config(path, context="test Decoder")
    assert vars(restored_cfg) == vars(cfg)

    target = TinyDecoder()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="config mismatch"):
        load_decoder_compatible(
            target,
            path,
            require_pc_complete=True,
            expected_pc_cfg=DinoPCHBMConfig(),
        )
    for key, value in before.items():
        assert torch.equal(value, target.state_dict()[key])


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
            pc_cfg=cfg,
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
            pc_cfg=cfg,
            restore_rng=True,
        )
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


def test_resume_restores_student_and_exact_rng_continuation(tmp_path):
    cfg = DinoPCHBMConfig()
    model = TinyDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    resume_path = tmp_path / "deterministic_resume.pth"
    save_training_resume(
        resume_path,
        epoch=2,
        model=model,
        optimizer=optimizer,
        pc_cfg=cfg,
        artifact_meta=_metadata(),
    )
    expected = (random.random(), np.random.rand(), torch.rand(4))
    saved_model = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    load_training_resume(
        resume_path,
        model=model,
        optimizer=optimizer,
        pc_cfg=cfg,
        restore_rng=True,
    )

    for name, value in saved_model.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2])


def test_resume_config_mismatch_is_rejected_before_mutation(tmp_path):
    cfg = DinoPCHBMConfig(tau_child=0.2)
    model = TinyDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    payload = save_training_resume(
        tmp_path / "resume_config.pth",
        epoch=2,
        model=model,
        optimizer=optimizer,
        pc_cfg=cfg,
        artifact_meta=_metadata(),
    )
    payload["model"]["baseline.weight"] = torch.full_like(
        payload["model"]["baseline.weight"],
        9.0,
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="config mismatch"):
        load_training_resume(
            payload,
            model=model,
            optimizer=optimizer,
            pc_cfg=DinoPCHBMConfig(),
            restore_rng=True,
        )
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route_environment_min_mass", 0.02),
        ("fg_boundary_kernel", 5),
        ("bg_near_kernel", 9),
        ("gt_binary_threshold", 0.4),
        ("region_max_quota", (40, 48)),
        ("region_min_quota", (4, 8)),
        ("region_sampling_ratio", (0.25, 0.5)),
    ],
)
def test_memory_builder_config_mismatch_is_atomic(field, value):
    source_cfg = DinoPCHBMConfig(**{field: value})
    source = _ready_memory(source_cfg)
    target = PCMemory(config=DinoPCHBMConfig())
    with pytest.raises(ValueError, match=field):
        target.load_state_dict(source.state_dict())
    assert not target.is_ready()


def test_memory_compat_excludes_training_only_configuration():
    baseline = DinoPCHBMConfig().expected_memory_meta()
    training_only = DinoPCHBMConfig(
        verify_start_epoch=2,
        full_pc_start_epoch=4,
        teacher_only_full_start_epoch=3,
        pc_injection_ramp_epochs=2,
        lambda_u=0.7,
        use_amp=False,
        grad_clip_norm=0.0,
        ema_momentum=1.0,
        diagnostic_window_epochs=5,
        warn_low_pair_valid_ratio=0.2,
        warn_pair_acc_near_random=0.2,
        warn_gate_inactive_threshold=0.2,
        warn_delta_large_threshold=0.0,
    ).expected_memory_meta()
    assert training_only == baseline


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


@pytest.mark.parametrize("count", [202, 404])
def test_run_labeled_split_accepts_variable_cardinality(tmp_path, count):
    keys = [f"COD10K/sample_{index:04d}" for index in range(count)]
    split_path = tmp_path / f"keys_{count}.pt"
    torch.save(keys, split_path)

    identity = validate_labeled_indices_pt(split_path)
    assert identity == LabeledSplitIdentity(
        count=count,
        fingerprint=identity.fingerprint,
    )

    loader = SimpleNamespace(dataset=SimpleNamespace(sample_keys=keys))
    assert _validate_memory_split(
        loader,
        expected_count=count,
        expected_fingerprint=identity.fingerprint,
    ) == identity


def test_run_labeled_split_rejects_duplicates_and_contract_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="unique"):
        validate_labeled_sample_keys(["CAMO/a", "CAMO\\a"])

    split_path = tmp_path / "keys.pt"
    torch.save(["CAMO/a", "CAMO/b"], split_path)
    with pytest.raises(RuntimeError, match="count differs"):
        validate_labeled_indices_pt(split_path, expected_count=3)
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        validate_labeled_indices_pt(
            split_path,
            expected_fingerprint="0" * 64,
        )


def test_labeled_split_source_falls_back_to_txt_and_pt_overrides(tmp_path):
    txt_path = tmp_path / "sampled_images.txt"
    txt_path.write_text("TR-CAMO/a\nTR-COD10K/b\n", encoding="utf-8")
    txt_identity = validate_labeled_sample_txt(txt_path)
    assert txt_identity.count == 2
    assert validate_labeled_split_source(None, txt_path) == txt_identity

    pt_path = tmp_path / "selected.pt"
    torch.save(["TR-CAMO/c"], pt_path)
    pt_identity = validate_labeled_indices_pt(pt_path)
    assert validate_labeled_split_source(pt_path, txt_path) == pt_identity


def test_base_student_checkpoint_requires_complete_state_and_matching_fingerprint(
    tmp_path,
):
    cfg = DinoPCHBMConfig()
    decoder = Decoder(
        in_dim=cfg.encoder_dim,
        out_dim=cfg.decoder_dim,
        pc_cfg=cfg,
    )
    non_pc_fingerprint = state_dict_fingerprint(
        {
            name: value
            for name, value in decoder.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
    )
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="base_student",
        labeled_split_fingerprint=CANONICAL_LABELED_SPLIT_FINGERPRINT,
        baseline_fingerprint=non_pc_fingerprint,
        pc_frozen=False,
    )
    payload = save_decoder_checkpoint(
        tmp_path / "base_student.pth",
        decoder,
        cfg,
        1,
        artifact_meta=metadata,
    )
    validated = validate_base_student_checkpoint(
        payload, CANONICAL_LABELED_SPLIT_FINGERPRINT
    )
    assert validated["artifact_role"] == "base_student"

    incomplete = copy.deepcopy(payload)
    pc_key = next(
        name for name in incomplete["decoder"] if name.startswith("pc_hbm.")
    )
    incomplete["decoder"].pop(pc_key)
    with pytest.raises(RuntimeError, match="not a complete Decoder state"):
        validate_base_student_checkpoint(
            incomplete, CANONICAL_LABELED_SPLIT_FINGERPRINT
        )
