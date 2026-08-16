from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.retrieval import PairVerifier
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    read_pc_config,
    save_decoder_checkpoint,
)


def _retrieval(
    parent_keys: torch.Tensor,
    child_keys: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if valid is None:
        valid = torch.ones(parent_keys.shape[:-1], dtype=torch.bool)
    return {
        "parent_keys": parent_keys,
        "paired_p2_keys": child_keys,
        "valid": valid,
    }


def _pcv(dim: int = 4, **kwargs) -> PairVerifier:
    options = {
        "dim": dim,
        "tau_parent": 1.0,
        "child_verification_mode": "parent_conditioned",
        "verification_strength_init": 0.25,
    }
    options.update(kwargs)
    return PairVerifier(**options)


def test_mode_specific_parameter_counts_and_identity_initialization() -> None:
    weighted = PairVerifier(dim=128, child_verification_mode="weighted_sum")
    pcv = PairVerifier(dim=128, child_verification_mode="parent_conditioned")

    assert sum(parameter.numel() for parameter in weighted.parameters()) == 1
    assert sum(parameter.numel() for parameter in pcv.parameters()) == 16_385
    assert not hasattr(pcv, "raw_child_mix")
    assert not hasattr(pcv, "raw_verification_abs_weight")
    assert not hasattr(pcv, "raw_verification_rel_weight")
    assert not hasattr(pcv, "verification_bias")
    assert torch.equal(pcv.parent_to_child.weight, torch.eye(128))
    torch.testing.assert_close(
        pcv.verification_strength, torch.tensor(0.25), rtol=0.0, atol=1.0e-7
    )


def test_zero_verify_logit_is_bitwise_parent_neutral() -> None:
    verifier = _pcv()
    q3 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_child = q3.clone()
    parent = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0, 0.0]]]])
    child = parent.clone()

    result = verifier(q3, q_child, _retrieval(parent, child), torch.ones(1, 1))

    assert result["child_match_logits"] is result["child_verify_logits"]
    assert torch.count_nonzero(result["child_match_logits"]) == 0
    assert torch.equal(result["verified_scores"], result["parent_scores"])
    assert torch.equal(result["pair_scores"], result["parent_scores"])


def test_absolute_child_evidence_supports_and_contradicts_candidates() -> None:
    verifier = _pcv()
    q3 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_child = q3.clone()
    parent = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]]])
    child = parent.clone()

    result = verifier(q3, q_child, _retrieval(parent, child), torch.ones(1, 1))

    assert result["child_abs_cosine"][0, 0, 0] == pytest.approx(1.0)
    assert result["child_abs_cosine"][0, 1, 0] == pytest.approx(-1.0)
    torch.testing.assert_close(
        result["child_match_logits"], 0.5 * result["child_abs_cosine"]
    )
    assert result["child_match_logits"][0, 0, 0] > 0.0
    assert result["child_match_logits"][0, 1, 0] < 0.0
    assert result["verified_scores"][0, 0, 0] > result["parent_scores"][0, 0, 0]
    assert result["verified_scores"][0, 1, 0] < result["parent_scores"][0, 1, 0]


def test_relation_cosine_uses_identity_alignment_and_masks_small_residuals() -> None:
    verifier = _pcv(relation_norm_eps=1.0e-4)
    q3 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_child = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    parent = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]]])
    child = parent + torch.tensor([0.0, 1.0, 0.0, 0.0])

    result = verifier(q3, q_child, _retrieval(parent, child), torch.ones(1, 1))
    assert result["relation_valid"].all()
    torch.testing.assert_close(
        result["child_relation_cosine"], torch.ones(1, 2, 1)
    )
    torch.testing.assert_close(
        result["child_match_logits"],
        0.5
        * (
            result["child_abs_cosine"]
            + result["child_relation_cosine"]
        ),
    )

    neutral = verifier(
        q3,
        q3.clone(),
        _retrieval(parent, parent.clone()),
        torch.ones(1, 1),
    )
    assert not neutral["relation_valid"].any()
    assert torch.count_nonzero(neutral["child_relation_cosine"]) == 0
    torch.testing.assert_close(
        neutral["child_match_logits"],
        0.5 * neutral["child_abs_cosine"],
    )


def test_verification_conditioning_detaches_parent_inputs_but_trains_child_path() -> None:
    verifier = _pcv()
    q3 = torch.randn(2, 4, requires_grad=True)
    q_child = torch.randn(2, 4, requires_grad=True)
    parent = torch.randn(2, 2, 2, 4, requires_grad=True)
    child = torch.randn(2, 2, 2, 4, requires_grad=True)

    result = verifier(
        q3,
        q_child,
        _retrieval(parent, child),
        torch.ones(2, 1),
    )
    result["child_match_logits"].sum().backward()

    assert q3.grad is None
    assert parent.grad is None
    assert q_child.grad is not None and torch.isfinite(q_child.grad).all()
    assert child.grad is not None and torch.isfinite(child.grad).all()
    assert verifier.parent_to_child.weight.grad is not None
    assert torch.isfinite(verifier.parent_to_child.weight.grad).all()


def test_parent_conditioned_verifier_masks_invalid_and_stays_finite() -> None:
    verifier = _pcv(
        verification_logit_clip=2.0,
    )
    q3 = torch.full((2, 4), 1.0e20)
    q_child = -q3
    parent = torch.full((2, 2, 2, 4), 1.0e20)
    child = -parent
    valid = torch.tensor(
        [[[True, False], [True, True]], [[False, False], [True, True]]]
    )

    result = verifier(
        q3,
        q_child,
        _retrieval(parent, child, valid),
        torch.ones(2, 1),
    )

    assert torch.isfinite(result["pair_scores"]).all()
    assert torch.isfinite(result["child_match_logits"]).all()
    assert result["child_match_logits"].abs().max() <= 2.0
    assert torch.count_nonzero(result["child_match_logits"][~valid]) == 0
    assert torch.count_nonzero(result["child_abs_cosine"][~valid]) == 0


class _MigrationDecoder(nn.Module):
    def __init__(self, cfg: DinoPCHBMConfig) -> None:
        super().__init__()
        self.pc_cfg = cfg
        self.baseline = nn.Linear(3, 2)
        self.pc_hbm = nn.Module()
        self.pc_hbm.shared = nn.Linear(2, 2)
        self.pc_hbm.pair_verifier = PairVerifier(
            dim=cfg.memory_dim,
            tau_parent=cfg.tau_parent,
            tau_child=cfg.tau_child,
            child_mix_init_logit=cfg.child_mix_init_logit,
            child_verification_mode=cfg.child_verification_mode,
            verification_strength_init=cfg.verification_strength_init,
            verification_logit_clip=cfg.verification_logit_clip,
            relation_norm_eps=cfg.relation_norm_eps,
        )


def test_explicit_legacy_decoder_migration_keeps_non_verifier_only(tmp_path) -> None:
    legacy_cfg = DinoPCHBMConfig(
        decoder_dim=4,
        memory_dim=4,
        child_verification_mode="weighted_sum",
        tau_parent=0.2,
    )
    source = _MigrationDecoder(legacy_cfg)
    with torch.no_grad():
        source.baseline.weight.fill_(2.0)
        source.pc_hbm.shared.weight.fill_(3.0)
        source.pc_hbm.pair_verifier.raw_child_mix.fill_(4.0)
    path = tmp_path / "legacy_decoder.pth"
    payload = save_decoder_checkpoint(path, source, legacy_cfg, 2)
    payload.pop("child_verifier_version")
    payload.pop("child_verification_mode")
    legacy_state = asdict(legacy_cfg)
    for name in tuple(legacy_state):
        if name.startswith("verification_") or name.startswith("parent_"):
            legacy_state.pop(name)
    for name in (
        "child_verification_mode",
        "relation_norm_eps",
        "lambda_candidate_verify",
        "lambda_parent_repair",
        "lambda_parent_preserve",
    ):
        legacy_state.pop(name, None)
    payload["pc_cfg"] = legacy_state
    torch.save(payload, path)

    with pytest.raises(RuntimeError):
        read_pc_config(path)
    migrated_cfg = read_pc_config(path, init_pcv_from_legacy=True)
    assert migrated_cfg.child_verification_mode == "parent_conditioned"
    assert migrated_cfg.tau_parent == pytest.approx(0.2)
    target = _MigrationDecoder(migrated_cfg)
    verifier_before = {
        key: value.clone()
        for key, value in target.pc_hbm.pair_verifier.state_dict().items()
    }

    load_decoder_compatible(
        target,
        path,
        require_pc_complete=True,
        expected_pc_cfg=migrated_cfg,
        init_pcv_from_legacy=True,
    )

    assert torch.equal(target.baseline.weight, source.baseline.weight)
    assert torch.equal(target.pc_hbm.shared.weight, source.pc_hbm.shared.weight)
    for key, value in verifier_before.items():
        assert torch.equal(value, target.pc_hbm.pair_verifier.state_dict()[key])


def test_explicit_v2_to_v3_migration_keeps_only_stable_verifier_state(
    tmp_path,
) -> None:
    cfg = DinoPCHBMConfig(decoder_dim=4, memory_dim=4)
    source = _MigrationDecoder(cfg)
    with torch.no_grad():
        source.baseline.weight.fill_(2.0)
        source.pc_hbm.shared.weight.fill_(3.0)
        source.pc_hbm.pair_verifier.parent_to_child.weight.fill_(4.0)
        source.pc_hbm.pair_verifier.raw_verification_strength.fill_(0.75)
    path = tmp_path / "v2_decoder.pth"
    payload = save_decoder_checkpoint(path, source, cfg, 2)
    payload["child_verifier_version"] = 2
    payload["pc_cfg"].update(
        {
            "verification_temperature": 0.10,
            "verification_abs_weight_init": 0.05,
            "verification_rel_weight_init": 0.05,
            "verification_bias_init": 0.0,
            "parent_hard_margin": 0.20,
            "parent_wrong_target_margin": 0.10,
            "verification_gain_margin": 0.10,
            "verification_preserve_tolerance": 0.05,
            "lambda_parent_repair": 0.50,
            "lambda_parent_preserve": 0.25,
            "lambda_candidate_verify": 0.50,
            "lambda_pair": 1.0,
            "feature_distill_p3_weight": 1.0,
        }
    )
    state = payload["decoder"]
    prefix = "pc_hbm.pair_verifier."
    state[prefix + "raw_verification_abs_weight"] = torch.tensor(-2.0)
    state[prefix + "raw_verification_rel_weight"] = torch.tensor(-3.0)
    state[prefix + "verification_bias"] = torch.tensor(1.0)
    torch.save(payload, path)

    target = _MigrationDecoder(cfg)
    target_before = {
        name: value.clone() for name, value in target.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="child_verifier_version=3"):
        load_decoder_compatible(
            target,
            path,
            require_pc_complete=True,
            expected_pc_cfg=cfg,
        )
    for name, value in target_before.items():
        assert torch.equal(value, target.state_dict()[name])

    migrated_cfg = read_pc_config(path, init_pcv_from_v2=True)
    for removed_name in (
        "verification_temperature",
        "lambda_candidate_verify",
        "lambda_pair",
        "feature_distill_p3_weight",
    ):
        assert not hasattr(migrated_cfg, removed_name)
    migrated = _MigrationDecoder(migrated_cfg)
    load_decoder_compatible(
        migrated,
        path,
        require_pc_complete=True,
        expected_pc_cfg=migrated_cfg,
        init_pcv_from_v2=True,
    )

    assert torch.equal(migrated.baseline.weight, source.baseline.weight)
    assert torch.equal(
        migrated.pc_hbm.shared.weight, source.pc_hbm.shared.weight
    )
    assert torch.equal(
        migrated.pc_hbm.pair_verifier.parent_to_child.weight,
        source.pc_hbm.pair_verifier.parent_to_child.weight,
    )
    assert torch.equal(
        migrated.pc_hbm.pair_verifier.raw_verification_strength,
        source.pc_hbm.pair_verifier.raw_verification_strength,
    )
    assert set(migrated.pc_hbm.pair_verifier.state_dict()) == {
        "parent_to_child.weight",
        "raw_verification_strength",
    }
