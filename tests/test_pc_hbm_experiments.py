from __future__ import annotations

from pathlib import Path
import sys

import pytest

from configs.pc_hbm_experiments import (
    apply_experiment_profile,
    build_experiment_profile,
    experiment_profile_names,
)
from configs.pc_hbm_dino_config import DinoPCHBMConfig, EncoderPCHBMConfig
import inference as inference_cli
import train_base_model_pc_hbm as base_cli
import train_ts_model_pseudo_pc_hbm as ts_cli


REQUIRED_PROFILES = {
    "encoder_pc",
    "default",
    "encoder_pc_f4_f3",
    "encoder_pc_no_route_loss",
    "legacy_pc",
    "legacy_off",
    "parent_only",
}


def test_profile_registry_is_exact_and_default_aliases_encoder_pc() -> None:
    assert set(experiment_profile_names()) == REQUIRED_PROFILES
    assert build_experiment_profile("default") is build_experiment_profile("encoder_pc")
    assert build_experiment_profile("encoder_pc").pc_placement == "encoder"
    assert not any(name.startswith("bgfbr_") for name in experiment_profile_names())


@pytest.mark.parametrize(
    "profile_name",
    ("encoder_pc", "encoder_pc_f4_f3", "encoder_pc_no_route_loss"),
)
def test_encoder_profiles_keep_the_original_decoder(profile_name: str) -> None:
    config = EncoderPCHBMConfig()
    profile = apply_experiment_profile(config, profile_name)

    assert profile.pc_placement == "encoder"
    assert config.pc_placement == "encoder"
    assert config.decoder_arch == "legacy_transformer"
    assert config.experiment_profile == profile_name


def test_encoder_ablations_change_only_the_requested_contract() -> None:
    f4_f3 = EncoderPCHBMConfig()
    apply_experiment_profile(f4_f3, "encoder_pc_f4_f3")
    assert f4_f3.enable_f2_f1_propagation is False
    assert f4_f3.lambda_route == pytest.approx(0.05)

    no_route = EncoderPCHBMConfig()
    apply_experiment_profile(no_route, "encoder_pc_no_route_loss")
    assert no_route.enable_f2_f1_propagation is True
    assert no_route.lambda_route == pytest.approx(0.0)


def test_enabled_false_remains_the_encoder_side_no_prototype_base_control() -> None:
    config = EncoderPCHBMConfig(enabled=False)
    apply_experiment_profile(config, "encoder_pc")

    assert config.enabled is False
    assert config.pc_placement == "encoder"
    assert config.decoder_arch == "legacy_transformer"


@pytest.mark.parametrize(
    ("profile_name", "epoch", "expected_mode"),
    [
        ("legacy_pc", 1, "off"),
        ("legacy_pc", 6, "parent_only"),
        ("legacy_pc", 11, "full"),
        ("legacy_off", 30, "off"),
        ("parent_only", 1, "parent_only"),
        ("parent_only", 30, "parent_only"),
    ],
)
def test_decoder_side_profiles_use_the_original_decoder_and_expected_schedule(
    profile_name: str,
    epoch: int,
    expected_mode: str,
) -> None:
    config = DinoPCHBMConfig()
    config.configure_training_design("two_stage")
    profile = apply_experiment_profile(config, profile_name)

    assert profile.pc_placement == "decoder"
    assert config.decoder_arch == "legacy_transformer"
    assert config.pc_mode_for_epoch(epoch) == expected_mode


def test_profile_type_mismatch_fails_fast() -> None:
    with pytest.raises(TypeError, match="Encoder profiles"):
        apply_experiment_profile(DinoPCHBMConfig(), "encoder_pc")
    with pytest.raises(TypeError, match="Decoder-side profiles"):
        apply_experiment_profile(EncoderPCHBMConfig(), "legacy_pc")


def test_cli_defaults_are_encoder_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_base_model_pc_hbm.py"])
    base_args = base_cli.parse_args()
    base_cli.validate_training_args(base_args)
    assert base_args.experiment_profile == "encoder_pc"
    assert not hasattr(base_args, "legacy_warm_start")

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ts_model_pseudo_pc_hbm.py", "--teacher-pc-checkpoint", "base.pth"],
    )
    ts_args = ts_cli.parse_args()
    ts_cli.validate_training_args(ts_args)
    assert ts_args.experiment_profile == "encoder_pc"

    monkeypatch.setattr(sys, "argv", ["inference.py"])
    inference_args = inference_cli.parse_args()
    assert inference_args.experiment_profile == "encoder_pc"


def test_removed_profiles_are_rejected() -> None:
    for old_profile in (
        "bgfbr_pc",
        "bgfbr_off",
        "no_gbe",
        "no_ode",
        "no_rcab",
        "no_pc_boundary_context",
    ):
        with pytest.raises(ValueError, match="Unknown experiment profile"):
            build_experiment_profile(old_profile)


def test_repository_has_no_removed_decoder_module_or_import() -> None:
    repository = Path(__file__).resolve().parents[1]
    assert not (repository / "Model" / "bgfbr_decoder.py").exists()
    assert not (repository / "Model" / "BGFBR").exists()

    forbidden_imports = (
        "from Model.bgfbr_decoder",
        "import Model.bgfbr_decoder",
        "from Model.BGFBR",
        "import Model.BGFBR",
    )
    offenders: list[str] = []
    for source in repository.rglob("*.py"):
        if source.resolve() == Path(__file__).resolve():
            continue
        text = source.read_text(encoding="utf-8")
        if any(fragment in text for fragment in forbidden_imports):
            offenders.append(str(source.relative_to(repository)))
    assert offenders == []
