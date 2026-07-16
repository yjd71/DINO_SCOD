from __future__ import annotations

from dataclasses import asdict
import sys

import pytest

from configs.bgfbr_experiments import (
    apply_experiment_profile,
    build_experiment_profile,
    experiment_profile_names,
)
from configs.pc_hbm_dino_config import DinoPCHBMConfig
import train_base_model_pc_hbm as base_cli
import inference as inference_cli
import train_ts_model_pseudo_pc_hbm as ts_cli


REQUIRED_PROFILES = {
    "bgfbr_pc",
    "default",
    "bgfbr_off",
    "parent_only",
    "no_gbe",
    "no_ode",
    "no_rcab",
    "no_pc_boundary_context",
    "legacy_off",
}


def test_profile_registry_and_default_alias_are_stable():
    assert REQUIRED_PROFILES.issubset(experiment_profile_names())
    assert build_experiment_profile("default") is build_experiment_profile("bgfbr_pc")


@pytest.mark.parametrize(
    ("profile_name", "field"),
    [
        ("no_gbe", "use_gbe"),
        ("no_ode", "use_ode"),
        ("no_rcab", "use_rcab"),
        ("no_pc_boundary_context", "use_pc_boundary_context"),
    ],
)
def test_component_ablations_change_only_the_selected_switch(profile_name, field):
    config = DinoPCHBMConfig()
    profile = apply_experiment_profile(config, profile_name)

    assert profile.name == profile_name
    for switch in ("use_gbe", "use_ode", "use_rcab", "use_pc_boundary_context"):
        assert getattr(config, switch) is (switch != field)
    assert config.decoder_arch == "bgfbr_pc_v1"
    assert asdict(config)["experiment_profile"] == profile_name


def test_base_mode_profiles_override_the_configured_curriculum():
    off = DinoPCHBMConfig()
    off.configure_training_design("two_stage")
    apply_experiment_profile(off, "bgfbr_off")
    off.configure_training_design("two_stage")
    assert off.pc_mode_for_epoch(1) == "off"
    assert off.pc_mode_for_epoch(30) == "off"

    parent = DinoPCHBMConfig()
    parent.configure_training_design("two_stage")
    apply_experiment_profile(parent, "parent_only")
    parent.configure_training_design("two_stage")
    assert parent.pc_mode_for_epoch(1) == "parent_only"
    assert parent.pc_mode_for_epoch(30) == "parent_only"


def test_legacy_profile_selects_legacy_decoder_and_disables_pc_schedule():
    config = DinoPCHBMConfig()
    apply_experiment_profile(config, "legacy-off")

    assert config.decoder_arch == "legacy_transformer"
    assert config.pc_mode_for_epoch(30) == "off"
    assert not any(
        (
            config.use_gbe,
            config.use_ode,
            config.use_rcab,
            config.use_pc_boundary_context,
        )
    )


def test_base_cli_accepts_profile_and_explicit_legacy_warm_start(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_base_model_pc_hbm.py",
            "--experiment-profile",
            "no_gbe",
            "--legacy-warm-start",
            "legacy.pth",
        ],
    )
    args = base_cli.parse_args()
    base_cli.validate_training_args(args)

    assert args.experiment_profile == "no_gbe"
    assert args.legacy_warm_start == "legacy.pth"


def test_legacy_warm_start_is_two_stage_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_base_model_pc_hbm.py",
            "--training-design",
            "joint",
            "--legacy-warm-start",
            "legacy.pth",
        ],
    )
    args = base_cli.parse_args()
    with pytest.raises(ValueError, match="two_stage"):
        base_cli.validate_training_args(args)


def test_legacy_warm_start_conflicts_with_normal_initialization(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_base_model_pc_hbm.py",
            "--legacy-warm-start",
            "legacy.pth",
            "--baseline-checkpoint",
            "bgfbr.pth",
        ],
    )
    with pytest.raises(SystemExit):
        base_cli.parse_args()


def test_ts_cli_accepts_component_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ts_model_pseudo_pc_hbm.py",
            "--teacher-pc-checkpoint",
            "teacher.pth",
            "--experiment-profile",
            "no_rcab",
        ],
    )
    args = ts_cli.parse_args()

    assert args.experiment_profile == "no_rcab"


def test_inference_cli_accepts_the_training_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["inference.py", "--experiment-profile", "no_pc_boundary_context"],
    )
    args = inference_cli.parse_args()

    assert args.experiment_profile == "no_pc_boundary_context"
