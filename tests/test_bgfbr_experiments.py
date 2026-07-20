from __future__ import annotations

from dataclasses import asdict
import sys

import pytest
import torch
import torch.nn as nn

from configs.bgfbr_experiments import (
    apply_experiment_profile,
    build_experiment_profile,
    experiment_profile_names,
)
from configs.pc_hbm_dino_config import DinoPCHBMConfig
import train_base_model_pc_hbm as base_cli
import inference as inference_cli
import train_ts_model_pseudo_pc_hbm as ts_cli
import Model.base_model as base_model_module
import Model.ts_model as ts_model_module


REQUIRED_PROFILES = {
    "bgfbr_pc",
    "default",
    "bgfbr_off",
    "encoder_pc",
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
    assert build_experiment_profile("bgfbr_pc").pc_placement == "decoder"


def test_encoder_profile_moves_pc_placement_without_changing_decoder_architecture():
    config = DinoPCHBMConfig()
    profile = apply_experiment_profile(config, "encoder_pc")

    assert profile.pc_placement == "encoder"
    assert config.pc_placement == "encoder"
    assert config.decoder_arch == "bgfbr_pc_v1"
    assert config.experiment_profile == "encoder_pc"


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


class _FakeDino(nn.Module):
    def load_state_dict(self, state_dict, strict=True):
        return None

    def get_intermediate_layers(self, x, **kwargs):
        batch = x.shape[0]
        patch_tokens = tuple(
            x.new_zeros(batch, 28 * 28, 768) for _ in range(4)
        )
        if kwargs.get("return_class_token", False):
            cls_tokens = tuple(x.new_zeros(batch, 768) for _ in range(4))
            return tuple(zip(patch_tokens, cls_tokens))
        return patch_tokens


class _RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc_hbm = None
        self.calls = []

    def forward(self, features, **kwargs):
        self.calls.append(dict(kwargs))
        sample = features[0]
        logits = sample.new_zeros(sample.shape[0], 1, 98, 98)
        outputs = (logits, logits, logits, logits, logits)
        return outputs, {"pc_active": False, "z_main": logits, "z_final": None}


def _patch_dino_loading(monkeypatch, module):
    monkeypatch.setattr(module.torch.hub, "load", lambda *args, **kwargs: _FakeDino())
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {})


def _encoder_config():
    config = DinoPCHBMConfig()
    apply_experiment_profile(config, "encoder_pc")
    return config


def test_base_encoder_profile_builds_a_detached_bgfbr_decoder(monkeypatch):
    _patch_dino_loading(monkeypatch, base_model_module)

    model = base_model_module.BaseModel(pc_cfg=_encoder_config())

    assert model.decoder.decoder_arch == "bgfbr_pc_v1"
    assert model.decoder.pc_hbm is None
    assert not any(name.startswith("pc_hbm.") for name, _ in model.decoder.named_parameters())
    assert not any(name.startswith("pc_hbm.") for name in model.decoder.state_dict())


def test_base_encoder_profile_forces_every_decoder_call_off(monkeypatch):
    _patch_dino_loading(monkeypatch, base_model_module)
    decoder = _RecordingDecoder()
    attach_values = []

    def fake_build_decoder(decoder_arch, pc_cfg, attach_pc):
        attach_values.append(attach_pc)
        return decoder

    monkeypatch.setattr(base_model_module, "build_decoder", fake_build_decoder)
    model = base_model_module.BaseModel(pc_cfg=_encoder_config())

    model(
        torch.zeros(2, 3, 392, 392),
        memory=object(),
        pc_mode="full",
        epoch=12,
        return_aux=True,
        query_image_ids=["a", "b"],
    )

    assert attach_values == [False]
    assert decoder.calls[-1]["pc_mode"] == "off"
    assert decoder.calls[-1]["memory"] is None
    assert decoder.calls[-1]["query_image_ids"] is None


def test_ts_encoder_profile_detaches_both_decoders_and_forces_off(monkeypatch):
    _patch_dino_loading(monkeypatch, ts_model_module)
    decoders = []
    attach_values = []

    def fake_build_decoder(decoder_arch, pc_cfg, attach_pc):
        decoder = _RecordingDecoder()
        decoder.decoder_arch = decoder_arch
        decoders.append(decoder)
        attach_values.append(attach_pc)
        return decoder

    monkeypatch.setattr(ts_model_module, "build_decoder", fake_build_decoder)
    monkeypatch.setattr(ts_model_module.TSModel, "load_teacher", lambda self, path: None)
    monkeypatch.setattr(ts_model_module.TSModel, "load_student", lambda self, path: None)

    model = ts_model_module.TSModel(
        teacher_pth="teacher.pth",
        student_pth="student.pth",
        pc_cfg=_encoder_config(),
        training_design="joint",
    )
    features = tuple(torch.zeros(2, 28 * 28, 768) for _ in range(4))
    image_rgb = torch.zeros(2, 3, 392, 392)
    memory = object()

    model.teacher_pseudo(features, memory, epoch=31, image_rgb=image_rgb)
    model.student_labeled(
        features,
        memory,
        epoch=31,
        query_image_ids=["a", "b"],
        image_rgb=image_rgb,
    )
    model.student_unlabeled(features, memory, epoch=31, image_rgb=image_rgb)

    assert attach_values == [False, False]
    assert all(decoder.pc_hbm is None for decoder in decoders)
    assert len(decoders[0].calls) == 1
    assert len(decoders[1].calls) == 2
    for decoder in decoders:
        for call in decoder.calls:
            assert call["pc_mode"] == "off"
            assert call["memory"] is None
