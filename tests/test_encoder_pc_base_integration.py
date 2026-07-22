from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.pc_hbm_dino_config import DinoPCHBMConfig, EncoderPCHBMConfig
import Model.base_model as base_model_module
from Model.PC_HBM.training.encoder_losses import encoder_bootstrap_loss
from Model.PC_HBM.training.supervision import build_gt_boundary
from Model.PC_HBM.encoder import (
    EncoderPCSegmentationHead,
    TeacherPseudoLabelRefiner,
)
from utils.trainer_base_model_no_pc import NoPCBaseTrainer


class _FakeDino(nn.Module):
    def load_state_dict(self, state_dict, strict=True):
        return None

    def get_intermediate_layers(self, x, **kwargs):
        batch = x.size(0)
        base = x.mean(dim=(1, 2, 3), keepdim=True).view(batch, 1, 1)
        patches = tuple(
            (base + float(level)).expand(batch, 784, 768).clone()
            for level in range(4)
        )
        cls = tuple(patch[:, 0].clone() for patch in patches)
        return tuple(zip(patches, cls))


class _RecordingDecoder(nn.Module):
    decoder_arch = "legacy_transformer"

    def __init__(self, pc_cfg=None) -> None:
        super().__init__()
        self.received_pc_cfg = pc_cfg
        self.pc_hbm = None
        self.scale = nn.Parameter(torch.ones(()))
        self.calls: list[dict] = []

    def forward(self, features, **kwargs):
        self.calls.append({"features": features, **kwargs})
        low = features[0].mean(dim=-1).view(features[0].size(0), 1, 28, 28)
        logits = self.scale * F.interpolate(
            low, size=(98, 98), mode="bilinear", align_corners=False
        )
        outputs = (logits, logits, logits, logits, logits)
        if kwargs.get("return_aux", False):
            return outputs, {
                "decoder_architecture": "legacy_transformer",
                "z_main": logits,
                "z_final": None,
                "pc_active": False,
            }
        return outputs


def _model(monkeypatch) -> tuple[nn.Module, _RecordingDecoder]:
    monkeypatch.setattr(
        base_model_module.torch.hub,
        "load",
        lambda *args, **kwargs: _FakeDino(),
    )
    monkeypatch.setattr(base_model_module.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(base_model_module, "Decoder", _RecordingDecoder)
    model = base_model_module.BaseModel(pc_cfg=EncoderPCHBMConfig())
    return model, model.decoder


def test_v3_base_bootstrap_runs_adapter_then_decoder_permanently_off(monkeypatch) -> None:
    model, decoder = _model(monkeypatch)
    output, aux = model(
        torch.randn(2, 3, 392, 392),
        memory=None,
        pc_mode="bootstrap",
        return_aux=True,
        query_image_ids=["a", "b"],
    )

    assert len(output) == 5
    assert aux["encoder_pc_hbm"]["coarse_logits"].shape == (2, 1, 28, 28)
    assert decoder.calls[-1]["pc_mode"] == "off"
    assert decoder.calls[-1]["memory"] is None
    assert decoder.calls[-1]["query_image_ids"] is None
    assert all(feature.shape == (2, 784, 768) for feature in decoder.calls[-1]["features"])


def test_v3_base_constructs_real_refiner_and_role_head(monkeypatch) -> None:
    model, decoder = _model(monkeypatch)

    assert isinstance(model.pseudo_refiner, TeacherPseudoLabelRefiner)
    assert isinstance(model.encoder_pc_head, EncoderPCSegmentationHead)
    assert model.encoder_pc_head.adapter is model.encoder_pc_hbm
    assert model.encoder_pc_head.decoder is decoder
    assert model.encoder_pc_head.pseudo_refiner is model.pseudo_refiner


def test_v3_off_path_is_elementwise_equal_to_bare_decoder(monkeypatch) -> None:
    model, decoder = _model(monkeypatch)
    images = torch.randn(1, 3, 392, 392)

    actual = model(images, pc_mode="off")
    raw = model.extract_features(images)
    expected = decoder(
        features=raw,
        memory=None,
        pc_mode="off",
        epoch=None,
        return_aux=False,
        query_image_ids=None,
    )

    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_bootstrap_aux_loss_trains_adapter_but_never_dino(monkeypatch) -> None:
    model, _ = _model(monkeypatch)
    images = torch.randn(1, 3, 392, 392)
    gt = torch.rand(1, 1, 392, 392).round()
    _, aux = model(images, pc_mode="bootstrap", return_aux=True)
    encoder_aux = aux["encoder_pc_hbm"]
    losses = encoder_bootstrap_loss(
        coarse_logits=encoder_aux["coarse_logits"],
        boundary_logits=encoder_aux["boundary_logits"],
        mask_target=gt,
        boundary_target=build_gt_boundary(gt, (28, 28)),
    )

    losses["total"].backward()

    assert all(parameter.grad is None for parameter in model.dino.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.encoder_pc_hbm.bootstrap.parameters()
        if parameter.requires_grad
    )


def test_enabled_false_is_structural_no_prototype_base(monkeypatch) -> None:
    monkeypatch.setattr(
        base_model_module.torch.hub,
        "load",
        lambda *args, **kwargs: _FakeDino(),
    )
    monkeypatch.setattr(base_model_module.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(base_model_module, "Decoder", _RecordingDecoder)
    cfg = EncoderPCHBMConfig(enabled=False)
    model = base_model_module.BaseModel(pc_cfg=cfg)
    decoder = model.decoder

    assert model.pc_enabled is False
    assert decoder.received_pc_cfg is None
    assert model.encoder_pc_hbm is None
    assert model.pseudo_refiner is None
    assert model.encoder_pc_head is None
    assert model.encoder_pc_profile_v3 is False
    assert not any("encoder_pc_hbm." in name for name in model.state_dict())
    assert not any("pseudo_refiner." in name for name in model.state_dict())

    marker_memory = object()
    _, aux = model(
        torch.randn(1, 3, 392, 392),
        memory=marker_memory,
        pc_mode="full",
        return_aux=True,
        query_image_ids=["self"],
        run_labeled_refiner=True,
    )
    call = decoder.calls[-1]
    assert call["memory"] is None
    assert call["pc_mode"] == "off"
    assert call["query_image_ids"] is None
    assert aux["pc_enabled"] is False
    assert aux["pc_active"] is False
    assert aux["fallback_reason"] == "pc_hbm_disabled_by_config"


def test_legacy_config_enabled_false_locks_schedule_off() -> None:
    cfg = DinoPCHBMConfig(enabled=False)

    assert cfg.pc_mode_for_epoch(1) == "off"
    assert cfg.pc_mode_for_epoch(1_000_000) == "off"
    assert cfg.injection_scale(1_000_000) == 0.0


class _NoPCDecoder(nn.Module):
    decoder_arch = "legacy_transformer"

    def __init__(self) -> None:
        super().__init__()
        self.pc_hbm = None
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        logits = images.mean(dim=1, keepdim=True)
        logits = F.interpolate(logits, size=(98, 98)) + self.bias
        return (logits, logits, logits, logits, logits)


class _NoPCTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dino = nn.Linear(1, 1)
        self.decoder = _NoPCDecoder()
        self.encoder_pc_hbm = None
        self.pseudo_refiner = None
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append(kwargs)
        return self.decoder(x)


def test_no_pc_trainer_updates_only_decoder_and_never_builds_memory(tmp_path) -> None:
    model = _NoPCTrainingModel()
    cfg = SimpleNamespace(
        device=torch.device("cpu"),
        use_amp=False,
        epochs=1,
        learning_rate=1.0e-3,
        min_lr=1.0e-7,
        weight_decay=0.0,
        checkpoint_interval=999,
        save_dir=str(tmp_path),
    )
    batch = (
        torch.randn(2, 3, 32, 32),
        torch.rand(2, 1, 98, 98).round(),
        ["a", "b"],
    )
    trainer = NoPCBaseTrainer(
        model,
        cfg,
        EncoderPCHBMConfig(enabled=False),
        labeled_loader=[batch],
    )
    old_bias = model.decoder.bias.detach().clone()

    metrics = trainer.train_epoch(1)

    assert metrics["pc_enabled"] == 0.0
    assert not torch.equal(model.decoder.bias.detach(), old_bias)
    assert all(parameter.grad is None for parameter in model.dino.parameters())
    assert model.calls[-1]["memory"] is None
    assert model.calls[-1]["pc_mode"] == "off"
    assert model.calls[-1]["query_image_ids"] is None
    assert not hasattr(trainer, "memory")
    assert not hasattr(trainer, "memory_loader")
