from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
import Model.base_model as base_model_module
from Model.PC_HBM.training.encoder_losses import encoder_bootstrap_loss
from Model.PC_HBM.training.supervision import build_gt_boundary


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
    decoder_arch = "bgfbr_pc_v1"

    def __init__(self) -> None:
        super().__init__()
        self.pc_hbm = None
        self.scale = nn.Parameter(torch.ones(()))
        self.calls: list[dict] = []

    def forward(self, features, image_rgb, **kwargs):
        self.calls.append({"features": features, "image_rgb": image_rgb, **kwargs})
        low = features[0].mean(dim=-1).view(features[0].size(0), 1, 28, 28)
        logits = self.scale * F.interpolate(
            low, size=(98, 98), mode="bilinear", align_corners=False
        )
        outputs = (logits, logits, logits, logits, logits)
        if kwargs.get("return_aux", False):
            return outputs, {
                "decoder_architecture": "bgfbr_pc_v1",
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
    decoder = _RecordingDecoder()
    monkeypatch.setattr(
        base_model_module,
        "build_decoder",
        lambda *args, **kwargs: decoder,
    )
    return base_model_module.BaseModel(pc_cfg=EncoderPCHBMConfig()), decoder


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


def test_v3_off_path_is_elementwise_equal_to_bare_decoder(monkeypatch) -> None:
    model, decoder = _model(monkeypatch)
    images = torch.randn(1, 3, 392, 392)

    actual = model(images, pc_mode="off")
    raw = model.extract_features(images)
    expected = decoder(
        features=raw,
        image_rgb=model.prepare_rgb(images),
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
