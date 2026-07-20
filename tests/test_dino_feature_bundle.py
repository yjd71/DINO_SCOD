from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from Model.base_model import BaseModel
from Model.PC_HBM.encoder import DinoFeatureBundle
from Model.ts_model import TSModel


class _FakeDino(nn.Module):
    def __init__(self, variant: str = "valid"):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.variant = variant
        self.calls = []

    def get_intermediate_layers(self, x, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        batch = x.shape[0]
        pairs = []
        for _ in range(4):
            patch = torch.ones(
                batch, 28 * 28, 768, device=x.device, dtype=x.dtype
            ) * self.anchor.to(dtype=x.dtype)
            cls = torch.ones(batch, 768, device=x.device, dtype=x.dtype) * self.anchor.to(
                dtype=x.dtype
            )
            pairs.append((patch, cls))

        if self.variant == "short":
            return tuple(pairs[:3])
        if self.variant == "malformed_pair":
            return (pairs[0][0], *pairs[1:])
        if self.variant == "bad_patch_shape":
            pairs[1] = (pairs[1][0][:, :-1], pairs[1][1])
        if self.variant == "bad_cls_shape":
            pairs[2] = (pairs[2][0], pairs[2][1][:, :-1])
        return tuple(pairs)


def _model_without_constructor(model_type, dino):
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.dino = dino
    model.pc_cfg = SimpleNamespace(dino_layer_indices=(2, 5, 8, 11))
    return model


@pytest.mark.parametrize("model_type", (BaseModel, TSModel))
def test_extract_feature_bundle_preserves_patch_and_cls_contract(model_type):
    dino = _FakeDino().train()
    model = _model_without_constructor(model_type, dino)
    images = torch.randn(1, 3, 392, 392, requires_grad=True)

    bundle = model.extract_feature_bundle(images)

    assert isinstance(bundle, DinoFeatureBundle)
    assert isinstance(bundle.patch_tokens, tuple)
    assert isinstance(bundle.cls_tokens, tuple)
    assert len(bundle.patch_tokens) == len(bundle.cls_tokens) == 4
    for patch, cls in zip(bundle.patch_tokens, bundle.cls_tokens):
        assert patch.shape == (1, 784, 768)
        assert cls.shape == (1, 768)
        assert patch.dtype == images.dtype
        assert cls.dtype == images.dtype
        assert patch.device == images.device
        assert cls.device == images.device
        assert not patch.requires_grad
        assert not cls.requires_grad

    call = dino.calls[-1]
    assert call["n"] == (2, 5, 8, 11)
    assert call["reshape"] is False
    assert call["return_class_token"] is True
    assert call["norm"] is True
    assert call["grad_enabled"] is False
    assert call["training"] is False
    assert dino.training is False
    assert dino.anchor.grad is None
    assert images.grad is None


@pytest.mark.parametrize("model_type", (BaseModel, TSModel))
def test_extract_features_keeps_the_patch_only_legacy_interface(model_type):
    dino = _FakeDino()
    model = _model_without_constructor(model_type, dino)
    images = torch.zeros(1, 3, 392, 392)

    patch_tokens = model.extract_features(images)
    alias_tokens = model._extract_features(images)

    assert isinstance(patch_tokens, tuple)
    assert isinstance(alias_tokens, tuple)
    assert len(patch_tokens) == len(alias_tokens) == 4
    assert all(tensor.shape == (1, 784, 768) for tensor in patch_tokens)
    assert all(torch.equal(left, right) for left, right in zip(patch_tokens, alias_tokens))
    assert all(call["return_class_token"] is True for call in dino.calls)


@pytest.mark.parametrize("model_type", (BaseModel, TSModel))
@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("short", "instead of four"),
        ("malformed_pair", "patch_tokens, cls_token"),
        ("bad_patch_shape", "patch level 2"),
        ("bad_cls_shape", "CLS level 3"),
    ],
)
def test_extract_feature_bundle_rejects_malformed_dino_outputs(
    model_type, variant, message
):
    model = _model_without_constructor(model_type, _FakeDino(variant))

    with pytest.raises((RuntimeError, ValueError), match=message):
        model.extract_feature_bundle(torch.zeros(1, 3, 392, 392))


@pytest.mark.parametrize("model_type", (BaseModel, TSModel))
def test_extract_feature_bundle_rejects_non_image_input(model_type):
    model = _model_without_constructor(model_type, _FakeDino())

    with pytest.raises(ValueError, match="image batch"):
        model.extract_feature_bundle(torch.zeros(3, 392, 392))


def test_dino_feature_bundle_is_frozen_and_validates_cross_level_consistency():
    patch = torch.zeros(1, 784, 768)
    cls = torch.zeros(1, 768)
    bundle = DinoFeatureBundle(
        patch_tokens=(patch, patch, patch, patch),
        cls_tokens=(cls, cls, cls, cls),
    )

    assert bundle.validate() is bundle
    with pytest.raises(FrozenInstanceError):
        bundle.patch_tokens = (patch, patch, patch, patch)

    bad_batch = DinoFeatureBundle(
        patch_tokens=(patch, patch, patch, torch.zeros(2, 784, 768)),
        cls_tokens=(cls, cls, cls, torch.zeros(2, 768)),
    )
    with pytest.raises(ValueError, match="share batch size"):
        bad_batch.validate()

    bad_dtype = DinoFeatureBundle(
        patch_tokens=(patch, patch, patch, patch.double()),
        cls_tokens=(cls, cls, cls, cls.double()),
    )
    with pytest.raises(ValueError, match="share batch size, device, and dtype"):
        bad_dtype.validate()
