from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder


@dataclass(frozen=True)
class _Compatibility:
    compatible: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.compatible


class _Memory:
    def __init__(self, compatible: bool = True) -> None:
        torch.manual_seed(31)
        self.compatible = compatible
        self.pairs = {
            "p3_keys": torch.randn(8, 128),
            "p2_keys": torch.randn(8, 128),
            "region_ids": torch.tensor([0] * 4 + [1] * 4),
            "pair_indices": torch.arange(8),
        }

    def is_ready(self) -> bool:
        return True

    def validate_compat(self, expected):
        return _Compatibility(
            self.compatible,
            None if self.compatible else "compat_mismatch:schema_version",
        )

    def route_query(
        self,
        *,
        q_global,
        q_environment,
        top_img_k,
        query_image_ids=None,
        exclude_self_match=True,
    ):
        batch = q_global.size(0)
        return {
            "top_img_ids": [["memory"] for _ in range(batch)],
            "scores": torch.zeros(batch, top_img_k),
            "valid": torch.ones(batch, top_img_k, dtype=torch.bool),
            "indices": torch.zeros(batch, top_img_k, dtype=torch.long),
        }

    def get_pair_subbank(
        self,
        top_img_ids,
        *,
        device=None,
        dtype=None,
        exclude_image_id=None,
    ):
        return {
            key: value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in self.pairs.items()
        }


@pytest.fixture(scope="module")
def decoder_inputs():
    torch.manual_seed(37)
    model = Decoder(pc_cfg=DinoPCHBMConfig())
    features = [torch.randn(1, 28 * 28, 768) for _ in range(4)]
    return model, features, _Memory()


@pytest.mark.parametrize("mode", ("verify_only", "full", "teacher_pseudo"))
def test_decoder_lite_modes_have_stable_output_and_aux(
    decoder_inputs, mode
) -> None:
    model, features, memory = decoder_inputs
    outputs, aux = model(
        features,
        memory=memory,
        pc_mode=mode,
        epoch=11,
        return_aux=True,
        query_image_ids=["query"],
    )

    assert len(outputs) == 5
    assert all(output.shape == (1, 1, 98, 98) for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    assert aux["pc_active"] is True
    assert aux["forward_mode"] == mode
    assert torch.equal(aux["z_final"], aux["z_main"])
    torch.testing.assert_close(aux["p_final"], torch.sigmoid(aux["z_main"]))
    assert aux["pc_hbm"]["pair_logits"].shape == (64, 2)
    assert aux["pc_hbm"]["query_valid"].shape == (64,)
    if mode == "verify_only":
        assert torch.equal(
            aux["pc_hbm"]["p3_corr"], aux["features"]["p3"]
        )
        assert torch.count_nonzero(aux["pc_hbm"]["p3_delta"]) == 0
    if mode == "teacher_pseudo":
        assert set(aux["distill_features"]) == {"p3_corr"}
        assert aux["distill_features"]["p3_corr"].shape == (1, 128, 28, 28)
    else:
        assert aux["distill_features"] is None


@pytest.mark.parametrize("mode", ("parent_only", "student_core", "joint"))
def test_decoder_rejects_removed_modes_without_aliases(
    decoder_inputs, mode
) -> None:
    model, features, memory = decoder_inputs
    with pytest.raises(ValueError, match="Unsupported pc_mode"):
        model(features, memory=memory, pc_mode=mode)


def test_decoder_missing_memory_falls_back_but_incompatible_memory_raises(
    decoder_inputs,
) -> None:
    model, features, _ = decoder_inputs
    baseline = model(features, pc_mode="off")
    fallback, aux = model(
        features, memory=None, pc_mode="full", return_aux=True
    )
    assert aux["pc_active"] is False
    assert aux["fallback_reason"] == "memory_missing"
    for actual, expected in zip(fallback, baseline):
        torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="Incompatible PC-HBM-Lite memory"):
        model(features, memory=_Memory(compatible=False), pc_mode="full")
    with pytest.raises(ValueError, match="Incompatible PC-HBM-Lite memory"):
        model(features, memory=_Memory(compatible=False), pc_mode="off")


def test_decoder_lite_rejects_nonfixed_dino_grid() -> None:
    model = Decoder(pc_cfg=DinoPCHBMConfig()).eval()
    features = tuple(torch.randn(1, 14 * 14, 768) for _ in range(4))
    with pytest.raises(ValueError, match="28x28"):
        model(features, pc_mode="off")


def test_decoder_memory_features_are_lite_builder_inputs(
    decoder_inputs,
) -> None:
    model, features, _ = decoder_inputs
    memory_features = model.forward_memory_features(features)
    assert set(memory_features) == {"x3", "p3", "p2", "m3"}
    assert memory_features["m3"].shape == (1, 1, 28, 28)
    for key in ("x3", "p3", "p2"):
        assert memory_features[key].shape == (1, 128, 28, 28)


def test_pair_loss_backward_reaches_child_mix_and_encoder(
    decoder_inputs,
) -> None:
    model, features, memory = decoder_inputs
    model.zero_grad(set_to_none=True)
    _, aux = model(
        features,
        memory=memory,
        pc_mode="full",
        epoch=13,
        return_aux=True,
        query_image_ids=["query"],
    )
    logits = aux["pc_hbm"]["pair_logits"]
    loss = (logits[:, 0] - logits[:, 1]).mean()
    loss.backward()
    assert model.pc_hbm.pair_verifier.raw_child_mix.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.pc_hbm.child_query.parameters()
    )
