from __future__ import annotations

from unittest.mock import Mock

import torch
import torch.nn as nn

from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCAdapterOutput,
    EncoderPCCoreResult,
    EncoderPCHBMAdapter,
    EncoderPCSegmentationHead,
    TeacherPseudoLabelRefiner,
)


def _bundle() -> DinoFeatureBundle:
    patches = tuple(torch.randn(1, 784, 768) for _ in range(4))
    cls = tuple(torch.randn(1, 768) for _ in range(4))
    return DinoFeatureBundle(patches, cls).validate()


def _evidence() -> dict[str, torch.Tensor]:
    return {
        "verified_evidence": torch.randn(1, 128, 28, 28),
        "boundary_probability": torch.rand(1, 1, 28, 28),
        "pc_gate": torch.rand(1, 1, 28, 28),
        "contradiction": torch.rand(1, 1, 28, 28),
        "semantic_support": torch.rand(1, 1, 28, 28),
        "detail_support": torch.rand(1, 1, 28, 28),
        "valid_map": torch.ones(1, 1, 28, 28),
        "route_confidence": torch.full((1,), 0.8),
    }


class _Decoder(nn.Module):
    decoder_arch = "bgfbr_pc_v1"

    def __init__(self) -> None:
        super().__init__()
        self.pc_hbm = None
        self.scale = nn.Parameter(torch.ones(()))
        self.calls: list[dict] = []

    def forward(self, features, image_rgb, **kwargs):
        self.calls.append(kwargs)
        z_core = self.scale * torch.ones(
            features[0].size(0), 1, 98, 98, device=features[0].device
        )
        outputs = (z_core, z_core, z_core, z_core, z_core)
        if kwargs.get("return_aux", False):
            return outputs, {
                "features": {
                    "p1": self.scale
                    * torch.ones(
                        features[0].size(0),
                        128,
                        98,
                        98,
                        device=features[0].device,
                    )
                }
            }
        return outputs


def _head() -> tuple[EncoderPCSegmentationHead, _Decoder, Mock]:
    bundle = _bundle()
    adapter = EncoderPCHBMAdapter()
    adapter.forward = Mock(
        return_value=EncoderPCAdapterOutput(
            bundle.patch_tokens,
            {
                "mode": "full",
                "pc_active": True,
                "refiner_evidence": _evidence(),
            },
        )
    )
    decoder = _Decoder()
    refiner = TeacherPseudoLabelRefiner()
    refiner_forward = Mock(wraps=refiner.forward)
    refiner.forward = refiner_forward
    return EncoderPCSegmentationHead(adapter, decoder, refiner), decoder, refiner_forward


def test_role_contract_runs_refiner_only_for_labeled_and_teacher_roles() -> None:
    head, decoder, refiner_forward = _head()
    bundle = _bundle()
    rgb = torch.randn(1, 3, 392, 392)

    labeled = head(
        role="labeled_core",
        bundle=bundle,
        image_rgb=rgb,
        mode="off",
        return_aux=True,
    )
    assert isinstance(labeled, EncoderPCCoreResult)
    assert refiner_forward.call_count == 0

    labeled_refined = head(
        role="labeled_refiner", core_result=labeled, epoch=21
    )
    assert labeled_refined["p_pseudo_refined"].shape == (1, 1, 98, 98)
    assert refiner_forward.call_count == 1

    student = head(
        role="student_core",
        bundle=bundle,
        image_rgb=rgb,
        return_aux=True,
    )
    assert isinstance(student, EncoderPCCoreResult)
    assert refiner_forward.call_count == 1

    z_core = head(
        role="inference",
        bundle=bundle,
        image_rgb=rgb,
        return_aux=False,
    )
    assert torch.equal(z_core, decoder.scale * torch.ones_like(z_core))
    assert refiner_forward.call_count == 1

    teacher = head(
        role="teacher_pseudo",
        bundle=bundle,
        image_rgb=rgb,
        epoch=1,
    )
    assert teacher["z_core"] is teacher["outputs"][3]
    assert teacher["pseudo_refiner"]["p_pseudo_refined"].shape == (
        1,
        1,
        98,
        98,
    )
    assert refiner_forward.call_count == 2
    assert all(call["pc_mode"] == "off" for call in decoder.calls)
    assert all(call["memory"] is None for call in decoder.calls)
    assert all(call["query_image_ids"] is None for call in decoder.calls)


def test_labeled_refiner_role_rejects_missing_core_result() -> None:
    head, _, _ = _head()

    try:
        head(role="labeled_refiner")
    except TypeError as error:
        assert "EncoderPCCoreResult" in str(error)
    else:
        raise AssertionError("labeled_refiner accepted a missing core result")
