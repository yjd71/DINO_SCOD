from __future__ import annotations

import pytest
import torch

from selection.bpus_v2 import (
    compute_boundary_score_bpus_v2,
    score_and_prototype_bpus_v2,
    sobel_magnitude_bpus_v2,
)


def _square_probability(size: int = 16) -> torch.Tensor:
    probability = torch.zeros(1, 1, size, size)
    probability[..., size // 4 : -size // 4, size // 4 : -size // 4] = 1.0
    return probability


def test_sobel_constant_is_exactly_zero_and_outer_ring_is_cleared() -> None:
    constant = torch.full((2, 1, 11, 13), 0.4)
    magnitude = sobel_magnitude_bpus_v2(constant)
    assert torch.count_nonzero(magnitude).item() == 0

    square = sobel_magnitude_bpus_v2(_square_probability())
    assert torch.count_nonzero(square).item() > 0
    assert torch.count_nonzero(square[..., 0, :]).item() == 0
    assert torch.count_nonzero(square[..., -1, :]).item() == 0
    assert torch.count_nonzero(square[..., :, 0]).item() == 0
    assert torch.count_nonzero(square[..., :, -1]).item() == 0


def test_constant_prediction_is_strictly_invalid_and_zero_value() -> None:
    probability = torch.full((1, 1, 12, 12), 0.3)
    transformed = torch.full_like(probability, 0.7)
    score = compute_boundary_score_bpus_v2(probability, transformed)

    assert score.boundary_mass.item() == 0.0
    assert score.boundary_disagreement.item() == 0.0
    assert score.boundary_value.item() == 0.0
    assert not score.valid_boundary.item()


def test_boundary_local_change_has_higher_value_than_global_change() -> None:
    probability = _square_probability()
    boundary = sobel_magnitude_bpus_v2(probability) > 0
    local = probability.clone()
    local[boundary] = 1.0 - local[boundary]
    global_change = (probability + 0.5).clamp(0.0, 1.0)

    local_score = compute_boundary_score_bpus_v2(probability, local)
    global_score = compute_boundary_score_bpus_v2(probability, global_change)

    assert local_score.boundary_value.item() > global_score.boundary_value.item()
    assert local_score.boundary_disagreement.item() > 0.0


class _TwoViewSelector(torch.nn.Module):
    def __init__(self, *, p2_shape: tuple[int, int, int] = (128, 28, 28)) -> None:
        super().__init__()
        self.forward_calls = 0
        self.p2_shape = p2_shape

    def forward(self, images: torch.Tensor, *, pc_mode: str, return_aux: bool):
        assert pc_mode == "off"
        assert return_aux
        self.forward_calls += 1
        probability = images[:, :1].sigmoid()
        base = torch.linspace(
            0.1,
            1.0,
            self.p2_shape[0] * self.p2_shape[1] * self.p2_shape[2],
            device=images.device,
        ).reshape(1, *self.p2_shape)
        p2 = base.expand(images.shape[0], -1, -1, -1)
        return (), {"p_final": probability, "features": {"p2": p2}}


def test_scoring_uses_two_explicit_forwards_and_aligns_flip() -> None:
    model = _TwoViewSelector()
    images = torch.randn(2, 3, 56, 56)
    result = score_and_prototype_bpus_v2(model, images)

    assert model.forward_calls == 2
    assert not model.training
    assert result.prototype.shape == (2, 128)
    assert torch.allclose(result.global_disagreement, torch.zeros(2))
    assert result.prototype.dtype == torch.float32


def test_scoring_rejects_non_production_p2_shape() -> None:
    with pytest.raises(ValueError, match="128,28,28"):
        score_and_prototype_bpus_v2(
            _TwoViewSelector(p2_shape=(64, 28, 28)), torch.randn(1, 3, 56, 56)
        )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf")])
def test_score_rejects_invalid_epsilon(eps: float) -> None:
    probability = _square_probability()
    with pytest.raises(ValueError, match="eps"):
        compute_boundary_score_bpus_v2(probability, probability, eps=eps)
