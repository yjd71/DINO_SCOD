from __future__ import annotations

import pytest
import torch

from selection.pc_bpus import (
    compute_boundary_utility,
    score_and_prototype,
    sobel_magnitude,
)


def _square_probability(size: int = 8) -> torch.Tensor:
    probability = torch.full((1, 1, size, size), 0.2)
    probability[..., 2:-2, 2:-2] = 0.8
    return probability


def test_sobel_constant_prediction_is_exactly_zero_and_border_is_cleared() -> None:
    constant = torch.full((2, 1, 7, 9), 0.5)
    assert torch.count_nonzero(sobel_magnitude(constant)) == 0

    ramp = torch.linspace(0.0, 1.0, 9).view(1, 1, 1, 9).expand(1, 1, 7, 9)
    magnitude = sobel_magnitude(ramp)
    assert torch.count_nonzero(magnitude[..., 1:-1, 1:-1]) > 0
    assert torch.count_nonzero(magnitude[..., 0, :]) == 0
    assert torch.count_nonzero(magnitude[..., -1, :]) == 0
    assert torch.count_nonzero(magnitude[..., :, 0]) == 0
    assert torch.count_nonzero(magnitude[..., :, -1]) == 0


def test_constant_prediction_has_zero_mass_value_and_is_invalid() -> None:
    probability = torch.full((2, 1, 8, 8), 0.3)
    transformed = torch.full_like(probability, 0.7)
    result = compute_boundary_utility(probability, transformed)

    assert torch.equal(result.boundary_mass, torch.zeros(2))
    assert torch.equal(result.d_boundary, torch.zeros(2))
    assert torch.equal(result.value, torch.zeros(2))
    assert not bool(result.valid.any())


def test_boundary_local_change_beats_equal_global_change() -> None:
    probability = _square_probability()
    local = probability.clone()
    local[..., 2:-2, 2] = 0.6
    global_change = (probability + 0.2).clamp_max(1.0)

    local_result = compute_boundary_utility(probability, local)
    global_result = compute_boundary_utility(probability, global_change)

    assert local_result.d_boundary.item() > local_result.d_global.item()
    assert local_result.value.item() > 0.0
    assert global_result.d_boundary.item() == pytest.approx(
        global_result.d_global.item(), abs=1e-6
    )
    assert global_result.value.item() == pytest.approx(0.0, abs=1e-6)


class _FlipEquivariantSelector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call: tuple[str, bool] | None = None

    def forward(
        self, images: torch.Tensor, *, pc_mode: str, return_aux: bool
    ):
        self.call = (pc_mode, return_aux)
        probability = images[:, :1].clamp(0.0, 1.0)
        p2 = torch.cat((images[:, :1], 1.0 - images[:, :1]), dim=1)
        outputs = tuple(torch.empty(0) for _ in range(5))
        return outputs, {"p_final": probability, "features": {"p2": p2}}


def test_score_and_prototype_aligns_both_flip_views() -> None:
    model = _FlipEquivariantSelector()
    images = _square_probability()
    result = score_and_prototype(model, images)

    assert model.call == ("off", True)
    assert result.d_boundary.item() == 0.0
    assert result.d_global.item() == 0.0
    assert result.value.item() == 0.0
    assert bool(result.valid.item())
    assert result.prototype.shape == (1, 2)
    assert torch.linalg.vector_norm(result.prototype, dim=1).item() == pytest.approx(
        1.0, abs=1e-6
    )


def test_score_rejects_unexpected_production_p2_shape() -> None:
    model = _FlipEquivariantSelector()
    images = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="Selector P2 must have shape"):
        score_and_prototype(
            model,
            images,
            expected_p2_shape=(128, 28, 28),
        )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_score_rejects_invalid_eps(eps: float) -> None:
    probability = _square_probability()
    with pytest.raises(ValueError, match="eps"):
        compute_boundary_utility(probability, probability, eps=eps)
