from __future__ import annotations

import math

import pytest
import torch

from utils.pc_bacs import compute_pc_bacs_score, score_pool, sobel_magnitude


def test_sobel_constant_prediction_has_exactly_zero_weight() -> None:
    probability = torch.full((2, 1, 9, 11), 0.37)
    magnitude = sobel_magnitude(probability)

    assert magnitude.dtype == torch.float32
    assert torch.count_nonzero(magnitude).item() == 0


def test_constant_disagreement_has_no_false_outer_boundary() -> None:
    probability = torch.full((1, 1, 8, 8), 0.2)
    transformed = torch.full((1, 1, 8, 8), 0.8)

    result = compute_pc_bacs_score(probability, transformed)

    assert result["boundary_disagreement"].item() == pytest.approx(0.0)
    assert result["global_disagreement"].item() == pytest.approx(0.6)
    assert result["score"].item() == pytest.approx(0.0)


def test_boundary_local_change_scores_above_global_mean_change() -> None:
    probability = torch.zeros(1, 1, 16, 16)
    probability[..., 8:] = 1.0
    transformed = probability.clone()
    transformed[..., 7:9] = 0.5

    result = compute_pc_bacs_score(probability, transformed)

    boundary = result["boundary_disagreement"].item()
    global_value = result["global_disagreement"].item()
    assert 0.0 < global_value < boundary <= 1.0
    assert result["score"].item() == pytest.approx(
        boundary * (1.0 - global_value), abs=1e-7
    )


def test_identical_views_have_zero_disagreement() -> None:
    probability = torch.rand(3, 1, 13, 15)
    result = compute_pc_bacs_score(probability, probability.clone())

    for value in result.values():
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0.0, atol=0.0)


def test_score_is_float32_finite_and_bounded_for_half_nan_inf_input() -> None:
    probability = torch.rand(2, 1, 7, 9, dtype=torch.float16)
    transformed = torch.rand(2, 1, 7, 9, dtype=torch.float16)
    probability[0, 0, 0, 0] = float("nan")
    transformed[0, 0, 0, 1] = float("inf")
    transformed[1, 0, 0, 2] = -float("inf")

    result = compute_pc_bacs_score(probability, transformed)

    for value in result.values():
        assert value.dtype == torch.float32
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
        assert bool(((0.0 <= value) & (value <= 1.0)).all())


@pytest.mark.parametrize(
    ("probability", "transformed", "error"),
    [
        (torch.zeros(1, 8, 8), torch.zeros(1, 8, 8), ValueError),
        (torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), ValueError),
        (torch.zeros(1, 1, 8, 8), torch.zeros(2, 1, 8, 8), ValueError),
    ],
)
def test_score_rejects_invalid_shapes(probability, transformed, error) -> None:
    with pytest.raises(error):
        compute_pc_bacs_score(probability, transformed)


def test_score_rejects_invalid_eps() -> None:
    probability = torch.zeros(1, 1, 8, 8)
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            compute_pc_bacs_score(probability, probability, eps=eps)


class _FlipEquivariantSelector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modes: list[str] = []

    def forward(self, images: torch.Tensor, *, pc_mode: str):
        self.modes.append(pc_mode)
        logits = images[:, :1]
        return None, None, None, logits


def test_score_pool_uses_output_three_off_mode_and_stable_keys() -> None:
    selector = _FlipEquivariantSelector()
    images = torch.linspace(-2.0, 2.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    loader = [(["TR-CAMO/a", "TR-COD10K/b"], images)]

    records = score_pool(selector, loader, "cpu", use_amp=True)

    assert selector.modes == ["off"]
    assert [record["sample_key"] for record in records] == [
        "TR-CAMO/a",
        "TR-COD10K/b",
    ]
    assert all(math.isfinite(record["score"]) for record in records)
    assert all(record["score"] == pytest.approx(0.0) for record in records)


def test_score_pool_rejects_duplicate_keys() -> None:
    selector = _FlipEquivariantSelector()
    loader = [(["TR-CAMO/a", "TR-CAMO/a"], torch.zeros(2, 3, 8, 8))]

    with pytest.raises(ValueError, match="duplicate"):
        score_pool(selector, loader, "cpu")
