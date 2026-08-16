from __future__ import annotations

import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.retrieval import PairVerifier
from Model.PC_HBM.training.losses import binary_pair_loss


GT = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])


def _config(**kwargs) -> DinoPCHBMConfig:
    options = {"child_verification_mode": "parent_conditioned"}
    options.update(kwargs)
    return DinoPCHBMConfig(**options)


def _aux(
    pair_logits: torch.Tensor,
    *,
    query_valid: torch.Tensor | None = None,
) -> dict:
    count = pair_logits.shape[0]
    if query_valid is None:
        query_valid = torch.ones(count, dtype=torch.bool)
    return {
        "pc_hbm": {
            "pair_logits": pair_logits,
            "query_valid": query_valid.to(device=pair_logits.device),
            "query_batch_ids": torch.zeros(
                count, dtype=torch.long, device=pair_logits.device
            ),
            "query_flat_indices": torch.arange(
                count, dtype=torch.long, device=pair_logits.device
            ),
            "query_mask_map": torch.ones(
                1, 1, 2, 2, device=pair_logits.device
            ),
        }
    }


def test_pair_loss_is_exactly_region_ce_without_candidate_inputs() -> None:
    pair_logits = torch.tensor(
        [[3.0, -3.0], [-2.0, 2.0]], requires_grad=True
    )
    loss, metrics = binary_pair_loss(
        _aux(pair_logits), GT, pair_logits, _config()
    )

    expected = F.cross_entropy(pair_logits.float(), torch.tensor([0, 1]))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["L_pair_region"], expected)
    assert "L_candidate_verify" not in metrics
    assert "candidate_valid_count" not in metrics


def test_parent_conditioned_empty_and_unsupervised_masks_are_zero() -> None:
    pair_logits = torch.randn(2, 2, requires_grad=True)
    aux = _aux(
        pair_logits,
        query_valid=torch.zeros(2, dtype=torch.bool),
    )
    loss, metrics = binary_pair_loss(aux, GT, pair_logits, _config())
    assert loss == 0.0
    assert metrics["pair_valid_count"] == 0.0
    loss.backward()
    assert pair_logits.grad is not None
    assert torch.count_nonzero(pair_logits.grad) == 0

    outside_logits = torch.randn(1, 2, requires_grad=True)
    outside = _aux(outside_logits)
    outside["pc_hbm"]["query_flat_indices"] = torch.tensor([5])
    outside["pc_hbm"]["query_mask_map"] = torch.ones(1, 1, 4, 4)
    outside_loss, outside_metrics = binary_pair_loss(
        outside,
        torch.ones(1, 1, 4, 4),
        outside_logits,
        _config(),
    )
    assert outside_loss == 0.0
    assert outside_metrics["pair_valid_count"] == 0.0


def test_parent_conditioned_fp16_loss_is_fp32_finite() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_logits = torch.randn(
        2, 2, device=device, dtype=torch.float16, requires_grad=True
    )
    loss, metrics = binary_pair_loss(
        _aux(pair_logits), GT.to(device), pair_logits, _config()
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert pair_logits.grad is not None
    assert torch.isfinite(pair_logits.grad).all()


def test_region_ce_routes_matcher_gradients() -> None:
    torch.manual_seed(29)
    verifier = PairVerifier(
        dim=4,
        tau_parent=1.0,
        child_verification_mode="parent_conditioned",
    )
    q3 = torch.randn(2, 4)
    q_child = torch.randn(2, 4, requires_grad=True)
    parent = torch.randn(2, 2, 2, 4)
    child = torch.randn(2, 2, 2, 4, requires_grad=True)
    retrieval = {
        "parent_keys": parent,
        "paired_p2_keys": child,
        "valid": torch.ones(2, 2, 2, dtype=torch.bool),
    }

    result = verifier(q3, q_child, retrieval, torch.ones(2, 1))
    aux = {
        "pc_hbm": {
            **result,
            "retrieval_valid": retrieval["valid"],
            "query_batch_ids": torch.zeros(2, dtype=torch.long),
            "query_flat_indices": torch.arange(2),
            "query_mask_map": torch.ones(1, 1, 2, 2),
        }
    }
    loss, metrics = binary_pair_loss(
        aux,
        GT,
        q_child,
        _config(),
    )
    loss.backward()
    torch.testing.assert_close(loss.detach(), metrics["L_pair_region"])
    assert verifier.parent_to_child.weight.grad is not None
    assert verifier.raw_verification_strength.grad is not None
