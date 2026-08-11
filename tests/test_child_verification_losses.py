from __future__ import annotations

import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.losses import binary_pair_loss


GT = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])


def _config(**kwargs) -> DinoPCHBMConfig:
    options = {
        "child_verification_mode": "parent_conditioned",
        "lambda_candidate_verify": 0.5,
        "lambda_parent_repair": 0.5,
        "lambda_parent_preserve": 0.25,
    }
    options.update(kwargs)
    return DinoPCHBMConfig(**options)


def _aux(
    verify_logits: torch.Tensor,
    parent_scores: torch.Tensor,
    *,
    retrieval_valid: torch.Tensor | None = None,
    query_valid: torch.Tensor | None = None,
) -> dict:
    count = verify_logits.shape[0]
    if retrieval_valid is None:
        retrieval_valid = torch.ones_like(verify_logits, dtype=torch.bool)
    if query_valid is None:
        query_valid = torch.ones(count, dtype=torch.bool)
    pair_logits = torch.tensor(
        [[4.0, -4.0], [-4.0, 4.0]][:count],
        dtype=verify_logits.dtype,
        device=verify_logits.device,
        requires_grad=True,
    )
    return {
        "pc_hbm": {
            "pair_logits": pair_logits,
            "query_valid": query_valid.to(device=verify_logits.device),
            "query_batch_ids": torch.zeros(
                count, dtype=torch.long, device=verify_logits.device
            ),
            "query_flat_indices": torch.arange(
                count, dtype=torch.long, device=verify_logits.device
            ),
            "query_mask_map": torch.ones(
                1, 1, 2, 2, device=verify_logits.device
            ),
            "child_verify_logits": verify_logits,
            "parent_scores": parent_scores,
            "retrieval_valid": retrieval_valid.to(device=verify_logits.device),
            "verification_strength": torch.tensor(
                0.5, device=verify_logits.device, requires_grad=True
            ),
        }
    }


def test_candidate_bce_is_normalized_per_query_and_masked() -> None:
    verify = torch.tensor(
        [
            [[3.0, -9.0], [-3.0, -3.0]],
            [[-2.0, -2.0], [2.0, 2.0]],
        ],
        requires_grad=True,
    )
    parent = torch.zeros_like(verify, requires_grad=True)
    valid = torch.tensor(
        [
            [[True, False], [True, True]],
            [[True, True], [True, True]],
        ]
    )
    cfg = _config(
        lambda_candidate_verify=1.0,
        lambda_parent_repair=0.0,
        lambda_parent_preserve=0.0,
    )

    loss, metrics = binary_pair_loss(
        _aux(verify, parent, retrieval_valid=valid), GT, verify, cfg
    )

    targets = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    raw = F.binary_cross_entropy_with_logits(verify, targets, reduction="none")
    expected = torch.stack(
        [raw[0][valid[0]].mean(), raw[1][valid[1]].mean()]
    ).mean()
    torch.testing.assert_close(metrics["L_candidate_verify"], expected)
    torch.testing.assert_close(
        loss,
        metrics["L_pair_region"] + expected,
    )


def test_repair_loss_rewards_gain_without_parent_prior_gradient() -> None:
    parent = torch.tensor(
        [
            [[-1.0, -1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    good_verify = torch.tensor(
        [
            [[3.0, 3.0], [-3.0, -3.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        requires_grad=True,
    )
    cfg = _config(
        lambda_candidate_verify=0.0,
        lambda_parent_repair=1.0,
        lambda_parent_preserve=0.0,
    )
    loss, metrics = binary_pair_loss(_aux(good_verify, parent), GT, good_verify, cfg)

    assert metrics["parent_repair_count"] == 1.0
    assert metrics["L_parent_repair"] == 0.0
    loss.backward()
    assert parent.grad is None
    assert good_verify.grad is not None and torch.isfinite(good_verify.grad).all()

    bad_verify = torch.zeros(2, 2, 2, requires_grad=True)
    _, bad_metrics = binary_pair_loss(
        _aux(bad_verify, parent.detach().clone().requires_grad_()),
        GT,
        bad_verify,
        cfg,
    )
    assert bad_metrics["L_parent_repair"] > 0.0


def test_preserve_loss_penalizes_harm_to_confident_correct_parent() -> None:
    parent = torch.tensor(
        [
            [[-1.0, -1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    verify = torch.tensor(
        [
            [[3.0, 3.0], [-3.0, -3.0]],
            [[3.0, 3.0], [-3.0, -3.0]],
        ],
        requires_grad=True,
    )
    cfg = _config(
        lambda_candidate_verify=0.0,
        lambda_parent_repair=0.0,
        lambda_parent_preserve=1.0,
    )

    loss, metrics = binary_pair_loss(_aux(verify, parent), GT, verify, cfg)

    assert metrics["parent_preserve_count"] == 1.0
    assert metrics["L_parent_preserve"] > 0.0
    loss.backward()
    assert parent.grad is None
    assert verify.grad is not None and torch.isfinite(verify.grad).all()


def test_parent_conditioned_empty_and_unsupervised_masks_are_zero() -> None:
    verify = torch.randn(2, 2, 2, requires_grad=True)
    parent = torch.randn(2, 2, 2, requires_grad=True)
    aux = _aux(
        verify,
        parent,
        query_valid=torch.zeros(2, dtype=torch.bool),
    )
    loss, metrics = binary_pair_loss(aux, GT, verify, _config())
    assert loss == 0.0
    assert metrics["candidate_valid_count"] == 0.0
    loss.backward()
    assert verify.grad is not None

    one_verify = torch.randn(1, 2, 2, requires_grad=True)
    one_parent = torch.randn(1, 2, 2, requires_grad=True)
    outside = _aux(one_verify, one_parent)
    outside["pc_hbm"]["query_flat_indices"] = torch.tensor([5])
    outside["pc_hbm"]["query_mask_map"] = torch.ones(1, 1, 4, 4)
    outside_loss, outside_metrics = binary_pair_loss(
        outside,
        torch.ones(1, 1, 4, 4),
        one_verify,
        _config(),
    )
    assert outside_loss == 0.0
    assert outside_metrics["pair_valid_count"] == 0.0


def test_parent_conditioned_fp16_loss_is_fp32_finite() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verify = torch.randn(
        2, 2, 2, device=device, dtype=torch.float16, requires_grad=True
    )
    parent = torch.randn(
        2, 2, 2, device=device, dtype=torch.float16, requires_grad=True
    )
    loss, metrics = binary_pair_loss(
        _aux(verify, parent), GT.to(device), verify, _config()
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert verify.grad is not None and torch.isfinite(verify.grad).all()
