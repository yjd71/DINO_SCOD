"""Cosine-only Parent/Child verification and binary region aggregation."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _masked_softmax_fp32(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    work = logits.float()
    mask = valid.to(device=work.device, dtype=torch.bool)
    masked = work.masked_fill(~mask, float("-inf"))
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=dim)
    return torch.where(mask, probability, torch.zeros_like(probability))


def _masked_logmeanexp_fp32(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    work = logits.float()
    mask = valid.to(device=work.device, dtype=torch.bool)
    count = mask.sum(dim=dim)
    masked = work.masked_fill(~mask, float("-inf"))
    safe = torch.logsumexp(masked, dim=dim) - count.clamp_min(1).float().log()
    return torch.where(count > 0, safe, torch.zeros_like(safe))


def _normalized_entropy_fp32(
    probability: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    probability = probability.float()
    valid = valid.to(device=probability.device, dtype=torch.bool)
    count = valid.sum(dim=dim)
    terms = torch.where(
        valid,
        -probability * probability.clamp_min(1.0e-12).log(),
        torch.zeros_like(probability),
    )
    entropy = terms.sum(dim=dim)
    denominator = count.clamp_min(2).float().log()
    normalized = entropy / denominator
    # A singleton distribution has no uncertainty by definition.
    return torch.where(count <= 1, torch.zeros_like(normalized), normalized)


class PairVerifier(nn.Module):
    """Verify aligned P3/P2 pairs using cosine similarity only."""

    def __init__(
        self,
        dim: int = 128,
        tau_parent: float = 0.07,
        tau_child: float = 0.10,
        child_mix_init_logit: float = 0.0,
        child_verification_mode: str = "weighted_sum",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.tau_parent = float(tau_parent)
        self.tau_child = float(tau_child)
        self.child_verification_mode = str(child_verification_mode).lower()
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if (
            not math.isfinite(self.tau_parent)
            or not math.isfinite(self.tau_child)
            or self.tau_parent <= 0
            or self.tau_child <= 0
        ):
            raise ValueError("cosine temperatures must be positive")
        if not math.isfinite(float(child_mix_init_logit)):
            raise ValueError("child_mix_init_logit must be finite")
        if self.child_verification_mode != "weighted_sum":
            raise ValueError(
                "PairVerifier currently supports child_verification_mode="
                "'weighted_sum' only"
            )
        # The verifier has exactly one learnable mixing scalar.
        self.raw_child_mix = nn.Parameter(
            torch.tensor(float(child_mix_init_logit), dtype=torch.float32)
        )

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_child_mix)

    def forward(
        self,
        q3: torch.Tensor,
        q_child: torch.Tensor,
        retrieval: Mapping[str, torch.Tensor],
        query_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if q3.ndim != 2 or q3.size(1) != self.dim:
            raise ValueError(f"q3 must be [M,{self.dim}]")
        if q_child.shape != q3.shape:
            raise ValueError(f"q_child must match q3, got {tuple(q_child.shape)}")
        parent_keys = retrieval["parent_keys"]
        child_keys = retrieval["paired_p2_keys"]
        valid = retrieval["valid"].to(device=q3.device, dtype=torch.bool)
        expected_prefix = (q3.size(0), 2)
        if (
            parent_keys.ndim != 4
            or child_keys.shape != parent_keys.shape
            or parent_keys.shape[:2] != expected_prefix
            or parent_keys.size(-1) != self.dim
        ):
            raise ValueError(
                "retrieval keys must both be [M,2,K,dim] and one-to-one aligned"
            )
        if valid.shape != parent_keys.shape[:-1]:
            raise ValueError("retrieval valid mask must be [M,2,K]")
        if query_score.shape != (q3.size(0), 1):
            raise ValueError("query_score must be [M,1]")

        parent_cosine = F.cosine_similarity(
            q3.float()[:, None, None, :],
            parent_keys.float(),
            dim=-1,
            eps=1.0e-6,
        )
        child_cosine = F.cosine_similarity(
            q_child.float()[:, None, None, :],
            child_keys.float(),
            dim=-1,
            eps=1.0e-6,
        )
        beta = self.beta
        pair_scores = (
            (1.0 - beta) * (parent_cosine / self.tau_parent)
            + beta * (child_cosine / self.tau_child)
        )
        pair_scores = torch.nan_to_num(
            pair_scores,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        ).clamp(-1.0e4, 1.0e4)
        pair_scores = torch.where(
            valid, pair_scores, pair_scores.new_full((), -1.0e4)
        )

        parent_scores = torch.nan_to_num(
            parent_cosine / self.tau_parent,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        ).clamp(-1.0e4, 1.0e4)
        parent_scores = torch.where(
            valid, parent_scores, parent_scores.new_full((), -1.0e4)
        )

        region_has_candidate = valid.any(dim=-1)
        query_valid = region_has_candidate.all(dim=-1)
        pair_weight = _masked_softmax_fp32(pair_scores, valid, dim=-1)
        region_logits = _masked_logmeanexp_fp32(pair_scores, valid, dim=-1)
        region_logits = torch.where(
            query_valid[:, None], region_logits, torch.zeros_like(region_logits)
        )
        region_probability = torch.softmax(region_logits, dim=-1)
        region_probability = torch.where(
            query_valid[:, None],
            region_probability,
            torch.zeros_like(region_probability),
        )
        parent_region_logits = _masked_logmeanexp_fp32(
            parent_scores, valid, dim=-1
        )
        parent_region_logits = torch.where(
            query_valid[:, None],
            parent_region_logits,
            torch.zeros_like(parent_region_logits),
        )
        parent_region_probability = torch.softmax(
            parent_region_logits, dim=-1
        )
        parent_region_probability = torch.where(
            query_valid[:, None],
            parent_region_probability,
            torch.zeros_like(parent_region_probability),
        )

        contexts = torch.einsum(
            "mrk,mrkd->mrd", pair_weight, parent_keys.float()
        )
        contexts = torch.where(
            query_valid[:, None, None],
            contexts,
            torch.zeros_like(contexts),
        )
        memory_context = (
            region_probability.unsqueeze(-1) * contexts
        ).sum(dim=1)
        memory_context = torch.where(
            query_valid[:, None],
            memory_context,
            torch.zeros_like(memory_context),
        )

        region_entropy = _normalized_entropy_fp32(
            pair_weight, valid, dim=-1
        )
        candidate_entropy = (
            region_probability * region_entropy
        ).sum(dim=-1).clamp(0.0, 1.0)
        candidate_entropy = torch.where(
            query_valid,
            candidate_entropy,
            torch.zeros_like(candidate_entropy),
        )
        region_margin = (
            region_probability[:, 0] - region_probability[:, 1]
        ).abs().clamp(0.0, 1.0)
        region_margin = torch.where(
            query_valid,
            region_margin,
            torch.zeros_like(region_margin),
        )
        confidence = (1.0 - candidate_entropy) * region_margin
        confidence = torch.where(
            query_valid, confidence, torch.zeros_like(confidence)
        )
        safe_query_score = torch.nan_to_num(
            query_score.float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        gate = confidence[:, None] * safe_query_score
        gate = torch.where(
            query_valid[:, None], gate, torch.zeros_like(gate)
        )
        correction = memory_context - q3.float()
        correction = torch.where(
            query_valid[:, None],
            correction,
            torch.zeros_like(correction),
        )

        parent_cosine_masked = torch.where(
            valid, parent_cosine, torch.zeros_like(parent_cosine)
        )
        child_cosine_masked = torch.where(
            valid, child_cosine, torch.zeros_like(child_cosine)
        )
        zero_evidence = torch.zeros_like(child_cosine_masked)
        return {
            "parent_cosine": parent_cosine_masked,
            "parent_scores": parent_scores,
            "parent_region_logits": parent_region_logits,
            "parent_region_prob": parent_region_probability,
            "child_abs_cosine": child_cosine_masked,
            "child_relation_cosine": zero_evidence,
            "relation_valid": torch.zeros_like(valid),
            "child_verify_logits": zero_evidence,
            "verification_strength": beta.new_zeros(()),
            "verification_abs_weight": beta.new_zeros(()),
            "verification_rel_weight": beta.new_zeros(()),
            "verification_bias": beta.new_zeros(()),
            "child_cosine": child_cosine_masked,
            "pair_scores": pair_scores,
            "verified_scores": pair_scores,
            "pair_weight": pair_weight,
            "pair_logits": region_logits,
            "verified_region_logits": region_logits,
            "region_prob": region_probability,
            "verified_region_prob": region_probability,
            "region_context": contexts.to(dtype=q3.dtype),
            "memory_context": memory_context.to(dtype=q3.dtype),
            "candidate_entropy": candidate_entropy,
            "region_margin": region_margin,
            "memory_confidence": confidence[:, None],
            "gate": gate,
            "correction": correction.to(dtype=q3.dtype),
            "query_valid": query_valid,
            "beta": beta,
        }
