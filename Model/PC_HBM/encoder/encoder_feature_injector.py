"""Zero-initialized restoration of memory evidence into DINO token space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _strength(maximum: float, progress: float, alpha: torch.Tensor) -> torch.Tensor:
    progress_value = min(max(float(progress), 0.0), 1.0)
    return alpha.new_tensor(float(maximum) * progress_value) * torch.tanh(alpha)


@dataclass(frozen=True)
class EncoderF4F3InjectionOutput:
    f3_tokens: torch.Tensor
    f4_tokens: torch.Tensor
    f3_delta: torch.Tensor
    f4_delta: torch.Tensor
    f3_strength: torch.Tensor
    f4_strength: torch.Tensor


class RouteTokenContextAdapter(nn.Module):
    """Condition every online F4 token on the routed 128-D memory context."""

    def __init__(self, encoder_dim: int = 768, memory_dim: int = 128) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.memory_dim = int(memory_dim)
        self.f4_projector = nn.Sequential(
            nn.Linear(encoder_dim, memory_dim),
            nn.GELU(),
            nn.LayerNorm(memory_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim * 2),
            nn.GELU(),
            nn.Linear(memory_dim * 2, memory_dim),
            nn.LayerNorm(memory_dim),
        )

    def forward(
        self, f4_tokens: torch.Tensor, route_context: torch.Tensor
    ) -> torch.Tensor:
        if f4_tokens.ndim != 3 or f4_tokens.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"f4_tokens must be [B,N,{self.encoder_dim}], got "
                f"{tuple(f4_tokens.shape)}."
            )
        batch, token_count, _ = f4_tokens.shape
        if route_context.shape != (batch, self.memory_dim):
            raise ValueError(
                f"route_context must be [B,{self.memory_dim}], got "
                f"{tuple(route_context.shape)}."
            )
        token_state = self.f4_projector(f4_tokens)
        route_state = route_context[:, None, :].expand(-1, token_count, -1)
        return self.fusion(
            torch.cat(
                (token_state, route_state, token_state * route_state), dim=-1
            )
        )


class EncoderFeatureInjector(nn.Module):
    """Inject route/F3 evidence while preserving exact initial identity."""

    def __init__(
        self,
        memory_dim: int = 128,
        encoder_dim: int = 768,
        *,
        max_f4: float = 0.25,
        max_f3: float = 1.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.encoder_dim = int(encoder_dim)
        self.max_f4 = float(max_f4)
        self.max_f3 = float(max_f3)
        self.restore4 = _zero_linear(memory_dim, encoder_dim)
        self.restore3 = _zero_linear(memory_dim, encoder_dim)
        self.route_context_adapter = RouteTokenContextAdapter(
            encoder_dim=encoder_dim,
            memory_dim=memory_dim,
        )
        self.alpha4 = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha3 = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        *,
        f3_tokens: torch.Tensor,
        f4_tokens: torch.Tensor,
        route_evidence: torch.Tensor,
        route_confidence: torch.Tensor,
        route_valid: torch.Tensor,
        verified_f3_map: torch.Tensor,
        f3_gate_map: torch.Tensor,
        progress: float,
    ) -> EncoderF4F3InjectionOutput:
        self._validate_tokens(f3_tokens, "f3_tokens")
        self._validate_tokens(f4_tokens, "f4_tokens")
        batch, token_count, _ = f3_tokens.shape
        if f4_tokens.shape[:2] != (batch, token_count):
            raise ValueError("F3 and F4 token layouts must match.")
        if route_evidence.shape != (batch, self.memory_dim):
            raise ValueError(
                f"route_evidence must be [B,{self.memory_dim}], got "
                f"{tuple(route_evidence.shape)}."
            )
        if route_confidence.shape not in {(batch,), (batch, 1)}:
            raise ValueError("route_confidence must be [B] or [B,1].")
        if route_valid.shape not in {(batch,), (batch, 1)}:
            raise ValueError("route_valid must be [B] or [B,1].")
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError("DINO token count must form a square map.")
        if verified_f3_map.shape != (batch, self.memory_dim, side, side):
            raise ValueError("verified_f3_map has an incompatible shape.")
        if f3_gate_map.shape != (batch, 1, side, side):
            raise ValueError("f3_gate_map has an incompatible shape.")

        route_tokens = self.route_context_adapter(f4_tokens, route_evidence)
        verified_tokens = verified_f3_map.flatten(2).transpose(1, 2)
        gate_tokens = f3_gate_map.flatten(2).transpose(1, 2).to(verified_tokens.dtype)
        f4_strength = _strength(self.max_f4, progress, self.alpha4)
        f3_strength = _strength(self.max_f3, progress, self.alpha3)
        route_gate = (
            route_confidence.reshape(batch, 1, 1).to(route_tokens)
            * route_valid.reshape(batch, 1, 1).to(route_tokens)
        )
        # Gates are deliberately outside the affine restore projections.  This
        # preserves strict local identity for invalid evidence even after the
        # restore biases have learned non-zero values.
        f4_delta = f4_strength * route_gate * self.restore4(route_tokens)
        f3_delta = f3_strength * gate_tokens * self.restore3(verified_tokens)
        return EncoderF4F3InjectionOutput(
            f3_tokens=f3_tokens + f3_delta,
            f4_tokens=f4_tokens + f4_delta,
            f3_delta=f3_delta,
            f4_delta=f4_delta,
            f3_strength=f3_strength,
            f4_strength=f4_strength,
        )

    def _validate_tokens(self, value: torch.Tensor, name: str) -> None:
        if value.ndim != 3 or value.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"{name} must be [B,N,{self.encoder_dim}], got {tuple(value.shape)}."
            )


__all__ = [
    "EncoderF4F3InjectionOutput",
    "EncoderFeatureInjector",
    "RouteTokenContextAdapter",
]
