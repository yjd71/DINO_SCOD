"""Decoder registry with an explicit legacy compatibility export."""

from __future__ import annotations

from typing import Any

from .legacy_decoder import (
    Attention,
    Decoder as LegacyDecoderAlias,
    FeedForwardLayer,
    LegacyTransformerDecoder,
    TransformerBlock,
    map_to_tokens,
    tokens_to_map,
)


SUPPORTED_DECODER_ARCHITECTURES = {
    "legacy_transformer",
    "bgfbr_pc_v1",
}


def resolve_decoder_arch(decoder_arch: str | None, pc_cfg: Any | None = None) -> str:
    """Resolve an architecture without inferring BGFBR from PC attachment."""

    resolved = decoder_arch
    if resolved is None and pc_cfg is not None:
        resolved = getattr(pc_cfg, "decoder_arch", None)
    if resolved is None:
        # Direct historical users (notably selector tooling) remain legacy.
        resolved = "legacy_transformer"
    resolved = str(resolved)
    if resolved not in SUPPORTED_DECODER_ARCHITECTURES:
        raise ValueError(
            f"Unsupported decoder_arch={resolved!r}; expected one of "
            f"{sorted(SUPPORTED_DECODER_ARCHITECTURES)}."
        )
    return resolved


def build_decoder(
    decoder_arch: str | None = None,
    pc_cfg: Any | None = None,
    attach_pc: bool = True,
    **kwargs: Any,
):
    """Build the concrete decoder without adding a state-dict wrapper prefix."""

    resolved = resolve_decoder_arch(decoder_arch, pc_cfg)
    if resolved == "legacy_transformer":
        decoder = LegacyTransformerDecoder(
            pc_cfg=pc_cfg if attach_pc else None, **kwargs
        )
    else:
        from .bgfbr_decoder import BGFBRDecoder

        decoder = BGFBRDecoder(
            pc_cfg=pc_cfg, attach_pc=bool(attach_pc), **kwargs
        )
    decoder.decoder_arch = resolved
    decoder.decoder_contract_version = int(
        getattr(pc_cfg, "decoder_contract_version", 1)
    )
    return decoder


# Preserve old direct imports and parity fixtures.  Application entry points
# must use ``build_decoder`` so the main configuration can default to BGFBR.
Decoder = LegacyTransformerDecoder


__all__ = [
    "Attention",
    "Decoder",
    "FeedForwardLayer",
    "LegacyDecoderAlias",
    "LegacyTransformerDecoder",
    "SUPPORTED_DECODER_ARCHITECTURES",
    "TransformerBlock",
    "build_decoder",
    "map_to_tokens",
    "resolve_decoder_arch",
    "tokens_to_map",
]
