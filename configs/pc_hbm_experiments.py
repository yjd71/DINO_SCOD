"""Reproducible original-Decoder PC-HBM experiment profiles.

Encoder-side profiles configure :class:`EncoderPCHBMConfig`; decoder-side
profiles configure :class:`DinoPCHBMConfig`.  Every profile uses the canonical
original Transformer Decoder.  Component ablations are restricted to the
encoder adapter so Decoder architecture and checkpoint identity remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


_DISABLED_STAGE_EPOCH = 1_000_000_000


@dataclass(frozen=True)
class PCHBMExperimentProfile:
    """Decision-complete PC placement, ablation, and Base-stage selection."""

    name: str
    pc_placement: Literal["decoder", "encoder"]
    base_mode: Literal["scheduled", "off", "parent_only"] = "scheduled"
    decoder_arch: Literal["legacy_transformer"] = "legacy_transformer"
    encoder_overrides: tuple[tuple[str, Any], ...] = ()
    description: str = ""

    def apply(self, config: Any) -> Any:
        """Apply this profile in-place and return ``config`` for composition."""

        required = ("decoder_arch", "experiment_profile", "pc_placement")
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise TypeError(
                "PC-HBM experiment profiles require a compatible config; "
                f"missing fields: {missing}"
            )

        architecture = getattr(config, "architecture", None)
        is_encoder_config = architecture == "DINO_SCOD_ENCODER_PC_HBM"
        if self.pc_placement == "encoder" and not is_encoder_config:
            raise TypeError("Encoder profiles require EncoderPCHBMConfig.")
        if self.pc_placement == "decoder" and is_encoder_config:
            raise TypeError("Decoder-side profiles require DinoPCHBMConfig.")

        config.experiment_profile = self.name
        config.decoder_arch = self.decoder_arch
        config.pc_placement = self.pc_placement

        if self.pc_placement == "encoder":
            for name, value in self.encoder_overrides:
                if not hasattr(config, name):
                    raise TypeError(
                        "Encoder experiment profile override is unsupported by "
                        f"the config: {name}"
                    )
                setattr(config, name, value)
            return config

        for name in ("experiment_base_mode", "parent_start_epoch", "full_pc_start_epoch"):
            if not hasattr(config, name):
                raise TypeError(
                    "Decoder-side profiles require DinoPCHBMConfig; "
                    f"missing field: {name}"
                )
        config.experiment_base_mode = self.base_mode
        if self.base_mode == "off":
            config.parent_start_epoch = _DISABLED_STAGE_EPOCH
            config.full_pc_start_epoch = _DISABLED_STAGE_EPOCH
        elif self.base_mode == "parent_only":
            config.parent_start_epoch = 1
            config.full_pc_start_epoch = _DISABLED_STAGE_EPOCH
        return config


_PROFILES = {
    "encoder_pc": PCHBMExperimentProfile(
        name="encoder_pc",
        pc_placement="encoder",
        description="Full encoder-side PC-HBM with the original Decoder.",
    ),
    "encoder_pc_f4_f3": PCHBMExperimentProfile(
        name="encoder_pc_f4_f3",
        pc_placement="encoder",
        encoder_overrides=(("enable_f2_f1_propagation", False),),
        description="Encoder PC-HBM without F2/F1 propagation.",
    ),
    "encoder_pc_no_route_loss": PCHBMExperimentProfile(
        name="encoder_pc_no_route_loss",
        pc_placement="encoder",
        encoder_overrides=(("lambda_route", 0.0),),
        description="Full encoder PC-HBM with route InfoNCE ablated.",
    ),
    "legacy_pc": PCHBMExperimentProfile(
        name="legacy_pc",
        pc_placement="decoder",
        description="Original Decoder with scheduled decoder-side PC-HBM.",
    ),
    "legacy_off": PCHBMExperimentProfile(
        name="legacy_off",
        pc_placement="decoder",
        base_mode="off",
        description="Original Decoder with decoder-side retrieval disabled.",
    ),
    "parent_only": PCHBMExperimentProfile(
        name="parent_only",
        pc_placement="decoder",
        base_mode="parent_only",
        description="Original Decoder with parent retrieval and no full correction.",
    ),
}

_ALIASES = {"default": "encoder_pc"}


def experiment_profile_names(*, include_aliases: bool = True) -> tuple[str, ...]:
    """Return stable CLI choices in experiment-table order."""

    names = tuple(_PROFILES)
    if include_aliases:
        names += tuple(_ALIASES)
    return names


def build_experiment_profile(
    name: str = "encoder_pc",
) -> PCHBMExperimentProfile:
    """Resolve a canonical profile without mutating a configuration."""

    normalized = str(name).strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return _PROFILES[normalized]
    except KeyError as error:
        choices = ", ".join(experiment_profile_names())
        raise ValueError(
            f"Unknown experiment profile {name!r}; choose one of: {choices}"
        ) from error


def apply_experiment_profile(
    config: Any,
    name: str = "encoder_pc",
) -> PCHBMExperimentProfile:
    """Resolve and apply a profile, returning its immutable specification."""

    profile = build_experiment_profile(name)
    profile.apply(config)
    return profile


__all__ = [
    "PCHBMExperimentProfile",
    "apply_experiment_profile",
    "build_experiment_profile",
    "experiment_profile_names",
]
