"""Reproducible BGFBR x PC-HBM experiment profiles.

Decoder-side profiles mutate :class:`DinoPCHBMConfig`; encoder-side profiles
mutate the independent strict :class:`EncoderPCHBMConfig`.  Model shapes stay
fixed across component ablations, so checkpoints and DDP graphs remain
structurally comparable.  ``bgfbr_off`` and ``parent_only`` express Base-stage
schedule overrides; Teacher-Student keeps its explicit pseudo/core modes while
still consuming the architecture and component switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


_DISABLED_STAGE_EPOCH = 1_000_000_000


@dataclass(frozen=True)
class BGFBRExperimentProfile:
    """Decision-complete architecture, ablation, and Base-mode selection."""

    name: str
    decoder_arch: Literal["bgfbr_pc_v1", "legacy_transformer"]
    pc_placement: Literal["decoder", "encoder"] = "decoder"
    base_mode: Literal["scheduled", "off", "parent_only"] = "scheduled"
    use_gbe: bool = True
    use_ode: bool = True
    use_rcab: bool = True
    use_pc_boundary_context: bool = True
    encoder_overrides: tuple[tuple[str, Any], ...] = ()
    description: str = ""

    def apply(self, config: Any) -> Any:
        """Apply this profile in-place and return ``config`` for composition."""

        # Encoder-side profiles use an independent strict v3 config.  Keep the
        # old decoder-side requirements unchanged, while allowing the profile
        # registry to apply decision-complete encoder ablations to that config.
        if self.pc_placement == "encoder" and getattr(
            config, "architecture", None
        ) == "DINO_SCOD_ENCODER_PC_HBM":
            for name in ("decoder_arch", "experiment_profile", "pc_placement"):
                if not hasattr(config, name):
                    raise TypeError(
                        "Encoder experiment profiles require an "
                        f"EncoderPCHBMConfig-like object; missing field: {name}"
                    )
            config.experiment_profile = self.name
            config.decoder_arch = self.decoder_arch
            config.pc_placement = self.pc_placement
            for name, value in self.encoder_overrides:
                if not hasattr(config, name):
                    raise TypeError(
                        "Encoder experiment profile override is unsupported by "
                        f"the config: {name}"
                    )
                setattr(config, name, value)
            return config

        required = (
            "decoder_arch",
            "parent_start_epoch",
            "full_pc_start_epoch",
            "use_gbe",
            "use_ode",
            "use_rcab",
            "use_pc_boundary_context",
        )
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise TypeError(
                "Experiment profiles require a DinoPCHBMConfig-like object; "
                f"missing fields: {missing}"
            )

        config.experiment_profile = self.name
        config.experiment_base_mode = self.base_mode
        config.decoder_arch = self.decoder_arch
        config.pc_placement = self.pc_placement
        config.use_gbe = self.use_gbe
        config.use_ode = self.use_ode
        config.use_rcab = self.use_rcab
        config.use_pc_boundary_context = self.use_pc_boundary_context

        if self.base_mode == "off":
            config.parent_start_epoch = _DISABLED_STAGE_EPOCH
            config.full_pc_start_epoch = _DISABLED_STAGE_EPOCH
        elif self.base_mode == "parent_only":
            config.parent_start_epoch = 1
            config.full_pc_start_epoch = _DISABLED_STAGE_EPOCH
        return config


_PROFILES = {
    "bgfbr_pc": BGFBRExperimentProfile(
        name="bgfbr_pc",
        decoder_arch="bgfbr_pc_v1",
        description="Full BGFBR decoder with the scheduled PC-HBM curriculum.",
    ),
    "bgfbr_off": BGFBRExperimentProfile(
        name="bgfbr_off",
        decoder_arch="bgfbr_pc_v1",
        base_mode="off",
        description="BGFBR-only Base training; PC retrieval and correction stay off.",
    ),
    "encoder_pc": BGFBRExperimentProfile(
        name="encoder_pc",
        decoder_arch="bgfbr_pc_v1",
        pc_placement="encoder",
        description=(
            "Encoder-side PC-HBM with a detached, permanently off BGFBR decoder."
        ),
    ),
    "encoder_pc_f4_f3": BGFBRExperimentProfile(
        name="encoder_pc_f4_f3",
        decoder_arch="bgfbr_pc_v1",
        pc_placement="encoder",
        encoder_overrides=(("enable_f2_f1_propagation", False),),
        description="Encoder PC-HBM without the F2/F1 propagation levels.",
    ),
    "encoder_pc_no_route_loss": BGFBRExperimentProfile(
        name="encoder_pc_no_route_loss",
        decoder_arch="bgfbr_pc_v1",
        pc_placement="encoder",
        encoder_overrides=(("lambda_route", 0.0),),
        description="Full encoder PC-HBM with same-image route InfoNCE ablated.",
    ),
    "parent_only": BGFBRExperimentProfile(
        name="parent_only",
        decoder_arch="bgfbr_pc_v1",
        base_mode="parent_only",
        description="BGFBR plus parent retrieval, without correction injection.",
    ),
    "no_gbe": BGFBRExperimentProfile(
        name="no_gbe",
        decoder_arch="bgfbr_pc_v1",
        use_gbe=False,
        description="Full PC-HBM with zero boundary context from GBE.",
    ),
    "no_ode": BGFBRExperimentProfile(
        name="no_ode",
        decoder_arch="bgfbr_pc_v1",
        use_ode=False,
        description="Full PC-HBM with ODE blocks bypassed as identity.",
    ),
    "no_rcab": BGFBRExperimentProfile(
        name="no_rcab",
        decoder_arch="bgfbr_pc_v1",
        use_rcab=False,
        description="Full PC-HBM with RCAB blocks bypassed as identity.",
    ),
    "no_pc_boundary_context": BGFBRExperimentProfile(
        name="no_pc_boundary_context",
        decoder_arch="bgfbr_pc_v1",
        use_pc_boundary_context=False,
        description="Keep PC input widths fixed but fill boundary context with zeros.",
    ),
    "legacy_off": BGFBRExperimentProfile(
        name="legacy_off",
        decoder_arch="legacy_transformer",
        base_mode="off",
        use_gbe=False,
        use_ode=False,
        use_rcab=False,
        use_pc_boundary_context=False,
        description="Legacy Transformer decoder with PC-HBM disabled.",
    ),
}

_ALIASES = {"default": "bgfbr_pc"}


def experiment_profile_names(*, include_aliases: bool = True) -> tuple[str, ...]:
    """Return stable CLI choices in report-table order."""

    names = tuple(_PROFILES)
    if include_aliases:
        names += tuple(_ALIASES)
    return names


def build_experiment_profile(name: str = "bgfbr_pc") -> BGFBRExperimentProfile:
    """Resolve a canonical profile without mutating a configuration."""

    normalized = str(name).strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return _PROFILES[normalized]
    except KeyError as error:
        choices = ", ".join(experiment_profile_names())
        raise ValueError(f"Unknown experiment profile {name!r}; choose one of: {choices}") from error


def apply_experiment_profile(config: Any, name: str = "bgfbr_pc") -> BGFBRExperimentProfile:
    """Resolve and apply a profile, returning its immutable specification."""

    profile = build_experiment_profile(name)
    profile.apply(config)
    return profile


__all__ = [
    "BGFBRExperimentProfile",
    "apply_experiment_profile",
    "build_experiment_profile",
    "experiment_profile_names",
]
