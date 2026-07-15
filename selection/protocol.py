"""Shared sample-count protocols used by all offline selection methods."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence


_FORMAL_PROTOCOL_NAMES: dict[tuple[int, int, int], str] = {
    (40, 200, 400): "kmeans_0040_0200_0400",
    (41, 202, 404): "scout_0041_0202_0404",
}
_FORMAL_BUDGET_COLUMNS = ({40, 41}, {200, 202}, {400, 404})


@dataclass(frozen=True, slots=True)
class SamplingProtocol:
    """Validated selection budgets with an explicit bootstrap size."""

    name: str
    target_counts: tuple[int, int, int]
    bootstrap_count: int
    is_formal: bool

    @classmethod
    def from_counts(
        cls,
        counts: Iterable[int],
        allow_custom: bool = False,
    ) -> "SamplingProtocol":
        if isinstance(counts, (str, bytes)):
            raise TypeError("target counts must be an iterable of integers")
        try:
            values = tuple(counts)
        except TypeError as error:
            raise TypeError("target counts must be an iterable of integers") from error
        if len(values) != 3:
            raise ValueError("target counts must contain exactly three budgets")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("target counts must contain only integers")
        if any(value <= 0 for value in values):
            raise ValueError("target counts must be positive")
        if tuple(sorted(set(values))) != values:
            raise ValueError("target counts must be strictly increasing and unique")

        typed_values = (int(values[0]), int(values[1]), int(values[2]))
        formal_name = _FORMAL_PROTOCOL_NAMES.get(typed_values)
        if formal_name is not None:
            return cls(
                name=formal_name,
                target_counts=typed_values,
                bootstrap_count=typed_values[0],
                is_formal=True,
            )
        if all(
            value in allowed
            for value, allowed in zip(typed_values, _FORMAL_BUDGET_COLUMNS)
        ):
            raise ValueError(
                f"mixed formal target-count protocols are not allowed: {typed_values}"
            )
        if not allow_custom:
            supported = ", ".join(str(item) for item in _FORMAL_PROTOCOL_NAMES)
            raise ValueError(
                f"unsupported target-count protocol {typed_values}; supported: {supported}. "
                "Custom counts are debug-only and require allow_custom=True."
            )
        custom_name = "custom_" + "_".join(f"{value:04d}" for value in typed_values)
        return cls(
            name=custom_name,
            target_counts=typed_values,
            bootstrap_count=typed_values[0],
            is_formal=False,
        )


def add_target_counts_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> argparse.Action:
    """Add the common, explicit three-budget CLI argument."""

    return parser.add_argument(
        "--target-counts",
        nargs=3,
        type=int,
        required=required,
        metavar=("SMALL", "MEDIUM", "LARGE"),
        help="Explicit selection budgets: 40 200 400 or 41 202 404.",
    )


def protocol_from_args(
    args: argparse.Namespace,
    *,
    allow_custom: bool | None = None,
) -> SamplingProtocol:
    if allow_custom is None:
        allow_custom = bool(getattr(args, "allow_custom_counts", False))
    return SamplingProtocol.from_counts(
        getattr(args, "target_counts"), allow_custom=allow_custom
    )


def formal_protocols() -> Sequence[tuple[int, int, int]]:
    return tuple(_FORMAL_PROTOCOL_NAMES)


__all__ = [
    "SamplingProtocol",
    "add_target_counts_argument",
    "formal_protocols",
    "protocol_from_args",
]
