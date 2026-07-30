"""Explicitly export only legacy Decoder weights from any checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.checkpoint_pc_hbm import (
    extract_non_pc_decoder_state,
    state_dict_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discard every pc_hbm.* tensor and export a baseline-only "
            "Decoder checkpoint."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def export_non_pc_decoder(
    source: Path,
    destination: Path,
    *,
    force: bool = False,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; pass --force to replace it"
        )
    state = extract_non_pc_decoder_state(source, clone=True)
    if not state:
        raise RuntimeError("Checkpoint contains no baseline Decoder tensors")
    if any(name.startswith("pc_hbm.") for name in state):
        raise AssertionError("Filtered state still contains PC tensors")
    payload = {
        "format_version": 2,
        "checkpoint_type": "baseline_decoder_export",
        "decoder": state,
        "baseline_fingerprint": state_dict_fingerprint(state),
        "source": str(source.resolve()),
        "filtered_prefix": "pc_hbm.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload


def main() -> None:
    args = parse_args()
    payload = export_non_pc_decoder(
        args.input,
        args.output,
        force=args.force,
    )
    print(
        f"Exported {len(payload['decoder'])} baseline tensors to "
        f"{args.output} ({payload['baseline_fingerprint']})"
    )


if __name__ == "__main__":
    main()
