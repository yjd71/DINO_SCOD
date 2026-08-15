from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.base_model_config import Config
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training.supervision import build_pair_label_map
from Model.base_model import BaseModel
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    load_memory_checkpoint,
    read_pc_config,
)
from utils.dataloader import LabeledMemoryDataset
from utils.pc_memory_runner import module_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Parent-only, true-paired, and same-region shuffled "
            "P2 Child verification on deterministic labeled data."
        )
    )
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--memory-checkpoint", required=True)
    parser.add_argument("--output-json", default="child_verification_eval.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--sample-txt", default=None)
    parser.add_argument("--labeled-indices-pt", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def exact_tie_aware_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().float().reshape(-1)
    targets = targets.detach().bool().reshape(-1)
    positive_count = targets.sum()
    negative_count = targets.numel() - positive_count
    if int(positive_count) == 0 or int(negative_count) == 0:
        return 0.0

    sorted_scores, order = torch.sort(scores)
    sorted_targets = targets[order]
    _, tie_counts = torch.unique_consecutive(
        sorted_scores, return_counts=True
    )
    tie_starts = tie_counts.cumsum(dim=0) - tie_counts
    average_ranks = tie_starts.float() + (tie_counts.float() + 1.0) * 0.5
    rank_per_item = torch.repeat_interleave(average_ranks, tie_counts)
    positive_count_fp = positive_count.float()
    u_statistic = (
        rank_per_item[sorted_targets].sum()
        - positive_count_fp * (positive_count_fp + 1.0) * 0.5
    )
    return float(
        (u_statistic / (positive_count_fp * negative_count.float())).item()
    )


def same_region_derangement(
    child_keys: torch.Tensor,
    valid: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Seeded non-zero cyclic shifts within every query/region valid set."""

    if child_keys.ndim != 4 or valid.shape != child_keys.shape[:-1]:
        raise ValueError("child_keys and valid must be [M,2,K,D] and [M,2,K]")
    shuffled = child_keys.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    eligible = 0
    groups = 0
    for query_index in range(valid.shape[0]):
        for region_index in range(valid.shape[1]):
            indices = valid[query_index, region_index].nonzero(
                as_tuple=False
            ).flatten()
            size = int(indices.numel())
            if size < 2:
                continue
            shift = int(
                torch.randint(1, size, (1,), generator=generator).item()
            )
            source_indices = indices.roll(shifts=shift)
            shuffled[query_index, region_index, indices] = child_keys[
                query_index, region_index, source_indices
            ]
            eligible += size
            groups += 1
    return shuffled, {
        "eligible_candidates": eligible,
        "valid_candidates": int(valid.sum().item()),
        "shuffled_groups": groups,
    }


def _query_supervision(
    pc: dict[str, Any], gt: torch.Tensor, pc_cfg
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = pc["query_valid"].detach().bool()
    query_map = pc["query_mask_map"]
    token_hw = tuple(int(value) for value in query_map.shape[-2:])
    labels_map = build_pair_label_map(
        gt,
        token_hw,
        boundary_kernel=pc_cfg.fg_boundary_kernel,
        bg_near_kernel=pc_cfg.bg_near_kernel,
        threshold=pc_cfg.gt_binary_threshold,
    ).reshape(gt.shape[0], -1)
    batch_ids = pc["query_batch_ids"].long()
    flat_indices = pc["query_flat_indices"].long()
    labels = labels_map[batch_ids, flat_indices]
    return valid & (labels >= 0), labels.long()


def _new_counts() -> dict[str, int]:
    return {
        "supervised_queries": 0,
        "parent_correct": 0,
        "verified_correct": 0,
        "repairs": 0,
        "harms": 0,
    }


def _update_counts(
    counts: dict[str, int],
    parent_logits: torch.Tensor,
    verified_logits: torch.Tensor,
    labels: torch.Tensor,
    supervised: torch.Tensor,
) -> None:
    parent_prediction = parent_logits[supervised].argmax(dim=1)
    verified_prediction = verified_logits[supervised].argmax(dim=1)
    selected_labels = labels[supervised]
    parent_correct = parent_prediction == selected_labels
    verified_correct = verified_prediction == selected_labels
    counts["supervised_queries"] += int(selected_labels.numel())
    counts["parent_correct"] += int(parent_correct.sum().item())
    counts["verified_correct"] += int(verified_correct.sum().item())
    counts["repairs"] += int(((~parent_correct) & verified_correct).sum().item())
    counts["harms"] += int((parent_correct & (~verified_correct)).sum().item())


def _candidate_observations(
    verify_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    labels: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = (
        torch.arange(2, device=labels.device).view(1, 2, 1)
        == labels.view(-1, 1, 1)
    ).expand_as(candidate_valid)
    mask = candidate_valid.bool() & supervised[:, None, None]
    return verify_logits.detach()[mask].cpu(), target[mask].cpu()


def _finalize_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = counts["supervised_queries"]
    denominator = max(total, 1)
    repair_rate = counts["repairs"] / denominator
    harm_rate = counts["harms"] / denominator
    return {
        "raw_counts": dict(counts),
        "parent_accuracy": counts["parent_correct"] / denominator,
        "verified_accuracy": counts["verified_correct"] / denominator,
        "repair_rate": repair_rate,
        "harm_rate": harm_rate,
        "net_gain": repair_rate - harm_rate,
    }


def _difference(true_values: dict[str, Any], shuffled_values: dict[str, Any]) -> dict[str, float]:
    names = ("verified_accuracy", "repair_rate", "harm_rate", "net_gain", "candidate_auroc")
    return {
        name: float(true_values[name]) - float(shuffled_values[name])
        for name in names
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    pc_cfg = read_pc_config(
        args.decoder_checkpoint, context="Child verification Decoder"
    )
    runtime_cfg = Config()
    dataset = LabeledMemoryDataset(
        l_image_root=args.image_root or runtime_cfg.train_imgs,
        l_gt_root=args.gt_root or runtime_cfg.train_gts,
        l_txt_root=args.sample_txt or runtime_cfg.train_sample_txt,
        l_train_size=pc_cfg.input_size,
        labeled_indices_pt=args.labeled_indices_pt,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BaseModel(pc_cfg=pc_cfg)
    load_decoder_compatible(
        model.decoder,
        args.decoder_checkpoint,
        require_pc_complete=True,
        expected_pc_cfg=pc_cfg,
    )
    memory = PCMemory(config=pc_cfg)
    load_memory_checkpoint(
        args.memory_checkpoint,
        memory,
        expected_compat=pc_cfg.expected_memory_meta(
            producer_fingerprint=module_fingerprint(model.decoder)
        ),
        require_producer_match=True,
    )
    model.to(device).eval()

    true_counts = _new_counts()
    shuffled_counts = _new_counts()
    true_scores: list[torch.Tensor] = []
    true_targets: list[torch.Tensor] = []
    shuffled_scores: list[torch.Tensor] = []
    shuffled_targets: list[torch.Tensor] = []
    shuffle_totals = {
        "eligible_candidates": 0,
        "valid_candidates": 0,
        "shuffled_groups": 0,
    }

    with torch.no_grad():
        for batch_index, (sample_keys, images, gt) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            _, aux = model(
                images,
                memory=memory,
                pc_mode="verify_only",
                return_aux=True,
                query_image_ids=list(sample_keys),
            )
            pc = aux["pc_hbm"]
            supervised, labels = _query_supervision(pc, gt, pc_cfg)
            parent_logits = pc["parent_region_logits"]
            _update_counts(
                true_counts,
                parent_logits,
                pc["pair_logits"],
                labels,
                supervised,
            )
            observations = _candidate_observations(
                pc["child_match_logits"],
                pc["retrieval_valid"],
                labels,
                supervised,
            )
            true_scores.append(observations[0])
            true_targets.append(observations[1])

            shuffled_keys, shuffle_batch = same_region_derangement(
                pc["retrieval_child_keys"],
                pc["retrieval_valid"],
                seed=args.seed + batch_index,
            )
            for name, value in shuffle_batch.items():
                shuffle_totals[name] += value
            shuffled_result = model.decoder.pc_hbm.pair_verifier(
                pc["verification_q3"],
                pc["verification_q_child"],
                {
                    "parent_keys": pc["retrieval_parent_keys"],
                    "paired_p2_keys": shuffled_keys,
                    "valid": pc["retrieval_valid"],
                },
                pc["query_scores"],
            )
            _update_counts(
                shuffled_counts,
                parent_logits,
                shuffled_result["pair_logits"],
                labels,
                supervised,
            )
            observations = _candidate_observations(
                shuffled_result["child_match_logits"],
                pc["retrieval_valid"],
                labels,
                supervised,
            )
            shuffled_scores.append(observations[0])
            shuffled_targets.append(observations[1])

    true_result = _finalize_counts(true_counts)
    shuffled_result = _finalize_counts(shuffled_counts)
    true_result["candidate_auroc"] = exact_tie_aware_auroc(
        torch.cat(true_scores) if true_scores else torch.empty(0),
        torch.cat(true_targets) if true_targets else torch.empty(0, dtype=torch.bool),
    )
    shuffled_result["candidate_auroc"] = exact_tie_aware_auroc(
        torch.cat(shuffled_scores) if shuffled_scores else torch.empty(0),
        torch.cat(shuffled_targets)
        if shuffled_targets
        else torch.empty(0, dtype=torch.bool),
    )
    parent_only = {
        "raw_counts": {
            "supervised_queries": true_counts["supervised_queries"],
            "correct": true_counts["parent_correct"],
        },
        "accuracy": true_result["parent_accuracy"],
    }
    report = {
        "child_verification_mode": pc_cfg.child_verification_mode,
        "seed": args.seed,
        "dataset_samples": len(dataset),
        "parent_only": parent_only,
        "true_paired": true_result,
        "same_region_shuffled": shuffled_result,
        "true_minus_shuffled": _difference(true_result, shuffled_result),
        "shuffle": {
            **shuffle_totals,
            "effective_candidate_ratio": (
                shuffle_totals["eligible_candidates"]
                / max(shuffle_totals["valid_candidates"], 1)
            ),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
