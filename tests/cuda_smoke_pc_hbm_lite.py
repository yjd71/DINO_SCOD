"""Required single-GPU forward/backward smoke for PC-HBM-Lite."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.base_model_config import Config
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    base_structure_loss,
    pc_hbm_labeled_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
)
from Model.base_model import BaseModel
from Model.decoder import Decoder
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    load_decoder_compatible,
    save_decoder_checkpoint,
    state_dict_fingerprint,
    validate_labeled_indices_pt,
)
from utils.dataloader import build_labeled_memory_loader
from utils.pc_memory_runner import module_fingerprint, rebuild_memory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--labeled-indices-pt", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_finite_gradients(module) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError("No gradients were produced")
    if not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise FloatingPointError("Non-finite gradient in CUDA smoke")


def assert_all_finite_gradients(module) -> None:
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise AssertionError(f"Unused trainable parameters: {missing}")
    nonfinite = [
        name
        for name, parameter in module.named_parameters()
        if not bool(torch.isfinite(parameter.grad).all())
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite gradients: {nonfinite}")


def assert_finite_loss(loss: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"Non-finite {name} in CUDA smoke")


def assert_dino_frozen(model: BaseModel) -> None:
    if any(parameter.requires_grad for parameter in model.dino.parameters()):
        raise AssertionError("DINO parameters unexpectedly require gradients")
    if any(parameter.grad is not None for parameter in model.dino.parameters()):
        raise AssertionError("Frozen DINO accumulated gradients")


def reset_peak_memory() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Lite smoke test")
    set_seed(args.seed)
    device = torch.device("cuda")
    cfg = Config()
    cfg.train_labeled_indices_pt = str(args.labeled_indices_pt)
    split_identity = validate_labeled_indices_pt(
        args.labeled_indices_pt
    )
    if split_identity.count != 202:
        raise RuntimeError(
            f"CUDA smoke requires 202 labeled samples, got {split_identity.count}"
        )
    split_fingerprint = split_identity.fingerprint
    pc_cfg = DinoPCHBMConfig()
    cfg.train_size = int(pc_cfg.input_size)
    model = BaseModel(pc_cfg=pc_cfg).to(device).train()
    verifier_parameter_count = sum(
        parameter.numel()
        for parameter in model.decoder.pc_hbm.pair_verifier.parameters()
    )
    if pc_cfg.child_verification_mode != "parent_conditioned":
        raise AssertionError("CUDA smoke must exercise parent_conditioned mode")
    if verifier_parameter_count != 16_388:
        raise AssertionError(
            f"Expected 16,388 verifier parameters, got {verifier_parameter_count}"
        )
    loader = build_labeled_memory_loader(
        l_image_root=cfg.train_imgs,
        l_gt_root=cfg.train_masks,
        l_txt_root=cfg.train_sample_txt,
        l_train_size=cfg.train_size,
        labeled_indices_pt=str(args.labeled_indices_pt),
        batch_size=16,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if len(loader.dataset) != 202:
        raise RuntimeError(
            f"Expected the fixed 202-key split, got {len(loader.dataset)}"
        )
    memory = PCMemory(config=pc_cfg)
    rebuild_memory(
        model=model,
        memory_decoder=model.decoder,
        memory_loader=loader,
        memory=memory,
        device=device,
        config=pc_cfg,
        use_amp=True,
    )
    if not memory.is_ready():
        raise RuntimeError("Real labeled-only memory rebuild failed")
    with tempfile.TemporaryDirectory(prefix="pcv-cuda-roundtrip-") as temporary:
        checkpoint_path = Path(temporary) / "decoder.pth"
        save_decoder_checkpoint(checkpoint_path, model.decoder, pc_cfg, epoch=0)
        roundtrip = Decoder(
            in_dim=pc_cfg.encoder_dim,
            out_dim=pc_cfg.decoder_dim,
            pc_cfg=pc_cfg,
        )
        load_decoder_compatible(
            roundtrip,
            checkpoint_path,
            require_pc_complete=True,
            expected_pc_cfg=pc_cfg,
        )
        for name, value in model.decoder.state_dict().items():
            if not torch.equal(value.detach().cpu(), roundtrip.state_dict()[name]):
                raise AssertionError(f"Checkpoint round-trip mismatch: {name}")
    # rebuild_memory intentionally evaluates the producer; restore training mode.
    model.train()
    assert_dino_frozen(model)

    # Base full: physical batch 16, two AMP optimizer steps.
    base_images = torch.randn(
        16,
        3,
        pc_cfg.input_size,
        pc_cfg.input_size,
        device=device,
    )
    base_gt = torch.rand(
        16,
        1,
        pc_cfg.output_size,
        pc_cfg.output_size,
        device=device,
    )
    base_optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.decoder.parameters()
            if parameter.requires_grad
        ),
        lr=1.0e-4,
    )
    reset_peak_memory()
    for _ in range(2):
        base_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            base_outputs, base_aux = model(
                base_images,
                memory=memory,
                pc_mode="full",
                epoch=13,
                return_aux=True,
            )
            base_loss, _ = pc_hbm_labeled_loss(
                base_outputs,
                base_aux,
                base_gt,
                13,
                pc_cfg,
                pc_mode="full",
                training_design="two_stage",
            )
        assert_finite_loss(base_loss, "Base loss")
        base_loss.backward()
        assert_finite_gradients(model.decoder)
        assert_all_finite_gradients(model.decoder.pc_hbm.pair_verifier)
        assert_dino_frozen(model)
        base_optimizer.step()
    base_peak = peak_memory()
    pc_aux = base_aux["pc_hbm"]
    query_ratio = float(pc_aux["query_mask_map"].detach().float().mean())
    candidate_valid_ratio = float(
        pc_aux["retrieval_valid"].detach().float().mean()
    )
    pair_valid_ratio = float(
        pc_aux["query_valid"].detach().float().mean()
    )
    teacher_pc_before = module_fingerprint(model.decoder.pc_hbm)

    # Teacher pseudo: physical batch 32, terminal injection scale.
    model.eval()
    teacher_images = torch.randn(
        32,
        3,
        pc_cfg.input_size,
        pc_cfg.input_size,
        device=device,
    )
    reset_peak_memory()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        features = model.extract_features(teacher_images)
        _, teacher_aux = model.decoder(
            features,
            memory=memory,
            pc_mode="teacher_pseudo",
            epoch=30,
            return_aux=True,
        )
    teacher_peak = peak_memory()
    pseudo = prepare_pseudo_targets(teacher_aux)
    if set(pseudo) != {"p_soft", "confidence", "p3_corr"}:
        raise AssertionError("Teacher pseudo target contains non-Lite fields")

    student = Decoder(
        in_dim=pc_cfg.encoder_dim,
        out_dim=pc_cfg.decoder_dim,
        pc_cfg=None,
    ).to(device).train()
    student.load_state_dict(
        {
            name: value
            for name, value in model.decoder.state_dict().items()
            if not name.startswith("pc_hbm.")
        },
        strict=True,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-4)
    labeled_gt = torch.rand(
        32,
        1,
        pc_cfg.output_size,
        pc_cfg.output_size,
        device=device,
    )

    # Raw Student labeled: physical batch 32, two AMP optimizer steps.
    reset_peak_memory()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            labeled_outputs, _ = student(
                features,
                pc_mode="off",
                return_aux=True,
            )
            labeled_loss = base_structure_loss(
                labeled_outputs,
                labeled_gt,
            )
        assert_finite_loss(labeled_loss, "Student labeled loss")
        labeled_loss.backward()
        assert_finite_gradients(student)
        optimizer.step()
    student_labeled_peak = peak_memory()

    # Raw Student unlabeled: physical batch 32, two AMP optimizer steps.
    reset_peak_memory()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            unlabeled_outputs, unlabeled_aux = student(
                features,
                pc_mode="off",
                return_aux=True,
            )
            unlabeled_loss, _ = pc_unlabeled_loss(
                unlabeled_outputs,
                unlabeled_aux,
                pseudo["p_soft"],
                pseudo["confidence"],
                pc_cfg,
                teacher_features={"p3_corr": pseudo["p3_corr"]},
            )
        assert_finite_loss(unlabeled_loss, "Student unlabeled loss")
        unlabeled_loss.backward()
        assert_finite_gradients(student)
        optimizer.step()
    student_unlabeled_peak = peak_memory()
    assert_dino_frozen(model)
    teacher_pc_after = module_fingerprint(model.decoder.pc_hbm)
    if teacher_pc_after != teacher_pc_before:
        raise AssertionError(
            "Teacher PC parameters changed during pseudo/Student steps"
        )

    baseline_fingerprint = state_dict_fingerprint(student.state_dict())
    metadata = build_artifact_metadata(
        training_design="teacher_only",
        artifact_role="student_raw",
        labeled_split_fingerprint=split_fingerprint,
        baseline_fingerprint=baseline_fingerprint,
        pc_frozen=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "student_raw.pth"
        payload = save_decoder_checkpoint(
            path,
            student,
            pc_cfg,
            1,
            artifact_meta=metadata,
        )
        if any(
            name.startswith("pc_hbm.") for name in payload["decoder"]
        ):
            raise AssertionError("student_raw.pth contains PC tensors")

    print(
        json.dumps(
            {
                "seed": args.seed,
                "labeled_split_fingerprint": split_fingerprint,
                "labeled_count": len(loader.dataset),
                "base_batch": 16,
                "teacher_batch": 32,
                "student_batch": 32,
                "base_loss": float(base_loss.detach()),
                "student_labeled_loss": float(labeled_loss.detach()),
                "student_unlabeled_loss": float(unlabeled_loss.detach()),
                "query_ratio": query_ratio,
                "candidate_valid_ratio": candidate_valid_ratio,
                "pair_valid_ratio": pair_valid_ratio,
                "peak_memory": {
                    "base_full": base_peak,
                    "teacher_pseudo": teacher_peak,
                    "student_labeled": student_labeled_peak,
                    "student_unlabeled": student_unlabeled_peak,
                },
                "teacher_pc_fingerprint_unchanged": True,
                "student_raw_contains_pc_keys": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
