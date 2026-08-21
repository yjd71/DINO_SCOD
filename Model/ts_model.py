from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from Model.decoder import Decoder
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import load_decoder_compatible


class TSModel(nn.Module):
    """Student-owned PC-HBM model with an EMA segmentation Teacher."""

    VALID_TRAINING_DESIGNS = frozenset({"student_joint"})

    def __init__(
        self,
        base_student_pth=None,
        pc_cfg=None,
        training_design="student_joint",
        allow_student_joint_defaults=False,
    ):
        super().__init__()
        if training_design != "student_joint":
            raise ValueError("PC-HBM-Lite TS supports only student_joint")
        if pc_cfg is None:
            raise ValueError("pc_cfg is required for student_joint TS")

        self.training_design = "student_joint"
        self.pc_cfg = pc_cfg
        self.dino = torch.hub.load(
            "./dinov2", "dinov2_vitb14", source="local", pretrained=False
        )
        self.dino.load_state_dict(
            torch.load(
                "./weight/dinov2_vitb14_pretrain.pth",
                map_location="cpu",
                weights_only=False,
            )
        )
        self.dino.requires_grad_(False).eval()

        decoder_kwargs = {
            "in_dim": int(pc_cfg.encoder_dim),
            "out_dim": int(pc_cfg.decoder_dim),
        }
        self.student = Decoder(pc_cfg=pc_cfg, **decoder_kwargs)
        if base_student_pth is not None:
            self.load_base_student(
                base_student_pth,
                allow_student_joint_defaults=allow_student_joint_defaults,
            )
        self.teacher = Decoder(pc_cfg=None, **decoder_kwargs)
        self._initialize_teacher_from_student()
        self.teacher.requires_grad_(False).eval()
        self.pc_hbm_readonly = deepcopy(self.student.pc_hbm)
        self.pc_hbm_readonly.requires_grad_(False).eval()
        self.sync_readonly_pc()

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.teacher.eval()
        self.pc_hbm_readonly.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x):
        expected_input = (
            3,
            int(self.pc_cfg.input_size),
            int(self.pc_cfg.input_size),
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected_input:
            raise ValueError(
                f"DINO input must be [B,{expected_input[0]},"
                f"{expected_input[1]},{expected_input[2]}], got {tuple(x.shape)}"
            )
        features = self.dino.get_intermediate_layers(
            x=x,
            n=list(self.pc_cfg.dino_layer_indices),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise RuntimeError("DINO must return exactly four feature layers")
        expected = (
            x.shape[0],
            int(self.pc_cfg.token_size) ** 2,
            int(self.pc_cfg.encoder_dim),
        )
        for index, feature in enumerate(features):
            if not torch.is_tensor(feature) or feature.shape != expected:
                raise RuntimeError(
                    f"DINO layer {index} must be {expected}, "
                    f"got {getattr(feature, 'shape', None)}"
                )
        return features

    @torch.inference_mode()
    def teacher_pseudo(self, features, memory, epoch):
        _, aux = self.teacher(
            features,
            memory=memory,
            pc_mode="teacher_pseudo",
            epoch=epoch,
            return_aux=True,
            pc_engine_override=self.pc_hbm_readonly,
        )
        if aux.get("pc_active") is not True:
            raise RuntimeError("Teacher PC-HBM-Lite path is inactive")
        if aux.get("pc_engine_source") != "external_readonly":
            raise RuntimeError("Teacher must use the readonly external PC-HBM engine")
        if not torch.is_tensor(aux.get("p_final")):
            raise RuntimeError("Teacher did not return p_final")
        return aux

    def student_joint(
        self,
        *,
        labeled_features,
        unlabeled_features,
        memory,
        epoch,
        labeled_image_ids,
    ):
        labeled = self.student(
            labeled_features,
            memory=memory,
            pc_mode="full",
            epoch=epoch,
            return_aux=True,
            query_image_ids=labeled_image_ids,
        )
        unlabeled = self.student(
            unlabeled_features,
            memory=memory,
            pc_mode="full",
            epoch=epoch,
            return_aux=True,
            pc_engine_override=self.pc_hbm_readonly,
        )
        return {"labeled": labeled, "unlabeled": unlabeled}

    def forward(
        self,
        *,
        branch,
        labeled_features=None,
        unlabeled_features=None,
        memory=None,
        epoch=None,
        labeled_image_ids=None,
    ):
        if branch != "student_joint":
            raise ValueError(f"Unsupported TS forward branch: {branch!r}")
        if labeled_features is None or unlabeled_features is None:
            raise ValueError("student_joint requires labeled and unlabeled DINO features")
        if memory is None:
            raise ValueError("student_joint requires finalized labeled Student memory")
        if labeled_image_ids is None:
            raise ValueError("student_joint requires stable labeled image IDs")
        return self.student_joint(
            labeled_features=labeled_features,
            unlabeled_features=unlabeled_features,
            memory=memory,
            epoch=epoch,
            labeled_image_ids=labeled_image_ids,
        )

    def inference(self, x, memory=None, epoch=None, disable_pc=False):
        features = self.extract_features(x)
        if disable_pc:
            return self.student(features, pc_mode="off")[3]
        if memory is None:
            raise RuntimeError("Student inference requires finalized PC-HBM memory")
        _, aux = self.student(
            features,
            memory=memory,
            pc_mode="full",
            epoch=epoch,
            return_aux=True,
        )
        return aux["z_final"]

    def load_base_student(self, path, *, allow_student_joint_defaults=False):
        load_decoder_compatible(
            self.student,
            path,
            require_pc_complete=True,
            expected_pc_cfg=self.pc_cfg,
            allow_student_joint_defaults=allow_student_joint_defaults,
        )

    def _initialize_teacher_from_student(self):
        teacher_state = {
            name: value
            for name, value in self.student.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
        self.teacher.load_state_dict(teacher_state, strict=True)

    @torch.no_grad()
    def update_teacher(self, momentum=0.995):
        update_ema_module(
            self.student,
            self.teacher,
            momentum=momentum,
            shared_only=True,
            exclude_prefixes=("pc_hbm.",),
        )
        self.teacher.requires_grad_(False).eval()

    @torch.no_grad()
    def sync_readonly_pc(self):
        self.pc_hbm_readonly.load_state_dict(
            self.student.pc_hbm.state_dict(), strict=True
        )
        self.pc_hbm_readonly.requires_grad_(False).eval()
        self.assert_readonly_pc_synced()

    @torch.no_grad()
    def assert_readonly_pc_synced(self):
        trainable = self.student.pc_hbm.state_dict()
        readonly = self.pc_hbm_readonly.state_dict()
        if trainable.keys() != readonly.keys():
            raise RuntimeError("Trainable and readonly PC-HBM state schemas differ")
        mismatched = [
            name for name in trainable if not torch.equal(trainable[name], readonly[name])
        ]
        if mismatched:
            raise RuntimeError(
                f"Readonly PC-HBM is not synchronized: {mismatched[:5]}"
            )
        return True

    @torch.no_grad()
    def readonly_pc_sync_max_abs_diff(self) -> float:
        maximum = 0.0
        readonly = self.pc_hbm_readonly.state_dict()
        for name, value in self.student.pc_hbm.state_dict().items():
            if value.numel() == 0:
                continue
            difference = (value.detach().float() - readonly[name].detach().float()).abs()
            maximum = max(maximum, float(difference.max().item()))
        return maximum


__all__ = ["TSModel"]
