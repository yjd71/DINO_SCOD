"""Teacher-only semi-supervised model for PC-HBM-Lite."""

from __future__ import annotations

import torch
import torch.nn as nn

from Model.decoder import Decoder
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import load_decoder_compatible


class TSModel(nn.Module):
    """Frozen Lite Teacher plus a raw legacy Decoder Student."""

    VALID_TRAINING_DESIGNS = frozenset({"teacher_only"})

    def __init__(
        self,
        teacher_pth=None,
        student_pth=None,
        pc_cfg=None,
        training_design="teacher_only",
    ):
        super().__init__()
        if training_design != "teacher_only":
            raise ValueError("PC-HBM-Lite TS supports only teacher_only")
        if pc_cfg is None:
            raise ValueError("pc_cfg is required for the Lite Teacher")

        self.training_design = "teacher_only"
        self.pc_cfg = pc_cfg
        self.dino = torch.hub.load(
            "./dinov2",
            "dinov2_vitb14",
            source="local",
            pretrained=False,
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
        self.teacher = Decoder(pc_cfg=pc_cfg, **decoder_kwargs)
        self.student = Decoder(pc_cfg=None, **decoder_kwargs)
        self.load_teacher(teacher_pth)
        if student_pth is None:
            self._initialize_raw_student_from_teacher()
        else:
            self.load_student(student_pth)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.teacher.eval()
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
        )
        if aux.get("pc_active") is not True:
            raise RuntimeError(
                "Teacher PC-HBM-Lite path is inactive: "
                f"{aux.get('fallback_reason')}"
            )
        if not torch.is_tensor(aux.get("p_final")):
            raise RuntimeError("Teacher did not return p_final")
        distill = aux.get("distill_features")
        if not isinstance(distill, dict) or not torch.is_tensor(
            distill.get("p3_corr")
        ):
            raise RuntimeError("Teacher did not return corrected P3 features")
        return aux

    def student_labeled(self, features):
        return self.student(features, pc_mode="off", return_aux=True)

    def student_unlabeled(self, features):
        return self.student(features, pc_mode="off", return_aux=True)

    def forward(
        self,
        *,
        branch,
        features,
    ):
        if features is None:
            raise ValueError(
                "Precomputed DINO features are required for Student dispatch"
            )
        if branch == "student_labeled":
            return self.student_labeled(features)
        if branch == "student_unlabeled":
            return self.student_unlabeled(features)
        raise ValueError(f"Unsupported TS forward branch: {branch!r}")

    def inference(self, x, memory=None, epoch=None):
        features = self.extract_features(x)
        return self.student(features, pc_mode="off")[3]

    def load_teacher(self, path):
        if path is None:
            raise ValueError("Teacher checkpoint path is required")
        load_decoder_compatible(
            self.teacher,
            path,
            require_pc_complete=True,
            expected_pc_cfg=self.pc_cfg,
        )
        self.teacher.requires_grad_(False).eval()

    def load_student(self, path):
        if path is None:
            raise ValueError("Student checkpoint path is required")
        load_decoder_compatible(
            self.student,
            path,
            require_pc_complete=False,
        )

    def _initialize_raw_student_from_teacher(self):
        raw_state = {
            name: value
            for name, value in self.teacher.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
        self.student.load_state_dict(raw_state, strict=True)

    @torch.no_grad()
    def update_teacher(self, momentum=0.995):
        update_ema_module(
            self.student,
            self.teacher,
            momentum=momentum,
            shared_only=True,
            exclude_prefixes=("pc_hbm.",),
        )
