from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import train_ts_model_pseudo_pc_hbm
from Model.ts_model import TSModel


class TinyPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))

    def forward(self, value):
        return value @ self.weight


class TinyDecoder(nn.Module):
    def __init__(self, with_pc):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.pc_hbm = TinyPC() if with_pc else None

    def forward(
        self,
        features,
        *,
        memory=None,
        pc_mode="off",
        return_aux=False,
        pc_engine_override=None,
        **kwargs,
    ):
        value = self.backbone(features)
        engine = pc_engine_override if pc_engine_override is not None else self.pc_hbm
        if pc_mode != "off":
            if engine is None or memory is None:
                raise RuntimeError("active PC path requires engine and memory")
            value = value + engine(value)
        source = None
        if pc_mode != "off":
            source = (
                "external_readonly"
                if pc_engine_override is not None
                else "internal_trainable"
            )
        outputs = (value, value, value, value, value)
        aux = {
            "z_main": value,
            "z_final": value,
            "p_final": value.sigmoid(),
            "pc_active": pc_mode != "off",
            "forward_mode": pc_mode,
            "pc_engine_source": source,
        }
        return (outputs, aux) if return_aux else outputs


def make_model():
    model = TSModel.__new__(TSModel)
    nn.Module.__init__(model)
    model.training_design = "student_joint"
    model.student = TinyDecoder(with_pc=True)
    model.teacher = TinyDecoder(with_pc=False)
    model._initialize_teacher_from_student()
    model.teacher.requires_grad_(False).eval()
    model.pc_hbm_readonly = deepcopy(model.student.pc_hbm)
    model.pc_hbm_readonly.requires_grad_(False).eval()
    model.sync_readonly_pc()
    return model


def _pc_grad(model):
    return model.student.pc_hbm.weight.grad


def test_student_joint_modes_and_teacher_use_readonly_pc():
    model = make_model()
    features_l = torch.randn(3, 2)
    features_u = torch.randn(3, 2)
    joint = model(
        branch="student_joint",
        labeled_features=features_l,
        unlabeled_features=features_u,
        memory=object(),
        epoch=1,
        labeled_image_ids=["a", "b", "c"],
    )
    assert joint["labeled"][1]["pc_engine_source"] == "internal_trainable"
    assert joint["unlabeled"][1]["pc_engine_source"] == "external_readonly"
    teacher_aux = model.teacher_pseudo(features_u, object(), 1)
    assert teacher_aux["pc_engine_source"] == "external_readonly"


def test_unlabeled_graph_cannot_update_trainable_pc():
    model = make_model()
    joint = model(
        branch="student_joint",
        labeled_features=torch.randn(3, 2),
        unlabeled_features=torch.randn(3, 2),
        memory=object(),
        epoch=1,
        labeled_image_ids=["a", "b", "c"],
    )
    joint["unlabeled"][0][3].sum().backward()
    assert model.student.backbone.weight.grad is not None
    assert _pc_grad(model) is None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(
        parameter.grad is None for parameter in model.pc_hbm_readonly.parameters()
    )


def test_labeled_and_combined_pc_gradients_are_identical():
    model = make_model()
    labeled = torch.randn(3, 2)
    unlabeled = torch.randn(3, 2)

    joint = model(
        branch="student_joint",
        labeled_features=labeled,
        unlabeled_features=unlabeled,
        memory=object(),
        epoch=1,
        labeled_image_ids=["a", "b", "c"],
    )
    joint["labeled"][0][3].sum().backward()
    labeled_pc_grad = _pc_grad(model).detach().clone()
    assert torch.count_nonzero(labeled_pc_grad) > 0

    model.zero_grad(set_to_none=True)
    joint = model(
        branch="student_joint",
        labeled_features=labeled,
        unlabeled_features=unlabeled,
        memory=object(),
        epoch=1,
        labeled_image_ids=["a", "b", "c"],
    )
    (joint["labeled"][0][3].sum() + joint["unlabeled"][0][3].sum()).backward()
    torch.testing.assert_close(_pc_grad(model), labeled_pc_grad)


def test_readonly_pc_sync_is_exact_after_student_update():
    model = make_model()
    with torch.no_grad():
        model.student.pc_hbm.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="not synchronized"):
        model.assert_readonly_pc_synced()
    model.sync_readonly_pc()
    assert model.assert_readonly_pc_synced()
    assert model.readonly_pc_sync_max_abs_diff() == 0.0


def test_ts_cli_requires_exactly_one_initialization_source(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train_ts_model_pseudo_pc_hbm.py", "--base-student-checkpoint", "base.pth"],
    )
    args = train_ts_model_pseudo_pc_hbm.parse_args()
    assert args.base_student_checkpoint == "base.pth"
    assert args.resume is None

    monkeypatch.setattr(
        "sys.argv",
        ["train_ts_model_pseudo_pc_hbm.py", "--resume", "resume.pth"],
    )
    args = train_ts_model_pseudo_pc_hbm.parse_args()
    assert args.resume == "resume.pth"
