"""DualUCOD-style BGFBR decoder integrated with the DINO PC-HBM engine."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BGFBR import (
    BGFBRStage,
    DinoGlobalPerceptionModule,
    F4Adapter,
    FinalFusion,
    GradientBoundaryEnhancement,
)
from .legacy_decoder import tokens_to_map


class BGFBRDecoder(nn.Module):
    """Four-stage foreground/background decoder with optional hierarchical PC."""

    decoder_arch = "bgfbr_pc_v1"
    decoder_architecture = "bgfbr_pc_v1"
    decoder_contract_version = 1
    VALID_PC_MODES = {"off", "parent_only", "full", "teacher_pseudo", "student_core"}

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 128,
        heads: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        pc_cfg: Any | None = None,
        attach_pc: bool = True,
    ) -> None:
        super().__init__()
        del heads, hidden_dim, dropout
        if int(out_dim) != 128:
            raise ValueError("bgfbr_pc_v1 requires decoder out_dim=128")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pc_cfg = pc_cfg
        self.token_size = int(getattr(pc_cfg, "token_size", 28))
        self.output_size = int(getattr(pc_cfg, "output_size", 98))
        self.input_size = int(getattr(pc_cfg, "input_size", 392))
        self.use_gbe = bool(
            getattr(pc_cfg, "bgfbr_use_gbe", getattr(pc_cfg, "use_gbe", True))
        )
        self.use_pc_boundary_context = bool(
            getattr(
                pc_cfg,
                "bgfbr_use_pc_boundary_context",
                getattr(pc_cfg, "use_pc_boundary_context", True),
            )
        )

        # Keep projector names and shapes stable for explicit legacy warm-start.
        self.linear_1 = self._projector(self.in_dim, self.out_dim)
        self.linear_2 = self._projector(self.in_dim, self.out_dim)
        self.linear_3 = self._projector(self.in_dim, self.out_dim)
        self.linear_4 = self._projector(self.in_dim, self.out_dim)

        norm = str(
            getattr(
                pc_cfg,
                "bgfbr_norm",
                "sync_bn" if bool(getattr(pc_cfg, "sync_bn", False)) else "bn",
            )
        )
        reduction = int(
            getattr(pc_cfg, "bgfbr_rcab_reduction", getattr(pc_cfg, "rcab_reduction", 16))
        )
        use_ode = bool(
            getattr(pc_cfg, "bgfbr_use_ode", getattr(pc_cfg, "use_ode", True))
        )
        use_rcab = bool(
            getattr(pc_cfg, "bgfbr_use_rcab", getattr(pc_cfg, "use_rcab", True))
        )
        dilations = tuple(
            getattr(
                pc_cfg,
                "bgfbr_gpm_dilations",
                getattr(pc_cfg, "gpm_dilations", (1, 3, 5)),
            )
        )
        bottleneck = int(getattr(pc_cfg, "bgfbr_adapter_bottleneck", 32))

        self.gbe = GradientBoundaryEnhancement(
            token_size=(self.token_size, self.token_size),
            output_size=(self.output_size, self.output_size),
            eps=float(getattr(pc_cfg, "gbe_eps", 1.0e-6)),
        )
        self.f4_adapter = F4Adapter(
            channels=self.out_dim, bottleneck=bottleneck, gamma=1.0
        )
        self.gpm = DinoGlobalPerceptionModule(
            channels=self.out_dim, dilations=dilations, norm=norm
        )
        stage_kwargs = {
            "channels": self.out_dim,
            "reduction": reduction,
            "norm": norm,
            "use_ode": use_ode,
            "use_rcab": use_rcab,
        }
        self.stage4 = BGFBRStage(**stage_kwargs)
        self.stage3 = BGFBRStage(**stage_kwargs)
        self.stage2 = BGFBRStage(**stage_kwargs)
        self.stage1 = BGFBRStage(**stage_kwargs)
        self.final_fusion = FinalFusion(
            channels=self.out_dim,
            output_size=(self.output_size, self.output_size),
            norm=norm,
        )

        if (
            bool(attach_pc)
            and pc_cfg is not None
            and bool(getattr(pc_cfg, "enabled", False))
        ):
            from Model.PC_HBM.dino_engine import DinoPCHBMEngine

            self.pc_hbm = DinoPCHBMEngine(pc_cfg)
        else:
            self.pc_hbm = None

    @staticmethod
    def _projector(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def _project_features(self, features) -> tuple[torch.Tensor, ...]:
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError("BGFBR decoder expects exactly four DINO feature tensors")
        maps: list[torch.Tensor] = []
        batch_size = None
        for index, (feature, projector) in enumerate(
            zip(features, (self.linear_1, self.linear_2, self.linear_3, self.linear_4)),
            start=1,
        ):
            if feature.ndim != 3 or feature.size(-1) != self.in_dim:
                raise ValueError(
                    f"DINO feature {index} must be [B,N,{self.in_dim}], got "
                    f"{tuple(feature.shape)}"
                )
            if batch_size is None:
                batch_size = feature.size(0)
            elif feature.size(0) != batch_size:
                raise ValueError("DINO feature batch dimensions differ")
            side = math.isqrt(int(feature.size(1)))
            if side * side != feature.size(1) or side != self.token_size:
                raise ValueError(
                    f"DINO feature {index} must form the fixed "
                    f"{self.token_size}x{self.token_size} grid"
                )
            maps.append(tokens_to_map(projector(feature), side, side))
        return tuple(maps)

    def _validate_image_rgb(
        self, image_rgb: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor:
        if image_rgb is None:
            raise ValueError("bgfbr_pc_v1 requires image_rgb; zero-edge fallback is forbidden")
        if image_rgb.ndim != 4 or image_rgb.size(1) != 3:
            raise ValueError(
                f"image_rgb must be [B,3,H,W], got {tuple(image_rgb.shape)}"
            )
        if image_rgb.size(0) != batch_size:
            raise ValueError("image_rgb and DINO features have different batch sizes")
        if tuple(image_rgb.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"image_rgb must use the fixed {self.input_size}x{self.input_size} input"
            )
        if not image_rgb.is_floating_point():
            raise TypeError("image_rgb must be a floating-point tensor")
        if not torch.isfinite(image_rgb).all():
            raise ValueError("image_rgb contains NaN or Inf")
        lower = float(image_rgb.detach().amin())
        upper = float(image_rgb.detach().amax())
        if lower < -1.0e-3 or upper > 1.0 + 1.0e-3:
            raise ValueError(
                f"image_rgb must be in [0,1] within tolerance, got [{lower}, {upper}]"
            )
        return image_rgb.clamp(0.0, 1.0)

    def _run_bgfbr(self, features, image_rgb: torch.Tensor) -> dict[str, Any]:
        f1, f2, f3, f4 = self._project_features(features)
        image_rgb = self._validate_image_rgb(image_rgb, f1.size(0))
        edges = self.gbe(image_rgb, decoder_dtype=f1.dtype)
        if not self.use_gbe:
            edges = {name: torch.zeros_like(value) for name, value in edges.items()}
        edge_28 = edges["edge_28"]
        edge_98 = edges["edge_98"]
        f4_adapted = self.f4_adapter(f4)
        cam_feat, cam_logit = self.gpm(f4_adapted)
        stages = (
            self.stage4(f4_adapted, cam_feat, cam_logit, edge_28),
            self.stage3(f3, cam_feat, cam_logit, edge_28),
            self.stage2(f2, cam_feat, cam_logit, edge_28),
            self.stage1(f1, cam_feat, cam_logit, edge_28),
        )
        output_hw = (self.output_size, self.output_size)
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4_adapted,
            "stages": stages,
            "cam_feat": cam_feat,
            "cam_logit": cam_logit,
            "global_logit": F.interpolate(
                cam_logit, size=output_hw, mode="bilinear", align_corners=False
            ),
            "edge_28": edge_28,
            "edge_98": edge_98,
            "edge_full": edges["edge_full"],
        }

    @torch.no_grad()
    def forward_memory_features(self, features, image_rgb=None):
        state = self._run_bgfbr(features, image_rgb)
        _, stage3, stage2, _ = state["stages"]
        return {
            "x3": state["f3"],
            "p3": stage3.fg_feature,
            "p2": stage2.fg_feature + state["f2"],
            "m3": stage3.fg_logit,
            "m2": stage2.fg_logit,
        }

    def _memory_fallback_reason(self, memory) -> str | None:
        if memory is None:
            return "memory_missing"
        if not hasattr(memory, "is_ready") or not memory.is_ready():
            return "memory_not_ready"
        if self.pc_cfg is not None and hasattr(memory, "validate_compat"):
            try:
                compatible = memory.validate_compat(self.pc_cfg.expected_memory_meta())
            except (KeyError, RuntimeError, ValueError) as error:
                return f"memory_incompatible:{error}"
            if not compatible:
                reason = getattr(compatible, "reason", None)
                return str(reason or "memory_incompatible")
        return None

    def _boundary_context(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.use_pc_boundary_context else torch.zeros_like(value)

    def _finalize_base(
        self, state: dict[str, Any], p2_refined: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage1 = state["stages"][3]
        return self.final_fusion(
            stage1.fg_feature,
            p2_refined,
            state["cam_feat"],
            state["edge_28"],
            state["edge_98"],
        )

    @staticmethod
    def _upsample(logit: torch.Tensor, output_size: int) -> torch.Tensor:
        return F.interpolate(
            logit,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )

    def _assemble(
        self,
        *,
        state: dict[str, Any],
        pc_mode: str,
        fallback_reason: str | None,
        pc_active: bool,
        pc_aux,
        bridge_aux,
        p2_aux,
        p1_aux,
        mix_aux,
        p3_corr: torch.Tensor,
        p2_refined: torch.Tensor,
        p1_98: torch.Tensor,
        z_main: torch.Tensor,
        m2_native: torch.Tensor,
        z_final: torch.Tensor | None,
        p_final: torch.Tensor | None,
        return_aux: bool,
    ):
        stage4, stage3, stage2, stage1 = state["stages"]
        m4 = self._upsample(stage4.fg_logit, self.output_size)
        m3 = self._upsample(stage3.fg_logit, self.output_size)
        m2 = self._upsample(m2_native, self.output_size)
        outputs = (m4, m3, m2, z_main, state["global_logit"])
        if not return_aux:
            return outputs, None

        fg1_98 = self._upsample(stage1.fg_logit, self.output_size)
        bg_output = tuple(
            self._upsample(stage.bg_logit, self.output_size)
            for stage in (stage4, stage3, stage2, stage1)
        )
        distill_features = None
        if pc_mode == "teacher_pseudo" and pc_active:
            distill_features = {
                "p3_corr": p3_corr,
                "p2_refined": p2_refined,
                "p1": {
                    key: p1_aux[key]
                    for key in (
                        "B1",
                        "G1_raw_map",
                        "R1_map",
                        "O1_map",
                        "R_sup_map",
                        "valid1_map",
                    )
                },
            }
        aux = {
            "decoder_architecture": self.decoder_arch,
            "m4": m4,
            "m3": m3,
            "m2": m2,
            "global_logit": state["global_logit"],
            "z_main": z_main,
            "z_nomix": z_main,
            "z_final": z_final,
            "p_final": p_final,
            "pc_active": bool(pc_active),
            "fallback_reason": fallback_reason,
            "pc_hbm": pc_aux,
            "pc_bridge": bridge_aux,
            "p2_bra": p2_aux,
            "p1_pra": p1_aux,
            "mixture": mix_aux,
            "mixture_skipped": mix_aux is None,
            "forward_mode": pc_mode,
            "distill_features": distill_features,
            "features": {
                "x3": state["f3"],
                "p3": stage3.fg_feature,
                "p3_corr": p3_corr,
                "p2": p2_refined,
                "p2_pre": stage2.fg_feature,
                "p2_refined": p2_refined,
                "p1": p1_98,
            },
            "bgfbr": {
                "edge_28": state["edge_28"],
                "edge_98": state["edge_98"],
                "edge_full": state["edge_full"],
                "cam_feat": state["cam_feat"],
                "cam_logit": state["cam_logit"],
                "fg_native": tuple(
                    stage.fg_logit for stage in (stage4, stage3, stage2, stage1)
                ),
                "bg_native": tuple(
                    stage.bg_logit for stage in (stage4, stage3, stage2, stage1)
                ),
                "fg_output": (m4, m3, m2, fg1_98),
                "bg_output": bg_output,
                "dual_uncertainty": tuple(
                    stage.dual_uncertainty
                    for stage in (stage4, stage3, stage2, stage1)
                ),
            },
        }
        if pc_active and self.pc_hbm is not None:
            aux = self.pc_hbm.slim_aux(aux, mode=pc_mode)
        return outputs, aux

    def _forward_impl(
        self,
        features,
        image_rgb,
        memory,
        pc_mode: str,
        epoch,
        return_aux: bool,
        query_image_ids=None,
    ):
        state = self._run_bgfbr(features, image_rgb)
        _, stage3, stage2, stage1 = state["stages"]
        p3_base = stage3.fg_feature
        p2_pre = stage2.fg_feature
        m2_native = stage2.fg_logit
        p3_corr = p3_base
        p2_refined = p2_pre
        pc_aux = bridge_aux = p2_aux = p1_aux = mix_aux = None
        pc_active = False
        fallback_reason = None

        if pc_mode != "off":
            if self.pc_hbm is None:
                fallback_reason = "pc_hbm_not_attached"
            else:
                fallback_reason = self._memory_fallback_reason(memory)

        if pc_mode != "off" and fallback_reason is not None and self.training:
            raise RuntimeError(
                "PC-HBM training requires an attached, ready, compatible memory; "
                f"got {fallback_reason}."
            )

        if pc_mode != "off" and fallback_reason is None:
            pc_active = True
            edge28_context = self._boundary_context(state["edge_28"])
            dual3_context = self._boundary_context(stage3.dual_uncertainty)
            if pc_mode == "parent_only":
                pc_aux = self.pc_hbm.forward_parent_only(
                    x3=state["f3"],
                    p3=p3_base,
                    m3=stage3.fg_logit,
                    memory=memory,
                    query_image_ids=query_image_ids,
                    edge_context=edge28_context,
                    dual_uncertainty=dual3_context,
                )
            else:
                pc_aux = self.pc_hbm.forward_parent_child(
                    x3=state["f3"],
                    p3=p3_base,
                    child_map=p2_pre + state["f2"],
                    m3=stage3.fg_logit,
                    m2_pre=stage2.fg_logit,
                    memory=memory,
                    epoch=epoch,
                    query_image_ids=query_image_ids,
                    edge_context=edge28_context,
                    dual_uncertainty=dual3_context,
                )
                p3_corr = pc_aux["p3_corr"]
                if self.pc_hbm.p3_p2_bridge is None:
                    raise RuntimeError("BGFBR PC-HBM engine is missing P3-to-P2 bridge")
                bridge_aux = self.pc_hbm.p3_p2_bridge(
                    p2_pre=p2_pre,
                    p3_base=p3_base,
                    p3_corr=p3_corr,
                    m2_pre=stage2.fg_logit,
                    edge_28=state["edge_28"],
                    valid_mask=pc_aux["pc_maps"]["valid3_map"],
                )
                m2_pc = bridge_aux["m2_pc"]
                p2_aux = self.pc_hbm.forward_p2(
                    p2=bridge_aux["p2_pc"],
                    prob2=torch.sigmoid(m2_pc),
                    pc_maps=pc_aux["pc_maps"],
                    edge_context=self._boundary_context(state["edge_28"]),
                    dual_uncertainty=self._boundary_context(stage2.dual_uncertainty),
                )
                p2_refined = p2_aux["p2_refined"]

        _, p1_98, z_main = self._finalize_base(state, p2_refined)
        if pc_active and pc_mode in {"full", "teacher_pseudo", "student_core"}:
            p1_aux = self.pc_hbm.forward_p1(
                p1=p1_98,
                z_main=z_main,
                p2_aux=p2_aux,
                edge_context=self._boundary_context(state["edge_98"]),
                dual_uncertainty=self._boundary_context(stage1.dual_uncertainty),
            )

        if pc_active and pc_mode in {"full", "teacher_pseudo"}:
            mixture_context = torch.cat(
                [
                    state["edge_98"],
                    self._upsample(torch.sigmoid(stage1.bg_logit), self.output_size),
                ],
                dim=1,
            )
            mixture_context = self._boundary_context(mixture_context)
            mix_aux = self.pc_hbm.forward_mixture(
                z_main=z_main,
                p1_aux=p1_aux,
                pc_maps=pc_aux["pc_maps"],
                epoch=epoch,
                ts_continuation=pc_mode == "teacher_pseudo",
                extra_context=mixture_context,
            )
            z_final = mix_aux["z_final"]
            p_final = mix_aux["p_final"]
        elif pc_mode == "student_core" and pc_active:
            z_final = None
            p_final = None
        else:
            z_final = z_main
            p_final = torch.sigmoid(z_main)

        return self._assemble(
            state=state,
            pc_mode=pc_mode,
            fallback_reason=fallback_reason,
            pc_active=pc_active,
            pc_aux=pc_aux,
            bridge_aux=bridge_aux,
            p2_aux=p2_aux,
            p1_aux=p1_aux,
            mix_aux=mix_aux,
            p3_corr=p3_corr,
            p2_refined=p2_refined,
            p1_98=p1_98,
            z_main=z_main,
            m2_native=m2_native,
            z_final=z_final,
            p_final=p_final,
            return_aux=return_aux,
        )

    def forward(
        self,
        features,
        image_rgb,
        memory=None,
        pc_mode="off",
        epoch=None,
        return_aux=False,
        query_image_ids=None,
    ):
        if pc_mode not in self.VALID_PC_MODES:
            raise ValueError(
                f"Unsupported pc_mode={pc_mode!r}; expected one of "
                f"{sorted(self.VALID_PC_MODES)}"
            )
        outputs, aux = self._forward_impl(
            features,
            image_rgb,
            memory,
            pc_mode,
            epoch,
            return_aux,
            query_image_ids=query_image_ids,
        )
        return (outputs, aux) if return_aux else outputs


__all__ = ["BGFBRDecoder"]
