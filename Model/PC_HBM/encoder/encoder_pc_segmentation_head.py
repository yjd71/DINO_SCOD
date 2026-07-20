"""Role-gated execution head for encoder-side PC-HBM v3.

The head is the single policy boundary around the trainable Encoder Adapter,
the permanently non-PC BGFBR Decoder, and the training-only pseudo refiner.
It deliberately makes the five supported execution roles explicit so that a
Student core or formal inference call cannot accidentally execute mixture
refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .contracts import DinoFeatureBundle
from .encoder_pc_adapter import EncoderPCHBMAdapter, EncoderPCStageFlags
from .teacher_pseudo_refiner import TeacherPseudoLabelRefiner


ENCODER_PC_SEGMENTATION_ROLES = frozenset(
    {
        "labeled_core",
        "labeled_refiner",
        "teacher_pseudo",
        "student_core",
        "inference",
    }
)


@dataclass(frozen=True)
class EncoderPCCoreResult:
    """Core output shared with the detached labeled-refiner role."""

    outputs: tuple[torch.Tensor, ...]
    aux: Mapping[str, Any]

    @property
    def z_core(self) -> torch.Tensor:
        if len(self.outputs) != 5:
            raise RuntimeError("BGFBR core must return exactly five outputs.")
        return self.outputs[3]


def _full_stage() -> EncoderPCStageFlags:
    return EncoderPCStageFlags(
        enable_f4_f3=True,
        f4_f3_progress=1.0,
        enable_f2_f1=True,
        f2_f1_progress=1.0,
        require_same_image_positive=False,
    )


class EncoderPCSegmentationHead(nn.Module):
    """Dispatch the strict encoder-PC execution roles.

    ``labeled_refiner`` accepts a previously computed
    :class:`EncoderPCCoreResult` and executes only the pseudo refiner.
    ``teacher_pseudo`` executes core then refiner.  ``student_core`` and
    ``inference`` never execute the refiner; inference returns ``outputs[3]``.
    Every Decoder call is hard-coded to ``pc_mode='off'`` and ``memory=None``.
    """

    def __init__(
        self,
        adapter: EncoderPCHBMAdapter,
        decoder: nn.Module,
        pseudo_refiner: TeacherPseudoLabelRefiner,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, EncoderPCHBMAdapter):
            raise TypeError("adapter must be EncoderPCHBMAdapter")
        if not isinstance(pseudo_refiner, TeacherPseudoLabelRefiner):
            raise TypeError("pseudo_refiner must be TeacherPseudoLabelRefiner")
        if getattr(decoder, "pc_hbm", None) is not None:
            raise RuntimeError("encoder-PC requires a Decoder without PC-HBM")
        if any(name.startswith("pc_hbm.") for name in decoder.state_dict()):
            raise RuntimeError("encoder-PC Decoder state must not contain pc_hbm.*")
        self.adapter = adapter
        self.decoder = decoder
        self.pseudo_refiner = pseudo_refiner

    def _run_core(
        self,
        bundle: DinoFeatureBundle,
        image_rgb: torch.Tensor,
        *,
        memory: Any,
        mode: str,
        stage: EncoderPCStageFlags | None,
        epoch: int | None,
        query_image_ids: Sequence[object] | None,
        allow_memory_fallback: bool,
        return_aux: bool,
    ) -> EncoderPCCoreResult:
        adapter_output = self.adapter(
            bundle,
            memory=memory,
            mode=mode,
            stage=stage,
            query_image_ids=query_image_ids,
            allow_memory_fallback=allow_memory_fallback,
        )
        decoder_result = self.decoder(
            features=adapter_output.features,
            image_rgb=image_rgb,
            memory=None,
            pc_mode="off",
            epoch=epoch,
            return_aux=return_aux,
            query_image_ids=None,
        )
        if return_aux:
            if not isinstance(decoder_result, (tuple, list)) or len(decoder_result) != 2:
                raise RuntimeError("BGFBR Decoder must return (outputs, aux).")
            outputs, decoder_aux = decoder_result
            combined_aux = dict(decoder_aux or {})
            combined_aux["encoder_pc_hbm"] = adapter_output.aux
            combined_aux["pc_active"] = bool(
                adapter_output.aux.get("pc_active", False)
            )
            combined_aux["pc_mode"] = str(adapter_output.aux.get("mode", mode))
        else:
            outputs = decoder_result
            combined_aux = {}
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise RuntimeError("BGFBR Decoder must return five core outputs.")
        return EncoderPCCoreResult(tuple(outputs), combined_aux)

    def _run_refiner(
        self,
        core: EncoderPCCoreResult,
        *,
        epoch: int | None,
        ts_continuation: bool,
    ) -> Mapping[str, torch.Tensor]:
        features = core.aux.get("features")
        if not isinstance(features, Mapping) or not torch.is_tensor(
            features.get("p1")
        ):
            raise KeyError("Decoder aux must provide features['p1'] for refinement")
        encoder_aux = core.aux.get("encoder_pc_hbm")
        if not isinstance(encoder_aux, Mapping):
            raise KeyError("core aux must provide encoder_pc_hbm evidence")
        return self.pseudo_refiner(
            core.z_core.detach(),
            features["p1"].detach(),
            encoder_aux,
            epoch=epoch,
            ts_continuation=ts_continuation,
        )

    def forward(
        self,
        *,
        role: str,
        bundle: DinoFeatureBundle | None = None,
        image_rgb: torch.Tensor | None = None,
        core_result: EncoderPCCoreResult | None = None,
        memory: Any = None,
        mode: str = "full",
        stage: EncoderPCStageFlags | None = None,
        epoch: int | None = None,
        query_image_ids: Sequence[object] | None = None,
        allow_memory_fallback: bool = False,
        return_aux: bool = True,
    ) -> Any:
        role = str(role)
        if role not in ENCODER_PC_SEGMENTATION_ROLES:
            raise ValueError(
                f"unsupported encoder-PC role {role!r}; "
                f"expected {sorted(ENCODER_PC_SEGMENTATION_ROLES)}"
            )
        if role == "labeled_refiner":
            if not isinstance(core_result, EncoderPCCoreResult):
                raise TypeError(
                    "labeled_refiner requires a prior EncoderPCCoreResult"
                )
            return self._run_refiner(
                core_result, epoch=epoch, ts_continuation=False
            )

        if bundle is None or image_rgb is None:
            raise ValueError(f"role={role!r} requires bundle and image_rgb")
        if role in {"teacher_pseudo", "student_core", "inference"}:
            mode = "full"
            stage = _full_stage() if stage is None else stage
        needs_aux = return_aux or role == "teacher_pseudo"
        core = self._run_core(
            bundle,
            image_rgb,
            memory=memory,
            mode=mode,
            stage=stage,
            epoch=epoch,
            query_image_ids=query_image_ids,
            allow_memory_fallback=allow_memory_fallback,
            return_aux=needs_aux,
        )
        if role == "inference":
            return core.z_core
        if role == "teacher_pseudo":
            refined = self._run_refiner(
                core, epoch=epoch, ts_continuation=True
            )
            return {
                "outputs": core.outputs,
                "aux": core.aux,
                "z_core": core.z_core,
                "pseudo_refiner": refined,
            }
        return core


__all__ = [
    "ENCODER_PC_SEGMENTATION_ROLES",
    "EncoderPCCoreResult",
    "EncoderPCSegmentationHead",
]
