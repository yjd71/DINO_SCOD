"""Typed contracts shared by the encoder-side PC-HBM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


Tensor4 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class DinoFeatureBundle:
    """Four frozen DINOv2 patch-token levels and their CLS tokens.

    The bundle retains the DINO sequence representation.  Spatial reshaping is
    owned by the encoder adapter, while the unchanged BGFBR decoder continues to
    receive the original ``[B, N, C]`` token layout.
    """

    patch_tokens: Tensor4
    cls_tokens: Tensor4

    def validate(
        self,
        *,
        token_count: int = 784,
        encoder_dim: int = 768,
    ) -> "DinoFeatureBundle":
        if len(self.patch_tokens) != 4 or len(self.cls_tokens) != 4:
            raise ValueError("DINO feature bundle must contain exactly four levels.")
        if token_count <= 0 or encoder_dim <= 0:
            raise ValueError("token_count and encoder_dim must be positive.")

        batch_size = None
        reference_device = None
        reference_dtype = None
        for level, (patch, cls) in enumerate(
            zip(self.patch_tokens, self.cls_tokens), start=1
        ):
            if not torch.is_tensor(patch) or not torch.is_tensor(cls):
                raise TypeError(f"DINO level {level} must contain tensors.")
            if patch.ndim != 3:
                raise ValueError(
                    f"DINO patch level {level} must be [B,N,C], got {tuple(patch.shape)}."
                )
            if cls.ndim != 2:
                raise ValueError(
                    f"DINO CLS level {level} must be [B,C], got {tuple(cls.shape)}."
                )
            if patch.shape[1:] != (token_count, encoder_dim):
                raise ValueError(
                    f"DINO patch level {level} must be [B,{token_count},{encoder_dim}], "
                    f"got {tuple(patch.shape)}."
                )
            if cls.shape[1] != encoder_dim or cls.shape[0] != patch.shape[0]:
                raise ValueError(
                    f"DINO CLS level {level} is incompatible with patch tokens: "
                    f"patch={tuple(patch.shape)}, cls={tuple(cls.shape)}."
                )
            if not patch.is_floating_point() or not cls.is_floating_point():
                raise TypeError(f"DINO level {level} tensors must be floating point.")
            if patch.device != cls.device or patch.dtype != cls.dtype:
                raise ValueError(
                    f"DINO level {level} patch/CLS device and dtype must match."
                )

            if batch_size is None:
                batch_size = patch.shape[0]
                reference_device = patch.device
                reference_dtype = patch.dtype
            elif (
                patch.shape[0] != batch_size
                or patch.device != reference_device
                or patch.dtype != reference_dtype
            ):
                raise ValueError(
                    "All DINO feature levels must share batch size, device, and dtype."
                )
        return self


__all__ = ["DinoFeatureBundle", "Tensor4"]
