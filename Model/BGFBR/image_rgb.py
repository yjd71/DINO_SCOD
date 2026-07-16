"""ImageNet de-normalization and static RGB Sobel boundary extraction."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNetRGBAdapter(nn.Module):
    """Recover clamped ``[0, 1]`` RGB images from ImageNet-normalized input."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("ImageNetRGBAdapter requires exactly three mean/std values")
        if any(value <= 0 for value in std):
            raise ValueError("ImageNetRGBAdapter std values must be positive")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.ndim != 4 or normalized.shape[1] != 3:
            raise ValueError(
                "normalized input must have shape [B,3,H,W], "
                f"got {tuple(normalized.shape)}"
            )
        if not normalized.is_floating_point():
            raise TypeError("normalized input must be floating point")
        if not torch.isfinite(normalized).all():
            raise ValueError("normalized input contains NaN or Inf")

        mean = self.mean.to(device=normalized.device, dtype=normalized.dtype)
        std = self.std.to(device=normalized.device, dtype=normalized.dtype)
        return (normalized * std + mean).clamp_(0.0, 1.0)


class GradientBoundaryEnhancement(nn.Module):
    """Compute a deterministic, per-image normalized RGB Sobel edge map.

    Sobel convolution always runs in FP32 with replicate padding, including
    under an outer autocast context. Returned maps are converted to the decoder
    dtype only after normalization and resizing.
    """

    def __init__(
        self,
        token_size: tuple[int, int] = (28, 28),
        output_size: tuple[int, int] = (98, 98),
        eps: float = 1e-12,
        range_tolerance: float = 1e-3,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        if len(token_size) != 2 or min(token_size) <= 0:
            raise ValueError("token_size must contain two positive values")
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError("output_size must contain two positive values")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.token_size = tuple(int(v) for v in token_size)
        self.output_size = tuple(int(v) for v in output_size)
        self.eps = float(eps)
        self.range_tolerance = float(range_tolerance)

    @staticmethod
    def _autocast_disabled(device_type: str):
        if device_type in {"cpu", "cuda"}:
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _validate_rgb(self, image_rgb: torch.Tensor) -> None:
        if image_rgb.ndim != 4 or image_rgb.shape[1] != 3:
            raise ValueError(f"image_rgb must have shape [B,3,H,W], got {tuple(image_rgb.shape)}")
        if not image_rgb.is_floating_point():
            raise TypeError("image_rgb must be floating point")
        if not torch.isfinite(image_rgb).all():
            raise ValueError("image_rgb contains NaN or Inf")
        lower = -self.range_tolerance
        upper = 1.0 + self.range_tolerance
        if torch.any((image_rgb < lower) | (image_rgb > upper)):
            minimum = float(image_rgb.detach().amin())
            maximum = float(image_rgb.detach().amax())
            raise ValueError(
                f"image_rgb must be in [0,1] within tolerance {self.range_tolerance}; "
                f"observed [{minimum:.6g},{maximum:.6g}]"
            )

    def forward(
        self,
        image_rgb: torch.Tensor,
        decoder_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_rgb(image_rgb)
        output_dtype = image_rgb.dtype if decoder_dtype is None else decoder_dtype
        if not torch.empty((), dtype=output_dtype).is_floating_point():
            raise TypeError("decoder_dtype must be a floating-point dtype")

        with self._autocast_disabled(image_rgb.device.type):
            rgb_fp32 = image_rgb.float()
            padded = F.pad(rgb_fp32, (1, 1, 1, 1), mode="replicate")
            grad_x = F.conv2d(padded, self.sobel_x, groups=3)
            grad_y = F.conv2d(padded, self.sobel_y, groups=3)

            # Subtracting sqrt(eps) preserves the documented stabilizer while
            # making a spatially constant image exactly zero.
            magnitude = (torch.sqrt(grad_x.square() + grad_y.square() + self.eps) - self.eps**0.5)
            magnitude = magnitude.clamp_min_(0.0).mean(dim=1, keepdim=True)
            spatial_max = magnitude.amax(dim=(-2, -1), keepdim=True)
            # A constant non-binary FP32 value can leave a ~1e-7 convolution
            # cancellation residue. Do not amplify that numerical noise to one.
            zero_threshold = max(self.eps**0.5, torch.finfo(torch.float32).eps * 16.0)
            edge_full = torch.where(
                spatial_max > zero_threshold,
                magnitude / (spatial_max + self.eps),
                torch.zeros_like(magnitude),
            )
            edge_28 = F.interpolate(edge_full, size=self.token_size, mode="bilinear", align_corners=False)
            edge_98 = F.interpolate(edge_full, size=self.output_size, mode="bilinear", align_corners=False)

        result = {
            "edge_full": edge_full.to(dtype=output_dtype),
            "edge_28": edge_28.to(dtype=output_dtype),
            "edge_98": edge_98.to(dtype=output_dtype),
        }
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("GBE produced a non-finite boundary map")
        return result


__all__ = ["ImageNetRGBAdapter", "GradientBoundaryEnhancement"]
