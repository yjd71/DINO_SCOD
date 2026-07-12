import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1x1Decoder(nn.Module):
    """Pure 1x1-convolution decoder for four DINO token feature levels."""

    def __init__(self, in_dim=768):
        super().__init__()
        self.in_dim = in_dim

        self.seg_head_1 = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        self.seg_head_2 = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        self.seg_head_3 = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        self.seg_head_4 = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        self.seg_global = nn.Conv2d(in_dim * 4, 1, kernel_size=1, bias=False)

    def _tokens_to_map(self, feature, patches):
        if feature.ndim != 3:
            raise ValueError(
                f"Expected token features with shape [B, N, C], got {tuple(feature.shape)}"
            )
        if feature.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected feature dimension {self.in_dim}, got {feature.shape[-1]}"
            )
        return feature.permute(0, 2, 1).reshape(
            feature.shape[0], self.in_dim, patches, patches
        )

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(f"Expected four feature levels, got {len(features)}")

        f_1, f_2, f_3, f_4 = features
        patch_num = f_1.shape[1]
        patches = math.isqrt(patch_num)
        if patches * patches != patch_num:
            raise ValueError(f"Token count must form a square grid, got {patch_num}")

        feature_maps = [
            self._tokens_to_map(feature, patches)
            for feature in (f_1, f_2, f_3, f_4)
        ]
        if any(feature.shape[1] != patch_num for feature in (f_2, f_3, f_4)):
            raise ValueError("All feature levels must have the same token count")

        map_1, map_2, map_3, map_4 = feature_maps
        seg_size = (patches * 14) // 4

        seg_1 = self.seg_head_1(map_1)
        seg_2 = self.seg_head_2(map_2)
        seg_3 = self.seg_head_3(map_3)
        seg_4 = self.seg_head_4(map_4)
        seg_g = self.seg_global(torch.cat(feature_maps, dim=1))

        outputs = (seg_4, seg_3, seg_2, seg_1, seg_g)
        return tuple(
            F.interpolate(
                output,
                size=(seg_size, seg_size),
                mode="bilinear",
                align_corners=False,
            )
            for output in outputs
        )
