from __future__ import annotations

import pytest
import torch

from Model.BGFBR import (
    BGFBRStage,
    DEFAULT_GPM_DILATIONS,
    DinoGlobalPerceptionModule,
    F4Adapter,
    FinalFusion,
    GradientBoundaryEnhancement,
    ImageNetRGBAdapter,
    ODEBlock,
    P3P2CorrectionBridge,
    RCAB,
)


def test_imagenet_rgb_adapter_round_trip_and_registered_buffers() -> None:
    adapter = ImageNetRGBAdapter()
    rgb = torch.rand(2, 3, 17, 19)
    normalized = (rgb - adapter.mean) / adapter.std

    recovered = adapter(normalized)

    torch.testing.assert_close(recovered, rgb, atol=1e-6, rtol=1e-6)
    assert dict(adapter.named_buffers()).keys() == {"mean", "std"}
    assert adapter(normalized.half()).dtype == torch.float16


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gbe_constant_image_is_zero_and_preserves_decoder_dtype(dtype: torch.dtype) -> None:
    gbe = GradientBoundaryEnhancement(token_size=(8, 8), output_size=(16, 16))
    image = torch.full((2, 3, 21, 23), 0.4, dtype=dtype)

    edges = gbe(image)

    assert edges["edge_full"].shape == (2, 1, 21, 23)
    assert edges["edge_28"].shape == (2, 1, 8, 8)
    assert edges["edge_98"].shape == (2, 1, 16, 16)
    assert all(edge.dtype == dtype for edge in edges.values())
    assert all(torch.count_nonzero(edge) == 0 for edge in edges.values())
    assert not gbe.sobel_x.requires_grad
    assert not gbe.sobel_y.requires_grad


def test_gbe_step_edge_is_finite_and_normalized_per_image() -> None:
    gbe = GradientBoundaryEnhancement(token_size=(8, 8), output_size=(16, 16))
    image = torch.zeros(2, 3, 24, 24)
    image[0, :, :, 12:] = 1.0
    image[1, :, 8:, :] = 0.5

    edges = gbe(image, decoder_dtype=torch.float64)

    full = edges["edge_full"]
    assert full.dtype == torch.float64
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full.amax(dim=(-2, -1)), torch.ones(2, 1, dtype=torch.float64))
    assert (full >= 0).all() and (full <= 1).all()


def test_gbe_rejects_non_rgb_non_finite_and_out_of_range_input() -> None:
    gbe = GradientBoundaryEnhancement()
    with pytest.raises(ValueError, match="shape"):
        gbe(torch.rand(1, 1, 10, 10))
    with pytest.raises(ValueError, match="NaN or Inf"):
        gbe(torch.full((1, 3, 10, 10), float("nan")))
    with pytest.raises(ValueError, match="must be in"):
        gbe(torch.full((1, 3, 10, 10), 1.1))


def test_f4_adapter_is_exact_identity_at_initialization_and_backpropagates() -> None:
    adapter = F4Adapter(channels=16, bottleneck=4, gamma=1.0)
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)

    output = adapter(feature)

    torch.testing.assert_close(output, feature, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(adapter.up.weight) == 0
    output.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert adapter.up.weight.grad is not None


def test_gpm_returns_separate_feature_and_logit_with_finite_gradients() -> None:
    module = DinoGlobalPerceptionModule(channels=32, dilations=(1, 3, 5), dropout=0.0)
    feature = torch.randn(2, 32, 8, 8, requires_grad=True)

    cam_feat, cam_logit = module(feature)

    assert cam_feat.shape == (2, 32, 8, 8)
    assert cam_logit.shape == (2, 1, 8, 8)
    (cam_feat.mean() + cam_logit.mean()).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert module.context_attention.query.out_channels == 4
    assert module.context_attention.value.out_channels == 32


def test_ode_shape_scalar_alpha_and_gradient() -> None:
    ode = ODEBlock(channels=16)
    feature = torch.randn(2, 16, 9, 9, requires_grad=True)

    output, alpha = ode(feature, return_alpha=True)

    assert output.shape == feature.shape
    assert alpha.shape == (2, 1, 1, 1)
    assert (alpha > 0).all() and (alpha < 1).all()
    output.mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_rcab_has_full_residual_body_and_gradient() -> None:
    rcab = RCAB(channels=16, reduction=4)
    feature = torch.randn(2, 16, 9, 9, requires_grad=True)

    output = rcab(feature)

    assert output.shape == feature.shape
    assert sum(isinstance(module, torch.nn.Conv2d) for module in rcab.modules()) == 4
    output.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


@pytest.mark.parametrize("use_ode,use_rcab", [(True, True), (False, False)])
def test_bgfbr_stage_foreground_background_contract_and_gradient(
    use_ode: bool,
    use_rcab: bool,
) -> None:
    stage = BGFBRStage(channels=16, reduction=4, use_ode=use_ode, use_rcab=use_rcab)
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)
    cam_feat = torch.randn(2, 16, 8, 8, requires_grad=True)
    cam_logit = torch.randn(2, 1, 8, 8, requires_grad=True)
    edge = torch.rand(2, 1, 8, 8)

    output = stage(feature, cam_feat, cam_logit, edge)

    assert output.fg_feature.shape == output.bg_feature.shape == (2, 16, 8, 8)
    assert output.fg_logit.shape == output.bg_logit.shape == (2, 1, 8, 8)
    assert output.dual_uncertainty.shape == (2, 1, 8, 8)
    assert (output.dual_uncertainty >= 0).all() and (output.dual_uncertainty <= 1).all()
    loss = output.fg_logit.mean() + output.bg_logit.mean()
    loss.backward()
    for tensor in (feature, cam_feat, cam_logit):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(stage.edge_embed(torch.zeros_like(edge))) == 0


def test_final_fusion_embeds_edges_internally_and_returns_logits() -> None:
    fusion = FinalFusion(channels=16, output_size=(20, 20))
    features = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    edge_28 = torch.rand(2, 1, 8, 8)
    edge_98 = torch.rand(2, 1, 20, 20)

    p1_28, p1_98, z_main = fusion(*features, edge_28, edge_98)

    assert p1_28.shape == (2, 16, 8, 8)
    assert p1_98.shape == (2, 16, 20, 20)
    assert z_main.shape == (2, 1, 20, 20)
    assert torch.isfinite(z_main).all()


def test_p3_p2_bridge_is_exact_identity_at_initialization() -> None:
    bridge = P3P2CorrectionBridge(channels=16)
    p2_pre = torch.randn(2, 16, 8, 8)
    p3_base = torch.randn(2, 16, 8, 8)
    p3_corr = torch.randn(2, 16, 8, 8)
    m2_pre = torch.randn(2, 1, 8, 8)
    edge = torch.rand(2, 1, 8, 8)

    result = bridge(p2_pre, p3_base, p3_corr, m2_pre, edge)

    assert set(result) == {"delta3", "gate32", "delta2", "p2_pc", "m2_pc"}
    torch.testing.assert_close(result["delta3"], p3_corr - p3_base)
    assert result["gate32"].shape == p2_pre.shape
    assert torch.count_nonzero(result["delta2"]) == 0
    torch.testing.assert_close(result["p2_pc"], p2_pre, atol=0.0, rtol=0.0)
    torch.testing.assert_close(result["m2_pc"], m2_pre, atol=0.0, rtol=0.0)
    assert bridge.gate.in_channels == 33


def test_p3_p2_bridge_stays_identity_without_valid_neighbors_after_training() -> None:
    bridge = P3P2CorrectionBridge(channels=8)
    with torch.no_grad():
        bridge.delta_projection.weight.normal_()
        bridge.delta_projection.bias.normal_()
        bridge.mask_head.weight.normal_()
        bridge.mask_head.bias.normal_()
    p2_pre = torch.randn(2, 8, 6, 6)
    p3_base = torch.randn(2, 8, 6, 6)
    m2_pre = torch.randn(2, 1, 6, 6)
    result = bridge(
        p2_pre,
        p3_base,
        p3_base.clone(),
        m2_pre,
        torch.rand(2, 1, 6, 6),
        valid_mask=torch.zeros(2, 1, 6, 6),
    )
    torch.testing.assert_close(result["delta2"], torch.zeros_like(result["delta2"]))
    torch.testing.assert_close(result["p2_pc"], p2_pre, atol=0.0, rtol=0.0)
    torch.testing.assert_close(result["m2_pc"], m2_pre, atol=0.0, rtol=0.0)


def test_default_128_channel_and_spatial_contracts() -> None:
    gpm = DinoGlobalPerceptionModule()
    stage = BGFBRStage()
    final_fusion = FinalFusion()
    bridge = P3P2CorrectionBridge()

    assert gpm.channels == stage.channels == final_fusion.channels == bridge.channels == 128
    assert gpm.dilations == DEFAULT_GPM_DILATIONS
    assert final_fusion.output_size == (98, 98)
    assert bridge.gate.in_channels == 257


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP validation")
def test_gbe_forces_fp32_compute_inside_cuda_autocast() -> None:
    gbe = GradientBoundaryEnhancement().cuda()
    image = torch.rand(2, 3, 64, 64, device="cuda", dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        edges = gbe(image)

    assert all(edge.dtype == torch.float16 for edge in edges.values())
    assert all(torch.isfinite(edge).all() for edge in edges.values())
