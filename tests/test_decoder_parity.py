import math

import torch
import torch.nn.functional as F

from Model.decoder import Attention, Decoder, map_to_tokens, tokens_to_map


class LegacyDecoder(Decoder):
    """Frozen copy of the pre-refactor RSBL decoder forward formula."""

    def forward(self, features):
        f_1, f_2, f_3, f_4 = features
        bs, patch_num, _ = f_1.shape
        patches = int(math.sqrt(patch_num))
        seg_res = (patches * 14) // 4

        query = self.linear_1234(torch.cat([f_1, f_2, f_3, f_4], dim=-1))
        global_mask = query.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        global_mask = F.interpolate(
            global_mask, size=(seg_res, seg_res), mode='bilinear', align_corners=False
        )
        global_mask = self.seg_global(global_mask)
        mask = torch.sigmoid(global_mask)

        kv_1 = self.linear_1(f_1)
        kv_2 = self.linear_2(f_2)
        kv_3 = self.linear_3(f_3)
        kv_4 = self.linear_4(f_4)

        seg_4 = self.TransBlock_seg4(q=query, kv=kv_4)
        seg_3 = self.TransBlock_seg3(
            q=query, kv=self.linear_34(torch.cat([kv_3, seg_4], dim=-1))
        )
        seg_2 = self.TransBlock_seg2(
            q=query, kv=self.linear_23(torch.cat([kv_2, seg_3], dim=-1))
        )
        seg_1 = self.TransBlock_seg1(
            q=query, kv=self.linear_12(torch.cat([kv_1, seg_2], dim=-1))
        )

        def predict(tokens, previous_mask, head):
            feature = tokens.permute(0, 2, 1).reshape(bs, -1, patches, patches)
            feature = F.interpolate(
                feature, size=(seg_res, seg_res), mode='bilinear', align_corners=False
            )
            return head(torch.cat([feature, feature * previous_mask], dim=1))

        seg_4 = predict(seg_4, mask, self.seg_head_4)
        seg_3 = predict(seg_3, torch.sigmoid(seg_4), self.seg_head_3)
        seg_2 = predict(seg_2, torch.sigmoid(seg_3), self.seg_head_2)
        seg_1 = predict(seg_1, torch.sigmoid(seg_2), self.seg_head_1)
        return seg_4, seg_3, seg_2, seg_1, global_mask


def test_decoder_off_mode_matches_legacy_formula():
    torch.manual_seed(2025)
    legacy = LegacyDecoder().eval()
    refactored = Decoder().eval()
    refactored.load_state_dict(legacy.state_dict(), strict=True)
    features = [torch.randn(2, 784, 768) for _ in range(4)]

    with torch.no_grad():
        expected = legacy(features)
        actual = refactored(features, memory=None, pc_mode='off')

    assert len(actual) == 5
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6
        )


def test_decoder_default_state_dict_keys_are_legacy_compatible():
    decoder = Decoder()
    assert len(decoder.state_dict()) == 101
    assert all(not key.startswith('pc_hbm.') for key in decoder.state_dict())


def test_token_map_round_trip():
    tokens = torch.randn(3, 28 * 28, 128)
    feature_map = tokens_to_map(tokens, 28, 28)
    assert feature_map.shape == (3, 128, 28, 28)
    torch.testing.assert_close(map_to_tokens(feature_map), tokens)


def test_sdpa_matches_manual_attention():
    torch.manual_seed(7)
    sdpa = Attention(dim=128, heads=8, hid_dim=128, dropout=0.0, use_sdpa=True).eval()
    manual = Attention(dim=128, heads=8, hid_dim=128, dropout=0.0, use_sdpa=False).eval()
    manual.load_state_dict(sdpa.state_dict())
    q = torch.randn(2, 37, 128)
    k = torch.randn(2, 41, 128)
    v = torch.randn(2, 41, 128)

    with torch.no_grad():
        actual = sdpa(q, k, v)
        expected = manual(q, k, v)

    assert actual.shape == (2, 37, 128)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
