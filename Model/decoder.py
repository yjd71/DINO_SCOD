import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class FeedForwardLayer(nn.Module):
    def __init__(self, dim, hidden_dim=768, dropout=0.):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=768, heads=16, hid_dim=768, dropout=0., use_sdpa=True):
        super(Attention, self).__init__()
        self.heads = heads
        assert hid_dim % heads == 0
        dim_head = hid_dim // heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = use_sdpa  # use SDPA or not
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, hid_dim)
        self.to_k = nn.Linear(dim, hid_dim)
        self.to_v = nn.Linear(dim, hid_dim)
        self.to_out = nn.Sequential(nn.Linear(hid_dim, dim), nn.Dropout(dropout))

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), (q, k, v))
        
        # use SDPA for faster attention computation
        if self.use_sdpa:
            # q = q * self.scale
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, hid_dim=hidden_dim, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardLayer(dim, hidden_dim=dim, dropout=dropout)

    def forward(self, q, kv):
        x = kv + self.attn(q, kv, kv)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        out = self.ffn_norm(x)

        return out


class Decoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=128, heads=16, hidden_dim=128, dropout=0.):
        super().__init__()

        self.linear_1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_4 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

        self.linear_1234 = nn.Sequential(
            nn.Linear(in_dim * 4, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        
        self.linear_34 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_23 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_12 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

        self.TransBlock_seg1 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg2 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg3 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg4 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)

        self.seg_global = nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.seg_head_1 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_2 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_3 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_4 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, features):
        f_1, f_2, f_3, f_4 = features
        bs, patch_num, _ = f_1.shape
        patches = int(math.sqrt(patch_num))
        seg_res = (patches * 14) // 4

        query = self.linear_1234(torch.cat([f_1, f_2, f_3, f_4], dim=-1))
        global_mask = query.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        global_mask = F.interpolate(global_mask, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        global_mask = self.seg_global(global_mask)

        mask = torch.sigmoid(global_mask)

        kv_1 = self.linear_1(f_1)
        kv_2 = self.linear_2(f_2)
        kv_3 = self.linear_3(f_3)
        kv_4 = self.linear_4(f_4)

        seg_4 = self.TransBlock_seg4(q=query, kv=kv_4)
        seg_3 = self.TransBlock_seg3(q=query, kv=self.linear_34(torch.cat([kv_3, seg_4], dim=-1)))
        seg_2 = self.TransBlock_seg2(q=query, kv=self.linear_23(torch.cat([kv_2, seg_3], dim=-1)))
        seg_1 = self.TransBlock_seg1(q=query, kv=self.linear_12(torch.cat([kv_1, seg_2], dim=-1)))

        seg_4 = seg_4.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        seg_4 = F.interpolate(seg_4, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        seg_4 = self.seg_head_4(torch.cat([seg_4, seg_4 * mask], dim=1))
        mask_4 = torch.sigmoid(seg_4)

        seg_3 = seg_3.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        seg_3 = F.interpolate(seg_3, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        seg_3 = self.seg_head_3(torch.cat([seg_3, seg_3 * mask_4], dim=1))
        mask_3 = torch.sigmoid(seg_3)

        seg_2 = seg_2.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        seg_2 = F.interpolate(seg_2, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        seg_2 = self.seg_head_2(torch.cat([seg_2, seg_2 * mask_3], dim=1))
        mask_2 = torch.sigmoid(seg_2)

        seg_1 = seg_1.permute(0, 2, 1).reshape(bs, -1, patches, patches)
        seg_1 = F.interpolate(seg_1, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        seg_1 = self.seg_head_1(torch.cat([seg_1, seg_1 * mask_2], dim=1))
        # mask_1 = torch.sigmoid(seg_1)

        return  seg_4, seg_3, seg_2, seg_1, global_mask
