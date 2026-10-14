"""Exact causal attention with rotary positions and per-head elapsed-time bias."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalAttention(nn.Module):
    def __init__(self, dim=128, heads=4, backend='sdpa'):
        super().__init__()
        if dim % heads or (dim // heads) % 2 or backend not in ('sdpa', 'manual'):
            raise ValueError('Even head width and an exact attention backend required')
        self.heads, self.width, self.backend = heads, dim // heads, backend
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.gap = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, heads))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, dim))
        self.register_buffer('frequency', 10000 ** (-torch.arange(0, self.width, 2).float() / self.width))

    def rotate(self, x, positions):
        angle = positions[:, None, :, None].to(x.dtype) * self.frequency.to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2)

    def forward(self, x, valid, times):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        # Timestamps stay float64 until subtraction, preserving small gaps at epoch scale.
        gap = (times[:, :, None] - times[:, None, :]).clamp_min(0).log1p().to(x.dtype)
        bias = self.gap(gap[..., None]).permute(0, 3, 1, 2)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None] & valid[:, None, :] & valid[:, :, None]
        # Give padded queries a safe self position, then explicitly zero their output.
        allowed |= (~valid[:, :, None]) & torch.eye(length, device=x.device, dtype=torch.bool)[None]
        bias = bias.masked_fill(~allowed[:, None], -torch.inf)
        if self.backend == 'sdpa':
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        else:
            y = ((q @ k.transpose(-1, -2)) / math.sqrt(self.width) + bias).softmax(-1) @ v
        y = y.transpose(1, 2).reshape(batch, length, dim)
        x = self.norm1(x + self.out(y)) * valid[..., None]
        return self.norm2(x + self.ff(x)) * valid[..., None]

    def forward_last(self, x, valid, times):
        """Exact final-position read without computing unused earlier outputs.

        K/V and their gradients still cover every history position. Both
        normalization and the FFN are positionwise, so earlier query outputs
        cannot affect the final one. Keep the shared QKV projection unchanged.
        """
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        q = self.rotate(q[:, :, -1:], positions[:, -1:])
        k = self.rotate(k, positions)
        gap = (times[:, -1:, None] - times[:, None, :]).clamp_min(0).log1p().to(x.dtype)
        bias = self.gap(gap[..., None]).permute(0, 3, 1, 2)
        allowed = valid[:, None, :] & valid[:, -1:, None]
        # Match forward's safe diagonal for an invalid final query exactly.
        diagonal = torch.arange(length, device=x.device) == length - 1
        allowed |= (~valid[:, -1:, None]) & diagonal[None, None, :]
        bias = bias.masked_fill(~allowed[:, None], -torch.inf)
        if self.backend == 'sdpa':
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        else:
            y = ((q @ k.transpose(-1, -2)) / math.sqrt(self.width) + bias).softmax(-1) @ v
        y = y.transpose(1, 2).reshape(batch, 1, dim)
        mask = valid[:, -1:, None]
        last = self.norm1(x[:, -1:] + self.out(y)) * mask
        return (self.norm2(last + self.ff(last)) * mask)[:, 0]


class ConvTemporalAttention(nn.Module):
    """One causal depthwise/pointwise CNN block before exact temporal attention."""
    def __init__(self, dim=128, heads=1, backend='sdpa'):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 3, groups=dim)
        self.pointwise = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attention = TemporalAttention(dim, heads, backend)

    def local(self, x, valid):
        mask = valid[..., None]
        # Left padding only: token t reads t-2, t-1 and t, never a future token.
        y = self.depthwise(F.pad((x * mask).transpose(1, 2), (2, 0))).transpose(1, 2)
        return self.norm(x + self.pointwise(F.gelu(y))) * mask

    def forward(self, x, valid, times):
        return self.attention(self.local(x, valid), valid, times)

    def forward_last(self, x, valid, times):
        return self.attention.forward_last(self.local(x, valid), valid, times)

