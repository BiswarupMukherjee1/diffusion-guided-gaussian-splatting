"""Denoising diffusion model (DDPM, noise prediction, cosine schedule) for 160 x 160 worm slices."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_STEPS = 1000


def cosine_alpha_bar(T=T_STEPS, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]
    return ab.clamp(1e-5, 1 - 1e-5).float()  # alpha_bar_t for t = 1..T, index t-1


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, i, o, tdim, drop=0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(min(32, i // 4), i), nn.Conv2d(i, o, 3, padding=1)
        self.t = nn.Linear(tdim, o)
        self.n2, self.c2 = nn.GroupNorm(min(32, o // 4), o), nn.Conv2d(o, o, 3, padding=1)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.qkv, self.out, self.heads = nn.GroupNorm(min(32, ch // 4), ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(a.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Levels at 160, 80, 40, 20 px; two residual blocks per level (three on the way up); attention at 20 px."""

    def __init__(self, ch=(32, 64, 128, 256), attn=(False, False, False, True), drop=0.1):
        super().__init__()
        tdim = 4 * ch[0]
        self.ch0 = ch[0]
        self.temb = nn.Sequential(nn.Linear(ch[0], tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        skips, c = [ch[0]], ch[0]
        for i, o in enumerate(ch):
            for _ in range(2):
                self.down.append(nn.ModuleList([ResBlock(c, o, tdim, drop), Attention(o) if attn[i] else nn.Identity()]))
                c = o; skips.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim, drop), Attention(c), ResBlock(c, c, tdim, drop)])
        for i, o in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.up.append(nn.ModuleList([ResBlock(c + skips.pop(), o, tdim, drop), Attention(o) if attn[i] else nn.Identity()]))
                c = o
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)))
        self.out = nn.Sequential(nn.GroupNorm(min(32, c // 4), c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(timestep_embedding(t, self.ch0))
        h = self.inp(x); hs = [h]
        for m in self.down:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](h, te))
            else:
                h = m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](torch.cat([h, hs.pop()], 1), te))
            else:
                h = m(h)
        return self.out(h)
