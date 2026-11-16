"""Conditional denoisers: they see the noisy full-angle slice, the SIRT reconstruction from the
measured views of the same slice, and which of the 100 views were measured.

All three predict the clean slice (x-prediction) and are trained with the same flow-matching
loss (train_cond.py):
* CondUNet  the U-Net of model.py with a second input channel and an acquisition embedding;
* JiT       a plain transformer on 8 x 8 pixel patches (as in JiT, Li & He, CVPR 2026), with
            DiT-style adaptive layer norm for time and acquisition;
* JiT with xattn=True: each block also cross-attends from the image patches to the measured
            projections of the slice, one token per measured view, tagged with its angle, so
            the network is told directly which directions the data cover.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import core as C
from model import UNet, timestep_embedding

N_VIEWS = C.c.N_VIEWS


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.SiLU(), nn.Linear(o, o))


class CondUNet(UNet):
    def __init__(self, **kw):
        super().__init__(**kw)
        tdim = 4 * self.ch0
        self.inp = nn.Conv2d(2, self.ch0, 3, padding=1)
        self.acq = mlp(N_VIEWS, tdim)

    def forward(self, x, cond, t, mask, tokens=None):
        te = self.temb(timestep_embedding(t * 1000, self.ch0)) + self.acq(mask)
        h = self.inp(torch.cat([x, cond], 1)); hs = [h]
        for m in self.down:
            h = m[1](m[0](h, te)) if isinstance(m, nn.ModuleList) else m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            h = m[1](m[0](torch.cat([h, hs.pop()], 1), te)) if isinstance(m, nn.ModuleList) else m(h)
        return self.out(h)


def sincos_2d(n, dim):
    """Fixed 2-D sine-cosine position embedding for an n x n grid of patches."""
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    om = 1.0 / 10000 ** (np.arange(dim // 4) / (dim // 4))
    e = [np.sin(y.reshape(-1, 1) * om), np.cos(y.reshape(-1, 1) * om),
         np.sin(x.reshape(-1, 1) * om), np.cos(x.reshape(-1, 1) * om)]
    return torch.from_numpy(np.concatenate(e, 1)).float()


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, dim, heads, xattn):
        super().__init__()
        self.heads, self.xattn = heads, xattn
        self.n1, self.n2 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        if xattn:
            self.nx = nn.LayerNorm(dim)
            self.q, self.kv, self.xproj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
            nn.init.zeros_(self.xproj.weight); nn.init.zeros_(self.xproj.bias)

    def _heads(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, c, views=None, vmask=None):
        sa, ca, ga, sf, cf, gf = self.ada(F.silu(c)).chunk(6, -1)
        q, k, v = self.qkv(modulate(self.n1(x), sa, ca)).chunk(3, -1)
        a = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v))
        x = x + ga[:, None] * self.proj(a.transpose(1, 2).flatten(2))
        if self.xattn:
            k, v = self.kv(views).chunk(2, -1)
            a = F.scaled_dot_product_attention(self._heads(self.q(self.nx(x))), self._heads(k), self._heads(v),
                                               attn_mask=vmask[:, None, None, :])
            x = x + self.xproj(a.transpose(1, 2).flatten(2))
        return x + gf[:, None] * self.ff(modulate(self.n2(x), sf, cf))


class JiT(nn.Module):
    def __init__(self, img=C.CROP, patch=8, dim=288, depth=12, heads=6, xattn=False, tok_len=C.TOKEN_LEN,
                 sino_scale=1.0):
        super().__init__()
        self.p, self.n, self.xattn = patch, img // patch, xattn
        self.embed = nn.Linear(2 * patch * patch, dim)
        self.register_buffer("pos", sincos_2d(self.n, dim)[None], persistent=False)
        self.t = mlp(256, dim)
        self.acq = mlp(N_VIEWS, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, xattn) for _ in range(depth)])
        self.nf = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaf = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, patch * patch)
        for m in (self.adaf, self.head):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if xattn:
            self.sino_scale = sino_scale
            self.view = nn.Linear(tok_len, dim)
            self.angle = mlp(16, dim)
            th = torch.deg2rad(torch.from_numpy(C.c.ANGLES_DEG).float())
            k = torch.arange(1, 9).float()
            self.register_buffer("ang_feat", torch.cat([torch.cos(th[:, None] * k), torch.sin(th[:, None] * k)], 1),
                                 persistent=False)  # (100, 16), periodic in 180 deg for even k only; both kept

    def patchify(self, x):
        b = x.shape[0]
        return x.view(b, -1, self.n, self.p, self.n, self.p).permute(0, 2, 4, 1, 3, 5).reshape(b, self.n * self.n, -1)

    def forward(self, x, cond, t, mask, tokens=None):
        b = x.shape[0]
        h = self.embed(self.patchify(torch.cat([x, cond], 1))) + self.pos
        c = self.t(timestep_embedding(t * 1000, 256)) + self.acq(mask)
        views = vmask = None
        if self.xattn:
            views = self.view(tokens / self.sino_scale) + self.angle(self.ang_feat)[None]  # (b, 100, dim)
            vmask = mask > 0.5
        for blk in self.blocks:
            h = blk(h, c, views, vmask)
        sh, sc = self.adaf(F.silu(c)).chunk(2, -1)
        out = self.head(modulate(self.nf(h), sh, sc))  # (b, n*n, p*p)
        return out.view(b, self.n, self.n, self.p, self.p).permute(0, 1, 3, 2, 4).reshape(b, 1, self.n * self.p,
                                                                                           self.n * self.p)


def make(arch, sino_scale=1.0):
    if arch == "unet":
        return CondUNet()
    return JiT(xattn=arch == "jit_xattn", sino_scale=sino_scale)
