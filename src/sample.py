"""Posterior sampling for limited-angle slices with the diffusion prior and data consistency.

DDS-style sampler (Chung et al., ICLR 2024), with SIRT in place of conjugate gradients:
at every reverse step the network's estimate of the clean slice is pasted into the full
310 x 310 slice and corrected by a few SIRT iterations on the measured views only. SIRT
updates lie in the range of the back-projector, so they change only what the measured views
see; the missing directions are left to the prior. The corrected estimate is then re-noised
to the next (lower) noise level. Several samples per case give a posterior mean and a
per-voxel spread.

Used by run_test.py (test cases) and tune.py (validation poses).
"""

import numpy as np
import torch

import core as C
from model import T_STEPS, UNet, cosine_alpha_bar


class DataConsistency:
    """k SIRT iterations on the measured views, weights precomputed once per slab."""

    def __init__(self, sino, ang_deg):
        import tigre
        self.tigre = tigre
        self.geo = C.c.tigre_geo(sino.shape[0])
        self.ang = np.deg2rad(ang_deg).astype(np.float32)
        self.b = C.c.to_tigre_proj(sino)
        self.R = 1.0 / np.maximum(tigre.Ax(np.ones(self.geo.nVoxel, np.float32), self.geo, self.ang), 1e-6)
        self.Cw = 1.0 / np.maximum(tigre.Atb(np.ones_like(self.b), self.geo, self.ang), 1e-6)

    def __call__(self, x, k, mode="sirt"):
        x = np.ascontiguousarray(x, np.float32)
        if mode == "cgls":
            return self.cgls(x, k)
        for _ in range(k):
            x = np.maximum(x + self.Cw * self.tigre.Atb(self.R * (self.b - self.tigre.Ax(x, self.geo, self.ang)),
                                                        self.geo, self.ang), 0)
        return x

    def cgls(self, x, k):
        """k conjugate-gradient iterations on ||A x - b||^2 started at x (as in DDS), then x >= 0.
        The updates lie in the range of A^T, so the part of x the measured views do not see is kept."""
        A = lambda v: self.tigre.Ax(np.ascontiguousarray(v, np.float32), self.geo, self.ang)  # noqa: E731
        At = lambda v: self.tigre.Atb(np.ascontiguousarray(v, np.float32), self.geo, self.ang)  # noqa: E731
        r = self.b - A(x)
        s = At(r)
        p, gamma = s.copy(), float((s * s).sum())
        for _ in range(k):
            q = A(p)
            alpha = gamma / max(float((q * q).sum()), 1e-30)
            x = x + alpha * p
            r = r - alpha * q
            s = At(r)
            g_new = float((s * s).sum())
            p = s + (g_new / max(gamma, 1e-30)) * p
            gamma = g_new
        return np.maximum(x, 0)


def load_prior(path=C.RUNS / "prior" / "ema.pt", device="cuda"):
    net = UNet().to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device))
    return net


@torch.no_grad()
def posterior_samples(net, sino, ang_deg, n_samples=4, steps=100, dc_iters=5, eta=0.15, final_iters=20, seed=0,
                      init=None, t_start=T_STEPS, dc_mode="sirt", device="cuda"):
    """sino: (z, n_meas, det) physical units. Returns (n_samples, z, 310, 310) densities in g/cm^3.

    With init (a (z, 310, 310) density volume), sampling starts from init noised to level t_start
    instead of from pure noise (SDEdit-style repair of an existing reconstruction)."""
    nz, n = sino.shape[0], C.c.N_DET
    cen, _ = C.centroid_from_sino(sino, ang_deg)
    corners = [C.crop(np.zeros((n, n), np.float32), *cen[k])[1] for k in range(nz)]
    dc = DataConsistency(sino, ang_deg)
    ab = cosine_alpha_bar().to(device)
    ts = torch.linspace(t_start, 1, steps).round().long().tolist()
    gen = torch.Generator(device=device).manual_seed(seed)

    def paste_all(crops):  # (nz, CROP, CROP) -> (nz, n, n)
        return np.stack([C.paste(crops[k], corners[k], (n, n)) for k in range(nz)])

    def crop_all(vol):
        return np.stack([C.crop(vol[k], *cen[k])[0] for k in range(nz)])

    out = []
    for _ in range(n_samples):
        x = torch.randn((nz, 1, C.CROP, C.CROP), device=device, generator=gen)
        if init is not None:
            a0 = ab[ts[0] - 1]
            x = a0.sqrt() * torch.from_numpy(C.to_model(crop_all(init))).to(device)[:, None].clamp(-1, 1) + (1 - a0).sqrt() * x
        for i, t in enumerate(ts):
            a = ab[t - 1]
            with torch.autocast("cuda", torch.bfloat16):
                eps = net(x, torch.full((nz,), t, device=device)).float()
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            d = dc(paste_all(C.to_density(x0[:, 0].cpu().numpy())), dc_iters, dc_mode)  # data consistency
            x0 = torch.from_numpy(C.to_model(crop_all(d))).to(device)[:, None].clamp(-1, 1)
            if i + 1 == len(ts):
                break
            a_prev = ab[ts[i + 1] - 1]
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps \
                + sigma * torch.randn(x.shape, device=device, generator=gen)
        d = dc(paste_all(C.to_density(x0[:, 0].cpu().numpy())), final_iters, dc_mode)
        out.append(paste_all(crop_all(d)))  # zero outside the crop
    return np.stack(out)
