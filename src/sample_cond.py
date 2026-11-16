"""Posterior sampling with a conditional x-prediction model, with or without data consistency.

Sampling runs the flow from t = 1 (noise) to t = 0 in K uniform steps. At each step the model
predicts the clean slice; with dc_iters > 0 that prediction is corrected by CGLS iterations on
the measured views of the full 310 x 310 slice (DDS-style; the updates lie in the range of A^T, so
only what the measured views see is changed). The next state is the corrected prediction
re-noised with the noise implied by the current state. Without data consistency the result is
whatever the network believes, which is the setting in which hallucinations are not checked
against the data.
"""

import json

import numpy as np
import torch

import core as C
from cond_models import make
from sample import DataConsistency


def load(arch, device="cuda"):
    d = C.RUNS / f"cond_{arch}"
    cfg = json.loads((d / "config.json").read_text())
    net = make(arch, cfg["sino_scale"]).to(device).eval()
    net.load_state_dict(torch.load(d / "ema.pt", map_location=device))
    return net


@torch.no_grad()
def sample(net, sino, meas, n_samples=4, steps=50, dc_iters=10, final_iters=10, seed=0, init=None, t_start=1.0,
           device="cuda"):
    """sino: (z, 100, det) supplied projections (only the views in `meas` are used).
    Returns (n_samples, z, 310, 310) densities in g/cm^3.

    With init (a (z, 310, 310) density volume) and t_start < 1, sampling starts from init mixed with
    noise at level t_start, x = (1 - t_start) init + t_start eps, and runs from t_start to 0: the
    model repairs an existing reconstruction instead of drawing one from scratch (SDEdit-style)."""
    nz, n = sino.shape[0], C.c.N_DET
    sm, am = sino[:, meas], C.c.ANGLES_DEG[meas]
    cen, _ = C.centroid_from_sino(sm, am)
    corners = [C.crop(np.zeros((n, n), np.float32), *cen[k])[1] for k in range(nz)]
    paste_all = lambda cr: np.stack([C.paste(cr[k], corners[k], (n, n)) for k in range(nz)])  # noqa: E731
    crop_all = lambda v: np.stack([C.crop(v[k], *cen[k])[0] for k in range(nz)])  # noqa: E731
    sirt = C.L.sirt_slab(sm, am, 200)
    cond = torch.from_numpy(C.to_model(crop_all(sirt))).to(device)[:, None]
    mask = torch.from_numpy(np.tile(meas.astype(np.float32), (nz, 1))).to(device)
    full = np.zeros_like(sino); full[:, meas] = sino[:, meas]  # hidden views are never passed in
    tok = torch.from_numpy(C.view_tokens(full, cen)).to(device)
    dc = DataConsistency(sm, am) if (dc_iters or final_iters) else None
    ts = torch.linspace(t_start, 0, steps + 1).tolist()
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(n_samples):
        x = torch.randn((nz, 1, C.CROP, C.CROP), device=device, generator=gen)
        if init is not None:
            x0 = torch.from_numpy(C.to_model(crop_all(init))).to(device)[:, None].clamp(-1, 1)
            x = (1 - t_start) * x0 + t_start * x
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            with torch.autocast("cuda", torch.bfloat16):
                xh = net(x, cond, torch.full((nz,), t, device=device), mask, tok).float().clamp(-1, 1)
            if dc_iters and i + 1 < steps:
                d = dc(paste_all(C.to_density(xh[:, 0].cpu().numpy())), dc_iters, "cgls")
                xh = torch.from_numpy(C.to_model(crop_all(d))).to(device)[:, None].clamp(-1, 1)
            if i + 1 < steps:
                eps = (x - (1 - t) * xh) / t
                x = (1 - tn) * xh + tn * eps
        d = paste_all(C.to_density(xh[:, 0].cpu().numpy()))
        if final_iters:
            d = paste_all(crop_all(dc(d, final_iters, "cgls")))
        out.append(d)
    return np.stack(out)
