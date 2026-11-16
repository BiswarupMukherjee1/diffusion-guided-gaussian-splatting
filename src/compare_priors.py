"""Speed against quality for the three priors, with the same data-consistency sampler.

For each test case and each number of network evaluations K, two posterior samples are drawn
and their mean is scored (worm PSNR against the reference, error on the hidden views,
seconds per sample). Data consistency (SIRT on the measured views, settings from tune.py)
is applied after every network evaluation in all three cases, so only the prior and its
sampler differ.

* DDPM: DDIM-style steps (sample.posterior_samples), K in {4, 8, 16, 32, 100}
* rectified flow: x0 = x_t - t v, K in {1, 2, 4, 8, 16, 32}
* shortcut model: x0 = x_t - t s(x_t, t, t) (one jump to t = 0), same K

  python compare_priors.py        # writes results/compare_priors.json
"""

import json
import time

import numpy as np
import torch

import core as C
from flow import FlowUNet, _pred
from sample import DataConsistency, load_prior, posterior_samples

K_DDPM, K_FLOW, N = (4, 8, 16, 32, 100), (1, 2, 4, 8, 16, 32), 2


@torch.no_grad()
def flow_posterior(net, sino, ang_deg, steps, dc_iters, final_iters=20, n_samples=N, seed=0, device="cuda"):
    nz, n = sino.shape[0], C.c.N_DET
    cen, _ = C.centroid_from_sino(sino, ang_deg)
    corners = [C.crop(np.zeros((n, n), np.float32), *cen[k])[1] for k in range(nz)]
    dc = DataConsistency(sino, ang_deg)
    paste_all = lambda cr: np.stack([C.paste(cr[k], corners[k], (n, n)) for k in range(nz)])  # noqa: E731
    crop_all = lambda v: np.stack([C.crop(v[k], *cen[k])[0] for k in range(nz)])  # noqa: E731
    gen = torch.Generator(device=device).manual_seed(seed)
    ts = torch.linspace(1, 0, steps + 1).tolist()
    out = []
    for _ in range(n_samples):
        x = torch.randn((nz, 1, C.CROP, C.CROP), device=device, generator=gen)
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            tt = torch.full((nz,), t, device=device)
            v = _pred(net, x, tt, tt if net.shortcut else None)
            x0 = (x - t * v).clamp(-1, 1)
            if i + 1 < steps:
                d = dc(paste_all(C.to_density(x0[:, 0].cpu().numpy())), dc_iters)
                x0 = torch.from_numpy(C.to_model(crop_all(d))).to(device)[:, None].clamp(-1, 1)
                eps = (x - (1 - t) * x0) / t
                x = (1 - tn) * x0 + tn * eps
        d = dc(paste_all(C.to_density(x0[:, 0].cpu().numpy())), final_iters)
        out.append(paste_all(crop_all(d)))
    return np.stack(out)


def load_flow(kind):
    net = FlowUNet(shortcut=kind == "shortcut").cuda().eval()
    net.load_state_dict(torch.load(C.RUNS / f"flow_{kind}" / "ema.pt", map_location="cuda"))
    return net


def main():
    best = json.loads((C.RESULTS / "tune.json").read_text())["best"]
    priors = {"ddpm": load_prior(), "rf": load_flow("rf"), "shortcut": load_flow("shortcut")}
    f = C.RESULTS / "compare_priors.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {(r["prior"], r["steps"], r["worm"], r["span"]) for r in rows}
    for w in C.TEST_POSES:
        for span in C.SPANS:
            sino, meas, _ = C.case_data(w, span)
            sm, am = sino[:, meas], C.c.ANGLES_DEG[meas]
            for name, net in priors.items():
                for k in (K_DDPM if name == "ddpm" else K_FLOW):
                    if (name, k, w, span) in done:
                        continue
                    torch.cuda.synchronize(); t0 = time.time()
                    if name == "ddpm":
                        s = posterior_samples(net, sm, am, n_samples=N, steps=k, **best)
                    else:
                        s = flow_posterior(net, sm, am, k, best["dc_iters"])
                    sec = (time.time() - t0) / N
                    sc = C.score(s.mean(0), w, span)
                    rows.append(dict(prior=name, steps=k, worm=w, span=span, seconds_per_sample=sec,
                                     worm_psnr_db=sc["voxel"]["worm_psnr_db"],
                                     hidden=sc["projection"]["proj_rel_l2_withheld"],
                                     measured=sc["projection"]["proj_rel_l2_measured"]))
                    C.write_json(f, rows)
                    r = rows[-1]
                    print(f"{name:8s} K={k:3d} w{w} {span}: PSNR {r['worm_psnr_db']:.2f} hidden {100 * r['hidden']:.2f}% "
                          f"{sec:.1f}s/sample", flush=True)


if __name__ == "__main__":
    main()
