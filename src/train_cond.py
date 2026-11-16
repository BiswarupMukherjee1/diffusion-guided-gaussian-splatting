"""Train a conditional denoiser (cond_models.py) with flow matching and x-prediction.

x_t = (1 - t) x + t eps with t from a logit-normal distribution. The network predicts the clean
slice x_hat; the loss is the velocity error ||v_hat - v||^2 = ||x_hat - x||^2 / t^2 (t >= 0.05 in
the denominator), as in JiT. Each training example gets a random acquisition setting from
core.SETTINGS (three arcs, two sparse-view settings). No flips or rotations: they would change
the acquisition geometry.

  python train_cond.py --arch unet|jit|jit_xattn --out runs/cond_<arch>
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import core as C
from cond_models import make

NAMES = list(C.SETTINGS)
MASKS = torch.from_numpy(np.stack([C.SETTINGS[k] for k in NAMES]).astype(np.float32))


def load(split):
    x = np.load(C.DATA / f"{split}.npz")["x"]
    d = np.load(C.DATA / f"cond_{split}.npz")
    cond = np.stack([d[k] for k in NAMES], 1)  # (n, settings, H, W) float16
    return x, cond, d["tokens"]


def batch(x, cond, tok, idx, sidx, dev):
    x0 = torch.from_numpy(C.to_model(x[idx].astype(np.float32)))[:, None]
    cd = torch.from_numpy(C.to_model(cond[idx, sidx].astype(np.float32)))[:, None]
    m = MASKS[sidx]
    tk = torch.from_numpy(tok[idx].astype(np.float32)) * m[:, :, None]  # unmeasured views are zeroed
    return x0.to(dev), cd.to(dev), m.to(dev), tk.to(dev)


def loss_fn(net, x0, cd, m, tk, gen):
    t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device, generator=gen))
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
    with torch.autocast("cuda", torch.bfloat16):
        xh = net(xt, cd, t, m, tk).float()
    return (((xh - x0) / t.clamp(min=0.05)[:, None, None, None]) ** 2).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["unet", "jit", "jit_xattn"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lr = a.lr or (2e-4 if a.arch == "unet" else 1e-4)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    xtr, ctr, ttr = load("train")
    xva, cva, tva = load("val")
    sino_scale = float(np.percentile(ttr[:500].astype(np.float32), 99.9))
    net = make(a.arch, sino_scale).to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    log, start, t0 = [], 0, time.time()
    ck = a.out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        start, log = s["step"], s["log"]
    (a.out / "config.json").write_text(json.dumps(dict(arch=a.arch, sino_scale=sino_scale, lr=lr, batch=a.batch,
                                                       steps=a.steps, settings=NAMES)))
    print(f"{a.arch}: {len(xtr)} train / {len(xva)} val slices x {len(NAMES)} settings, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, start at {start}", flush=True)
    vrng = np.random.default_rng(123)
    vidx = vrng.choice(len(xva), 256, replace=False)
    vset = vrng.integers(0, len(NAMES), 256)
    for step in range(start + 1, a.steps + 1):
        idx = rng.integers(0, len(xtr), a.batch)
        sidx = rng.integers(0, len(NAMES), a.batch)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 1000)
        loss = loss_fn(net, *batch(xtr, ctr, ttr, idx, sidx, dev), gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            dk = min(0.999, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - dk)
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.every == 0 or step == a.steps:
            g = torch.Generator(device=dev).manual_seed(7)
            with torch.no_grad():
                vl = float(np.mean([loss_fn(ema, *batch(xva, cva, tva, vidx[i:i + 64], vset[i:i + 64], dev), g).item()
                                    for i in range(0, 256, 64)]))
            log.append(dict(step=step, train_loss=loss.item(), val_loss=vl, seconds=time.time() - t0))
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, log=log), ck)
            torch.save(ema.state_dict(), a.out / "ema.pt")
            (a.out / "log.json").write_text(json.dumps(log, indent=1))
            print(f"== step {step} val_loss {vl:.4f}", flush=True)


if __name__ == "__main__":
    main()
