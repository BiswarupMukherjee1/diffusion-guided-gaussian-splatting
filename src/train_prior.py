"""Train the diffusion prior on full-angle worm slices (data/train.npz).

Noise-prediction DDPM with a cosine schedule, AdamW, bf16 autocast and an EMA copy of the
weights (used for sampling). Augmentation: the 8 flips/rotations of the square and a random
shift of up to 8 px; both give valid full-angle slices, since the prior only models images.
Every --every steps: validation loss on the held-out poses, a checkpoint, and a grid of
unconditional samples for visual checks.

  python train_prior.py --steps 40000 --out runs/prior
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import core as C
from model import T_STEPS, UNet, cosine_alpha_bar


def augment(x, gen):
    k = int(torch.randint(4, (1,), generator=gen))
    x = torch.rot90(x, k, (2, 3))
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(3)
    dy, dx = (int(v) for v in torch.randint(-8, 9, (2,), generator=gen))
    x = torch.roll(x, (dy, dx), (2, 3))  # the border of every crop is (almost) empty, so a roll is a shift
    return x


def loss_fn(net, x0, ab, gen=None):
    t = torch.randint(1, T_STEPS + 1, (x0.shape[0],), device=x0.device, generator=gen)
    a = ab[t - 1][:, None, None, None]
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    with torch.autocast("cuda", torch.bfloat16):
        pred = net(xt, t)
    return F.mse_loss(pred.float(), eps)


@torch.no_grad()
def ddim_sample(net, n, ab, steps=100, gen=None, device="cuda"):
    ts = torch.linspace(T_STEPS, 1, steps).round().long().tolist()
    x = torch.randn((n, 1, C.CROP, C.CROP), device=device, generator=gen)
    for i, t in enumerate(ts):
        a = ab[t - 1]
        with torch.autocast("cuda", torch.bfloat16):
            eps = net(x, torch.full((n,), t, device=device)).float()
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        a_prev = ab[ts[i + 1] - 1] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=C.RUNS / "prior")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    tr = torch.from_numpy(C.to_model(np.load(C.DATA / "train.npz")["x"].astype(np.float32))).half().to(dev)[:, None]
    va = torch.from_numpy(C.to_model(np.load(C.DATA / "val.npz")["x"].astype(np.float32)))[:, None].to(dev)
    net = UNet().to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    ab = cosine_alpha_bar().to(dev)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    log, start, t0 = [], 0, time.time()
    ck = a.out / "last.pt"
    if ck.exists():  # resume
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        start, log = s["step"], s["log"]
    nparam = sum(p.numel() for p in net.parameters())
    print(f"train {len(tr)} val {len(va)} slices, {nparam / 1e6:.1f}M parameters, start at {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        idx = torch.randint(len(tr), (a.batch,), generator=gen)
        x0 = augment(tr[idx.to(dev)].float(), gen)
        lr = a.lr * min(1.0, step / 1000)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = loss_fn(net, x0, ab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.every == 0 or step == a.steps:
            g = torch.Generator(device=dev).manual_seed(123)
            with torch.no_grad():
                vl = float(np.mean([loss_fn(ema, va[i:i + 64], ab, g).item() for i in range(0, len(va), 64)]))
            log.append(dict(step=step, train_loss=loss.item(), val_loss=vl, seconds=time.time() - t0))
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, log=log), ck)
            torch.save(ema.state_dict(), a.out / "ema.pt")
            smp = C.to_density(ddim_sample(ema, 16, ab, gen=torch.Generator(device=dev).manual_seed(0)).cpu().numpy())
            np.savez_compressed(a.out / "samples.npz", x=smp.astype(np.float16), step=step)
            (a.out / "log.json").write_text(json.dumps(log, indent=1))
            print(f"== step {step} val_loss {vl:.4f}", flush=True)


if __name__ == "__main__":
    main()
