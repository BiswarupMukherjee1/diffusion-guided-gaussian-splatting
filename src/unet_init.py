"""Learned starting volume for R2-Gaussian (Experiment 2).

A small 2-D U-Net maps the limited-arc SIRT slice to the full-arc (180 deg) SIRT slice of the
same pose. Parallel-beam slices are independent 2-D problems, so a 2-D network applied slice
by slice is exact in geometry. Training pairs come from the supplied sinograms of *other*
worm poses: their full 100-view scans give the target, their arc-restricted views give the
input. No phantom is used for training.

Leakage control, fixed before training: all eight phantom poses are excluded, and so is any
pose whose mid-slab sinogram has normalised cross-correlation > 0.9 with the mid slab of a
test pose. No flips or rotations are used for augmentation (they would move the missing
wedge). One ensemble of three seeds is trained per arc (120, 90, 60 deg from 0 deg).

  python unet_init.py data   --out DIR          # build training pairs (GPU SIRT)
  python unet_init.py train  --out DIR --span 90 --seed 0
"""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

import init_lib as L

PHANTOM_POSES = (4, 12, 293, 1031, 1035, 6896, 8916, 8945)
TEST_POSES = (4, 1031, 8916)
SPANS = (120, 90, 60)
NCC_MAX = 0.9
N_SLICES, SIRT_FULL_ITERS, PAD = 16, 300, 320


def all_poses():
    with zipfile.ZipFile(L.c.DEFAULT_ZIP) as zf:
        return sorted(int(n.split("_")[2]) for n in zf.namelist() if n.endswith("_gt.tiff"))


def mid_slab(stack):
    z0, z1 = L.c.slab_range(stack.shape[0], "mid")
    return stack[z0:z1]


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def build_data(out):
    out = Path(out); (out / "pairs").mkdir(parents=True, exist_ok=True)
    test_mid = {w: mid_slab(L.c.load_stack(w)[0]) for w in TEST_POSES}
    index = []
    for w in all_poses():
        f = out / "pairs" / f"worm{w:04d}.npz"
        rec = dict(worm=w, phantom_pose=w in PHANTOM_POSES)
        stack, _ = L.c.load_stack(w)
        rec["ncc_to_test"] = {str(t): ncc(mid_slab(stack), m) for t, m in test_mid.items()}
        rec["excluded"] = rec["phantom_pose"] or max(rec["ncc_to_test"].values()) > NCC_MAX
        if not rec["excluded"] and not f.exists():
            rng = np.random.default_rng(w)
            z0 = int(rng.integers(0, stack.shape[0] - N_SLICES + 1))
            sino = stack[z0:z0 + N_SLICES]
            full = L.sirt_slab(sino, L.c.ANGLES_DEG, SIRT_FULL_ITERS)
            arrs = dict(z0=z0, full=full.astype(np.float16))
            for s in SPANS:
                m = L.c.arc_mask(0, s)
                arrs[f"in{s}"] = L.sirt_slab(sino[:, m], L.c.ANGLES_DEG[m], 200).astype(np.float16)
            np.savez_compressed(f, **arrs)
        index.append(rec)
        print(w, "excluded" if rec["excluded"] else "kept", f"max ncc {max(rec['ncc_to_test'].values()):.3f}", flush=True)
    L.c.write_json(out / "index.json", index)


# ---------------------------------------------------------------- network
def make_net(base=32):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())

    class UNet(nn.Module):
        """Residual U-Net: output = relu(input + correction)."""

        def __init__(self):
            super().__init__()
            ch = [base, base * 2, base * 4, base * 8]
            self.enc = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(3)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(3)])
            self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(3)])
            self.head = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            skips, h = [], x
            for i, e in enumerate(self.enc):
                h = e(h)
                if i < 3:
                    skips.append(h); h = nn.functional.max_pool2d(h, 2)
            for i in (2, 1, 0):
                h = self.dec[i](torch.cat([self.up[i](h), skips[i]], 1))
            return torch.relu(x + self.head(h))

    return UNet()


def pad(x):
    p = (PAD - x.shape[-1]) // 2
    return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(p, PAD - x.shape[-2] - p), (p, PAD - x.shape[-1] - p)])


def unpad(x, n=L.c.N_DET):
    p = (PAD - n) // 2
    return x[..., p:p + n, p:p + n]


def load_pairs(out, span):
    index = json.loads((Path(out) / "index.json").read_text())
    kept = [r["worm"] for r in index if not r["excluded"]]
    X, Y, W = [], [], []
    for w in kept:
        d = np.load(Path(out) / "pairs" / f"worm{w:04d}.npz")
        X.append(d[f"in{span}"]); Y.append(d["full"]); W += [w] * d["full"].shape[0]
    return pad(np.concatenate(X).astype(np.float32)), pad(np.concatenate(Y).astype(np.float32)), np.array(W)


def train(out, span, seed, steps=4000, batch=12, lr=1e-3):
    import torch

    out = Path(out)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, Y, W = load_pairs(out, span)
    poses = np.unique(W)
    val_poses = np.random.default_rng(123).choice(poses, max(1, len(poses) // 10), replace=False)  # same for all seeds
    tr, va = np.flatnonzero(~np.isin(W, val_poses)), np.flatnonzero(np.isin(W, val_poses))
    dev = "cuda"
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    Xt, Yt = torch.from_numpy(X[:, None]), torch.from_numpy(Y[:, None])

    def loss_fn(p, y):  # L1, worm pixels weighted 10x (the worm fills ~2.5% of a slice)
        w = 1 + 9 * (y > L.DENSITY_THRESH).float()
        return (w * (p - y).abs()).sum() / w.sum()

    def val():
        net.eval(); errs = []
        with torch.no_grad():
            for i in range(0, len(va), 16):
                b = va[i:i + 16]
                p = net(Xt[b].to(dev)); y = Yt[b].to(dev)
                errs.append(((p - y).abs() * (y > L.DENSITY_THRESH)).sum().item() / max((y > L.DENSITY_THRESH).sum().item(), 1))
        net.train()
        return float(np.mean(errs))

    with torch.no_grad():
        m = Yt[va] > L.DENSITY_THRESH
        base = float(((Xt[va] - Yt[va]).abs() * m).sum() / m.sum())
    log, t0 = [], time.time()
    for it in range(1, steps + 1):
        b = rng.choice(tr, batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(Xt[b].to(dev))
        loss = loss_fn(p.float(), Yt[b].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == steps:
            log.append(dict(step=it, train_loss=float(loss), val_worm_mae=val(), seconds=time.time() - t0))
            print(span, seed, log[-1], flush=True)
    (out / "models").mkdir(exist_ok=True)
    torch.save(net.state_dict(), out / "models" / f"unet_span{span}_seed{seed}.pt")
    L.c.write_json(out / "models" / f"unet_span{span}_seed{seed}.json",
                   dict(span=span, seed=seed, steps=steps, batch=batch, lr=lr, n_train_slices=len(tr),
                        n_val_slices=len(va), n_train_poses=int(len(poses) - len(val_poses)),
                        val_poses=val_poses.tolist(), val_worm_mae_input=base, log=log))


def load_ensemble(out, span, seeds=(0, 1, 2)):
    import torch

    nets = []
    for s in seeds:
        n = make_net().cuda().eval()
        n.load_state_dict(torch.load(Path(out) / "models" / f"unet_span{span}_seed{s}.pt", map_location="cuda"))
        nets.append(n)
    return nets


def predict(nets, slab):
    """slab (z, y, x) limited-arc SIRT -> ensemble mean and std, (z, y, x)."""
    import torch

    x = torch.from_numpy(pad(np.asarray(slab, np.float32))[:, None]).cuda()
    with torch.no_grad():
        outs = np.stack([unpad(torch.cat([n(x[i:i + 16]) for i in range(0, len(x), 16)])[:, 0].cpu().numpy())
                         for n in nets])
    return outs.mean(0), outs.std(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("data", "train"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--span", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    if a.cmd == "data":
        build_data(a.out)
    else:
        train(a.out, a.span, a.seed, a.steps)


if __name__ == "__main__":
    main()
