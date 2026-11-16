"""Train a flow-matching prior (rectified flow or shortcut model) with the same data, network,
augmentation, optimiser, batch and number of steps as the DDPM prior (train_prior.py).

  python train_flow.py --kind rf --out runs/flow_rf
  python train_flow.py --kind shortcut --out runs/flow_shortcut
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import core as C
from flow import FlowUNet, flow_loss, sample_flow
from train_prior import augment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["rf", "shortcut"], required=True)
    ap.add_argument("--out", type=Path, required=True)
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
    net = FlowUNet(shortcut=a.kind == "shortcut").to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    gpu_gen = torch.Generator(device=dev).manual_seed(a.seed)
    log, start, t0 = [], 0, time.time()
    ck = a.out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        start, log = s["step"], s["log"]
    print(f"{a.kind}: train {len(tr)} val {len(va)}, {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, "
          f"start at {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        idx = torch.randint(len(tr), (a.batch,), generator=gen)
        x0 = augment(tr[idx.to(dev)].float(), gen)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 1000)
        loss = flow_loss(net, x0, gpu_gen)
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
            g = torch.Generator(device=dev).manual_seed(123)
            with torch.no_grad():
                vl = float(np.mean([flow_loss(ema, va[i:i + 64], g).item() for i in range(0, len(va), 64)]))
            log.append(dict(step=step, train_loss=loss.item(), val_loss=vl, seconds=time.time() - t0))
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, log=log), ck)
            torch.save(ema.state_dict(), a.out / "ema.pt")
            smp = {f"steps{k}": C.to_density(sample_flow(ema, 8, k, gen=torch.Generator(device=dev).manual_seed(0))
                                              .cpu().numpy()).astype(np.float16) for k in (1, 4, 32)}
            np.savez_compressed(a.out / "samples.npz", step=step, **smp)
            (a.out / "log.json").write_text(json.dumps(log, indent=1))
            print(f"== step {step} val_loss {vl:.4f}", flush=True)


if __name__ == "__main__":
    main()
