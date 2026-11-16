"""Flow-matching priors on the same data and network as the DDPM prior.

Convention: x_t = (1 - t) x0 + t eps, t in [0, 1] (t = 1 is pure noise); velocity v = eps - x0.

* Rectified flow (Liu et al. 2023; Lipman et al. 2023): the network predicts v(x_t, t).
* Shortcut model (Frans et al., "One Step Diffusion via Shortcut Models", ICLR 2025): the
  network also receives the step size d and predicts the average velocity over a jump of d,
  so that x_{t-d} = x_t - d s(x_t, t, d). Trained with flow matching at d = 0 plus a
  self-consistency target: one jump of 2d must equal two jumps of d (no gradient through
  the target). This allows sampling in 1 to 8 steps.

The network is the DDPM U-Net with one extra embedding for d (FlowUNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNet, timestep_embedding

SC_FRACTION, SC_LEVELS = 0.25, 7  # share of each batch used for self-consistency; d in {1, 1/2, ..., 1/128}


class FlowUNet(UNet):
    def __init__(self, shortcut=False, **kw):
        super().__init__(**kw)
        tdim = 4 * self.ch0
        self.shortcut = shortcut
        if shortcut:
            self.demb = nn.Sequential(nn.Linear(self.ch0, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

    def forward(self, x, t, d=None):
        te = self.temb(timestep_embedding(t * 1000, self.ch0))
        if self.shortcut:
            te = te + self.demb(timestep_embedding((d if d is not None else torch.zeros_like(t)) * 1000, self.ch0))
        h = self.inp(x); hs = [h]
        for m in self.down:
            h = m[1](m[0](h, te)) if isinstance(m, nn.ModuleList) else m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            h = m[1](m[0](torch.cat([h, hs.pop()], 1), te)) if isinstance(m, nn.ModuleList) else m(h)
        return self.out(h)


def _pred(net, x, t, d=None):
    with torch.autocast("cuda", torch.bfloat16):
        return net(x, t, d).float()


def flow_loss(net, x0, gen=None):
    """Rectified-flow loss; for a shortcut network, part of the batch gets the self-consistency loss."""
    b = x0.shape[0]
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    if not net.shortcut:
        t = torch.rand(b, device=x0.device, generator=gen)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
        return F.mse_loss(_pred(net, xt, t), eps - x0)
    nsc = int(b * SC_FRACTION)
    # flow matching at d = 0 on the first part of the batch
    t = torch.rand(b - nsc, device=x0.device, generator=gen)
    xt = (1 - t)[:, None, None, None] * x0[nsc:] + t[:, None, None, None] * eps[nsc:]
    loss_fm = F.mse_loss(_pred(net, xt, t, torch.zeros_like(t)), eps[nsc:] - x0[nsc:])
    # self-consistency on the rest: s(x_t, t, 2d) = (s(x_t, t, d) + s(x_{t-d}, t-d, d)) / 2
    k = torch.randint(0, SC_LEVELS, (nsc,), device=x0.device, generator=gen)
    d2 = 2.0 ** (-k.float())                 # big step 2d in {1, ..., 1/64}
    d = d2 / 2
    n = (1.0 / d2).long()                    # t on the grid of the big step: t = j * 2d, j >= 1
    j = torch.floor(torch.rand(nsc, device=x0.device, generator=gen) * n.float()).long() + 1
    t2 = (j.float() * d2).clamp(max=1.0)
    e = lambda v: v[:, None, None, None]  # noqa: E731
    xt2 = (1 - e(t2)) * x0[:nsc] + e(t2) * eps[:nsc]
    with torch.no_grad():
        s1 = _pred(net, xt2, t2, d)
        s2 = _pred(net, xt2 - e(d) * s1, t2 - d, d)
        target = (s1 + s2) / 2
    loss_sc = F.mse_loss(_pred(net, xt2, t2, d2), target)
    return (loss_fm * (b - nsc) + loss_sc * nsc) / b


@torch.no_grad()
def sample_flow(net, n, steps, size=160, gen=None, device="cuda"):
    """Unconditional samples with `steps` uniform jumps (Euler for rectified flow)."""
    x = torch.randn((n, 1, size, size), device=device, generator=gen)
    ts = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t, dt = ts[i].expand(n), ts[i] - ts[i + 1]
        x = x - dt * _pred(net, x, t, dt.expand(n) if net.shortcut else None)
    return x.clamp(-1, 1)
