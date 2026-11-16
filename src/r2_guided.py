"""R2-Gaussian training with an optional diffusion guidance term ("diffusion in the loop").

Runs the unchanged native training loop of R2-Gaussian  and
adds, through two hooks, a guidance loss:

  every --guide_every iterations:  w * (L_render / L_guide) * L_guide,
  L_guide = mean |V - G| over the voxelised Gaussians V and a guidance volume G.

G starts as the diffusion posterior mean of the case. At the --refresh iterations it is
replaced by a diffusion repair of the current reconstruction: V is noised to level --t_start
and denoised again with data consistency on the measured views (sample.posterior_samples
with init=V), so the measured views stay fixed and only what they do not see is redrawn by the
prior. The ratio L_render / L_guide (no gradient) keeps the guidance on the same scale as the
data term throughout training. Without --guide_target this is plain R2-Gaussian.

Arguments are those of train_worm.py, plus the --guide_* ones below.
"""

import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import core as C  # noqa: E402

os.chdir(C.R2_ROOT)
sys.path.insert(0, str(C.R2_ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402  (native R2-Gaussian training module, unchanged)
from r2_gaussian.gaussian import query  # noqa: E402


class Guide:
    g = sc = pipe = target = net = None
    it, every, weight, refresh, t_start, steps, dc = 0, 4, 1.0, (), 300, 30, {}
    cond = False  # True: conditional model (sample_cond.py), t_start is then a flow time in (0, 1)
    worm = span = None
    log = []


G = Guide()


def _vol():
    return query(G.g, G.sc["offOrigin"], G.sc["nVoxel"], G.sc["sVoxel"], G.pipe)["vol"]  # (x, y, z)


def _refresh():
    with torch.no_grad():
        cur = np.transpose(_vol().float().cpu().numpy(), (2, 1, 0))  # (z, y, x)
    sino, meas, _ = C.case_data(G.worm, G.span)
    t0 = time.time()
    if G.cond:
        from sample_cond import sample
        s = sample(G.net, sino, meas, n_samples=2, steps=G.steps, dc_iters=10, final_iters=10, seed=G.it,
                   init=cur, t_start=G.t_start)
    else:
        from sample import posterior_samples
        s = posterior_samples(G.net, sino[:, meas], C.c.ANGLES_DEG[meas], n_samples=2, steps=G.steps, init=cur,
                              t_start=int(G.t_start), seed=G.it, **G.dc)
    G.target = torch.from_numpy(np.ascontiguousarray(np.transpose(s.mean(0), (2, 1, 0)))).cuda()
    G.log.append(dict(iteration=G.it, refresh_seconds=time.time() - t0))


def install_hooks():
    orig_init, orig_l1, OrigScene = T.initialize_gaussian, T.l1_loss, T.Scene

    class SceneHook(OrigScene):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            G.sc = self.scanner_cfg

    def init_hook(g, *a, **k):
        G.g = g
        upd = g.update_learning_rate

        def upd_hook(it):
            G.it = it
            return upd(it)

        g.update_learning_rate = upd_hook
        return orig_init(g, *a, **k)

    def l1_hook(img, gt):
        loss = orig_l1(img, gt)
        if G.target is None or not torch.is_grad_enabled():
            return loss
        if G.it in G.refresh:
            _refresh()
        if G.it % G.every == 0:
            lg = (_vol() - G.target).abs().mean()
            loss = loss + G.weight * (loss.detach() / lg.detach().clamp(min=1e-12)) * lg
            if G.it % 1000 == 0:
                G.log.append(dict(iteration=G.it, render_l1=float(orig_l1(img, gt)), guide_l1=float(lg)))
        return loss

    T.Scene, T.initialize_gaussian, T.l1_loss = SceneHook, init_hook, l1_hook


def main():
    from argparse import ArgumentParser
    argv = sys.argv[1:]

    def pop(name, typ=str, n=None, default=None):
        """Remove --name and its value(s) from argv; with n, return a list of n values."""
        if name not in argv:
            return default
        i = argv.index(name)
        v = argv[i + 1:i + 1 + (n or 1)]
        del argv[i:i + 1 + (n or 1)]
        return [typ(x) for x in v] if n else typ(v[0])

    record = pop("--run_record")
    target = pop("--guide_target")
    G.worm, G.span = pop("--guide_case", int, 2, (None, None))
    G.every = pop("--guide_every", int, default=4)
    G.weight = pop("--guide_weight", float, default=1.0)
    cond_arch = pop("--guide_cond")  # e.g. jit_xattn: refresh with that conditional model instead of the prior
    G.cond = cond_arch is not None
    G.t_start = pop("--guide_t_start", float, default=0.4 if G.cond else 300)
    G.steps = pop("--guide_steps", int, default=30)
    prior = pop("--guide_prior", default=str(C.RUNS / "prior" / "ema.pt"))
    nref = pop("--guide_refresh_n", int, default=0)
    G.refresh = tuple(pop("--guide_refresh", int, nref)) if nref else ()
    if target:
        from sample import load_prior
        G.target = torch.from_numpy(np.ascontiguousarray(np.transpose(
            np.load(target)["mean"].astype(np.float32), (2, 1, 0)))).cuda()
        if G.cond:
            from sample_cond import load
            G.net = load(cond_arch)
        else:
            G.net = load_prior(prior)
        tune = C.RESULTS / "tune.json"
        G.dc = json.loads(tune.read_text())["best"] if tune.exists() else dict(dc_iters=5, eta=0.15)
    parser = ArgumentParser()
    lp, op, pp = T.ModelParams(parser), T.OptimizationParams(parser), T.PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    args.save_iterations.append(args.iterations)
    args.test_iterations += [args.iterations, 1]
    T.safe_state(args.quiet)
    tb = T.prepare_output_and_logger(args)
    if tb is not None:
        tb.close()
    G.pipe = pp.extract(args)
    install_hooks()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    T.training(lp.extract(args), op.extract(args), G.pipe, None, args.test_iterations, args.save_iterations,
               args.checkpoint_iterations, args.start_checkpoint)
    if record:
        import pickle
        pc = Path(args.model_path) / f"point_cloud/iteration_{args.iterations}/point_cloud.pickle"
        n = int(pickle.load(open(pc, "rb"))["xyz"].shape[0]) if pc.exists() else None
        Path(record).write_text(json.dumps(dict(seed=0, argv=sys.argv[1:], elapsed_seconds=time.time() - t0,
                                                peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                                final_gaussians=n, gpu=torch.cuda.get_device_name(0),
                                                guide=dict(target=target, every=G.every, weight=G.weight,
                                                           refresh=list(G.refresh), t_start=G.t_start, steps=G.steps,
                                                           cond=cond_arch,
                                                           log=G.log)), indent=1))


if __name__ == "__main__":
    main()
