"""Choose the sampler settings on validation poses only (never on the test poses).

For 4 validation poses, a 32-slice mid block is cut to the 60 and 120 degree arcs. Each
setting (data-consistency iterations per step, eta) is scored by the error on the views
outside the arc against the supplied data. This score needs no reference volume.

  python tune.py        # writes results/tune.json
"""

import itertools
import json
import time

import numpy as np

import core as C
from sample import load_prior, posterior_samples

GRID = dict(dc_iters=(2, 5, 10), eta=(0.0, 0.3))


def hidden_err(vol, sino, meas):
    from evaluate_run import projection_metrics
    pm = projection_metrics(np.ascontiguousarray(vol, np.float32), sino, sino, np.flatnonzero(meas))
    return pm["proj_rel_l2_withheld"], pm["proj_rel_l2_measured"]


def main():
    net = load_prior()
    _, val = C.kept_poses()
    rows = []
    for w in val[:4]:
        stack, _ = C.c.load_stack(w)
        nonempty = int(np.flatnonzero(stack.reshape(stack.shape[0], -1).max(1) > 0)[-1]) + 1
        z0, z1 = C.c.slab_range(nonempty, "mid", 32)
        sino = stack[z0:z1]
        for span in (60, 120):
            meas = C.c.arc_mask(0, span)
            sirt = C.L.sirt_slab(sino[:, meas], C.c.ANGLES_DEG[meas], 200)
            h, m = hidden_err(sirt, sino, meas)
            rows.append(dict(worm=w, span=span, method="sirt200", hidden=h, measured=m))
            for k, eta in itertools.product(GRID["dc_iters"], GRID["eta"]):
                t0 = time.time()
                s = posterior_samples(net, sino[:, meas], C.c.ANGLES_DEG[meas], n_samples=2, dc_iters=k, eta=eta)
                h, m = hidden_err(s.mean(0), sino, meas)
                h1, _ = hidden_err(s[0], sino, meas)
                rows.append(dict(worm=w, span=span, method="diffusion", dc_iters=k, eta=eta, hidden=h, measured=m,
                                 hidden_single=h1, seconds=time.time() - t0))
                print(rows[-1], flush=True)
            C.write_json(C.RESULTS / "tune.json", rows)
    # pick the setting with the lowest mean hidden-view error over poses and arcs
    keys = sorted({(r["dc_iters"], r["eta"]) for r in rows if r["method"] == "diffusion"})
    mean = {k: float(np.mean([r["hidden"] for r in rows if r.get("dc_iters") == k[0] and r.get("eta") == k[1]]))
            for k in keys}
    best = min(mean, key=mean.get)
    C.write_json(C.RESULTS / "tune.json", dict(rows=rows, mean_hidden={f"{k[0]}_{k[1]}": v for k, v in mean.items()},
                                               best=dict(dc_iters=best[0], eta=best[1])))
    print("best", best, mean, flush=True)


if __name__ == "__main__":
    main()
