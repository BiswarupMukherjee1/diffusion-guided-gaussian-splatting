"""Diffusion posterior sampling on the nine test cases (3 poses x 120/90/60 degree arcs).

Uses the settings chosen by tune.py. For each case: 8 samples, their mean (the
reconstruction) and per-voxel standard deviation (the uncertainty map), both saved as
float16, and the same scores as the R2-Gaussian runs for the mean and for a single
sample. Also scores how well the spread ranks the actual error inside the worm.

  python run_test.py            # writes results/test/*.json, *.npz
"""

import json
import time

import numpy as np

import core as C
from sample import load_prior, posterior_samples

N_SAMPLES = 8


def uncertainty_scores(std, err, worm):
    """How well the spread ranks the error: Spearman correlation and the share of the 10% worst
    voxels (by error) that are also among the 10% most uncertain."""
    from scipy.stats import spearmanr
    s, e = std[worm], np.abs(err[worm])
    top_e, top_s = e >= np.quantile(e, 0.9), s >= np.quantile(s, 0.9)
    return dict(spearman=float(spearmanr(s, e).correlation), top10_overlap=float((top_e & top_s).sum() / top_e.sum()),
                mean_std=float(s.mean()), mean_abs_err=float(e.mean()))


def main():
    best = json.loads((C.RESULTS / "tune.json").read_text())["best"]
    net = load_prior()
    out = C.RESULTS / "test"; out.mkdir(parents=True, exist_ok=True)
    for w in C.TEST_POSES:
        for span in C.SPANS:
            f = out / f"w{w:04d}_span{span}.json"
            if f.exists():
                continue
            sino, meas, rec = C.case_data(w, span)
            t0 = time.time()
            s = posterior_samples(net, sino[:, meas], C.c.ANGLES_DEG[meas], n_samples=N_SAMPLES, **best)
            sec = time.time() - t0
            mean, std = s.mean(0), s.std(0)
            truth = np.load(np.lib.npyio.os.path.join(rec["phantom_dir"], "phantom.npz"))["truth"]
            res = dict(worm=w, span=span, settings=best, n_samples=N_SAMPLES, seconds=sec,
                       mean=C.score(mean, w, span), single=C.score(s[0], w, span),
                       uncertainty=uncertainty_scores(std, mean - truth, truth > 0))
            np.savez_compressed(out / f"w{w:04d}_span{span}.npz", mean=mean.astype(np.float16), std=std.astype(np.float16))
            C.write_json(f, res)
            v, p = res["mean"]["voxel"], res["mean"]["projection"]
            print(f"w{w} {span}: PSNR {v['worm_psnr_db']:.2f} hidden {100 * p['proj_rel_l2_withheld']:.2f}% "
                  f"measured {100 * p['proj_rel_l2_measured']:.2f}% spearman {res['uncertainty']['spearman']:.2f} "
                  f"({sec:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
