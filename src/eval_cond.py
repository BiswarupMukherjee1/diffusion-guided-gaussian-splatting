"""Evaluation of the conditional models on the three test poses and five acquisition settings.

For each (pose, setting): baselines (SIRT-200 from the measured views; the U-Net start of the
earlier study for the arcs it was trained on; the unconditional prior with CGLS data consistency),
then each trained conditional model with and without data consistency, 4 samples each. Scores:
the voxel scores of evaluate_run.py and errors on measured and hidden views for the posterior mean, the same
for a single sample, and how well the spread across samples ranks the actual error.

  python eval_cond.py                     # all trained models, writes results/cond/*.json
  python eval_cond.py --sweep jit_xattn   # number-of-steps sweep (speed against quality)
"""

import argparse
import json
import time

import numpy as np
import torch

import core as C
from run_test import uncertainty_scores

OUT = C.RESULTS / "cond"
N = 4


def summary(sc):
    v, p = sc["voxel"], sc["projection"]
    return dict(worm_psnr_db=v["worm_psnr_db"], interior_cv_pct=v["interior_cv_pct_wmean"],
                density_bias_pct=v["abs_rel_bias_pct_wmean"], outline_f1=v["worm_bf1_1um"],
                hidden=p["proj_rel_l2_withheld"], measured=p["proj_rel_l2_measured"])


def done_training(arch):
    f = C.RUNS / f"cond_{arch}" / "log.json"
    cfg = C.RUNS / f"cond_{arch}" / "config.json"
    return f.exists() and cfg.exists() and json.loads(f.read_text())[-1]["step"] >= json.loads(cfg.read_text())["steps"]


def write(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT / f"{name}.json", res)


def run_samples(name, fn, w, setting, truth, keep):
    if (OUT / f"{name}.json").exists():
        return
    t0 = time.time()
    s = fn()
    sec = (time.time() - t0) / len(s)
    mean, std = s.mean(0), s.std(0)
    meas = C.SETTINGS[setting]
    res = dict(worm=w, setting=setting, n_samples=len(s), seconds_per_sample=sec,
               mean=summary(C.score_mask(mean, w, meas)[0]), single=summary(C.score_mask(s[0], w, meas)[0]),
               uncertainty=uncertainty_scores(std, mean - truth, truth > 0))
    if keep:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"{name}.npz", mean=mean.astype(np.float16), std=std.astype(np.float16))
    write(name, res)
    m = res["mean"]
    print(f"{name}: PSNR {m['worm_psnr_db']:.2f} hidden {100 * m['hidden']:.2f}% measured {100 * m['measured']:.2f}% "
          f"spearman {res['uncertainty']['spearman']:.2f} ({sec:.0f}s/sample)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=None)
    a = ap.parse_args()
    from sample_cond import load, sample
    if a.sweep:
        net = load(a.sweep)
        for w in C.TEST_POSES:
            for setting in ("arc60", "sparse10"):
                sino, meas, _ = C.setting_data(w, setting)
                truth = C.score_mask(np.zeros((64, 310, 310), np.float32), w, meas)[1]
                for k in (1, 2, 4, 8, 16, 50):
                    run_samples(f"sweep_{a.sweep}_K{k}_w{w:04d}_{setting}",
                                lambda: sample(net, sino, meas, n_samples=2, steps=k), w, setting, truth, False)
        return
    archs = [x for x in ("unet", "jit", "jit_xattn") if done_training(x)]
    nets = {x: load(x) for x in archs}
    for w in C.TEST_POSES:
        for setting in C.SETTINGS:
            sino, meas, _ = C.setting_data(w, setting)
            truth = C.score_mask(np.zeros((64, 310, 310), np.float32), w, meas)[1]
            tag = f"w{w:04d}_{setting}"
            if not (OUT / f"sirt200_{tag}.json").exists():
                v = C.L.sirt_slab(sino[:, meas], C.c.ANGLES_DEG[meas], 200)
                write(f"sirt200_{tag}", dict(worm=w, setting=setting, mean=summary(C.score_mask(v, w, meas)[0])))
            span = {"arc120": 120, "arc90": 90, "arc60": 60}.get(setting)
            if span and not (OUT / f"unet_start_{tag}.json").exists():
                v = np.load(C.case_dir(w, span) / "initvol_unet.npz")["vol"].astype(np.float32)
                write(f"unet_start_{tag}", dict(worm=w, setting=setting, mean=summary(C.score_mask(v, w, meas)[0])))
            for arch, net in nets.items():
                for dc in (True, False):
                    run_samples(f"cond_{arch}_{'dc' if dc else 'nodc'}_{tag}",
                                lambda: sample(net, sino, meas, n_samples=N, dc_iters=10 if dc else 0,
                                               final_iters=10 if dc else 0),
                                w, setting, truth, keep=(w == 4 and dc))
            torch.cuda.empty_cache()
    from sample import load_prior, posterior_samples  # unconditional prior, CGLS data consistency
    prior = load_prior()
    for w in C.TEST_POSES:
        for setting in C.SETTINGS:
            sino, meas, _ = C.setting_data(w, setting)
            truth = C.score_mask(np.zeros((64, 310, 310), np.float32), w, meas)[1]
            run_samples(f"uncond_cgls_w{w:04d}_{setting}",
                        lambda: posterior_samples(prior, sino[:, meas], C.c.ANGLES_DEG[meas], n_samples=2, dc_iters=10,
                                                  eta=0.3, final_iters=30, dc_mode="cgls"),
                        w, setting, truth, False)


if __name__ == "__main__":
    main()
