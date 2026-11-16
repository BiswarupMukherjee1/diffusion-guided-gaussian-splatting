"""R2-Gaussian runs started from the diffusion reconstruction (all outputs in this project).

Arms, each on the nine test cases, with the R2-Gaussian settings of the earlier R2-Gaussian
initialisation study (50,000 Gaussians, 30,000 iterations, lambda_TV = 0.05):
  diff_fit    start = diffusion posterior mean, Gaussians fitted to it (as the U-Net "fitted" start)
  diff_guide  same start, plus diffusion guidance during training (r2_guided.py)
The FBP and fitted U-Net results of that study serve as reference points.

--source prior (default) takes the start from the unconditional prior (results/test, run_test.py);
--source cond_<arch> (e.g. cond_jit_xattn) takes it from the conditional model: posterior mean of
4 samples with data consistency, made here once per case, and the guidance refreshes use the same
model. Results go to results/r2 or results/r2_<source>.

  python r2_runs.py --arms diff_fit diff_guide [--source cond_jit_xattn] [--worker K --workers N]
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import core as C

PY = sys.executable
OUT = C.RESULTS / "r2"
SOURCE = "prior"
REFRESH = (2000, 5000, 10000, 15000, 20000)


def prefit(cdir, ply, target_zyx, ckpt, iters=3000):
    """Fit the 50,000 initial Gaussians to a starting volume (same procedure as the earlier study)."""
    import argparse as ap_
    cwd = os.getcwd(); os.chdir(C.R2_ROOT); sys.path.insert(0, str(C.R2_ROOT))
    import torch
    from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query
    p = ap_.ArgumentParser(); lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    args = p.parse_args([]); args.source_path = str(cdir); args.ply_path = str(ply)
    ma, oa, pa = lp.extract(args), op.extract(args), pp.extract(args)
    sc = Scene(ma, False).scanner_cfg
    g = GaussianModel(np.array([ma.scale_min, ma.scale_max]) * max(sc["sVoxel"]))
    initialize_gaussian(g, ma, None)
    g.training_setup(oa)
    tgt = torch.from_numpy(np.ascontiguousarray(np.transpose(target_zyx, (2, 1, 0)), np.float32)).cuda()
    for _ in range(iters):
        loss = (query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"] - tgt).abs().mean()
        g.optimizer.zero_grad(set_to_none=True); loss.backward(); g.optimizer.step()
    with torch.no_grad():
        vol = query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"].cpu().numpy()
    g.training_setup(oa)
    torch.save((g.capture(), 0), ckpt)
    os.chdir(cwd)
    return np.transpose(vol, (2, 1, 0))


def make_start(path, w, span):
    """Posterior mean and spread of the conditional model (4 samples, data consistency) for one case."""
    import time as _t
    import torch
    from sample_cond import load, sample
    sino, meas, _ = C.case_data(w, span)
    assert np.array_equal(meas, C.SETTINGS[f"arc{span}"]), "case views differ from the training setting"
    t0 = _t.time()
    s = sample(load(SOURCE[len("cond_"):]), sino, meas, n_samples=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=s.mean(0).astype(np.float32), std=s.std(0).astype(np.float16))
    C.write_json(path.with_suffix(".json"), dict(seconds=_t.time() - t0, scores=C.score(s.mean(0), w, span)))
    torch.cuda.empty_cache()


def run(arm, w, span, log):
    cdir = C.case_dir(w, span)
    rid = f"w{w:04d}_span{span}_{arm}"
    res = OUT / f"{rid}.json"
    if res.exists():
        return
    work = OUT / "work" / f"w{w:04d}_span{span}"; work.mkdir(parents=True, exist_ok=True)
    if SOURCE == "prior":
        dmean = C.RESULTS / "test" / f"w{w:04d}_span{span}.npz"
    else:
        dmean = OUT / "starts" / f"w{w:04d}_span{span}.npz"
    ply, ckpt = work / "init_diff.npy", work / "init_diff_fit.ckpt"
    with open(work / ".lock", "w") as lk:  # both arms of a case share one start and one fitted start
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not dmean.exists():
            make_start(dmean, w, span)
        vol = np.load(dmean)["mean"].astype(np.float32)
        if not ply.exists():
            pts, _ = C.L.sample_points(C.L.to_xyz(vol), cdir)
            np.save(ply, pts)
        if not ckpt.exists():
            fitted = prefit(cdir, ply, vol, ckpt)
            C.write_json(work / "prefit_scores.json", C.score(fitted, w, span))
    rdir = OUT / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", C.PROJECT / "src" / "r2_guided.py", "--source_path", cdir, "--model_path", rdir, "--ply_path", ply,
           "--iterations", 30000, "--lambda_tv", 0.05, "--run_record", rdir / "run_record.json", "--quiet",
           "--test_iterations", 1000, 2500, 5000, 10000, 15000, 20000, 25000, "--start_checkpoint", ckpt]
    if arm == "diff_guide":
        cmd += ["--guide_target", dmean, "--guide_case", w, span, "--guide_refresh_n", len(REFRESH),
                "--guide_refresh", *REFRESH]
        if SOURCE != "prior":
            cmd += ["--guide_cond", SOURCE[len("cond_"):], "--guide_steps", 20]
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        if subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT).returncode:
            raise RuntimeError(f"train failed {rid}")
        ev = [PY, "-u", C.SRC / "evaluate_run.py", "--run", rdir, "--case", cdir,
              "--iteration", 30000, "--out", res, "--cleanup"] + (["--keep-volume"] if w == 4 else [])
        if subprocess.run(list(map(str, ev)), stdout=f, stderr=subprocess.STDOUT,
                          cwd=C.SRC).returncode:
            raise RuntimeError(f"evaluate failed {rid}")
    r = json.loads(res.read_text())
    print(f"{rid}: worm PSNR {r['voxel']['worm_psnr_db']:.2f} hidden "
          f"{100 * r['projection']['proj_rel_l2_withheld']:.2f}% ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["diff_fit", "diff_guide"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source", default="prior")
    a = ap.parse_args()
    global OUT, SOURCE
    SOURCE = a.source
    OUT = C.RESULTS / ("r2" if SOURCE == "prior" else f"r2_{SOURCE}")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    jobs = [(arm, w, s) for s in (60, 90, 120) for w in C.TEST_POSES for arm in a.arms]
    for i, (arm, w, s) in enumerate(jobs):
        if i % a.workers != a.worker:
            continue
        try:
            run(arm, w, s, C.LOGS / f"r2_worker{a.worker}.log")
        except Exception as e:  # keep going; the failure is in the log
            print("FAIL", arm, w, s, e, flush=True)


if __name__ == "__main__":
    main()
