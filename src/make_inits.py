"""Build the starting point clouds of the initialisation study for one or more cases.

Physics-only starting volumes (Experiment 1):
  sirt    non-negative SIRT (200 iterations) of the measured views.
Learned starting volumes (Experiment 2, need trained models, see unet_init.py):
  unet    mean of the U-Net ensemble applied slice by slice to the SIRT volume;
  unetuq  same volume, 30% of the points placed in the voxels where the ensemble
          disagrees most (uncertainty-guided placement).
Writes <case>/init_<method>.npy (the R2 point-cloud format), the starting volume as
float16 (<case>/initvol_<method>.npz) and diagnostics of the starting point.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import init_lib as L

SIRT_ITERS = 200


def build_case(cdir, methods, models=None, overwrite=False):
    cdir = Path(cdir)
    rec = json.loads((cdir / "case_record.json").read_text())
    sino, ang = L.load_measured(cdir)
    out = {}
    sirt = None
    for m in methods:
        f = cdir / f"init_{m}.npy"
        if f.exists() and not overwrite:
            continue
        t0 = time.time()
        extra = None
        if m in ("sirt", "unet", "unetuq"):
            if sirt is None:
                vf = cdir / "initvol_sirt.npz"
                sirt = (np.load(vf)["vol"].astype(np.float32) if vf.exists()
                        else L.sirt_slab(sino, ang, SIRT_ITERS).astype(np.float32))
            slab = sirt
        if m in ("unet", "unetuq"):
            import unet_init

            mean, std = unet_init.predict(models[int(rec["span"])], sirt)
            slab = mean
            if m == "unetuq":
                extra = L.to_xyz(std)
        vol = L.to_xyz(slab)
        pts, idx = L.sample_points(vol, cdir, extra=extra, extra_frac=0.3 if extra is not None else 0.0)
        np.save(f, pts)
        np.savez_compressed(cdir / f"initvol_{m}.npz", vol=np.asarray(slab, np.float16))
        d = dict(method=m, seconds=time.time() - t0, **L.init_diagnostics(vol, idx, cdir))
        if extra is not None:
            d["uq_mean_std_inside_worm"] = float(extra[np.load(cdir / "vol_gt.npy") > 0].mean())
        L.c.write_json(cdir / f"init_{m}_diag.json", d)
        out[m] = d
        print(cdir.name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
    return out


def fdk_diagnostics(cdir):
    """Same diagnostics for the native FDK start (volume recomputed, points from init.npy)."""
    import argparse as ap_
    import copy
    import os

    REPO = Path(os.environ["R2_GAUSSIAN_ROOT"])
    os.chdir(REPO); sys.path.insert(0, str(REPO))
    from r2_gaussian.arguments import ModelParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume
    from r2_gaussian.utils.general_utils import t2a

    cdir = Path(cdir)
    p = ap_.ArgumentParser(); lp = ModelParams(p); args = p.parse_args([])
    args.source_path = str(cdir)
    sc = Scene(lp.extract(args), False)
    cams = sc.getTrainCameras()
    projs = np.concatenate([t2a(cm.original_image) for cm in cams])
    ang = np.stack([t2a(cm.angle) for cm in cams])
    vol = recon_volume(projs, ang, copy.deepcopy(get_geometry_tigre(sc.scanner_cfg)), "fdk").astype(np.float32)
    nV, sVs, off = L.scene_box(cdir)
    pts = np.load(cdir / "init.npy")
    idx = np.clip(np.round((pts[:, :3] + sVs / 2 - off) / (sVs / nV)).astype(int), 0, nV.astype(int) - 1)
    d = dict(method="fdk", **L.init_diagnostics(vol, idx, cdir))
    L.c.write_json(cdir / "init_fdk_diag.json", d)
    np.savez_compressed(cdir / "initvol_fdk.npz", vol=np.transpose(vol, (2, 1, 0)).astype(np.float16))
    print(cdir.name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", type=Path)
    ap.add_argument("--methods", nargs="+", default=["sirt"])
    ap.add_argument("--models", type=Path, help="directory with trained U-Net ensembles")
    ap.add_argument("--fdk-diag", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    models = None
    if any(m.startswith("unet") for m in a.methods):
        import unet_init

        models = {s: unet_init.load_ensemble(a.models, s) for s in (120, 90, 60)}
    for cdir in a.cases:
        if a.fdk_diag:
            fdk_diagnostics(cdir)
        else:
            build_case(cdir, a.methods, models, a.overwrite)


if __name__ == "__main__":
    main()
