"""Shared helpers for the initialisation study (worm, parallel beam, R2-Gaussian).

Every initialiser produces a starting volume in g/cm^3 on the 1-um case grid, and the
Gaussians are then drawn from it with exactly the native rule of `initialize_pcd.py`
(voxels above 0.05 g/cm^3, 50,000 drawn uniformly, density x 0.15). Only the starting
volume (and, for the uncertainty variant, where part of the points go) changes.

Layouts: physical slabs are TIGRE order (z, y, x); R2 volumes (vol_gt.npy, recon_volume)
are (x, y, z) = transpose(2, 1, 0). Case projections are stored with v (= z) flipped.
Scene scaling multiplies lengths and projections by the same factor, so densities in the
scaled scene equal physical densities.
"""

import json
from pathlib import Path

import numpy as np

import common as c

N_POINTS, DENSITY_THRESH, DENSITY_RESCALE = 50000, 0.05, 0.15


def load_measured(case_dir):
    """Measured sinogram of a case in physical units: (z, n_meas, det), angles in degrees."""
    case_dir = Path(case_dir)
    meta = json.loads((case_dir / "meta_data.json").read_text())
    projs = [np.load(case_dir / p["file_path"])[::-1, :] for p in meta["proj_train"]]
    ang = np.array([np.rad2deg(p["angle"]) for p in meta["proj_train"]])
    return np.ascontiguousarray(np.stack(projs, 1)).astype(np.float64), ang


def sirt_slab(sino, ang_deg, iters, x0=None):
    """Non-negative SIRT of a slab sinogram (z, n, det) -> (z, y, x) in g/cm^3."""
    geo = c.tigre_geo(sino.shape[0])
    return c.sirt(c.to_tigre_proj(sino), geo, np.deg2rad(ang_deg), iters, x0=x0)


def to_xyz(slab):
    return np.ascontiguousarray(np.transpose(slab, (2, 1, 0))).astype(np.float32)


def scene_box(case_dir):
    sc = json.loads((Path(case_dir) / "meta_data.json").read_text())["scanner"]
    nV, sV = np.array(sc["nVoxel"], float), np.array(sc["sVoxel"], float)
    s = 2 / sV.max()
    return nV, sV * s, np.array(sc["offOrigin"], float) * s


def sample_points(vol_xyz, case_dir, seed=0, extra=None, extra_frac=0.0):
    """Native sampling rule. With `extra` (a per-voxel score) a fraction `extra_frac` of the
    points is drawn from the highest-scoring eligible voxels instead (uncertainty variant)."""
    nV, sVs, off = scene_box(case_dir)
    dV = sVs / nV
    rng = np.random.default_rng(seed)
    valid = np.argwhere(vol_xyz > DENSITY_THRESH)
    if len(valid) < N_POINTS:
        raise ValueError(f"only {len(valid)} voxels above threshold")
    n_extra = int(round(extra_frac * N_POINTS)) if extra is not None else 0
    if n_extra:
        sc = extra[valid[:, 0], valid[:, 1], valid[:, 2]]
        top = np.argsort(sc)[::-1][: n_extra * 4]  # pool = top 4x, drawn uniformly
        pick_x = rng.choice(top, n_extra, replace=False)
        rest = np.setdiff1d(np.arange(len(valid)), pick_x)
        pick = np.concatenate([pick_x, rng.choice(rest, N_POINTS - n_extra, replace=False)])
    else:
        pick = rng.choice(len(valid), N_POINTS, replace=False)
    idx = valid[pick]
    pos = idx * dV - sVs / 2 + off
    dens = vol_xyz[idx[:, 0], idx[:, 1], idx[:, 2]] * DENSITY_RESCALE
    return np.concatenate([pos, dens[:, None]], -1).astype(np.float64), idx


def init_diagnostics(vol_xyz, idx, case_dir):
    """How good is the starting point itself? Scored against the case reference volume."""
    gt = np.load(Path(case_dir) / "vol_gt.npy")
    worm = gt > 0
    err = (vol_xyz - gt)[worm]
    return dict(
        init_worm_psnr_db=float(10 * np.log10(gt.max() ** 2 / (err**2).mean())),
        init_corr=float(np.corrcoef(vol_xyz.ravel(), gt.ravel())[0, 1]),
        voxels_above_thresh=int((vol_xyz > DENSITY_THRESH).sum()),
        worm_voxels=int(worm.sum()),
        points_inside_worm_pct=float(100 * worm[idx[:, 0], idx[:, 1], idx[:, 2]].mean()),
        mass_ratio=float(vol_xyz.sum() / gt.sum()),
    )
