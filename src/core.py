"""Paths and shared helpers.

Environment variables:
  LACT_SINO_ZIP     zip of the sinogram stacks (see common.py)
  LACT_WORK         folder with the test cases, phantoms and the U-Net baseline (cases/, phantoms/, init_study/)
  R2_GAUSSIAN_ROOT  clone of R2-Gaussian with patches/r2_gaussian_parallel_beam.patch applied
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
R2_ROOT = Path(os.environ.get("R2_GAUSSIAN_ROOT", PROJECT / "third_party" / "r2_gaussian"))
RUNS_WORM = Path(os.environ.get("LACT_WORK", PROJECT / "work"))
UNET_DIR = RUNS_WORM / "init_study" / "unet"  # pose exclusion record and validation split of the U-Net baseline

sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache" / "matplotlib"))
import init_lib as L  # noqa: E402
c = L.c

DATA, RUNS, RESULTS, LOGS = (PROJECT / d for d in ("data", "runs", "results", "logs"))
CROP = 160          # diffusion works on 160 x 160 crops centred on the worm (worm extent <= 133 x 73 px)
SCALE = 0.35        # density (g/cm^3) -> model space: x / SCALE - 1, so 0 -> -1 and 0.7 -> +1
TEST_POSES = (4, 1031, 8916)
SPANS = (120, 90, 60)


def kept_poses():
    """Poses allowed for training (same exclusion as the U-Net study) and its validation poses."""
    index = json.loads((UNET_DIR / "index.json").read_text())
    kept = [r["worm"] for r in index if not r["excluded"]]
    val = json.loads((UNET_DIR / "models" / "unet_span90_seed0.json").read_text())["val_poses"]
    return [w for w in kept if w not in val], list(val)


def to_model(x):
    return x / SCALE - 1.0


def to_density(x):
    return (x + 1.0) * SCALE


def centroid_from_sino(sino, ang_deg, n=c.N_DET):
    """Centre of mass (y, x) in pixels of each slice, from its projections alone.

    For parallel beam the centre of mass of a projection is the projection of the slice's centre
    of mass, so two or more angles determine it exactly. sino: (z, n_views, det)."""
    th = np.deg2rad(ang_deg)
    u = np.arange(n) - (n - 1) / 2
    m = sino.sum(-1)
    uc = (sino * u).sum(-1) / np.maximum(m, 1e-9)  # (z, views)
    A = np.stack([np.cos(th), -np.sin(th)], 1)  # u_c = cos(th) y - sin(th) x (TIGRE convention, checked to 0.2 px)
    sol, *_ = np.linalg.lstsq(A, uc.T, rcond=None)  # (2, z)
    return sol.T + (n - 1) / 2, m.mean(1) > 1e-6


def crop(img, cy, cx, size=CROP):
    """Crop (size x size) around (cy, cx) with zero padding; returns crop and its top-left corner."""
    y0, x0 = int(round(cy)) - size // 2, int(round(cx)) - size // 2
    out = np.zeros((size, size), img.dtype)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, img.shape[0]), min(x0 + size, img.shape[1])
    out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out, (y0, x0)


def paste(crop_img, corner, shape):
    """Inverse of crop: place a crop back into a zero image of the given shape."""
    y0, x0 = corner
    size = crop_img.shape[-1]
    out = np.zeros(crop_img.shape[:-2] + tuple(shape), crop_img.dtype)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, shape[0]), min(x0 + size, shape[1])
    out[..., ys:ye, xs:xe] = crop_img[..., ys - y0:ye - y0, xs - x0:xe - x0]
    return out


def write_json(path, obj):
    c.write_json(path, obj)


def case_dir(worm, span):
    return RUNS_WORM / "cases" / f"A_w{worm:04d}_mid_span{span}_start0_f0_ns0"


def case_data(worm, span):
    """Supplied sinogram of the test block, measured-view mask and the case record (read-only)."""
    rec = json.loads((case_dir(worm, span) / "case_record.json").read_text())
    stack, _ = c.load_stack(worm)
    sino = stack[rec["z_range"][0]:rec["z_range"][1]]
    meas = np.zeros(c.N_VIEWS, bool)
    meas[rec["measured_view_indices"]] = True
    return sino, meas, rec


def score(vol_zyx, worm, span):
    """Same scores as the R2-Gaussian runs (evaluate_run.py), for a (z, y, x) volume."""
    from evaluate_run import evaluate, projection_metrics
    sino, meas, rec = case_data(worm, span)
    pdir = Path(rec["phantom_dir"])
    prep = json.loads((pdir / "phantom_report.json").read_text())
    ph = np.load(pdir / "phantom.npz")
    vol = np.ascontiguousarray(vol_zyx, np.float32)
    vm, _ = evaluate(vol, ph["truth"], ph["labels"], ph["pure"], np.array(prep["class_edges_g_cm3"]),
                     prep["class_density_g_cm3"])
    pm = projection_metrics(vol, sino, sino, rec["measured_view_indices"])
    pm.pop("per_view_rel_l2", None)
    return dict(voxel=vm, projection=pm)


# ---------------------------------------------------------------- acquisition settings (conditional models)
def _sparse(k):
    m = np.zeros(c.N_VIEWS, bool)
    m[::c.N_VIEWS // k] = True
    return m


SETTINGS = {"arc120": c.arc_mask(0, 120), "arc90": c.arc_mask(0, 90), "arc60": c.arc_mask(0, 60),
            "sparse20": _sparse(20), "sparse10": _sparse(10)}
TOKEN_LEN = 224  # detector samples kept per view around the crop centre (covers the 160 px crop diagonal)


def view_tokens(sino, cen, n=c.N_DET, L=TOKEN_LEN):
    """Re-centre every projection on the crop centre, so the tokens move with the crop.

    sino (z, views, det), cen (z, 2) centre (y, x) in pixels -> (z, views, L). With the TIGRE
    convention u = cos(th) y - sin(th) x, the crop centre projects to u_c on each view."""
    th = np.deg2rad(c.ANGLES_DEG[:sino.shape[1]])
    c0 = (n - 1) / 2
    u_c = np.cos(th)[None] * (cen[:, :1] - c0) - np.sin(th)[None] * (cen[:, 1:] - c0)  # (z, views)
    grid = np.arange(L) - (L - 1) / 2
    out = np.zeros(sino.shape[:2] + (L,), np.float32)
    det = np.arange(n)
    for i in range(sino.shape[0]):
        for v in range(sino.shape[1]):
            out[i, v] = np.interp(u_c[i, v] + c0 + grid, det, sino[i, v], left=0.0, right=0.0)
    return out


def setting_data(worm, setting):
    """Supplied sinogram of the mid test block of a pose and the measured-view mask of a setting.
    The block (z range) and reference are those of the test cases."""
    rec = json.loads((case_dir(worm, 120) / "case_record.json").read_text())
    stack, _ = c.load_stack(worm)
    return stack[rec["z_range"][0]:rec["z_range"][1]], SETTINGS[setting].copy(), rec


def score_mask(vol_zyx, worm, meas):
    """Scores of score() for any measured-view mask (the test block and reference of the pose)."""
    from evaluate_run import evaluate, projection_metrics
    rec = json.loads((case_dir(worm, 120) / "case_record.json").read_text())
    stack, _ = c.load_stack(worm)
    sino = stack[rec["z_range"][0]:rec["z_range"][1]]
    pdir = Path(rec["phantom_dir"])
    prep = json.loads((pdir / "phantom_report.json").read_text())
    ph = np.load(pdir / "phantom.npz")
    vol = np.ascontiguousarray(vol_zyx, np.float32)
    vm, _ = evaluate(vol, ph["truth"], ph["labels"], ph["pure"], np.array(prep["class_edges_g_cm3"]),
                     prep["class_density_g_cm3"])
    pm = projection_metrics(vol, sino, sino, np.flatnonzero(meas))
    pm.pop("per_view_rel_l2", None)
    return dict(voxel=vm, projection=pm), ph["truth"]
