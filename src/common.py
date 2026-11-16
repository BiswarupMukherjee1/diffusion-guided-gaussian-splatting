"""Data conventions and geometry helpers for the parallel-beam worm sinograms.

Conventions:
  * source TIFF axes (z, angle, detector) = (510, 100, 310); 100 views over [0, 180) deg;
    1 um voxels/detector pixels; rotation axis at the detector centre;
  * stored values are mg/cm^3 * um; we divide by 1000 -> g/cm^3 * um, so reconstructed
    densities are in g/cm^3 when voxels are 1 um;
  * empty slices (all zero) at the end of each stack are dropped;
  * TIGRE volumes are (z, y, x); TIGRE projections are (angle, v=z, u=detector);
  * R2-Gaussian stores volumes as (x, y, z) and projections as TIGRE output with v flipped.
"""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

N_VIEWS, STEP_DEG, N_DET = 100, 1.8, 310
ANGLES_DEG = np.arange(N_VIEWS) * STEP_DEG
# Fixed noise scale: RMS of the clean worm_004 z=255 sinogram (6210.1146 stored units),
# converted to g/cm^3*um.
NOISE_SCALE = 6.2101146
# TIGRE's parallel projector returns zero views at exactly 0/90 deg with a zero detector
# offset on even grids; a 0.01-pixel offset removes the degeneracy (mass error < 1%).
DET_EPS = 0.01
# Zip of the sinogram stacks (one TIFF stack per pose); set LACT_SINO_ZIP to its location.
DEFAULT_ZIP = Path(os.environ.get("LACT_SINO_ZIP", Path(__file__).resolve().parents[1] / "data" / "sino_selected.zip"))
# Slab centres as a fraction of each worm's non-empty length (mid = 255/415 for worm_004).
SLAB_POS = {"head": 0.15, "mid": 255 / 415, "tail": 0.85}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_stack(worm, zpath=DEFAULT_ZIP):
    """Return (sino[z, angle, det] in g/cm^3*um, float64; info dict). Reads from the zip."""
    import tifffile

    name = f"sino_selected/worm_{int(worm):03d}_gt.tiff" if str(worm).isdigit() else worm
    with zipfile.ZipFile(zpath) as zf:
        raw = zf.read(name)
    arr = tifffile.imread(io.BytesIO(raw))
    if arr.shape != (510, N_VIEWS, N_DET):
        raise ValueError(f"unexpected shape {arr.shape}")
    nonempty = np.flatnonzero(arr.reshape(arr.shape[0], -1).max(1) > 0)
    zl = int(nonempty[-1]) + 1
    if nonempty[0] != 0 or nonempty.size != zl:
        raise ValueError("non-contiguous content; empty-slice rule does not apply")
    info = dict(member=name, member_sha256=sha256_bytes(raw), n_nonempty=zl)
    return arr[:zl] / 1000.0, info


def slab_range(n_nonempty, position, nz=64):
    c = int(round(SLAB_POS[position] * n_nonempty)) if isinstance(position, str) else int(position)
    z0 = min(max(c - nz // 2, 0), n_nonempty - nz)
    return z0, z0 + nz


def arc_mask(start, span):
    """Measured-view mask for an arc [start, start+span) on the 180-deg periodic grid."""
    if not (0 <= start < 180 and 0 < span <= 180):
        raise ValueError("need 0 <= start < 180 and 0 < span <= 180")
    return ((ANGLES_DEG - start) % 180) < span - 1e-9


def add_noise(clean, f, seed):
    """Additive Gaussian noise, sigma = f * NOISE_SCALE, independent PCG64 stream per seed."""
    if f == 0:
        return clean.copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    return clean + rng.normal(0.0, f * NOISE_SCALE, clean.shape)


def tigre_geo(nz, n=N_DET, dvox=1.0, det_eps=DET_EPS):
    """Parallel-beam TIGRE geometry, lengths in um. Volume (nz, n/dvox... ) handled by caller."""
    import tigre

    nv = int(round(n / dvox))
    nzv = int(round(nz / dvox))
    geo = tigre.geometry(mode="parallel", nVoxel=np.array([nzv, nv, nv]))
    geo.DSD, geo.DSO = 1000.0, 500.0  # unused by the parallel projector
    geo.nDetector = np.array([nz, n])
    geo.dDetector = np.array([1.0, 1.0])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([nzv, nv, nv])
    geo.dVoxel = np.array([dvox, dvox, dvox])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.array([0.0, det_eps])
    geo.accuracy = 0.5 * min(dvox, 1.0)
    return geo


def to_tigre_proj(sino):
    """(z, angle, det) -> TIGRE (angle, z, det) float32."""
    return np.ascontiguousarray(np.transpose(sino, (1, 0, 2))).astype(np.float32)


def sirt(b, geo, ang, iters, x0=None, nonneg=True, lam=1.0):
    """Plain SIRT x += lam * C A^T R (b - A x) (TIGRE 2.3's own SIRT fails on numpy >= 1.24)."""
    import tigre

    R = 1.0 / np.maximum(tigre.Ax(np.ones(geo.nVoxel, np.float32), geo, ang), 1e-6)
    C = 1.0 / np.maximum(tigre.Atb(np.ones_like(b), geo, ang), 1e-6)
    x = np.zeros(geo.nVoxel, np.float32) if x0 is None else x0.astype(np.float32).copy()
    for _ in range(iters):
        x = x + lam * C * tigre.Atb(R * (b - tigre.Ax(x, geo, ang)), geo, ang)
        if nonneg:
            x = np.maximum(x, 0)
    return x


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def write_json(path, obj):
    """Atomic write: a failed write (e.g. disk quota) never truncates an existing file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    os.replace(tmp, path)
