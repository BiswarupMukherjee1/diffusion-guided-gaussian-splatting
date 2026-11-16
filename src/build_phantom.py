"""Track B: digital worm phantom with exact voxel truth for one pose and slab.

1. Reconstruct the slab from all 100 clean supplied views (nonnegative SIRT).
2. Upsample 2x (0.5 um), median-filter, split into density classes (multi-Otsu on the
   worm's values; background below half the lowest class).
3. Fit one density per class by least squares so the phantom's projections match the
   supplied sinogram (projections simulated at 0.5 um with a 0.5 um detector, then
   binned 2x2 to the real 1 um pixels: simulation and reconstruction never share a grid).
4. Save the 1 um truth (2x2x2 average of the fine phantom), 1 um class labels, the clean
   simulated 100-view sinogram, and a report of how well the phantom reproduces the
   supplied data.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tigre
from scipy import ndimage

import common as c


def project_fine(vol_fine, nz, ang):
    """Project a 0.5 um volume with a 0.5 um detector and bin 2x2 -> (angle, nz, 310)."""
    g = c.tigre_geo(2 * nz, n=2 * c.N_DET, dvox=1.0)
    # geometry with 0.5 um voxels and 0.5 um detector pixels: scale every length by 0.5
    g.dVoxel = g.dVoxel * 0.5
    g.sVoxel = g.nVoxel * g.dVoxel
    g.dDetector = g.dDetector * 0.5
    g.sDetector = g.nDetector * g.dDetector
    g.offDetector = np.array([0.0, c.DET_EPS * 0.5])
    g.accuracy = 0.25
    p = tigre.Ax(np.ascontiguousarray(vol_fine, dtype=np.float32), g, ang)
    return p.reshape(p.shape[0], nz, 2, c.N_DET, 2).mean((2, 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worm", type=int, required=True)
    ap.add_argument("--slab", default="mid")
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--classes", type=int, default=5, help="number of worm density levels")
    ap.add_argument("--sirt-iters", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out / f"worm{a.worm:04d}_{a.slab}"
    if out.exists():
        raise SystemExit(f"{out} exists; phantoms are never overwritten")
    t0 = time.time()
    stack, info = c.load_stack(a.worm)
    z0, z1 = c.slab_range(info["n_nonempty"], a.slab, a.nz)
    sino = stack[z0:z1]  # (z, angle, det), g/cm^3*um
    b = c.to_tigre_proj(sino)
    ang = np.deg2rad(c.ANGLES_DEG).astype(np.float32)
    g1 = c.tigre_geo(a.nz)
    rec = c.sirt(b, g1, ang, a.sirt_iters)
    rec_res = c.rel_l2(tigre.Ax(rec, g1, ang), b)

    fine = ndimage.zoom(rec, 2, order=1, prefilter=False)
    fine = ndimage.median_filter(fine, size=3)
    # density levels: 1-D k-means on the worm's interior values (above 0.2 g/cm^3, which
    # excludes blurred boundary voxels); class thresholds at midpoints between levels and
    # the worm/air boundary at half the lowest level (half-maximum edge position)
    vals = fine[fine > 0.2].astype(np.float64)
    lev = np.quantile(vals, (np.arange(a.classes) + 0.5) / a.classes)
    for _ in range(100):
        idx = np.abs(vals[:, None] - lev[None]).argmin(1)
        lev = np.array([vals[idx == k].mean() for k in range(a.classes)])
    lev.sort()
    edges = np.concatenate([[0.5 * lev[0]], 0.5 * (lev[1:] + lev[:-1])])
    lab_fine = np.digitize(fine, edges).astype(np.uint8)  # 0 = air
    n_cls = int(lab_fine.max())

    # least-squares density per class against the supplied sinogram
    P = [project_fine((lab_fine == k).astype(np.float32), a.nz, ang) for k in range(1, n_cls + 1)]
    M = np.stack([p.ravel() for p in P], 1).astype(np.float64)
    rho, *_ = np.linalg.lstsq(M, b.ravel().astype(np.float64), rcond=None)
    phantom_fine = np.zeros(fine.shape, np.float32)
    for k in range(1, n_cls + 1):
        phantom_fine[lab_fine == k] = rho[k - 1]
    sim = sum(r * p for r, p in zip(rho, P)).astype(np.float32)  # clean simulated views
    truth = phantom_fine.reshape(a.nz, 2, c.N_DET, 2, c.N_DET, 2).mean((1, 3, 5))
    # 1 um labels: majority class of the 8 sub-voxels
    lab1 = np.zeros(truth.shape, np.uint8)
    counts = np.stack([(lab_fine == k).reshape(a.nz, 2, c.N_DET, 2, c.N_DET, 2).sum((1, 3, 5))
                       for k in range(n_cls + 1)])
    lab1 = counts.argmax(0).astype(np.uint8)
    pure = counts.max(0) == 8  # voxel fully inside one class (no partial volume)

    rep = dict(
        worm=a.worm, slab=a.slab, z_range=[z0, z1], source=info,
        sirt_iters=a.sirt_iters, sirt_reprojection_residual=rec_res,
        class_edges_g_cm3=edges.tolist(), kmeans_levels_g_cm3=lev.tolist(), class_density_g_cm3=rho.tolist(),
        class_voxels_1um=[int((lab1 == k).sum()) for k in range(n_cls + 1)],
        phantom_vs_data_sino_rel_l2=c.rel_l2(sim, b),
        phantom_vs_data_sino_rel_l2_foreground=c.rel_l2(sim[b > 0], b[b > 0]),
        truth_range=[float(truth.min()), float(truth.max())],
        pure_voxel_fraction_in_worm=float(pure[lab1 > 0].mean()),
        units="density g/cm^3; projections g/cm^3*um; voxel 1 um; TIGRE order (z,y,x)",
        seconds=time.time() - t0,
    )
    out.mkdir(parents=True)
    np.savez_compressed(out / "phantom.npz", truth=truth.astype(np.float32), labels=lab1,
                        pure=pure, sirt_reference=rec.astype(np.float16))
    np.savez_compressed(out / "sino_sim_clean.npz", sino=np.transpose(sim, (1, 0, 2)).astype(np.float32),
                        angles_deg=c.ANGLES_DEG)
    c.write_json(out / "phantom_report.json", rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
