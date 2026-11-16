"""Evaluate one finished R2 run on a worm case (densities in g/cm^3, lengths in um).

Projection domain (valid for both tracks): reproject the final volume on all 100 angles
and compare with the clean sinogram on measured and on withheld views.
Voxel domain (exact for Track B; approximate reference for Track A), all restricted to
the worm, never the empty box:
  * contours    - Dice of worm/air and of each density class (same class thresholds as
                  the phantom), mean/95% surface distance, boundary F-score (1, 2 um);
  * homogeneity - std / CV inside each class after a 1-um erosion (truth is constant);
  * quantitative- mean bias per class (g/cm^3 and %), MAE/RMSE/PSNR inside the worm;
  * streaks     - RMS density in air >= 3 um away from the worm;
  * diagnostics - high-gradient boundary MAE (Week 3 definition) and gradient error by
                  in-plane edge orientation (Week 2 analysis).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

import common as c


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def boundary_scores(gt_mask, pr_mask):
    sg, sp = surface(gt_mask), surface(pr_mask)
    if not sp.any():
        return dict(mean_surface_dist_um=float("nan"), hd95_um=float("nan"), bf1_1um=0.0, bf1_2um=0.0)
    dg = ndimage.distance_transform_edt(~sg)
    dp = ndimage.distance_transform_edt(~sp)
    d_pg, d_gp = dg[sp], dp[sg]
    out = dict(mean_surface_dist_um=float(0.5 * (d_pg.mean() + d_gp.mean())),
               hd95_um=float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))))
    for tol in (1, 2):
        prec, rec = float((d_pg <= tol).mean()), float((d_gp <= tol).mean())
        out[f"bf1_{tol}um"] = 2 * prec * rec / max(prec + rec, 1e-12)
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def evaluate(pred, truth, labels, pure, edges, levels):
    worm = labels > 0
    ev = ndimage.binary_dilation(worm, iterations=2)
    err = pred - truth
    peak = float(truth.max())
    m = dict()
    m["worm_mae"] = float(np.abs(err[ev]).mean())
    m["worm_rmse"] = float(np.sqrt((err[ev] ** 2).mean()))
    m["worm_psnr_db"] = float(10 * np.log10(peak**2 / (err[ev] ** 2).mean()))
    m["box_psnr_peak1_db"] = float(10 * np.log10(1.0 / (err**2).mean()))  # R2 native definition
    lab_pred = np.digitize(pred, edges).astype(np.uint8)
    m["dice_worm"] = dice(worm, lab_pred > 0)
    m["dice_class"] = [dice(labels == k, lab_pred == k) for k in range(1, len(levels) + 1)]
    m["dice_class_mean"] = float(np.nanmean(m["dice_class"]))
    m.update({f"worm_{k}": v for k, v in boundary_scores(worm, lab_pred > 0).items()})
    # compartment boundaries: any label change
    def comp_edges(lab):
        e = np.zeros(lab.shape, bool)
        for ax in range(3):
            d = np.diff(lab.astype(np.int16), axis=ax) != 0
            sl = [slice(None)] * 3
            sl[ax] = slice(0, -1)
            e[tuple(sl)] |= d
        return e
    eg, ep = comp_edges(labels), comp_edges(lab_pred)
    dg, dp = ndimage.distance_transform_edt(~eg), ndimage.distance_transform_edt(~ep)
    for tol in (1, 2):
        prec = float((dg[ep] <= tol).mean()) if ep.any() else 0.0
        rec = float((dp[eg] <= tol).mean())
        m[f"compartment_bf1_{tol}um"] = 2 * prec * rec / max(prec + rec, 1e-12)
    # homogeneity and bias per class
    cls = []
    for k in range(1, len(levels) + 1):
        inner = ndimage.binary_erosion((labels == k) & pure, iterations=1)
        if inner.sum() < 20:
            cls.append(dict(k=k, n=int(inner.sum())))
            continue
        v = pred[inner]
        t = float(truth[inner].mean())
        cls.append(dict(k=k, n=int(inner.sum()), true=t, mean=float(v.mean()), bias=float(v.mean() - t),
                        rel_bias_pct=float(100 * (v.mean() - t) / t), std=float(v.std()),
                        cv_pct=float(100 * v.std() / max(v.mean(), 1e-9))))
    m["classes"] = cls
    ok = [d for d in cls if "std" in d]
    w = np.array([d["n"] for d in ok], float)
    m["interior_std_wmean"] = float(np.average([d["std"] for d in ok], weights=w))
    m["interior_cv_pct_wmean"] = float(np.average([d["cv_pct"] for d in ok], weights=w))
    m["abs_rel_bias_pct_wmean"] = float(np.average([abs(d["rel_bias_pct"]) for d in ok], weights=w))
    air = ndimage.distance_transform_edt(~worm) >= 3
    m["air_rms"] = float(np.sqrt((pred[air] ** 2).mean()))
    # Week 3 high-gradient boundary diagnostic (gradient on the 1-um grid)
    gt_g = np.stack(np.gradient(truth.astype(np.float64)))
    pr_g = np.stack(np.gradient(pred.astype(np.float64)))
    mag = np.sqrt((gt_g**2).sum(0))
    tau = np.percentile(mag[mag > 0], 90)
    B = mag >= tau
    m["highgrad_boundary_mae"] = float(np.abs(err[B]).mean())
    m["highgrad_gradient_err"] = float(np.sqrt(((pr_g - gt_g) ** 2).sum(0))[B].mean())
    # gradient error by in-plane edge-normal orientation (0..180 deg, 12 bins), axes (z, y, x)
    psi = np.degrees(np.arctan2(gt_g[1], gt_g[2])) % 180
    inplane = B & (np.hypot(gt_g[1], gt_g[2]) > 0.7 * mag)
    ge = np.sqrt(((pr_g - gt_g) ** 2).sum(0))
    bins = np.minimum((psi // 15).astype(int), 11)
    m["edge_err_by_normal_deg"] = {
        f"{15 * b}-{15 * b + 15}": float(ge[inplane & (bins == b)].mean() / mag[inplane & (bins == b)].mean())
        for b in range(12) if (inplane & (bins == b)).sum() > 50
    }
    return m, lab_pred


def projection_metrics(pred, clean, noisy, meas_idx, z_local=None):
    import tigre

    nz = pred.shape[0]
    g = c.tigre_geo(nz)
    ang = np.deg2rad(c.ANGLES_DEG).astype(np.float32)
    p = tigre.Ax(np.ascontiguousarray(pred, dtype=np.float32), g, ang)  # (angle, z, det)
    cl = c.to_tigre_proj(clean)
    meas = np.zeros(c.N_VIEWS, bool)
    meas[meas_idx] = True
    per_view = [float(np.linalg.norm(p[i] - cl[i]) / np.linalg.norm(cl[i])) for i in range(c.N_VIEWS)]
    out = dict(proj_rel_l2_measured=c.rel_l2(p[meas], cl[meas]), per_view_rel_l2=per_view)
    out["proj_rel_l2_withheld"] = c.rel_l2(p[~meas], cl[~meas]) if (~meas).any() else None
    nz_ = c.to_tigre_proj(noisy)[meas]
    out["residual_to_noisy_measured_rms"] = float(np.sqrt(((p[meas] - nz_) ** 2).mean()))
    if z_local is not None:
        s = p[:, z_local, :]
        sc = cl[:, z_local, :]
        out["slice"] = dict(z_local=int(z_local), measured=c.rel_l2(s[meas], sc[meas]),
                            withheld=c.rel_l2(s[~meas], sc[~meas]) if (~meas).any() else None)
    return out


def figure(pred, truth, levels, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = truth.shape[0]
    zs = [nz // 8, nz // 2, 7 * nz // 8]
    worm = truth > 0
    ys, xs = np.nonzero(worm.any(0))
    y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, truth.shape[1])
    x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, truth.shape[2])
    vmax = 1.1 * float(truth.max())
    fig, ax = plt.subplots(3, 4, figsize=(16, 11.5))
    for i, z in enumerate(zs):
        a0 = ax[i, 0].imshow(truth[z, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=vmax)
        ax[i, 1].imshow(pred[z, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=vmax)
        a2 = ax[i, 2].imshow((pred - truth)[z, y0:y1, x0:x1], cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        ax[i, 0].set_ylabel(f"slice z={z}")
        yy = int(np.argmax(worm[z].sum(1)))
        ax[i, 1].axhline(yy - y0, color="y", lw=0.7)
        ax[i, 3].plot(np.arange(x0, x1), truth[z, yy, x0:x1], "k", label="truth")
        ax[i, 3].plot(np.arange(x0, x1), pred[z, yy, x0:x1], "r", label="reconstruction")
        ax[i, 3].set_ylim(-0.05, vmax)
        ax[i, 3].grid(alpha=0.3)
        ax[i, 3].set_xlabel("x (µm)")
        ax[i, 3].set_ylabel("density (g/cm³)")
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].plot([3, 23], [y1 - y0 - 4] * 2, "w", lw=2)  # 20 um scale bar
    ax[0, 0].set_title("truth (phantom)"); ax[0, 1].set_title("reconstruction")
    ax[0, 2].set_title("signed error (recon − truth)"); ax[0, 3].set_title("profile along yellow line")
    ax[0, 3].legend(fontsize=8)
    fig.colorbar(a0, ax=ax[:, :2].ravel().tolist(), shrink=0.5, label="density (g/cm³)")
    fig.colorbar(a2, ax=ax[:, 2].ravel().tolist(), shrink=0.5, label="error (g/cm³)")
    fig.suptitle(title + "   (white bar = 20 µm)")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-volume", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="delete bulky native outputs after success")
    a = ap.parse_args()
    rec = json.loads((a.case / "case_record.json").read_text())
    pdir = Path(rec["phantom_dir"])
    prep = json.loads((pdir / "phantom_report.json").read_text())
    ph = np.load(pdir / "phantom.npz")
    truth, labels, pure = ph["truth"], ph["labels"], ph["pure"]
    pc = a.run / "point_cloud" / f"iteration_{a.iteration}"
    pred = np.ascontiguousarray(np.transpose(np.load(pc / "vol_pred.npy"), (2, 1, 0)))
    if rec["track"] == "A":
        stack, info = c.load_stack(rec["worm"])
        clean = stack[rec["z_range"][0]:rec["z_range"][1]]
    else:
        clean = np.load(pdir / "sino_sim_clean.npz")["sino"].astype(np.float64)
    noisy = c.add_noise(clean, rec["noise_fraction"], rec["noise_seed"])
    z_local = 255 - rec["z_range"][0] if rec["worm"] == 4 and rec["z_range"][0] <= 255 < rec["z_range"][1] else None
    res = dict(run=a.run.name, case=rec["case_id"], iteration=a.iteration,
               **{k: rec[k] for k in ("track", "worm", "slab", "span", "start", "noise_fraction", "noise_seed",
                                       "n_measured")})
    for f in ("run_record.json", "run_spec.json"):
        if (a.run / f).exists():
            res[f[:-5]] = json.loads((a.run / f).read_text())
    nat = a.run / "eval" / f"iter_{a.iteration:06d}" / "eval3d.yml"
    if nat.exists():
        import yaml

        res["native_eval3d"] = yaml.safe_load(nat.read_text())
    curves = {}
    for d in sorted((a.run / "eval").glob("iter_*")):
        try:
            import yaml

            curves[int(d.name[5:])] = yaml.safe_load((d / "eval3d.yml").read_text())["psnr_3d"]
        except Exception:
            pass
    res["native_psnr3d_curve"] = curves
    res["projection"] = projection_metrics(pred, clean, noisy, rec["measured_view_indices"], z_local)
    vm, lab_pred = evaluate(pred, truth, labels, pure, np.array(prep["class_edges_g_cm3"]),
                            prep["class_density_g_cm3"])
    res["voxel"] = vm
    res["voxel_reference"] = "exact phantom truth" if rec["track"] == "B" else "phantom fitted to the supplied data (approximate)"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(a.out, res)
    if a.figure:
        figure(pred, truth, prep["class_density_g_cm3"], a.out.with_suffix(".png"),
               f"{a.run.name}: worm PSNR {vm['worm_psnr_db']:.2f} dB, Dice {vm['dice_class_mean']:.3f}, "
               f"withheld-view error {100 * (res['projection']['proj_rel_l2_withheld'] or 0):.2f}%")
    if a.keep_volume:
        np.savez_compressed(a.out.with_suffix(".vol.npz"), pred=pred.astype(np.float16))
    if a.cleanup:
        for f in ("vol_pred.npy", "vol_gt.npy"):
            (pc / f).unlink(missing_ok=True)
        if not a.keep_volume:
            (pc / "point_cloud.pickle").unlink(missing_ok=True)
        for d in (a.run / "eval").glob("iter_*"):
            for f in d.glob("*.png"):
                f.unlink()
    print(json.dumps({k: res[k] for k in ("run", "case")}, indent=0),
          f"worm_psnr={vm['worm_psnr_db']:.3f} dice={vm['dice_class_mean']:.3f} "
          f"withheld={res['projection']['proj_rel_l2_withheld']}")


if __name__ == "__main__":
    main()
