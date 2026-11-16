"""Build one R2-Gaussian dataset (Blender format) from a worm slab.

Track A: projections are the supplied sinogram rows.
Track B: projections are simulated from the digital phantom (exact voxel truth).
Measured views = the arc [start, start+span) on the 1.8-deg grid, with additive Gaussian
noise drawn on the full 100-view grid (so arcs share the same noise realisation).
Test views = the withheld views, clean (all 100 clean views for a full 180-deg scan).
vol_gt = the phantom truth (exact for Track B, an approximate reference for Track A).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import common as c


def case_id(track, worm, slab, span, start, noise, noise_seed):
    return f"{track}_w{worm:04d}_{slab}_span{int(span)}_start{int(start)}_f{noise:g}_ns{noise_seed}"


def build(track, worm, slab, span, start, noise, noise_seed, phantoms, out_root):
    cid = case_id(track, worm, slab, span, start, noise, noise_seed)
    out = Path(out_root) / cid
    if (out / "meta_data.json").exists():
        return out, json.loads((out / "case_record.json").read_text())
    pdir = Path(phantoms) / f"worm{worm:04d}_{slab}"
    rep = json.loads((pdir / "phantom_report.json").read_text())
    truth = np.load(pdir / "phantom.npz")["truth"]  # (z, y, x)
    nz = truth.shape[0]
    if track == "A":
        stack, info = c.load_stack(worm)
        z0, z1 = rep["z_range"]
        if info["member_sha256"] != rep["source"]["member_sha256"]:
            raise ValueError("source changed since the phantom was built")
        clean = stack[z0:z1]  # (z, angle, det)
    elif track == "B":
        clean = np.load(pdir / "sino_sim_clean.npz")["sino"].astype(np.float64)
    else:
        raise ValueError(track)
    noisy = c.add_noise(clean, noise, noise_seed)
    meas = c.arc_mask(start, span)
    test_idx = np.flatnonzero(~meas) if (~meas).any() else np.arange(c.N_VIEWS)
    ang = np.deg2rad(c.ANGLES_DEG)
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(
        scanner=dict(
            mode="parallel", filter=None, DSD=1500.0, DSO=1000.0,
            nDetector=[nz, c.N_DET], sDetector=[float(nz), float(c.N_DET)],
            nVoxel=[c.N_DET, c.N_DET, nz], sVoxel=[float(c.N_DET), float(c.N_DET), float(nz)],
            offOrigin=[0.0, 0.0, 0.0], offDetector=[c.DET_EPS, 0.0], accuracy=0.5,
            totalAngle=float(span), startAngle=float(start), noise=False,
        ),
        vol="vol_gt.npy",
        bbox=[[-1, -1, -nz / c.N_DET], [1, 1, nz / c.N_DET]],
        proj_train=[], proj_test=[],
    )
    for split, idx, src in (("proj_train", np.flatnonzero(meas), noisy), ("proj_test", test_idx, clean)):
        (out / split).mkdir()
        for j, i in enumerate(idx):
            fn = f"{split}/{split}_{j:04d}.npy"
            # TIGRE projection (v=z, u=det) stored with v flipped, as the R2 generator does
            np.save(out / fn, np.ascontiguousarray(src[:, i, :][::-1, :]).astype(np.float32))
            meta[split].append(dict(file_path=fn, angle=float(ang[i]), view_index=int(i)))
    np.save(out / "vol_gt.npy", np.ascontiguousarray(np.transpose(truth, (2, 1, 0))).astype(np.float32))
    (out / "meta_data.json").write_text(json.dumps(meta, indent=1))
    rec = dict(case_id=cid, track=track, worm=worm, slab=slab, z_range=rep["z_range"], span=span,
               start=start, noise_fraction=noise, noise_sigma=noise * c.NOISE_SCALE,
               noise_scale_definition="RMS of clean worm_004 z=255 sinogram, g/cm^3*um",
               noise_seed=noise_seed, n_measured=int(meas.sum()), n_test=int(len(test_idx)),
               measured_view_indices=np.flatnonzero(meas).tolist(),
               test_is_withheld=bool((~meas).any()),
               phantom_dir=str(pdir), phantom_sha256=c.sha256_file(pdir / "phantom.npz"))
    c.write_json(out / "case_record.json", rec)
    return out, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=("A", "B"), required=True)
    ap.add_argument("--worm", type=int, required=True)
    ap.add_argument("--slab", default="mid")
    ap.add_argument("--span", type=float, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--phantoms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out, rec = build(a.track, a.worm, a.slab, a.span, a.start, a.noise, a.noise_seed, a.phantoms, a.out)
    print(out)
    print(json.dumps({k: v for k, v in rec.items() if k != "measured_view_indices"}, indent=1))


if __name__ == "__main__":
    main()
