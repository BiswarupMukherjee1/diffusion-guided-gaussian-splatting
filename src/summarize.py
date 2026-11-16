"""Markdown tables of every result so far (results/SUMMARY.md), averaged over the test poses that exist.

Tables: (1) single-volume reconstructions per acquisition setting (SIRT, supervised U-Net start,
unconditional prior with CGLS, conditional models with and without data consistency); (2) the
uncertainty scores of the sampled methods; (3) the steps sweep; (4) R2-Gaussian runs started from
the conditional model, next to the R2-Gaussian runs of an earlier initialisation study on the same data, if available.

  python summarize.py
"""

import csv
import glob
import json
import re
from collections import defaultdict

import numpy as np

import core as C

COND = C.RESULTS / "cond"
SETTINGS = list(C.SETTINGS)
METHODS = [("sirt200", "SIRT, 200 it. (measured views only)"),
           ("unet_start", "Supervised U-Net on SIRT (earlier study start)"),
           ("uncond_cgls", "Unconditional DDPM prior + CGLS data consistency"),
           ("cond_unet_nodc", "Conditional U-Net, no data consistency"),
           ("cond_unet_dc", "Conditional U-Net + data consistency"),
           ("cond_jit_nodc", "JiT, no data consistency"),
           ("cond_jit_dc", "JiT + data consistency"),
           ("cond_jit_xattn_nodc", "JiT + view attention, no data consistency"),
           ("cond_jit_xattn_dc", "JiT + view attention + data consistency")]


def load(prefix, setting):
    rows = []
    for f in glob.glob(str(COND / f"{prefix}_w*_{setting}.json")):
        if re.fullmatch(rf"{prefix}_w\d{{4}}_{setting}", f.split("/")[-1][:-5]):
            rows.append(json.loads(open(f).read()))
    return rows


def fmt(vals, scale=1.0, nd=2):
    vals = [v for v in vals if v is not None]
    return f"{scale * np.mean(vals):.{nd}f}" if vals else "–"


def table_main():
    out = []
    for key, label, scale, nd in (("worm_psnr_db", "Worm PSNR (dB, higher is better)", 1, 2),
                                  ("hidden", "Error on unmeasured views (relative L2, %, lower is better)", 100, 2),
                                  ("measured", "Error on measured views (relative L2, %)", 100, 2),
                                  ("outline_f1", "Outline F-score at 1 µm", 1, 3)):
        out.append(f"\n**{label}** (mean over test poses; n = number of poses)\n")
        out.append("| Method | " + " | ".join(SETTINGS) + " |")
        out.append("|---" * (len(SETTINGS) + 1) + "|")
        for prefix, name in METHODS:
            cells = []
            for s in SETTINGS:
                rows = load(prefix, s)
                cells.append(f"{fmt([r['mean'][key] for r in rows], scale, nd)} (n={len(rows)})" if rows else "–")
            if any(c != "–" for c in cells):
                out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_uncertainty():
    out = ["\n**Does the spread across samples point at the errors?** Spearman rank correlation between the "
           "per-voxel standard deviation and the absolute error of the posterior mean, inside the worm "
           "(1 = perfect ranking, 0 = none), mean over poses.\n",
           "| Method | " + " | ".join(SETTINGS) + " |", "|---" * (len(SETTINGS) + 1) + "|"]
    for prefix, name in METHODS:
        cells = []
        for s in SETTINGS:
            rows = [r for r in load(prefix, s) if "uncertainty" in r]
            cells.append(fmt([r["uncertainty"]["spearman"] for r in rows]) if rows else "–")
        if any(c != "–" for c in cells):
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_sweep():
    rows = defaultdict(list)
    for f in glob.glob(str(COND / "sweep_*.json")):
        m = re.search(r"sweep_(\w+?)_K(\d+)_w\d{4}_(\w+)\.json", f)
        r = json.loads(open(f).read())
        rows[(m.group(1), m.group(3), int(m.group(2)))].append(r)
    if not rows:
        return "\n(steps sweep not run yet)"
    out = ["\n| Model | Setting | Steps | Worm PSNR (dB) | Unmeasured-view error (%) | Seconds per sample | n |",
           "|---|---|---|---|---|---|---|"]
    for (arch, s, k) in sorted(rows):
        r = rows[(arch, s, k)]
        out.append(f"| {arch} | {s} | {k} | {fmt([x['mean']['worm_psnr_db'] for x in r])} | "
                   f"{fmt([x['mean']['hidden'] for x in r], 100)} | {fmt([x['seconds_per_sample'] for x in r], 1, 1)} | {len(r)} |")
    return "\n".join(out)


def table_r2():
    ref = defaultdict(list)
    f = C.RUNS_WORM / "init_study" / "analysis" / "all_runs.csv"  # optional: R2-Gaussian runs of the earlier study
    for r in (csv.DictReader(open(f)) if f.exists() else []):
        if r["init"] in ("fdk", "unet", "unet_fit") and int(r["worm"]) in C.TEST_POSES and r["seed"] == "0":
            ref[(r["init"], int(r["span"]))].append(r)
    out = ["\n| Start of R²-Gaussian | Arc | Start PSNR (dB) | Final worm PSNR (dB) | Final unmeasured-view error (%) | n |",
           "|---|---|---|---|---|---|"]
    names = {"fdk": "FBP (native R²-Gaussian)", "unet": "Supervised U-Net, points drawn from it",
             "unet_fit": "Supervised U-Net, Gaussians fitted to it"}
    for init in ("fdk", "unet", "unet_fit"):
        for span in (120, 90, 60):
            r = ref[(init, span)]
            if r:
                out.append(f"| {names[init]} (earlier study) | {span}° | {fmt([float(x['init_psnr']) for x in r])} | "
                           f"{fmt([float(x['worm_psnr_db']) for x in r])} | "
                           f"{fmt([float(x['proj_rel_l2_withheld']) for x in r], 100)} | {len(r)} |")
    d = C.RESULTS / "r2_cond_jit_xattn"
    for arm, name in (("diff_fit", "Conditional diffusion, Gaussians fitted to it"),
                      ("diff_guide", "Conditional diffusion, fitted + diffusion guidance during training")):
        for span in (120, 90, 60):
            rs, st = [], []
            for w in C.TEST_POSES:
                p = d / f"w{w:04d}_span{span}_{arm}.json"
                if p.exists():
                    rs.append(json.loads(p.read_text()))
                    s = d / "starts" / f"w{w:04d}_span{span}.json"
                    if s.exists():
                        st.append(json.loads(s.read_text())["scores"]["voxel"]["worm_psnr_db"])
            if rs:
                out.append(f"| {name} | {span}° | {fmt(st)} | {fmt([x['voxel']['worm_psnr_db'] for x in rs])} | "
                           f"{fmt([x['projection']['proj_rel_l2_withheld'] for x in rs], 100)} | {len(rs)} |")
    return "\n".join(out)


def main():
    text = ["# Results summary (generated by src/summarize.py)\n", "## Single-volume reconstructions", table_main(),
            "\n## Uncertainty", table_uncertainty(), "\n## Number of sampling steps", table_sweep(),
            "\n## Diffusion + R²-Gaussian", table_r2()]
    (C.RESULTS / "SUMMARY.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
