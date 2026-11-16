"""Conditioning data for the conditional diffusion models.

For every training and validation slice of data/train.npz and data/val.npz (same order), and for
each acquisition setting in core.SETTINGS (arcs of 120, 90, 60 degrees from 0, and 20 or 10
evenly spaced views over 180 degrees), a 200-iteration non-negative SIRT from the measured views
only, cropped like the target. Also the 100 projections of each slice re-centred on the crop
(core.view_tokens), from which a model can take the measured views.

  python build_cond.py      # writes data/cond_train.npz, data/cond_val.npz
"""

import time

import numpy as np

import core as C


def main():
    for split in ("val", "train"):
        out = C.DATA / f"cond_{split}.npz"
        if out.exists():
            continue
        d = np.load(C.DATA / f"{split}.npz")
        P, Z = d["pose"], d["z"]
        arrs = {k: [] for k in C.SETTINGS}
        toks = []
        for w in dict.fromkeys(P.tolist()):  # poses in file order
            t0 = time.time()
            zs = Z[P == w]
            stack, _ = C.c.load_stack(int(w))
            sino = stack[zs]
            cen, _ = C.centroid_from_sino(sino, C.c.ANGLES_DEG)
            for name, m in C.SETTINGS.items():
                img = C.L.sirt_slab(sino[:, m], C.c.ANGLES_DEG[m], 200)
                arrs[name].append(np.stack([C.crop(img[i], *cen[i])[0] for i in range(len(zs))]).astype(np.float16))
            toks.append(C.view_tokens(sino, cen).astype(np.float16))
            print(f"{split} worm {w}: {len(zs)} slices, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out, tokens=np.concatenate(toks), **{k: np.concatenate(v) for k, v in arrs.items()})
        print(split, "done", flush=True)


if __name__ == "__main__":
    main()
