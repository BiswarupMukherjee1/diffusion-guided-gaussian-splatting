"""Training images for the diffusion prior.

For every pose allowed by the U-Net study's exclusion rule (test poses, the other reference
poses and near-duplicates of the test poses are left out), reconstruct every 4th non-empty
slice with 300 iterations of non-negative SIRT from all 100 views, and keep a 160 x 160 crop
centred on the worm's centre of mass (computed from the projections). The prior therefore
learns what full-angle worm slices look like; it never sees a test pose.

  python build_data.py            # writes data/train.npz, data/val.npz
"""

import time

import numpy as np

import core as C

STRIDE, ITERS, CHUNK, MIN_MASS = 4, 300, 64, 50.0


def pose_slices(w):
    stack, _ = C.c.load_stack(w)  # (z, views, det), physical units
    mass = stack.sum((1, 2)) / stack.shape[1]
    zs = [z for z in range(0, stack.shape[0], STRIDE) if mass[z] > MIN_MASS]
    crops = []
    for i in range(0, len(zs), CHUNK):
        zz = zs[i:i + CHUNK]
        sino = stack[zz]
        img = C.L.sirt_slab(sino, C.c.ANGLES_DEG, ITERS)
        cen, _ = C.centroid_from_sino(sino, C.c.ANGLES_DEG)
        crops += [C.crop(img[k], *cen[k])[0] for k in range(len(zz))]
    return np.stack(crops).astype(np.float16), np.array(zs)


def main():
    train, val = C.kept_poses()
    C.DATA.mkdir(exist_ok=True)
    for name, poses in (("val", val), ("train", train)):
        X, P, Z = [], [], []
        for w in poses:
            t0 = time.time()
            x, zs = pose_slices(w)
            X.append(x); P += [w] * len(zs); Z += list(zs)
            print(f"{name} worm {w}: {len(zs)} slices, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(C.DATA / f"{name}.npz", x=np.concatenate(X), pose=np.array(P), z=np.array(Z))
        print(name, "done", sum(len(x) for x in X), "slices", flush=True)


if __name__ == "__main__":
    main()
