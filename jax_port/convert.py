"""Map a PyTorch state dict (exported by export_torch.py) onto the Flax UNet and check that both
give the same output, then run a few JAX training steps and a short DDIM sampling.

  .venv_jax/bin/python jax_port/convert.py jax_port/ref.npz
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from model_jax import UNet, cosine_alpha_bar, ddim_sample, make_train_step

NORMS = {"n1", "n2", "n", "out_norm"}


def flax_path(key, arr):
    p = key.split(".")
    leaf = p[-1]
    if p[0] == "temb":
        mods = (f"temb{p[1]}",)
    elif p[0] == "out":
        mods = ("out_norm" if p[1] == "0" else "out_conv",)
    elif p[0] == "inp":
        mods = ("inp",)
    elif p[0] == "mid":
        mods = (f"mid{p[1]}",) + tuple(p[2:-1])
    else:  # down / up
        top = f"{p[0]}_{p[1]}"
        if len(p) == 3:                                   # down-sampling conv: down.k.weight
            mods = (top,)
        elif len(p) == 4 and p[2] == "1" and p[0] == "up":  # up-sampling conv: up.k.1.weight
            mods = (top,)
        else:                                             # residual block (0) or attention (1)
            mods = (top, "res" if p[2] == "0" else "attn") + tuple(p[3:-1])
    if leaf == "weight":
        if mods[-1] in NORMS:
            return mods + ("scale",), arr
        if arr.ndim == 4:
            return mods + ("kernel",), arr.transpose(2, 3, 1, 0)  # (O, I, kh, kw) -> (kh, kw, I, O)
        return mods + ("kernel",), arr.T                        # (O, I) -> (I, O)
    return mods + ("bias",), arr


def main(ref_path):
    ref = np.load(ref_path)
    model = UNet()
    x = jnp.asarray(ref["x"].transpose(0, 2, 3, 1))
    t = jnp.asarray(ref["t"])
    init = flatten_dict(model.init(jax.random.PRNGKey(0), x, t)["params"])
    conv = dict(flax_path(k[2:], ref[k]) for k in ref.files if k.startswith("w/"))
    missing, extra = set(init) - set(conv), set(conv) - set(init)
    bad = [k for k in init if k in conv and init[k].shape != conv[k].shape]
    if missing or extra or bad:
        raise SystemExit(f"mismatch: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]} shape {bad[:5]}")
    params = unflatten_dict({k: jnp.asarray(v) for k, v in conv.items()})
    y = np.asarray(model.apply({"params": params}, x, t)).transpose(0, 3, 1, 2)
    err = np.abs(y - ref["y"]).max() / np.abs(ref["y"]).max()
    print(f"{len(conv)} tensors mapped; max |JAX - PyTorch| / max |PyTorch| = {err:.2e}")
    assert err < 1e-4, "outputs differ"
    # a few training steps on CPU (small batch) and a short sampling run
    ab = cosine_alpha_bar()
    tx, step = make_train_step(model, ab)
    opt_state = tx.init(params)
    key = jax.random.PRNGKey(1)
    xb = jnp.clip(jax.random.normal(key, (4, 160, 160, 1)) * 0.3, -1, 1)
    t0 = time.time()
    for i in range(3):
        key, k = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, xb, k)
        print(f"train step {i + 1}: loss {float(loss):.4f}")
    print(f"3 steps in {time.time() - t0:.1f}s (CPU, includes compilation)")
    s = ddim_sample(model, params, ab, jax.random.PRNGKey(2), 2, steps=5)
    print("DDIM sample", s.shape, "finite:", bool(jnp.isfinite(s).all()))


if __name__ == "__main__":
    main(sys.argv[1])
