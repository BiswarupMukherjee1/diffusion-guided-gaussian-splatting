"""JAX / Flax (linen) port of the diffusion prior (src/model.py), plus training step and sampler.

Layout is NHWC. Parameter names mirror the PyTorch modules so that convert.py can map a
PyTorch state dict onto this model one to one.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

T_STEPS = 1000


def cosine_alpha_bar(T=T_STEPS, s=0.008):
    t = jnp.arange(T + 1, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) / T
    f = jnp.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return jnp.clip((f / f[0])[1:], 1e-5, 1 - 1e-5).astype(jnp.float32)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)
    a = t.astype(jnp.float32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], 1)


def gn(ch, name):
    return nn.GroupNorm(num_groups=min(32, ch // 4), epsilon=1e-5, name=name)


class ResBlock(nn.Module):
    i: int
    o: int

    @nn.compact
    def __call__(self, x, te):
        h = nn.Conv(self.o, (3, 3), padding=1, name="c1")(nn.silu(gn(self.i, "n1")(x)))
        h = h + nn.Dense(self.o, name="t")(te)[:, None, None, :]
        h = nn.Conv(self.o, (3, 3), padding=1, name="c2")(nn.silu(gn(self.o, "n2")(h)))
        skip = nn.Conv(self.o, (1, 1), name="skip")(x) if self.i != self.o else x
        return h + skip


class Attention(nn.Module):
    ch: int
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        qkv = nn.Conv(3 * c, (1, 1), name="qkv")(gn(c, "n")(x))           # (b, h, w, 3c)
        qkv = qkv.reshape(b, h * w, 3, self.heads, c // self.heads)         # channel order as in PyTorch
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))   # (b, heads, hw, dh)
        a = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(c // self.heads), -1) @ v
        a = a.transpose(0, 2, 1, 3).reshape(b, h, w, c)
        return x + nn.Conv(c, (1, 1), name="out")(a)


class Pair(nn.Module):
    i: int
    o: int
    attn: bool

    @nn.compact
    def __call__(self, x, te):
        h = ResBlock(self.i, self.o, name="res")(x, te)
        return Attention(self.o, name="attn")(h) if self.attn else h


class UNet(nn.Module):
    ch: tuple = (32, 64, 128, 256)
    attn: tuple = (False, False, False, True)

    @nn.compact
    def __call__(self, x, t):
        ch, c0 = self.ch, self.ch[0]
        te = nn.Dense(4 * c0, name="temb0")(timestep_embedding(t, c0))
        te = nn.Dense(4 * c0, name="temb2")(nn.silu(te))
        h = nn.Conv(c0, (3, 3), padding=1, name="inp")(x)
        hs, skips, c, k = [h], [c0], c0, 0
        for i, o in enumerate(ch):
            for _ in range(2):
                h = Pair(c, o, self.attn[i], name=f"down_{k}")(h, te); k += 1
                c = o; hs.append(h); skips.append(c)
            if i < len(ch) - 1:
                h = nn.Conv(c, (3, 3), strides=2, padding=1, name=f"down_{k}")(h); k += 1
                hs.append(h); skips.append(c)
        h = ResBlock(c, c, name="mid0")(h, te)
        h = Attention(c, name="mid1")(h)
        h = ResBlock(c, c, name="mid2")(h, te)
        k = 0
        for i, o in reversed(list(enumerate(ch))):
            for _ in range(3):
                h = Pair(c + skips.pop(), o, self.attn[i], name=f"up_{k}")(jnp.concatenate([h, hs.pop()], -1), te)
                k += 1; c = o
            if i > 0:
                h = jnp.repeat(jnp.repeat(h, 2, 1), 2, 2)
                h = nn.Conv(c, (3, 3), padding=1, name=f"up_{k}")(h); k += 1
        return nn.Conv(1, (3, 3), padding=1, name="out_conv")(nn.silu(gn(c, "out_norm")(h)))


def make_train_step(model, ab, lr=2e-4):
    """One DDPM training step (noise prediction, MSE), as in src/train_prior.py."""
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.0))

    def loss_fn(params, x0, key):
        k1, k2 = jax.random.split(key)
        t = jax.random.randint(k1, (x0.shape[0],), 1, T_STEPS + 1)
        a = ab[t - 1][:, None, None, None]
        eps = jax.random.normal(k2, x0.shape)
        xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps
        return jnp.mean((model.apply({"params": params}, xt, t) - eps) ** 2)

    @jax.jit
    def step(params, opt_state, x0, key):
        loss, g = jax.value_and_grad(loss_fn)(params, x0, key)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    return tx, step


def ddim_sample(model, params, ab, key, n, steps=50, size=160):
    """Deterministic DDIM sampling (eta = 0), as train_prior.ddim_sample."""
    ts = [int(round(v)) for v in jnp.linspace(T_STEPS, 1, steps).tolist()]
    x = jax.random.normal(key, (n, size, size, 1))
    apply = jax.jit(lambda x, t: model.apply({"params": params}, x, t))
    x0 = x
    for i, t in enumerate(ts):
        a = ab[t - 1]
        eps = apply(x, jnp.full((n,), t))
        x0 = jnp.clip((x - jnp.sqrt(1 - a) * eps) / jnp.sqrt(a), -1, 1)
        a_prev = ab[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0
        x = jnp.sqrt(a_prev) * x0 + jnp.sqrt(1 - a_prev) * eps
    return x0
