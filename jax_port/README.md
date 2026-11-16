# JAX / Flax port of the diffusion prior

* `model_jax.py`: the U-Net of `src/model.py` in Flax (linen, NHWC), a DDPM training step
  (optax AdamW, gradient clipping) and a DDIM sampler.
* `export_torch.py` (PyTorch environment): exports a prior's weights and a reference output.
* `convert.py` (JAX environment, `.venv_jax`, CPU-only jaxlib): maps the 308 PyTorch tensors
  onto the Flax model, checks that both give the same output, runs 3 training steps and a
  short DDIM sampling.

Checked so far (9 Nov 2026), with randomly initialised weights: relative difference between
the JAX and PyTorch outputs 3.0e-6; training loss decreases over 3 steps; sampling runs.
Not yet done: the same check with the trained weights, and any training in JAX on a GPU
(the JAX environment here is CPU-only).

    python jax_port/export_torch.py --weights runs/prior/ema.pt --out jax_port/ref_trained.npz
    cd jax_port && JAX_PLATFORMS=cpu ../.venv_jax/bin/python convert.py ref_trained.npz
