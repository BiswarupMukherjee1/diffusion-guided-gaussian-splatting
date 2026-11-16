# Diffusion priors for limited-angle and sparse-view micro-CT, and their use inside 3D Gaussian splatting

This repository studies how far a learned generative prior can fill in what a tomographic scan does not
measure, and whether it can do so without inventing structure. It compares an unconditional diffusion prior
with posterior sampling, three conditional flow-matching models (a U-Net and two pixel-space transformers,
one of them with a geometry-aware cross-attention to the measured projections), physics-based data
consistency during sampling, and the use of the diffusion output as the start and as a guidance term for
R²-Gaussian, a 3D Gaussian splatting method for CT.

**Status (16 November 2026).** The conditional transformers are trained and evaluated on the three test
poses for all five acquisition settings. The unconditional baseline and the step-count sweep are done; the conditional U-Net and the
R²-Gaussian runs are in progress. Every number below says how many poses it is based on.
Sections marked *pending* will be filled in when those runs finish.

---

## 1. Problem

In computed tomography the object is reconstructed from its projections. Two acquisition shortcuts make the
problem ill-posed:

* **Limited angle.** Projections are recorded only over an arc (here 120°, 90° or 60° out of the 180° that a
  parallel-beam scan needs). Fourier components perpendicular to the missing directions are not measured at all
  (the "missing wedge"), so edges in those directions blur and densities smear along the arc.
* **Sparse view.** The full 180° is covered, but with few projections (here 20 or 10 out of 100). Every
  direction is seen, but coarsely, and classical reconstructions show streaks.

Classical regularisation (SIRT, total variation) suppresses noise but cannot recreate unmeasured
information. A learned prior can, which is both its value and its danger: a model that has seen many worms
will draw a plausible worm whether or not the data support it. In a medical or biological setting that is a
hallucination. The questions here are therefore not only *how good* the reconstruction is, but:

1. Is it consistent with the projections that **were** measured?
2. How wrong is it on projections that were **not** measured? (These are held out and only used for scoring.)
3. When the model is uncertain, does its sample spread say so, i.e. does the spread point at the errors?

A second motivation comes from earlier work on the same data with R²-Gaussian (a radiative 3D Gaussian
splatting method for CT). There, starting the Gaussians from a good learned reconstruction did not survive
training: a supervised U-Net start at 23.7–25.5 dB ended at 14–20 dB after 30,000 iterations, roughly where
a plain FBP start ends (table in section 5.4). The Gaussians fit the measured views and lose the prior
information about the unmeasured ones. A diffusion model that can repeatedly re-impose that information during
training is a natural candidate to stop this drift.

## 2. Data and evaluation protocol

* **Data.** Parallel-beam sinograms of a worm in many poses, provided by the supervising lab: 100 views
  equally spaced over [0°, 180°), 310 detector pixels of 1 µm, 510 slices per pose, values converted to
  g/cm³·µm. No noise is added in this study. The data are not part of this repository.
* **Splits.** The pose exclusion and the validation split of the earlier U-Net study are reused unchanged, so
  that the supervised baseline and the diffusion models never see the test poses or anything close to them.
  Training: 115 poses, 11,839 slices (every fourth non-empty slice). Validation: 12 poses, 1,268 slices.
  Test: poses 4, 1031 and 8916, the middle block of 64 slices.
* **Training targets.** Full-angle SIRT (300 iterations) of each slice, cropped to 160 × 160 pixels around
  the worm. The crop centre is computed from the projections themselves (centre of mass of each view, solved
  for the 2-D position), so it is available at test time without a reconstruction.
* **Acquisition settings.** `arc120`, `arc90`, `arc60` (arcs starting at 0°) and `sparse20`, `sparse10`
  (every 5th or every 10th view over 180°).
* **Scores** (the evaluation code of the earlier R²-Gaussian study, unchanged):
  * *Worm PSNR*: PSNR inside the worm against a reference volume. The reference is a phantom fitted to the
    supplied data and is approximate, so small PSNR differences should not be over-read.
  * *Error on unmeasured views*: relative L2 error between the projections of the reconstruction and the
    supplied projections at the views that were withheld. This uses only the data and is the most reliable
    hallucination check available here.
  * *Error on measured views*: the same on the views that were used.
  * *Outline F-score* at 1 µm, density bias and interior coefficient of variation per tissue class.
  * *Uncertainty quality*: Spearman rank correlation between the per-voxel standard deviation over samples and
    the absolute error of the posterior mean, inside the worm.

## 3. Methods

### 3.1 Baselines

* **SIRT-200**: non-negative SIRT on the measured views only (TIGRE projector).
* **Supervised U-Net start**: the 2-D U-Net ensemble of the earlier study, trained on (limited-arc SIRT →
  full-arc SIRT) pairs, applied to the test SIRT volume. It exists for the three arcs only.

### 3.2 Track 1: unconditional diffusion prior with posterior sampling

* **Prior.** A DDPM (noise prediction, cosine noise schedule, 1,000 steps) on single slices: U-Net with
  channels (32, 64, 128, 256), self-attention at the 20 × 20 resolution, 15.5 M parameters, EMA weights,
  40,000 steps at batch 32. Final validation loss 0.0057. Augmentation: the eight flips/rotations of the square
  plus random shifts.
* **Posterior sampling** in the style of the Decomposed Diffusion Sampler (DDS): at every reverse step the
  network's estimate of the clean slice is pasted into the full 310 × 310 slice, corrected by a few iterations
  of a data-consistency solver on the measured views, and re-noised to the next level. Two solvers were tried:
  SIRT, and conjugate gradients on the normal equations (CGLS). Both update only within the range of Aᵀ, so
  the part of the slice that the measured views cannot see is left to the prior. The CGLS adjoint pair was
  checked numerically (⟨Ax, y⟩ / ⟨x, Aᵀy⟩ = 1.00002).
* **Tuning** (4 validation poses, by error on unmeasured views): 10 data-consistency iterations per step,
  η = 0.3 (DDIM stochasticity).

### 3.3 Track 2: conditional flow-matching models

The unconditional prior has to discover the acquisition from the data-consistency step alone. The
conditional models are instead told what was measured. Each training example draws one of the five settings
at random; the network receives

* the noisy slice x_t,
* the SIRT-200 reconstruction of the same slice from that setting's measured views (a second input channel),
* the 100-dimensional mask of measured views, embedded by a small MLP and added to the time embedding,

and predicts the clean slice. No flips or rotations are used, because they would change the acquisition
geometry the network is told about.

**Training objective.** Flow matching on the straight path x_t = (1 − t) x + t ε with t drawn from a
logit-normal distribution. Following JiT (Li & He, CVPR 2026), the network predicts the clean image
(x-prediction) and the loss is written in velocity space, ‖x̂ − x‖² / t², with t clamped at 0.05 in the
denominator. AdamW, learning rate 1e-4 (transformers) or 2e-4 (U-Net) with 1,000 warm-up steps, gradient
clipping at 1, EMA 0.999, bf16, 40,000 steps at batch 32.

**Three architectures.**

| Model | Description | Parameters | Training time |
|---|---|---|---|
| Conditional U-Net | The U-Net of track 1 with a second input channel and the acquisition embedding | 15.5 M | *pending* |
| JiT | Plain transformer on 8 × 8 pixel patches of the two input channels (400 tokens), width 288, depth 12, 6 heads, fixed 2-D sine-cosine positions, DiT-style adaptive layer norm (adaLN-Zero) for time and acquisition | 18.5 M | 2.3 h |
| JiT + view attention | JiT plus, in every block, a cross-attention from the image patches to one token per projection view (below) | 22.6 M | 2.8 h* |

\*shared GPU, not directly comparable.

**Geometry-aware view attention.** Each of the 100 projection views becomes a token. The projection row of
the slice is resampled so that it is centred on the crop centre: under the projector's convention the crop
centre (y_c, x_c) projects to u_c = cos θ · y_c − sin θ · x_c on the view at angle θ, and 224 detector samples
around u_c are taken (enough to cover the diagonal of the crop). The token is a linear embedding of these
samples plus an embedding of the view angle (cos kθ and sin kθ for k = 1…8). Unmeasured views are zeroed and
masked out of the attention, so the model never sees a withheld projection. The output projection of each
cross-attention is initialised to zero, so the model starts as plain JiT. The idea is that the network is told
directly, in its own feature space, which directions the data cover and what they contain, instead of having
to infer this from a SIRT image.

**Sampling.** Euler steps of the flow from t = 1 to 0 (50 steps unless stated). At each step the network
predicts x̂; with data consistency, x̂ is corrected by 10 CGLS iterations on the measured views of the full
slice; the next state is x̂ re-noised with the noise implied by the current state,
x_{t'} = (1 − t') x̂ + t' ε̂ with ε̂ = (x_t − (1 − t) x̂) / t. A final 10 CGLS iterations follow. Four samples
per case give a posterior mean and a per-voxel spread. Every model is evaluated with and without data
consistency; the version without it shows what the network believes on its own.

### 3.4 Physics constraint, and what it can and cannot do

The data-consistency step is the physics constraint: after it, the reconstruction agrees (up to the solver's
convergence and the non-negativity clamp) with the projections that were measured. It cannot check anything
the measured views do not see: the null space of the measured projector is filled by the network alone. This
is why the error on withheld views and the calibration of the sample spread are reported separately; they are
the only handles on that null space.

### 3.5 3D Gaussian splatting: R²-Gaussian

R²-Gaussian (Zha et al., NeurIPS 2024) represents the volume as a set of 3D Gaussian kernels with a density
each, renders projections by integrating the kernels along the rays (an X-ray rasteriser instead of the
alpha-blending of view-synthesis splatting), and voxelises the kernels for the final volume. Training
minimises the L1 and D-SSIM difference to the measured projections plus a 3-D total-variation term on the
voxelised volume; Gaussians are densified and pruned as in 3D Gaussian splatting. The settings of the earlier
study are kept: 50,000 initial Gaussians, 30,000 iterations, λ_TV = 0.05.

Two ways of using the diffusion output are tested:

* **Fitted start (`diff_fit`).** The conditional model's posterior mean (4 samples, data consistency) is the
  starting volume. Initial Gaussian centres are drawn from it with the native rule, and the 50,000 Gaussians are
  then fitted to the volume (3,000 iterations of L1 on the voxelised Gaussians) before normal training starts.
  This mirrors the "fitted U-Net" start of the earlier study.
* **Diffusion guidance during training (`diff_guide`).** Same start. Every 4 iterations an extra loss
  |V − G|, between the voxelised Gaussians V and a guidance volume G, is added to the rendering loss, rescaled
  so that it has the same magnitude as the rendering loss at that moment. G is first the diffusion posterior
  mean. At iterations 2,000, 5,000, 10,000, 15,000 and 20,000 it is replaced by a diffusion *repair* of the
  current reconstruction: V is mixed with noise at flow time 0.4, x = 0.6 V + 0.4 ε, and sampled back to t = 0
  with data consistency (SDEdit-style). The measured views keep pinning the result; only what they do not see is
  redrawn by the prior.

The training loop of R²-Gaussian is not modified: both hooks are installed from outside (on the loss and on
the Gaussian initialisation).

### 3.6 Engineering side work

* **Rectified flow and shortcut models** (few-step generation, Frans et al., ICLR 2025) for the unconditional
  prior: implemented (`src/flow.py`, `src/train_flow.py`, `src/compare_priors.py`), not trained. The
  conditional models already use flow matching, and the step-count sweep in section 5.3 covers the
  speed–quality question for them.
* **JAX / Flax port** of the unconditional prior with conversion of the PyTorch weights (`jax_port/`). With
  random weights the outputs of the two implementations agree to a relative difference of 3.0e-6; a training
  step and sampling run. Not yet checked with the trained weights.

## 4. Related work

* Diffusion posterior sampling for CT and other linear inverse problems: DPS (Chung et al., ICLR 2023), the
  Decomposed Diffusion Sampler (Chung, Lee & Ye, ICLR 2024), and many CT-specific variants.
* Flow matching and rectified flow (Lipman et al., ICLR 2023; Liu et al., ICLR 2023); logit-normal time
  sampling (Esser et al., ICML 2024); x-prediction in pixel space with plain transformers (JiT, Li & He,
  CVPR 2026); adaptive layer norm conditioning (DiT, Peebles & Xie, ICCV 2023).
* Gaussian splatting for tomography: R²-Gaussian (Zha et al., NeurIPS 2024).
* Diffusion inside splatting-based CT: DiffNR (Su et al., AAAI 2026, arXiv 2604.21518) trains a single-step
  diffusion model that repairs slices of the current reconstruction and uses the repaired volumes, produced
  periodically, to supervise R²-Gaussian in sparse-view CT. The guidance arm of section 3.5 follows the same
  principle, with a different generator (a measurement-conditioned flow model trained from scratch, used by
  partial re-noising) and for limited-angle arcs. For natural scenes, Difix3D+ (Wu et al., CVPR 2025) distils
  views cleaned by a single-step diffusion model back into the 3D representation.
* Uncertainty in splatting-based CT: a 2026 preprint (arXiv 2607.13682) reports that the spread of Gaussian
  splatting reconstructions is not a reliable error map in sparse-view CT, which is one reason the sample
  spread is tested explicitly here instead of being assumed to work.

**What is and is not new here.** A search of arXiv and the major venues (November 2026) found no CT diffusion
model that uses one token per projection view, re-centred on the reconstruction crop and tagged with its
angle, as cross-attention context. A search is not proof that none exists. Combining diffusion priors with
Gaussian splatting already exists, including for CT (DiffNR, above), and no claim of novelty is made for the
combination itself. The specific questions tested are whether diffusion guidance stops the drift
of R²-Gaussian away from a good start, and what physics-based data consistency costs on the unmeasured views.

## 5. Results

### 5.1 Track 1: unconditional prior (validation and first test cases)

On validation pose 1090 (mean of 2 samples, error on unmeasured views):

| Arc | SIRT data consistency, 10 it./step | CGLS, 3 it./step | CGLS, 10 it./step | SIRT-200 alone |
|---|---|---|---|---|
| 60° | 27.4 % | 28.3 % | 23.6 % | 29.0 % |
| 120° | 22.8 % | 18.3 % | **5.7 %** | 18.3 % |

On the first four test cases with SIRT data consistency (8 samples), the posterior mean was poor: worm PSNR
11.9–14.4 dB and 20–24 % error on unmeasured views, no better than SIRT-200 alone. The sample spread ranked the
errors weakly (Spearman 0.25–0.33).

### 5.2 Track 2: conditional models (3 test poses)

Worm PSNR in dB / error on unmeasured views in %, posterior mean of 4 samples, mean over the 3 test poses.

| Method | 120° | 90° | 60° | 20 views | 10 views |
|---|---|---|---|---|---|
| SIRT-200 | 16.2 / 18.3 | 13.8 / 21.6 | 11.8 / 29.0 | 19.9 / 6.0 | 16.3 / 9.1 |
| Supervised U-Net start (one network per arc) | 25.3 / 3.5 | 24.8 / 3.9 | 23.6 / 4.6 | – | – |
| Unconditional prior + CGLS data consistency* | 23.6 / 6.4 | 17.3 / 12.5 | 11.7 / 25.7 | 20.2 / 8.0 | 13.9 / 16.0 |
| JiT, no data consistency | 25.4 / 3.5 | 25.1 / 3.7 | 24.6 / 4.0 | 25.1 / 3.6 | 24.6 / 3.7 |
| JiT + data consistency | 26.3 / 4.7 | 25.6 / 4.5 | 24.9 / 4.4 | 25.6 / 4.1 | 25.0 / 3.9 |
| JiT + view attention, no data consistency | 25.4 / 3.5 | 25.1 / 3.7 | 24.6 / 4.0 | 25.1 / 3.6 | 24.6 / 3.7 |
| JiT + view attention + data consistency | 26.2 / 4.7 | 25.6 / 4.5 | 24.9 / 4.4 | 25.7 / 4.1 | 25.0 / 3.9 |

\*2 samples per case; the two sparse-view columns are based on 2 poses so far.

Outline F-score at 1 µm: 0.992–0.996 for all conditional models, 0.988–0.993 for the U-Net start, 0.63–0.95 for
SIRT-200. Full tables, including the measured-view error, are generated by `src/summarize.py` into
`results/SUMMARY.md`.

Uncertainty (Spearman between sample spread and error): 0.33–0.40 without data consistency, 0.22–0.36 with it (3 poses).

### 5.3 Number of sampling steps (JiT + view attention, with data consistency, 2 samples, 3 poses)

| Steps | 60°: PSNR (dB) / unmeasured-view error (%) | 10 views: PSNR / error | Seconds per sample (60° / 10 views) |
|---|---|---|---|
| 1 | 24.45 / 4.57 | 24.74 / 4.04 | 4.2 / 3.3 |
| 2 | 24.73 / 4.47 | 24.85 / 4.00 | 4.4 / 3.7 |
| 4 | 24.82 / 4.44 | 24.87 / 3.99 | 5.4 / 4.6 |
| 8 | 24.85 / 4.42 | 24.90 / 3.98 | 7.4 / 6.1 |
| 16 | 24.84 / 4.42 | 24.92 / 3.96 | 11.9 / 9.2 |
| 50 | 24.88 / 4.39 | 24.97 / 3.92 | 30.1 / 23.2 |

A single step already gives all but 0.4 dB (60°) and 0.2 dB (10 views) of the 50-step result, and 4 steps are
within 0.1 dB at a sixth of the time. This is what x-prediction with a strong conditioning should give: the
network output is close to the conditional mean at every noise level, so the flow is nearly straight. Most of the
remaining time is the data-consistency solver (10 CGLS iterations on the full slice per step), not the network.
No distillation or few-step training (e.g. shortcut models) is needed for this problem.

### 5.4 Diffusion + R²-Gaussian (partial: 6 of 18 runs)

Worm PSNR (dB) of the start → after 30,000 R²-Gaussian iterations, and the final error on unmeasured views.

| Case | Diffusion start | Fitted start only (`diff_fit`) | Fitted start + diffusion guidance (`diff_guide`) |
|---|---|---|---|
| Pose 4, 60° | 25.9 | 14.9 (13.0 %) | **17.8 (8.2 %)** |
| Pose 1031, 60° | 23.7 | 14.6 (12.9 %) | **17.7 (8.0 %)** |
| Pose 8916, 60° | 25.2 | 14.7 (13.3 %) | *pending* |
| Pose 4, 90° | 26.5 | 17.6 (9.1 %) | *pending* |

Reference from the earlier study (3 test poses, seed 0):

| R²-Gaussian start | Arc | Start PSNR (dB) | Final PSNR (dB) | Final error on unmeasured views (%) |
|---|---|---|---|---|
| FBP (native) | 120° / 90° / 60° | 7.6 / 6.3 / 4.5 | 19.8 / 17.0 / 13.4 | 7.9 / 10.8 / 17.1 |
| Supervised U-Net, Gaussians fitted | 120° / 90° / 60° | 25.5 / 24.9 / 23.7 | 20.2 / 17.6 / 14.7 | 6.4 / 8.8 / 12.9 |

## 6. Analysis (from the results so far)

1. **Conditioning matters more than the prior's sophistication.** The unconditional prior with CGLS data
   consistency works at 120° (23.6 dB) but degrades quickly as the arc shrinks and is no better than SIRT at 60°
   (11.7 dB, 25.7 % error on unmeasured views). Every conditional model stays at 24.6–26.3 dB and 3.5–4.7 % error
   in all five settings, including 60° and 10 views. Telling the network what was measured, and giving it the
   measured-view reconstruction, removes most of the difficulty that the sampler otherwise has to solve.
2. **The conditional transformer beats the supervised U-Net, and the gap grows as the arc shrinks.** Without
   data consistency, JiT is +0.1 dB at 120°, +0.3 dB at 90° and +1.0 dB at 60° above the U-Net start, with lower
   error on unmeasured views in every case (3.5 vs 3.5 %, 3.7 vs 3.9 %, 4.0 vs 4.6 %). It is a single network for
   five acquisition settings, while the U-Net is trained separately for each arc. With three test poses the 60°
   difference is the only one large enough to be taken as more than an indication.
3. **The view attention gives no measurable gain here.** JiT with and without view attention differ by at most
   0.06 dB and 0.02 percentage points over the three poses, and their validation losses are identical (0.0026).
   The likely reason is that the SIRT channel already carries the same information in image space, so the view
   tokens are redundant; a setting where they could help is one without a good image-space conditioning (e.g. no
   SIRT input, or real data with geometry errors). This is reported as a negative result.
4. **Data consistency trades error on unmeasured views for PSNR.** With data consistency, PSNR rises by
   0.3–0.9 dB, but the error on unmeasured views rises too (3.5 → 4.7 % at 120°), and the sample spread ranks
   the errors less well. SIRT-200, the U-Net and every conditional model leave 2.7–4.1 % error on the measured
   views, so there is a floor that none of these reconstructions fits. Its cause is not isolated yet (candidates: mismatch
   between the reconstruction projector and the one behind the supplied data, the crop, which sets everything
   outside 160 × 160 pixels to zero, and incomplete convergence). Forcing the result towards the measured views
   fits part of this floor, and the correction leaks into directions that were not measured. With added noise
   the effect would likely be larger (not tested). Data consistency is therefore not automatically a safety
   measure; its strength should be tuned on held-out views.
5. **The conditional model is almost deterministic.** Repairing the supervised U-Net start (60 % start, 40 %
   noise) and sampling from pure noise give the same result to 0.01 dB (pose 4, 120°). The four samples barely
   differ, which explains the weak spread–error correlation. The model has learned something close to the
   conditional mean, not a broad posterior; its spread should not be used as an error map.
6. **R²-Gaussian loses a good start, whatever its quality; diffusion guidance recovers part of it.** In the
   earlier study, U-Net starts at 23.7–25.5 dB end at 14–20 dB. Starting from the better diffusion output does not
   change this: at 60° the fitted diffusion start ends at 14.6–14.9 dB, the same as the fitted U-Net start (14.7 dB).
   With diffusion guidance during training (5 refreshes), the result at 60° is 17.7–17.8 dB and the error on
   unmeasured views falls from about 13 % to 8 % (2 poses so far), the best R²-Gaussian result at this arc in either
   study. It is still well below the diffusion output itself (about 25 dB): the projection loss keeps pulling the
   Gaussians away from the prior between refreshes. A stronger or more frequent guidance is the obvious next test.

## 7. Limitations

* One dataset (one organism type, many poses, no added noise); voxel scores against an approximate phantom.
  Nothing here shows suitability for clinical data.
* Three test poses at most, one seed per model; differences below about 0.5 dB should not be read as real.
* 2-D slice models: consistency between slices is not modelled (the R²-Gaussian stage is 3-D).
* Training targets are full-angle SIRT reconstructions, not ground truth, so the models learn to reproduce SIRT
  quality at best.

## 8. Setup and reproduction

### Repository layout

| Path | Content |
|---|---|
| `src/model.py`, `src/cond_models.py` | DDPM U-Net; conditional U-Net, JiT and JiT with view attention |
| `src/sample.py`, `src/sample_cond.py` | posterior sampling with SIRT/CGLS data consistency; flow sampler with data consistency and partial-noise repair |
| `src/train_prior.py`, `src/train_cond.py`, `src/train_flow.py` | training of the unconditional prior, the conditional models, the rectified-flow / shortcut priors |
| `src/build_data.py`, `src/build_cond.py` | training data (full-angle SIRT targets, limited-view SIRT conditioning, view tokens) |
| `src/tune.py`, `src/run_test.py`, `src/eval_cond.py`, `src/summarize.py` | tuning, evaluation, result tables |
| `src/r2_guided.py`, `src/r2_runs.py` | R²-Gaussian with a diffusion start and diffusion guidance (hooks around the unchanged training loop) |
| `src/common.py`, `src/init_lib.py`, `src/evaluate_run.py` | data loading and geometry, SIRT and Gaussian initialisation, evaluation |
| `src/build_phantom.py`, `src/prepare_case.py`, `src/prepare_cases.py` | reference phantoms and R²-Gaussian test cases |
| `src/unet_init.py`, `src/make_inits.py` | supervised U-Net baseline and the starting volumes |
| `patches/r2_gaussian_parallel_beam.patch` | fix of the parallel-beam projection scale in R²-Gaussian (needed for detectors that do not span [-1, 1]) |
| `jax_port/` | JAX / Flax port of the unconditional prior |

### Setup

1. Python 3.9 with CUDA, `pip install -r requirements.txt`, and TIGRE.
2. R²-Gaussian with its CUDA extensions, patched:

       git clone --recursive https://github.com/Ruyi-Zha/r2_gaussian third_party/r2_gaussian
       cd third_party/r2_gaussian && git apply ../../patches/r2_gaussian_parallel_beam.patch
       # then install its two submodules as described in its README

   The results here were obtained with R²-Gaussian commit 72c8517 plus this patch; on the current upstream
   commit (f2579bf) the patched Python code is identical.
3. Environment variables:

       export LACT_SINO_ZIP=/path/to/sino_selected.zip   # sinogram stacks (not included)
       export LACT_WORK=$PWD/work                         # phantoms, test cases, U-Net baseline
       export R2_GAUSSIAN_ROOT=$PWD/third_party/r2_gaussian

### Steps (run from `src/`)

| Step | Command |
|---|---|
| Reference phantoms (3 test poses) | `python build_phantom.py --worm W --slab mid --out $LACT_WORK/phantoms` for W in 4, 1031, 8916 |
| Test cases and SIRT/FBP starts | `python prepare_cases.py` |
| U-Net baseline | `python unet_init.py data --out $LACT_WORK/init_study/unet`, then `python unet_init.py train --out $LACT_WORK/init_study/unet --span S --seed K --steps 2500` for S in 120, 90, 60 and K in 0, 1, 2, then `python make_inits.py $LACT_WORK/cases/* --methods unet --models $LACT_WORK/init_study/unet` |
| Training data | `python build_data.py`, `python build_cond.py` |
| Unconditional prior (track 1) | `python train_prior.py`, `python tune.py`, `python run_test.py` |
| Conditional models (track 2) | `python train_cond.py --arch jit_xattn --out ../runs/cond_jit_xattn` (and `jit`, `unet`) |
| Evaluation | `python eval_cond.py`, `python eval_cond.py --sweep jit_xattn` |
| Diffusion + R²-Gaussian | `python r2_runs.py --source cond_jit_xattn --arms diff_fit diff_guide` |
| Tables | `python summarize.py` (writes `results/SUMMARY.md`) |

Every training script resumes from its last checkpoint and every evaluation skips results that already exist.

## References

* Chung, H., Kim, J., McCann, M. T., Klasky, M. L., Ye, J. C. Diffusion posterior sampling for general noisy
  inverse problems. ICLR 2023.
* Chung, H., Lee, S., Ye, J. C. Decomposed diffusion sampler for accelerating large-scale inverse problems.
  ICLR 2024.
* Esser, P. et al. Scaling rectified flow transformers for high-resolution image synthesis. ICML 2024.
* Frans, K., Hafner, D., Levine, S., Abbeel, P. One step diffusion via shortcut models. ICLR 2025.
* Ho, J., Jain, A., Abbeel, P. Denoising diffusion probabilistic models. NeurIPS 2020.
* Li, T., He, K. Back to basics: let denoising generative models denoise. CVPR 2026 (arXiv 2511.13720).
* Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. Flow matching for generative modeling. ICLR 2023.
* Liu, X., Gong, C., Liu, Q. Flow straight and fast: learning to generate and transfer data with rectified
  flow. ICLR 2023.
* Meng, C. et al. SDEdit: guided image synthesis and editing with stochastic differential equations. ICLR 2022.
* Nichol, A., Dhariwal, P. Improved denoising diffusion probabilistic models. ICML 2021.
* Peebles, W., Xie, S. Scalable diffusion models with transformers. ICCV 2023.
* Su, S., Zha, R., Shi, D., Li, H., Cheng, X. DiffNR: diffusion-enhanced neural representation optimization
  for sparse-view 3D tomographic reconstruction. AAAI 2026 (arXiv 2604.21518).
* Wu, J. Z. et al. Difix3D+: improving 3D reconstructions with single-step diffusion models. CVPR 2025.
* Zha, R., Lin, T. J., Cai, Y., Cao, J., Zhang, Y., Li, H. R²-Gaussian: rectifying radiative Gaussian
  splatting for tomographic reconstruction. NeurIPS 2024.
* Biguri, A., Dosanjh, M., Hancock, S., Soleimani, M. TIGRE: a MATLAB-GPU toolbox for CBCT image
  reconstruction. Biomedical Physics & Engineering Express, 2016.
