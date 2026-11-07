# Implementation Log

Living document. It records what was built, why, how, and what was measured, in order. Each phase is appended as it is completed. Numbers below were produced by the code in this repo; none are copied from papers.

Reference: `DIP-Project-Proposal.pdf` (Single Image Haze Removal via Dark Channel Prior, with a Temporally Consistent Video Extension).

---

## 0. Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo, env, config | Done |
| 1 | Core single-image DCP | Done |
| 2 | Synthetic round-trip test | Done (4 tests pass); plus 14 per-stage unit tests in `tests/test_stages.py` (18 total, incl. a matting-convergence regression test) |
| 3 | Refinement: guided, fast guided, soft matting | Done |
| 4 | RESIDE SOTS benchmark and ablation | Done (`results/single_image_ablation.csv`, 40 configurations) |
| 5 | Flicker diagnosis | Done (REVIDE test, 6 clips) |
| 6 | Temporal variants V0-V3 | Done (REVIDE test, 6 clips, alpha sweep) |
| 7 | Failure-case analysis | Done (200 SOTS-outdoor images, one mitigation) |
| 8 | Demo, reports, README | Done (`figures/`, `reports/`, `presentation/`, `README.md`) |

---

## 1. Environment (Phase 0)

- Python 3.12.3. A venv lives at `.venv/`, created with `--system-site-packages`. The system already provides numpy 1.26, scipy 1.11, matplotlib 3.6, PyYAML and OpenCV 4.13. Only `pytest` and `scikit-image` were installed into the venv.
- `requirements.txt` is pinned to the versions used for all reported results. OpenCV came from the system install (4.13.0-dev); it is listed as `>=4.5` because only that one build was tested.
- **SciPy version issue:** SciPy 1.11's `cg` takes `tol`; 1.12+ renamed it `rtol`. `dcp/refine_matting.py` tries `rtol` and falls back to `tol`, so the code works on both.
- Running tests: `.venv/bin/python -m pytest -q tests` from the project root.
- Local git repo (branch `main`, one commit). `data/`, `.venv/` and run logs are git-ignored; nothing has been pushed to a remote.

### Layout
```
dcp/        single-image pipeline (this phase)
dcpvid/     video/temporal extension (empty)
eval/       metrics and benchmark runners; currently only synthetic.py
configs/    default.yaml
tests/      test_synthetic_roundtrip.py
scripts/    (empty)
```

### Configuration (`configs/default.yaml`)
All hyperparameters are in YAML and loaded by `dcp.load_config(path=None, **overrides)`. Overrides use dotted keys, e.g. `load_config(**{"refine.method": "matting"})`. This is what the one-factor-at-a-time ablation (Phase 4) will use.

| Key | Default | Meaning |
|---|---|---|
| `patch_size` | 15 | Dark-channel window (odd) |
| `omega` | 0.95 | Haze retention factor |
| `t0` | 0.1 | Lower bound on t during recovery |
| `atmosphere.estimator` | brightest | `brightest` or `mean` |
| `atmosphere.top_fraction` | 0.001 | Top 0.1% of dark-channel pixels |
| `refine.method` | guided | `none`, `guided`, `fast_guided`, `matting` |
| `refine.guided_radius` | 60 | About 4x patch size, as in the guided-filter paper |
| `refine.guided_eps` | 1e-3 | Guided filter regularization |
| `refine.fast_subsample` | 4 | Subsampling factor s |
| `refine.matting_eps` / `_lambda` / `_radius` | 1e-7 / 1e-4 / 1 | Levin Laplacian epsilon, data-term weight, 3x3 windows |

---

## 2. Single-image pipeline (Phase 1)

Input contract: `float64` RGB HxWx3 in [0,1]. `dcp.dehaze(img, cfg=None, A=None)` returns a dict with every intermediate: `dark`, `A`, `t_raw`, `t_ref`, `J`, `depth`. The optional `A` argument lets the video extension inject a temporally smoothed atmospheric light (V1-V3) without changing the pipeline.

### 2.1 Dark channel (`dcp/dark_channel.py`)
Per-pixel minimum over RGB, then a grayscale erosion with a square `patch_size` kernel (`cv2.erode`, rectangular structuring element). This is the "per-channel, per-patch minimum filter" from the proposal, computed as min-over-channels followed by min-over-patch, which is mathematically identical to the double min and cheaper.

### 2.2 Atmospheric light (`dcp/atmosphere.py`)
1. Select the top `top_fraction` of pixels by dark-channel value (`np.argpartition`, O(N); at least 1 pixel).
2. `brightest`: among those candidates, take the one with the largest R+G+B in the input image. This is He et al.'s rule.
3. `mean`: average colour of all candidates. This is the alternative for the A-estimator ablation. It is also the more stable choice and is a candidate for the video work.
Returns A as a length-3 vector.

### 2.3 Raw transmission (`dcp/transmission.py`)
`t_raw = 1 - omega * dark_channel(I / A)`, dividing each channel by its A component.

### 2.4 Recovery (`dcp/recover.py`)
`J = (I - A) / max(t, t0) + A`, clipped to [0,1]. The lower bound t0 stops noise amplification where the scene is dense with haze. `depth_from_transmission` computes `-ln(t)/beta` for visualization only; it is relative depth, not metric.

---

## 3. Transmission refinement (Phase 3 code, Phase 4 evaluation)

### 3.1 Guided filter (`dcp/refine_guided.py::guided_filter`)
Implements the colour-guide guided filter from He, Sun and Tang (ECCV 2010) with box filters (`cv2.boxFilter`, reflect border), so cost is O(N) independent of radius.
- The guide has c channels (1 or 3). `a = (Sigma + eps*I)^-1 cov(I,p)` is solved per pixel with a batched `np.linalg.solve`. `b = mean_p - a . mean_I`. The output is `mean_a . I + mean_b`.
- The pipeline currently uses the **grayscale** guide (`img.mean(axis=2)`), since the original paper filters t with the gray image. The code supports a 3-channel guide; the choice could be added to the ablation.

### 3.2 Fast guided filter (`fast_guided_filter`)
Subsamples I and p by factor s (nearest neighbour), computes the linear coefficients (a, b) at the reduced resolution with radius r/s, bilinearly upsamples a and b, and applies them to the full-resolution guide. The heavy per-pixel solve runs on N/s^2 pixels.

### 3.3 Soft matting (`dcp/refine_matting.py`)
- `matting_laplacian(img, eps, r)` builds the Levin closed-form matting Laplacian L as a sparse N x N matrix, fully vectorized over all (2r+1)^2 windows: for each window it computes the mean and covariance of the colour pixels, then `L_ij = delta_ij - (1/|w|)(1 + (I_i - mu)^T (Sigma + eps/|w| I)^-1 (I_j - mu))`, and accumulates entries through a COO matrix, summing duplicates on conversion to CSR.
- `soft_matting` solves `(L + lambda*I) t = lambda * t_raw` with conjugate gradients, warm-started at t_raw, then clips to [0,1].
- Windows are interior-only (centres from r to H-r-1). Border pixels are covered by fewer windows. This is a known simplification, and the effect on benchmark numbers is expected to be small.
- Memory and time grow with image size. A 120x160 crop took about 1.5 s. Full SOTS images (about 620x460) will take far longer, so Phase 4 should measure this and report it honestly. If it is too slow, options are a tighter CG tolerance or a downscaled solve; either must be stated in the report.

---

## 4. Synthetic correctness test (Phase 2)

Files: `eval/synthetic.py`, `tests/test_synthetic_roundtrip.py`.

### Scene and haze generation
- `make_scene`: 240x320 image of 60 random filled ellipses, colour values up to 0.6, each with one channel forced near zero (below 0.03), plus 0.01 Gaussian noise. This deliberately satisfies the dark-channel prior, so the test checks the algorithm's implementation rather than the prior's validity.
- `make_depth`: linear ramp from 0.2 (bottom) to 3.0 (top), so transmission runs from about 0.82 down to about 0.05.
- `add_haze`: `t = exp(-beta*d)`, `I = J*t + A*(1-t)`, with A_true = (0.85, 0.88, 0.92) and beta = 1.

### Tests (all pass)
1. A is recovered within 0.05 per channel (refinement off).
2. Mean absolute error of refined t versus ground truth is below 0.08 (guided).
3. Mean absolute error of J versus the clean scene is below half the hazy image's error.
4. All four refinement methods run on a crop, with t in [0,1] and the output shape preserved.

### Measured results (this run)
Hazy-to-clean MAE is 0.515.

| Refiner | max abs A err | t MAE | J MAE | time (s) | image |
|---|---|---|---|---|---|
| none | 0.044 | 0.0428 | 0.2411 | 0.02 | 240x320 |
| guided | 0.044 | 0.0479 | 0.2483 | 0.03 | 240x320 |
| fast_guided | 0.044 | 0.0471 | 0.2468 | 0.02 | 240x320 |
| matting | 0.042 | 0.0265 | 0.3884* | 1.48 | 120x160 crop |

\* The matting row was run on a crop taken from the top-left, which contains the densest haze, so its J MAE is **not comparable** with the other rows. Re-run all four on the same crop before drawing any conclusion about matting.

### Observations and caveats
- On this synthetic scene, guided filtering is slightly worse on t MAE than no refinement. The scene has hard-edged ellipses, and the guide is gray, so some edge information is lost. It should not be read as a verdict on guided filtering; the real-data benchmark (Phase 4) is the right place to compare.
- J MAE stays around 0.24 even with accurate t, because the far, dense-haze region (t near 0.05) is clamped at t0 = 0.1, which leaves residual haze and amplified noise. This reflects the t0 trade-off the ablation will explore.
- **Test-design lesson:** with colour values up to 0.9, the first run failed (A error 0.12 in the blue channel), since a bright object satisfied the "brightest among top dark-channel pixels" rule. This is the real, known DCP weakness (bright objects mistaken for airlight), so the scene's colours were capped at 0.6 to make the test target the implementation. The failure is worth citing in the failure-case section later.

---

## 5. Data (downloaded)

- **RESIDE SOTS**: HuggingFace mirror `desonglll/reside-SOTS` (`SOTS.zip`, 434 MB), unzipped to `data/reside/sots/SOTS/{indoor,outdoor}/{hazy,gt}`. Indoor: 500 hazy / 50 GT scenes (10 haze levels each, name `1400_3.png` -> GT `1400.png`). Outdoor: 500 hazy, every one has a GT (`0001_0.8_0.2.jpg` -> `0001.png`). The mirror is not the official site; the file layout matches the published SOTS.
- **REVIDE (indoor only)**: official Google Drive link from the dataset repo, 1.01 GB, unzipped to `data/revide/indoor/{Train,Test}/{hazy,gt}/<clip>/*.JPG`. Train has 42 clips, **Test has 6** (C005, E005, E006, J004, L006, W002; 36-58 frames each, 2708x1800). REVIDE outdoor and REVIDE-SYN are not released, so **all video results are indoor-only**; state this in the reports.
- Paths live in `configs/data.yaml`. Video frames are downscaled to width 640 (INTER_AREA) before processing, for both hazy and GT.
- `data/` and `results/` are generated/downloaded artefacts, not source.

---

## 6. Single-image benchmark (Phase 4)

Files: `eval/metrics.py` (PSNR, SSIM via scikit-image with `channel_axis`, CIEDE2000 = mean `deltaE_ciede2000` on Lab), `eval/reside_runner.py` (per-image metrics + wall-clock seconds measured inside each worker), `eval/ablation.py` (one-factor-at-a-time).

- The runner uses a fixed random subset (seed 0) when `--n` is given, so every configuration is scored on the same images.
- **Matting memory:** matting with 8 workers was killed (exit 137) on this 7 GB machine; the ablation now uses 2 workers for matting and 8 otherwise. Timings of runs that shared the CPU with other jobs are noisy, so treat `seconds` as approximate.
- The ablation is **resumable**: each finished configuration is appended to the CSV and skipped on re-run (jobs were repeatedly interrupted).
**Solver bug found by the reproducibility re-run (fixed).** `soft_matting` capped conjugate gradients at 2000 iterations, but about 2150-3650 iterations are needed to reach the 1e-6 tolerance at 620x460. The solve therefore stopped unconverged, and the iterate depended on floating-point noise: re-running gave 19.76 dB on the indoor matting row the first time and 20.01 dB the second. Fix: `maxiter` 20000, `atol=0`, and a warning if CG does not converge. With converged solves, PSNR is stable to 3 decimals across tolerances 1e-6 / 1e-8 / 1e-10 (checked on 3 images). The matting rows were recomputed; **the earlier 19.76 dB indoor figure (and the "-0.35 dB vs guided" claim built on it) was an artefact and is superseded**. Other rows never involved matting and are unaffected.

### Results (RESIDE SOTS; full 500 hazy images per split unless noted)
Default config (patch 15, omega 0.95, t0 0.1, brightest-pixel A, guided filter): **indoor 19.82 dB / SSIM 0.859 / CIEDE2000 7.43; outdoor 17.03 / 0.859 / 10.74**.

**Refinement method.** Matting is about 100x slower, so all four methods are compared on one fixed 25-image subset per split; the three cheap ones are also run on all 500.

| Method | Indoor PSNR / SSIM (25 img) | Outdoor PSNR / SSIM (25 img) | s/img indoor / outdoor | Full-500 PSNR indoor / outdoor |
|---|---|---|---|---|
| none (raw t) | 18.43 / 0.776 | 16.23 / 0.793 | 0.20 / 0.13 | 17.95 / 15.70 |
| guided | 20.11 / 0.850 | 17.59 / 0.857 | 0.35 / 0.30 | 19.82 / 17.03 |
| fast guided (s=4) | 20.10 / 0.849 | 17.56 / 0.855 | 0.21 / 0.19 | 19.81 / 16.99 |
| soft matting | 20.01 / 0.849 | 17.48 / 0.853 | 42.2 / 35.5 | not run |

Refinement is worth +1.9 dB (indoor) / +1.3 dB (outdoor) over raw t. Fast guided matches guided within 0.04 dB at roughly 60% of the time (the guided filter is already O(N), so the saving is modest). Matting did **not** beat guided on this subset (-0.10 dB indoor, -0.11 dB outdoor; effectively tied) and costs about 100x more; the differences on 25 images are small and untested for significance. Runtimes were measured while other jobs shared the CPU and the machine, so they are approximate and relative comparisons are the reliable part.

**One-factor-at-a-time (full split; others at default).**

| Factor | Value | Indoor PSNR / SSIM | Outdoor PSNR / SSIM |
|---|---|---|---|
| patch size | 7 | 17.52 / 0.807 | 14.98 / 0.801 |
| | **15** | 19.82 / 0.859 | 17.03 / 0.859 |
| | 31 | 21.37 / 0.880 | 19.18 / 0.899 |
| omega | 0.80 | 20.49 / 0.847 | 21.26 / 0.917 |
| | 0.90 | 20.73 / 0.861 | 18.78 / 0.894 |
| | **0.95** | 19.82 / 0.859 | 17.03 / 0.859 |
| | 1.00 | 18.06 / 0.833 | 15.13 / 0.798 |
| A estimator | brightest | 19.82 / 0.859 | 17.03 / 0.859 |
| | mean of candidates | 19.92 / 0.859 | 17.39 / 0.860 |
| t0 | 0.05 | 19.82 / 0.859 | 17.01 / 0.859 |
| | **0.10** | 19.82 / 0.859 | 17.03 / 0.859 |
| | 0.20 | 19.86 / 0.860 | 17.46 / 0.869 |
| | 0.30 | 20.28 / 0.865 | 18.36 / 0.883 |

Observations (measured, one factor at a time, no interaction effects tested):
- The paper-default omega = 0.95 and patch = 15 are **not** the best on SOTS: larger patches, smaller omega and a higher t0 all score higher. Plausible reason (not verified): SOTS haze is synthetic and homogeneous, and PSNR rewards conservative dehazing; this says nothing about visual quality or real haze. We keep the paper defaults as the reported baseline and do **not** pick a "best" configuration from these numbers, since that would tune on the test set.
- The mean-of-candidates A estimator is slightly better than the brightest pixel (+0.1 / +0.36 dB), as expected for a less outlier-sensitive estimator.
- t0 below 0.1 changes nothing (the floor rarely binds); raising it helps PSNR, consistent with t0 trading haze removal against noise gain.

**Published baselines (protocols differ; shown only to situate our numbers).** DCP on SOTS-indoor is reported as 16.95 dB / 0.792 in [Progressive Residual Learning for Single Image Dehazing](https://arxiv.org/pdf/2103.07973) and as 20.15 / 0.872 (indoor), 17.56 / 0.822 (outdoor) in [Unsupervised Single Image Dehazing Using Dark Channel Prior Loss](https://arxiv.org/pdf/1812.07051). Ours (indoor 19.82 / 0.859, outdoor 17.03 / 0.859) lies inside this spread. The spread itself shows these numbers depend on implementation details (patch size, omega, refinement, resolution), so we do not claim to be better or worse than "the" published DCP number. We did not check these papers' evaluation protocols beyond the table values.

---

## 7. Video extension (Phases 5-6)

Files: `dcpvid/flow.py`, `occlusion.py`, `video_pipeline.py`; `eval/temporal_metrics.py`, `eval/revide_runner.py`, `eval/flicker_diagnosis.py`.

### Implementation
- **Flow:** OpenCV DIS (PRESET_MEDIUM) on grayscale frames contrast-stretched to full range (hazy frames are low-contrast). Compared on 3 frame pairs against flow computed on the GT clip: min-max stretch 3.61 px mean end-point difference, CLAHE 4.31, flow on V0-dehazed frames 4.18, Farneback 13.2. Min-max + DIS-medium was kept. Note the reference is itself an estimate, not true flow.
- **Occlusion mask:** forward-backward check (Sundaram et al.): valid where `|f_cp(x)+f_pc(x+f_cp(x))|^2 < 0.01(|f_cp|^2+|f_pc|^2)+0.5`. About 76-82% of hazy-flow pixels pass, 86-89% on GT flow.
- **Variants** (alpha = weight on history, `s_t = alpha*s_{t-1} + (1-alpha)*x_t`):
  - V0: per-frame DCP.
  - V1: EMA on A. t is then re-estimated **under the smoothed A** (an earlier draft estimated t under the raw A and recovered J with the smoothed A, which was inconsistent; fixed).
  - V2: V1 + EMA on the refined t, no warping.
  - V3: V1 + EMA on t with the previous t warped to the current frame by flow cur->prev, blended only where the forward-backward mask is valid, raw per-frame t elsewhere.
- Per-frame DCP uses the default config (guided filter).

### Metrics (and a deviation from the proposal)
- Per-frame PSNR/SSIM vs GT, mean over frames.
- **Flicker index:** mean |J_t - warp(J_{t-1})| over non-occluded pixels, warped with flow computed on the *GT* clip (independent of dehazing artefacts), divided by the same quantity for the GT clip. 1.0 = same temporal change as the clean video; above 1 = excess temporal variation.
- **ROI variance (deviation):** REVIDE's camera moves 5-28 px/frame, so no pixel is static and the proposal's static ROI cannot be used. We instead split the frame into a 4x4 grid, take each block's mean colour error vs GT per frame, and report its temporal variance averaged over blocks (GT cancels legitimate scene change).
- FPS includes flow computation for V3, at 640 px width, CPU only.

### Results (REVIDE-indoor test, 6 clips, mean over clips)
| Variant | alpha | PSNR | SSIM | Flicker idx | ROI var | FPS |
|---|---|---|---|---|---|---|
| V0 | - | 12.72 | 0.678 | 2.04 | 0.0057 | 10.9 |
| V1 | 0.5 | 12.69 | 0.678 | 1.77 | 0.0055 | 9.9 |
| V1 | 0.7 | 12.63 | 0.677 | 1.69 | 0.0054 | 10.0 |
| V1 | 0.9 | 12.44 | 0.672 | **1.58** | 0.0051 | 9.8 |
| V2 | 0.5 | 12.66 | 0.673 | 2.21 | 0.0055 | 8.9 |
| V2 | 0.7 | 12.55 | 0.658 | 2.25 | 0.0055 | 8.8 |
| V2 | 0.9 | 12.07 | 0.607 | 2.19 | 0.0060 | 9.0 |
| V3 | 0.5 | 12.63 | 0.673 | 2.07 | 0.0057 | 5.5 |
| V3 | 0.7 | 12.53 | 0.666 | 2.12 | 0.0057 | 5.5 |
| V3 | 0.9 | 12.21 | 0.644 | 2.17 | 0.0055 | 5.5 |

**Headline (a negative result for the proposal's hypothesis):** smoothing the atmospheric light alone (V1) is the only variant that clearly reduces flicker (2.04 -> 1.58 at alpha 0.9) at a small PSNR cost. Smoothing t, naively (V2) or with flow warping and an occlusion mask (V3), does **not** reduce flicker in our setup, and V3 is about 2x slower. Absolute PSNR is low (about 12.7 dB) because DCP handles REVIDE's dense indoor haze poorly; the differences between variants are small relative to that, and with 6 clips they have not been tested for significance.

### Why V3 does not help (checked, partly unexplained)
- Not a flow-quality problem: V3 using **ground-truth** flow is also worse than V1 (C005: 2.58 vs V1 2.20; J004: 2.22 vs 1.48 at alpha 0.7).
- V3 does stabilize t itself: flow-warped t difference 0.0063 (V0) -> 0.0042 (V3) on C005. So t is smoother but J flicker is not lower. We have **no verified mechanism**; the hypothesis that hazy inputs carry exposure jitter was tested and rejected (hazy warped-difference is 0.66-0.94 of GT's). One untested idea: smoothing t decouples it from the per-frame I that DCP's own t co-varies with, so the 1/t gain is no longer self-compensated.

### Flicker diagnosis (Objective 4), mean flicker index over 6 clips
| Intervention on V0 | Flicker idx |
|---|---|
| V0 baseline | 2.04 |
| A frozen to the clip median (removes A jitter) | **1.58** |
| A frozen + V3 t smoothing (alpha 0.7) | 1.94 |
| t0 = 0.3 | 1.99 |
| t0 = 0.05 | 2.04 |

- A jitter is the largest single removable cause (but strongly clip-dependent: little effect on C005/E005, large on J004/W002/L006).
- Raising t0 barely helps (boundary clipping is a minor cause); t smoothing on top of frozen A **adds** flicker.
- Per-t-quartile breakdown of V0 flicker (mean raw warped diff, lowest t -> highest t): 0.0289, 0.0208, 0.0150, 0.0124 vs GT 0.0094. Flicker is 3x GT in the haziest quartile and 1.3x in the clearest: consistent with noise amplification by 1/t, a **fourth cause** not in the proposal's list, and the likely reason temporal smoothing of A/t cannot reach the GT floor.

---

## 8. Failure case: sky (Phase 7, Objective 6)

Files: `dcp/sky.py`, `eval/failure_analysis.py`, `scripts/make_figures.py` (figure `figures/sky_psnr_scatter.png`).

- **Sky fraction** (for analysis) is computed on the clean GT image: pixels with max channel > 0.7 and saturation (max-min) < 0.2, in connected components touching the top row. Classical, no learning.
- **Mitigation (sky-aware transmission floor):** on the hazy image find the region within 0.10 of A in every channel, low saturation, connected to the top row; feather its mask (Gaussian, sigma 5 px), and raise t to 0.6 inside it. This stops J being amplified by 1/t where DCP's prior is invalid.
- Evaluated on a fixed 200-image subset of SOTS outdoor:

| Sky fraction | n | baseline PSNR | mitigated PSNR |
|---|---|---|---|
| all | 200 | 16.85 | 18.04 |
| < 5% | 54 | 18.26 | 18.35 |
| 5-25% | 82 | 17.78 | 18.76 |
| >= 25% | 64 | 14.47 | **16.87** |

Pearson correlation of sky fraction with baseline PSNR is -0.50. The floor value (0.6) and thresholds were chosen once, not tuned on held-out data; they were evaluated on the same images, so the gain is somewhat optimistic.

---

## 9. Open items and decisions

1. Done: ablation, baselines, demo clip, transmission figure, README, `run_all.sh`.
2. Done: `reports/report1.pdf`, `reports/report2.pdf` (built from the `.md` with pandoc), `presentation/DIP_Project_Presentation.pptx` (built by `presentation/make_deck.js`, redesigned: native charts, speaker notes on slides 2-11).
   Note: the course's exact Report 1 / Report 2 checkpoint formats were not available to us, so the split (single-image / video + failure case) is our reading of the proposal; adapt headings if the course template differs.
3. Open question for the team: V3 does not beat V1. Options: report as a negative result (recommended, with the diagnosis above), or investigate further (e.g. blend t only in high-t regions, or smooth in the log/depth domain).
4. Matting uses interior-only windows and a CG solve with tol 1e-6; see section 3.3.
5. Git: local repo created; no remote configured. Speaker notes added to slides 2-10 of the deck.

---

## 10a. Reproducibility check (full `run_all.sh` from empty `results/` and `figures/`)

`scripts/run_all.sh` ran to completion (exit 0) after clearing `results/` and `figures/`. Comparison with the earlier run:
- video benchmark (60 rows), flicker diagnosis (30 rows) and t-quartile table: **identical** (max difference 0.0).
- sky failure analysis (200 images): sky fractions identical; PSNR/SSIM differ by up to 0.001 dB / 1e-5 (floating-point nondeterminism).
- single-image ablation: all rows identical to within 5e-4 dB except the indoor matting row, which exposed the unconverged-solver bug above and was recomputed.
- Not independent of the machine: timings (`seconds`, `fps`) vary run to run (ratio 0.7-1.1x here) and depend on CPU load.

---

## 10. Change log

- **Phase 0-2:** created the package layout, config, full single-image pipeline, three refiners, synthetic generator and four tests.
- **Final pass:** 14 per-stage unit tests (dark channel, A estimators, t, recovery, guided/fast guided, matting Laplacian symmetry and null space, sky mask, warp, occlusion mask, metrics); pinned requirements; references added to both reports; deck layout fixes; full `run_all.sh` re-run from empty `results/` and `figures/` (see section 11).
- **Phase 8:** full ablation finished (40 configurations); baselines with sources; stage figure; reports 1-2 (PDF), presentation deck.
- **Phase 4-7:** downloaded RESIDE SOTS and REVIDE; added metrics, SOTS runner and resumable ablation; added flow, occlusion, V0-V3, temporal metrics, REVIDE runner, flicker diagnosis, sky detection/mitigation and figure script.
