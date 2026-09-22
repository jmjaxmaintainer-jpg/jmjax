# jmjax MCMC Performance Investigation: Root Causes and Resolutions

**A technical report on diagnosing and resolving a suspected performance regression in jmjax's MCMC backend (`spline-PH-mcmc`/`weibull-PH-mcmc`)**

---

## Executive Summary

This investigation began from a well-founded concern: recent feature additions to jmjax's MCMC backend (functional-forms channels, `q=2` generalization, baseline covariates) appeared to have introduced a substantial, unexplained performance regression relative to an earlier, validated benchmark showing jmjax outperforming `JMbayes2` by roughly 12–13x in sampling efficiency (ESS/second). Real-world and simulated tests using the harder parameter regimes and covariate structures typical of clinical data showed jmjax's advantage collapsing to near-parity or worse.

Systematic investigation — testing one hypothesis at a time, discarding each when disproven rather than accumulating unverified assumptions — traced this apparent regression to **seven distinct, independently-confirmed issues**, none of which were the covariates, effect sizes, or "harder data" originally suspected. Six were genuine bugs or missed optimizations in jmjax's own MCMC implementation; fixing them did not merely restore the original 12–13x figure but, in several tested configurations, produced a package that is faster, more sample-efficient, and better-converging than before the regression was ever introduced.

The seventh finding was methodological: a substantial fraction of the *originally reported* 12–13x advantage turned out to rest on an unfair comparison (an unpenalized jmjax spline model compared against `JMbayes2`'s penalized default). Correcting for this, and applying all fixes, the honest, validated advantage of jmjax over `JMbayes2` for `q=2` spline-baseline joint models is approximately **2–4x in sampling efficiency**, with wall-clock time ranging from comparable to meaningfully faster depending on configuration and sample size.

This document exists to preserve the technical detail of what was found and fixed, both as a debugging record and as a foundation for a future methods paper or software documentation. A subsequent follow-up investigation, documented in Section 6, applied the same diagnostic rigor to a residual inefficiency the core fixes had narrowed but not resolved — and, in the course of validating a promising-looking fix, surfaced a genuine, quantitatively-characterized statistical limitation rather than a clean win, which is preserved here in full rather than omitted. A second follow-up (Section 7) confirmed that jmjax's efficiency advantage over JMbayes2 depends on covariate scale — reversing entirely for unstandardized, real-world-scale covariates — and traced this to a specific, well-documented architectural difference in how JMbayes2 seeds its MCMC proposals, pointing to a concrete, already-planned fix.

---

## 1. Background and Motivation

jmjax is an R package with a JAX/NumPyro Python backend implementing joint longitudinal-survival models, benchmarked throughout development against the established reference packages `JM` (maximum likelihood, adaptive Gauss-Hermite quadrature) and `JMbayes2` (Bayesian MCMC).

An earlier, carefully controlled benchmark (`n=200`/`500`, simulated data, `spline-PH-mcmc`, `q=2` random intercept + slope, no covariates, matched spline basis dimension) found jmjax approximately 12–13x more sample-efficient than `JMbayes2` (paired t-test, `p < 10⁻⁷`), with statistically indistinguishable parameter recovery accuracy.

Subsequent feature development added baseline covariates, additional functional-forms channels (delta, area, area-average), and generalized several code paths to `q=2`. When these newer capabilities were exercised on more realistic configurations — larger effect sizes matching a real clinical dataset (PBC2), genuine covariates, and larger sample sizes — jmjax's advantage over `JMbayes2` appeared to largely disappear, in some configurations reversing to a jmjax *disadvantage*. This was troubling enough to warrant halting further feature work and conducting a full diagnostic investigation before trusting any further performance claims.

---

## 2. Investigation Methodology

The investigation followed a strict discipline, stated explicitly because it materially affected the outcome:

1. **One hypothesis at a time.** Each candidate explanation (covariates, effect-size regime, time scaling, sample size, session state, prior structure) was isolated via a controlled comparison — holding every other factor fixed — before being accepted or discarded.
2. **Discard, don't accumulate.** A disproven hypothesis was dropped entirely rather than retained as a partial explanation. Several plausible-sounding early theories (raw covariate scale, the σ_b1/β1 ratio, effect-size magnitude alone) were tested directly and cleanly refuted.
3. **Distrust convenient numbers.** Several early "confirmations" turned out to be artifacts of measurement error rather than real findings, and were caught specifically because a *sanity-check reproduction* of a previously-validated result failed to reproduce — triggering deeper investigation rather than acceptance.
4. **Validate correctness before trusting speed.** Every new implementation (Wishart-Gibbs sampling, vectorized RW2 construction, targeted mass-matrix adaptation) was checked against known truth via simulation *before* any speed comparison was trusted.
5. **Fair comparison discipline.** The final, most important correction in this investigation was recognizing that an apparent software performance gap was partly a *model specification* mismatch, not a software efficiency difference — a reminder that cross-package benchmarking requires matching statistical models, not just software configuration.

---

## 3. Root Causes Identified

### 3.1 JAX Asynchronous Dispatch: `sampling_time_sec` Understated Real Compute Time

**Symptom:** Every "fast," small-scale diagnostic script built during this investigation showed absolute ESS/second figures roughly 10–100x lower than the original benchmark, even when reproducing the identical model, data, and settings.

**Root cause:** JAX dispatches computation asynchronously — `mcmc.run()` can return control to Python as soon as computation is *dispatched* to the device, without waiting for it to finish executing. The internal timer (`sampling_time_sec`) was measured immediately after `mcmc.run()` returned, capturing only dispatch time. The real computation continued in the background and only became visible (as a blocking wait) the first time *any* code tried to read an actual sampled value — which, in every diagnostic script built during this investigation, happened inside `numpyro_summary()`.

**Diagnostic confirmation:** Direct `cProfile` profiling of a slow-appearing `numpyro_summary()` call showed 38.286 of 38.310 total seconds spent inside a single line: `jax/_src/array.py:634(_value)`, called via `jax.device_get()`. NumPyro's own diagnostic computation (R-hat, ESS, autocorrelation) accounted for under 0.02 seconds of the same call. This eliminated "summary computation is slow" as an explanation and pointed directly at asynchronous dispatch.

**Fix:** Insert `jax.block_until_ready(mcmc.get_samples())` immediately after `mcmc.run()`, before measuring elapsed time. This forces real completion at the point the timer expects it, at zero cost to the actual sampling procedure — a timing-measurement fix, not a modeling or sampling change.

**Impact:** This single fix accounted for the majority of the apparent "penalized model class is catastrophically slow" and "covariates are extremely expensive" findings from earlier in the investigation — those effects were mostly measurement artifacts, not real. A single fresh-session verification (`n=200`, no covariates) recovered `alpha` ESS/second of 164.9, matching the original benchmark's 171.9 for the same scenario.

---

### 3.2 Wasted Diagnostic Computation on Per-Subject Latent Variables

**Symptom:** `numpyro_summary()` remained slower than expected even after the async-dispatch fix, for `q=2` models with `n` in the hundreds.

**Root cause:** `numpyro_summary()` was being called on the *entire* `samples_by_chain` dictionary, including the per-subject random-effects sites — `b` (500 subjects × 2 dimensions = 1000 individual traces, or more at larger `n`) and, for the non-centered `q=2` parameterization, `b_std` as well. Full R-hat/ESS diagnostics were computed for every individual subject's trace on every fit, despite neither site ever being reported in the final output (`fit$estimates` only reports population-level parameters).

**Fix:** Explicitly exclude `b`, `b_std`, and (after a related fix below) `z_step` from the dictionary passed to `numpyro_summary()`. Raw posterior samples for these sites remain fully available elsewhere in the pipeline for random-effects extraction (`ranef()`); only the unused, computationally expensive diagnostic pass is skipped.

**Note on discovery order:** This was tested and found to be a real, but *secondary*, cost — excluding these sites reduced but did not eliminate the discrepancy with the original benchmark, correctly pointing the investigation toward the larger async-dispatch issue described above.

---

### 3.3 Progress Bar Overhead

**Symptom:** An isolated test comparing `progress_bar=TRUE` (NumPyro's default) against `progress_bar=FALSE` on an identical model and dataset showed `sampling_time_sec` of 97.87 seconds versus 3.09 seconds — a 32x difference.

**Root cause:** NumPyro's console progress bar issues per-iteration updates; under reticulate's R–Python console bridging, this overhead appears to be dramatically amplified relative to native Python execution.

**Implication:** Every MCMC user leaving `progress_bar` at its (enabled) default pays this cost on every fit. This is a substantial, real-world usability issue independent of anything else in this report, and is flagged here as a priority follow-up (see Section 6).

---

### 3.4 Serial Execution of Multiple MCMC Chains

**Symptom:** Every multi-chain fit displayed the warning *"There are not enough devices to run parallel chains... Chains will be drawn sequentially."* — regardless of how many CPU cores were actually available on the host machine.

**Root cause:** JAX exposes exactly one computational "device" to NumPyro by default, irrespective of physical core count. Genuine parallel chain execution requires explicitly calling `numpyro.set_host_device_count(n)` **before** JAX's backend initializes for the first time in a process — a one-time, session-level configuration that jmjax was not performing.

**Fix:** `numpyro.set_host_device_count()` is now called automatically the first time jmjax's Python backend is imported (lazily on first `jm_fit()` call, or explicitly via `jmjax_setup(num_devices = ...)`), defaulting to `parallel::detectCores()`. Because this setting can only be applied once per R session, `jmjax_setup()` now warns if called after the backend has already loaded.

**Impact:** Multi-chain fits with `num_chains > 1` can now execute genuinely in parallel rather than queuing sequentially, subject to the host machine's core count and other hardware-level scaling limits (memory bandwidth contention, physical vs. logical core distinctions).

---

### 3.5 LKJCholesky Correlation Structure for `q=2` Random Effects

**Symptom:** Direct comparison of the correlated (`q=2`, random intercept + slope with estimated correlation) versus an independent (uncorrelated) random-effects specification showed the correlated model consistently slower, less sample-efficient, and exhibiting borderline convergence (R-hat = 1.07) despite genuine correlation being present in the simulated data.

**Investigation:** A hand-inspection of jmjax's existing implementation showed the correlation structure was built via an `LKJCholesky`-distributed Cholesky factor (`L_corr`) combined with a matrix transform (`b = b_std @ L.T`) — both fully explored by NUTS via gradient-based HMC on every leapfrog step, jointly with every other model parameter.

By contrast, `JMbayes2`'s own documented methodology (Rizopoulos, 2016, *Journal of Statistical Software*) specifies that the random-effects covariance matrix is updated via a **Wishart-conjugate Gibbs draw**: given a Wishart prior on the precision matrix and a Gaussian random-effects likelihood, the posterior conditional is *also* Wishart — a direct, closed-form sample requiring zero gradient evaluations, rather than gradient-based exploration of an `LKJCholesky` parameterization.

**Fix:** Implemented a Wishart-conjugate Gibbs update for the `q=2` random-effects precision matrix, hybridized with NUTS for all remaining parameters via NumPyro's `HMCGibbs` API (available as `control$random_effects_method = "wishart_gibbs"`, opt-in, defaulting to the prior `LKJCholesky`-based behavior for backward compatibility). The conjugate update formula was derived explicitly:

Given `bᵢ | D ~ N(0, D)` for `i = 1,...,N` and prior `D⁻¹ ~ Wishart(ν₀, S₀)`:

```
D⁻¹ | b ~ Wishart(ν₀ + N, (S₀⁻¹ + Σᵢ bᵢbᵢᵀ)⁻¹)
```

This was validated first via full truth-recovery testing (all seven reported parameters, including the newly-derived `rho`, recovered within 3 standard errors of ground truth on first implementation) before any speed comparison was trusted.

**Impact (single-run, `n=500`, same data):**

| Method | Wall-clock | Max R-hat | ESS/second | Estimated ρ (true 0.3) |
|---|---|---|---|---|
| Correlated (LKJCholesky, pre-fix) | 41.87s | 1.0700 | 0.98 | 0.397 |
| **Wishart-Gibbs (new)** | **36.53s** | **1.0407** | **1.48** | **0.322** |
| Independent (no correlation modeled) | 50.07s | 1.0125 | 2.46 | N/A |

The Wishart-Gibbs approach dominates the prior correlated implementation on every metric — faster, better-converging, and more accurate on the parameter of interest — while still narrowing, rather than fully closing, the gap to the independent (uncorrelated) model. Diagnostic evidence (per-parameter ESS in the truth-recovery run) indicates a *separate*, smaller residual source of difficulty remains: coupling between the population intercept (`β₀`) and the random-effects structure, which a Gibbs update to the covariance matrix alone does not resolve.

**A secondary finding surfaced during this fix:** the correlation parameter ρ was never being extracted or reported in `fit$estimates` for *any* prior MCMC fit with correlated `q=2` random effects, despite `L_corr` being sampled throughout. This reporting gap was corrected for both the existing LKJCholesky path and the new Wishart-Gibbs path as part of this fix.

---

### 3.6 Sequential Scan Implementation of the RW2 Smoothing Prior

**Background:** jmjax's `"penalized"` spline-prior option implements a second-order random-walk (RW2) smoothing prior on the baseline-hazard spline coefficients — mathematically equivalent to `JMbayes2`'s own multivariate-normal, banded-precision-matrix P-spline formulation (verified algebraically: both reduce to penalizing the sum of squared second differences of the coefficient sequence, scaled by a shared smoothing precision parameter).

**Symptom:** The existing implementation constructed this recursion via `numpyro.contrib.control_flow.scan`, executing one sequential step per spline coefficient, with a separate `numpyro.sample` call for the innovation at each step.

**Fix:** The RW2 recursion `Wₖ = 2Wₖ₋₁ - Wₖ₋₂ + σ_w zₖ` was shown to admit a closed-form solution:

```
Wₖ = W₂ + (k-2)(W₂ - W₁) + σ_w · S2ₖ,   k = 3,...,K
```

where `S2ₖ` is the double cumulative sum of the innovation vector `z`. This identity was verified numerically (pure NumPy, outside any MCMC context) against the sequential recursion across multiple sizes and random seeds, matching to floating-point precision (maximum absolute difference ≈ 3.6 × 10⁻¹⁴).

The vectorized form replaces the sequential `scan` — and its per-step sampling bookkeeping — with two `jnp.cumsum` calls, available as `control$rw2_implementation = "vectorized"` (opt-in; the original `scan`-based implementation remains the default).

**Impact (single-run, `n=500`):**

| Implementation | Wall-clock | Max R-hat | ESS/second |
|---|---|---|---|
| Sequential scan (prior default) | 66.45s | 1.0407 | 0.81 |
| **Vectorized closed-form** | **46.30s** | **1.0083** | **1.70** |

Population-level parameter estimates were essentially identical between implementations (`β₀`, `β₁`, `τ_w` all matching to within expected MCMC noise), confirming the two are computing the same statistical model with different, non-equivalent computational efficiency.

---

### 3.7 Diagonal Mass Matrix Insufficient for Correlated Spline Coefficients

**Symptom:** Even after the vectorization fix above, the `"penalized"` spline prior remained substantially more expensive than the unpenalized `"independent"` alternative, and — critically — this gap persisted even when the *same* model was compared against `JMbayes2`'s own penalized default, largely erasing jmjax's advantage on this specific configuration.

**Diagnostic instrumentation:** A new diagnostic was added exposing NUTS's actual per-iteration leapfrog step count (`mean_num_steps`, `max_num_steps`) via NumPyro's `extra_fields` mechanism. This directly confirmed the hypothesis rather than relying on indirect inference:

| Spline prior | Mean leapfrog steps/iteration |
|---|---|
| Independent | 31 |
| Penalized (unfixed) | **152** |

**Root cause:** NUTS's default mass matrix is diagonal, treating parameters as independent for the purposes of momentum resampling. The RW2 smoothing prior, however, induces genuine, known off-diagonal correlation among neighboring spline coefficients (the same non-diagonal precision structure derived in Section 3.6's mathematical equivalence to `JMbayes2`'s formulation). A diagonal mass matrix cannot represent this correlation, forcing NUTS into substantially deeper exploratory trees to compensate.

This diagnosis was informed directly by the resolution of Section 3.5: the same general principle (a diagonal treatment of a genuinely correlated block of parameters is expensive for gradient-based HMC) applied here, but — unlike the random-effects case — no closed-form conjugate update is available for spline coefficients, since they enter through a non-Gaussian survival likelihood. A **targeted, non-global** dense mass matrix was identified as the appropriate alternative: NumPyro's `dense_mass` parameter accepts a list of site-name groups, allowing a dense sub-block restricted to *only* the small (7–9 dimensional) spline-coefficient block, leaving the much larger random-effects block (hundreds to thousands of dimensions) diagonal.

*(An earlier, naive attempt to apply `dense_mass=True` globally — covering the per-subject random effects as well — was tested and found to be catastrophically counterproductive: approximately 12.5x slower, attributable to `O(n³)`-scaling matrix estimation and inversion cost across a ~1000-dimensional block. This failure was instructive: it confirmed the *targeting* of the fix, not merely the concept of a dense mass matrix, was essential.)*

**Fix:** `control$dense_mass_spline = TRUE` restricts NUTS's dense mass matrix to the spline-innovation parameters specifically (`W01`, `z_step`), leaving all other parameters under the default diagonal treatment.

**Impact (single-run, `n=500`):**

| Configuration | Wall-clock | Max R-hat | ESS/second | Mean leapfrog steps |
|---|---|---|---|---|
| Independent | 62.89s | 1.0093 | 1.80 | 31 |
| Penalized, vectorized RW2 (no targeted mass) | 75.30s | 1.0516 | 1.26 | 152 |
| **Penalized, vectorized RW2, targeted dense mass** | **16.95s** | **1.0081** | **7.80** | **63** |

The fully-fixed penalized configuration is not merely competitive with the unpenalized alternative — it outperforms it on every metric, while also providing the statistically preferable smoothed baseline hazard estimate.

---

## 4. A Methodological Correction: Confounded Model Specifications

Independent of the six implementation-level fixes above, a distinct and equally important issue was identified in the comparison methodology itself. Several jmjax-vs-`JMbayes2` speed comparisons conducted mid-investigation used jmjax's `spline_prior = "independent"` (no smoothing penalty) against `JMbayes2`'s default `jm()` behavior, which applies its own P-spline smoothing penalty unconditionally.

These comparisons — which had shown jmjax achieving a 5–8x sampling-efficiency advantage — were therefore not comparing software implementations of the *same statistical model*, but a simpler (unpenalized) model against a more complex (penalized) one. Rerunning the comparison with matched model specifications (`spline_prior = "penalized"` on the jmjax side) showed the advantage largely disappear (wall-clock ratio approximately 0.95–1.09x; ESS/second ratio 1.36–1.99x) *before* the Section 3.7 fix, and returned to a substantial, honestly-earned advantage after it (see Section 5).

This finding underscores a general principle for any future benchmarking work: **apparent software performance differences must be checked against the possibility of a model specification mismatch before being attributed to implementation quality.**

---

## 5. Final Validated Performance Comparison

Using jmjax's current best configuration — `random_effects_method = "wishart_gibbs"`, `rw2_implementation = "vectorized"`, `dense_mass_spline = TRUE`, genuine parallel chains — against `JMbayes2`'s own default (penalized spline baseline, matched basis dimension), on identical simulated data (`n=500`, `spline-PH-mcmc`, `q=2`):

| | Wall-clock | `alpha` ESS | `alpha` ESS/second |
|---|---|---|---|
| jmjax (all fixes) | 18.99s | 1584.5 | 83.44 |
| `JMbayes2` | 46.24s | 1011.6 | 21.88 |
| **Ratio** | **2.4x faster** | — | **3.81x more efficient** |

This result was validated via a preceding truth-recovery check confirming correct parameter recovery under the combined new features before the speed comparison was trusted, consistent with the validate-before-trust discipline followed throughout.

### Separately validated: MLE scaling behavior

Independent of the MCMC investigation above, the maximum-likelihood (adaptive Gauss-Hermite) estimation paths (`weibull-PH-aGH`, `spline-PH-aGH`) were found to scale markedly better with sample size than the `JM` reference package, attributable to full marginalization of random effects via Laplace approximation (keeping the optimization target's dimension fixed regardless of `n`) combined with vectorized (`vmap`-based) per-subject computation:

| n | jmjax time (weibull-PH-aGH) | `JM` time | Ratio |
|---|---|---|---|
| 1,000 | 5.60s | 6.28s | 0.89x |
| 5,000 | 8.46s | 25.33s | 0.33x |
| 10,000 | 20.47s | 125.30s | 0.16x |
| 20,000 | 14.60s | 182.97s | **0.08x (≈12x faster)** |

A crossover point occurs between `n≈1,000` and `n≈2,000`; the advantage grows substantially and consistently with sample size, confirmed for both Weibull and spline baseline hazards. This is a structurally different scaling story from the MCMC case (Section 5, main table): MLE's advantage compounds with `n` because the marginalized parameter count is `n`-independent, while MCMC's per-subject random-effects sampling means both packages' costs scale with `n`, yielding a stable *multiplier* rather than a widening one.

---

## 6. An Experimental Extension: Addressing the Residual β₀–Random-Intercept Coupling

Section 3.5 noted that the Wishart-Gibbs fix narrowed, but did not fully close, a residual source of inefficiency: per-parameter ESS diagnostics consistently showed the population intercept `β₀` mixing worse than any other parameter, across every random-effects configuration tested. This section documents a follow-up investigation that diagnosed the mechanism precisely, implemented an experimental fix as a fully separate, opt-in code path, and — in the course of validating it — surfaced an important, quantitatively-characterized statistical limitation.

### 6.1 Diagnosis: a near-perfect location degeneracy

Raw posterior traces for `β₀` and for the per-sample mean of the random intercepts (`mean(b_{i0})` across subjects) were extracted directly via `fit$posterior_samples` and correlated. The result was a posterior correlation of **−0.9510** (`n=500`): `β₀` and `mean(b_{i0})` move almost in perfect lockstep, in opposite directions, across posterior draws (`SD(β₀) = 0.037`, `SD(mean(b_{i0})) = 0.036`). This is the textbook signature of a mixed-model location degeneracy — the data constrain only `β₀ + mean(b_{i0})`, not the two separately — and directly explains why `β₀` consistently showed the lowest ESS of any population parameter.

### 6.2 Implementation: `wishart_gibbs_centered` (experimental, opt-in, fully separate)

A new random-effects method, `control$random_effects_method = "wishart_gibbs_centered"`, was implemented as an **additive, parallel branch** — the existing `"wishart_gibbs"` code path is not modified in any way, eliminating risk to previously-validated behavior. The raw random effects (`b_raw`) are sampled exactly as before, and feed the same Wishart-conjugate Gibbs update for `D⁻¹` unchanged. A new deterministic transform then subtracts the current posterior sample's own intercept-column mean:

```
b0_centered = b_raw[:, 0] - mean(b_raw[:, 0])
```

registered under the standard `"b"` site name so that `ranef()` extraction and all downstream reporting work unchanged. This forces `mean(b_{i0}) = 0` **exactly**, at every posterior draw, rather than merely encouraging it via the prior — removing the degeneracy by construction, since all population-level intercept information must then flow through `β₀` alone.

### 6.3 Initial validation (n=500)

| | Wall-clock | Max R-hat | `β₀` ESS |
|---|---|---|---|
| Original (uncentered) `wishart_gibbs` | 51.61s | 1.0166 | 75.7 |
| **`wishart_gibbs_centered` (new)** | **12.22s** | 1.0484 | **361.4 (4.8x)** |

The sum-to-zero constraint was confirmed to hold numerically to floating-point precision (`max |mean(b_{i0})|` across posterior samples ≈ 2.25 × 10⁻⁸), and all other parameter estimates (`β₀`, `α`, `σ_b0`, `ρ`) remained close to both truth and the uncentered version's own estimates.

### 6.4 A discovered limitation at larger n: credible-interval miscalibration

Following the same never-trust-a-single-run discipline used throughout this investigation, a 5-seed replication at a larger sample size (`n=1500`) was run. The point estimates were unbiased on average (mean error across seeds: **+0.0035**, errors roughly symmetric: `+0.024, −0.015, +0.041, +0.001, −0.034`) — but **3 of 5 seeds (60%) showed `|z| > 3`** for `β₀` against its own reported posterior SD, versus an expected rate of well under 1% for a well-calibrated model.

This was initially suspected to be a single-chain diagnostic blind spot (split-R-hat can look healthy even when a single chain hasn't fully explored a parameter's true posterior width). This hypothesis was tested directly and **decisively refuted**: rerunning the flagged case with 4 genuine, independent parallel chains gave `β₀` R-hat = 0.9995 and ESS = 2439.5 — near-perfect between-chain agreement. The MCMC itself is healthy; the miscalibration is a real property of the fitted model, not a sampling artifact.

**Mechanism.** In the true data-generating process, the sample mean of the simulated random intercepts is a genuine random quantity, `mean(b_{i0}) ~ N(0, σ_{b0}²/n)` — not exactly zero for any specific finite dataset. The hard sum-to-zero constraint forces the *fitted* model's random-intercept mean to exactly zero regardless, so `β₀`'s estimate must absorb each dataset's own specific sample-mean offset to fit the data — and the reported posterior SD, computed under the (violated) assumption that this offset is genuinely zero, does not include this source of variability.

**Quantitative confirmation.** Combining the reported SE with the missing finite-sample variance component in quadrature:

```
sqrt(0.0062^2 + (0.8/sqrt(1500))^2) = sqrt(0.0062^2 + 0.0207^2) ~= 0.0216
```

reduces the flagged case's z-score from 3.86 to approximately **1.11** — a thoroughly unremarkable deviation. This is a quantitatively consistent explanation, not merely a plausible post-hoc story, and is a known, documented cost of exact identifiability constraints in Bayesian hierarchical modeling generally, not a defect specific to this implementation.

### 6.5 Recommendation

**Hold `wishart_gibbs_centered` as experimental and opt-in; do not promote it to default status**, and do not implement a naive analytical variance correction without first formally deriving it for jmjax's specific model structure — the random intercepts here enter through a non-Gaussian survival hazard (and, when active, functional-forms channels) rather than a simple Gaussian likelihood, so the exact form of the missing variance term is not necessarily the same simple quadrature-sum used above for diagnostic purposes; that approximation confirmed the *mechanism*, not necessarily the *exact correction*. Two candidate long-term resolutions, neither implemented in this investigation:

- **A soft sum-to-zero penalty** (a tight ridge/Normal prior pulling `Σb_{i0}` toward zero, rather than constraining it exactly) — likely preserves most of the sampling-efficiency gain while retaining genuine finite-sample variability in the model, avoiding the miscalibration entirely.
- **A formally-derived post-hoc variance correction** accounting properly for the nonlinear hazard coupling, applied analytically to `fit$se` for this method specifically.



## 7. Covariate Scale Sensitivity: A Confirmed, Architecturally-Rooted Difference

A further line of investigation, prompted by an unexpectedly large (~3.4x) speedup observed after standardizing a real covariate on PBC2 data, led to a controlled replication and a mechanistic explanation with a direct, actionable implication for jmjax's design.

### 7.1 Direct confirmation via controlled replication

An earlier controlled test in this investigation (comparing a standardized covariate, `N(0,1)`, against an `age`-like-scale covariate, `N(50,10)`, with the true effect size held constant by construction) had found no statistically significant difference (paired t-test, `p = 0.16`) — but that test predated the `wishart_gibbs`, vectorized-RW2, and `dense_mass_spline` fixes documented above.

Rerunning the identical simulator, seeds, and design **after** those fixes were in place produced a clear, statistically significant effect:

| | Time | ESS/second |
|---|---|---|
| Standardized covariate | 10.30s | 75.12 |
| Age-like scale (mean 50, sd 10) | 25.78s | 24.86 |
| **Paired t-test** | **p = 0.0024** | **p = 0.012** |

The effect was real from the start; the earlier, larger implementation bottlenecks had simply been masking it — a second confirmed instance of the same "fix the dominant cost, and the next one becomes visible" pattern already seen in this investigation.

### 7.2 The consequential finding: jmjax's advantage over JMbayes2 depends on covariate scale

Computing the same comparison against `JMbayes2` on this identical, controlled dataset revealed something more consequential than a jmjax-internal optimization opportunity:

| | jmjax ESS/sec | JMbayes2 ESS/sec | **Ratio (jmjax/JMbayes2)** |
|---|---|---|---|
| Standardized covariate | 75.12 | 30.87 | **2.43x (jmjax ahead)** |
| Age-like scale | 24.86 | 32.66 | **0.76x (JMbayes2 ahead)** |

`JMbayes2`'s own efficiency was essentially unaffected by covariate scale (`30.87` vs `32.66` — statistically flat, if anything marginally better on the "harder" scale). jmjax's efficiency, by contrast, dropped by a factor of three. **The practical consequence: jmjax's efficiency advantage over JMbayes2 is not unconditional — it depends on covariates being reasonably well-scaled, and can reverse entirely when they are not.** Every jmjax-vs-JMbayes2 comparison reported earlier in this document used standardized or naturally well-scaled variables, and should be understood as characterizing this favorable regime specifically, not an unconditional guarantee.

### 7.3 Mechanism: JMbayes2 inherits scale-awareness architecturally, not incidentally

`JMbayes`'s own methodology paper (Rizopoulos, 2016, *JSS* — the same source that identified the Wishart-conjugate update in Section 3.5) documents the specific reason for this robustness:

> *"The implementation... takes full advantage of the separately fitted mixed effects and Cox models in order to appropriately define the covariance matrix of the normal proposal distributions for the random walk Metropolis algorithm. In particular, for β and bᵢ these covariance matrices are taken from the mixed model..."*

`JMbayes2`'s Metropolis proposal distributions are seeded directly from the covariance matrix of the pre-fitted `lme()`/`coxph()` maximum-likelihood models. Because MLE's Fisher-information-based covariance already reflects each covariate's natural scale by construction, `JMbayes2` never needs the *user* to standardize anything — the sampler starts with scale-appropriate proposal steps for every parameter simultaneously, before a single MCMC iteration runs.

NUTS has no equivalent source of prior scale information. It initializes generically and must *learn* an appropriate (diagonal, by default) mass matrix during warmup — evidently, for covariates of very different natural scales, this adaptation does not fully succeed within a typical warmup budget. This is a well-documented, general phenomenon in Bayesian computation, not specific to this implementation: a SAS Bayesian-procedures documentation example demonstrates the identical pattern in plain random-walk Metropolis, showing poor mixing with unstandardized covariates and good mixing once standardized. The broader Bayesian-software landscape handles this in two different ways — general-purpose HMC/NUTS tools (e.g., Stan) typically leave standardization to the user as documented best practice, while `JMbayes2` avoids the need architecturally.

### 7.4 Implication: a natural extension of the planned pre-fit input feature

This finding connects directly to a design consideration raised earlier in this investigation (Section 6.5's context and the recommendation below): `jm_fit()` already performs an internal `lme()`/`coxph()` pre-fit for empirical-Bayes prior centering, but does not currently accept user-supplied pre-fit objects as `JM`/`JMbayes2` do.

**The planned resolution should do more than accept pre-fitted objects for prior-centering convenience.** When implemented, the same pre-fit covariance matrices should also be used to **initialize NUTS's mass matrix**, giving jmjax the identical architectural scale-robustness `JMbayes2` achieves — without requiring users to manually standardize covariates as a precondition for good performance. This reframes what was originally a minor interface-consistency recommendation as a materially important fix: not merely "avoid redundant computation," but "close the specific, now-confirmed mechanism behind jmjax's covariate-scale sensitivity," using the same natural, already-planned entry point.

## 8. Recommendations for Follow-Up Work

1. **Adopt new defaults.** `random_effects_method = "wishart_gibbs"`, `rw2_implementation = "vectorized"`, and `dense_mass_spline = TRUE` (when `spline_prior = "penalized"`) each dominated their respective prior defaults on every tested metric with no observed accuracy cost. Promoting these to defaults (retaining the prior behavior as an opt-out) is a reasonable next step, pending replication (Section 9). `wishart_gibbs_centered` (Section 6) should remain experimental/opt-in given the calibration limitation documented there.

2. **Investigate and fix progress-bar overhead** (Section 3.3). This affects every user who has not explicitly disabled `progress_bar`, independent of any other finding in this report, and appears to be the single highest-impact remaining item for typical/default usage.

3. **Address the residual `β₀`–random-effects coupling** noted in Section 3.5. An experimental fix (`wishart_gibbs_centered`, Section 6) was implemented and shown to resolve the mixing problem directly (`β₀` ESS improved 4.8x at `n=500`), but introduced a distinct, quantitatively-characterized credible-interval miscalibration at larger `n` - resolving this properly (via a soft penalty or a formally-derived variance correction, per Section 6.5) remains open.

4. **Implement `lme_prefit`/`cox_prefit` input support for `jm_fit()` — elevated in priority following Section 7.** What began as an interface-consistency observation (jmjax re-fits internally where `JM`/`JMbayes2` accept pre-fitted objects) is now understood to have a second, more consequential purpose: `JMbayes2`'s own documented architecture (Section 7.3) uses exactly these pre-fit covariance matrices to seed scale-appropriate MCMC proposals, which is *why* it is robust to covariate scale while jmjax is not (Section 7.2). Implementing this feature should therefore include using the pre-fit covariance to initialize NUTS's mass matrix, not just for prior-centering convenience - directly closing the confirmed mechanism behind jmjax's covariate-scale sensitivity.

5. **Extend the Wishart-Gibbs and dense-mass fixes' validation.** All quantitative results in this report are single, unreplicated runs, consistent with the exploratory pace of this investigation. Before any figure here is cited externally, 2–3 replicate runs per configuration (and, ideally, additional sample sizes) are warranted given the session's own repeated demonstrations of real run-to-run MCMC timing variability.

6. **Document covariate standardization as an interim best practice** (Section 7) until the mass-matrix-initialization fix above is implemented. In the interim, users should standardize continuous covariates before fitting - the confirmed performance cost of not doing so (a reversal from a 2.43x jmjax advantage to a 0.76x disadvantage relative to JMbayes2) is large enough to warrant explicit documentation now, not only after the architectural fix ships.

---

## 9. Limitations

- All quantitative comparisons in this report are based on single runs without replication. Directional findings (which configuration is better, and why) are well-supported by consistent, mechanistically-grounded evidence across multiple related tests; exact multipliers should be treated as indicative rather than precise pending replication.
- All testing was conducted on CPU; GPU-accelerated execution, a plausible further lever given jmjax's JAX foundation, was not evaluated in this investigation.
- The Wishart-Gibbs and vectorized-RW2 implementations, while validated via truth recovery, are new, comparatively lightly-tested code paths relative to the package's longer-established MLE and unpenalized-MCMC paths.
- `wishart_gibbs_centered` (Section 6) carries a known, quantitatively-characterized credible-interval miscalibration for `β₀` at larger sample sizes and should not be used beyond experimental/exploratory contexts until resolved.
- Every jmjax-vs-JMbayes2 performance comparison in this report (Sections 3, 4, 5) used standardized or naturally well-scaled covariates. Section 7 confirms this represents jmjax's favorable case specifically; the reported advantages should not be assumed to hold with unstandardized, real-world-scale covariates until the mass-matrix-initialization fix (Section 7.4) is implemented.

---

## 10. Conclusion

What began as a troubling, seemingly inexplicable efficiency regression resolved into a well-characterized set of seven independent findings — six genuine, fixable inefficiencies in jmjax's own implementation, and one methodological correction in how the original benchmark comparison was framed. The systematic, one-hypothesis-at-a-time investigation discipline was essential to this outcome: several plausible early explanations (covariate cost, effect-size regime, raw data scale) were tested directly and correctly discarded, preventing the investigation from settling on a convenient but incorrect narrative.

The result is a materially improved package — faster, more sample-efficient, and better-converging than either the pre-regression baseline or the naive fixes first attempted — built on infrastructure (parallel chains, targeted mass-matrix adaptation, conjugate Gibbs updates within an HMC framework) that is positioned to scale further with additional hardware (GPU acceleration) and continued refinement, consistent with jmjax's founding motivation as a modern, JAX-based alternative to established joint-modeling software.

A natural follow-up investigation (Section 6) extended this same rigor to a residual inefficiency the core fixes had narrowed but not resolved, diagnosing it precisely, implementing a fix as an isolated, zero-risk experimental addition, and — critically — continuing to test rather than declaring victory once a speed improvement appeared, which is what surfaced the credible-interval miscalibration documented there. That outcome is itself a validation of the investigation's core discipline: the same willingness to keep questioning a promising result is what caught this limitation before it could be mistaken for an unqualified success.
