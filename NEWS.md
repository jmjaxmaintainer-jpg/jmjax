# jmjax (development)

## Changed default: the random effects are rotated for NUTS (`rotate_absorbable`)

* **The random effects are now rotated by default wherever the rotation
  applies**, which is any fit with `q >= 2`, `random_effects_corr = TRUE` and
  `random_effects_method = "nuts"`.
  - **The problem.** The likelihood identifies subject-constant fixed
    effects and the matching random-effect directions only through their
    sum. This "location degeneracy" recurs for the time slope in every
    model with a random slope, and it slowed the reported regression
    coefficients of a default fit.
  - **The fix.** Every standardized random-effect column is rotated by one
    fixed Householder matrix, so the absorbable directions become a small
    explicit site `b_gen_U`. `beta` and `b_gen_U` then share one dense
    mass-matrix block (`dense_mass_generator_beta`, also on by default).
  - **What stays the same.** The rotation is exact: the prior, the
    likelihood and every reported quantity, including `beta`, are
    unchanged. What was rotated is recorded in
    `fit$convergence$orthogonalize$rotation`.
* **Measured.**
  - Simulation (10 designs x 3 seeds): the intercept and time slope gain
    roughly 14x and 21x ESS/sec at about the same wall time.
  - Calibration (100 replicates): the same coverage as the unrotated fit.
  - `aids` and `pbc2`: the intercept and subject-constant covariates gain
    17x-28x, the time slope 2x-4x.
  - A stress grid of the designs the theory flags as weakest: above 5x in
    every cell.
  - Against `JMbayes2`, `jmjax` now matches it on the coefficients it
    hierarchically centres, where it previously trailed 18x-30x.
  - **A small cost, not a blocker.** `alpha`'s ESS/sec can fall to about 0.7x-1.0x the
    unrotated fit's in designs with about 9 visits per subject. It rises
    1.1x-2.4x with 17 or more visits, and on real data it was 2.0x (`aids`)
    and 0.95x (`pbc2`). See `vignette("jmjax-reparameterization")`.
* **Opting out.** `control$rotate_absorbable = FALSE` gives the unrotated
  fit. Explicit `TRUE` keeps the old strict behaviour, which is an error
  outside the rotation's scope. Unset (the default), the rotation is
  silently off where it does not apply, for example `q = 1` or when a
  legacy sweep option (`orthogonalize_b0/_b`) is requested.
* **Scripts that compare against "the default fit" now get the rotated
  fit.** The rotation studies (`dev/pilot_rotate_grid.R`,
  `dev/study_calibration.R`, `dev/study_realdata_rotate.R`) and the legacy
  sweep tests now set `rotate_absorbable = FALSE` for their unrotated
  baseline arm. Older benchmarking scripts do not, so to reproduce their
  published numbers, add it.

## Superseded: orthogonalize_b0 / orthogonalize_b (location-degeneracy sweep, now legacy)

*The entry below records the sweep as it was introduced. It has been
superseded by the rotation above and is kept, in `sweep.py`, only so the
development studies stay reproducible; see
`dev/notes/sweep-reparameterization.md`.*

* **`control$orthogonalize_b0`** (intercept column only) and
  **`control$orthogonalize_b`** (every random-effect column, intercept and
  slope) remove a location degeneracy between subject-constant fixed
  effects and the matching random-effect direction, which the likelihood
  identifies only through their *sum*. Implemented as a NUTS-space
  reparameterization (`_absorbable_basis()` in `mcmc_model.py` sweeps the
  absorbable directions, read from the actual design matrices, out of the
  sampled random-effect column) rather than an exact Gibbs constraint -
  see `?jm_fit` for the full mechanism and measured numbers.

* **The same degeneracy recurs for the random slope on time, independently
  of covariate count.** `orthogonalize_b0` fixes only the intercept-level
  case; `orthogonalize_b` also fixes a `beta_1`/`mean(b_i1)` degeneracy
  present as soon as there is any random slope on time, regardless of `k`.
  A replicated simulation sweep (k = 1, 4, 8, 12 extra covariates, n=1000,
  3 seeds) confirms `orthogonalize_b`'s regression-parameter ESS/sec
  advantage over the default fit is flat across k (~11-15x, not growing
  with covariate count), and is highly significant at k=8 (p=0.0064) and
  k=12 (p=0.0003) even though the aggregate all-parameters comparison is
  not (dominated by `JMbayes2`'s own random-effects-covariance R-hat
  collapsing with k - up to >10,000 at k=12 - a separate, `JMbayes2`-
  specific block-Gibbs fragility unrelated to this fix). Independently
  confirmed on the `aids` real-data benchmark (never used to build any of
  the diagnostics above): regression-only ESS/sec ratios of roughly
  5x-13x across 4 seeds.

* **Not the same fix as the earlier, retracted `wishart_gibbs_centered`**
  (see "Known limitations" below and the existing entry above): that
  option pinned only one direction of the degeneracy via an exact Gibbs
  constraint, and was found to miscalibrate `beta_0`'s credible interval.
  `orthogonalize_b0`/`orthogonalize_b` identify all absorbable directions
  from the actual design and operate as a continuous NUTS-space
  reparameterization instead. **Correctness** (fitted values and
  posterior means unaffected) is verified; **calibration/coverage has not
  yet been checked** - this is the reason the option remains experimental
  and opt-in, currently requiring `q >= 2`, `random_effects_corr = TRUE`
  and `random_effects_method = "nuts"` (other combinations error rather
  than silently no-op'ing).

* Full mechanism, literature grounding, and validation detail:
  `vignette("jmjax-reparameterization")`.

* **Calibration/coverage HAS now been checked, and the result is negative:
  `orthogonalize_b0`/`orthogonalize_b` reproduce `wishart_gibbs_centered`'s
  retracted failure mode.** `dev/study_calibration.R`'s simulation-based
  coverage study found unbiased point estimates (Corollary 1's model-
  invariance holds, confirmed per-replicate) but credible intervals for
  exactly the swept parameter(s) too narrow by roughly the theoretically
  predicted `sigma_b/sqrt(N)`: `sd_ratio` ~ 0.27-0.29 (1.0 is calibrated)
  and 95% coverage ~ 0.40 for `beta_0` under `orthogonalize_b0`/`_b` and for
  `beta_slope` under `orthogonalize_b`, against ~1.0 coverage for both the
  negative control (`alpha`, which never touches the swept subspace) and
  for these same parameters under a default fit. The mechanism: the swept
  component of `b_raw` is unidentified by the likelihood once removed from
  `b`, so it simply draws from its prior/conditional distribution, and
  nothing downstream reflects that uncertainty in `beta`'s own posterior.
  This refutes the hypothesis, previously recorded in
  `vignette("jmjax-reparameterization")` Section 4.7, that changing the
  *prior's support* (rather than imposing an exact Gibbs constraint) might
  avoid this failure mode - it does not.

* **New, opt-in analytical-correction track: `posterior_samples$beta_corrected`.**
  Since the swept component of `b_raw` is available in every fit's own
  draws (`b_std`, `sigma_b`, `L_corr`) and its relationship to `beta` is
  linear and known in closed form, the missing uncertainty can be added
  back as PURE POST-PROCESSING, with no refit: `fit_nuts()` now computes,
  whenever `orthogonalize_b0`/`orthogonalize_b` actually swept at least one
  direction, `beta_corrected = beta_orth - (the beta-shift matching the
  swept component of b_raw)` and reports it as an additional
  `posterior_samples` key alongside the untouched `beta` (new function
  `_absorbable_generator_matrix()` in `mcmc_model.py`; the standardized-
  covariate back-transform in `jm_fit()` now also carries `beta_corrected`
  when present, using the same transform already applied to `beta`). This
  is a SEPARATE code path from the orthogonalize machinery itself - it
  changes no sampled quantity - added specifically to test whether it
  restores arm-A-equivalent calibration while keeping the efficiency
  gains; empirical validation is in progress (see
  `dev/study_calibration.R`'s `*_corrected` estimands) and is not yet
  concluded. **New: `dev/verify_beta_correction.py`** - 34 checks
  confirming, independent of any MCMC run, that the beta-shift matrix
  satisfies its defining identity and that the derived correction exactly
  reproduces the unconstrained model's fitted values, on both coordinate-
  aligned and spline/orthogonal-polynomial (Example 1 style) designs, and
  with multiple random-effect columns swept simultaneously.

* **The absorbable-basis construction is now basis-independent, closing a
  silent failure mode.** `_absorbable_basis()` searched one fixed-effects
  column at a time, asking whether *that column* is a subject-constant
  multiple of the random-effect column. But a shift in `b[:, q]` is
  absorbed by `beta` whenever some *linear combination* of `X`'s columns
  reproduces it, so the absorbable set depends only on `X`'s column
  **space**, not on the basis `model.matrix()` happens to emit. The two
  questions differ exactly when the longitudinal mean uses a spline or
  orthogonal-polynomial basis: with `y ~ poly(time, 2)` or a natural
  spline, no single column is a multiple of `time`, so the per-column
  search returned nothing for the slope column and the `beta_1`/
  `mean(b_i1)` degeneracy was left fully in place - while the intercept
  column still produced a basis, so the "no absorbable direction" warning
  did not fire either. A new `_absorbable_basis_exact()` computes the
  absorbable set directly (one SVD of a `[sum n_i, p]` matrix; ~0.06s at
  n = 8000) and is now authoritative; the per-column search is retained
  for attribution, and a `RuntimeWarning` reports any disagreement. Every
  returned direction is verified against the defining identity before use,
  so the basis is sound by construction rather than by tolerance choice.
  **Results already reported are unaffected**: on designs that spell each
  term as its own column - every design in `dev/sim_joint.R` and both
  study scripts - the two constructions return the same subspace, which
  `dev/verify_absorbable_basis.py` checks directly.

* **Condition (S) is now checked rather than assumed.** Absorbability was
  established on the longitudinal grid, but `b` also enters the
  likelihood through the shared trajectory `m_i(t)` at the event and
  quadrature times. `_structural_extension_residual()` verifies the same
  proportionality holds there, using arrays the backend already builds,
  and warns if it does not - a design can satisfy the grid test by
  coincidence (few observations per subject) while failing off-grid, in
  which case the reparameterization would change the model rather than
  only its coordinates.

* **New: `dev/verify_absorbable_basis.py`** - 34 checks covering soundness,
  the completeness repair, the consensus imputation for subjects whose own
  data cannot determine their ratio, the Condition (S) check on a design
  constructed to violate it, and a randomized sweep confirming the exact
  construction never loses a direction the per-column search found.
  `fit$convergence$orthogonalize` gains `n_directions_column_search`, so
  the spline/polynomial case is detectable programmatically rather than by
  reading warning text. **`dev/check_orth_repair.sh`** runs the whole
  verification end to end in four stages, cheapest first, so a failure
  stops before the expensive part: static checks, the algebra, a reinstall
  plus confirmation that the edit actually reached the INSTALLED copy of
  `mcmc_model.py` (it is copied at install time, so editing the source
  tree alone changes nothing), and finally `test-orthogonalize.R`.

* Formal treatment - definition of the absorbable set, exact
  characterization, soundness and completeness propositions, and the
  worked counterexample above - is in
  `vignette("jmjax-reparameterization")`, Section 4.

* **Test coverage added** (`tests/testthat/test-orthogonalize.R`): the
  option's supported-configuration guards, that a default fit reports
  `fit$convergence$orthogonalize = NULL` while an orthogonalized one
  reports which columns were actually constrained, that fitted population
  estimates agree with the default fit within Monte Carlo tolerance, and
  a regression guard on `beta_1`'s ESS improving materially under
  `orthogonalize_b`. Previously validated only by one-off `dev/` scripts.
  `fit$convergence$orthogonalize` is new with this: a structured,
  R-readable record of what `orthogonalize_b0`/`orthogonalize_b` actually
  swept (`NULL` unless requested) - added because the existing
  `RuntimeWarning` describing this arrives on stderr from the Python
  backend, not as an R condition, so it is invisible to
  `expect_warning()`/`tryCatch()` on the R side (same reason
  `fit$convergence$warm_start` exists rather than relying on its own
  warning text).


## Warm start

* **The MCMC warm start now supplies the spline baseline hazard**, and under
  `spline_prior = "penalized"` this fixes a defect that made the warm start
  worse than useless on that prior.

  The warm start supplied `beta`, `sigma_e`, `sigma_b`, `L_corr` and
  `b_std` from the `lme()` pre-fit, and left the spline block to
  `init_to_uniform`. Under `independent` that is harmless, because `W` is
  sampled directly. Under `penalized` it is not: `W01 ~ Normal(0, 10)` is
  initialised uniformly on `[-2, 2]`, so the implied slope
  `W01[2] - W01[1]` can be 4, and the RW2 construction extrapolates it
  linearly - `w_rest[k] = W01[2] + k*(W01[2] - W01[1])` reaches about 22 by
  the fifth coefficient. The hazard is `exp(B W)`, and `exp(22)` is 3.6e9.

  Measured on the same data with the same supplied sites:

  | `spline_prior` | potential at warm start | at uniform start | accepted |
  |---|---|---|---|
  | `independent` | 860.2 | 2087.5 | yes |
  | `penalized` | 1,533,123,067 | 10,763.9 | no |

  The warm start is now seeded from the same Weibull-projected spline
  coefficients the maximum-likelihood path already used
  (`survreg` → log hazard at the event times → `lm.fit` onto the spline
  basis). Under `penalized` the RW2 construction is inverted so that the
  sampled sites `W01`, `z_step` and `tau_w` reproduce those coefficients
  exactly, with `sigma_w` chosen as the SD of the second differences so
  `z_step` starts on the unit scale its `N(0, 1)` prior expects.

  `alpha` is now supplied as 0 for the same class of reason: the hazard
  carries `exp(alpha * m)`, and a uniform-random `alpha` multiplying a
  correctly warm-started trajectory blows up the same way `W01` did.

  This was found only because enabling float64 changed the fixed PRNG draws
  in the self-check and the accident stopped holding. Under float32 the
  penalized warm start was accepted by a margin of 7.6% (2139.0 against
  2316.1) where it should have been better by a factor of two or more.

* **`control$init_values` supplied by the caller is no longer overwritten.**
  `jm_fit()` replaced it unconditionally with its own `lme` warm start, so
  user-supplied starting values were silently discarded and
  `fit$convergence$warm_start` then reported on sites the caller had never
  supplied. The warm start now only runs when no starting values were given.

* **The warm-start rejection test asserts on the fit object, not on a
  warning.** It wrapped the call in
  `expect_warning(regexp = "warm start REJECTED")`, but that warning comes
  from Python's `warnings.warn` inside the backend and arrives on stderr
  rather than as an R condition - so the text appeared in the test log
  while the expectation failed. `fit$convergence$warm_start$used` exists
  for exactly this, and the test now also checks that the corrupted values
  actually reached the model (`sites` is `beta`, `sigma_e` and nothing
  else), which was not true while `init_values` was being overwritten.

* **`test-scale-time.R` compares against Monte Carlo error** rather than a
  fixed 2% relative tolerance. The old bar was a statement about sampler
  noise rather than about reparameterization invariance, and whether it
  passed depended on how large a parameter happened to be in the generated
  data - `alpha` near -0.13 made 2% into 0.0026, well below what 250
  warmup / 250 samples can resolve. The bound is now
  `4 * sqrt(mcse_0^2 + mcse_1^2)` with `mcse = SD / sqrt(ESS)`, which
  cannot fail from noise alone while a genuine scaling bug (a ratio of 5 or
  25) misses it by orders of magnitude.

* **Three hardcoded `jnp.float32` casts removed** from the backend, now that
  it runs in double precision. The consequential one was in `common.py`: the
  probe that decides whether the pre-fit starting values are usable was
  evaluating finiteness in single precision on behalf of a double-precision
  optimizer.

## Numerical precision

* **The Python backend now runs in double precision (float64) by default.**
  This is a change to every number the package produces, and it is a
  behaviour change rather than a bug fix, so it is listed first.

  JAX defaults to float32 and silently downcasts any float64 array handed
  to it. Nothing in jmjax ever chose that default, and nothing recorded
  it, so every fit and every benchmark in the package's history was
  produced in single precision without saying so. Three consequences,
  measured rather than assumed:

  - The **maximum-likelihood path** was the worst affected and the least
    visible. `common.py` computes the gradient and Hessian in JAX, casts
    them into `np.float64` buffers, and hands them to `scipy`'s L-BFGS-B,
    whose default tolerances assume roughly 1e-15 relative accuracy - it
    was receiving about 1e-7. The Hessian is then inverted for the
    covariance matrix, so the reported standard errors were float64-shaped
    containers holding float32-quality numbers.
    Measured on the same data from the same starting values, `n = 400`:

    | | float64 | float32 |
    |---|---|---|
    | log-likelihood | -2593.641 | -2594.688 |
    | grad_max at the reported optimum | **0.0005** | **0.7822** |
    | iterations | 127 | 80 |
    | `converged` | TRUE | TRUE |

    At a stationary point the gradient is near zero. float32 stopped
    three orders of magnitude short, after 80 iterations instead of 127,
    and reported success. This is the pathology `common.py`'s own
    docstring already described - *"stopped after 5 iterations with
    converged = TRUE ... Nothing in the output flagged it"* - which had
    been attributed to parameter scaling. `parscale` helped, but it was
    not the whole story.
  - **MCMC is faster in float64 at every size measured**, not only large
    ones - 3 seeds each:

    | n | wall-time speedup | ESS_alpha/sec float64 | float32 |
    |---|---|---|---|
    | 200 | 1.20-1.26x | 254-353 | 168-213 |
    | 1,000 | 1.29-1.40x | 46-92 | 27-45 |
    | 8,000 | 4.24x | 1188 (ESS) | 4 (ESS) |

    The mechanism differs by size, which is worth recording: at
    `n = 8,000` float32 noise drove NUTS into roughly four times as many
    leapfrog steps, but at 200 and 1,000 the step counts are essentially
    identical (63.0 against 63.0) and float32 is simply slower in wall
    time for less effective sample. The only divergence in the whole
    sweep was float32's. An earlier version of this entry cited only the
    `n = 8,000` figure and flagged small-`n` as an open question where
    float64 might lose; it does not.
  - `exp()` overflows float32 at about 88 against float64's 709, and the
    hazard is `exp(linear predictor)`.

  R has no single-precision numeric type, so float64 is also what a user
  calling this from R already assumes they are getting.

  Opt out with `jmjax_setup(enable_x64 = FALSE)` or `JMJAX_ENABLE_X64=0`,
  before the backend is first imported in a session. Doing so now emits a
  warning, because it changes the numbers.

* **Fits now record the precision that produced them**, in
  `fit$convergence$precision`, shown by `print()`. This exists because of
  a concrete failure: the benchmark corpus was produced under float32 and
  nothing in any saved fit said so, which left the whole corpus suspect
  rather than selectively re-checkable. Fits saved by earlier versions
  have no such field and print exactly as before.

* **`jmjax_setup()` gains an `enable_x64` argument**, with the same
  lifecycle as `num_devices`: it must be set before the backend is first
  imported, and calling it afterwards warns rather than silently failing.
  Deliberately not a `jm_fit(control = )` option, since a per-call
  argument could not take effect from the second fit onward.

* **`dev/bench_x64.R` and `dev/bench_x64.sh`** measure the trade-off
  across sizes on a given machine, covering both the MCMC and
  maximum-likelihood paths, with pre-registered criteria.

## CRAN readiness

* **jmjax will no longer create a Python environment during a package
  check.** `.get_backend()` previously built the `r-jmjax` virtualenv
  whenever none was found, which in `R CMD check` on CRAN means a ~150MB
  download in a non-interactive session - against policy and an automatic
  rejection. It now refuses when `interactive()` is FALSE and `NOT_CRAN` is
  not `"true"` (testthat's convention: `devtools::test()` and r-lib's CI
  actions set it, CRAN does not), with an error naming
  `jmjax::jmjax_setup()` and the `JMJAX_ALLOW_SETUP=true` override.

  This is also what makes the suite CRAN-sized without annotating every
  test: with no jax available, `skip_if_no_backend()` skips every
  backend-dependent test for a stated reason instead of trying to install
  one. Local behaviour is unchanged.

* **`skip_if_slow_mcmc()` now also calls `skip_on_cran()`**, covering the
  remaining case - a maintainer running `R CMD check` on a machine where
  the venv already exists, so the backend works and the tests would
  otherwise run in full. The suite's MCMC blocks alone took 482s on a 2026
  laptop against CRAN's ~10 minute budget for the entire check.

  Three tiers now: CRAN skips slow MCMC; `JMJAX_SKIP_SLOW_TESTS=true` gives
  a fast subset for local iteration and CI pull requests; everything else
  runs the full suite.

* Added `.Rbuildignore` rules for study output. The benchmark scripts write
  results to `getwd()`, so running them from the package directory left
  `generator_misspec_n1000.rds` and friends inside the built tarball -
  which silently added a dependency on R >= 3.5.0, since serialize version
  3 cannot be read by older R. Anchored to the top level so `inst/extdata`
  is unaffected.

  Vignettes (all chunks `eval = FALSE`) and examples (all `\dontrun{}`)
  were already check-safe and are unchanged.

## Development tooling

* **New: `dev/check_undefined_names.py`** - a static check for names a
  Python function reads but never binds. `python -m py_compile` cannot
  catch these: the file compiles, and the `NameError` only appears when
  that line runs. One such bug shipped in this package and broke every
  `spline-PH-mcmc` fit until it was caught by running one. The check is an
  AST pass, deliberately biased toward silence (closure variables,
  comprehension targets and star-imports are all treated as bound), so a
  clean run is weak evidence while a dirty run is strong evidence. It runs
  in well under a second and is stage 1 of `dev/check_orth_repair.sh`.

## New

* **`control$mcmc_warm_start` (default `TRUE`) starts the NUTS chains from
  the internal `lme()` pre-fit** instead of NumPyro's `init_to_uniform`,
  supplying `beta`, `sigma_e`, `sigma_b`, `L_corr` and the per-subject
  random effects as `b_std` (the BLUPs pushed through the model's own
  Cholesky factor). This is what JMbayes2 has always done with the
  `lme`/`coxph` objects it is handed.

  It matters at scale, where a uniform start leaves every one of ~16,000
  random-effect coordinates in the wrong place and warm-up has to reach the
  typical set and adapt a step size in the same 500 iterations. On the two
  sampler seeds that failed at n = 8,000 from a cold start: bad fits **2/2
  to 0/2**, median divergences **98 to 0**, median ESS for `rho` **4 to
  101**, median leapfrog steps 155 to 111, and 28% faster. At n = 1,000
  across five seeds nothing measurable changed - 63 steps, zero
  divergences, 0/5 bad either way - which was the pre-registered
  do-no-harm condition for enabling it by default.

  Note `init_to_median` is NOT the fix it sounds like, and was measured
  doing harm: in d dimensions a standard normal's typical set is a shell at
  radius sqrt(d), so at 16,000 dimensions the prior median sits at radius
  ~0 against a shell at ~126, while `init_to_uniform`'s U(-2,2) lands at
  ~146. The default is accidentally well suited to this model.

  **The cost, shared with JMbayes2:** chains that start together agree more
  readily, so R-hat is a less sensitive diagnostic - Gelman-Rubin assumes
  overdispersed starts. Set `FALSE` to restore it. The evidence above does
  not rest on R-hat: ESS, divergences and step counts are within-chain
  measures a shared start cannot flatter.

  Per-chain jitter (`control$warm_start_jitter`, default 0.1) is applied in
  **unconstrained** space, so SDs stay positive and `L_corr` stays a valid
  Cholesky factor automatically - no special cases, unlike jittering
  constrained values and having to exclude the covariance matrix.

  Three layers of checking, because an inverse Cholesky is exactly the kind
  of transform this package has got wrong before: the algebra was verified
  offline (500 random SPD matrices, `L L' = D` to 3.8e-16 and the `b_std`
  round-trip to 3.1e-16); `jm_fit()` re-runs both checks on the actual fit;
  and the backend compares the log-density at the warm start against a
  uniform one and **discards the warm start if it is not better**. The
  first version of this feature omitted `beta` and was 267,000 log-density
  units WORSE than random - a tight `sigma_e` supplied alongside a
  uniform-random `beta` - and the self-check caught it before it ran.
  Outcome is recorded in `fit$convergence$warm_start`.

## Bug fixes

* **`control$rw2_implementation` now defaults to `"vectorized"` (was
  `"scan"`), because `"scan"` cannot run on the Python stack this package
  pins.** numpyro's `scan_wrapper` builds its carry with
  `device_put((0, rng_key, init))` - `device_put` applied to a tuple holding
  a Python int - which produces a weakly-typed `int32`. From jax 0.4.30
  onward that element's tangent is `float0` on one side of `lax.scan`'s
  backward pass and `int32` on the other, so every gradient through the scan
  fails with `body_fun output and input must have identical types`. This is
  jax-ml/jax#22045. Since `jmjax_setup()` pins jax 0.4.30 and numpyro 0.15.0,
  **every correct install of jmjax was unable to run
  `spline_prior = "penalized"` at all** - the configuration used by every
  benchmark script in the project.

  The new default is not a lossy workaround: the vectorized branch solves the
  same linear recursion in closed form, verified against the sequential
  version over 200 random cases (worst relative difference 4.9e-15), and the
  replication that introduced it measured both at an identical 63.0 mean
  leapfrog steps. `"scan"` remains selectable on numpyro >= 0.17.0, which
  carries the upstream fix (commit 4704656, "remove unnecessary device_put").

* **`.get_backend()` now selects the Python environment before the first
  `py_*` call.** reticulate binds one interpreter per R session at the first
  Python call and cannot be re-pointed afterwards. The old code probed first
  (`py_run_string()` to extend `sys.path`) and only created or selected the
  venv if that probe failed - by which time the probe had already bound
  Python to whatever the system offered, so `use_virtualenv(required = TRUE)`
  threw and the error was swallowed by the caller. On a fresh machine this
  built the `r-jmjax` venv during a test run and then skipped all 79
  backend-dependent tests with "Python backend not available". A venv that
  exists but is missing its dependencies is now rebuilt rather than bound to,
  and an unrecoverable binding reports which interpreter is attached and that
  a restart is required.

## Testing

* **Added `tests/testthat/test-penalized-spline.R`.** `spline_prior =
  "penalized"` appeared zero times in the test suite and in every benchmark
  script, so the breakage above sat behind a fully green 410-test run and was
  found only when a benchmark failed thirty times in a row. The new file
  exercises the penalized path with both `dense_mass_spline` settings, and
  includes a self-activating equivalence check between the two RW2
  implementations that skips (with the reason) on stacks where `"scan"`
  cannot run and turns itself on where it can.

* The backend skip helper now reports the underlying error in the skip
  message instead of discarding it.

## Packaging

* Added `.gitignore`, `.Rbuildignore`, `LICENSE`, `Authors@R` and
  `SystemRequirements`; `License:` corrected to `MIT + file LICENSE`.
  Added a GitHub Actions workflow that provisions the Python environment
  before `R CMD check` and fails the job if any backend test skipped - note
  that declaring dependencies via `Config/reticulate` would NOT serve this
  purpose, since reticulate's `configure_environment()` returns early in
  non-interactive sessions.

## New: optional EM warm start for the MLE paths

* **`control$em_warm_start = TRUE`** runs a short joint EM phase before
  the optimizer, following `JM`'s structure: E-step, closed-form
  longitudinal M-step, short BFGS step on the survival block, with JM's
  schedule (`maxit` 20 then 4) and stopping criteria (`tol1 = 1e-3`,
  `tol2 = 1e-4`, `tol3 = sqrt(eps)`).

* **Measured on `prothro` with a spline baseline**, where L-BFGS-B stops
  after 5 iterations 78 log-likelihood units short:

  | start | loglik | alpha |
  |---|---|---|
  | cold L-BFGS-B | -14083.14 | -0.0008 |
  | EM-warmed L-BFGS-B | -14004.49 | -0.0346 |
  | cold BFGS | -14004.80 | -0.0391 |
  | EM-warmed BFGS | **-14001.88** | **-0.0417** |

  against `JM`'s -0.0400 and jmjax's own MCMC -0.0411. 78.7 of the 78.3
  missing units recovered, with no iteration increasing the objective.

* **Default `FALSE`, unlike `JM` which always runs EM.** On fits that
  already converge it costs about 2.2x the runtime and changes nothing,
  and it can hand the optimizer a WORSE-conditioned point: EM stops on
  relative parameter change, so it may halt where the gradient is still
  substantial. Measured on a fit that converged cleanly from cold
  (`grad_max` 0.62), the EM-warmed run took 0 optimizer iterations and
  ended at `grad_max` 1.75 - above the threshold, so `converged` became
  FALSE for an essentially identical log-likelihood.

  `JM` runs EM unconditionally because it cannot distinguish a stalled
  optimizer from a converged one. jmjax can, via
  `fit$convergence$grad_max`, so the recommended use is
  diagnostic-driven: refit with `em_warm_start = TRUE` when a fit reports
  `converged = FALSE`.

* **Scope:** wired into the base `q = 1` Weibull path only - the single
  fitting function routed through the shared likelihood builder. On other
  models the option is currently inert.

* `em_phase()` refuses any step that raises the objective, and returns the
  better-conditioned of its endpoint and its starting values when the two
  likelihoods tie.

## `converged` is now a gradient criterion, not scipy's exit code

* scipy's `success` means different things per method and is wrong in
  both directions on the same data: `TRUE` for L-BFGS-B on `prothro` at
  `grad_max` 31.9 while 78 log-likelihood units short, `FALSE` for BFGS
  at `grad_max` 2.29 at the good optimum (its `gtol` default of 1e-5 is
  unreachable on a log-likelihood of 14000).

* `fit$convergence$converged` now reports whether the largest gradient
  component is small relative to the log-likelihood. When an EM phase has
  run, `JM`'s own rule is used as well - the optimizer succeeded or it
  improved on what EM reached. scipy's raw status remains available as
  `fit$convergence$optimizer_success`.


## MLE optimizer default changed to BFGS

* **`control$opt_method` now defaults to `"BFGS"`** (was `"L-BFGS-B"`).

  L-BFGS-B failed on the `prothro` data with a spline baseline: it stopped
  after 5 iterations reporting `converged = TRUE` with zero non-finite
  standard errors and a finite log-likelihood, having left 78
  log-likelihood units behind (-14083.14 against BFGS's -14004.80) and
  returning an association parameter of `-0.0008` where `JM` (-0.0400),
  BFGS (-0.0391), `trust-constr` (-0.0390) and jmjax's own MCMC (-0.0411)
  all agree. Tightening `ftol` changed nothing, and `maxiter` made no
  difference at 5, 20, 100 or 500 - a line search quit and was reported
  as convergence.

* **Evidence.** Across five dataset/method combinations L-BFGS-B was
  short of the best log-likelihood twice and never won; BFGS won twice
  and `trust-constr` three times without ever being short. The likely
  mechanism is L-BFGS-B's limited-memory curvature approximation (10
  correction pairs by default) on a problem with ~19 densely coupled
  parameters, 9 of them spline coefficients: handed an excellent
  EM-derived starting point it stopped after 2 iterations at
  `grad_max 6.65`, while BFGS ran 56 to 2.18.

  `JM` uses `method = "BFGS"` in every one of its fitters - `splinePHGH`,
  `piecewisePHGH`, `piecewiseAFTGH`, `flexCPH` - and `"L-BFGS-B"` in
  none.

* **New: `fit$convergence$grad_norm` and `grad_max`.** A stationary point
  has a small gradient; a failed line search does not, and `converged`
  reflects only the optimizer's exit status. `jm_fit()` now warns when the
  largest gradient component exceeds a threshold scaled to the
  log-likelihood. On `prothro` that separates the bad fits (`grad_max`
  17.4 and 31.9) from the good ones (2.3 and 6.2).

* **`init_theta` now accepts a prior fit's `$estimates`**, as documented.
  Previously the list was passed through unchanged and raised
  `float() argument must be a string or a real number` - the values need
  `log()`/`atanh()` transforming and reordering into the backend's theta
  layout, which is now done for you.


## `random_formula = NULL` no longer selects Z by column position

* **`Z` is now always built from an explicit formula.** `NULL` is
  materialized at entry into `~1` (q=1) or `~time` (q=2), and the
  `Z = X[, 1:q]` slicing convention is gone.

  That convention was correct only by coincidence of column order:
  `y ~ time + x` sliced `[intercept, time]` correctly, but `y ~ x + time`
  sliced `[intercept, x]` - silently putting a random effect on the
  baseline covariate instead of on time. jmjax guarded against this by
  refusing the configuration; the guard now has nothing to guard.

* **Verified before removal.** An assertion comparing the slice against
  the formula-built `Z` ran across the whole suite: they agree to `1e-12`
  on `Z_long`, `Z_time_surv` and `Z_time_quad`. Estimates are unchanged.

* **Why it mattered beyond tidiness.** `NULL` carried two meanings -
  "build Z by slicing" and "the user did not specify" - and every bug
  found while testing `scale_time` came from that overloading: a
  substitution that missed the NULL branch, a fix that defeated the
  ambiguity guard, and a test written against a configuration the package
  rejects. Roughly 70 lines came out, including five workarounds that
  existed solely to manage the sentinel.

* The ambiguity guard is retained for one release. Removing it will make
  `y ~ x + time` at q=2 fit correctly rather than error.

## `control$scale_time` now defaults to `"auto"`

* Time units are a computational artifact, not a modelling choice.
  Recording days instead of years does not change the inference, only
  whether NUTS can sample it - and the cost of leaving this off was
  measured, not hypothetical: on the `epileptic` data with time in days
  (random-effect SD ratio 796:1), an unscaled fit returns max R-hat 22
  and an association parameter of 7.1 where four independent routes agree
  on 0.12. Nothing in the output said the units caused it.

* **`"auto"` declines rather than errors** when the random-effects
  transformation is not diagonal (`log(time)`, a fixed-knot spline), so
  the new default cannot turn a working fit into a failure.
  `scale_time = TRUE` or a numeric divisor keeps the strict behaviour.

* **It changes no reported number.** Scaling is a reparameterization;
  coefficients are returned on the original scale through the same
  `qr.solve` projection that handles `standardize_covariates`. A test
  asserts the default and `scale_time = FALSE` agree on every parameter.

* Measured across five datasets: `epileptic` went from 1023 leapfrog
  steps and 292s to 63 steps and 41s; four datasets with SD ratios from
  1.9:1 to 8:1 changed by 0-15%, within noise.


## Correctness fix: standardization with interactions

* **`standardize_covariates` returned wrong coefficients when the formula
  contained an interaction involving a standardized variable.** The
  back-transformation made two elementwise adjustments - correct the
  intercept, divide the standardized column by its SD - which is right for
  an additive formula and wrong once the variable appears in an
  interaction, because `b3 * t * (x - m)/s = (b3/s) * t * x - (b3*m/s) * t`
  leaks into the time main effect. On `y ~ time * x` with `x ~ N(50, 10)`
  the time coefficient came back 0.700 against a true 0.500, and the
  interaction 0.0399 against 0.004.

  This was enabled by DEFAULT, so any model with such an interaction was
  affected. The whole test suite used `y ~ time + x`, which cannot expose
  it.

* **The fix** treats standardization as what it is - a linear map on the
  design, `X_std = X %*% A` - and recovers `A` once with `qr.solve`, so
  `beta_orig = A %*% beta_sampled` handles interactions, multi-way terms
  and polynomials with no formula parsing. `qr.solve` rather than the
  normal equations: `solve(crossprod(X), ...)` squares the condition
  number, and an uncentred design is exactly the ill-conditioned case that
  motivates standardizing.

* Three regression tests added, including an equivalence check that
  standardizing on and off give the same coefficients.

## New: `control$scale_time` (opt-in)

* Substitutes `I(time/c)` into `long_formula` and `random_formula`,
  scaling every time-derived design column while leaving survival times,
  spline knots and quadrature nodes raw. Coefficients are returned on the
  original scale through the same projection used above.

* **Measured.** `epileptic` (random-effect SD ratio 796:1, time in days):
  1023 leapfrog steps and 292s became 63 steps and 41s, max R-hat 22 ->
  1.01. Four datasets with ratios 1.9:1 to 8:1 changed by 0-15% in step
  count, within noise - so it is opt-in, not default.

* A three-arm experiment established that scaling the design columns alone
  matches scaling the time variable everywhere, to the leapfrog step. The
  survival submodel is scale-insulated, so `alpha`, the baseline hazard
  and `rho` need no back-transformation.

* **`jm_fit()` now reports the random-effect SD ratio** from its `lme`
  pre-fit when it exceeds 50:1, and points at
  `fit$convergence$mean_num_steps` - a value near `2^k - 1` for the
  maximum tree depth means the sampler exhausted its trajectory budget
  every iteration. No ratio threshold is documented: 8:1 sampled fine in
  testing and 796:1 failed, with nothing measured between.

* Three bugs were found and fixed during implementation, each identified
  by the ratio of a parameter with scaling on versus off: a correction
  living in the MCMC-only branch (ratio = c), the same correction applied
  in the wrong direction (ratio = c^2), and a first implementation that
  divided `data_long[[time_var]]` rather than rewriting the formula -
  `time_var` is also the slot `build_time_design()` fills with the
  SURVIVAL times, so the trajectory was evaluated at raw event times in a
  formula expecting `time/c`.


## MLE optimizer: parameter rescaling (`parscale`)

* **`control$parscale` now defaults to `0.01`**, matching `JM`. jmjax
  optimizes `u = theta/s` and maps back, chain-ruling the gradient - a
  pure change of variables, so the objective is unchanged and only the
  step geometry differs. scipy has no `parscale`, so this is implemented
  explicitly.

* **What it fixes.** jmjax optimizes `log_sigma_e`, `log_sigma_b` and
  `atanh_rho`, so `d/d(log sigma_b)` carries a factor of `sigma_b`. On
  the `prothro` data (`sigma_b` near 18, the largest tested) that made one
  gradient component an order of magnitude larger than the rest, and
  L-BFGS-B terminated at iteration 0 - with a FINITE objective and a
  FINITE gradient - returning the starting values as though they were
  estimates. With `parscale`, it converges in 54 iterations to
  `alpha = -0.0384`, matching `JM`'s `-0.0384` exactly. `heart.valve`
  likewise goes from a non-fit to `alpha = 1.1856` against `JM`'s 1.2241.

* **No regression on cases that already worked.** Across 3 simulated
  seeds (both baseline hazards) and the AIDS data, estimates shift by
  1e-4 to 1e-3 - optimizer tolerance - with no convergence regressions
  and no new non-finite standard errors. Set `parscale = NULL` to disable.

* **`control$opt_method`** selects the scipy optimizer (default
  `"L-BFGS-B"`). Worth noting `"BFGS"` and `"CG"` are NOT robust without
  `parscale` - both diverged badly on `prothro`, reaching
  `alpha = 1033.87` with every standard error non-finite. The rescaling
  matters more than the method.

* **The starting-value guard now checks the GRADIENT as well as the
  objective**, and warns explicitly when neither the pre-fit nor the
  default start is usable - a case that previously produced a silent
  non-fit whose "estimates" were just the starting values.

* **Weibull MLE starting values are now wired through.** The R side had
  been computing `lme()`/`coxph()`-derived starting values for both MLE
  methods, but only the spline backend read them; the Weibull backend
  ignored all of them. Also new: `log_lambda0` and `shape` are initialized
  from a `survreg()` fit rather than fixed at `-2.0` and `1.2`.

* **Known gap:** `JM` additionally runs an EM phase before any
  quasi-Newton step, which cannot fail a line search at all. jmjax does
  not, and that remains the more robust approach.


## Pre-fit interface refinements

* **`data_surv` is now optional in `jm_fit_prefit()`.** It is recovered by
  evaluating the data symbol recorded in `cox_object$call`, matching
  `JMbayes2::jm()`, and required explicitly only when that fails. Recovery
  works when `coxph()` was called at top level against a data frame that
  still exists under the same name; it fails when the object was renamed,
  removed, or created inside a function that has since returned, because
  `eval()` looks a symbol up in a frame rather than retrieving a stored
  copy. An earlier version required it unconditionally on the stated
  grounds that "coxph() does not retain its data" - imprecise: `$y` holds
  the `Surv` response by default and `$x` the design matrix with
  `x = TRUE`. What is genuinely absent is the subject identifier, which is
  not a model term and is exactly what aligns survival with longitudinal
  records.

* **The supplied `lme` object is now reused rather than refitted.**
  Previously `jm_fit_prefit()` used it for structure (formula, grouping
  variable, `q`, `data_long`) and then `jm_fit()` fitted a fresh `lme()`
  for the starting values - wasting a fit, and silently discarding any
  `lmeControl` settings the user had chosen. The object is now passed
  through and stripped before anything crosses into Python.

* **MLE starting values are derived from the pre-fits** (`beta`,
  `sigma_e`, `sigma_b`, `rho` from the mixed model; `alpha` and `gamma`
  from a two-stage Cox fit, mirroring `JM`'s `initial.surv()`; the spline
  baseline from a Weibull `survreg()` projected onto the basis, also
  mirroring `JM`). Measured across 5 seeds x 3 knot counts on simulated
  data: optimizer iterations fall to about 0.85-0.91x with identical
  estimates and no convergence cost. `control$init_from_prefit = FALSE`
  restores the previous hard-coded constants.

  A one-off 3.5x regression on the AIDS data at 7 knots did not reproduce
  across seeds (worst ratio 1.08), so it was specific to that hazard shape.

* **A non-finite starting point is now rejected.** `guard_initial_theta()`
  evaluates the objective at the pre-fit start and falls back to the
  defaults if it is not finite. This catches numerical overflow; it does
  NOT catch a start that is merely poor, which can still stop the
  optimizer in a wrong region while reporting convergence. See the new
  "Check the pre-fits" section in `?jm_fit_prefit`.

* **New: time-scale consistency warning.** `jm_fit()` warns when the
  survival times exceed the longitudinal time range by more than 5x. The
  joint model evaluates the longitudinal trajectory AT the survival times,
  so rescaling one without the other (months to years, say) silently
  produces wild extrapolation rather than an error. This was found the
  hard way - see `jmjax_cross_package_benchmark.md` section 5.


## New: longitudinal baseline covariates for `spline-PH-aGH`

* `long_formula = y ~ time + age` now works with `method = "spline-PH-aGH"`,
  at both `q = 1` and `q = 2`. Previously it was rejected by an R-side guard
  that listed only `weibull-PH-aGH` among the MLE methods.

* **The backend already supported it.** Unlike survival-side baseline
  covariates - which add a `gamma` parameter block and needed genuinely new
  likelihood functions - a longitudinal covariate adds no new parameters. It
  only makes `beta` longer, and the spline MLE's theta layout
  (`make_theta_layout()` gives `"beta": (0, p)`) and likelihood
  (`X_long_i @ beta`, `X_time_surv_i @ beta`, `X_time_quad_i @ beta`) were
  already generic in `p` at both `q = 1` and `q = 2`. The R-side design
  builders were likewise already generic. The guard was stale rather than
  protective, so this is a one-line relaxation plus truth-recovery tests
  rather than new numerical code.

* Surfaced while validating `standardize_covariates` on an MLE method: the
  natural test model `y ~ time + x` could not be fitted with
  `spline-PH-aGH` at all.

* Still not supported for the MLE methods: longitudinal covariates combined
  with `functional_forms` channels beyond the default value-only
  association. At `q = 2` an explicit `random_formula` remains required
  (method-agnostic, so that the random-effects design does not depend on
  `long_formula`'s column order).


## New interface: fit from pre-fitted component models

* **`jm_fit_prefit(lme_object, cox_object, data_surv, time_var, ...)`** takes
  fitted `nlme::lme()` and `survival::coxph()` objects directly, in the style
  of `JMbayes2::jm()`, instead of restating the formulas, grouping variable
  and random-effects structure. It validates, extracts, and delegates to
  `jm_fit()`, which is unchanged - the formula interface remains fully
  supported and is still the better choice when you do not already have
  fitted component models.

* **Why it is worth having.** Fitting the two submodels first is good
  practice regardless: convergence failures, questionable random-effects
  structures and implausible variances surface in seconds rather than after
  a long MCMC run. It also makes the standard diagnostic comparison - joint
  estimates against the separate fits - natural rather than an extra step.

* **It closes the last prior gap.** `coef(cox_object)` centres the prior on
  the survival covariate effects, matching `JMbayes2`'s `mean_gammas`
  (verified: `coef(coxph)` = -0.06697697 against `mean_gammas` = -0.067 on
  the same fit). The formula interface has no coxph object to read, so it
  continues to use a zero-centred prior. With this, every prior component
  compared against `JMbayes2` now matches.

* **`data_surv` is still required**, unlike `data_long`. `coxph()` does not
  retain the data frame it was fitted to - with `x = TRUE` it stores a
  design matrix but not the `Surv()` response columns and not the subject
  identifier. The fitted object records the *symbol* of its data argument,
  but evaluating a caller's symbol is fragile, so requiring it explicitly is
  the honest option. `lme()` does retain its data.

* **Structures jmjax cannot represent raise errors rather than being
  ignored**, since silently dropping them would fit a different model than
  the user specified: `lme` correlation structures (e.g. `corAR1()`),
  variance structures, more than one grouping level, list-form `random =`;
  and `coxph` `strata()`, `cluster()` or `tt()` terms, or a non-right-censored
  `Surv()` response.


## Default changes (covariate standardization)

* **`control$standardize_covariates` now defaults to `TRUE`.** Baseline
  covariates in `long_formula` are centred and scaled before any design
  matrix is built, then `beta` is back-transformed so coefficients are
  reported on the ORIGINAL scale. This is a pure reparameterization -
  `X %*% beta` is identical either way, so `alpha`, `sigma_b`, `rho` and
  `sigma_e` are unaffected and the reported inference does not change.
  Only sampling efficiency does.

* **Why.** With an uncentred covariate the posterior correlation between
  the intercept and that covariate's coefficient is approximately
  `-xbar / sqrt(xbar^2 + s^2)` - measured at `-0.9764` for `x ~ N(50,10)`
  against a first-principles prediction of `-0.9802`, and ~0 once centred.
  That near-degenerate ridge is expensive for NUTS to traverse. Across 4
  replicate raw-scale datasets, mean leapfrog steps fell from 228.3 to
  49.6 (a 78% reduction, in every seed) and sampling efficiency rose 2.31x
  on average (range 1.04x-3.11x, 4 of 4 seeds), moving jmjax from below
  `JMbayes2`'s raw-scale range to above it. On the MLE path the optimizer
  also converged faster (37 -> 23 iterations on the test case).

* **This succeeded where three mass-matrix interventions failed.**
  `seed_mass_matrix`, `dense_mass_beta` and `dense_mass_alpha` were all
  PRECONDITIONERS; this is a REPARAMETERIZATION, which is the same fix
  `JMbayes2` gets structurally through hierarchical centering. The
  mechanism is confirmed rather than assumed: leapfrog step counts fell,
  which is exactly the diagnostic `dense_mass_alpha` failed (213.0 either
  way).

* **Honest caveat.** On data whose covariates are ALREADY standardized
  there is no measured benefit and possibly a small cost (efficiency
  ratios 0.97x, 0.72x, 0.49x across 3 seeds - the last inflated by a
  timing artifact, with identical step counts but 53% more wall-clock).
  The standardized-data baseline itself ranged 40-92 ESS/second across
  seeds on identical settings, so this sits inside documented natural
  variability rather than being established harm. It is the default
  anyway because users supplying raw-scale covariates are least likely to
  know they need it, and because the transformation cannot change
  reported inference. Set `standardize_covariates = FALSE` to disable.

* **Correctness verified separately on both paths.** MCMC: back-transformed
  estimates matched an unstandardized fit on all 8 parameters, with
  `beta_2` (the covariate coefficient, the one divided by the scale)
  agreeing to 5 decimal places. MLE: the transformation is linear, so
  `vcov` propagates exactly as `A %*% vcov %*% t(A)` rather than by a
  delta-method approximation; estimates and SEs agreed to 3-4 significant
  figures while `n_iter` differed (37 vs 23), confirming the optimizer
  genuinely took a different path.

## New option (opt-in, narrow benefit)

* `control$dense_mass_alpha = TRUE` adds `alpha` to the spline dense mass
  block, capturing a measured -0.73 to -0.86 posterior correlation between
  `alpha` and the spline coefficients that the default block forces to
  zero. Across 10 replicate raw-scale datasets, `alpha`'s ESS/second
  improved on 8 of 10 (paired t-test on the log ratio, p = 0.007; 95% CI
  [1.23x, 2.66x]), mean rising 13.9 -> 22.3. The effect is floor-raising:
  worst case 4.8 -> 14.0, and the only two seeds that got worse were the
  two where the default already did best.
  **Not the default, for a specific reason**: minimum ESS across all
  parameters was essentially unchanged (279.9 vs 284.0, mixed by seed), so
  this improves the association parameter rather than overall inference
  quality. Mean leapfrog steps were also flat (210.3 vs 211.7), consistent
  with `alpha` decorrelating faster within trajectories whose length is set
  by other directions. Reasonable to enable when `alpha` is the parameter
  of primary interest, which is typical for joint models.

## Tested and not adopted (negative results, recorded deliberately)

* Two options were added, tested, and left at `FALSE` after failing
  pre-registered criteria. They are retained only so the negative results
  are not lost and the same ideas are not re-attempted blind. Full writeup:
  `mass_matrix_investigation_negative_results.md`.
  - `control$seed_mass_matrix`: seeds NUTS's diagonal inverse mass matrix
    for `beta` from the internal `lme()` pre-fit's standard errors. Failed
    catastrophically on 1 of 4 datasets (ESS/sec 64.4 -> 1.2, R-hat
    1.0163 -> 1.2922). A follow-up diagnostic showed NUTS's own adaptation
    already recovers essentially those same scales unaided (adapted
    `[0.00122, 0.00031, 0.00136]` vs `lme` se^2 `[0.00147, 0.00026,
    0.00144]`), so seeding supplied information the sampler finds anyway.
  - `control$dense_mass_beta`: dense mass block over `beta`, targeting a
    posterior correlation of `-0.9764` between the intercept and an
    unstandardized covariate's coefficient - itself predicted to `-0.9802`
    from first principles. Despite the diagnosis being exactly right, the
    intervention produced no consistent improvement (2 of 4 seeds worse in
    each condition).

* **Three diagnostic hypotheses were also refuted by data**, which is why
  this line of work is being closed as a route to the covariate-scale gap:
  - The ~1000-dimensional random-effects block was predicted to dominate
    the conditioning. It is the *best*-conditioned block measured
    (cond 9.4, vs 131.5 for the 9-dimensional spline block).
  - No funnel in the spline block (`corr(log tau_w, log sd(z_step)) =
    0.011`).
  - No funnel in the random effects: `sd(b_std)` sits at 1.0005, exactly
    what clean non-centering predicts, with an estimated slope of about
    `-0.45` against `sigma_b` - the opposite sign from funnel geometry.
  - Notably, no block's condition number explains the observed cost:
    the worst implies ~11.5 leapfrog steps against 254.3 observed.

* **The recommended answer to the covariate-scale gap remains user-side
  standardization of continuous covariates** - a one-line change with a
  replicated 4-8x effect, larger than anything these interventions
  achieved.

## Default changes

* **`control$dense_mass_spline` now defaults to `TRUE`** (was `FALSE`) for
  `spline_prior = "penalized"`. This is the one performance option that
  passed a pre-registered replication study (3 seeds x 2 sample sizes, all
  candidates compared against their own current default, paired by seed):
  ~2.1x faster wall-clock and ~2.3x ESS/second, consistent in direction on
  every individual run, with R-hat and truth recovery unaffected. Set
  `dense_mass_spline = FALSE` to restore the previous behavior.

* **Two other candidates were tested and NOT promoted** - recorded here
  because the negative results are as useful as the positive one:
  - `rw2_implementation = "vectorized"`: an initial single run suggested
    ~1.4x, but replication showed it consistently slightly WORSE at
    `n = 1000` (ESS/second ratio 0.92x, range [0.80, 0.99]) with no
    wall-clock saving, and marginally worse R-hat. The original figure was
    noise. Default stays `"scan"`.
  - `random_effects_method = "wishart_gibbs"`: failed the consistency
    criterion (mixed direction across seeds at `n = 500`, ESS/second range
    [0.97, 1.58]) AND showed worse truth recovery (max |z| rising from
    2.03 to 2.87 at `n = 500`) - consistent with the inverse-Wishart
    small-variance prior bias documented separately. Default stays
    `"nuts"`.

* **A related assumption was tested and disproven:** `dense_mass_spline`
  was believed to require `rw2_implementation = "vectorized"`, on the
  reasoning that `scan` produces many separate scalar `z_step` sites that
  could not form a single dense mass block. In fact numpyro's `scan`
  collects per-iteration samples into one stacked site of the same shape
  the vectorized branch produces directly. Both implementations reach an
  identical 63.0 mean leapfrog steps with dense mass enabled (down from
  255.0 and 232.5 respectively), so the two options are independent. This
  is why the default change above could be made on its own.

## Corrections

* **A previously-stated claim that `random_effects_method = "wishart_gibbs"`
  "matches JMbayes2's approach" was WRONG and has been corrected**
  throughout the documentation. Direct inspection of a fitted `jm`
  object's `$priors` confirmed JMbayes2 uses a **separation strategy**:
  independent Gamma priors on the random-effect standard deviations
  (`D_sds_mean`, centered exactly on the internal `lme()` fit's estimates)
  plus a separate `LKJ(eta = 3)` prior on the correlation. There is no
  Wishart prior on the covariance matrix anywhere in its prior list. The
  original claim came from a Rizopoulos (2016, JSS) quote describing the
  **predecessor** package JMbayes, and concerned *conjugacy* (that
  `D^-1`'s posterior conditional is Wishart - true regardless of prior)
  rather than prior specification.
* **Consequence:** jmjax's DEFAULT (`"nuts"`) path is the one that closely
  matches JMbayes2 - same Gamma family, same shape of 5, same
  `lme()`-derived mean, same separation of scale from correlation,
  differing only in LKJ concentration (jmjax `eta=2` vs JMbayes2
  `eta=3`). The `wishart_gibbs` option is the one that *departs* from
  JMbayes2. See the technical note
  `jmjax_vs_jmbayes2_priors_technical_note` for full derivations.

## New features

* `control$wishart_eb_scale = TRUE` (default `FALSE`): centers the
  Wishart prior's scale matrix on the internal `lme()` pre-fit's variance
  estimates instead of the fixed `S0 = I`, using `nu0 = q + 2` and
  `S0 = solve(diag(sigma_hat^2))` so the implied prior mean of `D` equals
  the pre-fit estimate exactly. Addresses a real risk with the original
  `S0 = I`: the inverse-Wishart family is documented (Alvarez, Niemi &
  Simpson, arXiv:1408.4050) to bias variances upward and correlations
  toward zero when the true variance is small relative to the prior mean,
  **with the bias persisting at large n**. Both the simulation used
  throughout this project (true `sigma_b1 = 0.2`) and PBC2 (`year` random
  slope, sd ~0.17) sit in that regime, with ~25-34x mismatches against an
  `S0 = I` prior centered near 1. NOT yet validated via truth recovery -
  treat as experimental.

## Bug fixes

* `control$n_gh_nodes_per_dim` and `control$n_newton_steps_q2` were
  effectively unusable from R: numeric literals (e.g.
  `n_gh_nodes_per_dim = 7`) are doubles in R, and reticulate forwarded
  them to Python as `7.0`, which `numpy.polynomial.hermite.hermgauss()`
  rejects with "deg must be an integer, received 7.0". Only the
  integer-literal form (`7L`) happened to work. Both values are now
  coerced with `int()` on the Python side (in both `weibull_model.py` and
  `spline_model.py`), so either form works. Found by attempting to
  actually use these options for a tuning experiment - they had been
  documented nowhere and, it turns out, silently broken for any normal
  usage.

## Developer/testing infrastructure

* Added `skip_if_slow_mcmc()` (see `tests/testthat/helper-python.R`),
  applied to all 18 MCMC-based correctness tests (concentrated in
  `test-functional-forms.R` and `test-baseline-covariates.R`, which
  together accounted for ~80% of the full suite's ~21 minute runtime).
  Set `JMJAX_SKIP_SLOW_TESTS=true` for a fast (~a few minutes) iteration
  loop during development; the default (unset, or `"false"`) runs
  everything, exactly as before this was added - no test's actual
  sampling budget or assertions were changed. See `dev/run_tests_both_ways.R`
  for a script that runs both modes and reports the time difference.

## New features

* MCMC methods (`spline-PH-mcmc`/`weibull-PH-mcmc`) can now run genuinely
  PARALLEL chains on CPU, instead of always falling back to sequential
  execution (previously showing "not enough devices... chains will be
  drawn sequentially" on every multi-chain fit, regardless of how many
  CPU cores were actually available). By default, JAX only exposes ONE
  device to numpyro's `MCMC(num_chains=...)` unless explicitly told
  otherwise via `numpyro.set_host_device_count()` - jmjax now calls this
  automatically (defaulting to `parallel::detectCores()`) the first time
  its Python backend is imported in a session, whether that happens via
  an explicit `jmjax_setup()` call or lazily on the first `jm_fit()`
  call. Configurable via `jmjax_setup(num_devices = ...)`, which MUST be
  called before the first `jm_fit()` of a session for the setting to take
  effect (XLA reads this once, when its backend first initializes - it
  cannot be changed mid-session without restarting R).

## Bug fixes

* Fixed `sampling_time_sec` (the internal timer used for ESS/sec
  reporting and benchmarking) systematically understating real MCMC
  compute time for `spline-PH-mcmc`/`weibull-PH-mcmc`, sometimes by an
  order of magnitude or more. Root cause: JAX dispatches computation
  ASYNCHRONOUSLY - `mcmc.run()` can return control to Python as soon as
  the computation is dispatched to the device, without waiting for it to
  actually finish executing in the background. The timer was measured
  immediately after `mcmc.run()` returns, capturing only dispatch time;
  the real wait then showed up later, wherever the code first tried to
  read an actual value - confirmed via direct `cProfile` profiling to be
  `jax.device_get()` blocking inside `numpyro_summary()`'s first array
  access (numpyro's own diagnostic math - R-hat, ESS - took milliseconds;
  effectively all of the "missing" time was JAX finishing the real
  computation, not numpyro doing slow work). Fixed by calling
  `jax.block_until_ready()` immediately after `mcmc.run()`, before
  measuring elapsed time. This was discovered while investigating an
  apparent, hard-to-explain jmjax-vs-JMbayes2 performance gap on
  PBC2-realistic simulated data that never reproduced on the original
  simple benchmark - after extensive elimination of covariates, effect
  sizes, time-scale, and session-state as explanations (each tested
  directly and ruled out), direct profiling of the actual bottleneck
  found this instead. Given `sampling_time_sec` has been used throughout
  this package's development to report ESS/sec figures, any such figures
  from a "harder" model configuration (more parameters, q=2, covariates -
  where the async gap is apparently more pronounced) predating this fix
  should be treated as unreliable; simpler configurations were less
  affected since the async gap was smaller relative to real compute time.
* Also excluded the per-subject random-effects sites ("b"/"b_std") from
  `numpyro_summary()`'s input - their full R-hat/ESS diagnostics were
  being computed (e.g. 400 separate per-subject computations for n=200,
  q=2) and then immediately discarded, since neither is ever included in
  the reported population-level parameters. A legitimate minor efficiency
  improvement, though direct profiling showed this was NOT the actual
  cause of the larger timing bug above (a natural, reasonable first
  hypothesis that turned out to be wrong once tested directly).

## New features (MCMC performance tuning)

* `control$random_effects_method = "wishart_gibbs"` for `q=2` MCMC fits -
  replaces the existing LKJCholesky-parameterized random-effects
  correlation structure (fully explored by NUTS via gradients, jointly
  with every other parameter) with a closed-form Wishart-conjugate Gibbs
  update, hybridized with NUTS via `numpyro.infer.HMCGibbs`. Matches
  `JMbayes2`'s own documented sampler design (Rizopoulos 2016, JSS: "the
  posterior conditional is a Wishart distribution"). Validated faster
  AND better-converging than the previous default on every tested
  metric, with no accuracy cost. `random_effects_corr = FALSE` was also
  added as a simpler, non-default alternative (independent, uncorrelated
  random intercept and slope - cheaper still, at the cost of being
  unable to estimate genuine correlation).
* `control$rw2_implementation = "vectorized"` for
  `spline_prior = "penalized"` - replaces the sequential `scan`-based
  construction of the RW2 spline-smoothing prior with an algebraically
  identical closed-form version (two cumulative sums), verified to match
  the original to floating-point precision. Validated faster with
  improved R-hat.
* `control$dense_mass_spline = TRUE` for `spline_prior = "penalized"` -
  a targeted (not global) dense NUTS mass matrix over just the small
  spline-smoothing parameter block. Fixes a substantial (multiple-fold)
  deep-tree inefficiency: the default diagonal mass matrix cannot
  represent the genuine correlation the RW2 prior induces among
  neighboring spline coefficients, confirmed directly via leapfrog
  step-count measurement. A *global* `control$dense_mass = TRUE` was
  also added but is NOT recommended - tested and found dramatically
  counterproductive (roughly 12x slower) due to the cost of estimating a
  mass matrix over the much larger random-effects block.
* Combined, the above three options were validated on real PBC2 data to
  restore and substantially exceed jmjax's original simulated-data
  efficiency advantage over `JMbayes2` (previously eroded to near-parity
  on real, covariate-laden data) - approximately 4.8x more
  sample-efficient in the final validated comparison.
* `rho` (the random-effects correlation) is now extracted and reported
  in `fit$estimates` for MCMC `q=2` fits, for both the existing
  LKJCholesky-based path and the new `"wishart_gibbs"` path. This closes
  a pre-existing gap: `rho` was previously never surfaced despite being
  sampled (via `L_corr`) throughout MCMC's `q=2` history.
* **Experimental:** `control$random_effects_method = "wishart_gibbs_centered"`
  additionally constrains the random intercepts to sum to exactly zero
  at every posterior draw, addressing a diagnosed near-perfect (-0.95)
  posterior correlation between the population intercept `beta_0` and
  the mean random intercept (a classic location degeneracy). Implemented
  as a fully separate, opt-in code path with zero effect on
  `"wishart_gibbs"`. Resolved the diagnosed mixing problem directly
  (`beta_0` ESS improved ~4.8x in testing) but was found, on further
  validation at larger sample sizes, to introduce a real, quantitatively
  -characterized credible-interval miscalibration for `beta_0` (point
  estimates remained unbiased; reported posterior SD did not fully
  reflect true uncertainty). Recommended for exploratory use only until
  resolved (candidate fixes - a soft rather than hard sum-to-zero
  penalty, or a formally-derived variance correction - are documented in
  the full investigation report but not yet implemented).


## Performance investigation findings (q=2 MLE) - documented, defaults unchanged

* **`n_newton_steps_q2 = 12` is a validated speedup, but is NOT the new
  default.** Measured ~1.35-1.43x faster than the default 20, with no
  detectable accuracy cost (identical log-likelihood to ~1e-2 on values in
  the thousands; unchanged parameter recovery) at n=500 and n=2000. Left
  non-default deliberately: validated on only ONE simulation
  configuration, and 8 steps was observed to FAIL to converge - so 12 sits
  between a known-good and a known-bad value on limited evidence. Newton
  convergence depends on likelihood curvature, which varies with
  random-effects variances, event rate, and visit schedule. The option is
  exposed, so users who want the speedup can set it and check
  `fit$convergence$converged` on their own data. Real-world usage across
  varied datasets will be a better guide to a safe default than further
  variations on one simulator; revisit if user reports support it.

* **Quadrature grid size is essentially free.** `n_gh_nodes_per_dim`
  (default 7 -> 49 points) was found to have almost no runtime effect -
  25 vs 81 points made no consistent difference, since `vmap` batches all
  nodes into one operation - and the log-likelihood was already fully
  converged at 25 points. An earlier hypothesis that the tensor-product
  grid was a major q=2 bottleneck was therefore WRONG, and is recorded
  here so it isn't re-investigated.

* **`cholesky_first_newton` - tested and rejected.** An attempt to skip
  the per-Newton-step eigendecomposition via a cheaper Cholesky
  factorization. Under JAX's `jit`, `try`/`except` cannot guard a traced
  operation, so the implementation must use `jnp.where`, which computes
  BOTH branches - adding work rather than replacing it. Measured 33%
  SLOWER at 20 Newton steps, 10% slower at 12. The option is retained
  (default `FALSE`) purely to document the negative result. Revisiting
  this idea requires a genuinely different approach (`lax.cond`, or
  dropping the safeguard for provably well-conditioned problems), not a
  retry.

* **Methodological note for future benchmarking:** several timing
  measurements during this investigation were initially confounded by
  JAX's one-time JIT compilation cost being paid by whichever
  configuration ran first (producing an apparent but false ~2.5x
  speedup, and the impossible result of two combined optimizations being
  slower than either alone). Always use a discarded warmup run per
  configuration, plus at least two measured runs so within-config
  run-to-run variance is visible - an effect smaller than a config's own
  spread should not be treated as real.

## Known limitations (documented, not yet fixed)

* jmjax's efficiency advantage over `JMbayes2` was found to depend on
  continuous covariates being reasonably well-scaled, and can reverse
  entirely for raw, unstandardized covariates (e.g. `age` on its natural
  scale). Confirmed via a controlled, replicated test (paired t-test,
  p < 0.02 on both wall-clock time and ESS/second): a 2.43x jmjax
  advantage with a standardized covariate became a 0.76x disadvantage
  with an unstandardized one, while `JMbayes2`'s own efficiency was
  essentially scale-invariant. Root cause: `JMbayes2` seeds its MCMC
  proposal distributions from the covariance matrix of its internal
  MLE pre-fits, which already reflects each covariate's natural scale;
  NUTS has no equivalent source of scale information and must learn it
  from scratch during warmup. Standardizing continuous covariates before
  fitting is recommended in the interim (see both vignettes' "Performance
  tuning" sections); a planned future enhancement (accepting user-
  supplied `lme`/`coxph` pre-fit objects, already useful for avoiding a
  redundant internal re-fit) should also use those objects' covariance
  matrices to seed NUTS's mass matrix, closing this gap architecturally
  rather than requiring users to standardize manually.

## New features

* Baseline covariates (both the survival-submodel "Piece 1" and
  longitudinal-submodel "Piece 2") extended to `weibull-PH-mcmc`/
  `spline-PH-mcmc`. Found via a real user script combining age (Piece 2)
  and drug (Piece 1) with `random_effects = "intercept_slope"` - which
  correctly errored, since baseline covariates had only ever been built
  for `weibull-PH-aGH`. Longitudinal covariates needed NO Python changes
  (the now-familiar "beta's length already generalizes" story). Survival
  covariates needed a new `gamma` parameter in `build_model()` -
  `gamma_term = W_surv @ gamma`, a per-subject constant added identically
  to both hazard branches (channel-present and channel-absent), following
  the exact `None`-check pattern used throughout this file. Unlike the
  MLE backends (where baseline covariates only support plain "value" -
  no combined "channel + covariates" MLE function was ever written), MCMC
  supports covariates combined with ANY functional_forms channel for
  free, since `build_model()`'s channel logic and `gamma_term` were
  already independent additions to the same shared formula.

* `delta`/`area`/`area_avg` extended to `method = "spline-PH-mcmc"`.
  Turned out to require ZERO Python changes: the channel logic added for
  `weibull-PH-mcmc` sits entirely after `build_model()`'s baseline-hazard
  branch and only ever touches `log_h0_T`/`log_h0_quad` (already computed
  correctly for either baseline choice above it) - so it was already
  baseline-hazard-agnostic without anyone having to plan for that
  explicitly. `area`/`area_avg` are supported at both q=1 and q=2; `delta`
  is q=2-only, carrying forward the CONFIRMED non-identifiability finding
  from `spline-PH-aGH` (a flexible spline baseline can fully absorb
  delta's purely-deterministic q=1 signal) - reasoned to apply equally to
  MCMC estimation since it's a property of the model class, not the
  algorithm, and confirmed by extending the same guard rather than
  re-deriving it. The claim that area/area_avg DON'T share this problem
  even at q=1 (their b-contributions are genuine subject-level variation
  a population curve can't mimic) was stated as a hypothesis in the guard
  and then directly validated by their q=1 truth-recovery tests actually
  passing.

* `delta`, `area`, and `area_avg` on `weibull-PH-mcmc` extended from q=1
  to q=2, using the exact same generalization already proven on the MLE
  side: the q=1-specific shortcut formulas (no b term / `b_i*T_i` / plain
  `b_i`) are each a special case of the general rule `Z_channel(t) @ b`.
  `build_model()` now dispatches on whether `Z_extra` was actually passed
  by the R side (only computed for q>1) - `None` falls back to the
  already-validated q=1 shortcuts unchanged, present uses the general
  rule directly.

## Known limitations (documented, not yet fixed)

* `random_effects = "intercept_slope"`/`random_formula` combined with a
  nonlinear `long_formula` can fail to converge (MLE: immediate L-BFGS-B
  failure; MCMC: very high R-hat) when `time_var` has a large raw
  numerical range (e.g. `year` spanning 0-14). Discovered via jmjax's
  first real-data test (PBC2). Confirmed via a real cross-package check:
  fitting the identical model/data in `JMbayes2` converged cleanly,
  ruling out a genuine model-identifiability explanation and pointing
  instead at raw coefficient/gradient scale conditioning, shared by both
  jmjax's MLE and MCMC backends. Resolved by rescaling `time_var` to a
  moderate range (e.g. `year/10`) before fitting - see
  `vignette("jmjax-validation")`'s "Time-scale sensitivity" section for
  the full diagnostic story, including a hypothesis that was tested and
  explicitly retracted when it didn't hold up. Not yet automated
  (deliberately, to avoid silently changing fitted coefficients' units
  without the user asking) - a future version could add a non-silent
  warning instead.

## New features

* Both baseline-covariate pieces extended from q=1 to q=2
  (`random_effects = "intercept_slope"`). Survival-side covariates
  (`gamma'W_i`) needed a new function built on the already-validated q=2
  scaffolding (the term itself is q-independent, same formula as q=1).
  Longitudinal-side covariates needed NO new Python code at all, same
  story as q=1 - but a real footgun was identified and guarded against:
  the legacy default `Z = X[:, :2]` slicing only correctly excludes a
  baseline covariate from Z if it happens to appear AFTER `time_var` in
  `long_formula`'s column order (e.g. `y ~ time + age` works by
  convention, `y ~ age + time` would silently be wrong) - rather than
  rely on that fragile convention, `random_formula` must now be given
  explicitly whenever longitudinal baseline covariates are combined with
  q=2, removing the ambiguity entirely (a clear error otherwise, not
  silent wrong behavior). Both pieces also validated working together
  simultaneously (covariates on both sides of the model at once).

* `jm_fit()` now supports baseline (time-constant) covariates in the
  LONGITUDINAL submodel too, e.g. `long_formula = y ~ time + age`
  (complementing the survival-submodel covariate support above - "Piece
  1" and "Piece 2" of the same underlying gap). The fix is isolated to
  `build_time_design()`: evaluating X(t) at a SYNTHETIC time point (the
  survival time, quadrature nodes) previously built a newdata frame
  containing only `time_var`, silently excluding any other covariate a
  user might have added to `long_formula` - `X_long` itself (built from
  the real `data_long`) was already correct via `model.matrix()`'s
  existing generality, so only the synthetic-time-point evaluation path
  had a gap. New `extract_baseline_covariates_long()` extracts each
  subject's covariate value once (validating it's genuinely
  time-CONSTANT, not a real time-varying covariate - a substantially
  harder, separate modeling problem deliberately kept out of scope) and
  threads it through. For the specific configuration implemented today
  (`weibull-PH-aGH`, q=1, plain value association), this required NO
  Python changes at all - `beta`'s length already generalizes to any `p`
  automatically. Other configurations raise a clear error rather than
  silently ignoring the covariates.

* `jm_fit()` now supports baseline (time-constant) covariates in the
  survival submodel, e.g. `surv_formula = Surv(time, event) ~ age + sex`.
  IMPORTANT: prior to this addition, `build_surv_arrays()` only ever read
  `surv_formula`'s LHS (the `Surv()` response) - any RHS covariates were
  SILENTLY DROPPED, not rejected. This is now built properly rather than
  left as a latent trap. The intercept column is explicitly excluded from
  the covariate design (a global multiplicative constant would be
  perfectly collinear with the baseline hazard's own intercept - the same
  reason `coxph()` itself never includes one). Mathematically the
  simplest possible model extension (a baseline covariate contributes a
  per-subject CONSTANT to the log-hazard, needing no new quadrature
  machinery at all, unlike `delta`/`area`) - implemented first for the
  single simplest configuration (`weibull-PH-aGH`, q=1, plain "value"
  association), matching the same incremental discipline used for
  `delta`/`area`/`random_formula`. Other configurations (q=2, spline
  baseline, MCMC methods, combined with `functional_forms`) raise a clear
  error rather than silently ignoring the covariates or producing a wrong
  fit - extending to those is planned incrementally.

* `jm_fit()` gained a `random_formula` argument, fixing a design
  assumption that only happened to be correct for `long_formula = y ~
  time`: previously, the random-effects design `Z` was ALWAYS derived by
  slicing the fixed-effects design `X`'s first `q` columns
  (`Z = X[, 1:q]`), which silently breaks for any `long_formula` where
  the first columns aren't simply "intercept" and "slope" (e.g.
  `y ~ time + I(time^2)`, or a spline-basis longitudinal formula). When
  `random_formula` is supplied (e.g. `random_formula = ~ time`), `Z` is
  now evaluated completely independently, the same way `X` is evaluated
  from `long_formula`. Supported across ALL FOUR fitting methods
  (`weibull-PH-aGH`, `spline-PH-aGH`, and - added shortly after the
  initial MLE-only version - `spline-PH-mcmc`/`weibull-PH-mcmc` too, both
  q=1 and q=2 throughout). The MCMC extension turned out to be a purely
  mechanical R-side wiring fix: `fit_nuts()` already accepted
  `Z_long`/`Z_time_surv`/`Z_time_quad` as optional overrides (falling back
  to the old internal slicing only when not given), so no Python changes
  were needed there at all. `random_formula = NULL` (the default)
  preserves the exact original behavior for full backward compatibility -
  confirmed via a dedicated test showing numerically identical results.
  Cross-validated with a genuine quadratic-in-time `long_formula` and an
  independent `random_formula = ~ time` on BOTH the MLE and MCMC paths:
  the originally-broken scenario now correctly recovers all true
  parameters (all 3 polynomial coefficients on the fixed-effects side,
  plus `sigma_b0`/`sigma_b1`/`rho`/`alpha` on the random-effects/
  association side) either way. Also fixed a related bug found while
  extending this: the internal empirical-Bayes `nlme::lme()` pre-fit used
  by the MCMC methods hardcoded `~time|id` as its random-effects formula
  whenever `q>1`, which only happened to be correct when the random slope
  was on `time_var` specifically - now uses `random_formula` directly
  when supplied. `q>2` is explicitly rejected everywhere (the underlying
  adaptive-GH/NUTS machinery is hardcoded for exactly 1 or 2 random
  effects, not arbitrary `q`).
* New `build_random_long_array()`, mirroring `build_long_arrays()`'s
  padding logic but driven by an independent formula - the low-level
  function underlying `random_formula`.

* `jm_fit()` gained a `functional_forms` argument, matching `JMbayes2`'s
  syntax (e.g. `~ value(y) + delta(y)`, `~ value(y) + area(y)`), for
  associating the survival hazard with functions of the longitudinal
  trajectory beyond its current value. Currently supported for
  `method = "weibull-PH-aGH"` with `random_effects = "intercept"`:
  - `delta(t) = m_i(t) - m_i(0)`: for a pure random intercept, this has NO
    random-effect contribution (it cancels exactly in the subtraction) -
    the simplest functional form to add.
  - `area(t) = integral_0^t m_i(s) ds`: DOES retain a random-effect
    contribution (`b_i * t`, since a constant integrates to something
    proportional to the integration length) and requires genuine nested
    Gauss-Kronrod quadrature on the R side (`build_area_channel()`) to
    evaluate the fixed-effects design at each outer quadrature node.
  - Combining `delta` and `area` simultaneously is not yet supported
    (each implemented as its own independent code path for now, validated
    separately - see package development notes for the plan to
    generalize into a combinable multi-channel system later).
  - `slope(t) = d/dt[m_i(t)]` remains planned (would need differentiating
    the design matrix w.r.t. time - trivial for the current linear-in-time
    scope, but real work once non-linear time bases like splines are
    supported for `long_formula`).
  - `area_avg(t) = area(t) / t`: added specifically because `JMbayes2`'s
    own `area()` function computes this TIME-AVERAGED integral, not the
    raw one - confirmed by directly inspecting a fitted `JMbayes2`
    object's `model_data$X_h` (the documentation for `standardise`/
    `time_window` was internally inconsistent, so this was verified
    empirically rather than trusted). `jmjax`'s own `area()` keeps its
    original raw-integral meaning; use `area_avg()` when replicating a
    `JMbayes2` model.
  - `delta`, `area`, and `area_avg` now all support `random_effects =
    "intercept_slope"` (q=2), not just the original q=1 (random intercept
    only). Derived from a general rule worked out while extending: each
    channel's random-effect contribution is `Z_channel(t) @ b`, where
    `Z_channel(t)` is that channel's own fixed-effects design restricted
    to its first `q` columns - the q=1 formulas (delta: none; area:
    `b*t`; area_avg: plain `b`) turned out to all be special cases of this
    one rule, which let the three q=2 extensions share a single generic
    implementation rather than needing three separate ones.
  - `delta` ported to `method = "spline-PH-aGH"` (previously
    `weibull-PH-aGH`-only) - but see the identifiability note below;
    `area`/`area_avg` not yet ported to the spline baseline.
* `method = "spline-PH-aGH"` now supports `random_effects =
  "intercept_slope"` (q=2) for the first time - previously restricted to
  `q=1` ("would need combining tensor-product adaptive GH with the spline
  basis - a separate, bigger undertaking not yet implemented"; this is
  that undertaking). Built by combining the already-validated q=2
  machinery (2D Newton mode-finding, tensor-product Gauss-Hermite
  quadrature, bivariate-normal random-effects prior) with the spline
  baseline hazard - cross-validated against `JM::jointModel`'s own
  spline-PH-aGH with a real random slope.
* `delta` further extended to `spline-PH-aGH` + q=2, using the same
  generic extra-channel machinery built for Weibull. This CONFIRMS the
  q=1 identifiability problem documented below is resolved by a genuine
  random slope: unlike q=1 (NaN standard errors, wrong-signed
  `alpha_delta`), q=2 gives finite standard errors throughout and
  correctly recovers both `alpha_value` and `alpha_delta` from simulated
  data - direct empirical confirmation of the mechanism proposed when the
  q=1 guard was added (a genuine random slope gives `delta(t)` real
  subject-level variation that a population-level spline curve cannot
  absorb).

## Confirmed non-identifiable combination (guarded, not fixed)

`functional_forms = ~ ... + delta(...)` combined with `method =
"spline-PH-aGH"` and `random_effects = "intercept"` (q=1) is now actively
rejected by `jm_fit()` with an informative error, rather than silently
producing a garbage fit. This is a genuine identifiability problem, not a
code bug: for a pure random intercept, `delta(t) = beta1*t` is a purely
deterministic function of `t` with zero subject-level variation (the
intercept and its random effect both cancel in the `t=0` subtraction). A
flexible multi-parameter spline baseline can fully absorb this signal
into its own coefficients, leaving `alpha_delta` unidentified. Confirmed
empirically before adding the guard: `beta`/`sigma_e`/`sigma_b`/
`alpha_value` recovered essentially perfectly, but `alpha_delta` came out
with the WRONG SIGN and a singular (`NaN`) standard error, with 8 of 9
spline coefficients also showing `NaN` SEs - the signature of a genuine
likelihood ridge. `weibull-PH-aGH + delta` remains fully valid at q=1
(a rigid 2-parameter Weibull baseline cannot freely mimic an arbitrary
added linear trend the way a flexible spline can). Expected to be
resolved once `delta` is extended to `random_effects = "intercept_slope"`
(q=2) for the spline baseline, where a genuine random slope gives
`delta(t)` real subject-level variation that a population-level spline
curve cannot absorb - not yet implemented.

# jmjax 0.2.0

## New features

* Added `method = "weibull-PH-mcmc"`: full Bayesian NUTS fit with the same
  closed-form Weibull baseline hazard as `weibull-PH-aGH`, useful for
  isolating whether a discrepancy is about spline/quadrature machinery
  specifically.
* `method = "weibull-PH-aGH"` now supports `random_effects =
  "intercept_slope"` via a new 2D adaptive Gauss-Hermite implementation
  (tensor-product quadrature with per-subject 2D Newton mode-finding).
  Cross-validated against `JM::jointModel` to 3-4 decimal places on every
  parameter.
* `spline-PH-mcmc` gained `control$spline_prior = "penalized"`: a P-spline
  (RW2 + Gamma precision hyperprior) alternative to the original
  independent-priors default, with hyperparameters (shape=5, rate=0.5)
  confirmed to match `JMbayes2`'s actual defaults.
* `spline-PH-mcmc` and `weibull-PH-mcmc` gained
  `control$empirical_bayes_prior` (default `TRUE`): centers the priors for
  `beta` and the random-effect standard deviations on an internally-fit
  `nlme::lme()` model's MLEs, matching `JMbayes2`'s confirmed default
  behavior, without requiring the user to pre-fit or pass an `lme` object
  themselves.
* `build_spline_knots()` gained `placement = "equal"`, matching
  `JMbayes2`'s exact knot-spacing scheme (confirmed by reading
  `JMbayes2:::knots`'s source), as an alternative to the original
  `"quantile"` placement (matching `JM`'s scheme).
* `gauss_kronrod_nodes()` now supports arbitrary node counts via a
  Gauss-Legendre fallback (Golub-Welsch algorithm), not just the original
  hardcoded 10-node table.
* New diagnostic-only functions `evaluate_log_density()` and
  `find_map_b_std()` in the Python backend, supporting a rigorous
  profile-log-density check for distinguishing genuine finite-sample
  estimation difficulty from real implementation bugs (see
  `vignette("jmjax-validation")`).

## Bug fixes

* Fixed a no-op mask in the MCMC longitudinal likelihood that scored
  padded (non-existent) observations as real, perfectly-fit data points,
  systematically deflating `sigma_e` by roughly 50% and cascading into
  biased random-effect variance and association-parameter estimates.
* Fixed a silent array-scrambling bug in the R-side construction of the
  spline basis at quadrature nodes (`apply()` + `array()` reshape used
  mismatched flattening order).
* Fixed the spline knot boundary formula to match `JM`'s actual
  `range(Time, st)` (observed times combined with Gauss-Kronrod
  quadrature evaluation points), not `range(Time)` alone.
* Fixed several R-to-Python type coercion issues (R doubles/length-1
  vectors arriving as Python floats or 0-dimensional arrays where JAX
  required a true integer or proper array).
* Reparameterized the P-spline (RW2) prior from a centered to a
  non-centered parameterization, fixing a funnel-geometry sampling
  inefficiency. Improved mean ESS/second by roughly 5x in benchmark
  testing (from 37.2 to 178.8).

## Validation

* Extensively cross-validated against R's `JM` and `JMbayes2` packages;
  see `vignette("jmjax-validation")` for full methodology and results.
* Added a replicated benchmark against `JMbayes2` (20 matched datasets):
  statistically indistinguishable accuracy (paired t-test p=0.079), 13.5x
  sampling efficiency (paired t-test p=1.8e-07, 95% CI on the ESS/sec
  ratio well clear of 1).

# jmjax 0.1.0

* Initial version: `method` in `"weibull-PH-aGH"`, `"spline-PH-aGH"`,
  `"spline-PH-mcmc"` (random intercept only for the two MLE methods;
  `spline-PH-mcmc` additionally supports `random_effects =
  "intercept_slope"`).
* R-side formula parsing and design-matrix construction; JAX/NumPyro
  Python backend via `reticulate`.
* Basic test suite: pure-R unit tests for knot placement and design
  matrices, truth-recovery tests, and cross-validation against `JM`.
