# jmjax 0.4.0

## The absorbable directions are defined jointly across random-effect columns

* The rotation (`control$rotate_absorbable`, on by default) is now built
  on the absorbable subspace defined jointly across random-effect columns,
  as in the revised Section 3 of the methods paper. Up to 0.3.x it was the
  union of per-column subspaces, which misses a fixed-effect shift that has
  to be offset in several random-effect columns at once: with
  `X = [1, t, x(1 + t)]` and `Z = [1, t]`, each column alone gives
  `span{1}`, but `span{1, x}` is absorbable. The new construction
  (`_absorbable_basis_joint()` in `absorbable.py`) takes `ker M` of the
  stacked `(I - P_Z) X` by one SVD, solves each subject's shift from the
  trajectory identity on its observation times, event time, quadrature
  nodes and any `delta`/`area`/`area_avg` rows, verifies every generator
  before use, and builds `Q` from a second SVD. It depends on the
  fixed-effects design only through its column space.
* Subjects whose `Z_i` is rank-deficient on their own observation rows
  (typically baseline-only subjects) now get the shift the survival
  submodel forces. The 0.3.x per-column construction filled them with the
  median of the other subjects' ratios, which is right for uniform
  directions but admitted a non-absorbable direction for a covariate-by-time
  column (`y ~ time * x`) with baseline-only subjects.
* Condition (S) is still checked at every fit (worst relative residual,
  warning above 1e-6). A direction that fails it is now left out of the
  rotation instead of only warned about.
* `fit$convergence$orthogonalize$rotation` gains `construction`,
  `dim_ker_M`, `n_witnesses`, `n_rejected_condition_s`,
  `n_rejected_verify`, and, for comparison with 0.3.x, `k_columnwise` and
  `columnwise_subspace_distance`.
* Every fit is exact either way (any fixed rotation is), so posteriors do
  not change. For designs in which every fixed-effect column is a
  subject-constant covariate times one random-effect basis function - all
  the designs in the vignettes and the methods paper - the two
  constructions give the same subspace. The orthonormal basis chosen
  inside it can differ, so draws from a given seed are not bit-for-bit
  those of 0.3.x, but the explicit block spans the same directions and
  mixing is equivalent. Otherwise the block can hold different directions
  and mixing can differ. To reproduce 0.3.x draws exactly, use 0.3.x.

## Bug fixes

* A user-supplied `control$beta_prior_mean`, `beta_prior_sd`,
  `sigma_e_prior_mean` or `sigma_b_prior_mean` now takes precedence over
  the empirical-Bayes pre-fit, as `?jm_fit` documented. Before, the
  pre-fit silently overwrote them whenever it ran (the default).

## Documentation

* `?jm_fit` states that, with `standardize_covariates = TRUE` (the
  default), the empirical-Bayes `beta` prior is placed on the standardized
  (and, for MCMC, time-rescaled) scale while coefficients are reported on
  the original scale.
* `q`: the docs said "`q >= 2`" where they meant `q = 2`. `jmjax` fits at
  most two random effects per subject; this is now stated under
  `random_formula`.
* The vignettes agree with each other, with the code's defaults and with
  the methods paper: `spline_prior = "independent"` and
  `rw2_implementation = "vectorized"` are the defaults; a 13.9 to 27.9
  ESS/sec improvement is about 2x, not 5x; the comparisons with `JMbayes2`
  use one ESS estimator for both packages (`dev/study_common_ess.R`); the
  Monte Carlo SE of 95% coverage from 100 replicates is 0.022; and the
  validation vignette's list of next steps records what has since been
  done. `vignette("jmjax-reparameterization")`, Section 4, describes the
  joint construction.

## Tests

* `tests/testthat/python/check_joint_construction.py` (numpy only) checks
  the construction against the methods paper's reference implementation:
  the coupled example, a baseline-only subject, a rank-deficient `Z_i`, a
  Condition (S) failure, `q = 3`, column-space invariance and 80 randomized
  designs. `dev/verify_joint_vs_columnwise.R` compares the two
  constructions on fits of the paper's designs.

# jmjax 0.3.1

## Bug fixes

* `jm_mle(baseline = "spline", init_theta = fit$estimates)` with the
  estimates of a Weibull fit failed with a shape error from the backend,
  because a Weibull fit has no spline coefficients. They now start at zero.
  Only this use of `init_theta` is affected; no fit that ran before gives a
  different result.

## Documentation

* `citation("jmjax")`, `CITATION.cff` and the README cite the software
  (DOI) and the preprint describing the random-effects rotation.
* The introduction vignette uses the `pbc2` data throughout, with output
  from a real run, and shows `init_theta = fit_w$estimates` for a warm
  start.
* The maintainer address is now jmjaxmaintainer@gmail.com.

## Packaging

* The methods-paper manuscript has moved out of the repository. The
  development studies behind it stay in `dev/`.

# jmjax 0.3.0

The full development record behind this release - measurements, failed
attempts and the reasoning for each default - is in
`dev/notes/development-log-0.3.0.md` in the source repository.

## Fitting functions

* **Two fitting functions, one per estimation approach:** `jm_mle()`
  (maximum likelihood) and `jm_bayes()` (Bayesian, NUTS), each with
  `baseline = "weibull"` or `"spline"`. Each documents only its own
  options and warns about, then ignores, `control` options that belong to
  the other. They call `jm_fit()`, so results are identical to the
  corresponding `jm_fit()` method:

  | `jm_fit(method = ...)` | New call |
  |---|---|
  | `"weibull-PH-aGH"` | `jm_mle()` or `jm_mle(baseline = "weibull")` |
  | `"spline-PH-aGH"` | `jm_mle(baseline = "spline")` |
  | `"spline-PH-mcmc"` | `jm_bayes()` or `jm_bayes(baseline = "spline")` |
  | `"weibull-PH-mcmc"` | `jm_bayes(baseline = "weibull")` |

  In `jm_bayes()`, `chains`, `warmup`, `samples` and `seed` are arguments;
  they set `control$num_chains`, `num_warmup`, `num_samples` and `seed`,
  which still work. `jm_bayes()` defaults to 2 chains (`jm_fit()` to 1).
* **`jm_fit()` is unchanged** and stays exported as the general interface
  that documents every option.
* **`jm_fit_prefit()` is no longer exported.** It gave a second way of
  specifying the same model, which users found confusing. It still exists
  as `jmjax:::jm_fit_prefit()`, with the same arguments and results.

## New features

* **`predict()`** for `jm_bayes()` fits: a subject's marker trajectory
  (`process = "longitudinal"`) and conditional survival probabilities
  P(T > u | T > t) (`process = "event"`), with posterior intervals. This
  first version covers subjects in the fitted data and the default `value`
  association; on `pbc2` it agrees with `JMbayes2::predict()` to within
  0.016 in survival probability. Dynamic prediction for new subjects is
  planned.
* **Standard model methods:** `coef()`, `vcov()`, `confint()`, `logLik()`
  and `nobs()`, so `AIC()` and `BIC()` also work for maximum-likelihood
  fits. For MCMC fits, `vcov()` and `confint()` come from the posterior
  draws.
* **`summary()` reports posterior credible intervals, R-hat and ESS** for
  MCMC fits (`CrI.lower`, `CrI.upper`, `Rhat`, `ESS`). The existing columns
  are kept, so code that reads `summary(fit)$p.value` still works.
* **`print()` is reorganized** around the association parameter (with its
  hazard ratio), with grouped tables, a compact line for the spline
  baseline hazard, and one-line convergence and sampler summaries.
* **Association structures:** `functional_forms = ~ value(y) + delta(y)`,
  `+ area(y)` or `+ area_avg(y)`, with the same syntax as JMbayes2.
* **Baseline covariates** in both submodels (e.g.
  `Surv(time, event) ~ drug`, `y ~ time + age`); `?jm_fit` lists which
  combinations each method supports. Survival-side covariates were
  previously dropped without a warning.
* **Parallel MCMC chains on CPU** (`jmjax_setup(num_devices = ...)`).
* **`jmjax_available()`** reports whether the Python backend can be used,
  without installing anything.
* Opt-in options: `control$em_warm_start` (EM start for the MLE paths),
  `control$dense_mass_alpha`, `control$wishart_eb_scale`,
  `control$keep_data`.

## Changed defaults

These change results or output. Each can be switched back through
`control`.

* **Double precision.** The backend now runs in float64. It previously ran
  in float32 without saying so, which mainly affected the MLE standard
  errors.
* **The random effects are rotated for NUTS** (`rotate_absorbable = TRUE`)
  wherever the rotation applies: random-intercept-only fits, and
  correlated random intercept and slope. The rotation leaves the model and
  the estimates unchanged and removes a degeneracy between the random
  effects and the subject-constant fixed effects; on `aids` and `pbc2` the
  affected coefficients mixed 19-53 times faster. In a simulation grid,
  unrotated random-intercept fits failed R-hat 1.05 in 17 of 21 runs.
* **Time is rescaled internally for MCMC** (`scale_time = "auto"`), and
  **longitudinal covariates are centred and scaled** (`standardize_covariates
  = TRUE`). Both are reparameterizations: every reported number is on the
  original scale.
* **MCMC chains start from the `lme()` pre-fit** (`mcmc_warm_start = TRUE`),
  as JMbayes2 does.
* **Penalized spline baseline:** `dense_mass_spline = TRUE` and
  `rw2_implementation = "vectorized"` (`"scan"` cannot run with the pinned
  jax/numpyro versions).
* **MLE optimizer:** `opt_method = "BFGS"` (was `"L-BFGS-B"`, which could
  stop early and report convergence) and `parscale = 0.01`, as in `JM`.
  `fit$convergence$converged` is now based on the gradient rather than on
  the optimizer's exit code.
* **Quiet by default.** Routine notes (warm-start seed, covariate
  standardization) appear only with `control$verbose = TRUE`, and the MCMC
  progress bar is off (`progress_bar = TRUE` turns it on; it costs a few
  percent at most). `options(jmjax.quiet = TRUE)` silences all
  informational messages.
* `random_formula = NULL` now means `~ 1` or `~ time` explicitly, rather
  than taking the first columns of the fixed-effects design.

## Bug fixes

* **`fit$vcov` is now the covariance of the reported parameters**, named,
  so that `sqrt(diag(vcov(fit)))` equals the standard errors. It was the
  inverse Hessian on the optimizer's internal scale (`log(sigma)`,
  `atanh(rho)`), so its diagonal disagreed with `fit$se` for every
  variance parameter.
* **Random-slope MCMC fits under the default `scale_time`:** `ranef()`,
  `posterior_samples$b` and the `sigma_b1` posterior draws are now in the
  original time units. They were off by the internal scaling factor, while
  the point estimates were correct.
* **`standardize_covariates` returned wrong coefficients** for formulas with
  an interaction involving a standardized covariate.
* With standardized covariates, the R-hat and ESS reported for `beta` now
  refer to the original-scale coefficients.
* `fit$mcmc_settings` records the run lengths actually used (it was `NULL`
  under the defaults, so `print()` omitted the sampler line).
* `control$n_gh_nodes_per_dim` and `control$n_newton_steps_q2` accept plain
  numbers as well as integers.
* The MCMC timing used for ESS per second no longer understates run time.

## Packaging

* jmjax no longer creates a Python environment during a non-interactive
  package check; `jmjax_setup()` does that once, on request.
* Examples run where the backend is available, and skip cleanly where it
  is not.
* `control$orthogonalize_b0`/`orthogonalize_b`, superseded by the rotation,
  remain available as legacy options.

## Known limitations

* `predict()` does not yet support maximum-likelihood fits, the
  `delta`/`area` associations or new subjects.
* `functional_forms` with `delta()` is rejected for `spline-PH-aGH` with a
  random intercept only, where it is not identifiable.

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
