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
