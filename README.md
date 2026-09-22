# jmjax — architecture sketch

## Data flow

```
User (R)
  |
  |  jm_fit(y ~ time, Surv(time,event) ~ 1, data_long, data_surv,
  |         id_var="id", time_var="time", method="spline-PH-aGH")
  v
R/jm_fit.R
  |-- build_long_arrays()      R/build_design.R   -> X_long, y_long, n_obs (padded arrays)
  |-- build_surv_arrays()      R/build_design.R   -> T_surv, event, t_quad, gk_weights
  |-- build_time_design()      R/build_design.R   -> X_time_surv, X_time_quad (shared m_i(t) term)
  |-- build_spline_knots()     R/knots.R          -> B_T, B_quad (if spline method; JM-compatible knots)
  |
  |  reticulate call: backend$spline_model$fit_mle(!!!backend_args)
  v
inst/python/jmjax_backend/
  |-- spline_model.py / weibull_model.py / mcmc_model.py
  |     (build theta layout, call into common.py's adaptive-GH + Newton
  |      machinery or numpyro's NUTS, run scipy L-BFGS-B or MCMC)
  |-- common.py
  |     (shared: theta layout, per-subject Newton mode-finding, adaptive
  |      GH quadrature, L-BFGS-B driver, Hessian-based SEs)
  |
  |  returns: {estimates, se, vcov, loglik, convergence, posterior_samples}
  v
R/jm_fit.R  -> wraps into class "jmjax"
  |
  v
R/summary.jmjax.R  -> summary(), print()  (formatted like JM::summary.jointModel)
```

## Why this split

- **R owns everything formula/data related**: parsing `y ~ time`,
  `Surv(time, event) ~ 1`, building design matrices, computing spline
  knots (ported from `JM`'s own logic so results are directly comparable),
  building the Gauss-Kronrod quadrature grid. R's formula/model.matrix
  machinery is mature; reimplementing it in Python would be wasted effort.
- **Python owns all the numerics**: JAX autodiff, adaptive Gauss-Hermite
  quadrature via per-subject Newton mode-finding (for the two MLE methods),
  and NumPyro NUTS (for the MCMC method). This is where the actual
  investigation in this conversation was spent, and it's the part that
  benefits most from JAX's autodiff/JIT/vmap.
- **The boundary is plain arrays**, not model objects: R never passes an
  R model object into Python, and Python never needs to know about
  formulas. This keeps `reticulate`'s job simple (numeric arrays convert
  cleanly) and keeps the Python backend testable/usable independent of R
  (e.g. directly from Python for development, or from a future non-R
  frontend).

## Current scope and limits

(Supersedes an earlier "deliberate v1 scope limits" list here, which
described several restrictions - MLE methods stuck at a 1D random
intercept, longitudinal formulas depending only on `time_var`,
intercept-only survival formulas - that have since been implemented.
`?jm_fit`'s Details section is the authoritative source for the exact
supported combinations; this is a summary.)

1. **Longitudinal formulas may include time-constant baseline
   covariates** (e.g. `y ~ time + age`), for all four methods. At
   `random_effects = "intercept_slope"` with a `long_formula` that
   isn't linear in `time_var` (e.g. a polynomial or spline term),
   supply `random_formula` explicitly rather than relying on the
   default "first `q` columns of `X`" convention.
2. **Survival formulas may include baseline covariates** (e.g.
   `Surv(time, event) ~ age + sex`). Fully supported by both MCMC
   methods at any `random_effects`/`functional_forms` combination;
   supported by the two MLE methods only with `random_effects` of
   `"intercept"` or `"intercept_slope"` and no `functional_forms`
   beyond the default value-only association.
3. **The two MLE methods now support 2D random effects**
   (`random_effects = "intercept_slope"`) via vector Newton
   mode-finding and tensor-product adaptive GH quadrature - the
   scalar-Newton limitation this section used to describe no longer
   applies. `functional_forms` beyond value-only association is more
   restricted for the MLE methods than for the MCMC methods, and in
   one case (`delta` at `q = 1` with a spline baseline hazard)
   restricted for a genuine identifiability reason rather than an
   implementation gap - see `?jm_fit`'s Details for the exact rules.
4. **Survival submodel must come from `coxph()`, not `survreg()`.**
   `survreg`-based (AFT) specs produced a confirmed internal-scale quirk
   in `JM` itself during prototyping (its `Time` ended up on an
   `exp(time)` scale) - this package sidesteps that class of issue
   entirely rather than replicating it.
5. **`control$orthogonalize_b0`/`control$orthogonalize_b`** (location-
   degeneracy reparameterization, experimental): requires
   `random_effects_corr = TRUE`, `q >= 2`, and the default
   `random_effects_method = "nuts"`. Correctness (fitted values
   unaffected) is verified; calibration/coverage has not yet been
   checked, which is why the option remains opt-in. See `?jm_fit` and
   `vignette("jmjax-reparameterization")`.

## Testing

Three tiers, from cheapest/always-run to most expensive/most-informative:

1. **`test-build-design.R`, `test-knots.R`** — pure R, no Python or `JM`
   needed. Test the design-matrix padding, formula-at-arbitrary-times
   evaluation, and knot placement directly. These always run and catch
   most R-side bugs fast.
2. **`test-recovers-truth.R`** — needs the Python backend (`jmjax_setup()`)
   but not `JM`. Fits on data simulated with known parameters (via
   `helper-simulate.R`, which numerically inverts the *actual*
   time-dependent hazard the backend fits - not a simplified closed-form
   stand-in, so it can only pass if `alpha*m_i(t)` round-trips correctly)
   and checks recovery within a few standard errors.
3. **`test-matches-JM.R`** — needs both the Python backend and R's `JM`
   package. Fits the same simulated data with both `jm_fit()` and
   `JM::jointModel()` (via `coxph`, matching this package's supported
   survival-submodel spec) and compares directly. Tolerances here are
   taken from the actual agreement observed during development, not
   guessed - see the comments in that file.
4. **`test-mcmc.R`** — needs the Python backend, not `JM` (there's no
   direct `JM` equivalent to compare against for the NUTS fit). Checks
   that population-level estimates and per-subject random effects stay
   properly separated (see `ranef()`), that diagnostics are real R-hat/ESS
   values rather than a placeholder, and that the fit recovers known
   simulation parameters within a loose tolerance.

Beyond these four tiers, a growing set of feature-specific files
(`test-baseline-covariates.R`, `test-functional-forms.R`,
`test-random-formula.R`, `test-spline-q2.R`, `test-penalized-spline.R`,
`test-precision.R`, `test-standardize-interaction.R`,
`test-orthogonalize.R` for `control$orthogonalize_b0`/`orthogonalize_b`,
and others) each targets one capability documented in `?jm_fit`; each
file's own header comment describes its scope rather than repeating it
here.

All Python-dependent tests call `skip_if_no_backend()` / `skip_if_no_JM()`
(in `helper-python.R`) so a missing environment shows up as **skipped**,
not **failed** - a fresh contributor without `jmjax_setup()` run yet, or a
CI job without R's `JM` installed, still gets a clean pass on what it can
actually run.

```r
devtools::test()
```

## Setup

```r
# once per machine (or in a Docker build step):
jmjax::jmjax_setup()

# thereafter, package load is fast and jm_fit() just works:
library(jmjax)
fit <- jm_fit(
  long_formula = y ~ time,
  surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim_long,
  data_surv = sim_surv,
  id_var = "id",
  time_var = "time",
  method = "spline-PH-aGH"
)
summary(fit)
```
