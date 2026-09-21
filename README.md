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

## Deliberate v1 scope limits (see design discussion this grew out of)

1. **MLE methods (`weibull-PH-aGH`, `spline-PH-aGH`) support only a 1D
   random intercept.** Adaptive GH's per-subject Newton mode-finding is
   scalar; multivariate random effects would need vector Newton + a real
   Hessian + tensor-product GH nodes, which is a bigger lift and was
   deliberately deferred. `spline-PH-mcmc` supports random intercept +
   slope today, because NUTS doesn't have this scaling problem.
2. **Longitudinal formulas may only depend on `time_var`.** `build_time_design()`
   evaluates the formula's RHS at survival/quadrature times, which
   requires knowing what to hold fixed for any other covariate - not yet
   implemented. Extending to `y ~ time * treatment`-style formulas is a
   natural v2 step once this path is validated end-to-end.
3. **Only intercept-only relative-risk survival formulas** (`Surv(time,
   event) ~ 1`) are supported; baseline covariates in the survival
   submodel are not yet wired through.
4. **Survival submodel must come from `coxph()`, not `survreg()`.**
   `survreg`-based (AFT) specs produced a confirmed internal-scale quirk
   in `JM` itself during prototyping (its `Time` ended up on an
   `exp(time)` scale) - this package sidesteps that class of issue
   entirely rather than replicating it.

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
