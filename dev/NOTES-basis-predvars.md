# Data-dependent longitudinal bases (`poly()`, `ns()`, `bs()`) are re-evaluated inconsistently

Status: **unconfirmed by execution** — found by reading `build_time_design()`
while designing `dev/study_calibration.R`, not by running a fit. The
verification below takes about ten seconds and settles it either way.

## What the code does

`R/build_design.R`, `build_time_design()`:

```r
rhs_terms <- stats::delete.response(stats::terms(long_formula))

eval_at <- function(t_vec) {
  newdata <- data.frame(t_vec)
  names(newdata) <- time_var
  if (!is.null(baseline_covariates)) newdata <- cbind(newdata, baseline_covariates)
  stats::model.matrix(rhs_terms, newdata)
}
```

`stats::terms()` is called on a **bare formula**, with no data, so the terms
object carries no `predvars` attribute. `predvars` is the mechanism by which R
records the basis a data-dependent term was built with — the centring and
scaling for `poly()`, the knot placement for `ns()`/`bs()` — so that
`model.matrix()` on new data reproduces *that* basis rather than fitting a
fresh one. Without it, every call to `eval_at()` rebuilds the basis from
whatever vector it was handed.

`eval_at()` is called on three different grids, and in the array branch it is
called **once per quadrature column**:

- `X_long` — the observed longitudinal times
- `X_time_surv` — the event times `T_i`
- `X_time_quad` — each Gauss-Legendre node, separately, in a loop

So with `long_formula = y ~ splines::ns(time, 3)`, the knots are placed at the
quantiles of the observed times for `X_long`, at the quantiles of the event
times for `X_time_surv`, and at the quantiles of *each individual quadrature
column* for `X_time_quad`. These are different bases. The same `beta` vector
then means something different in each one, and the shared trajectory
`m_i(t) = x_i(t)' beta + z_i(t)' b_i` is no longer the function the
longitudinal submodel fitted.

`I()` expressions are unaffected: `I(time^2)` and `I(time + 0.5 * time^2)` are
pure functions of the time value, with no dependence on the vector they are
evaluated over. This is why `dev/study_calibration.R`'s `mixing` design is
spelled with `I()` rather than `poly()`.

## Why it matters

Both `?jm_fit` (R/jm_fit.R:67) and `build_random_long_array`'s docs
(R/build_design.R:61) name `y ~ splines::ns(time, 3)` as a supported
longitudinal formula. If the above is right, that configuration is silently
wrong rather than unsupported — no error, no warning, just a model whose
survival submodel disagrees with its longitudinal one.

This is independent of the orthogonalization work. It predates it and would
affect any fit using a data-dependent longitudinal basis.

## Verification

Two parts. First, that the bases actually differ:

```r
rt <- stats::delete.response(stats::terms(y ~ splines::ns(time, 3)))
a  <- stats::model.matrix(rt, data.frame(time = seq(0, 10, length.out = 60)))
b  <- stats::model.matrix(rt, data.frame(time = c(2, 7)))
# Evaluate the SAME time points under both:
stats::model.matrix(rt, data.frame(time = c(2, 7)))          # basis from 2 points
a[c(13, 43), ]                                               # basis from 60 points
# If these rows disagree, the basis is not being carried across grids.
```

Then, that it reaches a fit. `_structural_extension_residual()` (added with
the absorbable-basis repair) checks exactly the property that breaks here:
whether a relation established on the longitudinal grid still holds at the
event and quadrature times. So:

```r
sim <- sim_joint(n = 200, seed = 1, k_extra = 1)
f <- jm_fit(long_formula = y ~ splines::ns(time, 3) + age,
            surv_formula = survival::Surv(time, event) ~ trt,
            data_long = sim$data_long, data_surv = sim$data_surv,
            id_var = "id", time_var = "time",
            method = "spline-PH-mcmc", random_effects = "intercept_slope",
            random_formula = ~ time,
            control = list(orthogonalize_b = TRUE, num_warmup = 200,
                           num_samples = 200, num_chains = 1,
                           progress_bar = FALSE))
```

If the Condition (S) warning fires here and does not fire for the equivalent
`I()` spelling, that is the bug reaching a real fit, caught by the check.
Note the (S) check only runs when `orthogonalize_b`/`_b0` is requested, so it
is a diagnostic for this, not a general guard — a default fit with an `ns()`
longitudinal formula would still be silently affected.

## If confirmed

The fix is to build the terms object against the data once, so `predvars` is
recorded, and reuse it for every grid:

```r
mf <- stats::model.frame(long_formula, data = data_long)
rhs_terms <- stats::delete.response(stats::terms(mf))   # carries predvars
```

and thread that object into `build_time_design()` instead of the bare formula.
The same applies to `random_formula` and `Z`.
