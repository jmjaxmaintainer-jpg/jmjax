# jmjax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22950083.svg)](https://doi.org/10.5281/zenodo.22950083)

Joint models for longitudinal and time-to-event data, fitted with a
[JAX](https://github.com/jax-ml/jax)/[NumPyro](https://num.pyro.ai)
backend. You write the model with ordinary R formulas; jmjax builds the
design in R and hands the numerical work to JAX - adaptive Gauss-Hermite
quadrature for maximum likelihood, and the No-U-Turn sampler (NUTS) for
Bayesian estimation.

The model is the standard shared-parameter joint model: a linear
mixed-effects model for the marker, and a proportional-hazards model whose
hazard depends on the subject's current marker value (or its change or
cumulative value). Results are cross-checked against the `JM` and
`JMbayes2` packages.

## Installation

```r
# install.packages("remotes")
remotes::install_github("jmjaxmaintainer-jpg/jmjax")

# Once per machine: creates a dedicated Python environment ("r-jmjax",
# about 150 MB) with pinned versions of jax, numpyro, numpy and scipy.
jmjax::jmjax_setup()
```

jmjax needs Python 3.9 or later. If you already have an environment with
the pinned versions (see `DESCRIPTION`), select it with
`reticulate::use_virtualenv()` or `reticulate::use_condaenv()` before the
first fit instead. `jmjax_available()` tells you whether the backend can be
used.

## Example

The primary biliary cirrhosis data from the `JM` package: log serum
bilirubin measured over time, and time to death.

```r
library(jmjax)
data("pbc2", "pbc2.id", package = "JM")
pbc2$log_bili <- log(pbc2$serBilir)

# Bayesian estimation (NUTS), spline baseline hazard, correlated random
# intercept and slope
fit <- jm_bayes(log_bili ~ year, survival::Surv(years, status2) ~ drug,
                data_long = pbc2, data_surv = pbc2.id,
                id_var = "id", time_var = "year",
                random_effects = "intercept_slope", random_formula = ~ year)

fit             # the association parameter and its hazard ratio first
summary(fit)    # estimates, credible intervals, R-hat and ESS
confint(fit, "alpha")

# Survival probabilities for a subject still alive at the end of their
# follow-up, from that point on
id_c <- pbc2.id$id[pbc2.id$status2 == 0 & pbc2.id$years < 8][1]
predict(fit, newdata = pbc2[pbc2$id == id_c, ], process = "event")

# Maximum likelihood with a Weibull baseline hazard: fast, and gives AIC
fit_ml <- jm_mle(log_bili ~ year, survival::Surv(years, status2) ~ drug,
                 data_long = pbc2, data_surv = pbc2.id,
                 id_var = "id", time_var = "year", baseline = "weibull")
AIC(fit_ml)
```

## Fitting functions

| Function | Estimation | Baseline hazard (`baseline =`) |
|---|---|---|
| `jm_mle()` | maximum likelihood, adaptive Gauss-Hermite quadrature | `"weibull"` (default) or `"spline"` |
| `jm_bayes()` | Bayesian, NUTS | `"spline"` (default; optionally penalized) or `"weibull"` |

Both take the same model arguments: a random intercept or a correlated
random intercept and slope, baseline covariates in both submodels, and
`functional_forms` to choose how the marker enters the hazard -
`value(y)`, `delta(y)`, `area(y)` or `area_avg(y)`, with JMbayes2's
syntax. Each documents only its own options. `jm_fit()`, which both call,
remains available as the general interface with every option; `?jm_fit`
also lists the combinations each method supports.

A fit works with the usual tools: `print()`, `summary()`, `coef()`,
`vcov()`, `confint()`, `logLik()` (so `AIC()` and `BIC()`), `nobs()`,
and, for `jm_bayes()` fits, `ranef()` and `predict()`.

## Documentation

- `vignette("jmjax-introduction")`: a tour of the package.
- `vignette("jmjax-validation")`: how the results were checked against
  `JM` and `JMbayes2`, with the benchmark results.
- `vignette("jmjax-reparameterization")`: the rotation of the random
  effects that jmjax uses by default for NUTS, and why it improves mixing.

## Status

Version 0.3.1. `predict()` currently covers `jm_bayes()` fits, subjects in
the fitted data and the `value` association; dynamic prediction for new
subjects is planned. See `NEWS.md` for the changes in this release.

## Development

Run the tests with `devtools::test()`. Tests that need the Python backend
or `JM`/`JMbayes2` are skipped when those are missing. Notes on how the
R and Python sides divide the work are in `dev/notes/architecture.md`.

## How to cite

If you use jmjax, please cite the software release:

> Guo, C. (2026). *jmjax: Joint Models for Longitudinal and Survival Data
> via JAX*. https://doi.org/10.5281/zenodo.22950083

That DOI always resolves to the latest release. To cite the exact version
you used, take its own DOI from the Zenodo record (version 0.3.0 is
https://doi.org/10.5281/zenodo.22950084).

For the rotation of the random effects that jmjax uses by default for
NUTS, please also cite the method:

> Guo, C. (2026). Rotating Away the Location Degeneracy: An Exact
> Reparameterization for Hamiltonian Monte Carlo in Hierarchical and Joint
> Models. Preprint. https://doi.org/10.5281/zenodo.22961412

`citation("jmjax")` gives both references, with BibTeX entries.

## License

MIT
