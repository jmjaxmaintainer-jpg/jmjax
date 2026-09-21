# NOTE: JM::jointModel()'s internal code calls some nlme generics (e.g.
# pdMatrix()) unqualified, relying on nlme being ATTACHED to the search
# path rather than just installed/namespace-accessible - otherwise you'll
# see "could not find function 'pdMatrix'" even though nlme is installed
# and nlme::lme() itself works fine. Same fix as test-matches-JM.R, which
# already documented and solved this - this file used nlme:: (namespaced)
# calls without ever attaching the package, which is what exposed the gap
# specifically here.
if (requireNamespace("JM", quietly = TRUE)) {
  library(nlme)
  library(survival)
}

test_that("spline-PH-aGH with random_effects = 'intercept_slope' (q=2) recovers true parameters", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 500, seed = 51)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    random_effects = "intercept_slope",
    control = list(n_interior_knots = 5)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("sigma_b0", "sigma_b1", "rho", "alpha") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["alpha"]] - sim$truth$alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sim$truth$sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sim$truth$sigma_b1), 3 * fit$se[["sigma_b1"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

test_that("spline-PH-aGH q=2 matches JM::jointModel's spline-PH-aGH with a real random slope", {
  skip_if_no_backend()
  skip_if_no_JM()

  sim <- simulate_joint_data_re2(n = 500, seed = 52)

  fit_jmjax <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    random_effects = "intercept_slope",
    control = list(n_interior_knots = 5)
  )

  lme_fit <- nlme::lme(y ~ time, random = ~ time | id, data = sim$data_long,
                        control = nlme::lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- survival::coxph(survival::Surv(time, event) ~ 1, data = sim$data_surv, x = TRUE)

  fit_jm <- JM::jointModel(
    lmeObject = lme_fit, survObject = cox_fit, timeVar = "time",
    method = "spline-PH-aGH"
  )

  # A generous but still meaningful tolerance - this is a genuinely new
  # capability being cross-validated for the first time (unlike the
  # extensively-tuned Weibull q=2 comparison, which matched to 3-4
  # decimal places after significant validation work). Catches a real
  # implementation problem (wrong sign, order-of-magnitude difference)
  # without demanding immediate near-exact agreement.
  expect_lt(abs(fit_jmjax$estimates[["alpha"]] - unname(fit_jm$coefficients$alpha)), 0.3)
})

test_that("spline-PH-aGH q=2 + delta RESOLVES the q=1 identifiability problem: recovers both true association parameters", {
  skip_if_no_backend()

  # THE key test of today's work: a genuine random slope should give
  # delta(t) = (beta1+b1_i)*t real subject-level variation that a
  # population-level spline curve cannot absorb, unlike the q=1 case
  # where delta(t) = beta1*t is a purely deterministic function of t that
  # the spline COULD (and did) fully absorb, leaving alpha_delta
  # unidentified (see jm_fit.R's guard and vignette("jmjax-validation")'s
  # "Confirmed non-identifiable" section for the q=1 story this resolves).
  sim <- simulate_joint_data_re2_channel("delta", n = 500, seed = 61,
                                          alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + delta(y),
    control = list(n_interior_knots = 5)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))

  # Crucially: unlike the q=1 case, expect FINITE (non-NaN) standard
  # errors here - a finite SE for alpha_delta and the spline coefficients
  # is itself direct evidence the identifiability problem is resolved,
  # independent of whether the point estimate also happens to be close to
  # truth.
  expect_true(is.finite(fit$se[["alpha_delta"]]))
  expect_true(all(is.finite(fit$se[grepl("^W", names(fit$se))])))

  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_delta"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sim$truth$sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sim$truth$sigma_b1), 3 * fit$se[["sigma_b1"]])
})
