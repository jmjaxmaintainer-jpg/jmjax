# ==============================================================================
# Cross-validation against JM::jointModel.
#
# Tolerances here are not arbitrary - they're taken directly from the
# agreement actually observed between jmjax's approach and JM::jointModel
# during development (see package history): once a coxph-based survObject
# is used on both sides, longitudinal parameters and variance components
# matched to ~3 decimal places, and alpha/log-likelihood matched to well
# within one standard error. Tests use somewhat looser tolerances than that
# best-case agreement to avoid flaking on the smaller n used here for test
# speed (300 vs 1500 subjects in the original comparison).
# ==============================================================================

# NOTE: JM::jointModel()'s internal code calls some nlme generics (e.g.
# pdMatrix()) unqualified, so nlme must be ATTACHED via library(), not just
# accessed via nlme:: - otherwise you'll see
# "could not find function 'pdMatrix'" even though nlme is installed and
# nlme::lme() itself works fine. This is a JM package quirk, not a jmjax bug.
if (requireNamespace("JM", quietly = TRUE)) {
  library(nlme)
  library(survival)
}

test_that("weibull-PH-aGH matches JM::jointModel(method='weibull-PH-aGH')", {
  skip_if_no_backend()
  skip_if_no_JM()

  sim <- simulate_joint_data(n = 300, seed = 44)

  fit_py <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  lme_fit <- nlme::lme(y ~ time, random = ~ 1 | id, data = sim$data_long)
  surv_fit <- survival::coxph(survival::Surv(time, event) ~ 1, data = sim$data_surv, x = TRUE)
  jm_fit_r <- JM::jointModel(
    lmeObject = lme_fit, survObject = surv_fit, timeVar = "time",
    method = "weibull-PH-aGH", control = list(GHk = 15)
  )

  # --- Longitudinal parameters: expect tight agreement ---
  expect_equal(fit_py$estimates[["beta_0"]], unname(jm_fit_r$coefficients$betas[1]), tolerance = 0.02)
  expect_equal(fit_py$estimates[["beta_1"]], unname(jm_fit_r$coefficients$betas[2]), tolerance = 0.02)
  expect_equal(fit_py$estimates[["sigma_e"]], jm_fit_r$coefficients$sigma, tolerance = 0.02)
  expect_equal(fit_py$estimates[["sigma_b"]], unname(sqrt(jm_fit_r$coefficients$D[1, 1])), tolerance = 0.05)

  # --- Association parameter: this is the one that took the longest to
  # reconcile during development (coxph vs survreg turned out to matter a
  # lot here) - looser tolerance, but should still be well within 3 SEs.
  alpha_py <- fit_py$estimates[["alpha"]]
  alpha_r <- unname(jm_fit_r$coefficients$alpha)
  alpha_se_py <- fit_py$se[["alpha"]]
  expect_lt(abs(alpha_py - alpha_r), 3 * alpha_se_py)

  # --- Log-likelihood: should agree to a small fraction per subject ---
  expect_equal(fit_py$loglik, jm_fit_r$logLik, tolerance = 0.02)
})

test_that("spline-PH-aGH matches JM::jointModel(method='spline-PH-aGH') on alpha and loglik", {
  skip_if_no_backend()
  skip_if_no_JM()

  sim <- simulate_joint_data(n = 300, seed = 45)

  fit_py <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    control = list(n_interior_knots = 5)
  )

  lme_fit <- nlme::lme(y ~ time, random = ~ 1 | id, data = sim$data_long)
  surv_fit <- survival::coxph(survival::Surv(time, event) ~ 1, data = sim$data_surv, x = TRUE)
  jm_fit_r <- JM::jointModel(
    lmeObject = lme_fit, survObject = surv_fit, timeVar = "time",
    method = "spline-PH-aGH", control = list(GHk = 15)
  )

  # NOTE: individual spline coefficients (W_i vs bs_i) are NOT compared
  # directly here, even though R/knots.R ports JM's exact knot placement -
  # per the h0(t) overlay investigation during development, raw coefficient
  # comparison is fragile in sparse-tail regions even when both fits are
  # correct. alpha and log-likelihood are the well-identified, comparable
  # quantities; a full h0(t)-curve overlay test (see test-baseline-hazard.R,
  # not yet implemented) would be the appropriate place for shape-level
  # comparison.
  alpha_py <- fit_py$estimates[["alpha"]]
  alpha_r <- unname(jm_fit_r$coefficients$alpha)
  alpha_se_py <- fit_py$se[["alpha"]]
  expect_lt(abs(alpha_py - alpha_r), 3 * alpha_se_py)

  # Per-subject log-lik difference should be small even with basis/knot
  # differences at this smaller n.
  expect_lt(abs(fit_py$loglik - jm_fit_r$logLik) / 300, 0.05)

  expect_equal(fit_py$estimates[["beta_0"]], unname(jm_fit_r$coefficients$betas[1]), tolerance = 0.02)
  expect_equal(fit_py$estimates[["beta_1"]], unname(jm_fit_r$coefficients$betas[2]), tolerance = 0.02)
})
