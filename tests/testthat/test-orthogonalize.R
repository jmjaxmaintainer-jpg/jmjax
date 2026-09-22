# ==============================================================================
# Regression tests for control$orthogonalize_b0 / control$orthogonalize_b
# (the location-degeneracy reparameterization; see ?jm_fit and
# vignette("jmjax-reparameterization") for the full mechanism).
#
# Prior to this file, everything validating this option lived in one-off
# dev/ scripts (diagnose_beta0*.R, test_orthogonalize_b0*.R,
# verify_orth_slope.R, and the dev/study_*.R simulation studies) - useful
# for the original investigation, but nothing here would catch a future
# change silently breaking the option. These tests are deliberately not a
# re-run of that investigation (no attempt to reproduce the measured 6-15x
# ESS ratios with statistical rigor - that needs the much larger,
# multi-seed replication in dev/study_orth_vs_jmbayes2.R and is recorded in
# vignette("jmjax-reparameterization") instead). They check the three
# things a silent regression would actually break: the option still
# refuses configurations it does not support, it still does not change
# what the model fits (the "model is unchanged" claim in ?jm_fit is a
# testable consequence of the construction, not just an algebra argument),
# and it still measurably helps the parameter it targets.
# ==============================================================================

test_that("orthogonalize_b0/orthogonalize_b refuse configurations outside their supported scope", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 11)

  # q = 1 (random_effects = "intercept"): orthogonalize_b0 requires q >= 2.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept",
      control = list(orthogonalize_b0 = TRUE,
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )

  # q = 2 but random_effects_corr = FALSE: orthogonalize_b requires
  # random_effects_corr = TRUE.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept_slope",
      control = list(orthogonalize_b = TRUE, random_effects_corr = FALSE,
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )

  # q = 2, corr = TRUE, but random_effects_method != "nuts": orthogonalize_b0
  # requires the default NUTS random-effects sampler.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept_slope",
      control = list(orthogonalize_b0 = TRUE,
                     random_effects_method = "wishart_gibbs",
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )
})

test_that("fit$convergence$orthogonalize reports which columns were actually constrained", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 150, seed = 12)

  # orthogonalize_b0: intercept column (0) only. _absorbable_basis() should
  # find the intercept as an absorbable direction (b[,0]); the slope column
  # (1) is not even attempted at this option, so it must come back
  # unconstrained - this is the routine, by-design case for orthogonalize_b0
  # (distinct from the "no fixed-effect column can absorb anything" warning
  # path, which is not exercised here).
  fit_d <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(orthogonalize_b0 = TRUE,
                   num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )

  orth_d <- fit_d$convergence$orthogonalize
  expect_false(is.null(orth_d))
  expect_identical(orth_d$requested, "b0")
  expect_true(isTRUE(orth_d$any_applied))
  expect_length(orth_d$columns, 2)

  col0_d <- orth_d$columns[[1]]
  expect_equal(col0_d$column, 0)
  expect_true(isTRUE(col0_d$applied))
  expect_gte(col0_d$n_directions, 1)

  col1_d <- orth_d$columns[[2]]
  expect_equal(col1_d$column, 1)
  expect_false(isTRUE(col1_d$applied))  # orthogonalize_b0 does not touch the slope

  # orthogonalize_b: every column. The design's fixed "time" term means the
  # slope column (1) should ALSO come back constrained - this is the
  # k-independent slope degeneracy documented in ?jm_fit and
  # vignette("jmjax-reparameterization"); it does not require any extra
  # covariates, just a fixed slope on time alongside the random one.
  fit_e <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(orthogonalize_b = TRUE,
                   num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )

  orth_e <- fit_e$convergence$orthogonalize
  expect_false(is.null(orth_e))
  expect_identical(orth_e$requested, "b")
  expect_true(isTRUE(orth_e$any_applied))
  expect_length(orth_e$columns, 2)
  expect_true(isTRUE(orth_e$columns[[1]]$applied))
  expect_true(isTRUE(orth_e$columns[[2]]$applied))
  expect_gte(orth_e$columns[[2]]$n_directions, 1)

  # A default fit (neither option requested) must report NULL, not an
  # empty/degenerate structure - callers use is.null() to check applicability.
  fit_a <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )
  expect_null(fit_a$convergence$orthogonalize)
})

test_that("orthogonalize_b0/orthogonalize_b do not change fitted population estimates", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # The construction guarantees this algebraically (every direction swept
  # out of a random-effect column lies in a column space beta already
  # spans - see ?jm_fit, "The model is unchanged"), but algebra and
  # implementation are different claims; this checks the second one. Loose,
  # multiple-of-SE tolerances, same spirit as the existing MCMC
  # truth-recovery tests - this is not a replication of the tight,
  # multi-seed agreement already established in
  # vignette("jmjax-reparameterization").
  sim <- simulate_joint_data_re2(n = 300, seed = 13)

  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 500, num_samples = 800, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_d <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b0 = TRUE)))))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  for (pname in c("alpha", "beta_0", "beta_1")) {
    tol_d <- 3 * max(fit_a$se[[pname]], fit_d$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_d$estimates[[pname]]), tol_d)

    tol_e <- 3 * max(fit_a$se[[pname]], fit_e$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_e$estimates[[pname]]), tol_e)
  }
})

test_that("orthogonalize_b materially improves beta_1 mixing over the default fit", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # The regression-guard for the headline efficiency claim: measured
  # regression-only ESS/sec ratios in the replicated k-sweep and aids
  # real-data study (vignette("jmjax-reparameterization")) range roughly
  # 5x-15x. A 1.5x threshold on a single seed leaves enormous margin
  # against Monte Carlo noise while still catching the failure mode this
  # test exists for: a future change that silently disables or breaks the
  # reparameterization.
  sim <- simulate_joint_data_re2(n = 300, seed = 14)

  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  ess_a <- fit_a$diagnostics$ess[["beta_1"]]
  ess_e <- fit_e$diagnostics$ess[["beta_1"]]

  expect_true(is.finite(ess_a) && is.finite(ess_e))
  expect_gt(ess_e, 1.5 * ess_a)
})
