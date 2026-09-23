# ==============================================================================
# Tests for control$rotate_absorbable + control$dense_mass_generator_beta,
# jmjax's construction for the location degeneracy (see ?jm_fit and
# vignette("jmjax-reparameterization")): every random-effect column of b_std is
# rotated by one fixed Householder matrix built from the absorbable
# directions, and beta plus the explicit generator block get one dense mass
# block. The rotation is exact, so the fitted model must be unchanged; these
# tests check that, and that unsupported combinations are refused. The
# efficiency and calibration studies are dev/pilot_rotate_grid.R and
# dev/study_calibration.R, not these tests.
# ==============================================================================

test_that("rotate_absorbable (no-sweep rotation) refuses to combine with the sweep", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 20)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 20, num_samples = 20, num_chains = 1, progress_bar = FALSE)
  expect_error(
    do.call(jm_fit, c(common, list(control = c(ctrl, list(
      rotate_absorbable = TRUE, orthogonalize_b0 = TRUE))))),
    "no-sweep alternative"
  )
  expect_error(
    do.call(jm_fit, c(common, list(control = c(ctrl, list(
      dense_mass_generator_beta = TRUE))))),
    "requires orthogonalize_rotate_all or rotate_absorbable"
  )
})

test_that("rotate_absorbable is exact: same fit as the default parameterization", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 200, seed = 21)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 400, num_samples = 600, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_r <- do.call(jm_fit, c(common, list(control = c(ctrl, list(
    rotate_absorbable = TRUE, dense_mass_generator_beta = TRUE)))))

  rot <- fit_r$convergence$orthogonalize$rotation
  expect_false(is.null(rot))
  expect_identical(rot$mode, "all_columns_no_sweep")
  expect_true(isTRUE(rot$applied))
  expect_gte(as.integer(rot$k), 1L)
  # No sweep: nothing to correct, and beta is the ordinary coefficient.
  expect_null(fit_r$posterior_samples$beta_corrected)

  for (pname in c("alpha", "beta_0", "beta_1", "sigma_e")) {
    tol <- 3 * max(fit_a$se[[pname]], fit_r$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_r$estimates[[pname]]), tol)
    # Same posterior spread, not just the same centre - a factor of 1.5
    # either way at this single seed and chain length.
    expect_lt(fit_r$se[[pname]], 1.5 * fit_a$se[[pname]])
    expect_gt(fit_r$se[[pname]], fit_a$se[[pname]] / 1.5)
  }
})
