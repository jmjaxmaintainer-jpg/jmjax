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
      rotate_absorbable = FALSE, dense_mass_generator_beta = TRUE))))),
    "requires orthogonalize_rotate_all or rotate_absorbable"
  )
})

test_that("the rotation is on by default where it applies, and opts out cleanly", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 22)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc"
  )
  ctrl <- list(num_warmup = 20, num_samples = 20, num_chains = 1, progress_bar = FALSE)

  # Default (unset): applied, with the dense block, for q = 2.
  fit_d <- do.call(jm_fit, c(common, list(random_effects = "intercept_slope",
                                          control = ctrl)))
  rot <- fit_d$convergence$orthogonalize$rotation
  expect_identical(rot$mode, "all_columns_no_sweep")
  expect_true(isTRUE(rot$applied))
  expect_false(is.null(fit_d$posterior_samples$b_gen_U))

  # Opt-out: nothing reported, no generator site.
  fit_o <- do.call(jm_fit, c(common, list(random_effects = "intercept_slope",
                                          control = c(ctrl, list(rotate_absorbable = FALSE)))))
  expect_null(fit_o$convergence$orthogonalize)
  expect_null(fit_o$posterior_samples$b_gen_U)

  # q = 1 (random intercept only) is rotated by default too.
  fit_q1 <- do.call(jm_fit, c(common, list(random_effects = "intercept",
                                           control = ctrl)))
  expect_true(isTRUE(fit_q1$convergence$orthogonalize$rotation$applied))
  expect_false(is.null(fit_q1$posterior_samples$b))

  # Outside its scope (q = 2 without correlation) the default is silently
  # off, not an error.
  fit_nc <- do.call(jm_fit, c(common, list(random_effects = "intercept_slope",
                                           control = c(ctrl, list(random_effects_corr = FALSE)))))
  expect_null(fit_nc$convergence$orthogonalize)

  # A legacy sweep option turns the automatic rotation off rather than
  # colliding with it.
  fit_s <- do.call(jm_fit, c(common, list(random_effects = "intercept_slope",
                                          control = c(ctrl, list(orthogonalize_b0 = TRUE)))))
  expect_null(fit_s$posterior_samples$b_gen_U)
})

test_that("rotate_absorbable is exact: same fit as the unrotated parameterization", {
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

  fit_a <- do.call(jm_fit, c(common, list(control = c(ctrl, list(
    rotate_absorbable = FALSE)))))
  # the default: rotate_absorbable and dense_mass_generator_beta unset
  fit_r <- do.call(jm_fit, c(common, list(control = ctrl)))

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

test_that("rotate_absorbable is exact for q = 1 (random intercept only)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 200, seed = 23)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept"
  )
  ctrl <- list(num_warmup = 400, num_samples = 600, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = c(ctrl, list(
    rotate_absorbable = FALSE)))))
  fit_r <- do.call(jm_fit, c(common, list(control = ctrl)))

  rot <- fit_r$convergence$orthogonalize$rotation
  expect_identical(rot$mode, "all_columns_no_sweep")
  expect_true(isTRUE(rot$applied))
  expect_gte(as.integer(rot$k), 1L)
  expect_null(fit_a$convergence$orthogonalize)

  # The random effects come back in the usual q = 1 shape: one value per
  # subject per draw.
  expect_length(fit_r$posterior_samples$b[[1]], length(fit_a$posterior_samples$b[[1]]))

  for (pname in c("alpha", "beta_0", "beta_1", "sigma_e")) {
    tol <- 3 * max(fit_a$se[[pname]], fit_r$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_r$estimates[[pname]]), tol)
    expect_lt(fit_r$se[[pname]], 1.5 * fit_a$se[[pname]])
    expect_gt(fit_r$se[[pname]], fit_a$se[[pname]] / 1.5)
  }
})
