test_that("spline-PH-mcmc runs cleanly, separates population params from random effects, and reports real diagnostics", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data(n = 150, seed = 99)  # smaller n: MCMC is slower than the MLE methods

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    random_effects = "intercept",
    control = list(num_warmup = 300, num_samples = 500, num_chains = 1, progress_bar = FALSE)
  )

  # --- Population estimates should NOT include per-subject effects ---
  pop_names <- names(fit$estimates)
  expect_false(any(grepl("^b[0-9]*$", pop_names)) || any(grepl("^b_mean", pop_names)))
  expect_true(all(c("beta_0", "beta_1", "sigma_e", "sigma_b", "alpha") %in% pop_names))
  expect_true(any(grepl("^W[0-9]+$", pop_names)))  # W0, W1, ... present

  # --- Random effects are separated out, one row per subject ---
  re <- ranef(fit)
  expect_equal(nrow(re), 150)
  expect_true(all(c("subject_index", "b_mean", "b_sd") %in% names(re)))

  # --- Real diagnostics, not a placeholder ---
  expect_true(!is.null(fit$diagnostics$rhat))
  expect_true(!is.null(fit$diagnostics$ess))
  rhat_vals <- unlist(fit$diagnostics$rhat)
  expect_true(all(is.finite(rhat_vals)))
  expect_true(all(rhat_vals > 0.8))  # sanity bound - real R-hat, not a stub value

  # loglik is deliberately NULL for MCMC (see mcmc_model.py comments) - this
  # should not error, and print() should handle it gracefully.
  expect_null(fit$loglik)
  expect_output(print(fit), "not reported for method")
})

test_that("spline-PH-mcmc recovers known simulation parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data(n = 200, seed = 100)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_true(fit$convergence$converged)

  # Loose tolerances - posterior mean vs a single simulated truth, similar
  # spirit to test-recovers-truth.R's MLE checks (3x posterior SD).
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["alpha"]] - sim$truth$alpha), 3 * fit$se[["alpha"]])

  # sigma_e specifically: this check was MISSING in the original version of
  # this test, which let a real bug through undetected (a no-op mask in the
  # longitudinal likelihood scored padded zero-observations as real
  # perfect-fit data points, deflating sigma_e by ~50%+ and cascading into
  # biased variance components elsewhere). Never drop this check.
  expect_equal(fit$estimates[["sigma_e"]], sim$truth$sigma_e, tolerance = 0.15)
})

test_that("spline-PH-mcmc with random_effects = 'intercept_slope' recovers truth on well-specified data", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Uses simulate_joint_data_re2() (GENUINE correlated random intercept +
  # slope), NOT simulate_joint_data() - forcing intercept_slope onto
  # slope-free data produces real attenuation bias that would make this
  # check meaningless (see helper-simulate.R's docs for that function).
  #
  # This exact test (checking sigma_e, sigma_b0, sigma_b1 - not just
  # beta/alpha) is what caught a real bug during development: a no-op mask
  # in the longitudinal likelihood was scoring padded observations as real
  # zero-residual data points, deflating sigma_e and cascading into a
  # ~75-120% overestimate of sigma_b1 and a biased alpha, persisting even
  # at n=1500 (ruling out "just needs more data"). Fixed in mcmc_model.py.
  sim <- simulate_joint_data_re2(n = 400, seed = 202)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    random_effects = "intercept_slope",
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_true(fit$convergence$converged)

  expect_equal(fit$estimates[["sigma_e"]], sim$truth$sigma_e, tolerance = 0.2)
  expect_equal(fit$estimates[["sigma_b0"]], sim$truth$sigma_b0, tolerance = 0.25)
  expect_equal(fit$estimates[["sigma_b1"]], sim$truth$sigma_b1, tolerance = 0.5)
  expect_lt(abs(fit$estimates[["alpha"]] - sim$truth$alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

test_that("ranef.jmjax warns and returns NULL for MLE methods (no per-subject effects)", {
  skip_if_no_backend()

  sim <- simulate_joint_data(n = 100, seed = 101)
  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  expect_warning(re <- ranef(fit), "only the MCMC methods")
  expect_null(re)
})
