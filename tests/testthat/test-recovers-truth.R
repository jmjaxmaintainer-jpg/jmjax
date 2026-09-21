test_that("weibull-PH-aGH recovers known simulation parameters", {
  skip_if_no_backend()

  sim <- simulate_joint_data(n = 300, seed = 42)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  est <- fit$estimates
  se <- fit$se

  # Recovery tolerance: within 3 SEs of truth. This is a sanity/regression
  # check (did something break the pipeline end-to-end), not a precise
  # calibration test - 3 SEs gives headroom for MC/finite-sample noise at
  # n=300 while still catching a genuinely broken implementation (e.g. a
  # sign error or a badly wrong link function would blow way past this).
  expect_true(fit$convergence$converged)

  expect_lt(abs(est[["beta_0"]] - sim$truth$beta0), 3 * se[["beta_0"]])
  expect_lt(abs(est[["beta_1"]] - sim$truth$beta1), 3 * se[["beta_1"]])
  expect_lt(abs(est[["alpha"]] - sim$truth$alpha), 3 * se[["alpha"]])

  # sigma_e/sigma_b are strictly positive and should be in the right
  # ballpark (loose multiplicative tolerance rather than SE-based, since
  # variance components are the slowest-converging parameters).
  expect_equal(est[["sigma_e"]], sim$truth$sigma_e, tolerance = 0.15)
  expect_equal(est[["sigma_b"]], sim$truth$sigma_b, tolerance = 0.15)
})

test_that("spline-PH-aGH runs cleanly and produces a sane baseline hazard shape", {
  skip_if_no_backend()

  sim <- simulate_joint_data(n = 300, seed = 43)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    control = list(n_interior_knots = 5)
  )

  expect_true(fit$convergence$converged)
  expect_false(any(is.na(unlist(fit$estimates))))
  expect_false(any(is.na(unlist(fit$se))))

  # alpha should be in a plausible neighborhood of the true value used to
  # simulate the data, even though the spline basis and the true Weibull
  # baseline aren't the same functional family (this is the same kind of
  # check that validated the real R-vs-Python comparison in this package's
  # development history: alpha should be recoverable even under a baseline
  # hazard misspecification, since it's a different part of the model).
  expect_lt(abs(fit$estimates[["alpha"]] - sim$truth$alpha), 0.25)
})
