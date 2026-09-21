test_that("build_random_long_array produces a genuinely different Z than X for a mismatched random_formula", {
  set.seed(1)
  data_long <- data.frame(
    id = rep(1:5, each = 4),
    time = rep(c(0, 1, 2, 3), 5)
  )

  # long_formula has 3 columns (intercept, time, time^2); random_formula
  # asks for only 2 (intercept, time) - these must NOT be related by
  # simple column slicing, since I(time^2) sits BETWEEN them in X's
  # column order in some model.matrix conventions, and more fundamentally
  # because Z is built from a genuinely different formula.
  X <- stats::model.matrix(~ time + I(time^2), data_long)
  subj_ids <- 1:5
  max_obs <- 4

  Z_long <- build_random_long_array(~ time, data_long, "id", subj_ids, max_obs)

  expect_equal(dim(Z_long), c(5, 4, 2))
  # Z's own columns should be exactly [1, time] - NOT [1, time] sliced
  # from X (which would coincidentally look the same here, but confirm it
  # via Z's own dimnames/values rather than assuming).
  expect_equal(dimnames(Z_long)[[3]], c("(Intercept)", "time"))
  expect_equal(Z_long[1, , 2], c(0, 1, 2, 3))  # time column, subject 1
})

test_that("build_random_long_array errors clearly on a missingness mismatch with X", {
  data_long <- data.frame(id = rep(1:3, each = 2), time = c(0, 1, 0, 1, 0, 1),
                           covariate = c(1, NA, 1, 1, 1, 1))
  expect_error(
    build_random_long_array(~ covariate, data_long, "id", 1:3, 2),
    "different missingness pattern"
  )
})

test_that("random_formula = NULL (default) gives IDENTICAL results to before - backward compatibility", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 300, seed = 71)

  fit_default <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope"
  )

  fit_explicit_random_formula <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_formula = ~ time  # equivalent to random_effects = "intercept_slope" here
  )

  # Should be numerically identical (or extremely close - both paths feed
  # the SAME underlying [1, time] design into the same optimizer) since
  # y ~ time IS the case where legacy slicing and independent evaluation
  # coincide exactly.
  expect_equal(fit_default$estimates, fit_explicit_random_formula$estimates, tolerance = 1e-6)
})

test_that("random_formula correctly fixes recovery for a polynomial long_formula (the originally-broken scenario)", {
  skip_if_no_backend()

  # THE decisive test: long_formula is a genuine quadratic-in-time
  # trajectory, so the legacy Z = X[:, :2] slicing would have taken
  # [intercept, time] OR [intercept, time^2] depending on model.matrix's
  # column order - neither is obviously "right", and more fundamentally
  # the whole premise of slicing X's columns to get Z breaks down for a
  # non-trivial fixed-effects basis. random_formula = ~ time asks for the
  # random-effects structure explicitly and correctly regardless of what
  # long_formula's own basis looks like.
  set.seed(81)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.3; beta2 <- 0.05
  sigma_b0 <- 0.7; sigma_b1 <- 0.15; rho <- 0.2; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)

  cum_hazard <- function(t, b0_i, b1_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      # QUADRATIC in time - the fixed-effects trajectory genuinely isn't
      # linear, so its design matrix's columns are NOT "intercept, slope".
      value_s <- (beta0 + b0_i) + (beta1 + b1_i) * s + beta2 * s^2
      h0 * exp(alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + beta2 * vt^2 + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + I(time^2),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_formula = ~ time  # independent from long_formula's 3-column design
  )

  expect_true(fit$convergence$converged)
  # 3 beta coefficients (intercept, time, time^2) - confirms X kept its
  # full polynomial structure, unaffected by Z's independent (2-column)
  # specification.
  expect_true(all(c("beta_0", "beta_1", "beta_2") %in% names(fit$estimates)))
  expect_true(all(c("sigma_b0", "sigma_b1", "rho", "alpha") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - beta2), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

# NOTE: an earlier version of this test asserted that random_formula +
# an MCMC method should error ("not yet supported for MCMC methods").
# This was true only until the MCMC dispatch was wired up to pass through
# fit_nuts()'s already-existing optional Z_long/Z_time_surv/Z_time_quad
# overrides (see the "random_formula now works for MCMC methods too"
# test above, and its accompanying truth-recovery test) - removed once
# that guard was lifted.

test_that("random_formula now works for MCMC methods too (guard removed)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2(n = 300, seed = 91)

  # Should NOT error anymore (an earlier version of this feature rejected
  # this combination while the MCMC backend hadn't been wired up yet -
  # fit_nuts() already accepted Z_long/Z_time_surv/Z_time_quad as optional
  # overrides all along, so this ended up being a pure R-side wiring fix,
  # no Python changes needed).
  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_formula = ~ time,
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_true(all(c("sigma_b0", "sigma_b1", "alpha") %in% names(fit$estimates)))
  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
})

test_that("MCMC + random_formula correctly fixes recovery for a polynomial long_formula", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Same decisive scenario as the MLE version of this test (see
  # test-random-formula.R's weibull-PH-aGH test) - a genuinely quadratic-
  # in-time longitudinal trajectory, with random_formula = ~ time asking
  # for the random-effects structure independently and correctly
  # regardless of long_formula's own (3-column) polynomial basis. Run via
  # weibull-PH-mcmc (Bayesian NUTS) rather than the MLE path, to confirm
  # the fix works identically through fit_nuts()'s optional Z overrides.
  set.seed(101)
  n <- 400
  beta0 <- 2.0; beta1 <- 0.3; beta2 <- 0.05
  sigma_b0 <- 0.7; sigma_b1 <- 0.15; rho <- 0.2; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)

  cum_hazard <- function(t, b0_i, b1_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + b0_i) + (beta1 + b1_i) * s + beta2 * s^2
      h0 * exp(alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + beta2 * vt^2 + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + I(time^2),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_formula = ~ time,
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("beta_0", "beta_1", "beta_2") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - beta2), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("jm_fit rejects random_formula producing more than 2 random-effects columns", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      random_formula = ~ time + I(time^2)  # 3 columns: intercept, time, time^2
    ),
    "only q=1 or q=2"
  )
})
