test_that("build_baseline_covariates drops the intercept column", {
  data_surv <- data.frame(id = 1:5, age = c(50, 60, 45, 70, 55),
                           sex = factor(c("f", "m", "f", "m", "f")))

  W <- build_baseline_covariates(survival::Surv(rep(1, 5), rep(1, 5)) ~ age + sex, data_surv)

  expect_false("(Intercept)" %in% colnames(W))
  expect_equal(colnames(W), c("age", "sexm"))
  expect_equal(unname(W[, "age"]), data_surv$age)
})

test_that("build_baseline_covariates returns 0 columns for ~1 (no covariates)", {
  data_surv <- data.frame(id = 1:5)
  W <- build_baseline_covariates(survival::Surv(rep(1, 5), rep(1, 5)) ~ 1, data_surv)
  expect_equal(ncol(W), 0)
  expect_equal(nrow(W), 5)
})

test_that("jm_fit rejects baseline covariates outside the supported configuration", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)
  sim$data_surv$age <- rnorm(50)

  # NOTE: q=2 for SURVIVAL-side baseline covariates is now correctly
  # SUPPORTED (gamma'W_i is q-independent) - see
  # "weibull-PH-aGH recovers a true baseline covariate effect at q=2"
  # below for its actual truth-recovery test. An earlier version of this
  # test asserted an error here; removed once that scope limit was lifted.

  # NOTE: spline-PH-aGH at q=1 AND q=2, value-only IS now supported - see
  # "spline-PH-aGH with a baseline covariate recovers the true gamma" and
  # "spline-PH-aGH recovers a true baseline covariate effect at q=2"
  # below. Two earlier versions of this test asserted an error for each of
  # those; both removed once that scope was implemented. The only
  # remaining restriction for spline-PH-aGH is combination with
  # functional_forms channels (delta/area/area_avg), tested via the
  # weibull-PH-aGH case below (the identical restriction applies to both
  # MLE methods, so testing it once is sufficient - see
  # "weibull-PH-aGH still rejects baseline covariates combined with
  # functional_forms" further down for the direct method='weibull-PH-aGH'
  # version of this same check).

  # combined with functional_forms: not yet supported
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ age,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      functional_forms = ~ value(y) + delta(y)
    ),
    "Baseline covariates"
  )
})

test_that("spline-PH-aGH with a baseline covariate recovers the true gamma", {
  skip_if_no_backend()

  # Mirrors "weibull-PH-aGH with a baseline covariate recovers the true
  # gamma" above exactly, swapping the Weibull baseline hazard for a
  # spline one - the extension this test file is validating.
  set.seed(202)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; sigma_b <- 0.8; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_age <- 0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b <- rnorm(n, 0, sigma_b)
  age <- rnorm(n)  # baseline covariate, standardized for a clean test

  # Truth is still generated from a Weibull hazard - spline-PH-aGH's
  # flexible basis should still recover gamma/alpha/beta correctly when
  # fit to data from this simpler special case, the same logic used
  # throughout this codebase's spline-vs-Weibull cross-checks.
  cum_hazard <- function(t, b_i, age_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- beta0 + b_i + beta1 * s
      h0 * exp(gamma_age * age_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b[i], age[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b[i], age[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, age = age)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- beta0 + b[i] + beta1 * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ age,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    control = list(n_interior_knots = 5)
  )

  expect_true(fit$convergence$converged)
  expect_true("gamma_0" %in% names(fit$estimates))
  expect_true("alpha" %in% names(fit$estimates))

  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_age), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
})

test_that("spline-PH-aGH recovers a true baseline covariate effect at q=2 (survival-side)", {
  skip_if_no_backend()

  # Mirrors "weibull-PH-aGH recovers a true baseline covariate effect at
  # q=2" below, swapping the Weibull baseline hazard for a spline one -
  # the q=1 -> q=2 extension of the spline baseline-covariates capability.
  set.seed(203)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5
  sigma_b0 <- 0.8; sigma_b1 <- 0.2; rho <- 0.3; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_drug <- -0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)
  drug <- rbinom(n, 1, 0.5)

  cum_hazard <- function(t, b0_i, b1_i, drug_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + b0_i) + (beta1 + b1_i) * s
      h0 * exp(gamma_drug * drug_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i], drug[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i], drug[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, drug = drug)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ drug,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    random_effects = "intercept_slope",
    control = list(n_interior_knots = 5)
  )

  expect_true(fit$convergence$converged)
  expect_true("gamma_0" %in% names(fit$estimates))
  expect_true(all(c("sigma_b0", "sigma_b1", "rho") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_drug), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-aGH with a baseline covariate recovers the true gamma", {
  skip_if_no_backend()

  set.seed(201)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; sigma_b <- 0.8; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_age <- 0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b <- rnorm(n, 0, sigma_b)
  age <- rnorm(n)  # baseline covariate, standardized for a clean test

  cum_hazard <- function(t, b_i, age_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- beta0 + b_i + beta1 * s
      h0 * exp(gamma_age * age_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b[i], age[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b[i], age[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, age = age)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- beta0 + b[i] + beta1 * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ age,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  expect_true(fit$convergence$converged)
  expect_true("gamma_0" %in% names(fit$estimates))
  expect_true("alpha" %in% names(fit$estimates))

  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_age), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
})

test_that("baseline covariates (surv_formula, Piece 1) work correctly with the default value-only association", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 200, seed = 42)
  sim$data_surv$age <- rnorm(200)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ age,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  expect_true(fit$convergence$converged)
  expect_true("gamma_0" %in% names(fit$estimates))
})

test_that("extract_baseline_covariates_long correctly extracts a time-constant covariate", {
  data_long <- data.frame(
    id = rep(c(3, 1, 2), each = 2),  # deliberately out-of-order ids
    time = c(0, 1, 0, 1, 0, 1),
    age = c(70, 70, 50, 50, 60, 60)  # constant within each subject
  )
  subj_ids <- c(1, 2, 3)  # sorted order, as build_long_arrays() would produce

  result <- extract_baseline_covariates_long(y ~ time + age, "time", data_long, "id", subj_ids)

  expect_equal(nrow(result), 3)
  expect_equal(result$age, c(50, 60, 70))  # matches subj_ids order (1,2,3), not data_long's row order
})

test_that("extract_baseline_covariates_long returns NULL when long_formula has no extra covariates", {
  data_long <- data.frame(id = rep(1:3, each = 2), time = c(0, 1, 0, 1, 0, 1))
  result <- extract_baseline_covariates_long(y ~ time, "time", data_long, "id", 1:3)
  expect_null(result)
})

test_that("extract_baseline_covariates_long errors clearly on a genuinely time-varying covariate", {
  data_long <- data.frame(
    id = rep(1:2, each = 2),
    time = c(0, 1, 0, 1),
    weight = c(70, 72, 60, 60)  # subject 1's weight CHANGES - not baseline
  )
  expect_error(
    extract_baseline_covariates_long(y ~ time + weight, "time", data_long, "id", 1:2),
    "varies WITHIN subject"
  )
})

test_that("build_time_design correctly carries forward baseline covariates at a synthetic time point", {
  baseline_cov <- data.frame(age = c(50, 60, 70))
  T_surv <- c(2, 3, 4)

  X <- build_time_design(y ~ time + age, "time", T_surv, baseline_cov)

  expect_equal(colnames(X), c("(Intercept)", "time", "age"))
  expect_equal(unname(X[, "time"]), T_surv)
  expect_equal(unname(X[, "age"]), baseline_cov$age)  # carried forward correctly
})

test_that("build_time_design's baseline_covariates work correctly for a quadrature-grid (matrix) t_values too", {
  baseline_cov <- data.frame(age = c(50, 60))
  t_quad <- matrix(c(0.5, 1.0, 1.5, 2.0), nrow = 2)  # [2 subjects, 2 quad nodes]

  X <- build_time_design(y ~ time + age, "time", t_quad, baseline_cov)

  expect_equal(dim(X), c(2, 2, 3))
  # age should be identical across BOTH quadrature nodes for each subject
  # (it's baseline/time-constant, not evaluated differently per node).
  expect_equal(X[, 1, "age"], baseline_cov$age)
  expect_equal(X[, 2, "age"], baseline_cov$age)
})

test_that("jm_fit rejects longitudinal baseline covariates outside the supported configuration", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)
  age_by_id <- setNames(rnorm(50), unique(sim$data_long$id))
  sim$data_long$age <- age_by_id[as.character(sim$data_long$id)]

  # q=2: not yet supported
  expect_error(
    jm_fit(
      long_formula = y ~ time + age,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      random_effects = "intercept_slope"
    ),
    "Baseline covariates"
  )

  # combined with functional_forms: not yet supported
  expect_error(
    jm_fit(
      long_formula = y ~ time + age,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      functional_forms = ~ value(y) + delta(y)
    ),
    "Baseline covariates"
  )
})

test_that("weibull-PH-aGH recovers a true baseline covariate effect in long_formula", {
  skip_if_no_backend()

  set.seed(301)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; beta_age <- 0.3  # age shifts the intercept
  sigma_b <- 0.8; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b <- rnorm(n, 0, sigma_b)
  age <- rnorm(n)  # baseline covariate

  cum_hazard <- function(t, b_i, age_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- beta0 + beta_age * age_i + b_i + beta1 * s
      h0 * exp(alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b[i], age[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b[i], age[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- beta0 + beta_age * age[i] + b[i] + beta1 * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y, age = age[i])
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + age,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("beta_0", "beta_1", "beta_2") %in% names(fit$estimates)))

  # model.matrix(~time+age) column order: (Intercept), time, age
  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - beta_age), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
})


test_that("spline-PH-aGH recovers a true baseline covariate effect in long_formula", {
  skip_if_no_backend()

  # Mirrors the weibull-PH-aGH test above, swapping the baseline hazard for
  # a spline one. This combination was previously rejected by an R-side
  # guard; the Python backend always supported it, since a longitudinal
  # covariate adds no new parameters - it only makes beta longer, and the
  # spline MLE's theta layout ("beta": (0, p)) and likelihood
  # (X_long_i @ beta) were already generic in p.
  set.seed(311)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; beta_age <- 0.3
  sigma_b <- 0.8; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b <- rnorm(n, 0, sigma_b)
  age <- rnorm(n)

  cum_hazard <- function(t, b_i, age_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- beta0 + beta_age * age_i + b_i + beta1 * s
      h0 * exp(alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b[i], age[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b[i], age[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- beta0 + beta_age * age[i] + b[i] + beta1 * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y, age = age[i])
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + age,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    control = list(n_interior_knots = 5L)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("beta_0", "beta_1", "beta_2") %in% names(fit$estimates)))

  # model.matrix(~time+age) column order: (Intercept), time, age
  expect_lt(abs(fit$estimates[["beta_0"]] - beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - beta_age), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
})


test_that("spline-PH-aGH handles a long_formula covariate at q=2", {
  skip_if_no_backend()

  # q=2 needs an explicit random_formula whenever a longitudinal covariate
  # is present - the legacy default Z = X[, 1:2] slicing would otherwise
  # depend on column order. That guard is method-agnostic and is tested
  # separately; here it is satisfied, and the point is that the spline q=2
  # path handles p=3 correctly.
  set.seed(312)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; beta_age <- 0.3
  sigma_b0 <- 0.8; sigma_b1 <- 0.2; rho <- 0.3; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)
  age <- rnorm(n)

  cum_hazard <- function(t, b0_i, b1_i, age_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + beta_age * age_i + b0_i) + (beta1 + b1_i) * s
      h0 * exp(alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i], age[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i], age[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + beta_age * age[i] + b0[i]) + (beta1 + b1[i]) * vt +
      stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y, age = age[i])
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + age,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-aGH",
    random_effects = "intercept_slope",
    random_formula = ~ time,
    control = list(n_interior_knots = 5L)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("beta_0", "beta_1", "beta_2", "sigma_b0", "sigma_b1", "rho")
                  %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["beta_1"]] - beta1), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - beta_age), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
})


test_that("jm_fit rejects longitudinal baseline covariates at q=2 without an explicit random_formula", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)
  age_by_id <- setNames(rnorm(50), unique(sim$data_long$id))
  sim$data_long$age <- age_by_id[as.character(sim$data_long$id)]

  expect_error(
    jm_fit(
      long_formula = y ~ time + age,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      random_effects = "intercept_slope"  # no random_formula given
    ),
    "require random_formula"
  )
})

test_that("weibull-PH-aGH recovers a true baseline covariate effect at q=2 (survival-side)", {
  skip_if_no_backend()

  set.seed(401)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5
  sigma_b0 <- 0.8; sigma_b1 <- 0.2; rho <- 0.3; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_drug <- -0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)
  drug <- rbinom(n, 1, 0.5)

  cum_hazard <- function(t, b0_i, b1_i, drug_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + b0_i) + (beta1 + b1_i) * s
      h0 * exp(gamma_drug * drug_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i], drug[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i], drug[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, drug = drug)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ drug,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope"
  )

  expect_true(fit$convergence$converged)
  expect_true("gamma_0" %in% names(fit$estimates))
  expect_true(all(c("sigma_b0", "sigma_b1", "rho") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_drug), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-aGH recovers baseline covariates on BOTH sides simultaneously at q=2", {
  skip_if_no_backend()

  # Mirrors the PBC2 real-data script: age in long_formula, drug in
  # surv_formula, q=2 with random_formula given explicitly.
  set.seed(402)
  n <- 500
  beta0 <- 2.0; beta1 <- 0.5; beta_age <- 0.3
  sigma_b0 <- 0.8; sigma_b1 <- 0.2; rho <- 0.3; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_drug <- -0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)
  age <- rnorm(n)
  drug <- rbinom(n, 1, 0.5)

  cum_hazard <- function(t, b0_i, b1_i, age_i, drug_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + beta_age * age_i + b0_i) + (beta1 + b1_i) * s
      h0 * exp(gamma_drug * drug_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i], age[i], drug[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i], age[i], drug[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, drug = drug)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + beta_age * age[i]) + b0[i] + (beta1 + b1[i]) * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y, age = age[i])
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + age,
    surv_formula = survival::Surv(time, event) ~ drug,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope",
    random_formula = ~ time
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("beta_0", "beta_1", "beta_2", "gamma_0", "alpha") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["beta_2"]] - beta_age), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_drug), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-mcmc recovers a true baseline covariate effect on both sides simultaneously at q=2 (mirrors real-data usage)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Mirrors the exact real-data script that surfaced this gap: age in
  # long_formula, drug in surv_formula, q=2 with random_formula given
  # explicitly.
  set.seed(501)
  n <- 400
  beta0 <- 2.0; beta1 <- 0.5; beta_age <- 0.3
  sigma_b0 <- 0.8; sigma_b1 <- 0.2; rho <- 0.3; sigma_e <- 0.3
  weibull_shape <- 1.2; log_lambda0 <- -2.0; alpha <- 0.5; gamma_drug <- -0.4
  max_time <- 5; visit_times <- seq(0, 5, by = 0.5)

  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)
  age <- rnorm(n)
  drug <- rbinom(n, 1, 0.5)

  cum_hazard <- function(t, b0_i, b1_i, age_i, drug_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      value_s <- (beta0 + beta_age * age_i + b0_i) + (beta1 + b1_i) * s
      h0 * exp(gamma_drug * drug_i + alpha * value_s)
    }
    stats::integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(stats::runif(1)); ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i], age[i], drug[i]) < target) return(Inf)
    stats::uniroot(function(t) cum_hazard(t, b0[i], b1[i], age[i], drug[i]) - target,
                    lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event, drug = drug)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + beta_age * age[i]) + b0[i] + (beta1 + b1[i]) * vt + stats::rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y, age = age[i])
  })
  data_long <- do.call(rbind, long_rows)

  fit <- jm_fit(
    long_formula = y ~ time + age,
    surv_formula = survival::Surv(time, event) ~ drug,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_effects = "intercept_slope",
    random_formula = ~ time,
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("beta_0", "beta_1", "beta_2", "gamma_0", "alpha") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["beta_2"]] - beta_age), 3 * fit$se[["beta_2"]])
  expect_lt(abs(fit$estimates[["gamma_0"]] - gamma_drug), 3 * fit$se[["gamma_0"]])
  expect_lt(abs(fit$estimates[["alpha"]] - alpha), 3 * fit$se[["alpha"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-aGH still rejects baseline covariates combined with functional_forms (MLE never built that combination)", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)
  sim$data_surv$drug <- rbinom(50, 1, 0.5)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ drug,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      functional_forms = ~ value(y) + delta(y)
    ),
    "not yet supported for method = 'weibull-PH-aGH'"
  )
})

test_that("weibull-PH-mcmc supports baseline survival covariates combined with delta (MCMC's generic model handles this)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_delta(n = 400, seed = 21, alpha_value = 0.5, alpha_delta = 0.3)
  drug_by_id <- setNames(rbinom(length(unique(sim$data_surv$id)), 1, 0.5), unique(sim$data_surv$id))
  sim$data_surv$drug <- drug_by_id[as.character(sim$data_surv$id)]

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ drug,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    functional_forms = ~ value(y) + delta(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_delta", "gamma_0") %in% names(fit$estimates)))
})
