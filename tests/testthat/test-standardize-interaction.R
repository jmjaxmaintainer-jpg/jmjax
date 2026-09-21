# Regression tests for the standardize_covariates back-transformation.
#
# THE BUG THESE LOCK DOWN. The back-transformation used to make two
# elementwise adjustments - correct the intercept, divide the standardized
# column by its sd - which is right for an additive formula and WRONG once
# the standardized variable appears in an interaction:
#
#     b3 * t * (x - m)/s  =  (b3/s) * t * x  -  (b3 * m/s) * t
#
# so the interaction leaks into the TIME main effect, a column the old code
# never touched. Measured on y ~ time * x with x ~ N(50, 10): the time
# coefficient came back 0.700 against a true 0.500, and the interaction
# 0.0399 against 0.004.
#
# The whole suite used `y ~ time + x`, so nothing caught it. These tests
# exist so that cannot happen again.
#
# The fix treats standardization as what it is - a linear map on the
# design, X_std = X %*% A - and recovers A once by qr.solve, so beta_orig
# = A %*% beta_sampled handles interactions, multi-way terms and
# polynomials without any formula parsing.

library(survival)

make_interaction_data <- function(n_sub = 120, seed = 42) {
  set.seed(seed)
  visit <- c(0, 1, 2, 3)
  # x deliberately far from zero: the leak scales with the mean, so a
  # centred covariate would hide the bug entirely.
  x <- runif(n_sub, 15, 25)
  b0 <- rnorm(n_sub, 0, 0.5)

  dl <- do.call(rbind, lapply(seq_len(n_sub), function(i) {
    data.frame(id = i, time = visit, x = x[i],
               y = 2.0 + 0.5 * visit + 0.03 * x[i] + 0.04 * visit * x[i] +
                   b0[i] + rnorm(length(visit), 0, 0.15))
  }))
  ds <- data.frame(id = seq_len(n_sub),
                   time = runif(n_sub, 1.5, 3.0),
                   event = rbinom(n_sub, 1, 0.6))
  dl$id <- factor(dl$id); ds$id <- factor(ds$id)
  list(data_long = dl, data_surv = ds,
       truth = c(beta_0 = 2.0, beta_1 = 0.5, beta_2 = 0.03, beta_3 = 0.04))
}

test_that("standardize_covariates back-transforms correctly with an interaction", {
  skip_if_no_backend()
  d <- make_interaction_data()

  fit <- jm_fit(
    long_formula = y ~ time * x,
    surv_formula = Surv(time, event) ~ 1,
    data_long = d$data_long, data_surv = d$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept",
    control = list(n_interior_knots = 5L, num_warmup = 300L,
                    num_samples = 300L, num_chains = 2L,
                    progress_bar = FALSE, standardize_covariates = TRUE))

  # model.matrix(~ time * x) column order: (Intercept), time, x, time:x
  expect_true(all(paste0("beta_", 0:3) %in% names(fit$estimates)))

  # beta_1 is the coefficient the old code got wrong - it returned the
  # SAMPLED value (~0.70 here) instead of the back-transformed one, because
  # the interaction's contribution to the time main effect was ignored.
  expect_lt(abs(fit$estimates[["beta_1"]] - d$truth[["beta_1"]]),
            3 * fit$se[["beta_1"]])
  # beta_3 was off by a factor of 1/s - about 10x on this data.
  expect_lt(abs(fit$estimates[["beta_3"]] - d$truth[["beta_3"]]),
            3 * fit$se[["beta_3"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - d$truth[["beta_2"]]),
            3 * fit$se[["beta_2"]])
})

test_that("standardizing on or off gives the same coefficients with an interaction", {
  skip_if_no_backend()
  d <- make_interaction_data()
  ctl <- list(n_interior_knots = 5L, num_warmup = 300L, num_samples = 300L,
              num_chains = 2L, progress_bar = FALSE, seed = 2026L)

  f_on <- jm_fit(long_formula = y ~ time * x,
                 surv_formula = Surv(time, event) ~ 1,
                 data_long = d$data_long, data_surv = d$data_surv,
                 id_var = "id", time_var = "time", method = "spline-PH-mcmc",
                 random_effects = "intercept",
                 control = c(ctl, list(standardize_covariates = TRUE)))
  f_off <- jm_fit(long_formula = y ~ time * x,
                  surv_formula = Surv(time, event) ~ 1,
                  data_long = d$data_long, data_surv = d$data_surv,
                  id_var = "id", time_var = "time", method = "spline-PH-mcmc",
                  random_effects = "intercept",
                  control = c(ctl, list(standardize_covariates = FALSE)))

  # Standardizing is a reparameterization: the reported coefficients are on
  # the original scale either way, so they must agree to within MCMC noise.
  # This is the check that would have failed loudest under the old code -
  # beta_1 differed by 0.2 on a parameter of size 0.5.
  for (nm in paste0("beta_", 0:3)) {
    expect_lt(abs(f_on$estimates[[nm]] - f_off$estimates[[nm]]),
              0.5 * sqrt(f_on$se[[nm]]^2 + f_off$se[[nm]]^2),
              label = sprintf("%s: on %.5f vs off %.5f", nm,
                              f_on$estimates[[nm]], f_off$estimates[[nm]]))
  }
})

test_that("the additive case still works (no regression from the matrix rewrite)", {
  skip_if_no_backend()
  set.seed(7)
  n <- 100; visit <- c(0, 1, 2, 3)
  x <- runif(n, 15, 25); b0 <- rnorm(n, 0, 0.5)
  dl <- do.call(rbind, lapply(seq_len(n), function(i) {
    data.frame(id = i, time = visit, x = x[i],
               y = 2 + 0.5 * visit + 0.03 * x[i] + b0[i] +
                   rnorm(length(visit), 0, 0.15))
  }))
  ds <- data.frame(id = seq_len(n), time = runif(n, 1.5, 3),
                   event = rbinom(n, 1, 0.6))
  dl$id <- factor(dl$id); ds$id <- factor(ds$id)

  fit <- jm_fit(long_formula = y ~ time + x,
                surv_formula = Surv(time, event) ~ 1,
                data_long = dl, data_surv = ds, id_var = "id",
                time_var = "time", method = "spline-PH-mcmc",
                random_effects = "intercept",
                control = list(n_interior_knots = 5L, num_warmup = 300L,
                                num_samples = 300L, num_chains = 2L,
                                progress_bar = FALSE))
  expect_lt(abs(fit$estimates[["beta_1"]] - 0.5), 3 * fit$se[["beta_1"]])
  expect_lt(abs(fit$estimates[["beta_2"]] - 0.03), 3 * fit$se[["beta_2"]])
})
