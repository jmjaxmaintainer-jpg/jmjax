# User-supplied prior components take precedence over the empirical-Bayes
# pre-fit, as ?jm_fit documents. Before 0.4.0 the pre-fit overwrote
# control$beta_prior_mean, beta_prior_sd, sigma_e_prior_mean and
# sigma_b_prior_mean without a word.

test_that("a user-supplied beta prior is not overwritten by the empirical-Bayes pre-fit", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 41)   # beta = (2, 0.5)
  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(num_warmup = 100, num_samples = 100, num_chains = 1,
                   progress_bar = FALSE, scale_time = FALSE,
                   beta_prior_mean = c(2.4, 0.3), beta_prior_sd = c(1e-3, 1e-3)))
  # A prior this tight dominates: the estimates sit at its mean, well away
  # from the lme() pre-fit's (about 2.0 and 0.5) that used to replace it.
  expect_lt(abs(fit$estimates[["beta_0"]] - 2.4), 0.05)
  expect_lt(abs(fit$estimates[["beta_1"]] - 0.3), 0.05)
})
