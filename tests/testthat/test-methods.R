# coef(), vcov(), confint(), logLik(), nobs() for jmjax fits.
#
# THE CHECK THAT MATTERS: sqrt(diag(vcov(fit))) must equal fit$se. Before
# these methods existed, fit$vcov held the inverse Hessian on the
# optimizer's INTERNAL scale - log(sigma), log(shape), atanh(rho), unnamed -
# while fit$se had been converted to natural units separately. The two
# disagreed for every variance parameter and nothing noticed, because
# nothing compared them. Each fit below exercises a different conversion:
# exp() for the sigmas and shape, tanh() for rho, and the linear
# standardized-covariate map applied on the R side.

expect_vcov_matches_se <- function(fit, tol = 1e-6) {
  V <- vcov(fit)
  expect_true(is.matrix(V))
  expect_identical(rownames(V), names(coef(fit)))
  expect_identical(colnames(V), names(coef(fit)))
  expect_equal(V, t(V), tolerance = 1e-10)
  se <- unlist(fit$se)[names(coef(fit))]
  ok <- is.finite(se)
  expect_equal(sqrt(diag(V))[ok], se[ok], tolerance = tol)
}

test_that("Weibull MLE: coef, vcov, confint, logLik and nobs agree with the fit", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 150, seed = 3)
  fit <- jm_fit(y ~ time, survival::Surv(time, event) ~ 1,
                data_long = sim$data_long, data_surv = sim$data_surv,
                id_var = "id", time_var = "time", method = "weibull-PH-aGH")

  expect_identical(coef(fit), setNames(as.numeric(unlist(fit$estimates)),
                                        names(fit$estimates)))
  expect_vcov_matches_se(fit)

  ci <- confint(fit, level = 0.9)
  expect_identical(colnames(ci), c("5 %", "95 %"))
  z <- stats::qnorm(0.95)
  expect_equal(ci[, 1], coef(fit) - z * unlist(fit$se)[rownames(ci)], tolerance = 1e-10)
  expect_equal(ci[, 2], coef(fit) + z * unlist(fit$se)[rownames(ci)], tolerance = 1e-10)
  expect_identical(rownames(confint(fit, "alpha")), "alpha")
  expect_error(confint(fit, "no_such_parameter"), "unknown parameter")

  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
  expect_equal(as.numeric(ll), fit$loglik)
  expect_identical(attr(ll, "df"), length(fit$estimates))
  expect_equal(AIC(fit), -2 * fit$loglik + 2 * length(fit$estimates))
  expect_identical(nobs(fit), 150L)
  expect_equal(BIC(fit), -2 * fit$loglik + log(150) * length(fit$estimates))
})

test_that("q = 2 MLE: vcov converts rho through tanh and the sigmas through exp", {
  skip_if_no_backend()
  sim <- simulate_joint_data_re2(n = 300, seed = 7)
  fit <- jm_fit(y ~ time, survival::Surv(time, event) ~ 1,
                data_long = sim$data_long, data_surv = sim$data_surv,
                id_var = "id", time_var = "time", method = "spline-PH-aGH",
                random_effects = "intercept_slope",
                control = list(n_interior_knots = 5))
  expect_true(all(c("sigma_b0", "sigma_b1", "rho") %in% rownames(vcov(fit))))
  expect_vcov_matches_se(fit)
})

test_that("standardized covariates: vcov carries beta back to the original scale", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 150, seed = 5)
  set.seed(5)
  x_by_id <- stats::rnorm(150) + 50          # large offset: the map is far from identity
  sim$data_long$x <- x_by_id[sim$data_long$id]
  fit <- jm_fit(y ~ time + x, survival::Surv(time, event) ~ 1,
                data_long = sim$data_long, data_surv = sim$data_surv,
                id_var = "id", time_var = "time", method = "weibull-PH-aGH",
                control = list(standardize_covariates = TRUE))
  expect_vcov_matches_se(fit)
})

test_that("MCMC: vcov and confint come from the posterior draws; logLik is refused", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2(n = 120, seed = 11)
  fit <- jm_fit(y ~ time, survival::Surv(time, event) ~ 1,
                data_long = sim$data_long, data_surv = sim$data_surv,
                id_var = "id", time_var = "time", method = "spline-PH-mcmc",
                random_effects = "intercept_slope", random_formula = ~ time,
                control = list(n_interior_knots = 5L, num_warmup = 250L,
                               num_samples = 250L, num_chains = 2L,
                               progress_bar = FALSE, seed = 11L))

  # fit$se is the posterior SD with divisor n (numpy); cov() divides by
  # n - 1. With 500 draws that is a 0.1% difference.
  expect_vcov_matches_se(fit, tol = 5e-3)

  # confint() at 95% must reproduce summary()'s credible interval exactly:
  # both are quantiles of the same draws.
  s <- summary(fit)
  ci <- confint(fit)
  expect_equal(unname(ci[, 1]), s[rownames(ci), "CrI.lower"], tolerance = 1e-12)
  expect_equal(unname(ci[, 2]), s[rownames(ci), "CrI.upper"], tolerance = 1e-12)

  expect_error(logLik(fit), "not available")
  expect_identical(nobs(fit), 120L)
})

test_that("an old fit object with an internal-scale vcov is refused, not misread", {
  old <- structure(list(estimates = c(beta_0 = 1, sigma_e = 0.3),
                        se = c(beta_0 = 0.1, sigma_e = 0.02),
                        vcov = list(list(0.01, 0), list(0, 0.004)),
                        method = "weibull-PH-aGH"),
                   class = "jmjax")
  expect_error(vcov(old), "internal")
  expect_equal(coef(old), c(beta_0 = 1, sigma_e = 0.3))
})
