# ==============================================================================
# Coverage for spline_prior = "penalized".
#
# WHY THIS FILE EXISTS. Every benchmark and diagnostic script in this project
# sets spline_prior = "penalized", and until now the test suite set it ZERO
# times. That gap let a total breakage of the option sit undetected behind a
# fully green suite: 410 tests passed while every penalized fit died at
# initialisation with
#
#   TypeError: body_fun output and input must have identical types
#   ... ShapedArray(int32[], weak_type=True) vs ShapedArray(float0[])
#
# The cause was numpyro's scan carry (jax-ml/jax#22045, jax >= 0.4.30), not
# anything in jmjax - but a suite that never exercises the configuration its
# own performance claims rest on cannot report that, and the failure was only
# found by a benchmark run failing thirty times in a row.
#
# So these tests are less about the RW2 algebra than about the path being
# executed at all, on whatever stack the package is installed against.
# ==============================================================================

test_that("spline_prior = 'penalized' runs and reports sane estimates", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data(n = 150, seed = 99)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    random_effects = "intercept",
    control = list(spline_prior = "penalized", n_interior_knots = 5L,
                   num_warmup = 300, num_samples = 500, num_chains = 1,
                   progress_bar = FALSE)
  )

  pop <- names(fit$estimates)
  expect_true(all(c("beta_0", "beta_1", "sigma_e", "sigma_b", "alpha") %in% pop))
  expect_true(any(grepl("^W[0-9]+$", pop)))

  # tau_w is the smoothing precision and exists ONLY under this prior - its
  # presence is what distinguishes a penalized fit from an independent one
  # that silently fell back.
  expect_true("tau_w" %in% pop)

  rh <- unlist(fit$diagnostics$rhat)
  expect_true(all(is.finite(rh)))
  expect_true(all(rh > 0.8))
})

test_that("the penalized prior's dense mass block runs", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # dense_mass_spline defaults to TRUE for this prior, so the block above
  # already covers it. This pins the OTHER branch: an earlier default flip
  # broke every weibull and independent-spline fit with KeyError: 'W01'
  # because the requested sites did not exist, so both settings need to run.
  sim <- simulate_joint_data(n = 150, seed = 99)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept",
    control = list(spline_prior = "penalized", dense_mass_spline = FALSE,
                   num_warmup = 200, num_samples = 300, num_chains = 1,
                   progress_bar = FALSE)
  )
  expect_true("tau_w" %in% names(fit$estimates))
})

test_that("the two RW2 implementations agree when both can run", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # SELF-ACTIVATING. rw2_implementation = "scan" cannot run on numpyro
  # < 0.17.0 with jax >= 0.4.30 (jax-ml/jax#22045), which is what this
  # package currently pins, so on that stack this test skips with the reason
  # rather than failing. On a stack where numpyro's device_put fix is
  # present, it turns itself on and checks that the default "vectorized"
  # branch - a closed-form solution of the same linear recursion - actually
  # agrees with the sequential one it replaced.
  sim <- simulate_joint_data(n = 150, seed = 7)
  args <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept")

  ctrl <- function(impl) list(spline_prior = "penalized", rw2_implementation = impl,
                              n_interior_knots = 5L, num_warmup = 400,
                              num_samples = 600, num_chains = 1, seed = 1L,
                              progress_bar = FALSE)

  f_scan <- tryCatch(do.call(jm_fit, c(args, list(control = ctrl("scan")))),
                     error = function(e) e)
  if (inherits(f_scan, "error")) {
    skip(paste0("rw2_implementation='scan' cannot run on this stack ",
                "(numpyro < 0.17.0 with jax >= 0.4.30, jax-ml/jax#22045): ",
                substr(conditionMessage(f_scan), 1, 120)))
  }

  f_vec <- do.call(jm_fit, c(args, list(control = ctrl("vectorized"))))

  # Same model, same seed, different arithmetic route - so these agree to
  # MCMC noise, not to machine precision. Compared on the spline block, which
  # is what the two implementations actually construct differently.
  w <- grep("^W[0-9]+$", names(f_scan$estimates), value = TRUE)
  expect_gt(length(w), 0)
  a <- unlist(f_scan$estimates[w]); b <- unlist(f_vec$estimates[w])
  expect_equal(length(a), length(b))
  expect_lt(max(abs(a - b)) / max(1, max(abs(a))), 0.25)

  expect_lt(abs(f_scan$estimates[["alpha"]] - f_vec$estimates[["alpha"]]), 0.15)
})

# ==============================================================================
# Warm start from the lme() pre-fit (control$mcmc_warm_start, default TRUE).
#
# The first version of this feature shipped WITHOUT beta, on the reasoning
# that standardize_covariates would put lme()'s coefficients on the wrong
# scale. That was backwards - standardization rewrites data_long in place
# and leaves the formula - and the omission made the start 267,000
# log-density units WORSE than random, because a tight sigma_e was supplied
# alongside a uniform-random beta. The backend's self-check caught it.
# These tests pin down that the check exists, that it passes, and that a
# deliberately corrupted start is still rejected.
# ==============================================================================

test_that("the warm start is built, accepted, and better than a cold start", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 150, seed = 3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time,
    control = list(spline_prior = "penalized", mcmc_warm_start = TRUE,
                   num_warmup = 200, num_samples = 200, num_chains = 2,
                   progress_bar = FALSE)
  )

  ws <- fit$convergence$warm_start
  expect_false(is.null(ws))                      # it ran at all
  expect_true(isTRUE(as.logical(ws$used)))       # the self-check accepted it

  # b_std is the site that matters - the population parameters were never
  # the bottleneck - so its absence would make the feature pointless even
  # while the flag reported success.
  expect_true("b_std" %in% unlist(ws$sites))
  expect_true("beta" %in% unlist(ws$sites))

  # Lower potential energy is higher log-density. The margin is large when
  # the transform is right; this only asserts the direction.
  expect_lt(as.numeric(ws$potential_warm), as.numeric(ws$potential_uniform))
})

test_that("a warm start on the wrong scale is rejected rather than used", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # The guard that makes the feature safe to default on. Supplying a tight
  # sigma_e with a wildly wrong beta reproduces exactly the failure mode
  # that shipped in the first version: plausible-looking values that are
  # collectively far worse than random.
  sim <- simulate_joint_data_re2(n = 150, seed = 3)

  expect_warning(
    fit <- jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept_slope",
      random_formula = ~ time,
      control = list(spline_prior = "penalized",
                     init_values = list(beta = c(500, 500), sigma_e = 0.01),
                     num_warmup = 100, num_samples = 100, num_chains = 1,
                     progress_bar = FALSE)
    ),
    regexp = "warm start REJECTED"
  )

  expect_false(isTRUE(as.logical(fit$convergence$warm_start$used)))
  # Rejected, not fatal: the fit still completes from the default start.
  expect_true(all(is.finite(unlist(fit$diagnostics$rhat))))
})
