# control$verbose: routine notes are opt-in; warnings are never affected.
#
# Before this, every MCMC fit printed a multi-line warm-start note from
# Python (print(), so neither suppressMessages() nor any option could stop
# it), and every fit with an off-centre covariate printed a standardization
# note. Examples and loops were dominated by output that described normal
# operation.

sim_with_offset_covariate <- function(n = 150, seed = 5) {
  sim <- simulate_joint_data(n = n, seed = seed)
  set.seed(seed)
  x_by_id <- stats::rnorm(n) + 50
  sim$data_long$x <- x_by_id[sim$data_long$id]
  sim
}

test_that("MLE fits are quiet by default and chatty with verbose = TRUE", {
  skip_if_no_backend()
  sim <- sim_with_offset_covariate()
  fit_it <- function(ctl) jm_fit(
    y ~ time + x, survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", method = "weibull-PH-aGH",
    control = c(list(standardize_covariates = TRUE), ctl))

  expect_no_message(fit_it(list()))
  expect_message(fit_it(list(verbose = TRUE)), "standardize_covariates")

  old <- options(jmjax.quiet = TRUE); on.exit(options(old), add = TRUE)
  expect_no_message(fit_it(list(verbose = TRUE)))
})

test_that("MCMC fits print no warm-start note by default", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2(n = 120, seed = 11)
  fit_it <- function(ctl) jm_fit(
    y ~ time, survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", method = "spline-PH-mcmc",
    random_effects = "intercept_slope", random_formula = ~ time,
    control = c(list(n_interior_knots = 5L, num_warmup = 100L,
                     num_samples = 100L, num_chains = 1L,
                     progress_bar = FALSE, seed = 11L), ctl))

  msgs <- character(0)
  out <- capture.output(
    fit <- withCallingHandlers(fit_it(list()), message = function(m) {
      msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage")
    }))
  expect_false(any(grepl("warm start", c(out, msgs), ignore.case = TRUE)))

  # The progress bar is off by default, so nothing of its output either.
  expect_false(any(grepl("warmup|sample:", out)))
  # Run lengths are recorded as used, not as given (were NULL by default).
  expect_identical(fit$mcmc_settings$num_warmup, 100L)
  expect_identical(fit$mcmc_settings$num_samples, 100L)

  # verbose = TRUE: one line announcing the run, and - when the backend
  # chose the conservative seed, which is recorded on the fit either way -
  # the warm-start note, both as R messages.
  vmsgs <- character(0)
  withCallingHandlers(fit_it(list(verbose = TRUE)), message = function(m) {
    vmsgs <<- c(vmsgs, conditionMessage(m)); invokeRestart("muffleMessage")
  })
  expect_true(any(grepl("sampling 1 chain\\(s\\) x 200 iterations", vmsgs)))
  ws <- fit$convergence$warm_start
  if (is.list(ws) && identical(ws$tier, "conservative")) {
    expect_true(any(grepl("CONSERVATIVE seed", vmsgs)))
  }
})
