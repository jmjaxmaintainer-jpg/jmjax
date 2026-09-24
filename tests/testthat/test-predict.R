# predict() - phase 1: MCMC fits, in-sample subjects, value association.
#
# What these check, and what they deliberately do not: the agreement with
# JMbayes2 (the acceptance test for shipping predict()) lives in
# dev/verify_predict_jmbayes2.R, because fitting both packages on pbc2
# takes minutes. The tests here pin down the plumbing - that predictions
# are rebuilt on the ORIGINAL scale, from the right draws, and that the
# survival integral is right - each against something computed a
# different way.

fit_pred <- function(sim, ctl = list(), method = "spline-PH-mcmc",
                     long_f = y ~ time + x, surv_f = survival::Surv(time, event) ~ w) {
  jm_fit(long_f, surv_f,
         data_long = sim$data_long, data_surv = sim$data_surv,
         id_var = "id", time_var = "time", method = method,
         random_effects = "intercept_slope", random_formula = ~ time,
         control = c(list(n_interior_knots = 5L, num_warmup = 250L,
                          num_samples = 250L, num_chains = 2L, seed = 3L), ctl))
}

sim_pred <- function(n = 150, seed = 21) {
  sim <- simulate_joint_data_re2(n = n, seed = seed)
  set.seed(seed)
  x <- stats::rnorm(n) + 50                 # far off-centre: standardization bites
  sim$data_long$x <- x[sim$data_long$id]
  sim$data_surv$w <- stats::rbinom(n, 1, 0.5)
  sim
}

test_that("longitudinal predictions track the observed data, on the original scale", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- sim_pred()
  fit <- fit_pred(sim)

  ids <- head(fit$model_info$subj_ids, 8)
  nd <- sim$data_long[sim$data_long$id %in% ids, ]
  obs_t <- sort(unique(nd$time))
  p <- predict(fit, newdata = nd, times = obs_t)
  expect_identical(names(p), c("id", "time", "estimate", "lower", "upper"))
  expect_true(all(p$lower <= p$estimate & p$estimate <= p$upper))

  # Marker means at the visit times against the observed values: with
  # sigma_e = 0.3 and subject-specific random effects, they must line up
  # closely. A scale error (standardized x, or time/c) would not.
  m <- merge(nd, p, by = c("id", "time"))
  expect_gt(stats::cor(m$y, m$estimate), 0.95)
  expect_lt(sqrt(mean((m$y - m$estimate)^2)), 3 * fit$estimates[["sigma_e"]])
})

test_that("predictions do not depend on scale_time", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- sim_pred()
  f_on  <- fit_pred(sim, list(scale_time = 5))
  f_off <- fit_pred(sim, list(scale_time = FALSE))
  expect_match(f_on$scale_time_status, "^applied")

  ids <- head(f_on$model_info$subj_ids, 5)
  nd <- sim$data_long[sim$data_long$id %in% ids, ]
  pl_on  <- predict(f_on,  nd, times = c(0, 1, 2, 3))
  pl_off <- predict(f_off, nd, times = c(0, 1, 2, 3))
  # Two independent MCMC runs of the same posterior: agreement at a few
  # posterior SDs. An unconverted slope (factor c = 5) misses by far more.
  sd_off <- (pl_off$upper - pl_off$lower) / (2 * 1.96)
  expect_true(all(abs(pl_on$estimate - pl_off$estimate) < 1.5 * sd_off + 0.02))

  pe_on  <- predict(f_on,  nd, process = "event", times = NULL)
  pe_off <- predict(f_off, nd, process = "event", times = NULL)
  expect_true(all(abs(pe_on$estimate - pe_off$estimate) < 0.05))
})

test_that("conditional survival: equals 1 at t_from, decreases, and matches integrate()", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- sim_pred()
  fit <- fit_pred(sim)
  # The simulated data have no censoring (every subject has an event), so
  # condition on survival to t = 1 rather than on each subject's
  # censoring time.
  ids <- head(fit$model_info$subj_ids, 3)
  nd <- sim$data_long[sim$data_long$id %in% ids, ]

  pe <- predict(fit, nd, process = "event", t_from = 1, return_draws = TRUE)
  expect_true(all(c("t_from", "time") %in% names(pe)))
  for (id in ids) {
    s <- pe[pe$id == id, ]
    expect_equal(s$estimate[1], 1)                    # u = t_from
    expect_true(all(diff(s$estimate) <= 1e-12))
    expect_true(all(s$estimate >= 0 & s$estimate <= 1))
  }

  # One draw, recomputed with stats::integrate() and the hazard written out
  # longhand - a different quadrature and a different code path.
  id <- ids[1]; i <- match(id, fit$model_info$subj_ids)
  ps <- fit$posterior_samples; d <- 7L
  beta <- .jmjax_site_matrix(ps, "beta")[d, ]
  bi <- .jmjax_b_draws(ps, i, 2L)[d, ]
  W <- .jmjax_site_matrix(ps, "W")[d, ]
  alpha <- .jmjax_site_matrix(ps, "alpha")[d, 1]
  gamma <- .jmjax_site_matrix(ps, "gamma")[d, 1]
  xi <- sim$data_long$x[sim$data_long$id == id][1]
  wi <- sim$data_surv$w[sim$data_surv$id == id]
  haz <- function(t) {
    m <- beta[1] + beta[2] * t + beta[3] * xi + bi[1] + bi[2] * t
    exp(as.numeric(fit$spline_info$basis(t) %*% W) + gamma * wi + alpha * m)
  }
  s <- pe[pe$id == id, ]
  u <- s$time[10]; t0 <- s$t_from[1]
  ref <- exp(-stats::integrate(haz, t0, u, rel.tol = 1e-10)$value)
  expect_equal(attr(pe, "draws")[[as.character(id)]][d, 10], ref, tolerance = 1e-8)
})

test_that("Weibull MCMC: event predictions are valid survival curves", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- sim_pred(n = 120, seed = 4)
  fit <- fit_pred(sim, method = "weibull-PH-mcmc")
  pe <- predict(fit, process = "event")
  expect_setequal(unique(pe$id), fit$model_info$subj_ids)
  expect_true(all(pe$estimate >= 0 & pe$estimate <= 1))
  expect_true(all(tapply(pe$estimate, pe$id, function(v) all(diff(v) <= 1e-12))))
})

test_that("out-of-scope requests stop with a clear message", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 80, seed = 2)
  mle <- jm_fit(y ~ time, survival::Surv(time, event) ~ 1,
                data_long = sim$data_long, data_surv = sim$data_surv,
                id_var = "id", time_var = "time", method = "weibull-PH-aGH")
  expect_error(predict(mle), "MCMC methods only")

  fake <- structure(list(posterior_samples = list(beta = list(1)), model_info = list(
    assoc_types = c("value", "delta"))), class = "jmjax")
  expect_error(predict(fake), "'value' association")
  old <- structure(list(posterior_samples = list(beta = list(1))), class = "jmjax")
  expect_error(predict(old), "before jmjax 0.3.0")
  ok <- structure(list(posterior_samples = list(beta = list(1)),
                       model_info = list(assoc_types = "value", id_var = "id",
                                         subj_ids = 1:3),
                       data = list(long = data.frame(id = 1:3))), class = "jmjax")
  expect_error(predict(ok, newdata = data.frame(id = integer(0))), "no subjects selected")
})
