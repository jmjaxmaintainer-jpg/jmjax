# Tests for jm_fit_prefit() - the pre-fitted lme/coxph object interface.
#
# Two things matter here and they are tested separately:
#   1. EXTRACTION: does it recover the same model the formula interface
#      would have built? The check is that both routes give matching
#      estimates on the same data, since a mis-derived random_formula or
#      id_var would change the model rather than error.
#   2. REJECTION: does it refuse structures jmjax cannot represent? These
#      are correctness guards - silently ignoring a user's corAR1() would
#      fit a different model than they asked for.

library(nlme)
library(survival)

make_prefit_data <- function(n = 60, seed = 7) {
  set.seed(seed)
  b0 <- rnorm(n, 0, 0.8)
  b1 <- rnorm(n, 0, 0.2)
  visit <- c(0, 1, 2, 3)
  dl <- do.call(rbind, lapply(seq_len(n), function(i) {
    data.frame(id = i, time = visit,
               y = 2 + b0[i] + (0.5 + b1[i]) * visit + rnorm(length(visit), 0, 0.3))
  }))
  ds <- data.frame(id = seq_len(n),
                   time = runif(n, 1, 3),
                   event = rbinom(n, 1, 0.6),
                   trt = rbinom(n, 1, 0.5))
  dl$id <- factor(dl$id); ds$id <- factor(ds$id)
  list(data_long = dl, data_surv = ds)
}

test_that("jm_fit_prefit derives the same model as the formula interface", {
  skip_if_no_backend()
  d <- make_prefit_data()

  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", msMaxIter = 200,
                                        niterEM = 100, returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)

  ctl <- list(n_interior_knots = 5L, num_warmup = 200L, num_samples = 200L,
              num_chains = 1L, progress_bar = FALSE, seed = 2026L,
              # zero out the coxph-derived gamma prior so the two routes are
              # comparing the SAME model - jm_fit() has no coxph to read from,
              # so leaving it in would be a genuine (intended) difference.
              gamma_prior_mean = NULL,
              # Same reasoning, for the same reason, on the warm start.
              #
              # This test asks whether the two routes build the same MODEL,
              # and it can only ask that sharply if everything else is held
              # identical - with a shared seed and identical initialisation
              # the two fits are bit-identical, so a 1e-4 tolerance detects
              # a wrong random_formula or id_var immediately.
              #
              # mcmc_warm_start breaks that, legitimately: jm_fit_prefit
              # warm-starts from the lme object the USER handed it, while
              # jm_fit has none and fits its own (with msMaxEval = 500,
              # which the fit above does not set). Two different lme fits
              # give two different starting points, so the chains diverge
              # and alpha lands at 0.1306 against 0.1340 - statistically
              # the same answer, 33x the tolerance this test needs.
              #
              # Turning it off removes a confound rather than weakening the
              # assertion. Loosening the tolerance to MCMC noise instead
              # would have cost the test the ability to catch the
              # specification bugs it exists for.
              mcmc_warm_start = FALSE)

  fit_obj <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$data_surv,
                            time_var = "time", method = "spline-PH-mcmc",
                            control = ctl)

  fit_fml <- jm_fit(
    long_formula = y ~ time,
    surv_formula = Surv(time, event) ~ trt,
    data_long = d$data_long, data_surv = d$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time, control = ctl)

  # Same model, same MCMC seed -> the extraction must have produced an
  # identical specification. Any difference in random_formula, id_var or
  # random_effects would move these materially.
  #
  # gamma_prior_mean = NULL above is honoured as an explicit opt-out
  # (jm_fit_prefit tests membership, not nullity), so both routes use the
  # same zero-centred gamma prior here. Without that the prefit route
  # would legitimately differ, since it reads coef(coxph) and the formula
  # interface has no coxph to read - a real feature that would otherwise
  # look like an extraction bug.
  for (pp in c("beta_0", "beta_1", "alpha", "sigma_e")) {
    expect_equal(fit_obj$estimates[[pp]], fit_fml$estimates[[pp]],
                 tolerance = 1e-4,
                 info = sprintf(
                   "%s: prefit %.6f vs formula %.6f. A difference here means the two routes built DIFFERENT models - check random_formula, id_var, random_effects and whether a prior differs between them.",
                   pp, fit_obj$estimates[[pp]], fit_fml$estimates[[pp]]))
  }
})

test_that("jm_fit_prefit uses coef(coxph) to centre the gamma prior", {
  skip_if_no_backend()
  d <- make_prefit_data()

  # Give the treatment a real, large effect so coef(coxph) is far from
  # zero and the prior's influence is unambiguous.
  set.seed(11)
  d$data_surv$event <- rbinom(nrow(d$data_surv), 1,
                               plogis(-0.5 + 2.5 * d$data_surv$trt))

  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)
  expect_gt(abs(unname(coef(cox_fit))), 0.2)   # precondition for the test

  ctl <- list(n_interior_knots = 5L, num_warmup = 200L, num_samples = 200L,
              num_chains = 1L, progress_bar = FALSE, seed = 2026L)

  with_prior <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$data_surv,
                               time_var = "time", method = "spline-PH-mcmc",
                               control = ctl)
  # explicit NULL must be honoured as an opt-out
  without_prior <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$data_surv,
                                  time_var = "time", method = "spline-PH-mcmc",
                                  control = c(ctl, list(gamma_prior_mean = NULL)))

  # Same data, same seed - so any difference is the gamma prior, which is
  # the point. If these are identical the coxph coefficients are not
  # reaching the model at all.
  expect_false(isTRUE(all.equal(with_prior$estimates[["gamma_0"]],
                                 without_prior$estimates[["gamma_0"]],
                                 tolerance = 1e-8)),
               info = "gamma is identical with and without the coxph-derived prior mean - the prior is not reaching the backend")
})

test_that("jm_fit_prefit handles a random-intercept-only lme (q=1)", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_fit <- lme(y ~ time, random = ~ 1 | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)

  fit <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$data_surv,
                        time_var = "time", method = "spline-PH-mcmc",
                        control = list(n_interior_knots = 5L, num_warmup = 200L,
                                        num_samples = 200L, num_chains = 1L,
                                        progress_bar = FALSE))
  # q=1 must map to random_effects = "intercept", which reports a single
  # sigma_b rather than sigma_b0/sigma_b1/rho.
  expect_true("sigma_b" %in% names(fit$estimates))
  expect_false("rho" %in% names(fit$estimates))
})

test_that("jm_fit_prefit rejects an lme correlation structure", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_cor <- tryCatch(
    lme(y ~ time, random = ~ time | id, data = d$data_long,
        correlation = corAR1(form = ~ time | id),
        control = lmeControl(opt = "optim", returnObject = TRUE)),
    error = function(e) NULL)
  skip_if(is.null(lme_cor), "corAR1 lme did not converge on this test data")

  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)
  expect_error(
    jm_fit_prefit(lme_cor, cox_fit, data_surv = d$data_surv, time_var = "time"),
    "correlation structure")
})

test_that("jm_fit_prefit rejects a coxph strata() term", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_str <- coxph(Surv(time, event) ~ strata(trt), data = d$data_surv)
  expect_error(
    jm_fit_prefit(lme_fit, cox_str, data_surv = d$data_surv, time_var = "time"),
    "strata")
})

test_that("jm_fit_prefit requires data_surv and validates its id column", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)

  # data_surv is OPTIONAL: jmjax recovers it by evaluating the data symbol
  # from cox_object's call, as JMbayes2::jm() does. Here d$data_surv is in
  # scope, so recovery should succeed and the fit should run.
  expect_error(
    jm_fit_prefit(lme_fit, cox_fit, time_var = "time",
                  control = list(n_interior_knots = 5L, num_warmup = 50L,
                                  num_samples = 50L, num_chains = 1L,
                                  progress_bar = FALSE)),
    NA)

  # But recovery is convenience, not a guarantee - eval() looks the symbol
  # up in the calling frame rather than retrieving a stored copy. Fitting
  # inside a local scope that then exits makes it unrecoverable, and the
  # error must say so rather than failing obscurely.
  gone_cox <- local({
    tmp_surv <- d$data_surv
    coxph(Surv(time, event) ~ trt, data = tmp_surv)
  })
  expect_error(
    jm_fit_prefit(lme_fit, gone_cox, time_var = "time"),
    "could not recover the data")

  # the grouping variable must be present, or subjects cannot be aligned
  ds_noid <- d$data_surv; ds_noid$id <- NULL
  expect_error(
    jm_fit_prefit(lme_fit, cox_fit, data_surv = ds_noid, time_var = "time"),
    "not a\\s+column of data_surv|grouping variable")
})

test_that("jm_fit_prefit rejects non-lme / non-coxph inputs", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)

  expect_error(
    jm_fit_prefit(lm(y ~ time, data = d$data_long), cox_fit,
                  data_surv = d$data_surv, time_var = "time"),
    "must be a fitted nlme::lme")
  expect_error(
    jm_fit_prefit(lme_fit, lm(time ~ trt, data = d$data_surv),
                  data_surv = d$data_surv, time_var = "time"),
    "must be a fitted survival::coxph")
})

test_that("jm_fit_prefit validates time_var against the lme data", {
  skip_if_no_backend()
  d <- make_prefit_data()
  lme_fit <- lme(y ~ time, random = ~ time | id, data = d$data_long,
                  control = lmeControl(opt = "optim", returnObject = TRUE))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = d$data_surv)
  expect_error(
    jm_fit_prefit(lme_fit, cox_fit, data_surv = d$data_surv,
                  time_var = "not_a_column"),
    "is not a column of the data attached to lme_object")
})
