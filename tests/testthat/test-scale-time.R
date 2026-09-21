# Tests for control$scale_time.
#
# WHAT IT DOES. Substitutes I(time/c) into long_formula and random_formula
# before the design matrices are built, so every time-derived column in X
# and Z is scaled. Coefficients come back on the ORIGINAL scale through
# the same qr.solve projection that handles standardize_covariates.
#
# WHY THESE TESTS EXIST. The feature took three attempts, and each failure
# was invisible except through the ratio of a parameter with scaling on
# versus off:
#   ratio = c      the sigma_b correction sat in the MCMC-only branch and
#                  never ran for the aGH methods
#   ratio = c^2    it ran in the wrong DIRECTION (diag(A_z) is 1/c, so
#                  dividing by it multiplies by c)
#   iter 2, and    the first version divided data_long[[time_var]] instead
#   survival       of rewriting the formula - time_var is ALSO the slot
#   params moved   build_time_design() fills with the SURVIVAL times
#
# The first test below would have caught all three.

library(survival)

make_data <- function(n = 120, seed = 9) {
  set.seed(seed)
  visit <- c(0, 1, 2, 3)
  b0 <- rnorm(n, 0, 0.8); b1 <- rnorm(n, 0, 0.2); x <- rnorm(n)
  dl <- do.call(rbind, lapply(seq_len(n), function(i) {
    data.frame(id = i, time = visit, x = x[i],
               y = (2 + 0.3 * x[i] + b0[i]) + (0.5 + b1[i]) * visit +
                   rnorm(length(visit), 0, 0.2))
  }))
  ds <- data.frame(id = seq_len(n), time = runif(n, 1.5, 3),
                   event = rbinom(n, 1, 0.6))
  dl$id <- factor(dl$id); ds$id <- factor(ds$id)
  list(data_long = dl, data_surv = ds)
}

# scale_time applies to the MCMC methods ONLY - the MLE paths use
# control$parscale for conditioning, and applying both moved estimates
# that previously matched JM to four decimals. So these tests run MCMC,
# with a small budget since they check reparameterization invariance
# rather than convergence quality.
fit_it <- function(d, ctl = list(), long_f = y ~ time + x, rand_f = ~ time) {
  jm_fit(long_formula = long_f, surv_formula = Surv(time, event) ~ 1,
         data_long = d$data_long, data_surv = d$data_surv,
         id_var = "id", time_var = "time", method = "spline-PH-mcmc",
         random_effects = "intercept_slope", random_formula = rand_f,
         control = c(list(n_interior_knots = 5L, num_warmup = 250L,
                           num_samples = 250L, num_chains = 2L,
                           progress_bar = FALSE, seed = 11L), ctl))
}

test_that("scale_time leaves every reported parameter unchanged", {
  skip_if_no_backend()
  d <- make_data()
  f0 <- fit_it(d)
  f1 <- fit_it(d, list(scale_time = 5))

  # A KNOWN divisor, so a failure lands on a recognisable multiple: a ratio
  # of 5 means the correction never fired, 25 means it fired backwards.
  for (nm in c("beta_0", "beta_1", "beta_2", "sigma_e",
               "sigma_b0", "sigma_b1", "alpha")) {
    expect_equal(f1$estimates[[nm]], f0$estimates[[nm]], tolerance = 0.02,
                 label = sprintf("%s: off %.6f vs on %.6f (ratio %.4f)", nm,
                                 f0$estimates[[nm]], f1$estimates[[nm]],
                                 f1$estimates[[nm]] / f0$estimates[[nm]]))
  }
})

test_that("scale_time leaves the survival submodel alone", {
  skip_if_no_backend()
  d <- make_data()
  f0 <- fit_it(d); f1 <- fit_it(d, list(scale_time = 5))
  # The baseline hazard must be untouched: scale_time moves only the
  # design COLUMNS, leaving survival times, spline knots and quadrature
  # nodes raw. Under the spline MCMC path those are the W coefficients and
  # the smoothing precision tau_w.
  #
  # The spline sites are named W0, W1, ... with NO underscore - `_site_names`
  # uses f"W{i}" while beta uses f"beta_{i}". An earlier version of this
  # test matched on "^W_", found nothing, and the loop body never ran:
  # testthat reported it as an EMPTY TEST rather than a pass, which is the
  # only reason it was noticed.
  .wn <- grep("^W[0-9]+$|^tau_w$", names(f0$estimates), value = TRUE)
  expect_gt(length(.wn), 0)          # fail loudly if the names change again
  for (nm in .wn) {
    expect_equal(f1$estimates[[nm]], f0$estimates[[nm]], tolerance = 0.15,
                 label = nm)
  }
})

test_that("scale_time composes with standardize_covariates", {
  skip_if_no_backend()
  d <- make_data()
  d$data_long$x <- d$data_long$x + 50      # make standardization bite
  f0 <- fit_it(d, list(standardize_covariates = TRUE))
  f1 <- fit_it(d, list(standardize_covariates = TRUE, scale_time = 5))
  for (nm in c("beta_1", "beta_2", "sigma_b1")) {
    expect_equal(f1$estimates[[nm]], f0$estimates[[nm]], tolerance = 0.02,
                 label = nm)
  }
})

test_that("scale_time handles a non-linear time term", {
  skip_if_no_backend()
  d <- make_data()
  # I(time^2) scales by c^2, not c. The projection is recovered from the
  # design matrices rather than parsed from the formula, so it discovers
  # that without being told.
  f0 <- fit_it(d, list(), long_f = y ~ time + I(time^2) + x)
  f1 <- fit_it(d, list(scale_time = 5), long_f = y ~ time + I(time^2) + x)
  for (nm in c("beta_1", "beta_2", "beta_3")) {
    if (nm %in% names(f0$estimates))
      expect_equal(f1$estimates[[nm]], f0$estimates[[nm]], tolerance = 0.05,
                   label = nm)
  }
})

test_that("scale_time errors rather than returning mixed scales", {
  skip_if_no_backend()
  d <- make_data()
  # log(time) in random_formula gives a SHIFT, not a scale:
  # log(t/c) = log(t) - log(c). A_z is then upper triangular, so sigma_b
  # would need a full congruence and the per-component correction cannot
  # be right. Returning beta on the original scale and sigma_b on the
  # scaled one would be internally inconsistent, so the fit stops.
  d$data_long$time <- d$data_long$time + 1      # keep log() finite
  expect_error(
    fit_it(d, list(scale_time = 5), long_f = y ~ log(time) + x,
           rand_f = ~ log(time)),
    "not diagonal|cannot be recovered")
})

test_that("a divisor of 1 is a no-op", {
  skip_if_no_backend()
  d <- make_data()
  f0 <- fit_it(d)
  f1 <- fit_it(d, list(scale_time = 1))
  expect_equal(f1$estimates[["beta_1"]], f0$estimates[["beta_1"]],
               tolerance = 0.01)
})

test_that('scale_time = "auto" declines silently where TRUE errors', {
  skip_if_no_backend()
  d <- make_data()
  d$data_long$time <- d$data_long$time + 1        # keep log() finite

  # ~ log(time) gives log(t/c) = log(t) - log(c), a SHIFT: A_z comes out
  # upper triangular, so the per-component sigma_b correction cannot be
  # right. An explicit request errors; "auto" was not asked for on this
  # particular model, so turning a working fit into a hard failure would
  # be a regression - it declines instead.
  expect_error(
    fit_it(d, list(scale_time = 5), long_f = y ~ log(time) + x,
           rand_f = ~ log(time)),
    "not diagonal|cannot be recovered")

  f <- fit_it(d, list(scale_time = "auto"), long_f = y ~ log(time) + x,
              rand_f = ~ log(time))
  expect_true(grepl("^declined_", f$scale_time_status))
})

test_that('scale_time = "auto" applies where the geometry allows it', {
  skip_if_no_backend()
  d <- make_data()
  f_auto <- fit_it(d, list(scale_time = "auto"))
  expect_true(grepl("^applied", f_auto$scale_time_status))

  # and the estimates still match the unscaled fit
  f_off <- fit_it(d)
  for (nm in c("beta_1", "sigma_b1", "alpha")) {
    expect_equal(f_auto$estimates[[nm]], f_off$estimates[[nm]],
                 tolerance = 0.02, label = nm)
  }
})

test_that("scale_time_status is always present", {
  skip_if_no_backend()
  d <- make_data()

  # The DEFAULT is now "auto", so an unconfigured fit reports "applied"
  # on any model where the random-effects transformation is diagonal.
  # This asserted "off" while the default was FALSE, and failing when the
  # default changed is the test doing its job.
  expect_true(grepl("^applied", fit_it(d)$scale_time_status))

  # Explicitly off still reports off - the field records what happened,
  # not what was asked for.
  expect_identical(fit_it(d, list(scale_time = FALSE))$scale_time_status, "off")
})

test_that("the default is auto and can be turned off", {
  skip_if_no_backend()
  d <- make_data()
  f_default <- fit_it(d)
  f_off     <- fit_it(d, list(scale_time = FALSE))

  # Scaling is a reparameterization: turning it on by default must not
  # change a single reported number. This is the assertion that makes the
  # default defensible - if it fails, the default is wrong regardless of
  # what it does for convergence.
  for (nm in c("beta_0", "beta_1", "beta_2", "sigma_e",
               "sigma_b0", "sigma_b1", "alpha")) {
    expect_equal(f_default$estimates[[nm]], f_off$estimates[[nm]],
                 tolerance = 0.02, label = nm)
  }
})

test_that("scale_time treats random_formula = NULL like the explicit form", {
  skip_if_no_backend()
  d <- make_data()

  # With random_effects = "intercept_slope" and random_formula = NULL,
  # build_time_design() constructs Z from time_var internally rather than
  # from a formula object. An earlier version substituted I(time/c) only
  # when random_formula was non-NULL, so X was scaled and Z was not - a
  # mixed-scale model in which the two specifications of the SAME model
  # stopped agreeing. It surfaced only when scale_time was made the
  # default and the backward-compatibility test in test-random-formula.R
  # started failing.
  # NOTE the formula is `y ~ time`, with NO baseline covariate. jmjax
  # REJECTS covariates in long_formula at q=2 when random_formula is NULL,
  # because Z is then taken as X's first q columns and which columns those
  # are depends on the order terms were written in. An earlier version of
  # this test used `y ~ time + x`, which is exactly that rejected
  # configuration - the test was asking for something the package does not
  # support, and only surfaced once the guard was fixed to fire correctly.
  f_null <- jm_fit(long_formula = y ~ time,
                   surv_formula = Surv(time, event) ~ 1,
                   data_long = d$data_long, data_surv = d$data_surv,
                   id_var = "id", time_var = "time",
                   method = "spline-PH-mcmc",
                   random_effects = "intercept_slope", random_formula = NULL,
                   control = list(scale_time = 5, n_interior_knots = 5L,
                                   num_warmup = 250L, num_samples = 250L,
                                   num_chains = 2L, progress_bar = FALSE,
                                   seed = 11L))
  f_expl <- jm_fit(long_formula = y ~ time,
                   surv_formula = Surv(time, event) ~ 1,
                   data_long = d$data_long, data_surv = d$data_surv,
                   id_var = "id", time_var = "time",
                   method = "spline-PH-mcmc",
                   random_effects = "intercept_slope", random_formula = ~ time,
                   control = list(scale_time = 5, n_interior_knots = 5L,
                                   num_warmup = 250L, num_samples = 250L,
                                   num_chains = 2L, progress_bar = FALSE,
                                   seed = 11L))

  for (nm in names(f_expl$estimates)) {
    expect_equal(f_null$estimates[[nm]], f_expl$estimates[[nm]],
                 tolerance = 1e-4, label = nm)
  }
  expect_identical(f_null$scale_time_status, f_expl$scale_time_status)
})

test_that("the q=2 ambiguity guard still fires under scale_time", {
  skip_if_no_backend()
  d <- make_data()

  # jmjax rejects covariates in long_formula at q=2 with random_formula
  # NULL, because Z = X[, 1:q] picks columns by POSITION: `y ~ time + x`
  # slices [intercept, time] but `y ~ x + time` slices [intercept, x],
  # silently putting a random effect on the covariate instead of on time.
  #
  # scale_time materialises a NULL random_formula into ~ time so that X
  # and Z scale together. That must NOT defeat this guard - it did in an
  # earlier version, letting through a specification jmjax had decided was
  # ambiguous. The guard tests what the USER passed, not what the
  # internals substituted.
  for (st in list(FALSE, 5, "auto")) {
    expect_error(
      fit_it(d, list(scale_time = st), long_f = y ~ time + x,
             rand_f = NULL),
      "require random_formula to be given explicitly",
      label = sprintf("scale_time = %s", format(st)))
  }
})

test_that("scale_time is ignored for the MLE methods", {
  skip_if_no_backend()
  d <- make_data()
  mle <- function(ctl) jm_fit(
    long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
    data_long = d$data_long, data_surv = d$data_surv, id_var = "id",
    time_var = "time", method = "weibull-PH-aGH",
    random_effects = "intercept_slope", random_formula = ~ time,
    control = ctl)

  # Applying scale_time to an MLE fit changed estimates that previously
  # matched JM to four decimal places (prothro -0.0384 -> -0.0395, liver
  # -0.0385 -> -0.0240). The MLE paths condition with control$parscale
  # instead, so scale_time must be a no-op here - and say so rather than
  # silently ignoring the request.
  f_off <- mle(list())
  expect_message(f_on <- mle(list(scale_time = TRUE)),
                 "applies to the MCMC methods only")
  for (nm in names(f_off$estimates)) {
    expect_equal(f_on$estimates[[nm]], f_off$estimates[[nm]],
                 tolerance = 1e-8, label = nm)
  }
  expect_identical(f_on$scale_time_status, "off")
})
