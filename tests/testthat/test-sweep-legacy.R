# ==============================================================================
# LEGACY. The location sweep (control$orthogonalize_b0 / orthogonalize_b and
# its rotations) is not part of jmjax's recommended path - see the backend
# module sweep.py and dev/notes/sweep-reparameterization.md. These tests are
# kept so that code, which the dev/ comparison studies still exercise, does
# not silently break. The recommended construction is tested in
# test-rotate-absorbable.R.
#
# Regression tests for control$orthogonalize_b0 / control$orthogonalize_b
# (the location-degeneracy reparameterization; see ?jm_fit and
# vignette("jmjax-reparameterization") for the full mechanism).
#
# Prior to this file, everything validating this option lived in one-off
# dev/ scripts (diagnose_beta0*.R, test_orthogonalize_b0*.R,
# verify_orth_slope.R, and the dev/study_*.R simulation studies) - useful
# for the original investigation, but nothing here would catch a future
# change silently breaking the option. These tests are deliberately not a
# re-run of that investigation (no attempt to reproduce the measured 6-15x
# ESS ratios with statistical rigor - that needs the much larger,
# multi-seed replication in dev/study_orth_vs_jmbayes2.R and is recorded in
# vignette("jmjax-reparameterization") instead). They check the three
# things a silent regression would actually break: the option still
# refuses configurations it does not support, it still does not change
# what the model fits (the "model is unchanged" claim in ?jm_fit is a
# testable consequence of the construction, not just an algebra argument),
# and it still measurably helps the parameter it targets.
# ==============================================================================

test_that("orthogonalize_b0/orthogonalize_b refuse configurations outside their supported scope", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 11)

  # q = 1 (random_effects = "intercept"): orthogonalize_b0 requires q >= 2.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept",
      control = list(orthogonalize_b0 = TRUE,
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )

  # q = 2 but random_effects_corr = FALSE: orthogonalize_b requires
  # random_effects_corr = TRUE.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept_slope",
      control = list(orthogonalize_b = TRUE, random_effects_corr = FALSE,
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )

  # q = 2, corr = TRUE, but random_effects_method != "nuts": orthogonalize_b0
  # requires the default NUTS random-effects sampler.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time",
      method = "spline-PH-mcmc", random_effects = "intercept_slope",
      control = list(orthogonalize_b0 = TRUE,
                     random_effects_method = "wishart_gibbs",
                     num_warmup = 20, num_samples = 20, num_chains = 1,
                     progress_bar = FALSE)
    ),
    "requires q >= 2"
  )
})

test_that("fit$convergence$orthogonalize reports which columns were actually constrained", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 150, seed = 12)

  # orthogonalize_b0: intercept column (0) only. _absorbable_basis() should
  # find the intercept as an absorbable direction (b[,0]); the slope column
  # (1) is not even attempted at this option, so it must come back
  # unconstrained - this is the routine, by-design case for orthogonalize_b0
  # (distinct from the "no fixed-effect column can absorb anything" warning
  # path, which is not exercised here).
  fit_d <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(orthogonalize_b0 = TRUE,
                   num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )

  orth_d <- fit_d$convergence$orthogonalize
  expect_false(is.null(orth_d))
  expect_identical(orth_d$requested, "b0")
  expect_true(isTRUE(orth_d$any_applied))
  expect_length(orth_d$columns, 2)

  col0_d <- orth_d$columns[[1]]
  expect_equal(col0_d$column, 0)
  expect_true(isTRUE(col0_d$applied))
  expect_gte(col0_d$n_directions, 1)

  col1_d <- orth_d$columns[[2]]
  expect_equal(col1_d$column, 1)
  expect_false(isTRUE(col1_d$applied))  # orthogonalize_b0 does not touch the slope

  # orthogonalize_b: every column. The design's fixed "time" term means the
  # slope column (1) should ALSO come back constrained - this is the
  # k-independent slope degeneracy documented in ?jm_fit and
  # vignette("jmjax-reparameterization"); it does not require any extra
  # covariates, just a fixed slope on time alongside the random one.
  fit_e <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(orthogonalize_b = TRUE,
                   num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )

  orth_e <- fit_e$convergence$orthogonalize
  expect_false(is.null(orth_e))
  expect_identical(orth_e$requested, "b")
  expect_true(isTRUE(orth_e$any_applied))
  expect_length(orth_e$columns, 2)
  expect_true(isTRUE(orth_e$columns[[1]]$applied))
  expect_true(isTRUE(orth_e$columns[[2]]$applied))
  expect_gte(orth_e$columns[[2]]$n_directions, 1)

  # A default fit (neither option requested) must report NULL, not an
  # empty/degenerate structure - callers use is.null() to check applicability.
  fit_a <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    control = list(num_warmup = 150, num_samples = 200, num_chains = 1,
                   progress_bar = FALSE)
  )
  expect_null(fit_a$convergence$orthogonalize)
})

test_that("orthogonalize_b0/orthogonalize_b do not change fitted population estimates", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # The construction guarantees this algebraically (every direction swept
  # out of a random-effect column lies in a column space beta already
  # spans - see ?jm_fit, "The model is unchanged"), but algebra and
  # implementation are different claims; this checks the second one. Loose,
  # multiple-of-SE tolerances, same spirit as the existing MCMC
  # truth-recovery tests - this is not a replication of the tight,
  # multi-seed agreement already established in
  # vignette("jmjax-reparameterization").
  sim <- simulate_joint_data_re2(n = 300, seed = 13)

  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 500, num_samples = 800, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_d <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b0 = TRUE)))))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  for (pname in c("alpha", "beta_0", "beta_1")) {
    tol_d <- 3 * max(fit_a$se[[pname]], fit_d$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_d$estimates[[pname]]), tol_d)

    tol_e <- 3 * max(fit_a$se[[pname]], fit_e$se[[pname]])
    expect_lt(abs(fit_a$estimates[[pname]] - fit_e$estimates[[pname]]), tol_e)
  }
})

test_that("orthogonalize_b materially improves beta_1 mixing over the default fit", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # The regression-guard for the headline efficiency claim: measured
  # regression-only ESS/sec ratios in the replicated k-sweep and aids
  # real-data study (vignette("jmjax-reparameterization")) range roughly
  # 5x-15x. A 1.5x threshold on a single seed leaves enormous margin
  # against Monte Carlo noise while still catching the failure mode this
  # test exists for: a future change that silently disables or breaks the
  # reparameterization.
  sim <- simulate_joint_data_re2(n = 300, seed = 14)

  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  ess_a <- fit_a$diagnostics$ess[["beta_1"]]
  ess_e <- fit_e$diagnostics$ess[["beta_1"]]

  expect_true(is.finite(ess_a) && is.finite(ess_e))
  expect_gt(ess_e, 1.5 * ess_a)
})

# ==============================================================================
# beta_corrected: the SEPARATE, opt-in analytical-correction track added
# after dev/study_calibration.R's pilot run found orthogonalize_b/_b0
# reproduce wishart_gibbs_centered's retracted miscalibration (unbiased
# point estimates, too-narrow credible intervals for the swept
# parameter(s) -- see NEWS.md and vignette("jmjax-reparameterization")
# Section 9). fit_nuts() computes beta_corrected as pure post-processing of
# draws it already produces (b_std, sigma_b, L_corr, beta) -- no new
# numpyro site, no refit -- so these tests check it is present exactly
# when expected, shaped like beta, and materially restores the posterior
# spread orthogonalization removed. The underlying algebra (the beta-shift
# matrix's defining identity, and that it exactly reproduces the
# unconstrained model's fitted values) is checked independently of any
# MCMC run in dev/verify_beta_correction.py; these tests check the
# INTEGRATION into a real fit instead.
# ==============================================================================

test_that("beta_corrected is present iff orthogonalize_b0/_b actually swept a direction", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data_re2(n = 150, seed = 16)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 150, num_samples = 200, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_d <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b0 = TRUE)))))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  # Default fit: nothing swept, nothing to correct.
  expect_null(fit_a$posterior_samples$beta_corrected)

  # Both D and E sweep the intercept column on this design (see the
  # "fit$convergence$orthogonalize reports..." test above), so both should
  # report a correction shaped exactly like beta: same number of draws,
  # same per-draw length.
  for (f in list(fit_d, fit_e)) {
    bc <- f$posterior_samples$beta_corrected
    b  <- f$posterior_samples$beta
    expect_false(is.null(bc))
    expect_length(bc, length(b))
    expect_length(bc[[1]], length(b[[1]]))
  }
})

test_that("beta_corrected materially restores the posterior spread orthogonalize_b removes", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Same n/settings as the ESS-improvement test above, so this reuses a
  # cost profile already accepted into the suite. Loose, single-seed
  # tolerances in the same spirit as that test -- this is a regression
  # guard against the correction silently breaking or no-op'ing, not a
  # replication of dev/study_calibration.R's calibration study.
  sim <- simulate_joint_data_re2(n = 300, seed = 17)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 500, num_samples = 800, num_chains = 1, progress_bar = FALSE)

  fit_a <- do.call(jm_fit, c(common, list(control = ctrl)))
  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))

  bc0 <- vapply(fit_e$posterior_samples$beta_corrected,
                function(z) as.numeric(unlist(z))[1], numeric(1))
  bc0_sd     <- stats::sd(bc0)
  b0_orth_sd <- fit_e$se[["beta_0"]]
  b0_def_sd  <- fit_a$se[["beta_0"]]

  # The whole point: beta_corrected's spread for the swept parameter should
  # be materially larger than the (miscalibrated, too-narrow) orthogonalized
  # beta_0's own SE -- not just numerically different from it.
  expect_gt(bc0_sd, 1.5 * b0_orth_sd)
  # And in the right ballpark of the default fit's SE -- generous bounds
  # (a factor of 3 either way) because this is a single seed at moderate n,
  # not a claim of exact agreement.
  expect_lt(bc0_sd, 3 * b0_def_sd)
  expect_gt(bc0_sd, b0_def_sd / 3)
})

test_that("orthogonalize_rotate_all refuses unsupported combinations", {
  skip_if_no_backend()

  sim <- simulate_joint_data_re2(n = 100, seed = 18)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 20, num_samples = 20, num_chains = 1, progress_bar = FALSE)

  # Only changes how a swept degeneracy is sampled: nothing swept, refuse.
  expect_error(
    do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_rotate_all = TRUE))))),
    "requires orthogonalize_b0 or orthogonalize_b"
  )
  # The two rotations are alternatives, not layers.
  expect_error(
    do.call(jm_fit, c(common, list(control = c(ctrl, list(
      orthogonalize_b = TRUE, orthogonalize_rotate_all = TRUE,
      orthogonalize_b0_rotate = TRUE))))),
    "request at most one"
  )
})

test_that("orthogonalize_rotate_all is exact: same fit as the unrotated sweep", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Proposition 8 of vignette("jmjax-reparameterization"), Section 4.10:
  # rotating every column of b_std by one fixed orthogonal matrix cannot
  # change the posterior of any model quantity. This checks the
  # implementation, with loose multiple-of-SE tolerances as elsewhere in
  # this file - it is not the efficiency study (dev/pilot_rotate_grid.R).
  sim <- simulate_joint_data_re2(n = 200, seed = 19)
  common <- list(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope"
  )
  ctrl <- list(num_warmup = 400, num_samples = 600, num_chains = 1, progress_bar = FALSE)

  fit_e <- do.call(jm_fit, c(common, list(control = c(ctrl, list(orthogonalize_b = TRUE)))))
  fit_r <- do.call(jm_fit, c(common, list(control = c(ctrl, list(
    orthogonalize_b = TRUE, orthogonalize_rotate_all = TRUE)))))

  rot <- fit_r$convergence$orthogonalize$rotation
  expect_false(is.null(rot))
  expect_identical(rot$mode, "all_columns")
  expect_true(isTRUE(rot$applied))
  expect_gte(as.integer(rot$k), 1L)
  expect_false(is.null(fit_r$posterior_samples$b_gen_U))

  for (pname in c("alpha", "beta_0", "beta_1", "sigma_e")) {
    tol <- 3 * max(fit_e$se[[pname]], fit_r$se[[pname]])
    expect_lt(abs(fit_e$estimates[[pname]] - fit_r$estimates[[pname]]), tol)
  }
  bc <- function(f, j) mean(vapply(f$posterior_samples$beta_corrected,
                                   function(z) as.numeric(unlist(z))[j], numeric(1)))
  bcs <- function(f, j) stats::sd(vapply(f$posterior_samples$beta_corrected,
                                         function(z) as.numeric(unlist(z))[j], numeric(1)))
  for (j in 1:2) {
    expect_lt(abs(bc(fit_e, j) - bc(fit_r, j)), 3 * max(bcs(fit_e, j), bcs(fit_r, j)))
  }
})
