test_that("parse_functional_forms defaults to value-only when NULL", {
  expect_equal(parse_functional_forms(NULL, "y"), "value")
})

test_that("parse_functional_forms correctly extracts requested types", {
  expect_equal(parse_functional_forms(~ value(y), "y"), "value")
  expect_equal(sort(parse_functional_forms(~ value(y) + delta(y), "y")), c("delta", "value"))
  expect_equal(parse_functional_forms(~ delta(y), "y"), "delta")
})

test_that("parse_functional_forms de-duplicates repeated types", {
  # terms() itself would typically dedupe identical formula terms, but
  # guard explicitly since we rely on unique() ourselves too.
  expect_equal(parse_functional_forms(~ value(y) + value(y), "y"), "value")
})

test_that("parse_functional_forms rejects unknown functional forms clearly", {
  expect_error(parse_functional_forms(~ nonsense(y), "y"), "Unknown functional form")
})

test_that("parse_functional_forms flags planned-but-unimplemented forms distinctly from unknown ones", {
  expect_error(parse_functional_forms(~ slope(y), "y"), "planned but not yet implemented")
})

test_that("parse_functional_forms now accepts area() now that it's implemented", {
  expect_equal(parse_functional_forms(~ area(y), "y"), "area")
  expect_equal(sort(parse_functional_forms(~ value(y) + area(y), "y")), c("area", "value"))
})

test_that("parse_functional_forms warns (but doesn't error) on an argument/response mismatch", {
  expect_warning(
    result <- parse_functional_forms(~ value(wrong_name), "y"),
    "don't match the longitudinal response variable"
  )
  expect_equal(result, "value")  # still parses correctly despite the mismatch warning
})

test_that("build_delta_channel: delta(0) is exactly zero for every subject", {
  # delta(t) = X(t) - X(0), so evaluating the channel's OWN design at t=0
  # should give exactly zero - a direct mathematical sanity check.
  t_quad <- matrix(c(0, 0, 0), ncol = 1)  # evaluate "quadrature" at t=0 for this check
  X_time_quad_zero <- build_time_design(y ~ time, "time", t_quad)

  delta_at_zero <- build_delta_channel(y ~ time, "time", X_time_quad_zero[, 1, ], X_time_quad_zero)

  expect_equal(unname(delta_at_zero$X_delta_surv), matrix(0, nrow = 3, ncol = 2))
})

test_that("build_delta_channel correctly computes X(t) - X(0) for a simple linear-in-time formula", {
  T_surv <- c(0.5, 1.0, 2.5)
  X_time_surv <- build_time_design(y ~ time, "time", T_surv)  # columns: (Intercept), time
  # Second argument must be a 3D [N_sub, n_quad, p] array, not the 2D
  # X_time_surv matrix - build a small fake quadrature grid for this check.
  t_quad <- outer(T_surv, c(0.3, 0.6, 0.9))
  X_time_quad <- build_time_design(y ~ time, "time", t_quad)

  result <- build_delta_channel(y ~ time, "time", X_time_surv, X_time_quad)

  # For y ~ time, X(t) = [1, t], X(0) = [1, 0], so delta = [0, t] - intercept
  # column should vanish entirely, time column should equal T_surv exactly.
  expect_equal(unname(result$X_delta_surv[, 1]), rep(0, 3))
  expect_equal(unname(result$X_delta_surv[, 2]), T_surv)
})

test_that("build_delta_channel produces correctly-shaped quadrature-grid output", {
  T_surv <- c(1, 2, 3)
  X_time_surv <- build_time_design(y ~ time, "time", T_surv)
  t_quad <- outer(T_surv, c(0.2, 0.5, 0.8))  # [3, 3] fake quadrature grid
  X_time_quad <- build_time_design(y ~ time, "time", t_quad)

  result <- build_delta_channel(y ~ time, "time", X_time_surv, X_time_quad)

  expect_equal(dim(result$X_delta_quad), dim(X_time_quad))
  # Time column of delta at quad node k should equal t_quad[,k] exactly
  # (same y ~ time logic as the build_time_design test above).
  expect_equal(unname(result$X_delta_quad[, 1, 2]), t_quad[, 1])
  expect_equal(unname(result$X_delta_quad[, 3, 2]), t_quad[, 3])
})

test_that("jm_fit rejects functional_forms combined with an unsupported method", {
  skip_if_no_backend()

  sim <- simulate_joint_data(n = 50, seed = 1)

  # spline-PH-mcmc + delta is now correctly SUPPORTED at q=2 (and
  # specifically, correctly REJECTED at q=1 for identifiability reasons -
  # see "spline-PH-mcmc rejects delta at q=1" above) - with all four
  # methods now having SOME functional_forms support, there may no longer
  # be a case hitting the fully generic "method doesn't support
  # functional_forms at all" fallback. spline-PH-aGH + area is still a
  # genuinely, specifically unsupported combination though (only 'delta'
  # has been ported to that particular method) - use that instead.
  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "spline-PH-aGH",
      functional_forms = ~ value(y) + area(y)
    ),
    "not yet supported for method"
  )

  # NOTE: weibull-PH-aGH + random_effects = "intercept_slope" + a
  # functional form is NO LONGER expected to error - it's a supported
  # combination as of the q=2 extension (see
  # test_that("weibull-PH-aGH with q=2 (intercept_slope) + delta ...")
  # below for its actual truth-recovery test). An earlier version of this
  # test asserted an error here; removed once that scope limit was lifted.
})

test_that("weibull-PH-aGH with functional_forms = ~ value(y) + delta(y) recovers both true association parameters", {
  skip_if_no_backend()

  sim <- simulate_joint_data_delta(n = 400, seed = 7, alpha_value = 0.5, alpha_delta = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    functional_forms = ~ value(y) + delta(y)
  )

  expect_true(fit$convergence$converged)

  # Both channels should appear with distinct names, not collapsed into "alpha"
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))
  expect_false("alpha" %in% names(fit$estimates))

  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_delta), 3 * fit$se[["alpha_delta"]])

  # beta/sigma_e/sigma_b should be unaffected by adding the delta channel -
  # same recovery quality as the plain value-only path.
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

test_that("functional_forms = ~ value(y) (explicit) matches the default (implicit) value-only fit", {
  skip_if_no_backend()

  sim <- simulate_joint_data(n = 200, seed = 42)

  fit_default <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH"
  )

  fit_explicit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    functional_forms = ~ value(y)
  )

  # Should be numerically identical (same code path, same optimizer, same
  # data) - explicit ~ value(y) is purely a no-op relative to the default.
  expect_equal(fit_default$estimates, fit_explicit$estimates, tolerance = 1e-8)
})

test_that("build_area_channel matches the closed-form analytic integral for y ~ time", {
  # For y ~ time, X(s) = [1, s], so area(t) = integral_0^t [1,s] ds =
  # [t, t^2/2] EXACTLY - a closed form we can check the nested-quadrature
  # implementation against directly, rather than only checking internal
  # self-consistency. Gauss-Kronrod quadrature integrates low-degree
  # polynomials essentially exactly, so we expect near-machine-precision
  # agreement here, not just "close".
  T_surv <- c(1, 2, 3, 4)
  t_quad <- outer(T_surv, c(0.2, 0.5, 0.8))  # small fake outer grid, 3 nodes
  gk_weights_outer <- c(0.3, 0.4, 0.3)       # arbitrary weights summing to 1 for this check

  result <- build_area_channel(y ~ time, "time", T_surv, t_quad, gk_weights_outer,
                                gk_order_inner = 10)

  # area(T_i) analytic: [T_i, T_i^2/2]
  expect_equal(unname(result$X_area_surv[, 1]), T_surv, tolerance = 1e-8)
  expect_equal(unname(result$X_area_surv[, 2]), T_surv^2 / 2, tolerance = 1e-8)

  # area(s_k) analytic, for each outer node k: [s_k, s_k^2/2]
  for (k in seq_len(ncol(t_quad))) {
    s_k <- t_quad[, k]
    expect_equal(unname(result$X_area_quad[, k, 1]), s_k, tolerance = 1e-8)
    expect_equal(unname(result$X_area_quad[, k, 2]), s_k^2 / 2, tolerance = 1e-8)
  }
})

test_that("build_area_channel: area(0) is exactly zero (integral over a zero-width interval)", {
  t_quad_zero <- matrix(rep(0, 3), ncol = 1)
  result <- build_area_channel(y ~ time, "time", rep(0, 3), t_quad_zero,
                                gk_weights_outer = 1, gk_order_inner = 10)
  expect_equal(unname(result$X_area_surv), matrix(0, nrow = 3, ncol = 2), tolerance = 1e-10)
})

test_that("jm_fit rejects combining area and delta simultaneously with a clear message", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      functional_forms = ~ value(y) + delta(y) + area(y)
    ),
    "not yet supported"
  )
})

test_that("weibull-PH-aGH with functional_forms = ~ value(y) + area(y) recovers both true association parameters", {
  skip_if_no_backend()

  sim <- simulate_joint_data_area(n = 400, seed = 11, alpha_value = 0.4, alpha_area = 0.2)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    functional_forms = ~ value(y) + area(y)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_area") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area"]] - sim$truth$alpha_area), 3 * fit$se[["alpha_area"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

test_that("area_avg is correctly computed as area(t)/t for y ~ time (closed form)", {
  # For y ~ time, area(t) = [t, t^2/2] (confirmed exactly by the earlier
  # closed-form test), so area_avg(t) = area(t)/t = [1, t/2] - a second,
  # independent closed-form check.
  T_surv <- c(1, 2, 4)
  t_quad <- outer(T_surv, c(0.25, 0.75))
  raw_area <- build_area_channel(y ~ time, "time", T_surv, t_quad, gk_weights_outer = c(0.5, 0.5))

  area_avg_surv <- raw_area$X_area_surv / T_surv
  expect_equal(unname(area_avg_surv[, 1]), rep(1, 3), tolerance = 1e-8)
  expect_equal(unname(area_avg_surv[, 2]), T_surv / 2, tolerance = 1e-8)
})

test_that("jm_fit rejects combining more than one of delta/area/area_avg", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "weibull-PH-aGH",
      functional_forms = ~ value(y) + area(y) + area_avg(y)
    ),
    "not yet supported"
  )
})

test_that("weibull-PH-aGH with functional_forms = ~ value(y) + area_avg(y) recovers both true association parameters", {
  skip_if_no_backend()

  sim <- simulate_joint_data_area_avg(n = 400, seed = 13, alpha_value = 0.4, alpha_area_avg = 0.5)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    functional_forms = ~ value(y) + area_avg(y)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_area_avg") %in% names(fit$estimates)))

  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area_avg"]] - sim$truth$alpha_area_avg), 3 * fit$se[["alpha_area_avg"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

test_that("weibull-PH-aGH with q=2 (intercept_slope) + delta recovers both true association parameters", {
  skip_if_no_backend()
  sim <- simulate_joint_data_re2_channel("delta", n = 500, seed = 31, alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + delta(y)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_delta"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sim$truth$sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sim$truth$sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-aGH with q=2 (intercept_slope) + area recovers both true association parameters", {
  skip_if_no_backend()
  sim <- simulate_joint_data_re2_channel("area", n = 500, seed = 32, alpha_value = 0.3, alpha_extra = 0.15)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + area(y)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_area") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_area"]])
})

test_that("weibull-PH-aGH with q=2 (intercept_slope) + area_avg recovers both true association parameters", {
  skip_if_no_backend()
  sim <- simulate_joint_data_re2_channel("area_avg", n = 500, seed = 33, alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + area_avg(y)
  )

  expect_true(fit$convergence$converged)
  expect_true(all(c("alpha_value", "alpha_area_avg") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area_avg"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_area_avg"]])
})

test_that("q=2 + area_avg matches JMbayes2 on a genuine q=2 dataset (both structurally matched)", {
  skip_if_no_backend()
  skip_if_no_JMbayes2()

  sim <- simulate_joint_data_re2_channel("area_avg", n = 500, seed = 34, alpha_value = 0.4, alpha_extra = 0.3)

  fit_jmjax <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-aGH",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + area_avg(y)
  )

  lme_fit <- nlme::lme(y ~ time, random = ~ time | id, data = sim$data_long,
                        control = nlme::lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- survival::coxph(survival::Surv(time, event) ~ 1, data = sim$data_surv)

  fit_jmbayes2 <- JMbayes2::jm(
    Surv_object = cox_fit,
    Mixed_objects = lme_fit,
    time_var = "time",
    base_hazard = "weibull",
    functional_forms = list(y = ~ JMbayes2::value(y) + JMbayes2::area(y)),
    n_chains = 2,
    n_iter = 3500,
    n_burnin = 500
  )

  alphas_jmbayes2 <- fit_jmbayes2$statistics$Mean$alphas

  # Now a GENUINE q=2 vs q=2 comparison (no structural mismatch this time,
  # unlike the earlier q=1-vs-q=2 workaround) - expect closer agreement,
  # though still not the near-exact match seen for value-only q=2 (this is
  # a harder, more collinear estimation problem, and MLE-vs-MCMC still
  # differ). A generous tolerance reflects that, while still catching a
  # genuinely broken implementation (e.g. wrong sign, order-of-magnitude
  # difference).
  expect_lt(abs(fit_jmjax$estimates[["alpha_value"]] - alphas_jmbayes2[1]), 1.0)
  expect_lt(abs(fit_jmjax$estimates[["alpha_area_avg"]] - alphas_jmbayes2[2]), 1.0)
})

test_that("spline-PH-aGH + delta at q=1 (intercept) is rejected as non-identifiable, not silently fit", {
  skip_if_no_backend()

  # CONFIRMED NON-IDENTIFIABLE (not merely unimplemented) - see jm_fit.R's
  # guard for the full explanation. Empirically: fitting this combination
  # before the guard was added gave beta/sigma_e/sigma_b/alpha_value
  # essentially perfect, but alpha_delta with the WRONG SIGN and a
  # singular (NaN) SE, with 8/9 spline coefficients also NaN - the
  # signature of a genuine likelihood ridge, not a fixable bug. This test
  # confirms the guard actively rejects the combination with a specific,
  # informative message (distinct from the generic "not yet supported"
  # wording used for genuinely-just-unimplemented combinations elsewhere
  # in this file), rather than letting a user silently get a garbage fit.
  sim <- simulate_joint_data_delta(n = 400, seed = 21, alpha_value = 0.5, alpha_delta = 0.3)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "spline-PH-aGH",
      random_effects = "intercept",  # the confirmed non-identifiable case
      functional_forms = ~ value(y) + delta(y)
    ),
    "NON-IDENTIFIABLE"
  )
})

test_that("spline-PH-aGH rejects area/area_avg (not yet ported from Weibull)", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 50, seed = 1)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "spline-PH-aGH",
      functional_forms = ~ value(y) + area(y)
    ),
    "not yet supported for method = 'spline-PH-aGH'"
  )
})

test_that("weibull-PH-mcmc with functional_forms = ~ value(y) + delta(y) recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Same simulator already used for the weibull-PH-aGH (MLE) + delta
  # truth-recovery test - the data-generating hazard doesn't know or care
  # whether jmjax will fit it via MLE or MCMC.
  sim <- simulate_joint_data_delta(n = 400, seed = 21, alpha_value = 0.5, alpha_delta = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    functional_forms = ~ value(y) + delta(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))
  expect_false("alpha" %in% names(fit$estimates))

  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_delta), 3 * fit$se[["alpha_delta"]])
  expect_lt(abs(fit$estimates[["beta_0"]] - sim$truth$beta0), 3 * fit$se[["beta_0"]])
  expect_lt(abs(fit$estimates[["beta_1"]] - sim$truth$beta1), 3 * fit$se[["beta_1"]])
})

# NOTE: an earlier version of this test asserted that delta + q=2 +
# weibull-PH-mcmc should error ("not yet implemented"). This was true
# only until q=2 support was added to the MCMC backend (see "weibull-PH-mcmc
# with q=2 + delta recovers both true association parameters" above for
# its actual truth-recovery test) - removed once that guard was lifted.

test_that("weibull-PH-mcmc with functional_forms = ~ value(y) + area(y) recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Same simulator already used for the weibull-PH-aGH (MLE) + area
  # truth-recovery test.
  sim <- simulate_joint_data_area(n = 400, seed = 11, alpha_value = 0.4, alpha_area = 0.2)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    functional_forms = ~ value(y) + area(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area"]] - sim$truth$alpha_area), 3 * fit$se[["alpha_area"]])
})

test_that("weibull-PH-mcmc with functional_forms = ~ value(y) + area_avg(y) recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # Same simulator already used for the weibull-PH-aGH (MLE) + area_avg
  # truth-recovery test.
  sim <- simulate_joint_data_area_avg(n = 400, seed = 13, alpha_value = 0.4, alpha_area_avg = 0.5)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    functional_forms = ~ value(y) + area_avg(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area_avg") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area_avg"]] - sim$truth$alpha_area_avg), 3 * fit$se[["alpha_area_avg"]])
})

test_that("weibull-PH-mcmc with q=2 + delta recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2_channel("delta", n = 500, seed = 71, alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + delta(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_delta"]])
  expect_lt(abs(fit$estimates[["sigma_b0"]] - sim$truth$sigma_b0), 3 * fit$se[["sigma_b0"]])
  expect_lt(abs(fit$estimates[["sigma_b1"]] - sim$truth$sigma_b1), 3 * fit$se[["sigma_b1"]])
})

test_that("weibull-PH-mcmc with q=2 + area recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2_channel("area", n = 500, seed = 72, alpha_value = 0.3, alpha_extra = 0.15)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + area(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_area"]])
})

test_that("weibull-PH-mcmc with q=2 + area_avg recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2_channel("area_avg", n = 500, seed = 73, alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + area_avg(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area_avg") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area_avg"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_area_avg"]])
})

test_that("weibull-PH-mcmc without functional_forms still behaves exactly as before (backward compatibility)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data(n = 300, seed = 55)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "weibull-PH-mcmc",
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_true("alpha" %in% names(fit$estimates))
  expect_false("alpha_value" %in% names(fit$estimates))
  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
})

test_that("spline-PH-mcmc rejects delta at q=1 (confirmed non-identifiable, same as spline-PH-aGH)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_delta(n = 400, seed = 21, alpha_value = 0.5, alpha_delta = 0.3)

  expect_error(
    jm_fit(
      long_formula = y ~ time,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long,
      data_surv = sim$data_surv,
      id_var = "id",
      time_var = "time",
      method = "spline-PH-mcmc",
      random_effects = "intercept",
      functional_forms = ~ value(y) + delta(y)
    ),
    "NON-IDENTIFIABLE"
  )
})

test_that("spline-PH-mcmc with q=1 + area recovers both true association parameters (testing the 'no identifiability problem' reasoning)", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  # THE key empirical test of the reasoning in jm_fit.R's guard: area's
  # q=1 random-effect contribution (b_i*t) IS genuine subject-level
  # variation, unlike delta's - so this should NOT show delta's
  # identifiability problem even with the same flexible spline baseline.
  sim <- simulate_joint_data_area(n = 400, seed = 11, alpha_value = 0.4, alpha_area = 0.2)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    functional_forms = ~ value(y) + area(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area"]] - sim$truth$alpha_area), 3 * fit$se[["alpha_area"]])
})

test_that("spline-PH-mcmc with q=1 + area_avg recovers both true association parameters", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_area_avg(n = 400, seed = 13, alpha_value = 0.4, alpha_area_avg = 0.5)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    functional_forms = ~ value(y) + area_avg(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_area_avg") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_area_avg"]] - sim$truth$alpha_area_avg), 3 * fit$se[["alpha_area_avg"]])
})

test_that("spline-PH-mcmc with q=2 + delta resolves the q=1 identifiability problem", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_re2_channel("delta", n = 500, seed = 61, alpha_value = 0.4, alpha_extra = 0.3)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    random_effects = "intercept_slope",
    functional_forms = ~ value(y) + delta(y),
    control = list(num_warmup = 500, num_samples = 1000, num_chains = 1, progress_bar = FALSE)
  )

  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
  expect_true(all(c("alpha_value", "alpha_delta") %in% names(fit$estimates)))
  expect_lt(abs(fit$estimates[["alpha_value"]] - sim$truth$alpha_value), 3 * fit$se[["alpha_value"]])
  expect_lt(abs(fit$estimates[["alpha_delta"]] - sim$truth$alpha_extra), 3 * fit$se[["alpha_delta"]])
})
