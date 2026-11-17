# ==============================================================================
# More than two random effects (q >= 3) with the MCMC methods (jmjax 0.5.0).
#
# build_model() was already general in q (sigma_b of length q, LKJCholesky(q),
# b = b_std %*% t(L)); what changed is the plumbing around it: jm_fit()'s
# q > 2 stop now applies to the MLE methods only, the backend reads q from
# Z, pairwise correlations are reported as rho_i_j, VarCorr() gives the q x q
# matrix, and ns()/bs()/poly() terms keep the knots of the longitudinal data
# at event times and quadrature nodes.
# ==============================================================================

q3_ctrl <- function(...) {
  utils::modifyList(list(num_warmup = 30, num_samples = 30, num_chains = 1,
                         progress_bar = FALSE), list(...))
}

test_that("fixed-knot terms reproduce the longitudinal basis at new times (no backend)", {
  d <- data.frame(time = c(0, 0.5, 1, 2, 3.5, 5, 0, 1, 4), id = rep(1:3, each = 3))
  for (f in list(~ splines::ns(time, 2), ~ poly(time, 2), ~ time + I(time^2))) {
    tt <- jmjax:::.jmjax_fixed_terms(f, d)
    # Evaluated at the data's own times, the knot-fixed design equals the
    # design built on the data.
    Z_data <- stats::model.matrix(f, d)
    Z_new <- jmjax:::build_time_design(tt, "time", d$time)
    expect_equal(c(Z_new), c(Z_data), tolerance = 1e-12)
    # And it does not depend on which other times are evaluated with it:
    # one time on its own gives the same row as inside a vector.
    expect_equal(c(jmjax:::build_time_design(tt, "time", 2)),
                 c(Z_new[d$time == 2, , drop = FALSE]), tolerance = 1e-12)
  }
})

test_that("q = 3 (random quadratic) fits with the MCMC methods and reports a 3 x 3 correlation", {
  skip_if_no_backend()
  sim <- simulate_joint_data_q3(n = 150, seed = 3)
  fit <- jm_fit(
    long_formula = y ~ time + I(time^2) + x,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", method = "spline-PH-mcmc",
    random_effects = "intercept_slope", random_formula = ~ time + I(time^2),
    control = q3_ctrl())

  nm <- names(fit$estimates)
  expect_true(all(c("sigma_b0", "sigma_b1", "sigma_b2",
                    "rho_0_1", "rho_0_2", "rho_1_2") %in% nm))
  expect_false("rho" %in% nm)

  vc <- VarCorr(fit)
  expect_s3_class(vc, "VarCorr.jmjax")
  expect_identical(names(vc$sd), c("(Intercept)", "time", "I(time^2)"))
  expect_equal(dim(vc$corr), c(3L, 3L))
  expect_equal(unname(diag(vc$corr)), rep(1, 3))
  expect_equal(vc$corr, t(vc$corr))
  expect_equal(unname(vc$corr[1, 2]), unname(fit$estimates[["rho_0_1"]]))
  # suppressWarnings: print() warns about R-hat on this deliberately tiny
  # run, which is not what is being tested.
  expect_output(suppressWarnings(print(fit)), "Random effects")
  expect_output(suppressWarnings(print(summary(fit))), "I\\(time\\^2\\)")
  s <- summary(fit)
  expect_true(all(s[c("rho_0_1", "rho_0_2", "rho_1_2"), "Group"] == "Variance components"))

  # The rotation is on by default at q = 3: absorbable span{1, x}.
  rot <- fit$convergence$orthogonalize$rotation
  expect_true(isTRUE(rot$applied))
  expect_identical(as.integer(rot$k), 2L)
  expect_lt(rot$condition_s_residual, 1e-6)
  expect_false(is.null(fit$posterior_samples$b_gen_U))

  # Random effects come back per subject and per column.
  re <- ranef(fit)
  expect_true(all(c("b0_mean", "b1_mean", "b2_mean") %in% names(re)))

  # predict() at q = 3.
  ids <- head(fit$model_info$subj_ids, 3)
  nd <- sim$data_long[sim$data_long$id %in% ids, ]
  pl <- predict(fit, nd, times = c(0, 1, 2))
  expect_identical(nrow(pl), 9L)
  expect_true(all(is.finite(pl$estimate)))
  pe <- predict(fit, nd, process = "event", t_from = 0.5)
  expect_true(all(pe$estimate >= 0 & pe$estimate <= 1))
})

test_that("q = 3 with a natural-spline basis, ~ ns(time, 2), fits and satisfies Condition (S)", {
  skip_if_no_backend()
  sim <- simulate_joint_data_q3(n = 150, seed = 4)
  fit <- jm_fit(
    long_formula = y ~ splines::ns(time, 2),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", method = "spline-PH-mcmc",
    random_effects = "intercept_slope", random_formula = ~ splines::ns(time, 2),
    control = q3_ctrl())
  expect_true(all(c("sigma_b0", "sigma_b1", "sigma_b2") %in% names(fit$estimates)))
  # ns() of time/c, with its knots chosen on time/c, is the same basis as
  # ns(time), so scale_time = "auto" may apply here (identity transform).
  # With the knots fixed on the longitudinal data, X and Z at the event
  # times and quadrature nodes use the same basis as the longitudinal rows,
  # so the trajectory identity holds there to rounding error. With knots
  # recomputed per evaluation (before 0.5.0) it did not.
  rot <- fit$convergence$orthogonalize$rotation
  expect_lt(rot$condition_s_residual, 1e-6)
  expect_identical(as.integer(rot$n_rejected_condition_s), 0L)
  ids <- head(fit$model_info$subj_ids, 2)
  pl <- predict(fit, sim$data_long[sim$data_long$id %in% ids, ], times = c(0, 2, 4))
  expect_true(all(is.finite(pl$estimate)))
})

test_that("q > 2 is refused by the MLE methods and by the Wishart-Gibbs samplers, with clear messages", {
  skip_if_no_backend()
  sim <- simulate_joint_data_q3(n = 60, seed = 5)
  common <- list(long_formula = y ~ time + I(time^2),
                 surv_formula = survival::Surv(time, event) ~ 1,
                 data_long = sim$data_long, data_surv = sim$data_surv,
                 id_var = "id", time_var = "time",
                 random_effects = "intercept_slope",
                 random_formula = ~ time + I(time^2))
  expect_error(do.call(jm_fit, c(common, list(method = "weibull-PH-aGH"))),
               "maximum-likelihood methods.*at most 2.*jm_bayes")
  expect_error(do.call(jm_fit, c(common, list(method = "spline-PH-aGH"))),
               "maximum-likelihood methods")
  expect_error(do.call(jm_fit, c(common, list(
    method = "spline-PH-mcmc",
    control = q3_ctrl(random_effects_method = "wishart_gibbs")))),
    "q = 2 random effects only")
})

test_that("the rotation is exact at q = 3 and q = 2: same log density at the same point", {
  skip_if_no_backend()
  if (!identical(jmjax:::.backend_precision(), "float64")) {
    skip("needs the float64 backend (jmjax_setup(enable_x64 = TRUE))")
  }
  mm <- jmjax:::.get_backend()$mcmc_model
  for (rf in list(~ time + I(time^2), ~ time)) {
    sim <- simulate_joint_data_q3(n = 80, seed = 6)
    fit <- jm_fit(
      long_formula = y ~ time + I(time^2) + x,
      surv_formula = survival::Surv(time, event) ~ 1,
      data_long = sim$data_long, data_surv = sim$data_surv,
      id_var = "id", time_var = "time", method = "spline-PH-mcmc",
      random_effects = "intercept_slope", random_formula = rf,
      control = q3_ctrl(num_warmup = 5, num_samples = 5,
                        return_backend_data = TRUE))
    bd <- fit$backend_data
    N <- dim(bd$X_long)[1]; p <- dim(bd$X_long)[3]; q <- dim(bd$Z_long)[3]

    jq <- reticulate::py_get_attr(mm, "_absorbable_basis_joint")(
      bd$X_long, bd$Z_long, as.integer(bd$n_obs), bd$X_time_surv,
      bd$Z_time_surv, bd$X_time_quad, bd$Z_time_quad)
    Q <- jq[[1]]
    expect_false(is.null(Q))
    V <- reticulate::py_get_attr(mm, "_householder_reflectors")(Q)

    # A random point: any point will do, since the two parameterizations
    # must agree everywhere.
    set.seed(10 + q)
    A <- matrix(stats::rnorm(q * q), q)
    Lc <- t(chol(stats::cov2cor(crossprod(A) + diag(q))))
    args <- list(bd$X_long, bd$y_long, as.integer(bd$n_obs), bd$X_time_surv,
                 bd$X_time_quad, bd$T_surv, bd$event, bd$t_quad, bd$gk_weights,
                 bd$B_T, bd$B_quad, as.integer(bd$n_splines),
                 bd$Z_long, bd$Z_time_surv, bd$Z_time_quad,
                 beta = stats::rnorm(p, 0, 0.5), sigma_e = 0.4, alpha = 0.3,
                 W = stats::rnorm(as.integer(bd$n_splines), -1, 0.3),
                 sigma_b = stats::runif(q, 0.2, 1),
                 b_std = matrix(stats::rnorm(N * q), N, q), L_corr = Lc)
    ld0 <- do.call(mm$evaluate_log_density, args)
    ld1 <- do.call(mm$evaluate_log_density, c(args, list(gen_reflectors = V)))
    expect_true(is.finite(ld0))
    expect_lt(abs(ld0 - ld1), 1e-10 * max(1, abs(ld0)),
              label = sprintf("q = %d: |%.12f - %.12f|", q, ld0, ld1))
  }
})

test_that("q = 3 recovers beta, the random-effects covariance and alpha", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_q3(n = 400, seed = 7)
  fit <- jm_fit(
    long_formula = y ~ time + I(time^2) + x,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", method = "spline-PH-mcmc",
    random_effects = "intercept_slope", random_formula = ~ time + I(time^2),
    control = list(num_warmup = 800, num_samples = 800, num_chains = 1,
                   progress_bar = FALSE, seed = 7))
  tr <- sim$truth
  est <- fit$estimates; se <- fit$se
  within <- function(nm, truth) {
    expect_lt(abs(est[[nm]] - truth), 3 * se[[nm]],
              label = sprintf("%s: %.4f vs truth %.4f (se %.4f)", nm, est[[nm]], truth, se[[nm]]))
  }
  for (k in 0:2) within(paste0("beta_", k), tr$beta[k + 1])
  within("beta_3", tr$beta_x)
  for (k in 0:2) within(paste0("sigma_b", k), tr$sd_b[k + 1])
  within("rho_0_1", tr$corr[1, 2]); within("rho_0_2", tr$corr[1, 3])
  within("rho_1_2", tr$corr[2, 3])
  within("alpha", tr$alpha)
  within("sigma_e", tr$sigma_e)
  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
})
