# ==============================================================================
# Several longitudinal markers (jmjax 0.5.0; R/jm_fit_multi.R and
# inst/python/jmjax_backend/multi_model.py).
# ==============================================================================

mm_ctrl <- function(...) {
  utils::modifyList(list(num_warmup = 30, num_samples = 30, num_chains = 1,
                         progress_bar = FALSE, seed = 11), list(...))
}
mm_fit <- function(sim, ...) {
  jm_bayes(
    long_formula = list(y1 = y ~ time + age, y2 = y ~ time),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time", random_formula = ~ time,
    marker_var = "marker", ...)
}

test_that("multi-marker construction: union over markers, Lemma 1 under a full L (numpy)", {
  skip_if_no_backend()
  jmjax:::.get_backend()
  script <- normalizePath(testthat::test_path("python", "check_multi_marker_construction.py"))
  out <- suppressWarnings(system2(
    reticulate::py_exe(),
    c(shQuote(script), shQuote(system.file("python", package = "jmjax"))),
    stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (is.null(status)) status <- 0L
  out <- paste(out, collapse = "\n")
  expect_identical(as.integer(status), 0L, info = out)
  expect_match(out, "ALL PASS", fixed = TRUE)
})

test_that("two markers fit, with per-marker names, a 4 x 4 correlation and one rotation basis", {
  skip_if_no_backend()
  sim <- simulate_joint_data_2m(n = 120, seed = 1)
  fit <- mm_fit(sim, control = mm_ctrl())

  nm <- names(fit$estimates)
  expect_true(all(c("beta_y1_0", "beta_y1_1", "beta_y1_2", "beta_y2_0", "beta_y2_1",
                    "sigma_e_y1", "sigma_e_y2", "alpha_y1", "alpha_y2",
                    paste0("sigma_b", 0:3)) %in% nm))
  expect_true(all(c("rho_0_1", "rho_0_2", "rho_1_3", "rho_2_3") %in% nm))
  expect_identical(fit$model_info$beta_map$term,
                   c("(Intercept)", "time", "age", "(Intercept)", "time"))

  vc <- VarCorr(fit)
  expect_identical(names(vc$sd), c("y1:(Intercept)", "y1:time", "y2:(Intercept)", "y2:time"))
  expect_equal(dim(vc$corr), c(4L, 4L))

  # One basis for the union: y1 absorbs span{1, age}, y2 span{1} -> k = 2,
  # dense block p + k q = 5 + 2 * 4 = 13.
  rot <- fit$convergence$orthogonalize$rotation
  expect_true(isTRUE(rot$applied))
  expect_identical(as.integer(rot$k), 2L)
  expect_identical(as.integer(rot$dense_block_size), 13L)
  expect_lt(rot$condition_s_residual, 1e-6)

  s <- summary(fit)
  expect_true(all(s[c("beta_y1_0", "sigma_e_y2"), "Group"] == "Longitudinal"))
  expect_true(all(s[c("alpha_y1", "alpha_y2"), "Group"] == "Association"))
  expect_output(suppressWarnings(print(fit)), "alpha_y2")
  expect_true(all(c("b0_mean", "b3_mean") %in% names(ranef(fit))))
  expect_error(predict(fit, sim$data_long[sim$data_long$id == 1, ]),
               "not yet available")
  ci <- confint(fit)
  expect_true(all(is.finite(ci[c("alpha_y1", "beta_y2_1"), ])))
})

test_that("a list of data frames gives the same fit as one data frame with a marker column", {
  skip_if_no_backend()
  sim <- simulate_joint_data_2m(n = 80, seed = 2)
  f1 <- mm_fit(sim, control = mm_ctrl(num_warmup = 15, num_samples = 15))
  f2 <- jm_bayes(
    long_formula = list(y1 = y ~ time + age, y2 = y ~ time),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_list, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    random_formula = list(y1 = ~ time, y2 = ~ time),
    control = mm_ctrl(num_warmup = 15, num_samples = 15))
  expect_equal(f1$estimates, f2$estimates, tolerance = 1e-10)
})

test_that("a one-marker list gives the same fit as the plain formula", {
  skip_if_no_backend()
  sim <- simulate_joint_data_re2(n = 100, seed = 3)
  ctl <- mm_ctrl(num_warmup = 40, num_samples = 40)
  f_old <- jm_bayes(y ~ time, survival::Surv(time, event) ~ 1,
                    sim$data_long, sim$data_surv, id_var = "id", time_var = "time",
                    random_formula = ~ time, control = ctl)
  f_new <- jm_bayes(list(y = y ~ time), survival::Surv(time, event) ~ 1,
                    list(y = sim$data_long), sim$data_surv, id_var = "id",
                    time_var = "time", random_formula = list(y = ~ time), control = ctl)
  expect_equal(f_new$estimates, f_old$estimates, tolerance = 1e-12)
  expect_null(f_new$model_info$multi)
})

test_that("several markers are refused by the MLE methods", {
  sim <- list(data_long = data.frame(id = 1, time = 0, y = 0, marker = "y1"),
              data_surv = data.frame(id = 1, time = 1, event = 1))
  expect_error(
    jm_mle(list(y1 = y ~ time, y2 = y ~ time), survival::Surv(time, event) ~ 1,
           sim$data_long, sim$data_surv, id_var = "id", time_var = "time",
           marker_var = "marker"),
    "jm_mle\\(\\) fits one longitudinal marker")
})

test_that("re_structure = 'block' gives independent markers", {
  skip_if_no_backend()
  sim <- simulate_joint_data_2m(n = 100, seed = 4)
  fit <- mm_fit(sim, control = mm_ctrl(re_structure = "block"))
  nm <- names(fit$estimates)
  expect_true(all(c("rho_0_1", "rho_2_3") %in% nm))
  expect_false(any(c("rho_0_2", "rho_0_3", "rho_1_2", "rho_1_3") %in% nm))
  vc <- VarCorr(fit)
  expect_equal(unname(vc$corr[1:2, 3:4]), matrix(0, 2, 2))
  expect_identical(fit$model_info$re_structure, "block")
})

test_that("the multi-marker rotation is exact (full and block covariance)", {
  skip_if_no_backend()
  if (!identical(jmjax:::.backend_precision(), "float64")) {
    skip("needs the float64 backend")
  }
  be <- jmjax:::.get_backend()
  ab <- reticulate::py_get_attr(be$multi_model, "_absorbable_basis_multi")
  hr <- reticulate::py_get_attr(be$multi_model, "_householder_reflectors")
  sim <- simulate_joint_data_2m(n = 60, seed = 5)
  fit <- mm_fit(sim, control = mm_ctrl(num_warmup = 5, num_samples = 5,
                                       return_backend_data = TRUE))
  a <- fit$backend_data
  N <- length(a$T_surv); P <- ncol(a$X); Q <- ncol(a$Z)
  Qb <- ab(a$X, a$Z, a$subj, N, a$XS, a$ZS, a$XQ, a$ZQ)[[1]]
  V <- hr(Qb)
  set.seed(9)
  for (st in c("full", "block")) {
    A <- matrix(stats::rnorm(Q * Q), Q)
    R <- stats::cov2cor(crossprod(A) + diag(Q))
    if (st == "block") { R[1:2, 3:4] <- 0; R[3:4, 1:2] <- 0 }
    args <- list(a$y, a$X, a$Z, a$subj, a$marker, a$XS, a$ZS, a$XQ, a$ZQ,
                 a$T_surv, a$event, a$t_quad, a$gk_weights, a$B_T, a$B_quad,
                 a$n_splines, a$block_sizes,
                 beta = stats::rnorm(P, 0, 0.3), sigma_e = c(0.4, 0.6),
                 alpha = c(0.3, -0.2), W = stats::rnorm(a$n_splines, -1, 0.3),
                 sigma_b = stats::runif(Q, 0.2, 0.8), L_corr = t(chol(R)),
                 b_std = matrix(stats::rnorm(N * Q), N, Q), re_structure = st)
    ld0 <- do.call(be$multi_model$evaluate_log_density_multi, args)
    ld1 <- do.call(be$multi_model$evaluate_log_density_multi,
                   c(args, list(gen_reflectors = V)))
    expect_true(is.finite(ld0))
    expect_lt(abs(ld0 - ld1), 1e-10 * max(1, abs(ld0)),
              label = sprintf("%s: |%.12f - %.12f|", st, ld0, ld1))
  }
})

test_that("two markers: recovery of beta, sigma_e, alpha and the cross-marker covariance", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data_2m(n = 400, seed = 6)
  fit <- mm_fit(sim, control = list(num_warmup = 800, num_samples = 800,
                                    num_chains = 1, progress_bar = FALSE, seed = 6))
  tr <- sim$truth; est <- fit$estimates; se <- fit$se
  within <- function(nm, truth) {
    expect_lt(abs(est[[nm]] - truth), 3 * se[[nm]],
              label = sprintf("%s: %.4f vs truth %.4f (se %.4f)", nm, est[[nm]], truth, se[[nm]]))
  }
  for (j in 0:2) within(paste0("beta_y1_", j), tr$beta1[j + 1])
  for (j in 0:1) within(paste0("beta_y2_", j), tr$beta2[j + 1])
  within("sigma_e_y1", tr$sigma_e[1]); within("sigma_e_y2", tr$sigma_e[2])
  within("alpha_y1", tr$alpha[1]); within("alpha_y2", tr$alpha[2])
  for (j in 0:3) within(paste0("sigma_b", j), tr$sd_b[j + 1])
  within("rho_0_2", tr$corr[1, 3])   # y1 intercept with y2 intercept
  within("rho_1_3", tr$corr[2, 4])   # y1 slope with y2 slope
  expect_lt(max(unlist(fit$diagnostics$rhat), na.rm = TRUE), 1.1)
})
