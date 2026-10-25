# ==============================================================================
# The absorbable subspace defined jointly across random-effect columns
# (inst/python/jmjax_backend/absorbable.py, _absorbable_basis_joint; methods
# paper, revised Section 3). From 0.4.0 the rotation (control$rotate_absorbable)
# is built on it instead of the union of per-column bases.
#
# The algebra is checked in Python against the paper's reference
# implementation (tests/testthat/python/check_joint_construction.py, numpy
# only): the coupled example, a baseline-only subject, a rank-deficient Z_i,
# a Condition (S) failure, q = 3, column-space invariance, and 80 randomized
# designs. The fits below check that the backend uses it and reports it.
# ==============================================================================

test_that("joint absorbable construction passes the reference checks", {
  skip_if_no_backend()
  jmjax:::.get_backend()
  # Run the numpy-only check in its own process, with the interpreter the
  # backend uses, so its module-level names and sys.argv stay its own.
  script <- normalizePath(testthat::test_path("python", "check_joint_construction.py"))
  out <- suppressWarnings(system2(
    reticulate::py_exe(),
    c(shQuote(script), shQuote(system.file("python", package = "jmjax"))),
    stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (is.null(status)) status <- 0L
  out <- paste(out, collapse = "\n")
  expect_identical(as.integer(status), 0L, info = out)
  expect_match(out, "ALL PASS", fixed = TRUE)
  expect_false(grepl("FAIL", out, fixed = TRUE), info = out)
})

coupled_data <- function(n = 120, seed = 31) {
  sim <- simulate_joint_data_re2(n = n, seed = seed)
  set.seed(seed + 1)
  x <- stats::rnorm(n)
  sim$data_long$x <- x[sim$data_long$id]
  sim
}

test_that("the rotation uses the joint construction (coupled design: k = 2)", {
  skip_if_no_backend()

  # X_i = [1, t, x_i (1 + t)], Z_i = [1, t]: column by column only span{1}
  # is absorbable, jointly span{1, x} (beta_x + c, b_i0 - c x_i,
  # b_i1 - c x_i leaves every trajectory unchanged).
  sim <- coupled_data()
  fit <- jm_fit(
    long_formula = y ~ time + I(x * (1 + time)),
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time,
    control = list(num_warmup = 20, num_samples = 20, num_chains = 1,
                   progress_bar = FALSE,
                   # keep the design exactly as written: time rescaling would turn
                   # x(1 + time) into x(1 + time/c), which standardization cannot
                   # back-transform
                   scale_time = FALSE, standardize_covariates = FALSE))
  rot <- fit$convergence$orthogonalize$rotation
  expect_identical(rot$construction, "joint")
  expect_true(isTRUE(rot$applied))
  expect_identical(as.integer(rot$k), 2L)
  expect_identical(as.integer(rot$k_columnwise), 1L)
  expect_lt(rot$condition_s_residual, 1e-6)
  expect_identical(as.integer(rot$n_rejected_condition_s), 0L)
})

test_that("on a separate-column design the joint and per-column subspaces agree", {
  skip_if_no_backend()

  # Intercept, time and a subject-constant covariate; random intercept and
  # slope; subjects with an early event are observed only at baseline. This
  # is the kind of design every fit in the methods paper uses, so the
  # subspace must be the one the per-column construction gave.
  sim <- coupled_data(n = 150, seed = 32)
  expect_true(any(table(sim$data_long$id) == 1L))
  fit <- jm_fit(
    long_formula = y ~ time + x,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time,
    control = list(num_warmup = 20, num_samples = 20, num_chains = 1,
                   progress_bar = FALSE))
  rot <- fit$convergence$orthogonalize$rotation
  expect_identical(as.integer(rot$k), 2L)
  expect_identical(as.integer(rot$k_columnwise), 2L)
  expect_lt(rot$columnwise_subspace_distance, 1e-8)
})
