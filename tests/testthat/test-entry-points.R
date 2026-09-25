# jm_mle() and jm_bayes(): front ends to jm_fit(). What must hold is that
# they are ONLY front ends - the same fit as the jm_fit() method they map
# to - plus their own argument handling.

args_for <- function(sim) list(
  long_formula = y ~ time, surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim$data_long, data_surv = sim$data_surv,
  id_var = "id", time_var = "time")

test_that("jm_mle() is jm_fit() with the matching MLE method", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 120, seed = 8)
  a <- args_for(sim)
  f_new <- do.call(jm_mle, a)
  f_old <- do.call(jm_fit, c(a, method = "weibull-PH-aGH"))
  expect_identical(f_new$method, "weibull-PH-aGH")
  expect_equal(coef(f_new), coef(f_old), tolerance = 1e-10)
  expect_equal(vcov(f_new), vcov(f_old), tolerance = 1e-10)
  expect_identical(as.character(f_new$call[[1]]), "jm_mle")

  f_sp <- do.call(jm_mle, c(a, baseline = "spline"))
  expect_identical(f_sp$method, "spline-PH-aGH")

  # Warm start of a spline fit from Weibull estimates: those have no spline
  # coefficients, which used to leave init_theta too short (a backend
  # shape error) instead of starting the coefficients at zero.
  f_ws <- do.call(jm_mle, c(a, list(baseline = "spline",
                                    init_theta = f_new$estimates)))
  expect_true(is.finite(f_ws$loglik))
})

test_that("jm_mle() warns about, and ignores, Bayesian-only control options", {
  skip_if_no_backend()
  sim <- simulate_joint_data(n = 80, seed = 9)
  expect_warning(do.call(jm_mle, c(args_for(sim),
                                   list(control = list(num_chains = 4)))),
                 "apply only to jm_bayes")
})

test_that("jm_bayes() argument handling needs no backend", {
  # A setting given both as an argument and in control, with different
  # values, is refused before anything is fitted.
  expect_error(jm_bayes(y ~ time, survival::Surv(time, event) ~ 1,
                        data_long = NULL, data_surv = NULL, id_var = "id",
                        time_var = "time", chains = 2,
                        control = list(num_chains = 4)),
               "both given")
  expect_warning(.jmjax_warn_foreign_controls(list(opt_method = "BFGS"),
                                              .jmjax_mle_only_controls,
                                              "jm_bayes", "jm_mle"),
                 "apply only to jm_mle")
  expect_silent(.jmjax_warn_foreign_controls(list(n_interior_knots = 5),
                                             .jmjax_mle_only_controls,
                                             "jm_bayes", "jm_mle"))
})

test_that("jm_bayes() is jm_fit() with the matching MCMC method", {
  skip_if_no_backend()
  skip_if_slow_mcmc()
  sim <- simulate_joint_data(n = 100, seed = 10)
  a <- args_for(sim)
  f_new <- do.call(jm_bayes, c(a, list(chains = 1, warmup = 150, samples = 150,
                                      seed = 5, control = list(n_interior_knots = 4L))))
  f_old <- do.call(jm_fit, c(a, list(method = "spline-PH-mcmc",
                                     control = list(num_chains = 1L, num_warmup = 150L,
                                                    num_samples = 150L, seed = 5L,
                                                    n_interior_knots = 4L))))
  expect_identical(f_new$method, "spline-PH-mcmc")
  # Same seed, same inputs: the same draws.
  expect_equal(coef(f_new), coef(f_old), tolerance = 1e-8)
  expect_identical(f_new$mcmc_settings$num_warmup, 150L)

  # An argument left at its default gives way to the same setting in
  # control, so older code passing control$num_warmup keeps working.
  f_ctl <- do.call(jm_bayes, c(a, list(chains = 1, samples = 150, seed = 5,
                                      control = list(num_warmup = 120L,
                                                     n_interior_knots = 4L))))
  expect_identical(f_ctl$mcmc_settings$num_warmup, 120L)
})

test_that("jm_fit_prefit() is internal; jm_mle(), jm_bayes() and jm_fit() are exported", {
  # Read NAMESPACE itself: under devtools::load_all() every function is
  # exported, so getNamespaceExports() cannot tell.
  ns <- readLines(system.file("NAMESPACE", package = "jmjax"))
  expect_false("export(jm_fit_prefit)" %in% ns)
  expect_true(all(c("export(jm_mle)", "export(jm_bayes)", "export(jm_fit)") %in% ns))
  expect_true(is.function(jmjax:::jm_fit_prefit))
})
