# ==============================================================================
# Verification for tonight's changes: print.jmjax/summary.jmjax rework,
# the new print.summary.jmjax method, and the small additive mcmc_settings
# field on jm_fit()'s return object.
#
# Written without access to R/JAX (this session has neither), so it has NOT
# been executed - only read carefully against the fit object's actual field
# structure and the existing test suite. Run this before anything here gets
# committed.
#
# Usage:
#   Rscript dev/verify_print_summary_rework.R
# or, interactively:
#   devtools::load_all("."); source("dev/verify_print_summary_rework.R")
# ==============================================================================

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("devtools is required to run this check (install.packages('devtools')).")
}

devtools::document(quiet = FALSE)   # regenerates NAMESPACE/man/*.Rd from roxygen
                                     # comments - confirms the hand-edited
                                     # NAMESPACE line (S3method(print,summary.jmjax))
                                     # matches what roxygen2 itself would produce,
                                     # and creates man/print.summary.jmjax.Rd,
                                     # which was NOT hand-written tonight.
devtools::load_all(".", quiet = FALSE)

cat("\n=== 1. Existing test suite - nothing here should have broken ===\n")
Sys.setenv(NOT_CRAN = "true")
res <- testthat::test_dir("tests/testthat", stop_on_failure = FALSE)

cat("\n=== 2. print()/summary() smoke test against a REAL MLE fit ===\n")
set.seed(1)
sim <- simulate_joint_data_re2(n = 150, seed = 21)
fit_mle <- jm_fit(
  long_formula = y ~ time,
  surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim$data_long, data_surv = sim$data_surv,
  id_var = "id", time_var = "time",
  method = "spline-PH-aGH"
)
print(fit_mle)
cat("\n--- summary(fit_mle) ---\n")
print(summary(fit_mle))
cat("\n--- summary(fit_mle) is still a data.frame with the original columns ---\n")
stopifnot(is.data.frame(summary(fit_mle)))
stopifnot(all(c("Estimate", "Std.Err", "z.value", "p.value") %in% names(summary(fit_mle))))
cat("OK: summary(fit_mle) backward-compatible check passed.\n")

cat("\n=== 3. print()/summary() smoke test against a REAL MCMC fit ===\n")
cat("    (uses whatever control$rotate_absorbable currently defaults to,\n")
cat("     so this also exercises the new Rotation line if it applies here)\n")
fit_mcmc <- jm_fit(
  long_formula = y ~ time,
  surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim$data_long, data_surv = sim$data_surv,
  id_var = "id", time_var = "time",
  method = "spline-PH-mcmc", random_effects = "intercept_slope",
  control = list(num_warmup = 300, num_samples = 500, num_chains = 1,
                 progress_bar = FALSE)
)
print(fit_mcmc)
cat("\n--- summary(fit_mcmc) ---\n")
print(summary(fit_mcmc))

cat("\n--- fit_mcmc$mcmc_settings (new field) ---\n")
str(fit_mcmc$mcmc_settings)
stopifnot(!is.null(fit_mcmc$mcmc_settings))
stopifnot(isTRUE(fit_mcmc$mcmc_settings$num_warmup == 300))  # stored as integer since 0.3.0
stopifnot(isTRUE(fit_mcmc$mcmc_settings$num_samples == 500))
cat("OK: mcmc_settings populated as expected.\n")

cat("\n--- fit_mle$mcmc_settings should be NULL (MLE method) ---\n")
stopifnot(is.null(fit_mle$mcmc_settings))
cat("OK.\n")

cat("\n=== 4. Backward-compat: a hand-built minimal fit object must still print ===\n")
old_fit <- structure(
  list(call = quote(jm_fit()), method = "spline-PH-mcmc",
       estimates = c(beta_0 = 1), se = c(beta_0 = 0.1),
       loglik = NULL,
       convergence = list(converged = TRUE, message = "ok"),
       n_subjects = 10L, n_obs_long = 40L, n_events = 5L),
  class = "jmjax")
out <- tryCatch({
  paste(utils::capture.output(print(old_fit)), collapse = "\n")
}, error = function(e) {
  cat("FAILED: print(old_fit) errored:", conditionMessage(e), "\n")
  NULL
})
if (!is.null(out)) {
  stopifnot(!grepl("Precision", out))
  cat("OK: minimal/legacy fit object still prints without error.\n")
}

cat("\n=== 5. Recursion guard: print(summary(fit)) must not stack-overflow ===\n")
# The real risk this checks: [.data.frame subsetting inside
# print.summary.jmjax retaining the "summary.jmjax" class on the subset,
# which would make print() on that subset call print.summary.jmjax again.
# Should return normally, not hang or error with "C stack usage too close
# to the limit".
invisible(capture.output(print(summary(fit_mcmc))))
cat("OK: no recursion.\n")

cat("\n=== Done. If everything above says OK and the test suite is green, ===\n")
cat("=== this is ready to commit.                                        ===\n")
