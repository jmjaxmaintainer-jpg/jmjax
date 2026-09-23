# ==============================================================================
# Verification for the print/summary "polish" follow-up to the rework in
# dev/verify_print_summary_rework.R (already committed as 40cdbed). Five
# changes, all confined to R/summary.jmjax.R:
#
#   1. p-values are formatted with format.pval() at print time (digits=3,
#      eps=1e-4, scientific=FALSE) - a very small p-value now reads
#      "<0.0001" instead of a bare "0" or "0.0000". Applied in
#      .jmjax_print_group_table(), NOT stored back into summary()'s
#      returned data.frame (p.value stays numeric there).
#   2. summary()'s columns are now ADDITIVE, not swapped: MLE fits are
#      unchanged (Estimate/Std.Err/z.value/p.value/Group); MCMC fits keep
#      those same columns AND gain CrI.lower/CrI.upper (95% credible
#      interval from posterior_samples, already on the original covariate
#      scale) and Rhat/ESS (per-parameter, from $diagnostics). print()
#      (both print.jmjax's compact view and print.summary.jmjax) shows
#      only one set per fit via the internal .jmjax_display_cols() - the
#      other set stays in summary(fit) either way.
#      (Revised from an earlier version of this change that DROPPED
#      z.value/p.value for MCMC fits - the manuscript-thread review
#      flagged that as unsafe for any user code reading those columns
#      unconditionally, per the original verify_print_summary_rework.R's
#      own backward-compatibility check.)
#      (Revised AGAIN: the first CrI implementation looked up
#      object$posterior_samples[[p_nm]] directly using per-element names
#      like "beta_1"/"W3"/"sigma_b1" - but posterior_samples is keyed by
#      the backend's raw NumPyro SITE name ("beta", "W", "sigma_b", one
#      entry per site holding every element's draws), not the expanded
#      per-element names _site_names() uses for estimates/se/diagnostics.
#      That lookup returned NULL for every vector-site element, silently
#      falling back to a Wald normal approximation for most of the table -
#      only the scalar sites (alpha, sigma_e, tau_w, rho, sigma_b at q=1)
#      ever got a real credible interval. The original verify script's
#      `!anyNA(s_mcmc$CrI.lower)` check didn't catch this because the Wald
#      fallback fills every row with a plausible-looking, non-NA number.
#      Fixed via the new .jmjax_param_draws() helper, which maps a
#      per-element name back to its site + 0-based column index; the
#      fallback for a genuine lookup miss is now NA, not Wald, so a future
#      mismatch is visible instead of silently disguised - see section 3b
#      below for a check that independently recomputes two of these
#      columns straight from posterior_samples and confirms they match.)
#   3. print(fit)'s compact "Estimates by submodel" view collapses the
#      baseline-hazard block to one line whenever it has more than 3 rows
#      (every row in that group is already a baseline-hazard parameter by
#      construction, so more than 3 only happens for a spline basis) -
#      naming the W-coefficient range and listing anything else (tau_w,
#      say) alongside it. summary(fit) still has the full per-coefficient
#      table.
#      (Revised from an earlier version that required EVERY row in the
#      group to match "^W[0-9]+$" - which meant a spline basis with tau_w
#      riding along in the same group wouldn't collapse at all. Fixed
#      here and covered by its own synthetic test in section 4b below,
#      since none of the real fits built in this script happen to
#      populate a tau_w estimate to exercise the mixed case.)
#   4. print(fit) no longer prints "Log-Likelihood: not reported for
#      method '...'" for MCMC fits - the line is omitted entirely rather
#      than printing a placeholder.
#   5. print(fit)'s compact view drops the one-row Association table
#      (redundant with the alpha headline above it); summary(fit) still
#      includes it.
#
# Written without access to R/JAX (this session has neither), so it has
# NOT been executed - only read carefully against the fit object's actual
# field structure, the existing test suite, and dev/verify_print_summary_
# rework.R's own real-fit smoke tests. Run this before anything here gets
# committed. Also re-run (per the manuscript-thread review):
#
#   Rscript -e 'devtools::test(filter = "rotate-absorbable|scale-time|mcmc|prefit-interface")'
#   Rscript dev/verify_print_summary_rework.R
#
# ...and confirm verify_print_summary_rework.R's section 2 still prints
# "OK: summary(fit_mle) backward-compatible check passed."
#
# Usage:
#   Rscript dev/verify_print_summary_polish.R
# or, interactively:
#   devtools::load_all("."); source("dev/verify_print_summary_polish.R")
# ==============================================================================

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("devtools is required to run this check (install.packages('devtools')).")
}

devtools::document(quiet = FALSE)
devtools::load_all(".", quiet = FALSE)  # exposes internal .jmjax_* helpers too

cat("\n=== 1. Existing test suite - nothing here should have broken ===\n")
Sys.setenv(NOT_CRAN = "true")
res <- testthat::test_dir("tests/testthat", stop_on_failure = FALSE)

cat("\n=== 2. Real MLE fit: summary() columns/format unchanged ===\n")
set.seed(1)
sim <- simulate_joint_data_re2(n = 150, seed = 21)
fit_mle <- jm_fit(
  long_formula = y ~ time,
  surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim$data_long, data_surv = sim$data_surv,
  id_var = "id", time_var = "time",
  method = "spline-PH-aGH"
)
s_mle <- summary(fit_mle)
stopifnot(is.data.frame(s_mle))
stopifnot(all(c("Estimate", "Std.Err", "z.value", "p.value", "Group") %in% names(s_mle)))
stopifnot(!any(c("CrI.lower", "CrI.upper", "Rhat", "ESS") %in% names(s_mle)))
stopifnot(identical(.jmjax_display_cols(s_mle), c("Estimate", "Std.Err", "z.value", "p.value")))
cat("OK: summary(fit_mle) still has exactly the original MLE columns.\n")

mle_summary_out <- paste(utils::capture.output(print(s_mle)), collapse = "\n")
stopifnot(grepl("Association", mle_summary_out))
cat("OK: summary(fit_mle) still prints the Association table (item 5 only removes it from print(fit), not summary(fit)).\n")

cat("\n=== 3. Real MCMC fit: columns are additive, print() picks one set ===\n")
fit_mcmc <- jm_fit(
  long_formula = y ~ time,
  surv_formula = survival::Surv(time, event) ~ 1,
  data_long = sim$data_long, data_surv = sim$data_surv,
  id_var = "id", time_var = "time",
  method = "spline-PH-mcmc", random_effects = "intercept_slope",
  control = list(num_warmup = 300, num_samples = 500, num_chains = 1,
                 progress_bar = FALSE)
)

s_mcmc <- summary(fit_mcmc)
stopifnot(is.data.frame(s_mcmc))
stopifnot(all(c("Estimate", "Std.Err", "z.value", "p.value",
                "CrI.lower", "CrI.upper", "Rhat", "ESS", "Group") %in% names(s_mcmc)))
stopifnot(!anyNA(s_mcmc$CrI.lower) && !anyNA(s_mcmc$CrI.upper))
stopifnot(identical(.jmjax_display_cols(s_mcmc),
                     c("Estimate", "Std.Err", "CrI.lower", "CrI.upper", "Rhat", "ESS")))
cat("OK: summary(fit_mcmc) keeps z.value/p.value AND adds CrI.lower/CrI.upper/Rhat/ESS (item 2, additive).\n")

cat("\n=== 3b. CrI.lower/CrI.upper actually come from the right posterior_samples column ===\n")
# The real test for the site-name-mapping bugfix: independently rebuild
# the credible interval straight from posterior_samples (NOT via
# .jmjax_param_draws() - that would just re-run the same code under test)
# and confirm it matches what summary() reported. beta_1 exercises the
# "beta_<i>" naming (element 1, 0-based -> column 2 of the beta matrix);
# W3 exercises the "W<i>" naming (element 3, 0-based -> column 4).
# !anyNA() above already passed even with the OLD, buggy lookup, because
# its Wald fallback fills every row with a plausible non-NA number - this
# is the check that actually would have caught that bug.
.beta_mat <- do.call(rbind, lapply(fit_mcmc$posterior_samples$beta,
                                    function(z) as.numeric(unlist(z))))
.beta1_q <- stats::quantile(.beta_mat[, 2], c(0.025, 0.975), names = FALSE)
stopifnot(isTRUE(all.equal(s_mcmc["beta_1", "CrI.lower"], .beta1_q[1])))
stopifnot(isTRUE(all.equal(s_mcmc["beta_1", "CrI.upper"], .beta1_q[2])))

.w_mat <- do.call(rbind, lapply(fit_mcmc$posterior_samples$W,
                                 function(z) as.numeric(unlist(z))))
.w3_q <- stats::quantile(.w_mat[, 4], c(0.025, 0.975), names = FALSE)
stopifnot(isTRUE(all.equal(s_mcmc["W3", "CrI.lower"], .w3_q[1])))
stopifnot(isTRUE(all.equal(s_mcmc["W3", "CrI.upper"], .w3_q[2])))
cat("OK: CrI.lower/CrI.upper for beta_1 and W3 match independently-recomputed quantiles of the\n")
cat("    correct posterior_samples column - the site-name mapping fix is working.\n")

mcmc_print_out <- paste(utils::capture.output(print(fit_mcmc)), collapse = "\n")

stopifnot(!grepl("Log-Likelihood", mcmc_print_out))
cat("OK: print(fit_mcmc) has no Log-Likelihood line at all (item 4).\n")

# The alpha headline block ("Association (alpha): effect of the ...") should
# still be there - only the redundant one-row *table* under "Estimates by
# submodel" is removed. Check specifically for the table's own group header
# ("Association:" on its own line, as .print_estimates_compact writes it),
# not the headline sentence.
stopifnot(grepl("Association \\(alpha\\): effect", mcmc_print_out))
stopifnot(!grepl("\nAssociation:\n", mcmc_print_out))
cat("OK: print(fit_mcmc) keeps the alpha headline but drops the Association table (item 5).\n")

# print(fit_mcmc)'s compact view should show the credible-interval columns,
# not a z.value/p.value table, for whichever group prints in full (e.g.
# Longitudinal).
stopifnot(grepl("CrI.lower", mcmc_print_out) || grepl("CrI\\.lower", mcmc_print_out))
cat("OK: print(fit_mcmc)'s compact tables show the credible-interval columns.\n")

cat("\n=== 4a. Baseline-hazard collapse on the real MCMC fit ===\n")
stopifnot(grepl("spline coefficients", mcmc_print_out))
stopifnot(!grepl("\nW0 ", mcmc_print_out))
mcmc_summary_out <- paste(utils::capture.output(print(s_mcmc)), collapse = "\n")
stopifnot(grepl("\nW0", mcmc_summary_out))
cat("OK: print(fit_mcmc) collapses the baseline-hazard spline table (item 3); summary(fit_mcmc) still has it in full.\n")

cat("\n=== 4b. Baseline-hazard collapse when tau_w rides along with the W's (synthetic) ===\n")
# None of the real fits above are guaranteed to populate a tau_w estimate,
# so this is the one place the tau_w-specific fix from item 3's revision
# actually gets exercised: a spline basis (W0..W8) plus tau_w in the same
# "Survival (baseline hazard)" group must still collapse, tau_w named
# alongside the W-range rather than the whole table skipping collapse.
fake_spline_fit <- structure(
  list(call = quote(jm_fit()), method = "spline-PH-aGH",
       estimates = c(setNames(rep(-1, 9), paste0("W", 0:8)), tau_w = 0.5),
       se = c(setNames(rep(0.1, 9), paste0("W", 0:8)), tau_w = 0.05),
       loglik = -100,
       convergence = list(converged = TRUE, message = "ok", precision = NA),
       n_subjects = 10L, n_obs_long = 40L, n_events = 5L),
  class = "jmjax")
fake_spline_out <- paste(utils::capture.output(print(fake_spline_fit)), collapse = "\n")
stopifnot(grepl("spline coefficients \\(W0-W8\\), tau_w; see summary\\(fit\\)", fake_spline_out))
stopifnot(!grepl("\nW0 ", fake_spline_out))
cat("OK: a spline basis with tau_w mixed into the same group still collapses, naming both.\n")

cat("\n=== 5. p-value formatting (item 1) ===\n")
# Exercise format.pval() directly against a synthetic summary.jmjax object
# spanning a tiny, a middling, and a NA p-value, independent of what the
# real fits above happened to produce.
fake <- structure(
  data.frame(
    Estimate = c(1, 2), Std.Err = c(0.1, 0.2),
    z.value = c(10, 1.5), p.value = c(1e-15, 0.13),
    Group = c("Longitudinal", "Longitudinal"),
    row.names = c("beta_0", "beta_1")
  ),
  class = c("summary.jmjax", "data.frame")
)
fake_out <- paste(utils::capture.output(print(fake)), collapse = "\n")
stopifnot(grepl("<0.0001", fake_out))
stopifnot(grepl("0.13", fake_out))
cat("OK: a very small p-value prints as \"<0.0001\"; an ordinary one prints as a plain number.\n")
cat("--- fake print output ---\n")
cat(fake_out, "\n")

cat("\n=== 6. Backward-compat: hand-built minimal fit object still prints ===\n")
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
  stopifnot(!grepl("Log-Likelihood", out))  # loglik is NULL, not numeric(1) -> no line at all now
}
old_s <- summary(old_fit)
stopifnot(all(c("z.value", "p.value") %in% names(old_s)))
stopifnot(!any(c("CrI.lower", "Rhat", "ESS") %in% names(old_s)))
cat("OK: minimal/legacy fit object (no posterior_samples/diagnostics/mcmc_settings) still prints and summarizes on the MLE code path.\n")

cat("\n=== 7. Recursion guard: print(summary(fit)) must not stack-overflow, both shapes ===\n")
invisible(capture.output(print(summary(fit_mle))))
invisible(capture.output(print(summary(fit_mcmc))))
cat("OK: no recursion for either the MLE or MCMC summary shape.\n")

cat("\n=== Done. If everything above says OK and the test suite is green, ===\n")
cat("=== this is ready to commit.                                        ===\n")
