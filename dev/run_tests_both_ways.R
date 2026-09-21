# ==============================================================================
# Run the jmjax test suite both ways: fast (slow MCMC correctness tests
# skipped) and full (everything, as CI/a release should always run).
#
# Background: as of the run that motivated this script, the full suite
# took ~21 minutes (1272s), with 18 MCMC-based correctness tests -
# concentrated in test-functional-forms.R and test-baseline-covariates.R -
# accounting for roughly 80% of that time. These tests check correctness
# (R-hat health, truth recovery within 3 SE), not sampling speed, so they
# don't need their current heavy sampling budget to do their job -
# skip_if_slow_mcmc() (see tests/testthat/helper-python.R) lets them be
# skipped via an environment variable without changing what any test
# actually checks.
#
# Usage:
#   source("dev/run_tests_both_ways.R")
# or just run this file directly with Rscript/devtools::load_all() active.
#
# Day-to-day development: just run the FAST pass yourself
#   (Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "true"); devtools::test())
# Before a commit, PR, or release: always run the FULL pass (this script's
#   second half, or simply devtools::test() with the env var unset/false -
#   that's the default, so a bare devtools::test() call is always "full").
# ==============================================================================

pkg_path <- "."  # adjust if running from outside the package root

cat("############## FAST PASS (slow MCMC tests skipped) ##############\n")
Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "true")
t_fast <- system.time({
  fast_results <- devtools::test(pkg_path, reporter = "summary")
})
cat(sprintf("\nFast pass wall-clock time: %.1f sec\n", t_fast["elapsed"]))

cat("\n\n############## FULL PASS (everything, incl. slow MCMC) ##############\n")
Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "false")
t_full <- system.time({
  full_results <- devtools::test(pkg_path, reporter = "summary")
})
cat(sprintf("\nFull pass wall-clock time: %.1f sec\n", t_full["elapsed"]))

cat("\n\n================ SUMMARY ================\n")
cat(sprintf("Fast pass: %.1f sec\n", t_fast["elapsed"]))
cat(sprintf("Full pass: %.1f sec\n", t_full["elapsed"]))
cat(sprintf("Time saved by skipping slow tests: %.1f sec (%.1fx faster)\n",
            t_full["elapsed"] - t_fast["elapsed"],
            t_full["elapsed"] / t_fast["elapsed"]))
cat("\nReminder: the fast pass is for quick iteration only. Always run (or\n")
cat("confirm CI has run) the FULL pass - with JMJAX_SKIP_SLOW_TESTS unset\n")
cat("or 'false' - before a commit, PR, or release. A passing fast pass is\n")
cat("not a substitute for a passing full pass; the fast pass simply\n")
cat("doesn't exercise the MCMC-based correctness checks at all.\n")

# Restore the env var to the safe default (run everything) so a stray
# leftover "true" doesn't silently skip tests in a later, unrelated
# devtools::test() call in the same R session.
Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "false")
