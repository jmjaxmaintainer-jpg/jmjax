# ==============================================================================
# Skip helper for Python-dependent tests.
#
# CI environments (or a fresh contributor's machine) may not have run
# jmjax_setup() yet, and installing a virtualenv + jax/numpyro during a test
# run is slow and network-dependent. Python-dependent tests use
# skip_if_no_backend() so the pure-R unit tests (design matrices, knots)
# always run, while the integration tests degrade to "skipped" rather than
# "failed" when the backend isn't available - a missing environment isn't a
# code regression.
# ==============================================================================

# Returns TRUE, or the error message explaining why not.
#
# Reporting the MESSAGE rather than a bare FALSE is deliberate. An earlier
# version returned FALSE and discarded the condition, so a session in which
# every backend test skipped said only "Python backend not available" - which
# reads like "you never ran jmjax_setup()" and is the one explanation that
# was not true, the venv having been built by that very run. The real error
# ("failed to initialize requested version of Python", from reticulate
# refusing to re-point an already-attached interpreter) was thrown away 79
# times. Whatever goes wrong next, the reason should be in the test log.
#
# Cached because .get_backend() re-runs its whole discovery path on every
# call while the import is failing, and that includes a subprocess probe.
.backend_state <- new.env(parent = emptyenv())

backend_available <- function() {
  if (is.null(.backend_state$result)) {
    .backend_state$result <- tryCatch({
      jmjax:::.get_backend()
      TRUE
    }, error = function(e) conditionMessage(e))
  }
  .backend_state$result
}

skip_if_no_backend <- function() {
  testthat::skip_if_not_installed("reticulate")
  state <- backend_available()
  if (!isTRUE(state)) {
    testthat::skip(paste0(
      "Python backend not available (run jmjax::jmjax_setup() to enable ",
      "this test). Reason: ", state))
  }
}

# Cross-package comparison tests. These fit the SAME model with another
# package and compare - validation work, not correctness checks for the
# package itself, and among the most expensive things in the suite.
#
# skip_on_cran() matters here for a reason a local check cannot show you:
# on this machine JM and JMbayes2 are typically absent, so these skip and
# look free. On CRAN both ARE installed, so without this guard they would
# run in full during a check that is supposed to finish inside ten minutes.
skip_if_no_JM <- function() {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("JM")
  testthat::skip_if_not_installed("nlme")
  testthat::skip_if_not_installed("survival")
}

skip_if_no_JMbayes2 <- function() {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("JMbayes2")
  testthat::skip_if_not_installed("nlme")
  testthat::skip_if_not_installed("survival")
}

# ==============================================================================
# Skip helper for slow, MCMC-based correctness tests.
#
# The full suite's ~21 minute runtime (as of this writing) is dominated by
# 18 MCMC test_that() blocks - functional-forms.R and
# baseline-covariates.R alone account for roughly 80% of total runtime
# (766.8s and 266.5s of 1272.3s in the run that motivated this helper).
# These tests check CORRECTNESS (R-hat health, truth recovery within 3 SE)
# using a heavy, production-grade sampling budget
# (num_warmup=500/num_samples=1000) that a correctness check doesn't
# actually need - they were never intended as speed benchmarks.
#
# Default behavior (JMJAX_SKIP_SLOW_TESTS unset or "false"): every test
# runs, exactly as before this helper was added - safe for CI and for any
# workflow that doesn't opt in. Set JMJAX_SKIP_SLOW_TESTS=true for a fast
# iteration loop during day-to-day development; run the full suite
# (unset, or explicitly "false") before a commit, PR, or release.
#
# Usage in a test file, right after skip_if_no_backend():
#   skip_if_slow_mcmc()
#
# To run fast-only from the R console:
#   Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "true"); devtools::test()
# To run the full suite (the default - equivalent to not setting this at all):
#   Sys.setenv(JMJAX_SKIP_SLOW_TESTS = "false"); devtools::test()
# ==============================================================================
skip_if_slow_mcmc <- function() {
  # TIER 1 - CRAN. R CMD check should finish well inside ten minutes on
  # CRAN's hardware, and this suite's MCMC blocks alone took 482s on a
  # 2026 Apple laptop. skip_on_cran() uses testthat's NOT_CRAN convention:
  # devtools::test() and r-lib's CI actions set it to "true", CRAN does
  # not, so local runs are completely unaffected.
  #
  # Mostly belt and braces. The primary guard is in .get_backend(), which
  # refuses to CREATE a Python environment in a non-interactive session
  # with NOT_CRAN unset - so on CRAN every backend test skips for want of
  # jax rather than trying to download it. This tier covers the remaining
  # case: a maintainer running R CMD check on their own machine, where the
  # r-jmjax venv already exists and the backend therefore works.
  testthat::skip_on_cran()

  # TIER 2 - fast local iteration, and CI pull requests (the workflow sets
  # this from github.event_name). Unset or "false" runs everything.
  if (tolower(Sys.getenv("JMJAX_SKIP_SLOW_TESTS", "false")) == "true") {
    testthat::skip("Slow MCMC test skipped (JMJAX_SKIP_SLOW_TESTS=true) - run the full suite before committing/releasing")
  }

  # TIER 3 - the default everywhere else: run it.
}
