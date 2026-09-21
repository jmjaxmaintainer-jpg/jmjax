# ==============================================================================
# Coverage for floating-point precision.
#
# WHY THIS FILE EXISTS. jmjax ran in float32 for its entire benchmarked
# history without a single line of code choosing that, and without any saved
# fit recording it. JAX defaults to single precision and silently downcasts
# float64 arrays handed to it, so the maximum-likelihood path was computing
# gradients and a Hessian at ~1e-7 relative accuracy, casting them back up
# into float64 containers, and handing them to a scipy optimizer whose
# default tolerances assume ~1e-15. The standard errors came from inverting
# that Hessian.
#
# Nothing failed. The suite was green, the fits converged, and the numbers
# looked like double-precision numbers because they were stored in doubles.
# The only visible symptom was a 4.24x speed difference at n = 8,000 that
# took a separate investigation to explain.
#
# The lesson is that a silent default which changes every number in the
# package needs a test asserting what it is - not a test that fits still
# run. These tests therefore check the SETTING, not model quality.
# ==============================================================================

test_that("the backend runs in float64 by default", {
  skip_if_no_backend()

  # The headline assertion of this file. If this fails, every number the
  # package produces has changed, including the vignette benchmarks.
  expect_identical(jmjax:::.backend_precision(), "float64")
})

test_that("R and the Python backend agree on the precision in force", {
  skip_if_no_backend()

  # .backend_precision() reads get_precision() out of the running backend
  # rather than echoing JMJAX_ENABLE_X64. That distinction is the point:
  # if another package imported jax before jmjax did, the environment
  # variable was set too late to be read and only the config.update()
  # fallback in __init__.py applied. Asserting against jax's own view
  # catches a disagreement; asserting against the env var would not.
  be <- jmjax:::.get_backend()
  expect_identical(as.character(be$get_precision()),
                   jmjax:::.backend_precision())

  # And jax itself must agree that x64 is on, independently of the helper.
  expect_true(isTRUE(as.logical(
    reticulate::py_eval("__import__('jax').config.jax_enable_x64"))))
})

test_that("a fit records the precision that produced it", {
  skip_if_no_backend()
  skip_if_slow_mcmc()

  sim <- simulate_joint_data(n = 80, seed = 404)

  fit <- jm_fit(
    long_formula = y ~ time,
    surv_formula = survival::Surv(time, event) ~ 1,
    data_long = sim$data_long,
    data_surv = sim$data_surv,
    id_var = "id",
    time_var = "time",
    method = "spline-PH-mcmc",
    random_effects = "intercept",
    control = list(n_interior_knots = 5L, num_warmup = 100, num_samples = 100,
                   num_chains = 1, progress_bar = FALSE)
  )

  # The whole reason this field exists: a saved fit must describe the
  # arithmetic that produced it. jmjax's benchmark corpus had to be treated
  # as wholesale suspect precisely because no fit carried this.
  expect_identical(fit$convergence$precision, "float64")
  expect_identical(fit$convergence$precision, jmjax:::.backend_precision())

  # Stamped at jm_fit()'s single assembly point, so it must survive
  # alongside the fields the backend itself sets rather than replacing them.
  expect_true(!is.null(fit$convergence$message))
  expect_true("sampling_time_sec" %in% names(fit$convergence))

  # ...and be visible without digging into the object.
  out <- paste(utils::capture.output(print(fit)), collapse = "\n")
  expect_match(out, "Precision: float64")
})

test_that("print.jmjax tolerates a fit saved before precision was recorded", {
  # Backward compatibility, tested without the backend: an .rds written by
  # an older jmjax has no convergence$precision, and must print rather than
  # error or claim "unknown". Constructed by hand so this runs anywhere.
  old_fit <- structure(
    list(call = quote(jm_fit()), method = "spline-PH-mcmc",
         estimates = c(beta_0 = 1), se = c(beta_0 = 0.1),
         loglik = NULL,
         convergence = list(converged = TRUE, message = "ok"),
         n_subjects = 10L, n_obs_long = 40L, n_events = 5L),
    class = "jmjax")

  expect_error(utils::capture.output(print(old_fit)), NA)
  out <- paste(utils::capture.output(print(old_fit)), collapse = "\n")
  expect_false(grepl("Precision", out))
})

test_that("jmjax_setup refuses to change precision after the backend loads", {
  skip_if_no_backend()

  # JAX reads x64 once, when its backend initializes, and reticulate binds
  # one interpreter per R session. So this cannot take effect, and the only
  # safe behaviour is to say so loudly. Silently accepting the argument -
  # or worse, setting the environment variable so a LATER session picks up
  # a change the user thought applied now - is the failure mode guarded
  # against here.
  jmjax:::.get_backend()   # ensure it is loaded
  before <- Sys.getenv("JMJAX_ENABLE_X64")

  # try() because jmjax_setup() goes on to call use_virtualenv(required =
  # TRUE) after warning, which throws if this session's Python is attached
  # to an environment the user selected themselves rather than to r-jmjax.
  # That is a legitimate configuration and not what this test is about -
  # the warning is. try() lets the warning through and swallows only the
  # unrelated error.
  expect_warning(try(jmjax_setup(enable_x64 = FALSE), silent = TRUE),
                 "already loaded")

  expect_identical(Sys.getenv("JMJAX_ENABLE_X64"), before)
  expect_identical(jmjax:::.backend_precision(), "float64")
})

test_that("JMJAX_ENABLE_X64=0 actually produces a float32 backend", {
  skip_on_cran()
  # system2()'s `env` argument is ignored on Windows (with a warning), so
  # the subprocess would inherit this session's settings, run in float64,
  # and fail this test for a reason that has nothing to do with the code.
  testthat::skip_on_os("windows")
  skip_if_no_backend()

  # The opt-out cannot be tested in this session - precision is fixed at
  # the first jax import - so it needs a fresh R process. That makes this
  # the one test here that depends on jmjax being INSTALLED rather than
  # load_all()'d, which is not true under some devtools workflows. Skip
  # rather than fail in that case: this asserts a real behaviour, but it
  # is not worth a red suite on a machine that simply cannot run it.
  rscript <- file.path(R.home("bin"), "Rscript")
  skip_if_not(file.exists(rscript), "Rscript not found")
  skip_if_not(nzchar(system.file(package = "jmjax")),
              "jmjax is not installed (load_all session); cannot spawn a fresh process")

  code <- paste(
    "suppressWarnings(suppressMessages(library(jmjax)))",
    "cat(jmjax:::.backend_precision())",
    sep = "; ")

  out <- suppressWarnings(tryCatch(
    system2(rscript, c("-e", shQuote(code)),
            env = c("JMJAX_ENABLE_X64=0", "NOT_CRAN=true"),
            stdout = TRUE, stderr = FALSE),
    error = function(e) NA_character_))

  skip_if(all(is.na(out)) || !length(out),
          "could not run a subprocess to test the opt-out")

  expect_match(paste(out, collapse = ""), "float32",
               info = paste0("A fresh R process with JMJAX_ENABLE_X64=0 ",
                             "should report float32. Got: ",
                             paste(out, collapse = " ")))
})
