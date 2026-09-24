# ==============================================================================
# Package-level Python environment management.
#
# Design choice: a dedicated virtualenv ("r-jmjax"), created and populated on
# first use, rather than requiring the user to manage a Python install
# themselves. This mirrors the pattern used by the `torch` and `tensorflow`
# R packages. The environment is created once per machine and reused across
# sessions; jmjax_setup() is exported for the user to call explicitly (e.g.
# in a Docker build step) so first-use latency isn't a surprise mid-analysis.
# ==============================================================================

.jmjax_env <- new.env(parent = emptyenv())

JMJAX_VENV_NAME <- "r-jmjax"

#' Set up the Python backend for jmjax
#'
#' Creates (if needed) a dedicated virtualenv and installs the pinned
#' Python dependencies (jax, numpyro, numpy, scipy). Safe to call multiple
#' times - subsequent calls are no-ops if the environment already satisfies
#' the pinned versions.
#'
#' @param recreate Logical; if TRUE, delete and rebuild the environment even
#'   if it already exists. Useful after a jmjax package upgrade that bumps
#'   the required backend versions.
#' @param num_devices Number of CPU "devices" JAX/numpyro should expose for
#'   genuinely parallel MCMC chains (\code{spline-PH-mcmc}/
#'   \code{weibull-PH-mcmc} with \code{num_chains > 1}). By default, JAX
#'   only exposes ONE device regardless of how many CPU cores are actually
#'   available, silently forcing chains to run sequentially even when
#'   \code{num_chains > 1} is requested (with a
#'   "not enough devices... chains will be drawn sequentially" warning).
#'   Defaults to \code{parallel::detectCores()}. MUST be set before the
#'   Python backend is first imported in a session (either via this
#'   function or the first \code{jm_fit()} call) - it cannot be changed
#'   afterward without restarting R, since XLA reads this setting once,
#'   the first time its backend actually initializes.
#' @param enable_x64 Use double precision (float64) in the Python backend.
#'   Defaults to \code{TRUE}, which is NOT JAX's own default - JAX defaults
#'   to float32 and silently downcasts any float64 array passed to it.
#'
#'   \code{TRUE} is the right default here for three measured reasons.
#'
#'   The \strong{maximum-likelihood path} is the worst affected and the
#'   least visible. It computes its gradient and Hessian in JAX and hands
#'   them to \code{scipy}'s optimizer, whose default tolerances assume
#'   roughly 1e-15 relative accuracy; under float32 it receives about
#'   1e-7. Measured on the same data from the same starting values at
#'   \code{n = 400}, float32 reported \code{converged = TRUE} with a
#'   maximum gradient component of \strong{0.78} against float64's
#'   \strong{0.0005}, stopping after 80 iterations rather than 127. At a
#'   stationary point that gradient is near zero, so the float32 fit had
#'   not converged and said it had. The Hessian is inverted for the
#'   standard errors, which compounds it.
#'
#'   For \strong{MCMC}, float64 was faster at every size measured, not
#'   only large ones: \strong{1.20-1.26x} at \code{n = 200},
#'   \strong{1.29-1.40x} at \code{n = 1,000} and \strong{4.24x} at
#'   \code{n = 8,000}, with higher ESS per second throughout. The
#'   mechanism differs by size - at \code{n = 8,000} float32 noise
#'   inflated the leapfrog step count roughly fourfold, while at the
#'   smaller sizes step counts matched and float32 was simply slower for
#'   less effective sample.
#'
#'   And \code{exp()} overflows float32 at about 88 against float64's
#'   709, which matters because the hazard is
#'   \code{exp(linear predictor)} and NUTS explores wild regions in warmup.
#'
#'   R has no single-precision numeric type, so \code{TRUE} is also what a
#'   user calling this from R already assumes they are getting. Set
#'   \code{FALSE} only to trade accuracy for memory. Like
#'   \code{num_devices}, this MUST be set before the backend is first
#'   imported in a session.
#' @return \code{TRUE}, invisibly.
#' @export
jmjax_setup <- function(recreate = FALSE, num_devices = parallel::detectCores(),
                        enable_x64 = TRUE) {
  if (is.na(num_devices)) num_devices <- 4L  # detectCores() can return NA in restricted environments
  if (is.null(.jmjax_env$backend)) {
    Sys.setenv(JMJAX_NUM_DEVICES = as.integer(num_devices))
    Sys.setenv(JMJAX_ENABLE_X64 = if (isTRUE(enable_x64)) "1" else "0")
  } else {
    warning("jmjax: the Python backend is already loaded in this session - ",
            "num_devices and enable_x64 cannot be changed now (XLA and JAX ",
            "read these settings once, the first time the backend ",
            "initializes). Restart R and call jmjax_setup(num_devices = ",
            num_devices, ", enable_x64 = ", isTRUE(enable_x64), ") before any ",
            "jm_fit() call to apply this.")
  }

  if (recreate && reticulate::virtualenv_exists(JMJAX_VENV_NAME)) {
    reticulate::virtualenv_remove(JMJAX_VENV_NAME, confirm = FALSE)
  }

  if (!reticulate::virtualenv_exists(JMJAX_VENV_NAME)) {
    message("jmjax: creating Python virtualenv '", JMJAX_VENV_NAME, "' ...")
    reticulate::virtualenv_create(
      envname = JMJAX_VENV_NAME,
      packages = c(
        "numpy==1.26.4",
        "scipy==1.13.0",
        "jax==0.4.30",
        "jaxlib==0.4.30",
        "numpyro==0.15.0"
      )
    )
  }

  reticulate::use_virtualenv(JMJAX_VENV_NAME, required = TRUE)
  invisible(TRUE)
}

#' Is the Python backend available?
#'
#' \code{TRUE} when a Python environment with jax and numpyro is loaded or
#' can be loaded, without creating or installing anything. Useful to guard
#' code - package examples use it - that should run only where
#' \code{\link{jmjax_setup}()} has been done.
#'
#' @return A single logical.
#' @export
#' @examples
#' jmjax_available()
jmjax_available <- function() {
  if (!is.null(.jmjax_env$backend)) return(TRUE)
  if (reticulate::py_available(initialize = FALSE)) {
    return(isTRUE(tryCatch(
      reticulate::py_module_available("jax") &&
        reticulate::py_module_available("numpyro"),
      error = function(e) FALSE)))
  }
  .python_has_backend_deps(tryCatch(reticulate::py_exe(), error = function(e) NULL))
}

.onLoad <- function(libname, pkgname) {
  # Point reticulate at the venv WITHOUT forcing creation/install at package
  # load time - that belongs in jmjax_setup(), called explicitly by the user
  # or lazily on first jm_fit() call. Package load should stay fast and not
  # require network access.
  if (reticulate::virtualenv_exists(JMJAX_VENV_NAME)) {
    reticulate::use_virtualenv(JMJAX_VENV_NAME, required = FALSE)
  }
}

# May this session build a Python environment on its own?
#
# TRUE when a person is present to have asked for it, or when the caller has
# said so. FALSE in the one case that matters: a non-interactive check with
# NOT_CRAN unset, which is how CRAN runs R CMD check. See .get_backend().
.auto_setup_permitted <- function() {
  interactive() ||
    identical(tolower(Sys.getenv("NOT_CRAN")), "true") ||
    identical(tolower(Sys.getenv("JMJAX_ALLOW_SETUP")), "true")
}

# Does the interpreter at `python` have the backend's dependencies?
#
# Deliberately a SUBPROCESS check rather than reticulate::py_module_available():
# the latter starts Python in THIS session, which is the one thing the caller
# below must not do before it has chosen an environment (see .get_backend()).
# find_spec() only looks the modules up on sys.path, so this costs a few
# milliseconds - importing jax for real would cost seconds.
.python_has_backend_deps <- function(python) {
  if (is.null(python) || !nzchar(python) || !file.exists(python)) return(FALSE)
  probe <- paste(
    "import importlib.util as u, sys",
    "sys.exit(0 if u.find_spec('jax') and u.find_spec('numpyro') else 1)",
    sep = "; "
  )
  status <- tryCatch(
    suppressWarnings(system2(python, c("-c", shQuote(probe)),
                             stdout = FALSE, stderr = FALSE)),
    error = function(e) 1L
  )
  identical(as.integer(status), 0L)
}

# Lazily import and cache the backend module.
#
# Strategy: use whatever Python reticulate is ALREADY configured to use, if
# that interpreter has jax/numpyro (e.g. the user ran
# reticulate::use_condaenv("myenv") themselves before calling jm_fit() -
# common on Windows, where the WindowsApps stub python.exe/python3.exe can't
# be used to create a venv, but a real conda environment with jax/numpyro
# already installed works fine). Otherwise fall back to the package-managed
# "r-jmjax" virtualenv, creating it via jmjax_setup() if needed.
#
# ORDERING IS LOAD-BEARING, and was the cause of a real failure. reticulate
# binds to exactly one interpreter per R session, at the first py_* call, and
# cannot be re-pointed afterwards: use_python(required = TRUE) checks
# is_python_initialized() and calls stop("failed to initialize requested
# version of Python") if a different interpreter is already attached. So the
# environment has to be chosen BEFORE any Python is touched.
#
# An earlier version of this function probed first - py_run_string() to put
# the backend on sys.path, then import() - and only created/selected the venv
# if that failed. On a fresh machine with no "r-jmjax" venv, .onLoad() leaves
# reticulate unpointed, so the py_run_string() probe itself bound Python to
# whatever the system offered; the import then failed for want of numpyro;
# jmjax_setup() duly built the venv; and the use_virtualenv(required = TRUE)
# at the end of it threw, because Python was already attached elsewhere. The
# error was swallowed by the caller's tryCatch, so the whole test suite
# skipped its 79 backend tests with "Python backend not available" while a
# perfectly good venv sat on disk, built moments earlier by that same run.
.get_backend <- function() {
  if (is.null(.jmjax_env$backend)) {
    # If the user hasn't already configured this via jmjax_setup(), set a
    # sensible default now - the Python backend is about to be imported
    # for the first time either way, and JMJAX_NUM_DEVICES must be set
    # before that happens for parallel MCMC chains to work (see
    # jmjax_setup()'s docs for why this can't be changed afterward).
    if (Sys.getenv("JMJAX_NUM_DEVICES") == "") {
      .default_num_devices <- parallel::detectCores()
      if (is.na(.default_num_devices)) .default_num_devices <- 4L
      Sys.setenv(JMJAX_NUM_DEVICES = as.integer(.default_num_devices))
    }

    # Same lifecycle for precision. The Python side defaults to float64 on
    # its own if this is unset, so setting it here changes nothing about
    # what runs - it makes the setting VISIBLE from R (Sys.getenv) rather
    # than an invisible default buried in __init__.py, which matters when
    # someone is trying to work out why two runs differ.
    if (Sys.getenv("JMJAX_ENABLE_X64") == "") {
      Sys.setenv(JMJAX_ENABLE_X64 = "1")
    }

    # ---- choose the environment, while there is still a choice to make ----
    if (!reticulate::py_available(initialize = FALSE)) {
      # py_exe() reports the interpreter reticulate WOULD use - honouring
      # RETICULATE_PYTHON and any use_*() call made earlier in this session -
      # without starting it.
      candidate <- tryCatch(reticulate::py_exe(), error = function(e) NULL)
      if (!.python_has_backend_deps(candidate)) {
        # Creating a virtualenv downloads ~150MB. That is fine when a person
        # asked for a fit and is watching; it is a policy violation and an
        # automatic rejection during R CMD check on CRAN, which runs
        # non-interactively and forbids writing outside the session
        # temporary directory or reaching the network.
        #
        # The NOT_CRAN convention is testthat's: devtools::test() and
        # r-lib's CI actions set it to "true", CRAN does not. So this
        # refuses to build an environment in exactly the situation where
        # building one would be wrong, and nowhere else. The resulting
        # error is what makes every backend test skip on CRAN with a
        # sensible reason instead of the suite trying to install jax.
        if (!.auto_setup_permitted()) {
          stop("jmjax: no Python environment with jax/numpyro is available, ",
               "and this is a non-interactive session with NOT_CRAN unset, ",
               "so one will NOT be created automatically - that would mean a ",
               "~150MB download during a package check. Run ",
               "jmjax::jmjax_setup() once beforehand, point reticulate at an ",
               "existing jax/numpyro environment, or set JMJAX_ALLOW_SETUP=true ",
               "to permit automatic creation here.", call. = FALSE)
        }
        if (!reticulate::virtualenv_exists(JMJAX_VENV_NAME)) {
          message("jmjax: no Python environment with jax/numpyro found - ",
                  "running jmjax_setup() to create '", JMJAX_VENV_NAME, "'. ",
                  "If you already have jax/numpyro installed in an existing ",
                  "conda/virtualenv, run reticulate::use_condaenv() or ",
                  "reticulate::use_virtualenv() to select it BEFORE the first ",
                  "jm_fit() call to skip this step next time.")
          jmjax_setup()
        } else if (!.python_has_backend_deps(
                     tryCatch(reticulate::virtualenv_python(JMJAX_VENV_NAME),
                              error = function(e) NULL))) {
          # The venv is on disk but its dependencies are not - an install
          # that ran out of disk, lost the network, or was interrupted.
          # jmjax_setup() only installs when CREATING, so without recreate
          # it would bind to the broken env and fail at import.
          message("jmjax: the '", JMJAX_VENV_NAME, "' virtualenv exists but ",
                  "does not have jax/numpyro - a previous install did not ",
                  "finish. Rebuilding it.")
          jmjax_setup(recreate = TRUE)
        } else {
          reticulate::use_virtualenv(JMJAX_VENV_NAME, required = TRUE)
        }
      }
    }

    backend_path <- system.file("python", package = "jmjax")
    add_path_code <- sprintf(
      "import sys\nif r'%s' not in sys.path:\n    sys.path.insert(0, r'%s')",
      backend_path, backend_path
    )

    .jmjax_env$backend <- tryCatch({
      reticulate::py_run_string(add_path_code)
      reticulate::import("jmjax_backend", delay_load = FALSE)
    }, error = function(e) {
      # Reachable when Python was already attached - by another package, or
      # by an earlier call in this session - to an interpreter without the
      # backend's dependencies. Nothing can be done about that without a
      # restart, so say exactly that rather than re-raising an ImportError.
      stop("jmjax: could not load the Python backend from '",
           tryCatch(reticulate::py_exe(), error = function(e2) "<unknown>"),
           "'. Python is already attached to that interpreter in this R ",
           "session and reticulate cannot be re-pointed without restarting. ",
           "Restart R, then run jmjax::jmjax_setup() (or select your own ",
           "jax/numpyro environment with reticulate::use_condaenv() / ",
           "use_virtualenv()) before anything else touches Python.\n",
           "  original error: ", conditionMessage(e), call. = FALSE)
    })

    # Read the precision back from the RUNNING backend rather than trusting
    # what was requested. The two can disagree: if another package imported
    # jax earlier in this session, JMJAX_ENABLE_X64 was set too late to be
    # read, and only the config.update() fallback in __init__.py applied.
    # Reporting the request instead of the reality is how a package ends up
    # with a benchmark corpus whose precision nobody can reconstruct.
    .jmjax_env$precision <- tryCatch(
      as.character(.jmjax_env$backend$get_precision()),
      error = function(e) NA_character_)

    # Only reachable when someone opted out, or when a foreign jax import
    # won the race described above. Either way it changes the numbers, so
    # it should not be silent.
    if (identical(.jmjax_env$precision, "float32")) {
      warning("jmjax: the Python backend is running in SINGLE precision ",
              "(float32), not the float64 default. Standard errors from the ",
              "maximum-likelihood methods are computed by inverting a ",
              "float32 Hessian and should be treated as approximate - in a ",
              "measured comparison the float32 optimizer reported ",
              "convergence with a gradient max of 0.78 against float64's ",
              "0.0005 on the same data. MCMC was also slower in float32 at ",
              "every size measured (1.2x at n = 200, 1.3-1.4x at n = 1,000, ",
              "4.24x at n = 8,000). If this was not deliberate, restart R ",
              "and call ",
              "jmjax::jmjax_setup(enable_x64 = TRUE) before anything else ",
              "touches Python.", call. = FALSE)
    }
  }
  .jmjax_env$backend
}

# The precision the backend is ACTUALLY running in: "float64", "float32",
# or NA_character_ if it could not be determined. Recorded on every fit so
# that a saved result describes the arithmetic that produced it.
.backend_precision <- function() {
  .get_backend()   # populates .jmjax_env$precision as a side effect
  .jmjax_env$precision %||% NA_character_
}
