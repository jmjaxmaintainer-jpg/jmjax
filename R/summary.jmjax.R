#' Print a jmjax fit
#'
#' @param x A \code{"jmjax"} fit object.
#' @param ... Further arguments (currently unused).
#' @return \code{x}, invisibly.
#' @export
print.jmjax <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nMethod:", x$method, "\n")

  # Precision belongs next to the method: it is part of HOW the fit was
  # computed, not a diagnostic of how well it went. NULL for fits saved
  # before jmjax recorded this, which is why it is skipped rather than
  # printed as "unknown" - an old object is not a broken one.
  .prec <- x$convergence$precision
  if (!is.null(.prec) && !is.na(.prec)) {
    cat("Precision: ", .prec,
        if (identical(.prec, "float32"))
          "  (single - standard errors are approximate; see ?jmjax_setup)"
        else "",
        "\n", sep = "")
  }

  if (is.numeric(x$loglik) && length(x$loglik) == 1) {
    cat("Log-Likelihood:", round(x$loglik, 2), "\n")
  } else {
    # MCMC methods return loglik = NULL deliberately - a single marginal
    # log-likelihood isn't directly available from posterior samples the
    # way it is for the MLE methods (would need a separate log-density
    # evaluation at, e.g., posterior means - not yet implemented).
    cat("Log-Likelihood: not reported for method '", x$method, "'\n", sep = "")
  }

  cat("Subjects:", x$n_subjects, " | Longitudinal obs:", x$n_obs_long,
      " | Events:", x$n_events, sprintf("(%.1f%%)", 100 * x$n_events / x$n_subjects), "\n")

  if (!is.null(x$diagnostics)) {
    rhat <- unlist(x$diagnostics$rhat)
    ess <- unlist(x$diagnostics$ess)
    cat(sprintf("MCMC diagnostics: max R-hat = %.4f, min ESS = %.0f\n",
                max(rhat, na.rm = TRUE), min(ess, na.rm = TRUE)))
  }

  if (!is.null(x$convergence) && !isTRUE(x$convergence$converged)) {
    warning("Fit did not report tolerance-based convergence: ", x$convergence$message)
  }
  invisible(x)
}

#' Summarize a jmjax fit
#'
#' @param object A \code{"jmjax"} fit object.
#' @param ... Further arguments (currently unused).
#' @return A data frame with one row per parameter: \code{Estimate},
#'   \code{Std.Err}, \code{z.value}, and \code{p.value} (Wald test against
#'   zero). For MCMC fits, \code{Estimate}/\code{Std.Err} are the posterior
#'   mean/SD and the Wald framing is an approximation - inspect the full
#'   posterior (\code{object$posterior_samples}) for skewed parameters.
#' @export
summary.jmjax <- function(object, ...) {
  z <- object$estimates / object$se
  p <- 2 * stats::pnorm(-abs(z))
  data.frame(
    Estimate = object$estimates,
    Std.Err = object$se,
    z.value = z,
    p.value = p,
    row.names = names(object$estimates)
  )
}

#' @export
nlme::ranef

#' Extract per-subject random effects from a jmjax fit
#'
#' Populated for the two MCMC methods (\code{"spline-PH-mcmc"},
#' \code{"weibull-PH-mcmc"}), which sample each subject's random effect(s)
#' directly. Not available for the two MLE methods, which marginalize the
#' random effect via adaptive Gauss-Hermite quadrature and never estimate
#' individual b_i directly.
#'
#' @param object A \code{"jmjax"} fit object.
#' @param ... Further arguments (currently unused).
#' @return A data frame with one row per subject (posterior mean and SD of
#'   each random-effect dimension), or \code{NULL} with a warning if the
#'   fit method doesn't produce per-subject random effects.
#' @importFrom nlme ranef
#' @export
ranef.jmjax <- function(object, ...) {
  if (is.null(object$random_effects)) {
    warning("No per-subject random effects available for method '", object$method,
             "' - only the MCMC methods (spline-PH-mcmc, weibull-PH-mcmc) estimate ",
             "these directly (MLE methods marginalize the random effect via quadrature).")
    return(NULL)
  }
  as.data.frame(object$random_effects)
}
