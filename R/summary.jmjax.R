#' Print a jmjax fit
#'
#' Leads with the association parameter (\code{alpha}, the effect of the
#' longitudinal marker on the hazard) as the headline result, since it is
#' usually the reason a joint model was fit at all, then the full set of
#' estimates grouped by submodel, a one-line convergence summary, and
#' (for the MCMC methods) the sampler settings and whether the
#' \code{control$rotate_absorbable} reparameterization was applied.
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

  cat("Subjects:", x$n_subjects, " | Longitudinal obs:", x$n_obs_long,
      " | Events:", x$n_events, sprintf("(%.1f%%)", 100 * x$n_events / x$n_subjects), "\n")

  .print_alpha_headline(x)

  if (!is.null(x$estimates) && length(x$estimates)) {
    cat("\nEstimates by submodel:\n")
    print(summary(x))
  }

  if (is.numeric(x$loglik) && length(x$loglik) == 1) {
    cat("\nLog-Likelihood:", round(x$loglik, 2), "\n")
  } else {
    # MCMC methods return loglik = NULL deliberately - a single marginal
    # log-likelihood isn't directly available from posterior samples the
    # way it is for the MLE methods (would need a separate log-density
    # evaluation at, e.g., posterior means - not yet implemented).
    cat("Log-Likelihood: not reported for method '", x$method, "'\n", sep = "")
  }

  .print_convergence_line(x)
  .print_sampler_line(x)

  if (!is.null(x$convergence) && !isTRUE(x$convergence$converged)) {
    warning("Fit did not report tolerance-based convergence: ", x$convergence$message)
  }
  invisible(x)
}

# ---- internal helpers, not exported --------------------------------------

# Association-parameter site names the backend can produce, in the order
# they are preferred for the headline. "alpha" (the default value-only
# association) is almost always what is present; the functional_forms
# channels are mutually exclusive with it and with each other (see
# mcmc_model.py's population_sites: "alpha absent when a channel is used").
.jmjax_assoc_candidates <- c("alpha", "alpha_value", "alpha_delta",
                              "alpha_area", "alpha_area_avg")

.jmjax_assoc_param <- function(nm_all) {
  hit <- .jmjax_assoc_candidates[.jmjax_assoc_candidates %in% nm_all]
  if (length(hit)) hit[1] else NA_character_
}

# One parameter name -> one submodel group, for both print.jmjax's headline
# search and summary.jmjax's grouped table. Falls through to "Other" rather
# than dropping anything, so a future/unrecognised site name still appears
# somewhere rather than silently vanishing from the printed output.
.jmjax_param_group <- function(nm) {
  if (grepl("^beta_[0-9]+$", nm) || identical(nm, "sigma_e")) return("Longitudinal")
  if (nm %in% .jmjax_assoc_candidates) return("Association")
  if (grepl("^gamma_[0-9]+$", nm)) return("Survival (baseline covariates)")
  if (grepl("^W[0-9]+$", nm) || nm %in% c("tau_w", "log_lambda0", "shape")) {
    return("Survival (baseline hazard)")
  }
  if (grepl("^sigma_b[0-9]*$", nm) || identical(nm, "rho")) return("Variance components")
  "Other"
}

.print_alpha_headline <- function(x) {
  nm_all <- names(x$estimates)
  a <- .jmjax_assoc_param(nm_all)
  if (is.na(a)) return(invisible(NULL))

  est <- unname(x$estimates[[a]])
  se_a <- if (a %in% names(x$se)) unname(x$se[[a]]) else NA_real_

  # Prefer the posterior draws when they exist (a real credible interval),
  # and fall back to a Wald normal approximation from the SE for the MLE
  # methods, which have no posterior_samples.
  .samp <- x$posterior_samples[[a]]
  if (!is.null(.samp) && length(unlist(.samp)) > 1) {
    ci <- stats::quantile(unlist(.samp), c(0.025, 0.975), names = FALSE)
    ci_kind <- "95% credible interval"
  } else if (is.finite(se_a) && se_a > 0) {
    ci <- est + c(-1, 1) * stats::qnorm(0.975) * se_a
    ci_kind <- "95% CI"
  } else {
    ci <- NULL
    ci_kind <- NULL
  }

  cat("\nAssociation (", a, "): effect of the longitudinal marker on the hazard\n", sep = "")
  if (is.null(ci_kind)) {
    cat(sprintf("  %s = %.4f (SE unavailable)\n", a, est))
    return(invisible(NULL))
  }
  cat(sprintf("  %s = %.4f, %s = [%.4f, %.4f]\n", a, est, ci_kind, ci[1], ci[2]))
  # "Hazard ratio per unit of marker" is the literal reading only for the
  # default value-only association (alpha * m_i(t)); the functional_forms
  # channels (delta/area/area_avg) scale a different derived quantity, so
  # they get the more neutral exp(coefficient) phrasing instead.
  .label <- if (identical(a, "alpha")) "hazard ratio per unit of marker" else "exp(coefficient)"
  cat(sprintf("  %s = %.4f, %s = [%.4f, %.4f]\n",
              .label, exp(est), ci_kind, exp(ci[1]), exp(ci[2])))
  invisible(NULL)
}

.print_convergence_line <- function(x) {
  cv <- x$convergence
  dg <- x$diagnostics
  parts <- character(0)
  if (!is.null(dg) && !is.null(dg$rhat) && !is.null(dg$ess)) {
    rhat <- unlist(dg$rhat)
    ess <- unlist(dg$ess)
    if (any(is.finite(rhat)) && any(is.finite(ess))) {
      parts <- c(parts, sprintf("max R-hat = %.4f", max(rhat, na.rm = TRUE)),
                 sprintf("min ESS = %.0f", min(ess, na.rm = TRUE)))
    }
  }
  if (!is.null(cv) && !is.null(cv$n_divergences)) {
    parts <- c(parts, sprintf("divergences = %s", format(cv$n_divergences)))
  }
  if (length(parts)) {
    cat("\nConvergence:", paste(parts, collapse = ", "), "\n")
  }
  invisible(NULL)
}

.print_sampler_line <- function(x) {
  cv <- x$convergence
  ms <- x$mcmc_settings
  bits <- character(0)
  if (!is.null(ms)) {
    if (!is.null(ms$num_chains) && !is.null(ms$num_samples) && !is.null(ms$num_warmup)) {
      bits <- c(bits, sprintf("%s chain%s x %s samples (%s warmup)",
                               ms$num_chains, if (identical(ms$num_chains, 1L)) "" else "s",
                               ms$num_samples, ms$num_warmup))
    }
  }
  if (!is.null(cv) && !is.null(cv$sampling_time_sec)) {
    bits <- c(bits, sprintf("%.1fs", cv$sampling_time_sec))
  }
  if (length(bits)) cat("Sampler:", paste(bits, collapse = ", "), "\n")

  # Whether the location-degeneracy reparameterization was applied. Two
  # shapes: convergence$orthogonalize$rotation (control$rotate_absorbable,
  # the recommended, default-on path) and the legacy sweep report
  # (control$orthogonalize_b0/_b, mutually exclusive with rotation - see
  # mcmc_model.py's ValueError guard). NULL (both absent) means neither was
  # requested or applicable, which is the common case and prints nothing.
  o <- cv$orthogonalize
  if (!is.null(o$rotation)) {
    k <- as.integer(o$rotation$k %||% 0L)
    cat(sprintf("Rotation (control$rotate_absorbable): %s%s\n",
                if (isTRUE(o$rotation$applied)) "applied" else "not applied",
                if (isTRUE(o$rotation$applied)) sprintf(" (%d direction%s)", k, if (k == 1L) "" else "s") else ""))
  } else if (!is.null(o) && !identical(o$requested, "none")) {
    cat(sprintf("Legacy reparameterization (control$orthogonalize_%s): %s\n",
                o$requested, if (isTRUE(o$any_applied)) "applied" else "not applied"))
  }
  invisible(NULL)
}

#' Summarize a jmjax fit
#'
#' @param object A \code{"jmjax"} fit object.
#' @param ... Further arguments (currently unused).
#' @return A data frame with one row per parameter: \code{Estimate},
#'   \code{Std.Err}, \code{z.value}, \code{p.value} (Wald test against
#'   zero), and \code{Group} (which submodel the parameter belongs to -
#'   Association, Longitudinal, Survival (baseline hazard), Survival
#'   (baseline covariates), or Variance components). For MCMC fits,
#'   \code{Estimate}/\code{Std.Err} are the posterior mean/SD and the Wald
#'   framing is an approximation - inspect the full posterior
#'   (\code{object$posterior_samples}) for skewed parameters, or the alpha
#'   credible interval \code{print()} reports up front. Carries class
#'   \code{c("summary.jmjax", "data.frame")}; the extra class only changes
#'   how it prints (grouped, via \code{print.summary.jmjax}) - every
#'   existing \code{data.frame} access (\code{summary(fit)$Estimate},
#'   subsetting, etc.) still works exactly as before.
#' @export
summary.jmjax <- function(object, ...) {
  z <- object$estimates / object$se
  p <- 2 * stats::pnorm(-abs(z))
  nm <- names(object$estimates)
  df <- data.frame(
    Estimate = object$estimates,
    Std.Err = object$se,
    z.value = z,
    p.value = p,
    Group = if (length(nm)) vapply(nm, .jmjax_param_group, character(1)) else character(0),
    row.names = nm
  )
  class(df) <- c("summary.jmjax", "data.frame")
  df
}

#' Print a summarized jmjax fit
#'
#' @param x A \code{"summary.jmjax"} object, from \code{summary.jmjax}.
#' @param ... Further arguments passed to \code{print.data.frame} for each
#'   group's table.
#' @return \code{x}, invisibly.
#' @export
print.summary.jmjax <- function(x, ...) {
  # Fixed, meaningful order rather than alphabetical: the parameter the
  # model exists to estimate first, then the submodels roughly in the
  # order jm_fit()'s own arguments introduce them, "Other" (should
  # normally be empty - see .jmjax_param_group) last.
  groups <- c("Association", "Longitudinal", "Survival (baseline hazard)",
              "Survival (baseline covariates)", "Variance components", "Other")
  .cols <- c("Estimate", "Std.Err", "z.value", "p.value")
  any_printed <- FALSE
  for (g in groups) {
    sub <- x[x$Group == g, .cols, drop = FALSE]
    if (!nrow(sub)) next
    any_printed <- TRUE
    # Force plain "data.frame" before printing. [.data.frame subsetting can
    # retain the "summary.jmjax" class on `sub` (behaviour that is not
    # guaranteed either way across R versions), and round() (Math.data.frame)
    # would then propagate that class to its result too - print() on THAT
    # dispatches straight back to this function and recurses forever. This
    # line is the only thing standing between a class-preserving subset and
    # an infinite loop, so it stays even though it looks defensive/unneeded
    # in a session where subsetting happens to drop the class.
    class(sub) <- "data.frame"
    cat(g, ":\n", sep = "")
    print(round(sub, 4), ...)
    cat("\n")
  }
  if (!any_printed) cat("(no parameter estimates)\n")
  invisible(x)
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
