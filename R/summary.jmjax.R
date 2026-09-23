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
    .print_estimates_compact(x)
  }

  # MCMC methods return loglik = NULL deliberately - a single marginal
  # log-likelihood isn't directly available from posterior samples the way
  # it is for the MLE methods (would need a separate log-density evaluation
  # at, e.g., posterior means - not yet implemented). Rather than print a
  # placeholder line explaining that every time, just omit the line: the
  # sampler/convergence lines just below already make clear this is an
  # MCMC fit, and summary(fit) never had a Log-Likelihood line to begin with.
  if (is.numeric(x$loglik) && length(x$loglik) == 1) {
    cat("\nLog-Likelihood:", round(x$loglik, 2), "\n")
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

# Prints one submodel group's estimate table: forces plain "data.frame"
# (see the recursion-guard note in print.summary.jmjax for why), rounds the
# numeric columns to 4dp, and - when a p.value column is present - formats
# it with format.pval() so a very small p-value reads as "<0.0001" instead
# of rounding to a bare "0" or spilling out to scientific notation. Shared
# by print.summary.jmjax and print.jmjax's compact view (.print_estimates_
# compact) so both print p-values the same way.
.jmjax_print_group_table <- function(sub, ...) {
  class(sub) <- "data.frame"
  # format.pval() on the RAW p-value, before round() touches it: eps is
  # compared against the actual value ("pv < eps"), and rounding first
  # could push a value just under eps (e.g. 0.00009996) up to exactly
  # 0.0001, which is no longer "< eps" and would then print as a plain
  # "0.0001" instead of "<0.0001". round()ing the other columns below still
  # clobbers this column too, but it's immediately overwritten with the
  # formatted version in its original position.
  p_fmt <- if ("p.value" %in% names(sub)) {
    format.pval(sub$p.value, digits = 3, eps = 1e-4, scientific = FALSE)
  } else NULL
  sub <- round(sub, 4)
  if (!is.null(p_fmt)) sub$p.value <- p_fmt
  print(sub, ...)
}

# summary.jmjax() always carries z.value/p.value (Wald, against zero) PLUS,
# additively for MCMC fits, CrI.lower/CrI.upper/Rhat/ESS - see its roxygen
# doc for why this is additive rather than a swap. print.jmjax's compact
# view and print.summary.jmjax both need to pick ONE set to actually show
# (showing both a z-test and a credible interval for the same row is
# clutter, and Rhat/ESS mean nothing for an MLE fit), so this is the one
# place that choice is made, by which columns summary.jmjax() happened to
# add: CrI.lower present means an MCMC fit, its columns win; otherwise the
# original MLE set. Returns only the names that are actually present, so a
# legacy/hand-built object missing some of them still prints what it has.
.jmjax_display_cols <- function(x) {
  wanted <- if ("CrI.lower" %in% names(x)) {
    c("Estimate", "Std.Err", "CrI.lower", "CrI.upper", "Rhat", "ESS")
  } else {
    c("Estimate", "Std.Err", "z.value", "p.value")
  }
  intersect(wanted, names(x))
}

# object$posterior_samples is keyed by the backend's raw NumPyro SITE name
# ("beta", "W", "gamma", "sigma_b" - one entry holding every draw for every
# element of that site), not by the expanded per-element names R uses
# elsewhere ("beta_0", "W3", "sigma_b1", ...). See mcmc_model.py's
# _site_names(): estimates/se/diagnostics$rhat/diagnostics$ess ARE keyed by
# those expanded names, but posterior_samples is built straight from the
# raw `samples` dict via `{k: np.asarray(v).tolist() for k, v in
# samples.items()}`, so it never goes through _site_names() at all. Naively
# indexing object$posterior_samples[["beta_0"]] therefore returns NULL for
# every vector-site element - only the scalar sites (alpha, sigma_e, tau_w,
# rho, and sigma_b when q=1, where the per-element name IS the site name)
# ever hit directly. Maps a per-element name back to its site and 0-based
# element index (matching _site_names()'s naming exactly: "beta"/"gamma"
# use an underscore before the index, "W"/"sigma_b" don't) and returns that
# element's column of draws.
.jmjax_param_draws <- function(ps, nm) {
  if (!is.null(ps[[nm]])) return(as.numeric(unlist(ps[[nm]])))  # scalar site: name == site
  m <- regmatches(nm, regexec("^(beta|gamma)_([0-9]+)$|^(W|sigma_b)([0-9]+)$", nm))[[1]]
  if (!length(m)) return(NULL)
  site <- if (nzchar(m[2])) m[2] else m[4]
  idx <- as.integer(if (nzchar(m[3])) m[3] else m[5]) + 1L
  v <- ps[[site]]
  if (is.null(v)) return(NULL)
  # reticulate hands a plain nested Python list (built via .tolist(), not a
  # numpy array) back as an R list of per-draw vectors rather than a
  # matrix - reassemble it into one either way before column-indexing.
  M <- if (is.list(v)) do.call(rbind, lapply(v, function(z) as.numeric(unlist(z)))) else as.matrix(v)
  if (idx > ncol(M)) return(NULL)
  M[, idx]
}

# The compact "Estimates by submodel:" block inside print.jmjax - a shorter
# view than summary(fit)'s full grouped table, meant to fit on one screen
# alongside the alpha headline and convergence/sampler lines. Two
# differences from print.summary.jmjax:
#   - Association is skipped entirely: print.jmjax already leads with
#     .print_alpha_headline()'s narrative version of the same one row, so
#     repeating it as a table here is redundant. It still appears in
#     summary(fit).
#   - The baseline-hazard block collapses to one line naming its
#     coefficients when it has more than 3 rows - a spline basis (W0, W1,
#     ...) is typically 8-10 coefficients that exist to make the hazard
#     shape flexible, not individually interpretable, and often comes with
#     its own smoothing-variance parameter (tau_w) riding along in the same
#     group. Both get folded into the one summary line; see summary(fit)
#     for the full table. A small/parametric baseline-hazard block (e.g.
#     just a Weibull "shape") stays under the 3-row threshold and still
#     prints in full.
.print_estimates_compact <- function(x) {
  s <- summary(x)
  groups <- c("Longitudinal", "Survival (baseline hazard)",
              "Survival (baseline covariates)", "Variance components", "Other")
  .cols <- .jmjax_display_cols(s)
  any_printed <- FALSE
  for (g in groups) {
    sub <- s[s$Group == g, .cols, drop = FALSE]
    if (!nrow(sub)) next
    any_printed <- TRUE

    if (identical(g, "Survival (baseline hazard)") && nrow(sub) > 3) {
      # Every row here is already a "Survival (baseline hazard)" parameter
      # by construction (.jmjax_param_group), so nrow > 3 alone is enough
      # to identify a spline basis - the only way this group gets that
      # large. Name the W-coefficient range and list anything else
      # (tau_w, say) separately, rather than requiring every row to match
      # "^W[0-9]+$": that stricter check used to mean a spline basis with
      # tau_w mixed in wouldn't collapse at all, defeating the point.
      is_w <- grepl("^W[0-9]+$", rownames(sub))
      w_nm <- rownames(sub)[is_w]
      other_nm <- rownames(sub)[!is_w]
      bits <- character(0)
      if (length(w_nm)) {
        bits <- c(bits, sprintf("%d spline coefficient%s (%s-%s)", length(w_nm),
                                 if (length(w_nm) == 1) "" else "s",
                                 w_nm[1], w_nm[length(w_nm)]))
      }
      if (length(other_nm)) bits <- c(bits, other_nm)
      cat(g, ": ", paste(bits, collapse = ", "),
          "; see summary(fit) for the full table\n", sep = "")
      cat("\n")
      next
    }

    cat(g, ":\n", sep = "")
    .jmjax_print_group_table(sub)
    cat("\n")
  }
  if (!any_printed) cat("(no parameter estimates)\n")
  invisible(NULL)
}

#' Summarize a jmjax fit
#'
#' @param object A \code{"jmjax"} fit object.
#' @param ... Further arguments (currently unused).
#' @return A data frame with one row per parameter: \code{Estimate},
#'   \code{Std.Err}, \code{z.value}, \code{p.value} (Wald test against
#'   zero) and \code{Group} (which submodel the parameter belongs to -
#'   Association, Longitudinal, Survival (baseline hazard), Survival
#'   (baseline covariates), or Variance components) - always present, for
#'   both MLE and MCMC fits, so existing code reading these columns keeps
#'   working unchanged. For the MCMC methods, four more columns are added
#'   (not substituted): \code{CrI.lower}/\code{CrI.upper} (95\% credible
#'   interval from the matching column of \code{object$posterior_samples}
#'   - already on the original covariate scale there, so these line up
#'   with \code{Estimate} above; \code{NA} for any parameter whose draws
#'   can't be located, via the internal \code{.jmjax_param_draws()}) and
#'   \code{Rhat}/\code{ESS} (per-parameter convergence diagnostics from
#'   \code{object$diagnostics}, \code{NA} where not available).
#'   \code{print()} (both \code{print.jmjax}'s
#'   compact view and \code{print.summary.jmjax}) shows only one set per
#'   fit - the credible-interval/Rhat/ESS columns for MCMC fits, the
#'   z-test columns for MLE fits - since a z-test against zero is an
#'   MLE-flavoured framing that doesn't fit a posterior distribution, but
#'   both sets stay in the returned data frame either way. Carries class
#'   \code{c("summary.jmjax", "data.frame")}; the extra class only changes
#'   how it prints (grouped, via \code{print.summary.jmjax}) - every
#'   existing \code{data.frame} access (\code{summary(fit)$Estimate},
#'   subsetting, etc.) still works exactly as before.
#' @export
summary.jmjax <- function(object, ...) {
  nm <- names(object$estimates)
  grp <- if (length(nm)) vapply(nm, .jmjax_param_group, character(1)) else character(0)

  z <- object$estimates / object$se
  p <- 2 * stats::pnorm(-abs(z))
  df <- data.frame(
    Estimate = object$estimates,
    Std.Err = object$se,
    z.value = z,
    p.value = p,
    Group = grp,
    row.names = nm
  )

  # posterior_samples is populated only by the MCMC methods (spline-PH-mcmc,
  # weibull-PH-mcmc); MLE fits never have it. That's the same test
  # print.jmjax's alpha headline already uses to decide whether a
  # z-test/Wald framing or a posterior credible interval applies.
  is_mcmc <- !is.null(object$posterior_samples) && length(object$posterior_samples) > 0

  if (is_mcmc) {
    cr_lo <- setNames(rep(NA_real_, length(nm)), nm)
    cr_hi <- cr_lo
    for (p_nm in nm) {
      samp <- .jmjax_param_draws(object$posterior_samples, p_nm)
      if (!is.null(samp) && length(samp) > 1) {
        q <- stats::quantile(samp, c(0.025, 0.975), names = FALSE)
        cr_lo[[p_nm]] <- q[1]
        cr_hi[[p_nm]] <- q[2]
      }
      # else: leave NA. A lookup failure here means .jmjax_param_draws()
      # couldn't map this name to a site/column at all (not just "this
      # particular site has no draws") - substituting a Wald normal
      # approximation would look like a real credible interval while
      # actually being a different, silently-wrong methodology, which is
      # worse than a visible NA. (This used to fall back to Wald on the
      # assumption that a lookup miss was a rare edge case; it turned out
      # to be the common case for every vector-site element, because the
      # lookup itself was wrong - see .jmjax_param_draws()'s doc comment.)
    }

    rhat_v <- setNames(rep(NA_real_, length(nm)), nm)
    ess_v <- rhat_v
    dg <- object$diagnostics
    if (!is.null(dg)) {
      for (p_nm in nm) {
        if (!is.null(dg$rhat[[p_nm]])) rhat_v[[p_nm]] <- unname(dg$rhat[[p_nm]])
        if (!is.null(dg$ess[[p_nm]])) ess_v[[p_nm]] <- unname(dg$ess[[p_nm]])
      }
    }

    df$CrI.lower <- unname(cr_lo)
    df$CrI.upper <- unname(cr_hi)
    df$Rhat <- unname(rhat_v)
    df$ESS <- unname(ess_v)
  }

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
  # summary.jmjax() always returns both the z-test columns and, for MCMC
  # fits, the credible-interval/Rhat/ESS columns (additive - see its
  # roxygen doc). Printing both at once would be clutter, so pick the one
  # set that fits this object via .jmjax_display_cols(); the columns not
  # shown are still sitting in `x` for anyone who wants them directly.
  .cols <- .jmjax_display_cols(x)
  any_printed <- FALSE
  for (g in groups) {
    sub <- x[x$Group == g, .cols, drop = FALSE]
    if (!nrow(sub)) next
    any_printed <- TRUE
    # .jmjax_print_group_table() forces plain "data.frame" before printing.
    # [.data.frame subsetting can retain the "summary.jmjax" class on `sub`
    # (behaviour that is not guaranteed either way across R versions), and
    # round() (Math.data.frame) would then propagate that class to its
    # result too - print() on THAT dispatches straight back to this
    # function and recurses forever. That line is the only thing standing
    # between a class-preserving subset and an infinite loop, so it stays
    # even though it looks defensive/unneeded in a session where
    # subsetting happens to drop the class.
    cat(g, ":\n", sep = "")
    .jmjax_print_group_table(sub, ...)
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
