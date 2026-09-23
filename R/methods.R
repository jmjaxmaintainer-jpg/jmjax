# Standard model-object methods: coef(), vcov(), confint(), logLik(), nobs().
#
# A jmjax fit stores its results as fit$estimates / fit$se rather than the
# $coefficients that stats' default methods look for, so without these
# coef(fit) silently returned NULL. Each method below reads the fit's own
# fields; none of them recomputes a model quantity.

# Internal: turn whatever the backend sent as "vcov" into a numeric matrix
# named by the reported parameters, or NULL. Called once in jm_fit(), right
# after the backend returns. A matrix whose size does not match the
# estimates is dropped rather than labelled wrongly.
.jmjax_as_vcov <- function(v, nm) {
  if (is.null(v) || !length(nm)) return(NULL)
  V <- tryCatch({
    if (is.matrix(v)) v else do.call(rbind, lapply(v, function(r) as.numeric(unlist(r))))
  }, error = function(e) NULL)
  if (is.null(V)) return(NULL)
  storage.mode(V) <- "double"
  if (!identical(dim(V), c(length(nm), length(nm)))) return(NULL)
  dimnames(V) <- list(nm, nm)
  V
}

.jmjax_is_mcmc <- function(object) {
  !is.null(object$posterior_samples) && length(object$posterior_samples) > 0
}

# Internal: the posterior draws of every reported parameter as a
# [n_draws x n_params] matrix, columns in coef() order. A parameter whose
# draws cannot be found gets an NA column rather than being dropped, so the
# result always lines up with coef().
.jmjax_draws_matrix <- function(object) {
  nm <- names(object$estimates)
  dl <- lapply(nm, function(p) .jmjax_param_draws(object$posterior_samples, p))
  n <- max(0L, vapply(dl, function(d) if (is.null(d)) 0L else length(d), integer(1)))
  if (n < 2L) {
    stop("no posterior draws found in this fit", call. = FALSE)
  }
  D <- vapply(dl, function(d) {
    if (is.null(d) || length(d) != n) rep(NA_real_, n) else as.numeric(d)
  }, numeric(n))
  D <- matrix(D, nrow = n, dimnames = list(NULL, nm))
  D
}

.jmjax_select_parm <- function(object, parm) {
  nm <- names(object$estimates)
  if (missing(parm) || is.null(parm)) return(nm)
  if (is.numeric(parm)) return(nm[parm])
  bad <- setdiff(parm, nm)
  if (length(bad)) {
    stop("unknown parameter(s): ", paste(bad, collapse = ", "), call. = FALSE)
  }
  parm
}

#' Standard methods for jmjax fits
#'
#' Extract the estimates, their covariance matrix, confidence or credible
#' intervals, the log-likelihood and the number of subjects from a fit
#' returned by [jm_fit()] or [jm_fit_prefit()].
#'
#' The estimates and standard errors these return are the same numbers
#' [summary.jmjax()] reports, on the same (original) scale.
#'
#' * `coef()`: the point estimates - maximum-likelihood estimates, or
#'   posterior means for the MCMC methods.
#' * `vcov()`: for the maximum-likelihood methods, the inverse Hessian
#'   converted to the reported scale (so the variances of `sigma_*`,
#'   `shape` and `rho` refer to those parameters, not to their logs); for
#'   the MCMC methods, the covariance of the posterior draws. In both cases
#'   `sqrt(diag(vcov(fit)))` equals the standard errors in `summary(fit)`.
#' * `confint()`: Wald intervals (estimate +/- z * SE) for the
#'   maximum-likelihood methods; equal-tailed posterior credible intervals
#'   (quantiles of the draws) for the MCMC methods.
#' * `logLik()`: the maximized log-likelihood, with `df` the number of
#'   estimated parameters, so [stats::AIC()] and [stats::BIC()] work.
#'   Maximum-likelihood methods only: the MCMC methods do not compute it.
#' * `nobs()`: the number of subjects - the independent units of a joint
#'   model, and the sample size [stats::BIC()] uses.
#'
#' @param object A `"jmjax"` fit.
#' @param parm Parameters to include, by name or position. Default: all.
#' @param level Coverage of the intervals.
#' @param ... Unused.
#' @return `coef()`: named numeric vector. `vcov()`: named square matrix.
#'   `confint()`: two-column matrix, one row per parameter. `logLik()`: an
#'   object of class `"logLik"`. `nobs()`: an integer.
#' @name jmjax-methods
#' @importFrom stats coef vcov confint logLik nobs
NULL

#' @rdname jmjax-methods
#' @export
coef.jmjax <- function(object, ...) {
  est <- object$estimates
  setNames(as.numeric(unlist(est)), names(est))
}

#' @rdname jmjax-methods
#' @export
vcov.jmjax <- function(object, ...) {
  if (.jmjax_is_mcmc(object)) {
    return(stats::cov(.jmjax_draws_matrix(object)))
  }
  V <- object$vcov
  nm <- names(object$estimates)
  if (is.null(V)) {
    stop("no covariance matrix is available for this fit", call. = FALSE)
  }
  if (!is.matrix(V) || !identical(rownames(V), nm)) {
    # Fits made before jmjax 0.3.0 stored the inverse Hessian on the
    # optimizer's internal scale (log sigma, atanh rho) without names.
    # Returning that as vcov() would be wrong for every variance parameter.
    stop("this fit's stored covariance matrix is on the optimizer's internal ",
         "scale (fits made before jmjax 0.3.0); refit it to use vcov()",
         call. = FALSE)
  }
  V
}

#' @rdname jmjax-methods
#' @export
confint.jmjax <- function(object, parm, level = 0.95, ...) {
  nm <- .jmjax_select_parm(object, if (missing(parm)) NULL else parm)
  a <- (1 - level) / 2
  probs <- c(a, 1 - a)
  pct <- paste(format(100 * probs, trim = TRUE, scientific = FALSE, digits = 3), "%")
  out <- matrix(NA_real_, length(nm), 2L, dimnames = list(nm, pct))
  if (.jmjax_is_mcmc(object)) {
    for (p in nm) {
      d <- .jmjax_param_draws(object$posterior_samples, p)
      if (!is.null(d) && length(d) > 1L) {
        out[p, ] <- stats::quantile(d, probs, names = FALSE)
      }
    }
  } else {
    est <- coef(object)[nm]
    se <- as.numeric(unlist(object$se))[match(nm, names(object$se))]
    z <- stats::qnorm(1 - a)
    out[, 1] <- est - z * se
    out[, 2] <- est + z * se
  }
  out
}

#' @rdname jmjax-methods
#' @export
logLik.jmjax <- function(object, ...) {
  if (!is.numeric(object$loglik) || length(object$loglik) != 1L) {
    stop("logLik() is not available for method '", object$method, "': ",
         "the MCMC methods do not compute the maximized log-likelihood",
         call. = FALSE)
  }
  structure(object$loglik,
            df = length(object$estimates),
            nobs = object$n_subjects,
            class = "logLik")
}

#' @rdname jmjax-methods
#' @export
nobs.jmjax <- function(object, ...) {
  as.integer(object$n_subjects)
}
