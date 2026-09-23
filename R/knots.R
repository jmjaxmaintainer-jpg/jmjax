# ==============================================================================
# Spline baseline hazard knot placement.
#
# Ported from JM::jointModel's internal logic for method = "spline-PH-aGH"
# (single-stratum / equal.strata.knots = TRUE case), so that a spline fit
# from this package uses the same basis convention as JM and the two are
# directly comparable - this was verified during prototyping via
# debug(jointModel) + all.equal() against jm_fit$x$W2 %*% jm_fit$coefficients$gamma.
#
# IMPORTANT: this operates on RAW time, i.e. assumes a survival submodel
# built from coxph() (semi-parametric PH), not survreg() (AFT). Passing a
# survreg-style survival spec produced a confirmed exp(time)-scale quirk in
# JM itself during prototyping; this package intentionally supports only the
# coxph-style relative-risk specification to sidestep that entirely.
# ==============================================================================

#' Build the padded B-spline knot vector and basis-evaluation function for
#' the spline baseline hazard.
#'
#' @param T_surv Numeric vector of observed event/censoring times.
#' @param n_interior Number of interior knots (JM's default: lng.in.kn = 5).
#' @param ord Spline order (JM's default: ord = 4, i.e. cubic).
#' @param placement \code{"quantile"} (default) places interior knots at
#'   quantiles of \code{T_surv}, matching \code{JM}'s scheme. \code{"equal"}
#'   places interior knots at EQUALLY SPACED positions across a padded time
#'   range instead, matching \code{JMbayes2}'s scheme (confirmed exactly
#'   from \code{JMbayes2:::knots}'s source).
#' @param t_quad Optional matrix of Gauss-Kronrod quadrature evaluation
#'   times (e.g. \code{surv_arr$t_quad}). Only used when
#'   \code{placement = "quantile"}: JM's actual boundary knots are computed
#'   from \code{range(Time, st)} - i.e. observed times COMBINED WITH the
#'   quadrature evaluation points \code{st}, not \code{range(Time)} alone
#'   (confirmed by comparing a real fitted JM object's
#'   \code{control$knots} against this package's reconstruction during
#'   development: JM's lower boundary was far below \code{min(T_surv)},
#'   consistent with a small Gauss-Kronrod node scaled by a large subject's
#'   T_i landing closer to zero than any single observed time). If NULL
#'   (default), falls back to \code{range(T_surv)} alone - a reasonable
#'   approximation, but not an exact JM replica for the boundary.
#' @return A list with `knots` (full padded knot vector) and `basis(t)`, a
#'   function evaluating the B-spline design matrix at arbitrary times t
#'   (via splines::splineDesign, outer.ok = TRUE so it can be safely
#'   evaluated on a fine grid for plotting even slightly beyond the range
#'   used to place the knots).
#' @keywords internal
build_spline_knots <- function(T_surv, n_interior = 5L, ord = 4L,
                                placement = c("quantile", "equal"),
                                t_quad = NULL) {
  placement <- match.arg(placement)
  n_interior <- as.integer(n_interior)
  ord <- as.integer(ord)

  if (placement == "quantile") {
    pp <- seq(0, 1, length.out = n_interior + 2)
    pp <- utils::tail(utils::head(pp, -1), -1)  # drop 0 and 1 -> n_interior probs
    kk <- stats::quantile(T_surv, pp, names = FALSE)
    boundary_range <- if (!is.null(t_quad)) range(c(T_surv, t_quad)) else range(T_surv)
    kk <- kk[kk < boundary_range[2]]
    knots <- sort(c(rep(boundary_range, ord), kk))
  } else {
    # "equal": EXACT replication of JMbayes2:::knots(x, ndx, deg, basis).
    # Their `deg` parameter is degree (not order); ord here = degree + 1.
    # Their `ndx` (base_hazard_segments) relates to n_interior via
    # ndx = n_interior + 1 (confirmed: segments=8 -> 7 interior knots).
    degree <- ord - 1L
    lower <- sqrt(.Machine$double.eps)
    upper <- max(T_surv) + 0.001
    kn <- seq(lower, upper, length.out = n_interior + 2)  # includes both endpoints
    knots <- c(rep(lower, degree), kn, rep(upper, degree))
  }

  basis <- function(t) {
    splines::splineDesign(knots = knots, x = t, ord = ord, outer.ok = TRUE)
  }

  # IMPORTANT: must stay a true R integer (not double) - this value flows
  # through reticulate into Python array-shape arithmetic (n_theta = p + 2 +
  # n_splines + 1), and numpy/JAX reject a float shape argument (e.g.
  # "expected a sequence of integers ... got '14.0'") if this leaks through
  # as a double.
  n_splines <- as.integer(length(knots) - ord)

  list(knots = knots, basis = basis, n_splines = n_splines, ord = ord, placement = placement)
}
