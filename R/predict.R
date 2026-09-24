# predict() for jmjax fits - phase 1.
#
# Scope: MCMC fits, subjects that were in the fitted data, the "value"
# association, q = 1 or 2, spline or Weibull baseline hazard, baseline
# survival covariates. Everything is computed per posterior draw from
# fit$posterior_samples, which jm_fit() has already put back on the
# original scale (standardized covariates and scale_time undone), and the
# design is rebuilt from the formulas and data as the user supplied them
# (fit$model_info, fit$data). Dynamic prediction for NEW subjects - which
# needs their random effects re-estimated from their own history - is
# planned for a later release.

# Internal: one sampling site as a [n_draws x k] matrix.
.jmjax_site_matrix <- function(ps, site) {
  d <- ps[[site]]
  if (is.null(d)) return(NULL)
  if (is.matrix(d)) return(d)
  if (is.list(d)) {
    return(do.call(rbind, lapply(d, function(z) as.numeric(unlist(z)))))
  }
  matrix(as.numeric(d), ncol = 1L)
}

# Internal: the random-effect draws of subject position `i` (in fit order)
# as [n_draws x q]. posterior_samples$b holds, per draw, either a flat
# [N] vector (q = 1), a list of per-subject [q] vectors, or an [N x q]
# matrix (q >= 2) - see the scale_time conversion in jm_fit(), which
# handles the same three shapes.
.jmjax_b_draws <- function(ps, i, q) {
  t(vapply(ps$b, function(d) {
    if (q == 1L) return(as.numeric(unlist(d))[i])
    if (is.list(d)) return(as.numeric(unlist(d[[i]])))
    as.numeric(as.matrix(d)[i, ])
  }, numeric(q)))
}

.jmjax_summarise_draws <- function(D, level) {
  a <- (1 - level) / 2
  cbind(estimate = colMeans(D),
        lower = apply(D, 2, stats::quantile, probs = a, names = FALSE),
        upper = apply(D, 2, stats::quantile, probs = 1 - a, names = FALSE))
}

#' Predictions from a jmjax fit
#'
#' Subject-specific predictions for subjects in the fitted data, from the
#' posterior draws of an MCMC fit: the longitudinal marker trajectory, or
#' the conditional survival probability \eqn{P(T > u \mid T > t)}.
#'
#' For each posterior draw \eqn{s}, the marker mean is
#' \eqn{m_i^{(s)}(t) = x_i(t)'\beta^{(s)} + z_i(t)'b_i^{(s)}} (the mean
#' trajectory, without measurement error), and the hazard is
#' \eqn{h_0^{(s)}(t)\exp(w_i'\gamma^{(s)} + \alpha^{(s)} m_i^{(s)}(t))}.
#' The survival probability is \eqn{\exp\{-\int_t^u h^{(s)}(v)\,dv\}},
#' integrated with composite 8-point Gauss-Legendre quadrature split at
#' the spline knots. The estimate is the
#' posterior mean over draws and the interval the equal-tailed posterior
#' interval.
#'
#' The random effects are the ones sampled with the model, so they condition
#' on each subject's full observed data - all marker values, and survival up
#' to the subject's follow-up time. For a subject who was censored, that is
#' exactly the information \eqn{P(T > u \mid T > t_{cens})} should use, and
#' is the default \code{t_from}.
#'
#' Phase 1 scope: MCMC methods, subjects present in the fit, and the
#' default \code{value} association. Maximum-likelihood fits, the
#' \code{delta}/\code{area}/\code{area_avg} associations and new subjects
#' stop with an error saying so.
#'
#' @param object A \code{"jmjax"} fit from an MCMC method.
#' @param newdata Optional long-format data frame whose \code{id_var}
#'   values select the subjects to predict for. They must all be subjects
#'   of the fit. Default: every subject. Required (and then also the source
#'   of covariate values) when the fit was made with
#'   \code{control$keep_data = FALSE}.
#' @param process \code{"longitudinal"} for the marker trajectory,
#'   \code{"event"} for conditional survival probabilities.
#' @param times Prediction times, shared by all subjects. Default: 20 points
#'   per subject - for \code{"longitudinal"} from the first visit to the end
#'   of follow-up, for \code{"event"} from \code{t_from} to the end of
#'   follow-up in the data. Event predictions are only available up to the
#'   largest follow-up time in the fit (the baseline hazard is not defined
#'   beyond it); later times are dropped with a warning.
#' @param t_from For \code{"event"}: the time the subject is known to have
#'   survived to. A single value, or one per subject. Default: each
#'   subject's follow-up time in the fit.
#' @param level Coverage of the posterior intervals.
#' @param return_draws If \code{TRUE}, the per-draw values are attached as
#'   attribute \code{"draws"}: a list, one \code{[n_draws x n_times]}
#'   matrix per subject.
#' @param ... Unused.
#' @return A data frame with columns \code{id}, \code{time},
#'   \code{estimate}, \code{lower}, \code{upper} (and \code{t_from} for
#'   \code{"event"}), one row per subject and time.
#' @importFrom stats predict
#' @export
predict.jmjax <- function(object, newdata = NULL,
                          process = c("longitudinal", "event"),
                          times = NULL, t_from = NULL, level = 0.95,
                          return_draws = FALSE, ...) {
  process <- match.arg(process)
  if (!.jmjax_is_mcmc(object)) {
    stop("predict() currently supports the MCMC methods only ",
         "(spline-PH-mcmc, weibull-PH-mcmc): maximum-likelihood fits do not ",
         "store per-subject random effects.", call. = FALSE)
  }
  mi <- object$model_info
  if (is.null(mi)) {
    stop("this fit was made before jmjax 0.3.0 and lacks what predict() ",
         "needs; refit it.", call. = FALSE)
  }
  if (!identical(mi$assoc_types, "value")) {
    stop("predict() currently supports the default 'value' association ",
         "only (this fit uses: ", paste(mi$assoc_types, collapse = ", "), ").",
         call. = FALSE)
  }
  id_var <- mi$id_var; time_var <- mi$time_var

  # --- data: stored on the fit, or supplied --------------------------------
  dl <- object$data$long
  ds <- object$data$surv
  if (is.null(dl)) {
    if (is.null(newdata)) {
      stop("this fit was made with control$keep_data = FALSE, so `newdata` ",
           "is required.", call. = FALSE)
    }
    dl <- newdata
    ds <- newdata[!duplicated(newdata[[id_var]]), , drop = FALSE]
  }
  ids <- if (is.null(newdata)) mi$subj_ids else unique(newdata[[id_var]])
  if (!length(ids)) {
    stop("no subjects selected: `newdata` has no rows (or no '", id_var,
         "' column).", call. = FALSE)
  }
  pos <- match(as.character(ids), as.character(mi$subj_ids))
  if (anyNA(pos)) {
    stop("predict() currently predicts for subjects in the fitted data ",
         "only; not in the fit: ",
         paste(utils::head(ids[is.na(pos)], 5), collapse = ", "),
         if (sum(is.na(pos)) > 5) ", ..." else "",
         ". Dynamic prediction for new subjects is planned.", call. = FALSE)
  }

  # Covariates are built for ALL fitted subjects at once, exactly as
  # jm_fit() built them, then indexed - building them per subject would
  # drop unused factor levels and change the design.
  bcl <- extract_baseline_covariates_long(mi$long_formula, time_var, dl,
                                          id_var, mi$subj_ids)

  ps <- object$posterior_samples
  beta <- .jmjax_site_matrix(ps, "beta")
  S <- nrow(beta)
  q <- ncol(stats::model.matrix(stats::delete.response(stats::terms(mi$random_formula)),
                                stats::setNames(data.frame(0), time_var)))

  X_at <- function(i, t) {
    bc <- if (is.null(bcl)) NULL else bcl[rep(i, length(t)), , drop = FALSE]
    X <- build_time_design(mi$long_formula, time_var, t, bc)
    if (ncol(X) != ncol(beta)) {
      stop("internal: rebuilt design has ", ncol(X), " columns but beta has ",
           ncol(beta), call. = FALSE)
    }
    X
  }
  Z_at <- function(t) build_time_design(mi$random_formula, time_var, t)
  m_draws <- function(i, t, bi) {          # [S x length(t)]
    beta %*% t(X_at(i, t)) + bi %*% t(Z_at(t))
  }

  t_end <- max(mi$T_surv)
  out <- vector("list", length(ids)); draws <- vector("list", length(ids))

  if (process == "longitudinal") {
    for (k in seq_along(ids)) {
      i <- pos[k]
      tt <- times
      if (is.null(tt)) {
        t_obs <- dl[[time_var]][as.character(dl[[id_var]]) == as.character(ids[k])]
        tt <- seq(min(c(t_obs, 0), na.rm = TRUE), mi$T_surv[i], length.out = 20)
      }
      D <- m_draws(i, tt, .jmjax_b_draws(ps, i, q))
      out[[k]] <- data.frame(id = ids[k], time = tt,
                             .jmjax_summarise_draws(D, level), row.names = NULL)
      draws[[k]] <- D
    }
  } else {
    alpha <- as.numeric(.jmjax_site_matrix(ps, "alpha"))
    gamma <- .jmjax_site_matrix(ps, "gamma")
    if (!is.null(gamma)) {
      ds_ord <- ds[match(as.character(mi$subj_ids), as.character(ds[[id_var]])), , drop = FALSE]
      Wsurv <- build_baseline_covariates(mi$surv_formula, ds_ord)
    }
    is_spline <- !is.null(object$spline_info)
    if (is_spline) {
      Wsp <- .jmjax_site_matrix(ps, "W")
    } else {
      log_lambda0 <- as.numeric(.jmjax_site_matrix(ps, "log_lambda0"))
      shape <- as.numeric(.jmjax_site_matrix(ps, "shape"))
    }
    log_h0 <- function(s) {                 # [S x length(s)]
      if (is_spline) return(Wsp %*% t(object$spline_info$basis(s)))
      outer(log(shape), rep(1, length(s))) +
        outer(shape - 1, log(s)) + log_lambda0
    }
    # Composite Gauss-Legendre: the interval is split at every prediction
    # time and at every spline knot, and an 8-point rule is used on each
    # piece. The spline baseline hazard has a kink in a higher derivative
    # at each knot, which a single rule across the whole interval
    # integrates only to ~1e-5; on knot-free pieces the integrand is
    # smooth and the rule is accurate to ~1e-10 (checked against
    # stats::integrate() in test-predict.R).
    gl <- gauss_legendre_01(8L)
    brk <- if (!is.null(object$spline_info)) unique(object$spline_info$knots) else numeric(0)
    t0_all <- if (is.null(t_from)) mi$T_surv[pos] else rep_len(t_from, length(ids))

    for (k in seq_along(ids)) {
      i <- pos[k]; t0 <- t0_all[k]
      uu <- if (is.null(times)) seq(t0, t_end, length.out = 20) else sort(times)
      if (any(uu > t_end)) {
        warning("times after the last follow-up time in the fit (", signif(t_end, 4),
                ") dropped: the baseline hazard is not defined there.", call. = FALSE)
        uu <- uu[uu <= t_end]
      }
      if (any(uu < t0)) {
        warning("times before t_from dropped for subject ", ids[k], call. = FALSE)
        uu <- uu[uu >= t0]
      }
      if (!length(uu)) next
      bi <- .jmjax_b_draws(ps, i, q)
      lin <- if (is.null(gamma)) rep(0, S) else as.numeric(gamma %*% Wsurv[i, ])
      # Pieces: consecutive points of t_from, the prediction times and the
      # knots in between. Integrate each piece once, then accumulate, so
      # the cumulative hazard at u_j is the sum of the pieces up to u_j.
      grid <- sort(unique(c(t0, uu, brk[brk > t0 & brk < max(uu)])))
      G <- length(grid) - 1L
      K <- length(gl$nodes)
      if (G > 0L) {
        a <- grid[-length(grid)]; w_len <- diff(grid)
        s <- as.vector(outer(gl$nodes, w_len) + matrix(a, K, G, byrow = TRUE))
        s_eval <- pmax(s, 1e-12)                       # log(s) for Weibull at t0 = 0
        h <- exp(log_h0(s_eval) + lin + alpha * m_draws(i, s_eval, bi))  # [S x (K*G)]
        Wm <- matrix(0, K * G, G)
        Wm[cbind(seq_len(K * G), rep(seq_len(G), each = K))] <-
          as.vector(outer(gl$weights, w_len))
        Hseg <- h %*% Wm                               # [S x G], one column per piece
        Hcum <- cbind(0, Hseg %*% upper.tri(diag(G), diag = TRUE))  # at each grid point
      } else {
        Hcum <- matrix(0, S, 1L)
      }
      H <- Hcum[, match(uu, grid), drop = FALSE]
      D <- exp(-H)
      out[[k]] <- data.frame(id = ids[k], t_from = t0, time = uu,
                             .jmjax_summarise_draws(D, level), row.names = NULL)
      draws[[k]] <- D
    }
  }

  res <- do.call(rbind, out)
  if (isTRUE(return_draws)) attr(res, "draws") <- stats::setNames(draws, as.character(ids))
  res
}
