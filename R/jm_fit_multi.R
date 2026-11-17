# ==============================================================================
# Joint models with several longitudinal markers (jmjax 0.5.0, MCMC only).
#
# jm_fit() / jm_bayes() dispatch here when long_formula is a named list of two
# or more formulas. A list of length one is unwrapped and fitted by the
# single-marker code, so the list API and the old API give the same fit.
#
# Each marker k has its own fixed effects beta_k, residual SD sigma_e[k] and
# association parameter(s) alpha_k (one per functional form). The random
# effects are stacked, b_i = (b_i1, ..., b_iK), with a full covariance (LKJ
# prior; the default, which links the markers) or a block-diagonal one
# (control$re_structure = "block"). The likelihood and the rotation live in
# inst/python/jmjax_backend/multi_model.py.
#
# SCALES. As in the single-marker path, the model is fitted on rescaled data
# (time / c, standardized subject-level covariates) and reported on the
# user's scale. The maps back are found by least squares between the designs
# on the two scales and verified exact; a marker whose map is not exact
# (e.g. ns() of a covariate) is fitted on its original scale, with a message.
# ==============================================================================

.jmjax_is_multi <- function(long_formula) {
  is.list(long_formula) && !inherits(long_formula, "formula")
}

# Unwrap a one-marker list so the single-marker code fits it.
.jmjax_unwrap_single <- function(long_formula, random_formula, data_long,
                                 functional_forms, marker_var) {
  nm <- names(long_formula) %||% "1"
  pick <- function(x) {
    if (is.null(x) || inherits(x, "formula")) return(x)
    if (is.list(x)) return(if (!is.null(names(x))) x[[nm]] %||% x[[1]] else x[[1]])
    x
  }
  dl <- data_long
  if (is.data.frame(dl) && !is.null(marker_var)) {
    dl <- dl[as.character(dl[[marker_var]]) == nm, , drop = FALSE]
  } else if (is.list(dl) && !is.data.frame(dl)) {
    dl <- pick(dl)
  }
  list(long_formula = long_formula[[1]], random_formula = pick(random_formula),
       data_long = dl, functional_forms = pick(functional_forms))
}

# Subject-level covariates of a formula (all variables except the response
# and time), one row per subject in subj_ids order.
.mm_subject_covariates <- function(formula, time_var, data, id_var, subj_ids) {
  rhs <- stats::delete.response(stats::terms(formula))
  vars <- setdiff(all.vars(rhs), time_var)
  if (!length(vars)) return(NULL)
  for (v in vars) {
    nu <- tapply(data[[v]], data[[id_var]], function(x) length(unique(x)))
    if (any(nu > 1, na.rm = TRUE)) {
      stop("covariate '", v, "' varies within subject; only baseline (subject-",
           "level) covariates are supported in long_formula.", call. = FALSE)
    }
  }
  first <- data[!duplicated(data[[id_var]]), c(id_var, vars), drop = FALSE]
  out <- first[match(subj_ids, first[[id_var]]), vars, drop = FALSE]
  if (anyNA(match(subj_ids, first[[id_var]]))) {
    stop("a marker with subject-level covariates (", paste(vars, collapse = ", "),
         ") has no rows for some subjects, so their covariates are unknown. ",
         "Give every subject at least one row in that marker's data.", call. = FALSE)
  }
  out
}

# Design of one functional form of one formula at the event times [N, p]
# and at the quadrature nodes [N, n_quad, p]. Subject-level covariates are
# carried along, so every form works for markers with covariates.
.mm_form_design <- function(tt, time_var, form, T_surv, t_quad, gk_weights, bc,
                            gk_inner_order = 10L) {
  bd <- function(t) build_time_design(tt, time_var, t, bc)
  if (form == "value") {
    return(list(S = bd(T_surv), Q = bd(t_quad)))
  }
  if (form == "delta") {
    X0 <- bd(rep(0, length(T_surv)))
    S <- bd(T_surv) - X0
    XQ <- bd(t_quad)
    for (k in seq_len(dim(XQ)[2])) XQ[, k, ] <- XQ[, k, ] - X0
    return(list(S = S, Q = XQ))
  }
  # area and area_avg: integral_0^t x(u) du by Gauss-Kronrod on (0, t).
  gk_in <- gauss_kronrod_nodes(gk_inner_order)
  integ <- function(tv, nodes, w) {          # tv [N], returns [N, p]
    Xi <- bd(outer(tv, nodes))               # [N, n, p]
    p <- dim(Xi)[3]
    out <- matrix(0, length(tv), p)
    for (pp in seq_len(p)) out[, pp] <- tv * as.vector(matrix(Xi[, , pp], length(tv)) %*% w)
    out
  }
  gk_out_nodes <- t_quad[1, ] / T_surv[1]
  S <- integ(T_surv, gk_out_nodes, gk_weights)
  XQ <- array(0, dim = c(nrow(t_quad), ncol(t_quad), ncol(S)))
  for (k in seq_len(ncol(t_quad))) XQ[, k, ] <- integ(t_quad[, k], gk_in$nodes, gk_in$weights)
  if (form == "area_avg") {
    S <- S / T_surv
    for (k in seq_len(ncol(t_quad))) XQ[, k, ] <- XQ[, k, ] / t_quad[, k]
  }
  list(S = S, Q = XQ)
}

# Least-squares linear map A with X_orig %*% A == X_int, or NULL if inexact.
.mm_linear_map <- function(X_orig, X_int, tol = 1e-8) {
  A <- tryCatch(qr.solve(X_orig, X_int), error = function(e) NULL)
  if (is.null(A)) return(NULL)
  err <- max(abs(X_orig %*% A - X_int))
  if (!is.finite(err) || err > tol * max(1, max(abs(X_int)))) return(NULL)
  unname(as.matrix(A))
}

.jm_fit_multi <- function(long_formula, surv_formula, data_long, data_surv,
                          id_var, time_var, method, random_formula,
                          functional_forms, marker_var, control) {
  if (!method %in% c("spline-PH-mcmc", "weibull-PH-mcmc")) {
    stop("several longitudinal markers are supported by the Bayesian methods only ",
         "(jm_bayes(), method = 'spline-PH-mcmc' or 'weibull-PH-mcmc'); ",
         "maximum likelihood (jm_mle()) fits one marker.", call. = FALSE)
  }
  markers <- names(long_formula)
  K <- length(long_formula)
  if (is.null(markers) || any(!nzchar(markers)) || anyDuplicated(markers)) {
    stop("long_formula must be a list with unique, non-empty names, one per marker ",
         "(e.g. list(bili = log(serBilir) ~ year, alb = albumin ~ year)).", call. = FALSE)
  }
  if (!all(vapply(long_formula, inherits, logical(1), "formula"))) {
    stop("every element of long_formula must be a formula.", call. = FALSE)
  }
  per_marker <- function(x, what, default = NULL) {
    if (is.null(x)) return(stats::setNames(rep(list(default), K), markers))
    if (inherits(x, "formula")) return(stats::setNames(rep(list(x), K), markers))
    if (!is.list(x) || is.null(names(x)) || !all(markers %in% names(x))) {
      stop(what, " must be a formula applied to every marker, or a list named like ",
           "long_formula (", paste(markers, collapse = ", "), ").", call. = FALSE)
    }
    x[markers]
  }
  random_formula <- per_marker(random_formula, "random_formula")
  if (any(vapply(random_formula, is.null, logical(1)))) {
    stop("random_formula is required for several markers (e.g. ~ year, or a list ",
         "with one per marker).", call. = FALSE)
  }
  functional_forms <- per_marker(functional_forms, "functional_forms")

  # ---- data per marker ------------------------------------------------
  if (is.data.frame(data_long)) {
    if (is.null(marker_var) || !marker_var %in% names(data_long)) {
      stop("with several markers, data_long is either a named list of data frames ",
           "(one per marker) or one data frame with a marker column named by ",
           "marker_var.", call. = FALSE)
    }
    dls <- lapply(markers, function(m)
      data_long[as.character(data_long[[marker_var]]) == m, , drop = FALSE])
    names(dls) <- markers
  } else {
    if (!all(markers %in% names(data_long))) {
      stop("data_long must have one data frame per marker, named like long_formula.",
           call. = FALSE)
    }
    dls <- data_long[markers]
  }
  for (m in markers) {
    resp <- all.vars(long_formula[[m]])[1]
    keep <- stats::complete.cases(dls[[m]][, intersect(
      c(id_var, time_var, all.vars(long_formula[[m]]), all.vars(random_formula[[m]])),
      names(dls[[m]])), drop = FALSE])
    dls[[m]] <- dls[[m]][keep, , drop = FALSE]
    if (!nrow(dls[[m]])) stop("marker '", m, "' has no complete rows.", call. = FALSE)
  }
  ids_long <- unique(unlist(lapply(dls, function(d) d[[id_var]])))
  subj_ids <- sort(ids_long)
  if (anyNA(match(subj_ids, data_surv[[id_var]]))) {
    stop("some subjects with longitudinal data are missing from data_surv.", call. = FALSE)
  }
  N <- length(subj_ids)
  surv_arr <- build_surv_arrays(surv_formula, data_surv, subj_ids, id_var,
                                gk_order = control$gk_order %||% 10)
  dsurv_ord <- data_surv[match(subj_ids, data_surv[[id_var]]), , drop = FALSE]
  W_surv <- build_baseline_covariates(surv_formula, dsurv_ord)
  if (!ncol(W_surv)) W_surv <- NULL

  # ---- internal scale: time / c, standardized subject-level covariates
  st <- control$scale_time %||% "auto"
  c_time <- 1
  if (!isFALSE(st)) {
    c_time <- if (is.numeric(st)) as.numeric(st)[1] else
      max(abs(unlist(lapply(dls, function(d) d[[time_var]]))), na.rm = TRUE)
    if (!is.finite(c_time) || c_time <= 0) c_time <- 1
  }
  do_std <- !isFALSE(control$standardize_covariates)
  internal_data <- function(m, scale_on) {
    d <- dls[[m]]
    if (!scale_on) return(d)
    d[[time_var]] <- d[[time_var]] / c_time
    if (do_std) {
      cand <- setdiff(all.vars(stats::delete.response(stats::terms(long_formula[[m]]))),
                      c(time_var, all.vars(random_formula[[m]])))
      for (v in cand) {
        if (!is.numeric(d[[v]])) next
        first <- d[!duplicated(d[[id_var]]), v]
        s <- stats::sd(first)
        if (is.finite(s) && s > 0) d[[v]] <- (d[[v]] - mean(first)) / s
      }
    }
    d
  }

  # ---- designs per marker -------------------------------------------
  blocks <- vector("list", K); names(blocks) <- markers
  notes <- character(0)
  for (m in markers) {
    lf <- long_formula[[m]]; rf <- random_formula[[m]]
    d0 <- dls[[m]]
    tl0 <- .jmjax_fixed_terms(lf, d0); tr0 <- .jmjax_fixed_terms(rf, d0)
    X0 <- stats::model.matrix(tl0, stats::model.frame(tl0, d0))
    Z0 <- stats::model.matrix(tr0, stats::model.frame(tr0, d0))
    use_scale <- !isFALSE(st) || do_std
    A <- diag(ncol(X0)); B <- diag(ncol(Z0)); di <- d0; ck <- 1
    if (use_scale) {
      di <- internal_data(m, TRUE)
      tli <- .jmjax_fixed_terms(lf, di); tri <- .jmjax_fixed_terms(rf, di)
      Xi <- stats::model.matrix(tli, stats::model.frame(tli, di))
      Zi <- stats::model.matrix(tri, stats::model.frame(tri, di))
      Am <- .mm_linear_map(X0, Xi); Bm <- .mm_linear_map(Z0, Zi)
      if (!is.null(Am) && !is.null(Bm)) {
        A <- Am; B <- Bm; ck <- if (isFALSE(st)) 1 else c_time
      } else {
        di <- d0
        notes <- c(notes, m)
      }
    }
    tl <- .jmjax_fixed_terms(lf, di); tr <- .jmjax_fixed_terms(rf, di)
    X <- stats::model.matrix(tl, stats::model.frame(tl, di))
    Z <- stats::model.matrix(tr, stats::model.frame(tr, di))
    y <- stats::model.response(stats::model.frame(stats::terms(lf), di))
    bcx <- .mm_subject_covariates(lf, time_var, di, id_var, subj_ids)
    bcz <- .mm_subject_covariates(rf, time_var, di, id_var, subj_ids)
    forms <- if (is.null(functional_forms[[m]])) "value" else
      parse_functional_forms(functional_forms[[m]], all.vars(lf)[1])
    Ts <- surv_arr$T_surv / ck; tq <- surv_arr$t_quad / ck
    chans <- lapply(forms, function(f) {
      xs <- .mm_form_design(tl, time_var, f, Ts, tq, surv_arr$gk_weights, bcx)
      zs <- .mm_form_design(tr, time_var, f, Ts, tq, surv_arr$gk_weights, bcz)
      # area/area_avg integrate over internal time; the integral of x over
      # (0, t/c) in internal time is (1/c) times the integral over (0, t),
      # so the channel's alpha would be on a rescaled axis. Rescale the
      # design instead, so alpha keeps the user's units.
      if (f == "area") { xs$S <- xs$S * ck; xs$Q <- xs$Q * ck; zs$S <- zs$S * ck; zs$Q <- zs$Q * ck }
      list(form = f, XS = xs$S, XQ = xs$Q, ZS = zs$S, ZQ = zs$Q)
    })
    blocks[[m]] <- list(X = X, Z = Z, y = as.numeric(y),
                        subj = match(di[[id_var]], subj_ids) - 1L,
                        x_names = colnames(X0), z_names = colnames(Z0),
                        A = A, B = B, chans = chans, data_int = di, lf = lf, rf = rf)
  }
  if (length(notes)) {
    message("jm_bayes(): marker(s) ", paste(notes, collapse = ", "), " fitted on ",
            "the original scale (the rescaled design is not an exact linear map ",
            "of the original one).")
  }

  # ---- stack: block-diagonal X and Z, flat rows ----------------------
  p <- vapply(blocks, function(b) ncol(b$X), integer(1))
  q <- vapply(blocks, function(b) ncol(b$Z), integer(1))
  P <- sum(p); Q <- sum(q)
  po <- c(0L, cumsum(p)); qo <- c(0L, cumsum(q))
  n_k <- vapply(blocks, function(b) nrow(b$X), integer(1))
  Xf <- matrix(0, sum(n_k), P); Zf <- matrix(0, sum(n_k), Q)
  yf <- numeric(0); sf <- integer(0); mf <- integer(0); r0 <- 0L
  for (k in seq_len(K)) {
    b <- blocks[[k]]; rr <- r0 + seq_len(n_k[k])
    Xf[rr, po[k] + seq_len(p[k])] <- b$X
    Zf[rr, qo[k] + seq_len(q[k])] <- b$Z
    yf <- c(yf, b$y); sf <- c(sf, b$subj); mf <- c(mf, rep(k - 1L, n_k[k]))
    r0 <- r0 + n_k[k]
  }
  chan_list <- unlist(lapply(seq_len(K), function(k)
    lapply(blocks[[k]]$chans, function(ch) c(ch, k = k))), recursive = FALSE)
  C <- length(chan_list); nq <- ncol(surv_arr$t_quad)
  XS <- array(0, c(C, N, P)); ZS <- array(0, c(C, N, Q))
  XQ <- array(0, c(C, N, nq, P)); ZQ <- array(0, c(C, N, nq, Q))
  alpha_names <- character(C)
  for (cc in seq_len(C)) {
    ch <- chan_list[[cc]]; k <- ch$k
    XS[cc, , po[k] + seq_len(p[k])] <- ch$XS
    ZS[cc, , qo[k] + seq_len(q[k])] <- ch$ZS
    XQ[cc, , , po[k] + seq_len(p[k])] <- ch$XQ
    ZQ[cc, , , qo[k] + seq_len(q[k])] <- ch$ZQ
    alpha_names[cc] <- if (ch$form == "value") paste0("alpha_", markers[k]) else
      paste0("alpha_", markers[k], "_", ch$form)
  }
  A_all <- matrix(0, P, P); B_all <- matrix(0, Q, Q)
  for (k in seq_len(K)) {
    A_all[po[k] + seq_len(p[k]), po[k] + seq_len(p[k])] <- blocks[[k]]$A
    B_all[qo[k] + seq_len(q[k]), qo[k] + seq_len(q[k])] <- blocks[[k]]$B
  }
  beta_names <- unlist(lapply(seq_len(K), function(k)
    paste0("beta_", markers[k], "_", seq_len(p[k]) - 1L)))
  sigma_e_names <- paste0("sigma_e_", markers)
  re_names <- unlist(lapply(seq_len(K), function(k)
    paste0(markers[k], ":", blocks[[k]]$z_names)))

  # ---- empirical-Bayes priors: one lme() per marker -----------------
  prefit <- stats::setNames(rep("none", K), markers)
  ctl_py <- control
  if (!isFALSE(control$empirical_bayes_prior)) {
    bm <- numeric(0); bs <- numeric(0); sem <- numeric(0); sbm <- numeric(0)
    for (k in seq_len(K)) {
      b <- blocks[[k]]
      pf <- .jmjax_lme_prefit(b$lf, b$rf, id_var, b$data_int,
                              nlme::lmeControl(opt = "optim"))
      prefit[k] <- pf$structure
      if (!is.null(pf$fit)) {
        bm <- c(bm, unname(nlme::fixef(pf$fit)))
        bs <- c(bs, sqrt(pmin(14400 * diag(stats::vcov(pf$fit)), 1000)))
        sem <- c(sem, stats::sigma(pf$fit))
        sbm <- c(sbm, sqrt(diag(as.matrix(nlme::getVarCov(pf$fit)))))
      } else {
        # Data-free for beta; the residual and random-effect scales from
        # the marker's own spread so the Gamma priors stay on its scale.
        bm <- c(bm, rep(0, p[k])); bs <- c(bs, rep(5, p[k]))
        sy <- stats::sd(b$y)
        sem <- c(sem, sy / 2); sbm <- c(sbm, rep(sy / 2, q[k]))
      }
    }
    if (length(bm) == P) {
      ctl_py$beta_prior_mean <- bm; ctl_py$beta_prior_sd <- bs
      ctl_py$sigma_e_prior_mean <- sem; ctl_py$sigma_b_prior_mean <- sbm
    }
  }
  if (any(prefit == "diagonal")) {
    message("jm_bayes(): the lme() pre-fit used a diagonal random-effects covariance ",
            "for marker(s) ", paste(markers[prefit == "diagonal"], collapse = ", "),
            "; the correlations keep their LKJ prior.")
  }

  # ---- baseline hazard -------------------------------------------------
  spline_info <- NULL
  if (method == "spline-PH-mcmc") {
    spline_info <- build_spline_knots(
      surv_arr$T_surv, n_interior = as.integer(control$n_interior_knots %||% 5L),
      ord = as.integer(control$spline_order %||% 4L),
      placement = control$knot_placement %||% "quantile", t_quad = surv_arr$t_quad)
    B_T <- spline_info$basis(surv_arr$T_surv)
    B_quad <- array(0, dim = c(N, nq, spline_info$n_splines))
    for (k in seq_len(nq)) B_quad[, k, ] <- spline_info$basis(surv_arr$t_quad[, k])
    n_splines <- as.integer(spline_info$n_splines)
  } else {
    B_T <- matrix(0, N, 1); B_quad <- array(0, dim = c(N, nq, 1)); n_splines <- 1L
  }
  ctl_py$baseline_hazard <- if (method == "spline-PH-mcmc") "spline" else "weibull"
  ctl_py$re_structure <- control$re_structure %||% "full"
  ctl_py$lkj_concentration <- control$lkj_concentration %||% 3.0

  backend <- .get_backend()
  args <- list(y = yf, X = Xf, Z = Zf, subj = as.integer(sf), marker = as.integer(mf),
               XS = XS, ZS = ZS, XQ = XQ, ZQ = ZQ,
               T_surv = as.numeric(surv_arr$T_surv), event = as.numeric(surv_arr$event),
               t_quad = surv_arr$t_quad, gk_weights = as.numeric(surv_arr$gk_weights),
               B_T = B_T, B_quad = B_quad, n_splines = n_splines,
               block_sizes = as.integer(q), beta_names = beta_names,
               sigma_e_names = sigma_e_names, alpha_names = alpha_names,
               beta_transform = A_all, re_transform = B_all, W_surv = W_surv,
               control = ctl_py)
  res <- do.call(backend$multi_model$fit_nuts_multi, args)

  cv <- res$convergence
  cv$precision <- .backend_precision()
  cv$prefit <- as.list(prefit)
  structure(
    list(
      call = .jmjax_named_call(sys.call(-1), "jm_fit"),
      method = method,
      long_formula = long_formula,
      surv_formula = surv_formula,
      spline_info = spline_info,
      estimates = unlist(res$estimates),
      se = unlist(res$se),
      vcov = NULL, loglik = NULL,
      convergence = cv,
      backend_data = if (isTRUE(control$return_backend_data)) args else NULL,
      scale_time_status = if (isFALSE(st)) "off" else sprintf("applied (c = %.6g)", c_time),
      diagnostics = res$diagnostics,
      random_effects = res$random_effects,
      n_subjects = N,
      n_obs_long = sum(n_k),
      n_events = sum(surv_arr$event),
      mcmc_settings = list(num_warmup = .mcmc_n_warmup(control),
                           num_samples = .mcmc_n_samples(control),
                           num_chains = .mcmc_n_chains(control)),
      posterior_samples = res$posterior_samples,
      model_info = list(
        multi = TRUE, markers = markers, id_var = id_var, time_var = time_var,
        long_formula = long_formula, random_formula = random_formula,
        surv_formula = surv_formula, re_names = re_names,
        re_structure = ctl_py$re_structure,
        beta_map = data.frame(
          name = beta_names,
          marker = rep(markers, p),
          term = unlist(lapply(blocks, `[[`, "x_names")), stringsAsFactors = FALSE),
        alpha_names = alpha_names, sigma_e_names = sigma_e_names,
        n_obs_per_marker = n_k,
        subj_ids = subj_ids,
        T_surv = as.numeric(surv_arr$T_surv), event = as.numeric(surv_arr$event)),
      data = if (isFALSE(control$keep_data)) NULL else
        list(long = dls, surv = data_surv)
    ),
    class = "jmjax")
}
