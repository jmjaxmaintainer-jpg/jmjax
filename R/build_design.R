# ==============================================================================
# Build the numeric arrays the Python backend needs, from R formulas + data.
# This is the "R does formula parsing, Python does numerics" boundary.
# ==============================================================================

#' Build padded longitudinal arrays (subject x max_obs), analogous to the
#' subj_time / subj_y / subj_n arrays built by hand in the prototyping scripts.
#'
#' @param long_formula A formula for the fixed effects, e.g. y ~ time.
#' @param data_long Long-format longitudinal data.
#' @param id_var Name of the subject id column (must be integer-codeable,
#'   0-indexed after conversion, to match array indexing on the Python side).
#' @return A list with X_long (padded [N_sub, max_obs, p] design array),
#'   y_long (padded [N_sub, max_obs]), n_obs ([N_sub] integer vector), and
#'   subj_ids (sorted unique ids, defines row order used everywhere else).
#' @keywords internal
build_long_arrays <- function(long_formula, data_long, id_var) {
  subj_ids <- sort(unique(data_long[[id_var]]))
  N_sub <- length(subj_ids)
  id_index <- match(data_long[[id_var]], subj_ids)  # 1-indexed row per obs

  mf <- stats::model.frame(long_formula, data = data_long)
  X <- stats::model.matrix(long_formula, mf)          # [N_obs, p]
  y <- stats::model.response(mf)                       # [N_obs]
  p <- ncol(X)

  n_obs <- as.integer(table(factor(id_index, levels = seq_len(N_sub))))
  max_obs <- max(n_obs)

  X_long <- array(0, dim = c(N_sub, max_obs, p))
  y_long <- matrix(0, N_sub, max_obs)

  # Fill row-by-row per subject, preserving within-subject order as given
  row_cursor <- rep(0L, N_sub)
  for (i in seq_len(nrow(X))) {
    s <- id_index[i]
    row_cursor[s] <- row_cursor[s] + 1L
    X_long[s, row_cursor[s], ] <- X[i, ]
    y_long[s, row_cursor[s]] <- y[i]
  }

  list(
    X_long = X_long,      # [N_sub, max_obs, p]
    y_long = y_long,       # [N_sub, max_obs]
    n_obs = n_obs,         # [N_sub]
    subj_ids = subj_ids,   # length N_sub, defines row order
    long_terms = attr(mf, "terms"),
    p_long = p
  )
}

#' Build the padded random-effects design array Z_long from an INDEPENDENT
#' random-effects formula, using the same subject ordering/padding already
#' established by \code{build_long_arrays()} for \code{X_long} (same
#' \code{subj_ids}, same \code{max_obs}) so the two arrays align row-for-
#' row without needing to be built from the same formula.
#'
#' Added to fix a design assumption that only happened to be correct for
#' \code{long_formula = y ~ time}: previously, \code{Z} was always derived
#' by slicing \code{X}'s first \code{q} columns (\code{Z = X[, 1:q]}),
#' which silently breaks for a non-trivial fixed-effects formula (e.g.
#' \code{y ~ time + I(time^2)} or \code{y ~ splines::ns(time, 3)}) where
#' the first columns are not simply "intercept" and "slope". Evaluating
#' \code{random_formula} independently, the same way \code{long_formula}
#' is evaluated for \code{X}, removes that assumption entirely.
#'
#' @param random_formula A one-sided formula for the random-effects design,
#'   e.g. \code{~ time} for a random intercept and slope, or \code{~ 1}
#'   for a random intercept only. Evaluated against \code{data_long} the
#'   same way \code{long_formula} is - completely independent of what
#'   \code{long_formula} itself contains.
#' @param data_long Long-format longitudinal data (same data frame used to
#'   build \code{X_long} - see \code{build_long_arrays()}).
#' @param id_var Name of the subject id column.
#' @param subj_ids Sorted unique ids from \code{build_long_arrays()} -
#'   MUST be the same value used there, so \code{Z_long}'s rows align with
#'   \code{X_long}'s.
#' @param max_obs Max observations per subject from \code{build_long_arrays()}
#'   (\code{max(n_obs)}) - ensures \code{Z_long}'s padding matches
#'   \code{X_long}'s exactly.
#' @return \code{Z_long}: a padded \code{[N_sub, max_obs, q]} array, where
#'   \code{q = ncol(model.matrix(random_formula, data_long))}.
#' @keywords internal
build_random_long_array <- function(random_formula, data_long, id_var, subj_ids, max_obs) {
  N_sub <- length(subj_ids)
  id_index <- match(data_long[[id_var]], subj_ids)

  rhs_terms <- stats::terms(random_formula)  # already one-sided; no response to delete
  mf_z <- stats::model.frame(rhs_terms, data = data_long)  # default na.action (na.omit)
  Z <- stats::model.matrix(rhs_terms, mf_z)

  if (nrow(Z) != nrow(data_long)) {
    stop("random_formula's variables have a different missingness pattern than ",
         "long_formula's - this would misalign Z_long with X_long. Ensure both ",
         "formulas reference only complete-case variables, or handle NAs ",
         "consistently before calling jm_fit().")
  }

  q <- ncol(Z)
  Z_long <- array(0, dim = c(N_sub, max_obs, q), dimnames = list(NULL, NULL, colnames(Z)))

  # Same fill order as build_long_arrays() (row-by-row per subject,
  # preserving within-subject order as given) - required for row-for-row
  # alignment with X_long.
  row_cursor <- rep(0L, N_sub)
  for (i in seq_len(nrow(Z))) {
    s <- id_index[i]
    row_cursor[s] <- row_cursor[s] + 1L
    Z_long[s, row_cursor[s], ] <- Z[i, ]
  }

  Z_long
}

#' Build the survival-side arrays: event times, event indicator, and the
#' Gauss-Kronrod time-quadrature grid needed for the cumulative hazard
#' integral (used by both the Weibull and spline baseline backends).
#'
#' @param surv_formula A survival::Surv() formula, e.g. Surv(time, event) ~ 1.
#'   Only intercept-only relative-risk models are currently supported (the
#'   baseline hazard carries all the time-dependence; additional baseline
#'   covariates are a natural v2 extension once this path is validated).
#' @param data_surv One row per subject, in the SAME subject order as
#'   subj_ids from build_long_arrays() - reordered internally to guarantee this.
#' @param subj_ids Sorted unique ids from build_long_arrays(), defines row order.
#' @param id_var Name of the subject id column in data_surv.
#' @param gk_order Number of Gauss-Kronrod nodes for the time quadrature
#'   (10, matching the prototyping scripts; JM itself defaults GKk=15 for
#'   non-piecewise methods - keep these aligned if exact parity matters).
#' @keywords internal
build_surv_arrays <- function(surv_formula, data_surv, subj_ids, id_var, gk_order = 10) {
  data_surv <- data_surv[match(subj_ids, data_surv[[id_var]]), , drop = FALSE]

  surv_resp <- eval(surv_formula[[2]], data_surv)  # the Surv(time, event) object
  T_surv <- surv_resp[, 1]
  event <- surv_resp[, 2]

  gk <- gauss_kronrod_nodes(gk_order)  # see quadrature.R
  t_quad <- outer(T_surv, gk$nodes)     # [N_sub, gk_order]

  list(
    T_surv = T_surv,
    event = event,
    t_quad = t_quad,
    gk_weights = gk$weights
  )
}

#' Extract the baseline (time-constant) covariate design matrix from
#' \code{surv_formula}'s RHS, e.g. \code{Surv(time, event) ~ age + sex}.
#'
#' The intercept column is explicitly dropped: a global multiplicative
#' constant on the hazard is perfectly collinear with the baseline
#' hazard's own intercept (\code{log_lambda0} for Weibull, the spline
#' coefficients' own level for the spline baseline) - exactly why
#' \code{coxph()} itself never includes an intercept term either.
#'
#' @param surv_formula A \code{survival::Surv()} formula, e.g.
#'   \code{Surv(time, event) ~ age + sex}. \code{~ 1} (no covariates) is
#'   valid and returns a matrix with 0 columns.
#' @param data_surv One-row-per-subject data frame, already reordered to
#'   match \code{subj_ids}'s row order (see \code{build_surv_arrays()}).
#' @return A \code{[N_sub, n_gamma]} matrix (\code{n_gamma} may be 0).
#' @keywords internal
build_baseline_covariates <- function(surv_formula, data_surv) {
  rhs_terms <- stats::delete.response(stats::terms(surv_formula))
  W <- stats::model.matrix(rhs_terms, data_surv)
  if ("(Intercept)" %in% colnames(W)) {
    W <- W[, colnames(W) != "(Intercept)", drop = FALSE]
  }
  W
}

#' Extract baseline (time-constant) covariates referenced in
#' \code{long_formula} beyond \code{time_var} itself (e.g. \code{y ~ time
#' + age} - extracts \code{age}), one row per subject in \code{subj_ids}
#' order, for use with \code{build_time_design()}'s \code{baseline_covariates}
#' argument.
#'
#' Validates that each such covariate is genuinely CONSTANT within every
#' subject - jmjax currently only supports baseline (time-constant)
#' longitudinal covariates, not genuinely time-varying ones (a
#' substantially harder, separate modeling problem, deliberately out of
#' scope - see package development notes).
#'
#' @param long_formula The longitudinal fixed-effects formula.
#' @param time_var Name of the time variable in long_formula.
#' @param data_long Long-format longitudinal data.
#' @param id_var Name of the subject id column.
#' @param subj_ids Sorted unique ids (row order to match).
#' @return \code{NULL} if long_formula references no covariates beyond
#'   time_var (preserves the original behavior exactly), otherwise a data
#'   frame with one row per subject (in subj_ids order) and one column
#'   per baseline covariate.
#' @keywords internal
extract_baseline_covariates_long <- function(long_formula, time_var, data_long, id_var, subj_ids) {
  all_vars <- all.vars(long_formula)
  response_var <- all_vars[1]
  covariate_vars <- setdiff(all_vars, c(response_var, time_var))
  if (length(covariate_vars) == 0) return(NULL)

  for (v in covariate_vars) {
    n_unique_per_subject <- tapply(data_long[[v]], data_long[[id_var]], function(x) length(unique(x)))
    if (any(n_unique_per_subject > 1)) {
      bad_id <- names(n_unique_per_subject)[which(n_unique_per_subject > 1)[1]]
      stop("Covariate '", v, "' in long_formula varies WITHIN subject '", bad_id, "' - ",
           "jmjax currently only supports BASELINE (time-constant) longitudinal ",
           "covariates, not genuinely time-varying ones (a separate, harder ",
           "modeling problem - see package development notes). If '", v, "' should ",
           "be constant, check your data for typos or inconsistent duplicate rows.")
    }
  }

  first_rows <- data_long[!duplicated(data_long[[id_var]]), c(id_var, covariate_vars), drop = FALSE]
  first_rows <- first_rows[match(subj_ids, first_rows[[id_var]]), , drop = FALSE]
  first_rows[, covariate_vars, drop = FALSE]
}

#' Evaluate the longitudinal formula's FIXED-EFFECTS design matrix at
#' arbitrary time values, needed for the shared m_i(t) = X(t) beta + b_i
#' term in the survival hazard (this is what real JM calls Xtime/Xs).
#'
#' (The matrix product is written by juxtaposition on purpose. The percent
#' sign opens a comment in Rd and swallows the rest of the line, and
#' roxygen2 does not escape it for you - this title previously used the
#' matrix-multiply operator and was silently truncated at it. Elsewhere in
#' the package the escaped form is used inside \code{}; here there is no
#' need to write one at all.)
#'
#' @param long_formula The longitudinal fixed-effects formula.
#' @param time_var Name of the time variable in long_formula.
#' @param t_values Numeric vector (evaluated once, e.g. at T_surv) OR numeric
#'   matrix [N_sub, n_quad] (evaluated column-by-column, e.g. at t_quad).
#' @param baseline_covariates Optional data frame (see
#'   \code{extract_baseline_covariates_long()}), one row per subject in
#'   the SAME order as \code{t_values}'s rows, carrying forward each
#'   subject's baseline covariate value(s) when evaluating X at a
#'   synthetic time point (t_values isn't itself drawn from data_long, so
#'   it has no covariate columns of its own to reuse). \code{NULL}
#'   (default) preserves the original time_var-only behavior exactly.
#' @return If t_values is a vector: a [N_sub, p] matrix. If a matrix: a
#'   [N_sub, n_quad, p] array.
#' @keywords internal
build_time_design <- function(long_formula, time_var, t_values, baseline_covariates = NULL) {
  rhs_terms <- stats::delete.response(stats::terms(long_formula))

  eval_at <- function(t_vec) {
    newdata <- data.frame(t_vec)
    names(newdata) <- time_var
    if (!is.null(baseline_covariates)) {
      newdata <- cbind(newdata, baseline_covariates)
    }
    stats::model.matrix(rhs_terms, newdata)
  }

  if (is.matrix(t_values)) {
    first_col <- eval_at(t_values[, 1])
    p <- ncol(first_col)
    out <- array(0, dim = c(nrow(t_values), ncol(t_values), p),
                 dimnames = list(NULL, NULL, colnames(first_col)))
    out[, 1, ] <- first_col
    for (k in seq_len(ncol(t_values))[-1]) out[, k, ] <- eval_at(t_values[, k])
    out
  } else {
    eval_at(t_values)
  }
}

#' Parse a functional_forms specification into a vector of requested
#' association type names, following JMbayes2's syntax
#' (\code{~ value(y) + delta(y)}) but simplified for jmjax's current
#' single-longitudinal-outcome scope: the argument inside each term (e.g.
#' the \code{y} in \code{value(y)}) is validated against the longitudinal
#' formula's response variable but doesn't otherwise affect computation,
#' since there is only one outcome to associate. This keeps the syntax
#' forward-compatible with a future multi-outcome
#' \code{list(y = ~ value(y) + delta(y))} form without requiring it now.
#'
#' @param functional_forms \code{NULL} (default: value-only, jmjax's
#'   original behavior) or a one-sided formula built from
#'   \code{value()}/\code{delta()}/\code{area()}/\code{area_avg()} terms,
#'   e.g. \code{~ value(y) + delta(y)}. \code{area(t) = integral_0^t m_i(s)
#'   ds} is the RAW cumulative integral; \code{area_avg(t) = area(t) / t}
#'   is the TIME-AVERAGED version - confirmed by direct inspection of a
#'   fitted \code{JMbayes2} object's \code{model_data$X_h} during
#'   development that \code{JMbayes2::area()} computes \code{area_avg},
#'   NOT the raw integral, despite the name - the two are NOT simply a
#'   constant rescaling of each other (the divisor \code{t} differs by
#'   subject), so a fitted \code{alpha} under one convention cannot be
#'   converted to the other's scale after the fact. Use \code{area_avg()}
#'   specifically when comparing against or replicating a \code{JMbayes2}
#'   model that uses its \code{area()}.
#' @param response_var Name of the longitudinal response variable (from
#'   \code{long_formula}), used only to validate the argument inside each
#'   term and produce a clearer error message on a likely typo.
#' @return A character vector of requested association types, e.g.
#'   \code{c("value", "delta")}. Always includes at least one type.
#' @keywords internal
parse_functional_forms <- function(functional_forms, response_var) {
  if (is.null(functional_forms)) return("value")

  tt <- stats::terms(functional_forms)
  term_labels <- attr(tt, "term.labels")
  if (length(term_labels) == 0) {
    stop("functional_forms must contain at least one of value()/delta(), ",
         "e.g. ~ value(y) or ~ value(y) + delta(y).")
  }

  # Each term looks like "value(y)" or "delta(y)" - split into type and arg.
  parsed <- lapply(term_labels, function(lbl) {
    m <- regmatches(lbl, regexec("^([a-zA-Z_]+)\\((.*)\\)$", lbl))[[1]]
    if (length(m) != 3) {
      stop("Could not parse functional_forms term '", lbl, "' - expected ",
           "a function-call form like value(", response_var, ") or delta(",
           response_var, ").")
    }
    list(type = m[2], arg = trimws(m[3]))
  })

  types <- vapply(parsed, `[[`, character(1), "type")
  args <- vapply(parsed, `[[`, character(1), "arg")

  # SLOPE_NOTE: "slope" is planned but not yet implemented (needs
  # differentiating the design matrix w.r.t. time, real work once
  # long_formula supports non-linear-in-time bases like splines) - named
  # explicitly here rather than falling through to a generic "unsupported"
  # message so a user trying it gets a clear signal this is a known, scoped
  # -for-later gap, not a typo.
  valid_types <- c("value", "delta", "area", "area_avg")
  planned_types <- c("slope")
  unsupported <- setdiff(types, c(valid_types, planned_types))
  if (length(unsupported) > 0) {
    stop("Unknown functional form(s): ", paste(unsupported, collapse = ", "),
         ". Supported: ", paste(valid_types, collapse = ", "), ".")
  }
  not_yet <- intersect(types, planned_types)
  if (length(not_yet) > 0) {
    stop("functional form(s) ", paste(not_yet, collapse = ", "), " are ",
         "planned but not yet implemented. Currently supported: ",
         paste(valid_types, collapse = ", "), ".")
  }

  mismatched <- args[args != response_var]
  if (length(mismatched) > 0) {
    warning("functional_forms argument(s) (", paste(unique(mismatched), collapse = ", "),
            ") don't match the longitudinal response variable '", response_var,
            "' - jmjax currently has a single longitudinal outcome, so the ",
            "argument name doesn't affect computation, but a mismatch often ",
            "indicates a typo.")
  }

  unique(types)
}

#' Build the DELTA association channel's fixed-effects design:
#' \code{delta(t) = X(t) - X(0)} evaluated at the survival time and at each
#' quadrature node. For a pure random INTERCEPT (\code{random_effects =
#' "intercept"}), the random-effect contribution to delta is always exactly
#' zero (the intercept doesn't depend on time, so it cancels in the
#' subtraction) - this function therefore only handles the fixed-effects
#' part; the caller should not add a random-effect term for this channel
#' under that scope.
#'
#' @param long_formula The longitudinal fixed-effects formula.
#' @param time_var Name of the time variable in long_formula.
#' @param X_time_surv Already-built value-channel design at the survival
#'   times (\code{[N_sub, p]}), from \code{build_time_design()}.
#' @param X_time_quad Already-built value-channel design at the quadrature
#'   nodes (\code{[N_sub, n_quad, p]}), from \code{build_time_design()}.
#' @return A list with \code{X_delta_surv} (\code{[N_sub, p]}) and
#'   \code{X_delta_quad} (\code{[N_sub, n_quad, p]}).
#' @keywords internal
build_delta_channel <- function(long_formula, time_var, X_time_surv, X_time_quad) {
  n_sub <- nrow(X_time_surv)
  p <- ncol(X_time_surv)

  x_zero_row <- build_time_design(long_formula, time_var, 0)  # [1, p]

  X_zero_surv <- matrix(x_zero_row, nrow = n_sub, ncol = p, byrow = TRUE)
  X_delta_surv <- X_time_surv - X_zero_surv

  X_zero_quad <- array(0, dim = dim(X_time_quad))
  for (pp in seq_len(p)) {
    X_zero_quad[, , pp] <- x_zero_row[pp]
  }
  X_delta_quad <- X_time_quad - X_zero_quad

  list(X_delta_surv = X_delta_surv, X_delta_quad = X_delta_quad)
}

#' Build the AREA association channel's fixed-effects design:
#' \code{area(t) = integral_0^t X(s) ds}, evaluated at the survival time
#' (reusing the ALREADY-COMPUTED outer Gauss-Kronrod grid directly - it's
#' exactly the same integral already used for the cumulative hazard, just
#' of the design vector \code{X(s)} instead of the hazard itself, so no
#' extra quadrature is needed for this piece) and at each outer quadrature
#' node (requiring a genuine NESTED inner quadrature from 0 to that node,
#' since the upper integration limit differs node-by-node - this is the
#' real complexity increase relative to \code{delta}).
#'
#' @param long_formula The longitudinal fixed-effects formula.
#' @param time_var Name of the time variable in long_formula.
#' @param T_surv Observed event/censoring times.
#' @param t_quad Outer Gauss-Kronrod quadrature grid
#'   (\code{[N_sub, n_quad_outer]}), from \code{build_surv_arrays()}.
#' @param gk_weights_outer Weights matching \code{t_quad}'s construction.
#' @param gk_order_inner Number of nodes for the INNER (nested) quadrature.
#'   Defaults to 10, matching the outer scheme.
#' @return A list with \code{X_area_surv} (\code{[N_sub, p]}) and
#'   \code{X_area_quad} (\code{[N_sub, n_quad_outer, p]}).
#' @keywords internal
build_area_channel <- function(long_formula, time_var, T_surv, t_quad, gk_weights_outer,
                                gk_order_inner = 10) {
  n_sub <- nrow(t_quad)
  n_quad_outer <- ncol(t_quad)

  # area(T_i): reuses the EXISTING outer quadrature grid directly.
  X_time_quad_outer <- build_time_design(long_formula, time_var, t_quad)  # [N_sub, n_quad_outer, p]
  p <- dim(X_time_quad_outer)[3]

  X_area_surv <- matrix(0, n_sub, p)
  for (pp in seq_len(p)) {
    # matrix(..., nrow=, ncol=) explicitly reshapes regardless of whether
    # R's default [,,pp] slicing already dropped a size-1 dimension (e.g.
    # n_quad_outer == 1) - the same class of silent-dimension-drop footgun
    # that caused a real bug in the B_quad construction earlier in this
    # package's development (see jm_fit.R comments there).
    Xp <- matrix(X_time_quad_outer[, , pp], nrow = n_sub, ncol = n_quad_outer)
    X_area_surv[, pp] <- T_surv * as.vector(Xp %*% gk_weights_outer)
  }

  # area(s_k) for each outer node s_k: a GENUINE nested integral - the
  # upper integration limit (s_k) differs for every outer node, so each
  # needs its own inner Gauss-Kronrod quadrature from 0 to s_k.
  gk_inner <- gauss_kronrod_nodes(gk_order_inner)
  X_area_quad <- array(0, dim = c(n_sub, n_quad_outer, p))
  for (k in seq_len(n_quad_outer)) {
    s_k <- t_quad[, k]                          # [N_sub]
    u_inner_k <- outer(s_k, gk_inner$nodes)     # [N_sub, n_inner]
    X_inner_k <- build_time_design(long_formula, time_var, u_inner_k)  # [N_sub, n_inner, p]
    n_inner <- length(gk_inner$nodes)
    for (pp in seq_len(p)) {
      Xp <- matrix(X_inner_k[, , pp], nrow = n_sub, ncol = n_inner)
      X_area_quad[, k, pp] <- s_k * as.vector(Xp %*% gk_inner$weights)
    }
  }

  list(X_area_surv = X_area_surv, X_area_quad = X_area_quad)
}
