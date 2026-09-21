#' Fit a joint model from pre-fitted `lme` and `coxph` objects
#'
#' A convenience wrapper around [jm_fit()] that takes the two fitted
#' component models directly, in the style of `JMbayes2::jm()`, rather than
#' restating the formulas, grouping variable and random-effects structure.
#' Everything it can recover from the objects is recovered; everything it
#' cannot represent is rejected with an explicit error rather than silently
#' ignored.
#'
#' @details
#' **Why fit the components first.** The longitudinal and survival submodels
#' are the constituent parts of a joint model, and fitting them separately
#' first is good practice regardless: convergence failures, questionable
#' random-effects structures and implausible variance estimates surface in
#' seconds rather than after a long MCMC run. Having them in hand also makes
#' the standard diagnostic comparison - joint-model estimates against the
#' separate fits - natural rather than an extra step.
#'
#' **What is derived from `lme_object`:** the longitudinal formula, the
#' random-effects formula, the grouping variable, the longitudinal data, and
#' whether the model has a random intercept only or an intercept and slope.
#'
#' **What is derived from `cox_object`:** the survival formula, and the
#' coefficients used to centre the prior on the survival covariate effects
#' (`gamma`), matching `JMbayes2`'s `mean_gammas`.
#'
#' **`data_surv` is optional, but often needed.** jmjax first tries to
#' recover it by evaluating the data symbol recorded in `cox_object$call`,
#' the same approach `JMbayes2::jm()` takes. That works when `coxph()` was
#' called at the top level against a data frame that still exists under the
#' same name. It fails when the object has been renamed or removed, or was
#' created inside a function that has since returned - `eval()` looks the
#' symbol up in the calling frame rather than retrieving a stored copy, so
#' this is convenience rather than a guarantee. In those cases pass
#' `data_surv` explicitly; the error message says so.
#'
#' Reconstructing it from the fitted object is not an option: `coxph()`
#' retains the `Surv()` response in `$y` by default, and the design matrix
#' in `$x` when `x = TRUE`, but never the subject identifier - that is not a
#' term in a Cox model, and it is exactly what is needed to align survival
#' records with longitudinal ones. `lme()` does retain its data, so
#' `data_long` is never needed.
#'
#' **What is rejected.** jmjax's longitudinal submodel assumes i.i.d.
#' Gaussian residuals and a single grouping factor, and its survival
#' submodel has no equivalent of strata, clusters or time-transform terms.
#' Passing an `lme` fit with a correlation or variance structure and having
#' it silently ignored would be a correctness problem, not a documented
#' limitation, so these raise errors:
#' \itemize{
#'   \item `lme`: any `correlation =` (e.g. `corAR1()`), any `weights =`
#'     variance structure, more than one grouping level, or `random =`
#'     given in list form
#'   \item `coxph`: `strata()`, `cluster()` or `tt()` terms, or a `Surv()`
#'     response that is not right-censored
#' }
#'
#' @param lme_object A fitted [nlme::lme()] model for the longitudinal
#'   outcome.
#' @param cox_object A fitted [survival::coxph()] model for the event
#'   process.
#' @param data_surv The subject-level survival data frame, containing the
#'   grouping variable used by `lme_object`. Optional: jmjax will try to
#'   recover it from `cox_object`'s call, and errors asking for it only when
#'   that fails - see Details for when it does.
#' @param time_var Name of the time variable in the longitudinal data.
#' @param method,functional_forms,init_theta,control Passed through to
#'   [jm_fit()] unchanged.
#'
#' @return An object of class `jmjax`, exactly as returned by [jm_fit()].
#'
#' @section Check the pre-fits before trusting the joint fit:
#'   The starting values for the MLE optimizers are taken from the models
#'   you supply, so a poorly-fitted `lme_object` propagates into the joint
#'   fit rather than being detected. jmjax rejects a starting point that
#'   produces a non-finite log-likelihood and falls back to its own
#'   defaults, but that only catches the extreme case. A start that is
#'   merely bad - rather than numerically invalid - can stop the optimizer
#'   in the wrong region after very few iterations, reporting
#'   `converged = TRUE` alongside estimates that are plainly wrong.
#'
#'   In testing, an `lme` fit whose fixed effects were around 70 where the
#'   data supported 2 produced a joint fit that terminated after 6
#'   iterations with an association parameter of -0.002 against a true 0.5.
#'   Nothing in the output flagged it.
#'
#'   So inspect `lme_object` and `cox_object` before fitting - convergence,
#'   the magnitude of the coefficients, the variance components - and treat
#'   a joint fit that converges suspiciously fast as a reason to look at
#'   the component models again. Setting
#'   `control$init_from_prefit = FALSE` makes jmjax ignore the pre-fits for
#'   starting values and use its own defaults, which is a useful
#'   cross-check if a result looks wrong.
#'
#' @seealso [jm_fit()] for the formula interface, which remains fully
#'   supported and is the better choice when you do not already have fitted
#'   component models.
#'
#' @examples
#' \dontrun{
#' lme_fit <- nlme::lme(log_serBilir ~ year, random = ~ year | id, data = pbc2)
#' cox_fit <- survival::coxph(Surv(years, status2) ~ drug, data = pbc2.id)
#' # data_surv recovered from cox_fit's call:
#' fit <- jm_fit_prefit(lme_fit, cox_fit, time_var = "year")
#' # or passed explicitly, which always works:
#' fit <- jm_fit_prefit(lme_fit, cox_fit, data_surv = pbc2.id, time_var = "year")
#' }
#'
#' @export
jm_fit_prefit <- function(lme_object,
                           cox_object,
                           data_surv = NULL,
                           time_var,
                           method = c("spline-PH-mcmc", "weibull-PH-mcmc",
                                      "weibull-PH-aGH", "spline-PH-aGH"),
                           functional_forms = NULL,
                           init_theta = NULL,
                           control = list()) {

  method <- match.arg(method)

  # ---- 1. class checks --------------------------------------------------
  if (!inherits(lme_object, "lme")) {
    stop("lme_object must be a fitted nlme::lme() model (got class '",
         paste(class(lme_object), collapse = "/"), "'). For the formula ",
         "interface, use jm_fit() instead.", call. = FALSE)
  }
  if (!inherits(cox_object, "coxph")) {
    stop("cox_object must be a fitted survival::coxph() model (got class '",
         paste(class(cox_object), collapse = "/"), "').", call. = FALSE)
  }
  # data_surv is optional: recover it from the coxph call when possible,
  # require it when not. This mirrors JMbayes2::jm(), which does:
  #     dataS <- try(eval(Surv_object$call$data, envir = parent.frame()), ...)
  #     if (inherits(dataS, "try-error")) stop("... provide 'data_Surv' ...")
  #
  # Recovery works for the common case of fitting coxph() at the top level
  # against a named data frame that still exists. It FAILS when the object
  # has been renamed or removed, or was created inside a function that has
  # since returned - `eval()` looks the symbol up in the caller's frame,
  # it does not retrieve a stored copy. coxph() itself keeps only the Surv
  # response ($y, by default) and optionally the design matrix ($x); it
  # never stores the subject identifier, which is not a model term, so a
  # partial reconstruction is not enough to align subjects.
  if (is.null(data_surv) || !is.data.frame(data_surv)) {
    # Try the caller's frame first (matching JMbayes2), then the global
    # environment. The second covers the common case of calling
    # jm_fit_prefit() from inside a function while the data lives at top
    # level - where parent.frame() alone would fail.
    .recovered <- try(eval(cox_object$call$data, envir = parent.frame()),
                      silent = TRUE)
    if (inherits(.recovered, "try-error") || !is.data.frame(.recovered)) {
      .recovered <- try(eval(cox_object$call$data, envir = globalenv()),
                        silent = TRUE)
    }
    if (inherits(.recovered, "try-error") || !is.data.frame(.recovered)) {
      stop("could not recover the data used to fit cox_object - pass it ",
           "explicitly via the data_surv argument. (coxph() stores only the ",
           "Surv response and, with x = TRUE, the design matrix; it never ",
           "stores the subject identifier, so the original data frame is ",
           "needed to align survival records with longitudinal ones.)",
           call. = FALSE)
    }
    data_surv <- as.data.frame(.recovered)
  }

  # ---- 2. reject what jmjax cannot represent ----------------------------
  # These are correctness guards, not style preferences: jmjax assumes
  # i.i.d. Gaussian residuals and a single grouping factor, so honouring
  # the call while ignoring the structure would give the user a different
  # model than the one they specified.
  if (!is.null(lme_object$modelStruct$corStruct)) {
    stop("lme_object has a correlation structure (",
         class(lme_object$modelStruct$corStruct)[1], "). jmjax's longitudinal ",
         "submodel assumes independent Gaussian residuals and cannot ",
         "represent it. Refit without `correlation =`, or use jm_fit() with ",
         "the formula interface if you accept that assumption.", call. = FALSE)
  }
  if (!is.null(lme_object$modelStruct$varStruct)) {
    stop("lme_object has a variance structure (",
         class(lme_object$modelStruct$varStruct)[1], "). jmjax assumes ",
         "homoscedastic residuals and cannot represent it. Refit without ",
         "`weights =`.", call. = FALSE)
  }
  n_levels <- length(lme_object$groups)
  if (n_levels != 1L) {
    stop("lme_object has ", n_levels, " grouping levels (nested or crossed ",
         "random effects). jmjax supports a single grouping factor - one ",
         "subject identifier.", call. = FALSE)
  }

  cox_terms <- stats::terms(cox_object)
  sp <- attr(cox_terms, "specials")
  for (bad in c("strata", "cluster", "tt")) {
    if (!is.null(sp[[bad]])) {
      stop("cox_object contains a ", bad, "() term. jmjax's survival submodel ",
           "has no equivalent; the baseline hazard is common to all subjects ",
           "and covariate effects are time-constant.", call. = FALSE)
    }
  }

  surv_resp <- tryCatch(stats::model.response(stats::model.frame(cox_object)),
                         error = function(e) NULL)
  s_type <- if (!is.null(surv_resp)) attr(surv_resp, "type") else NULL
  if (!is.null(s_type) && !identical(s_type, "right")) {
    stop("cox_object's Surv() response is of type '", s_type, "'. jmjax ",
         "supports right-censored data only.", call. = FALSE)
  }

  # ---- 3. extract the longitudinal structure ----------------------------
  long_formula <- stats::formula(lme_object)

  grp_form <- nlme::getGroupsFormula(lme_object)
  id_var <- all.vars(grp_form)
  if (length(id_var) != 1L) {
    stop("could not determine a single grouping variable from lme_object ",
         "(found: ", paste(id_var, collapse = ", "), ").", call. = FALSE)
  }

  # random_formula: parsed from the call's `random = ~ x | g` form. The
  # alternative, formula(lme_object$modelStruct$reStruct), returns a LIST
  # rather than a formula, so it is not usable directly.
  re_call <- lme_object$call$random
  if (is.null(re_call)) {
    stop("lme_object has no `random =` specification that jmjax can parse.",
         call. = FALSE)
  }
  re_eval <- tryCatch(eval(re_call, envir = parent.frame()),
                       error = function(e) re_call)
  if (is.list(re_eval) && !inherits(re_eval, "formula")) {
    stop("lme_object specifies `random =` in list form. jmjax needs the ",
         "formula form, e.g. random = ~ ", time_var, " | <id>.", call. = FALSE)
  }
  re_rhs <- if (inherits(re_eval, "formula")) re_eval[[length(re_eval)]] else re_call[[2]]
  if (is.call(re_rhs) && identical(as.character(re_rhs[[1]]), "|")) {
    re_rhs <- re_rhs[[2]]
  }
  random_formula <- stats::as.formula(paste("~", paste(deparse(re_rhs), collapse = "")))

  # q determines the random_effects argument. getVarCov() is the reliable
  # source - its dimension is the number of random effects actually fitted.
  q <- ncol(nlme::getVarCov(lme_object))
  random_effects <- if (q == 1L) "intercept" else "intercept_slope"
  if (q > 2L) {
    stop("lme_object has ", q, " random effects. jmjax supports a random ",
         "intercept (q=1) or a random intercept and slope (q=2).", call. = FALSE)
  }
  if (q == 1L) random_formula <- NULL   # jm_fit's default for intercept-only

  data_long <- lme_object$data
  if (is.null(data_long)) {
    stop("lme_object does not carry its data (`lme_object$data` is NULL). ",
         "Refit with the data available, or use jm_fit() and pass data_long ",
         "explicitly.", call. = FALSE)
  }
  data_long <- as.data.frame(data_long)

  if (!time_var %in% names(data_long)) {
    .nm <- names(data_long)
    .show <- if (length(.nm) > 20) c(.nm[1:20], "...") else .nm
    stop("time_var '", time_var, "' is not a column of the data attached to ",
         "lme_object (available: ", paste(.show, collapse = ", "), ").",
         call. = FALSE)
  }
  if (!id_var %in% names(data_surv)) {
    stop("the grouping variable '", id_var, "' from lme_object is not a ",
         "column of data_surv. The two data sets must share a subject ",
         "identifier so survival and longitudinal records can be aligned.",
         call. = FALSE)
  }

  # ---- 4. extract the survival structure -------------------------------
  surv_formula <- stats::formula(cox_object)

  # coef(cox_object) centres the prior on gamma, matching JMbayes2's
  # mean_gammas (verified: coef = -0.06697697 against mean_gammas = -0.067
  # on the same fit). jmjax otherwise centres that prior at zero.
  #
  # Membership is tested rather than nullity, so that passing
  # control$gamma_prior_mean = NULL explicitly is honoured as "do not use
  # the coxph coefficients" rather than being treated as "unset" and
  # silently overridden. list(x = NULL) creates the element, so is.null()
  # cannot distinguish the two cases.
  cox_coef <- stats::coef(cox_object)
  if (length(cox_coef) && !("gamma_prior_mean" %in% names(control))) {
    control$gamma_prior_mean <- unname(cox_coef)
  }
  # Starting values for the MLE optimizers. The longitudinal pieces are
  # picked up from lme_prefit inside jm_fit(); the survival coefficients
  # are only available here, since jm_fit() performs no coxph fit of its
  # own. Distinct from gamma_prior_mean above: that is a prior, this is a
  # starting point, and they happen to take the same value.
  if (length(cox_coef) && !("init_gamma" %in% names(control))) {
    control$init_gamma <- unname(cox_coef)
  }

  # ---- Starting values from the SUPPLIED lme, not a refit -------------
  # jm_fit()'s own MLE starting-value block fits a fresh lme() when these
  # are absent. Supplying them here means the user's OWN fit is used,
  # which matters for two reasons beyond saving a redundant fit:
  #
  #   - jm_fit() refits with its own lmeControl(opt = "optim",
  #     msMaxIter = 200, niterEM = 100). A user who tuned their lme to
  #     converge, or chose different settings deliberately, would
  #     otherwise have that silently discarded.
  #   - It is the point of this interface. Passing a fitted object and
  #     having it refitted internally undermines the reason to fit the
  #     components first at all.
  #
  # jm_fit() uses `%||%` throughout, so anything set here wins and the
  # internal refit is skipped. alpha is NOT set: the two-stage Cox
  # approximation needs X_time_surv, which exists only after the design
  # matrices are built, so it stays inside jm_fit().
  #
  # Measured effect of these starting values (5 seeds x 3 knot counts on
  # simulated data): optimizer iterations fall to about 0.85-0.91x with
  # identical estimates and no convergence cost. A one-off 3.5x
  # REGRESSION seen on the AIDS data at 7 knots did not reproduce - worst
  # ratio across seeds was 1.08 - so it was specific to that hazard shape.
  # Hand the fitted object itself to jm_fit(), which uses it for the
  # coefficients AND for the two-stage alpha (which needs ranef/fixef).
  # jm_fit() strips it from control before the backend call.
  control$.lme_object <- lme_object

  .Dp <- tryCatch(nlme::getVarCov(lme_object), error = function(e) NULL)
  control$init_beta    <- control$init_beta    %||% unname(nlme::fixef(lme_object))
  control$init_sigma_e <- control$init_sigma_e %||% stats::sigma(lme_object)
  if (!is.null(.Dp)) {
    control$init_sigma_b <- control$init_sigma_b %||% sqrt(diag(.Dp))
    if (nrow(.Dp) >= 2) {
      control$init_rho <- control$init_rho %||%
        (.Dp[1, 2] / sqrt(.Dp[1, 1] * .Dp[2, 2]))
    }
  }

  # ---- 5. delegate ------------------------------------------------------
  jm_fit(long_formula = long_formula,
         surv_formula = surv_formula,
         data_long = data_long,
         data_surv = data_surv,
         id_var = id_var,
         time_var = time_var,
         method = method,
         random_effects = random_effects,
         random_formula = random_formula,
         functional_forms = functional_forms,
         init_theta = init_theta,
         control = control)
}
