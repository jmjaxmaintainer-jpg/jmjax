# Internal: the nlme::lme() pre-fit behind the empirical-Bayes priors and
# the MCMC warm start.
#
# With three or more random effects an unstructured (pdSymm) covariance
# often fails to converge, or converges to a singular matrix. For q >= 3 a
# diagonal (pdDiag) structure is tried next: it still gives the fixed
# effects, the residual SD and a scale for each random effect, which is all
# the priors use (the correlations get the LKJ prior either way). If that
# fails too, the caller falls back to priors that do not depend on the
# data. At q <= 2 the behaviour is exactly the previous one: one attempt,
# NULL on error.
#
# Returns list(fit = <lme or NULL>, structure = "unstructured" |
# "diagonal" | "none", message = <last error, or NULL>).
.jmjax_lme_prefit <- function(long_formula, random_formula, id_var, data_long,
                              lme_control) {
  rf_txt <- paste(deparse(random_formula), collapse = " ")
  q <- tryCatch(
    ncol(stats::model.matrix(stats::delete.response(stats::terms(random_formula)),
                             data = data_long)),
    error = function(e) NA_integer_)

  fit_with <- function(random) {
    tryCatch(nlme::lme(long_formula, random = random, data = data_long,
                       control = lme_control),
             error = function(e) e)
  }
  usable <- function(f) {
    if (inherits(f, "error")) return(FALSE)
    sdb <- tryCatch(sqrt(diag(as.matrix(nlme::getVarCov(f)))),
                    error = function(e) NA_real_)
    all(is.finite(sdb)) && all(sdb > 0) && is.finite(stats::sigma(f))
  }

  f1 <- fit_with(stats::as.formula(paste(rf_txt, "|", id_var)))
  if (!is.na(q) && q <= 2L) {
    # Previous behaviour, unchanged.
    if (inherits(f1, "error")) {
      return(list(fit = NULL, structure = "none", message = conditionMessage(f1)))
    }
    return(list(fit = f1, structure = "unstructured", message = NULL))
  }
  if (usable(f1)) return(list(fit = f1, structure = "unstructured", message = NULL))

  f2 <- fit_with(stats::setNames(list(nlme::pdDiag(random_formula)), id_var))
  if (usable(f2)) {
    return(list(fit = f2, structure = "diagonal",
                message = if (inherits(f1, "error")) conditionMessage(f1) else
                  "unstructured covariance gave a non-finite or zero variance"))
  }
  list(fit = NULL, structure = "none",
       message = if (inherits(f2, "error")) conditionMessage(f2) else
         "diagonal covariance gave a non-finite or zero variance")
}
