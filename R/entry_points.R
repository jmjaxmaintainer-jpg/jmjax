# jm_mle() and jm_bayes(): the two user-facing ways to fit a joint model.
#
# Both are thin front ends to jm_fit(), which holds all the fitting code;
# they choose the method from `baseline`, lift the options users set most
# often out of `control`, and warn about control options that belong to
# the other path (they would otherwise be silently ignored). A fit from
# jm_bayes(baseline = "spline") is the same object, with the same
# numbers, as one from jm_fit(method = "spline-PH-mcmc").

# control options read only by the MCMC path, and only by the MLE path.
# Anything not listed (spline knots, standardize_covariates, verbose, ...)
# is shared.
.jmjax_mcmc_only_controls <- c(
  "num_warmup", "num_samples", "num_chains", "seed", "progress_bar",
  "target_accept_prob", "max_tree_depth", "find_heuristic_step_size",
  "init_strategy", "init_values", "mcmc_warm_start", "warm_start_jitter",
  "random_effects_method", "random_effects_corr", "rotate_absorbable",
  "dense_mass", "dense_mass_spline", "dense_mass_alpha", "dense_mass_beta",
  "dense_mass_generator", "dense_mass_generator_beta",
  "dense_mass_b0_generator", "seed_mass_matrix", "orthogonalize_b",
  "orthogonalize_b0", "orthogonalize_b0_rotate", "orthogonalize_rotate_all",
  "spline_prior", "spline_penalty_shape", "spline_penalty_rate",
  "rw2_implementation", "prior_set", "empirical_bayes_prior",
  "alpha_prior_sd", "beta_prior_mean", "beta_prior_sd", "gamma_prior_mean",
  "gamma_prior_sd", "sigma_b_prior_mean", "sigma_b_prior_shape",
  "sigma_e_prior_mean", "sigma_e_prior_shape", "lkj_concentration",
  "wishart_eb_scale", "scale_time")

.jmjax_mle_only_controls <- c(
  "opt_method", "parscale", "maxiter", "ftol", "em_warm_start",
  "n_gh_nodes", "n_gh_nodes_per_dim", "n_newton_steps", "n_newton_steps_q2",
  "cholesky_first_newton", "init_alpha", "init_beta", "init_gamma",
  "init_log_lambda0", "init_log_shape", "init_rho", "init_sigma_b",
  "init_sigma_e", "init_spline")

.jmjax_warn_foreign_controls <- function(control, foreign, fun, other) {
  bad <- intersect(names(control), foreign)
  if (length(bad)) {
    warning(fun, "(): control option(s) ", paste(bad, collapse = ", "),
            " apply only to ", other, "() and are ignored here.",
            call. = FALSE)
  }
}

# The recorded call, with the function NAME in first position. Under
# do.call(jm_mle, args) match.call() puts the function object itself there,
# and print(fit) would then print the function's source code.
.jmjax_named_call <- function(cl, name) {
  cl[[1L]] <- as.name(name)
  cl
}

# Put an argument into control, refusing two different values for one
# setting.
.jmjax_set_control <- function(control, key, value, arg) {
  if (is.null(value)) return(control)
  if (!is.null(control[[key]]) && !isTRUE(all.equal(as.numeric(control[[key]]), as.numeric(value)))) {
    stop("`", arg, "` and control$", key, " are both given, with different ",
         "values; give it once.", call. = FALSE)
  }
  control[[key]] <- value
  control
}

#' Fit a joint model by maximum likelihood
#'
#' Fits a shared-parameter joint model - a linear mixed-effects model for
#' the longitudinal marker and a proportional-hazards model for the event
#' time, linked through the marker's current value - by maximum likelihood.
#' The random effects are integrated out with adaptive Gauss-Hermite
#' quadrature, and the likelihood is maximized with BFGS, with gradients
#' and the Hessian from JAX's automatic differentiation.
#'
#' Maximum likelihood is fast and gives AIC/BIC through \code{logLik()},
#' but does not estimate individual subjects' random effects, so
#' \code{ranef()} and \code{predict()} need a Bayesian fit from
#' \code{\link{jm_bayes}()}.
#'
#' @inheritParams jm_fit
#' @param baseline The baseline hazard: \code{"weibull"} (default) or
#'   \code{"spline"} (a B-spline on the log-hazard scale; see
#'   \code{n_interior_knots} under Control options).
#' @param control A list of further options. Those for this path:
#'   \describe{
#'     \item{\code{opt_method}}{Optimizer, default \code{"BFGS"}.}
#'     \item{\code{maxiter}, \code{ftol}, \code{parscale}}{Optimizer
#'       iteration limit, tolerance and parameter scaling (default
#'       \code{parscale = 0.01}, as in \code{JM}).}
#'     \item{\code{em_warm_start}}{Run a short EM phase before the
#'       optimizer (default \code{FALSE}).}
#'     \item{\code{n_gh_nodes}, \code{n_gh_nodes_per_dim},
#'       \code{n_newton_steps}, \code{n_newton_steps_q2}}{Adaptive
#'       Gauss-Hermite settings for one and two random effects.}
#'     \item{\code{n_interior_knots}, \code{knot_placement},
#'       \code{spline_order}}{Spline baseline: number of interior knots
#'       (default 5), their placement (\code{"quantile"}, as in \code{JM},
#'       or \code{"equal"}, as in \code{JMbayes2}) and the spline order
#'       (default 4, cubic).}
#'     \item{\code{standardize_covariates}}{Centre and scale baseline
#'       covariates internally (default \code{TRUE}); estimates are always
#'       reported on the original scale.}
#'     \item{\code{verbose}}{Informational messages; see \code{\link{jm_fit}}.}
#'   }
#'   \code{\link{jm_fit}} documents every option in full. Options that only
#'   the Bayesian path uses (\code{num_chains}, priors, ...) are ignored
#'   with a warning.
#' @return An object of class \code{"jmjax"}; see \code{\link{jm_fit}}.
#' @seealso \code{\link{jm_bayes}} for Bayesian estimation;
#'   \code{\link{jmjax-methods}} for \code{coef()}, \code{vcov()},
#'   \code{confint()} and \code{logLik()}.
#' @examplesIf jmjax_available() && requireNamespace("JM", quietly = TRUE)
#' data("pbc2", "pbc2.id", package = "JM")
#' pbc2$log_bili <- log(pbc2$serBilir)
#' fit <- jm_mle(log_bili ~ year, survival::Surv(years, status2) ~ drug,
#'               data_long = pbc2, data_surv = pbc2.id,
#'               id_var = "id", time_var = "year")
#' fit
#' confint(fit, "alpha")
#' AIC(fit)
#' @export
jm_mle <- function(long_formula, surv_formula, data_long, data_surv,
                   id_var, time_var, baseline = c("weibull", "spline"),
                   random_effects = "intercept", random_formula = NULL,
                   functional_forms = NULL, init_theta = NULL,
                   control = list()) {
  baseline <- match.arg(baseline)
  .jmjax_warn_foreign_controls(control, .jmjax_mcmc_only_controls,
                               "jm_mle", "jm_bayes")
  control <- control[setdiff(names(control), .jmjax_mcmc_only_controls)]
  fit <- jm_fit(long_formula = long_formula, surv_formula = surv_formula,
                data_long = data_long, data_surv = data_surv,
                id_var = id_var, time_var = time_var,
                method = if (baseline == "weibull") "weibull-PH-aGH" else "spline-PH-aGH",
                random_effects = random_effects, random_formula = random_formula,
                functional_forms = functional_forms, init_theta = init_theta,
                control = control)
  fit$call <- .jmjax_named_call(match.call(), "jm_mle")
  fit
}

#' Fit a joint model by Bayesian MCMC
#'
#' Fits a shared-parameter joint model - a linear mixed-effects model for
#' the longitudinal marker and a proportional-hazards model for the event
#' time, linked through the marker's current value - by Bayesian
#' estimation, sampling all parameters and each subject's random effects
#' with the No-U-Turn sampler (NUTS) in NumPyro.
#'
#' By default the random effects are rotated so that NUTS does not have to
#' traverse the ridge between them and the subject-constant fixed effects
#' (\code{control$rotate_absorbable}); the model and its estimates are
#' unchanged, only mixing improves. The chains start from a preliminary
#' \code{nlme::lme()} fit, as in \code{JMbayes2}. Time and covariates are
#' rescaled internally where that helps the sampler, and every result is
#' reported on the original scale.
#'
#' @inheritParams jm_fit
#' @param baseline The baseline hazard: \code{"spline"} (default; a
#'   B-spline on the log-hazard scale) or \code{"weibull"}.
#' @param chains,warmup,samples Number of chains, warm-up iterations per
#'   chain and retained draws per chain. Default 2 chains of 500 + 1000.
#'   Chains run in parallel when \code{\link{jmjax_setup}()} made several
#'   CPU devices available.
#' @param seed Random seed for the sampler, or \code{NULL}.
#' @param control A list of further options. Those for this path:
#'   \describe{
#'     \item{\code{spline_prior}}{\code{"independent"} (default) or
#'       \code{"penalized"} (a second-order random-walk prior on the spline
#'       coefficients, as in \code{JMbayes2}).}
#'     \item{\code{n_interior_knots}, \code{knot_placement},
#'       \code{spline_order}}{Spline baseline: number of interior knots
#'       (default 5), their placement (\code{"quantile"} or \code{"equal"})
#'       and the spline order (default 4, cubic).}
#'     \item{\code{prior_set}, \code{empirical_bayes_prior} and the
#'       \code{*_prior_*} options}{Priors; see \code{\link{jm_fit}}.}
#'     \item{\code{rotate_absorbable}}{Rotate the random effects (default
#'       \code{TRUE} where it applies).}
#'     \item{\code{target_accept_prob}, \code{max_tree_depth}}{NUTS
#'       settings.}
#'     \item{\code{progress_bar}}{Show NumPyro's progress bar (default
#'       \code{FALSE}).}
#'     \item{\code{scale_time}, \code{standardize_covariates}}{Internal
#'       rescaling of time and covariates (default on); estimates are
#'       always reported on the original scale.}
#'     \item{\code{verbose}}{Informational messages; see \code{\link{jm_fit}}.}
#'   }
#'   \code{\link{jm_fit}} documents every option in full. Options that only
#'   the maximum-likelihood path uses (\code{opt_method}, quadrature
#'   settings, ...) are ignored with a warning.
#' @return An object of class \code{"jmjax"}; see \code{\link{jm_fit}}.
#'   \code{print()} and \code{summary()} report R-hat and effective sample
#'   sizes.
#' @seealso \code{\link{jm_mle}} for maximum likelihood;
#'   \code{\link{predict.jmjax}}, \code{\link{ranef.jmjax}} and
#'   \code{\link{jmjax-methods}}.
#' @examplesIf jmjax_available() && requireNamespace("JM", quietly = TRUE)
#' \donttest{
#' data("pbc2", "pbc2.id", package = "JM")
#' pbc2$log_bili <- log(pbc2$serBilir)
#' fit <- jm_bayes(log_bili ~ year, survival::Surv(years, status2) ~ drug,
#'                 data_long = pbc2, data_surv = pbc2.id,
#'                 id_var = "id", time_var = "year",
#'                 random_effects = "intercept_slope", random_formula = ~ year,
#'                 warmup = 500, samples = 500)
#' summary(fit)
#' head(ranef(fit))
#'
#' # Survival probabilities for a subject still alive at the end of their
#' # follow-up, from that point on
#' id_c <- pbc2.id$id[pbc2.id$status2 == 0 & pbc2.id$years < 8][1]
#' predict(fit, newdata = pbc2[pbc2$id == id_c, ], process = "event")
#' }
#' @export
jm_bayes <- function(long_formula, surv_formula, data_long, data_surv,
                     id_var, time_var, baseline = c("spline", "weibull"),
                     random_effects = "intercept", random_formula = NULL,
                     functional_forms = NULL, chains = 2L, warmup = 500L,
                     samples = 1000L, seed = NULL, control = list()) {
  baseline <- match.arg(baseline)
  .jmjax_warn_foreign_controls(control, .jmjax_mle_only_controls,
                               "jm_bayes", "jm_mle")
  control <- control[setdiff(names(control), .jmjax_mle_only_controls)]
  # An argument left at its default gives way to the same setting in
  # control (older code passes control$num_chains etc.); one given
  # explicitly must agree with it.
  for (a in list(list("chains", "num_chains", chains, missing(chains)),
                 list("warmup", "num_warmup", warmup, missing(warmup)),
                 list("samples", "num_samples", samples, missing(samples)),
                 list("seed", "seed", seed, missing(seed)))) {
    if (is.null(a[[3]])) next
    if (a[[4]] && !is.null(control[[a[[2]]]])) next
    control <- .jmjax_set_control(control, a[[2]], as.integer(a[[3]]), a[[1]])
  }
  fit <- jm_fit(long_formula = long_formula, surv_formula = surv_formula,
                data_long = data_long, data_surv = data_surv,
                id_var = id_var, time_var = time_var,
                method = if (baseline == "spline") "spline-PH-mcmc" else "weibull-PH-mcmc",
                random_effects = random_effects, random_formula = random_formula,
                functional_forms = functional_forms, control = control)
  fit$call <- .jmjax_named_call(match.call(), "jm_bayes")
  fit
}
