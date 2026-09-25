#' jmjax: Joint Models for Longitudinal and Survival Data via JAX
#'
#' Fits shared-parameter joint models (a linear mixed-effects longitudinal
#' submodel plus a relative-risk survival submodel with Weibull or
#' spline-approximated baseline hazard) using a JAX/NumPyro backend, with R
#' providing the formula interface and Python providing the numerical
#' fitting via \pkg{reticulate}.
#'
#' @section Getting started: Fit by maximum likelihood with
#'   \code{\link{jm_mle}()} or by Bayesian MCMC with \code{\link{jm_bayes}()};
#'   \code{\link{jm_fit}()} is the general interface behind both. See
#'   \code{vignette("jmjax-introduction")} for a worked walkthrough covering
#'   the fitting functions, diagnostics, predictions, and how to choose
#'   between them.
#'
#' @section Validation: This package's estimates have been extensively
#'   cross-validated against R's \pkg{JM} and \pkg{JMbayes2} packages
#'   during development - see \code{vignette("jmjax-validation")} for the
#'   full methodology, bugs found and fixed along the way, and benchmark
#'   results.
#'
#' @section Citation: If you use jmjax, please cite the software release
#'   (\doi{10.5281/zenodo.22950083}); \code{citation("jmjax")} gives the
#'   reference.
#'
#' @section One-time setup: Run \code{\link{jmjax_setup}()} once per
#'   machine to create the managed Python environment, or point
#'   \pkg{reticulate} at an existing environment with \code{jax}/
#'   \code{numpyro} installed (see \code{\link{jmjax_setup}} for details on
#'   both paths, including Windows-specific notes).
#'
#' @keywords internal
"_PACKAGE"
