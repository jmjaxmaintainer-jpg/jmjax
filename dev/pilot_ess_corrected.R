# ==============================================================================
# Pilot: does the efficiency gain SURVIVE the calibration correction?
#
# THE QUESTION. control$orthogonalize_b/_b0 buy a large regression-parameter
# ESS gain (Sections 6-7 of vignette("jmjax-reparameterization")), but
# dev/study_calibration.R showed the credible intervals of the swept
# coefficients are miscalibrated, so `beta` is not the estimand a user may
# report. The identified estimand is `beta_corrected` (Section 4.9). Its
# effective sample size is therefore the figure that actually matters, and
# it is NOT currently reported: the correction is applied to the pooled
# draws, downstream of the per-chain summary that produces R-hat and ESS.
#
# If beta_corrected's ESS sits near beta's, the option delivers what it
# claims. If it collapses back to a default fit's, the option buys fast
# sampling of a quantity nobody can report and slow effective sampling of
# the one they can - which would substantially weaken the paper's headline.
# This script measures which.
#
# WHY NO REINSTALL IS NEEDED. beta_corrected's draws are already present in
# any orthogonalized fit, and the backend already exposes
# recompute_site_diagnostics() - the same routine jm_fit() uses to rebuild
# ESS after the standardize_covariates back-transform. So the measurement
# is pure post-processing of a fitted object, using the PACKAGE'S OWN
# estimator rather than a second implementation that would disagree with it
# in the third digit. Nothing here requires changing or reinstalling the
# backend, so it is safe to run alongside a study already in flight.
#
# WHAT IS COMPARED, per parameter, all three via the same estimator so the
# numbers are directly comparable:
#
#   A / beta             the reference: no option, the identified estimand
#   E / beta             the constrained estimand - the headline ESS claim
#   E / beta_corrected   the identified estimand under the option - the question
#
# ESS/sec is reported alongside raw ESS because that is the metric the
# vignette's efficiency claims are stated in, and the correction costs no
# sampling time (it is arithmetic on stored draws), so any change in
# ESS/sec is a change in ESS alone.
#
# Arm E rather than D: it sweeps both the intercept and the slope column,
# so one fit answers the question for both beta_0 and beta_slope.
#
#   Rscript dev/pilot_ess_corrected.R
#   ESS_SEEDS=4 Rscript dev/pilot_ess_corrected.R      # more seeds
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(survival)
})
`%||%` <- function(a, b) if (is.null(a)) b else a

.self <- sub("^--file=", "",
             grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.gen <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
if (!file.exists(.gen)) stop("cannot find sim_joint.R next to this script", call. = FALSE)
source(.gen)

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d
}

SEEDS   <- .envi("ESS_SEEDS", 2L)
N_SUB   <- .envi("ESS_N", 300L)
K_EXTRA <- .envi("ESS_K_EXTRA", 2L)
WARMUP  <- .envi("ESS_WARMUP", 500L)
SAMPLES <- .envi("ESS_SAMPLES", 1000L)
CHAINS  <- .envi("ESS_CHAINS", 2L)

# Matches dev/study_calibration.R's `linear` design exactly, so these ESS
# figures sit alongside that study's calibration figures for the same fits.
COVS  <- c("age", "sex", "trt")[seq_len(K_EXTRA)]
LFORM <- stats::as.formula(paste("y ~ time +", paste(COVS, collapse = " + ")))
SLOPE_IDX <- 2L

as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) {
    return(t(vapply(v, function(z) as.numeric(unlist(z)),
                    numeric(length(unlist(v[[1]]))))))
  }
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}

# The package's own estimator, via the same entry point jm_fit() uses.
# Draws arrive with chains CONCATENATED; recompute_site_diagnostics reshapes
# to [chains, n_per_chain, p] so R-hat is genuinely between-chain.
site_diag <- function(M, nc) {
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  list(ess = as.numeric(unlist(rd$n_eff)),
       rhat = as.numeric(unlist(rd$r_hat)))
}

fit_one <- function(arm, seed) {
  sim <- sim_joint(n = N_SUB, seed = 7000L + seed, k_extra = K_EXTRA)

  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (arm == "E") ctl$orthogonalize_b <- TRUE

  f <- jm_fit(
    long_formula = LFORM,
    surv_formula = survival::Surv(time, event) ~ trt,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time, control = ctl)

  # Sampling time only - the correction is arithmetic on stored draws and
  # adds no sampling cost, so ESS/sec differences are ESS differences.
  secs <- as.numeric(f$convergence$sampling_time_sec %||% NA_real_)

  ps <- f$posterior_samples
  B  <- as_draws(ps[["beta"]])
  if (is.null(B)) stop("no beta draws returned", call. = FALSE)
  dB <- site_diag(B, CHAINS)

  rows <- list()
  push <- function(quantity, idx, label, d) {
    rows[[length(rows) + 1L]] <<- data.frame(
      seed = seed, arm = arm, quantity = quantity, param = label,
      ess = d$ess[idx], rhat = d$rhat[idx],
      ess_per_sec = d$ess[idx] / secs, sec = secs,
      stringsAsFactors = FALSE)
  }

  Xd <- stats::model.matrix(LFORM, sim$data_long)
  i0 <- match("(Intercept)", colnames(Xd))
  if (is.na(i0)) stop("no intercept column in the design", call. = FALSE)
  # Assert the slope column is time-varying WITHIN subject, as
  # dev/study_calibration.R does: a covariate is subject-constant, so this
  # catches a formula edit that silently moved what beta_slope tracks.
  wv <- max(tapply(Xd[, SLOPE_IDX], sim$data_long$id,
                   function(v) diff(range(v))))
  if (!is.finite(wv) || wv <= 1e-8) {
    stop("column ", SLOPE_IDX, " ('", colnames(Xd)[SLOPE_IDX],
         "') is subject-constant; it cannot be the fixed slope on time",
         call. = FALSE)
  }

  push("beta", i0, "beta_0", dB)
  push("beta", SLOPE_IDX, "beta_slope", dB)

  BC <- as_draws(ps[["beta_corrected"]])
  if (!is.null(BC)) {
    if (ncol(BC) != ncol(B)) {
      stop(sprintf("beta_corrected has %d columns, beta has %d", ncol(BC), ncol(B)),
           call. = FALSE)
    }
    dC <- site_diag(BC, CHAINS)
    push("beta_corrected", i0, "beta_0", dC)
    push("beta_corrected", SLOPE_IDX, "beta_slope", dC)
  } else if (arm == "E") {
    stop("arm E returned no beta_corrected - the installed backend predates ",
         "the correction track; reinstall before running this pilot",
         call. = FALSE)
  }

  do.call(rbind, rows)
}

cat(sprintf("\nESS pilot: n = %d, %d warmup + %d samples x %d chains, %d seed(s)\n",
            N_SUB, WARMUP, SAMPLES, CHAINS, SEEDS))
cat(sprintf("design: %s\n\n", paste(deparse(LFORM), collapse = " ")))

all_rows <- list()
for (s in seq_len(SEEDS)) {
  for (a in c("A", "E")) {
    r <- try(fit_one(a, s), silent = TRUE)
    if (inherits(r, "try-error")) {
      cat(sprintf("  [seed %d arm %s] FAILED: %s", s, a, as.character(r)))
      next
    }
    all_rows[[length(all_rows) + 1L]] <- r
    cat(sprintf("  [seed %d arm %s] %.1fs  max R-hat %.3f\n",
                s, a, r$sec[1], max(r$rhat, na.rm = TRUE)))
  }
}
R <- do.call(rbind, all_rows)
if (is.null(R) || !nrow(R)) stop("no fits succeeded", call. = FALSE)

cat("\n")
cat("=========================================================================\n")
cat(" ESS of the CONSTRAINED estimand (E/beta) vs the IDENTIFIED one\n")
cat(" (E/beta_corrected), against a default fit (A/beta) as reference.\n")
cat(" beta_corrected is what a user may report; its ESS is the figure that\n")
cat(" decides whether the efficiency gain is usable.\n")
cat("=========================================================================\n\n")
cat(sprintf("%-11s %-16s %4s %9s %11s %7s\n",
            "param", "quantity", "n", "ESS", "ESS/sec", "R-hat"))

key <- function(p, a, q) R[R$param == p & R$arm == a & R$quantity == q, , drop = FALSE]
means <- list()
for (p in c("beta_0", "beta_slope")) {
  for (aq in list(c("A", "beta"), c("E", "beta"), c("E", "beta_corrected"))) {
    x <- key(p, aq[1], aq[2])
    if (!nrow(x)) next
    lbl <- paste0(aq[1], " / ", aq[2])
    cat(sprintf("%-11s %-16s %4d %9.1f %11.2f %7.3f\n",
                p, lbl, nrow(x), mean(x$ess), mean(x$ess_per_sec),
                max(x$rhat, na.rm = TRUE)))
    means[[paste(p, aq[1], aq[2])]] <- mean(x$ess_per_sec)
  }
  cat("\n")
}

cat("-------------------------------------------------------------------------\n")
cat(" The decisive ratios (ESS/sec relative to a default fit):\n\n")
for (p in c("beta_0", "beta_slope")) {
  ref <- means[[paste(p, "A", "beta")]]
  eb  <- means[[paste(p, "E", "beta")]]
  ec  <- means[[paste(p, "E", "beta_corrected")]]
  if (is.null(ref) || !is.finite(ref) || ref <= 0) next
  cat(sprintf("  %-11s  E/beta %6.2fx    E/beta_corrected %6.2fx\n",
              p, eb / ref, ec / ref))
}
cat("\n Read the SECOND column. If it is close to the first, the efficiency\n")
cat(" gain survives the correction and the option delivers a reportable\n")
cat(" speedup. If it is close to 1.00x, the gain is confined to an estimand\n")
cat(" that cannot be reported, and the headline claim needs restating.\n")
cat(sprintf("\n %d seed(s) only - this is a pilot. Treat a ratio near the\n", SEEDS))
cat(" boundary as undecided rather than as a result.\n")
