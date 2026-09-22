# ==============================================================================
# Pilot: does isolating the swept generator direction as its OWN NUTS site
# (control$orthogonalize_b0_rotate, + control$dense_mass_b0_generator) close
# the ESS gap dev/pilot_ess_corrected.R found for beta_corrected?
#
# BACKGROUND. Section 9.1.1 of vignette("jmjax-reparameterization") measured
# that orthogonalize_b0/_b relocate a location degeneracy into the b-block
# rather than removing it: beta_corrected's own ESS/sec gain over a default
# fit collapses to ~1.25-1.31x, far short of the swept (unreportable) beta's
# ~9-15x. The diagnosis: the relocated direction lives inside b_std[:, 0], a
# several-hundred-dimensional per-subject block, and inherits whatever
# mixing rate NUTS's single global mass matrix gives that block - not the
# near-prior-rate mixing the vignette's own earlier hypothesis assumed.
#
# THE IDEA THIS SCRIPT TESTS. b_raw[:, 0] = sigma_b[0] * b_std[:, 0] always
# (a valid Cholesky factor of a correlation matrix has first row
# [1, 0, ..., 0]), so a FIXED, exact rotation of b_std[:, 0] - via the
# orthonormal basis Q that orthogonalize_b0 already computes, plus its
# orthogonal complement Qperp - splits it into:
#   b0_gen_u  (k-dim, k = number of absorbable directions, typically 1-4):
#             EXACTLY Q.T @ b_std[:, 0], the generator coordinate the
#             degeneracy lives in.
#   b0_gen_v  ((N_sub - k)-dim): the residual, orthogonal to b0_gen_u.
# This is an EXACT reparameterization (a rotation of a spherical Gaussian -
# see _orthogonal_complement()'s docstring in mcmc_model.py) - it changes
# nothing about the model, only which coordinates NUTS can adapt a mass
# matrix to. Giving the small b0_gen_u block its own dense mass-matrix
# entry (dense_mass_b0_generator) is then a one-line use of NumPyro's
# existing block-diagonal mass-matrix mechanism - the same one
# dense_mass_beta and dense_mass_spline already use elsewhere in this file.
#
# WHY THIS MIGHT NOT WORK. dense_mass_beta targeted an even more strongly
# correlated ridge (-0.977) and produced a NULL result - a large
# correlation is necessary but evidently not sufficient. b0_gen_u is a
# different kind of block (a newly-isolated, previously-nonexistent site,
# not a slice of an already-well-conditioned one), so this is a genuinely
# open question, not a confirmation exercise. Read a null result here the
# same way: informative, not a failure.
#
# WHAT THIS HAS NOT BEEN TESTED WITH. The implementing session had no
# JAX/NumPyro available to trace the model directly (network-restricted
# sandbox). The rotation identity itself was verified independently in
# pure numpy to ~1e-15, and the code was reviewed line by line against the
# existing orthogonalize_b0 machinery it reuses, but THIS SCRIPT'S FIRST
# RUN IS ALSO THE FIRST TIME THE NEW CODE PATH IS ACTUALLY TRACED BY JAX.
# Treat any shape or tracing error as a real bug report, not a
# configuration mistake - please share the exact traceback if one comes up.
#
# KNOWN LIMITATION - control$init_values (warm starts) AND
# orthogonalize_b0_rotate DO NOT COMBINE. fit_nuts()'s warm-start path
# (init_to_value_jittered) only substitutes actual numpyro.sample sites;
# with the rotation active, "b_std" becomes a numpyro.deterministic (it is
# rebuilt from the new "b0_gen_u"/"b0_gen_v"/"b_std_rest" sample sites), so
# any pre-fit-seeded initial value supplied for "b_std" is silently
# skipped for the rotated columns - a soft degradation (NUTS still gets a
# valid, if less-informed, starting point from the prior), not a crash or
# an error. This pilot does not exercise control$init_values at all, so it
# will not surface this; it only matters if you later combine
# orthogonalize_b0_rotate with a warm-started fit elsewhere.
#
# WHAT IS COMPARED, per seed, all via the same estimator
# (recompute_site_diagnostics, as in pilot_ess_corrected.R) so the numbers
# are directly comparable:
#
#   A        default fit                                    - reference
#   D        orthogonalize_b0 (plain, as shipped)            - the ESS-
#                                                               collapse
#                                                               baseline
#   D_rot    orthogonalize_b0 + orthogonalize_b0_rotate      - rotation
#                                                               alone
#   D_dense  orthogonalize_b0 + orthogonalize_b0_rotate +
#            dense_mass_b0_generator                          - rotation +
#                                                               its own
#                                                               mass block
#
# For each: beta_corrected's ESS/sec for beta_0 (the only column
# orthogonalize_b0 sweeps), AND - for D_rot/D_dense - b0_gen_u's own
# ESS/sec directly, read straight off fit$posterior_samples$b0_gen_u. If
# D_dense's beta_corrected ESS/sec approaches D's swept-beta number (not
# just improves a little over D's beta_corrected), the option is workable
# as a fix for the Section 9.1.1 finding, not just a partial mitigation.
#
# A quick correctness sanity check is also printed: beta_corrected's
# posterior mean should agree closely between D and D_rot at a shared seed
# (same target distribution, different sampling coordinates - Proposition 6
# says the rotation is exact, so this should track the same tight
# agreement dev/study_calibration.R already measured between arms).
#
#   Rscript dev/pilot_b0_rotate.R
#   ROT_SEEDS=4 Rscript dev/pilot_b0_rotate.R      # more seeds
#
# THIS FILE: identical to dev/pilot_b0_rotate.R except method =
# "weibull-PH-mcmc" instead of "spline-PH-mcmc" - a closed-form Weibull
# baseline hazard instead of the penalized-spline one, for a faster smoke
# test. The rotation targets b_std/L_corr in the LONGITUDINAL submodel
# (build_model()'s random-effects block), which is shared code regardless
# of baseline_hazard - orthogonalize_b0/_b0_rotate do not read
# baseline_hazard at all (grep confirms), so this is expected to behave
# identically to the spline variant, just without the spline coefficients'
# own dimensions and dense_mass_spline overhead. n_interior_knots/
# spline_prior/rw2_implementation/dense_mass_spline are dropped from ctl
# below since R only reads them when method is one of the spline-PH-*
# variants (R/jm_fit.R) - leaving them in would not error, just be unused.
#
#   Rscript dev/pilot_b0_rotate_weibull.R
#   ROT_SEEDS=4 Rscript dev/pilot_b0_rotate_weibull.R
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

SEEDS   <- .envi("ROT_SEEDS", 2L)
N_SUB   <- .envi("ROT_N", 300L)
K_EXTRA <- .envi("ROT_K_EXTRA", 2L)
WARMUP  <- .envi("ROT_WARMUP", 500L)
SAMPLES <- .envi("ROT_SAMPLES", 1000L)
CHAINS  <- .envi("ROT_CHAINS", 2L)

# Matches dev/study_calibration.R's `linear` design and pilot_ess_corrected.R.
COVS  <- c("age", "sex", "trt")[seq_len(K_EXTRA)]
LFORM <- stats::as.formula(paste("y ~ time +", paste(COVS, collapse = " + ")))

as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) {
    return(t(vapply(v, function(z) as.numeric(unlist(z)),
                    numeric(length(unlist(v[[1]]))))))
  }
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}

site_diag <- function(M, nc) {
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  list(ess = as.numeric(unlist(rd$n_eff)),
       rhat = as.numeric(unlist(rd$r_hat)))
}

fit_one <- function(arm, seed) {
  sim <- sim_joint(n = N_SUB, seed = 8000L + seed, k_extra = K_EXTRA)

  ctl <- list(num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (arm %in% c("D", "D_rot", "D_dense")) ctl$orthogonalize_b0 <- TRUE
  if (arm %in% c("D_rot", "D_dense"))      ctl$orthogonalize_b0_rotate <- TRUE
  if (arm == "D_dense")                    ctl$dense_mass_b0_generator <- TRUE

  f <- jm_fit(
    long_formula = LFORM,
    surv_formula = survival::Surv(time, event) ~ trt,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "weibull-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time, control = ctl)

  secs <- as.numeric(f$convergence$sampling_time_sec %||% NA_real_)
  ps <- f$posterior_samples
  max_rhat_all <- tryCatch(max(unlist(f$diagnostics$rhat), na.rm = TRUE),
                            error = function(e) NA_real_)

  Xd <- stats::model.matrix(LFORM, sim$data_long)
  i0 <- match("(Intercept)", colnames(Xd))
  if (is.na(i0)) stop("no intercept column in the design", call. = FALSE)

  rows <- list()
  push <- function(quantity, label, mat, idx, d) {
    rows[[length(rows) + 1L]] <<- data.frame(
      seed = seed, arm = arm, quantity = quantity, param = label,
      est = mean(mat[, idx]), ess = d$ess[idx], rhat = d$rhat[idx],
      ess_per_sec = d$ess[idx] / secs, sec = secs, max_rhat_all = max_rhat_all,
      stringsAsFactors = FALSE)
  }

  B  <- as_draws(ps[["beta"]])
  if (!is.null(B)) push("beta", "beta_0", B, i0, site_diag(B, CHAINS))

  BC <- as_draws(ps[["beta_corrected"]])
  if (!is.null(BC)) {
    push("beta_corrected", "beta_0", BC, i0, site_diag(BC, CHAINS))
  } else if (arm != "A") {
    stop("arm ", arm, " returned no beta_corrected - unexpected", call. = FALSE)
  }

  U <- as_draws(ps[["b0_gen_u"]])
  if (!is.null(U)) {
    dU <- site_diag(U, CHAINS)
    for (j in seq_len(ncol(U))) push("b0_gen_u", paste0("u", j), U, j, dU)
  } else if (arm %in% c("D_rot", "D_dense")) {
    stop("arm ", arm, " requested orthogonalize_b0_rotate but returned no ",
         "b0_gen_u - the installed backend predates this change; reinstall ",
         "before running this pilot", call. = FALSE)
  }

  do.call(rbind, rows)
}

cat(sprintf("\nb0-rotate pilot (WEIBULL baseline hazard): n = %d, %d warmup + %d samples x %d chains, %d seed(s)\n",
            N_SUB, WARMUP, SAMPLES, CHAINS, SEEDS))
cat(sprintf("design: %s\n\n", paste(deparse(LFORM), collapse = " ")))

all_rows <- list()
for (s in seq_len(SEEDS)) {
  for (a in c("A", "D", "D_rot", "D_dense")) {
    r <- try(fit_one(a, s), silent = TRUE)
    if (inherits(r, "try-error")) {
      cat(sprintf("  [seed %d arm %-7s] FAILED: %s", s, a, as.character(r)))
      next
    }
    all_rows[[length(all_rows) + 1L]] <- r
    cat(sprintf("  [seed %d arm %-7s] %6.1fs  max R-hat(all) %.3f\n",
                s, a, r$sec[1], r$max_rhat_all[1]))
  }
}
R <- do.call(rbind, all_rows)
if (is.null(R) || !nrow(R)) stop("no fits succeeded", call. = FALSE)

cat("\n=========================================================================\n")
cat(" ESS/sec by arm and quantity (mean over seeds)\n")
cat("=========================================================================\n\n")
cat(sprintf("%-16s %-16s %4s %9s %11s %7s\n",
            "arm", "quantity/param", "n", "ESS", "ESS/sec", "R-hat"))
means <- list()
for (a in c("A", "D", "D_rot", "D_dense")) {
  sub <- R[R$arm == a, ]
  if (!nrow(sub)) next
  for (q in unique(sub$quantity)) {
    for (p in unique(sub$param[sub$quantity == q])) {
      x <- sub[sub$quantity == q & sub$param == p, ]
      if (!nrow(x)) next
      cat(sprintf("%-16s %-16s %4d %9.1f %11.2f %7.3f\n",
                  a, paste0(q, "/", p), nrow(x), mean(x$ess), mean(x$ess_per_sec),
                  max(x$rhat, na.rm = TRUE)))
      if (q == "beta_corrected" && p == "beta_0") means[[a]] <- mean(x$ess_per_sec)
      # A never has beta_corrected (it is not orthogonalized) - ITS baseline
      # is its own beta/beta_0 ESS/sec, the actual reported estimand for a
      # default fit. Tracked separately so the comparison below has a
      # reference even though arm A contributes nothing to `means` above.
      if (a == "A" && q == "beta" && p == "beta_0") ref_A <- mean(x$ess_per_sec)
    }
  }
  cat("\n")
}

cat("-------------------------------------------------------------------------\n")
cat(" The decisive comparison: beta_corrected/beta_0 ESS/sec, relative to A's\n")
cat(" own beta/beta_0 (A never has beta_corrected - it is not orthogonalized;\n")
cat(" its own beta IS the reported estimand for a default fit)\n\n")
ref <- if (exists("ref_A")) ref_A else NULL
if (!is.null(ref) && is.finite(ref) && ref > 0) {
  for (a in c("D", "D_rot", "D_dense")) {
    if (!is.null(means[[a]]))
      cat(sprintf("  %-8s %6.2fx\n", a, means[[a]] / ref))
  }
}
cat("\n D is the Section 9.1.1 baseline (expect ~1.2-1.3x). If D_dense is\n")
cat(" close to D, the rotation + dense block did not help. If D_dense\n")
cat(" approaches the SWEPT beta's own ~9-15x (see the 'beta/beta_0' rows\n")
cat(" above for D_rot/D_dense), it closed the gap.\n")

cat("\n-------------------------------------------------------------------------\n")
cat(" Correctness sanity check: beta_corrected's posterior mean, D vs D_rot,\n")
cat(" same seed (should agree closely - same target, different coordinates)\n\n")
bc <- R[R$quantity == "beta_corrected" & R$param == "beta_0", c("seed", "arm", "est")]
if (nrow(bc)) {
  wide <- reshape(bc, idvar = "seed", timevar = "arm", direction = "wide")
  print(wide)
  if (all(c("est.D", "est.D_rot") %in% names(wide))) {
    cat(sprintf("\n  mean |D - D_rot| = %.4f\n",
                mean(abs(wide$est.D - wide$est.D_rot), na.rm = TRUE)))
  }
}

cat(sprintf("\n %d seed(s) only - this is a pilot.\n", SEEDS))
