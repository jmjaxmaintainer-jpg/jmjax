# ==============================================================================
# Simulation-based calibration / coverage study for
# control$orthogonalize_b0 and control$orthogonalize_b.
#
# WHY THIS EXISTS. vignette("jmjax-reparameterization") Section 4 proves the
# reparameterization is LIKELIHOOD-invariant: the set of achievable mean
# structures is identical with and without it (Corollary 1). That says
# nothing about the POSTERIOR. The prior on the reparameterized random-effect
# column is singular - confined to the orthogonal complement of the swept
# basis - while sigma_b still governs the unconstrained b_raw, and the swept
# directions simply sample their prior. Whether the credible intervals a user
# reports are still calibrated is an empirical question, and it is the one
# open item blocking submission (Section 9).
#
# It is not an idle worry. The earlier, retracted attempt at the same problem
# (random_effects_method = "wishart_gibbs_centered") was likelihood-invariant
# in exactly the same sense, improved beta_0's ESS 4.8x, and was RETRACTED
# because its beta_0 credible interval was too narrow: the exact zero-sum
# constraint removed real finite-sample variability in the mean of the true
# random intercepts that beta_0 must otherwise absorb. Point estimates stayed
# unbiased throughout - which is precisely why a bias check would have passed
# it. Only an interval check caught it.
#
# WHAT IS MEASURED, per arm and estimand:
#
#   coverage   - the fraction of replicates in which the true value falls
#                inside the central 50 / 80 / 95% credible interval. This is
#                what a reviewer asks for, and it is the direct analogue of
#                the check wishart_gibbs_centered failed.
#   sd_ratio   - mean(posterior SD) / SD(posterior means across replicates).
#                A calibrated posterior reports, on average, the spread its
#                point estimates actually have, so this sits at 1. Below 1
#                means intervals are too narrow. This is a far more sensitive
#                instrument than coverage at a fixed replicate count - it uses
#                the whole interval rather than a binary hit - and it is the
#                exact shape of the wishart_gibbs_centered failure.
#   bias       - mean(posterior mean) - truth, in units of the Monte Carlo
#                standard error. Reported so that "unbiased but miscalibrated"
#                remains visible AS that, rather than being read as success.
#
# ESTIMANDS, declared here rather than chosen after seeing results:
#
#   beta_0     - the intercept. The parameter the intercept-level degeneracy
#                targets and the one wishart_gibbs_centered miscalibrated.
#   beta_slope - the coefficient carrying the fixed slope on time. The
#                parameter the k-independent slope degeneracy targets, and the
#                one orthogonalize_b (arm E) treats but orthogonalize_b0 (D)
#                does not.
#   alpha      - the association parameter. NEGATIVE CONTROL: it lives in the
#                survival submodel and never touches the subject-constant
#                column space, so no arm should move its calibration. If alpha
#                under-covers in EVERY arm including the default, the finding
#                is about this model or these settings, not about
#                orthogonalization.
#   sigma_b0   - the random-intercept SD. Flagged in mcmc_model.py's own
#                source comment as the open empirical question: the swept
#                directions of b_raw sample their prior unmodified, and
#                whether that shifts sigma_b0's posterior "is an empirical
#                question, not something to assert". This is that measurement.
#
#   beta_0_corrected / beta_slope_corrected - the SEPARATE, OPT-IN
#                analytical-correction track (mcmc_model.py's
#                "beta_corrected" post-processing block, added after this
#                study's pilot run found orthogonalize_b/_b0 reproduce
#                wishart_gibbs_centered's miscalibration - see NEWS.md and
#                vignette("jmjax-reparameterization") Section 9). Computed
#                from arm D/E's own stored draws (b_std, sigma_b, L_corr),
#                with NO refit: beta_corrected = beta_orth - the beta-shift
#                matching the swept component of b_raw. The hypothesis is
#                that this recovers arm A's calibration while keeping D/E's
#                point-estimate efficiency gains. Absent for arm A (nothing
#                is swept there) and skipped automatically by add() below.
#
# ARMS. A (default), D (orthogonalize_b0), E (orthogonalize_b). A is not
# optional: it is the reference that separates "the reparameterization
# miscalibrates" from "this model at this n miscalibrates".
#
#   D_rot / D_dense (not run by default - opt in via CAL_ARMS). The rotation
#   prototype (control$orthogonalize_b0_rotate, +
#   control$dense_mass_b0_generator for D_dense; see mcmc_model.py's
#   _orthogonal_complement() and dev/pilot_b0_rotate.R) isolates D's swept
#   intercept direction as its own small NUTS site instead of leaving it
#   inside the several-hundred-dimensional b_std block. dev/pilot_b0_rotate.R
#   found this recovers beta_corrected's ESS/sec from D's ~1x-1.3x (over a
#   default fit) to roughly 5.8x, over 6 seeds - but that pilot only checked
#   posterior MEANS agree with D's (to within Monte Carlo noise), which is
#   necessary but not sufficient for calibration: the wishart_gibbs_centered
#   retraction this study's header describes ALSO had unbiased point
#   estimates throughout, and only an interval check caught the problem.
#   The rotation is a STRONGER guarantee than wishart_gibbs_centered's
#   likelihood-invariance - it is a proven rotation of a spherical Gaussian,
#   so the joint prior (and hence posterior) over every original parameter
#   is unchanged, not merely the likelihood - but "proven exact" and
#   "empirically confirmed by this study's own instrument" are different
#   claims, and this is the cheap way to check the second: beta_0_corrected's
#   coverage/sd_ratio/bias for D_rot (and D_dense) should equal D's own,
#   since correction is computed identically for both (mcmc_model.py's
#   beta_corrected block reads b_std/L_corr/sigma_b generically and does not
#   know whether b_std was rotated). Run alongside D directly comparably:
#   CAL_ARMS=A,D,D_rot Rscript dev/study_calibration.R
#
#   D_rotall / E_rotall (also opt-in): control$orthogonalize_rotate_all on
#   top of D / E - the all-column Householder rotation of vignette Section
#   4.10. Same logic as D_rot: equality with D / E is predicted exactly.
#   CAL_OUT=dev/calibration_results_rotall.csv CAL_ARMS=A,D,D_rotall,E,E_rotall \
#     CAL_REPS=100 CAL_DESIGNS=linear,mixing Rscript dev/study_calibration.R
#
# DESIGNS.
#   linear  y ~ time + covariates. Each term is its own column, so the
#           per-column and exact basis constructions agree (vignette
#           Section 4.5, condition (C)) - this is the configuration every
#           number in the vignette was measured on.
#   mixing  y ~ I(time + 0.5*time^2) + I(time^2) + covariates. The SAME
#           column space as {1, t, t^2}, but no single column is a
#           subject-constant multiple of `time` - Example 1 of Section 4.5,
#           the design the per-column search silently failed on. The true
#           mean is still linear in t and is exactly representable in this
#           basis, so the model is correctly specified and the truth is
#           recovered by projection (verified per replicate, not assumed).
#           This arm exercises the repair: under the old construction arm E
#           would have left the slope degeneracy untouched here.
#
#   NOTE the mixing design is spelled with I() rather than poly() or ns()
#   deliberately. See dev/NOTES-basis-predvars.md - build_time_design()
#   re-evaluates the longitudinal formula at the event and quadrature times
#   from a bare terms() object with no predvars, so a data-dependent basis
#   would silently differ between grids. I() expressions are pure functions
#   of time and re-evaluate correctly.
#
# POWER. With R replicates, coverage at nominal 0.95 has Monte Carlo SE
# sqrt(0.95*0.05/R): 2.2% at R = 100. That detects a true coverage at or
# below about 0.90, which is the size of miscalibration worth acting on. The
# sd_ratio diagnostic resolves considerably finer - SD(posterior means) is
# itself estimated to within about 100/sqrt(2R) percent - so a 15-20% width
# error is unmistakable at R = 100 even where coverage alone is equivocal.
#
# COST. One fit is a few tens of seconds. reps x arms x designs fits:
# 100 x 3 x 2 = 600. Run one design at a time and overnight.
#
# RESUMABLE. Every row is appended to the CSV as it completes, and re-reading
# that CSV on startup skips finished cells, so an interrupted run loses at
# most one fit.
#
#   Rscript dev/study_calibration.R --dry-run     # verify setup, no MCMC
#   caffeinate -i Rscript dev/study_calibration.R 2>&1 | tee calib.log
#   CAL_DESIGNS=mixing CAL_REPS=100 caffeinate -i Rscript dev/study_calibration.R
#   CAL_ARMS=A,D,D_rot CAL_REPS=100 caffeinate -i Rscript dev/study_calibration.R \
#     2>&1 | tee calib_rotate.log                  # does the rotation stay calibrated?
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
.envc <- function(nm, d) {
  v <- Sys.getenv(nm, "")
  if (nzchar(v)) strsplit(v, "[,[:space:]]+")[[1]] else d
}

DRY      <- any(commandArgs(trailingOnly = TRUE) == "--dry-run")
REPS     <- .envi("CAL_REPS", 100L)
N_SUB    <- .envi("CAL_N", 300L)
K_EXTRA  <- .envi("CAL_K_EXTRA", 2L)
WARMUP   <- .envi("CAL_WARMUP", 500L)
SAMPLES  <- .envi("CAL_SAMPLES", 1000L)
CHAINS   <- .envi("CAL_CHAINS", 2L)
ARMS     <- .envc("CAL_ARMS", c("A", "D", "E"))
DESIGNS  <- .envc("CAL_DESIGNS", c("linear", "mixing"))
RHAT_MAX <- 1.05
OUT      <- Sys.getenv("CAL_OUT", "dev/calibration_results.csv")

COVS <- if (K_EXTRA <= 3L) {
  c("age", "sex", "trt")[seq_len(K_EXTRA)]
} else {
  c("age", "sex", "trt", paste0("x", 4:K_EXTRA))
}

# ---- design definitions ------------------------------------------------
# Each returns the longitudinal formula and the name of the column carrying
# the fixed slope on time, which is what "beta_slope" means in that basis.
design_spec <- function(design) {
  cov_part <- if (length(COVS)) paste("+", paste(COVS, collapse = " + ")) else ""
  if (identical(design, "linear")) {
    list(lform = stats::as.formula(paste("y ~ time", cov_part)),
         slope_idx = 2L)
  } else if (identical(design, "mixing")) {
    list(lform = stats::as.formula(
           paste("y ~ I(time + 0.5 * time^2) + I(time^2)", cov_part)),
         slope_idx = 2L)
  } else {
    stop("unknown design: ", design, call. = FALSE)
  }
}

# ---- truth in the FITTED basis, by projection --------------------------
# Rather than matching coefficient names (brittle for I() terms, and silently
# wrong if model.matrix reorders), solve for the coefficient vector that
# reproduces the true fixed part of the mean exactly. This also VERIFIES the
# design is correctly specified: if the basis cannot represent the true mean
# the residual is non-zero and the run aborts rather than measuring coverage
# of a misspecified model.
true_beta <- function(lform, dl, truth) {
  X <- stats::model.matrix(lform, dl)
  cov_eff <- rep(0, nrow(dl))
  if (length(COVS)) {
    cm <- as.matrix(dl[, COVS, drop = FALSE])
    cov_eff <- as.numeric(cm %*% truth$beta_cov[seq_along(COVS)])
  }
  mu_fixed <- truth$beta0 + cov_eff + truth$beta1 * dl$time
  bt <- qr.solve(X, mu_fixed)
  resid <- max(abs(as.numeric(X %*% bt) - mu_fixed))
  scale <- max(1, max(abs(mu_fixed)))
  if (!is.finite(resid) || resid > 1e-8 * scale) {
    stop(sprintf("design cannot represent the true mean (residual %.2e) - ",
                 resid), "the model would be misspecified", call. = FALSE)
  }
  names(bt) <- colnames(X)
  bt
}

# ---- draws -> summary ---------------------------------------------------
as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
interval_row <- function(draws, truth_val) {
  q <- as.numeric(stats::quantile(draws, c(0.025, 0.10, 0.25, 0.75, 0.90, 0.975),
                                  names = FALSE, na.rm = TRUE))
  data.frame(
    truth = truth_val,
    post_mean = mean(draws), post_sd = stats::sd(draws),
    q025 = q[1], q10 = q[2], q25 = q[3], q75 = q[4], q90 = q[5], q975 = q[6],
    cov50 = as.integer(truth_val >= q[3] && truth_val <= q[4]),
    cov80 = as.integer(truth_val >= q[2] && truth_val <= q[5]),
    cov95 = as.integer(truth_val >= q[1] && truth_val <= q[6]),
    stringsAsFactors = FALSE)
}

# ---- one fit ------------------------------------------------------------
fit_one <- function(design, arm, rep_id) {
  spec <- design_spec(design)
  sim  <- sim_joint(n = N_SUB, seed = 7000L + rep_id, k_extra = K_EXTRA)
  dl   <- sim$data_long; ds <- sim$data_surv
  bt   <- true_beta(spec$lform, dl, sim$truth)

  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = rep_id, progress_bar = FALSE)
  if (arm == "D") ctl$orthogonalize_b0 <- TRUE
  if (arm == "E") ctl$orthogonalize_b  <- TRUE
  # D_rot/D_dense: the rotation prototype (mcmc_model.py's
  # _orthogonal_complement()), scoped to D's intercept-only sweep exactly
  # as D itself is - see the "D_rot / D_dense" paragraph above.
  if (arm == "D_rot")   { ctl$orthogonalize_b0 <- TRUE; ctl$orthogonalize_b0_rotate <- TRUE }
  if (arm == "D_dense") { ctl$orthogonalize_b0 <- TRUE; ctl$orthogonalize_b0_rotate <- TRUE
                           ctl$dense_mass_b0_generator <- TRUE }
  # D_rotall / E_rotall: control$orthogonalize_rotate_all (every
  # random-effect column rotated by one Householder matrix; vignette
  # Section 4.10, Corollary 4). Proposition 8 predicts calibration EQUAL to
  # D's / E's, so these arms check the implementation, as D_rot did.
  if (arm == "D_rotall") { ctl$orthogonalize_b0 <- TRUE; ctl$orthogonalize_rotate_all <- TRUE }
  if (arm == "E_rotall") { ctl$orthogonalize_b  <- TRUE; ctl$orthogonalize_rotate_all <- TRUE }
  # A_rotdense: the no-sweep alternative (rotate_absorbable + a dense block
  # over beta and b_gen_U). Reports the ordinary beta, like A, so its
  # calibration rows are directly comparable to A's.
  if (arm == "A_rotdense") { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  if (arm == "E_rotall_dense") { ctl$orthogonalize_b <- TRUE; ctl$orthogonalize_rotate_all <- TRUE
                                 ctl$dense_mass_generator <- TRUE }

  tm <- system.time(f <- jm_fit(
    long_formula = spec$lform,
    surv_formula = survival::Surv(time, event) ~ trt,
    data_long = dl, data_surv = ds,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time, control = ctl))

  ps <- f$posterior_samples
  B  <- as_draws(ps[["beta"]])
  if (is.null(B)) stop("no beta draws returned", call. = FALSE)
  if (ncol(B) != length(bt))
    stop(sprintf("beta has %d columns but the design matrix has %d - ",
                 ncol(B), length(bt)),
         "the truth mapping would be misaligned", call. = FALSE)
  colnames(B) <- names(bt)

  rh  <- unlist(f$diagnostics$rhat)
  rh  <- rh[is.finite(rh)]
  reg <- rh[grepl("^(beta|alpha|gamma)", names(rh))]

  rows <- list()
  add <- function(param, draws, truth_val) {
    if (is.null(draws) || !length(draws)) return(invisible(NULL))
    r <- interval_row(as.numeric(draws), truth_val)
    r$param <- param
    rows[[length(rows) + 1L]] <<- r
  }
  i0 <- match("(Intercept)", colnames(B))
  if (is.na(i0)) stop("no intercept column in the design", call. = FALSE)
  add("beta_0", B[, i0], bt[[i0]])
  j <- spec$slope_idx
  # Assert the slope column is time-varying WITHIN subject. A covariate is
  # subject-constant, so this catches a formula edit that silently moved the
  # column beta_slope is supposed to track.
  Xd <- stats::model.matrix(spec$lform, dl)
  wv <- max(tapply(Xd[, j], dl$id, function(v) diff(range(v))))
  if (!is.finite(wv) || wv <= 1e-8)
    stop("column ", j, " ('", colnames(Xd)[j], "') is subject-constant; ",
         "it cannot be the fixed slope on time", call. = FALSE)
  add("beta_slope", B[, j], bt[[j]])

  # Corrected estimand (SEPARATE, OPT-IN TRACK - see ESTIMANDS above).
  # Present only for arms where orthogonalize_b/_b0 actually found and
  # swept a direction; NULL for arm A and skipped there by add()'s guard.
  # Same column order as B (both come from the same p-length beta), so the
  # same i0/j indices apply without re-deriving them.
  Bc <- as_draws(ps[["beta_corrected"]])
  if (!is.null(Bc)) {
    if (ncol(Bc) != length(bt))
      stop(sprintf("beta_corrected has %d columns but the design matrix has %d - ",
                   ncol(Bc), length(bt)),
           "the truth mapping would be misaligned", call. = FALSE)
    colnames(Bc) <- names(bt)
    add("beta_0_corrected", Bc[, i0], bt[[i0]])
    add("beta_slope_corrected", Bc[, j], bt[[j]])
  }

  A <- as_draws(ps[["alpha"]])
  if (!is.null(A)) add("alpha", A[, 1L], sim$truth$alpha)
  SB <- as_draws(ps[["sigma_b"]])
  if (!is.null(SB)) add("sigma_b0", SB[, 1L], sim$truth$sigma_b0)

  out <- do.call(rbind, rows)
  out$design <- design; out$arm <- arm; out$rep <- rep_id
  out$rhat_reg <- if (length(reg)) max(reg) else NA_real_
  out$rhat_all <- if (length(rh)) max(rh) else NA_real_
  out$ndiv <- as.numeric(f$convergence$n_divergences %||% NA_real_)
  out$sec <- as.numeric(tm[["elapsed"]])
  # The settings a row was produced under are part of its identity. Without
  # them a cheap smoke row (CAL_WARMUP=50) silently satisfies the resume
  # check and contaminates a real run. Same reason the JMbayes2 study keys
  # its cache on jb_iter.
  out$n_sub <- N_SUB; out$warmup <- WARMUP
  out$samples <- SAMPLES; out$chains <- CHAINS
  out[, c("design", "arm", "rep", "param", "truth", "post_mean", "post_sd",
          "q025", "q10", "q25", "q75", "q90", "q975",
          "cov50", "cov80", "cov95", "rhat_reg", "rhat_all", "ndiv", "sec",
          "n_sub", "warmup", "samples", "chains")]
}

# ---- dry run ------------------------------------------------------------
if (DRY) {
  cat("\nDRY RUN - setup only, no MCMC\n\n")
  for (d in DESIGNS) {
    spec <- design_spec(d)
    sim  <- sim_joint(n = N_SUB, seed = 7001L, k_extra = K_EXTRA)
    bt   <- true_beta(spec$lform, sim$data_long, sim$truth)
    cat(sprintf("design '%s'\n  formula   : %s\n", d,
                paste(deparse(spec$lform), collapse = " ")))
    cat(sprintf("  slope col : [%d] %s\n", spec$slope_idx,
                colnames(stats::model.matrix(spec$lform, sim$data_long))[spec$slope_idx]))
    cat("  true beta in this basis (recovered by projection, residual 0):\n")
    for (nm in names(bt)) cat(sprintf("     %-28s %+.4f\n", nm, bt[[nm]]))
    cat(sprintf("  event rate %.1f%%, %.1f obs/subject, %d long obs\n\n",
                100 * sim$event_rate, sim$obs_per_subject, sim$n_obs))
  }
  cat(sprintf("grid: %d reps x %d arms (%s) x %d designs (%s) = %d fits\n",
              REPS, length(ARMS), paste(ARMS, collapse = ","),
              length(DESIGNS), paste(DESIGNS, collapse = ","),
              REPS * length(ARMS) * length(DESIGNS)))
  cat(sprintf("n = %d, %d warmup + %d samples x %d chains\n",
              N_SUB, WARMUP, SAMPLES, CHAINS))
  cat(sprintf("coverage MC SE at nominal 0.95 with %d reps: %.3f\n",
              REPS, sqrt(0.95 * 0.05 / REPS)))
  cat("\nsetup verified. Drop --dry-run to fit.\n")
  quit(status = 0)
}

# ---- main loop ----------------------------------------------------------
# The row schema is part of the file format. Appending rows with more columns
# than the existing header produces a ragged CSV that read.csv cannot parse,
# and the failure would land hours into a run. If the header does not match,
# move the old file aside rather than appending to it or deleting it.
EXPECTED_COLS <- c("design", "arm", "rep", "param", "truth", "post_mean",
                   "post_sd", "q025", "q10", "q25", "q75", "q90", "q975",
                   "cov50", "cov80", "cov95", "rhat_reg", "rhat_all", "ndiv",
                   "sec", "n_sub", "warmup", "samples", "chains")
if (file.exists(OUT)) {
  hdr <- tryCatch(names(utils::read.csv(OUT, nrows = 1L,
                                        stringsAsFactors = FALSE)),
                  error = function(e) character(0))
  if (!identical(hdr, EXPECTED_COLS)) {
    bak <- paste0(OUT, ".bak-", format(Sys.time(), "%Y%m%d-%H%M%S"))
    ok <- file.rename(OUT, bak)
    if (!ok) stop("cannot move the incompatible CSV aside: ", OUT, call. = FALSE)
    cat(sprintf("Row schema changed since %s was written.\n", OUT))
    cat(sprintf("Moved it to %s and starting a fresh file.\n", bak))
  }
}

done <- if (file.exists(OUT)) {
  utils::read.csv(OUT, stringsAsFactors = FALSE)
} else {
  NULL
}
settings_cols <- c("n_sub", "warmup", "samples", "chains")
have <- function(d, a, r) {
  if (is.null(done) || !nrow(done)) return(FALSE)
  hit <- done$design == d & done$arm == a & done$rep == r
  # Rows from an older CSV without the settings columns cannot be shown to
  # match, so they are re-fitted rather than trusted.
  if (!all(settings_cols %in% names(done))) return(FALSE)
  hit <- hit & done$n_sub == N_SUB & done$warmup == WARMUP &
         done$samples == SAMPLES & done$chains == CHAINS
  any(hit)
}
append_rows <- function(df) {
  # Compute this ONCE. Passed as two lazily-evaluated arguments, `append`
  # is forced first and opens (creating) the file, after which
  # `!file.exists(OUT)` is FALSE and the header is silently skipped. It
  # happens to work today; it should not depend on argument evaluation
  # order inside write.table.
  first <- !file.exists(OUT)
  utils::write.table(df, OUT, sep = ",", row.names = FALSE,
                     col.names = first, append = !first)
}

t_start <- Sys.time(); n_done <- 0L
total <- length(DESIGNS) * length(ARMS) * REPS
n_skipped <- 0L
for (d in DESIGNS) for (r in seq_len(REPS)) for (a in ARMS) {
  if (have(d, a, r)) n_skipped <- n_skipped + 1L
}
if (n_skipped > 0L)
  cat(sprintf("resuming: %d of %d fits already in %s\n", n_skipped, total, OUT))
for (d in DESIGNS) {
  for (r in seq_len(REPS)) {
    for (a in ARMS) {
      if (have(d, a, r)) next
      res <- try(fit_one(d, a, r), silent = TRUE)
      if (inherits(res, "try-error")) {
        cat(sprintf("  [%s r%03d %s] FAILED: %s", d, r, a,
                    as.character(res)))
        next
      }
      append_rows(res)
      n_done <- n_done + 1L
      el <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
      left <- total - n_skipped - n_done
      eta <- if (n_done > 0L && left > 0L) el / n_done * left else 0
      cat(sprintf("  [%s r%03d %s] %5.1fs  rhat_reg %.3f  div %s  (%d/%d, %.1f min, ETA %.0f min)\n",
                  d, r, a, res$sec[1], res$rhat_reg[1],
                  format(res$ndiv[1]), n_done, total - n_skipped, el, eta))
    }
  }
}

# ---- summary ------------------------------------------------------------
R <- utils::read.csv(OUT, stringsAsFactors = FALSE)
R <- R[R$design %in% DESIGNS & R$arm %in% ARMS, , drop = FALSE]
if (all(settings_cols %in% names(R))) {
  n_before <- nrow(R)
  R <- R[R$n_sub == N_SUB & R$warmup == WARMUP &
         R$samples == SAMPLES & R$chains == CHAINS, , drop = FALSE]
  if (nrow(R) < n_before)
    cat(sprintf("\nIgnoring %d row(s) produced under different sampler settings.\n",
                n_before - nrow(R)))
}
keep <- is.finite(R$rhat_reg) & R$rhat_reg <= RHAT_MAX
cat(sprintf("\n\nConvergence gate (regression-only R-hat <= %.2f): %d of %d rows kept\n",
            RHAT_MAX, sum(keep), nrow(R)))
drop_tab <- table(R$design[!keep], R$arm[!keep])
if (sum(!keep)) { cat("dropped per design x arm:\n"); print(drop_tab) }
R <- R[keep, , drop = FALSE]
if (!nrow(R)) {
  cat("\nNo rows survived the convergence gate - nothing to summarise.\n")
  quit(status = 1)
}

cat("\n")
cat("=========================================================================\n")
cat(" CALIBRATION. coverage = fraction of replicates containing the truth.\n")
cat(" sd_ratio = mean(posterior SD) / SD(posterior means); 1 is calibrated,\n")
cat(" below 1 means intervals too narrow. bias is in MC standard errors.\n")
cat("=========================================================================\n\n")
cat(sprintf("%-8s %-4s %-11s %4s %7s %7s %7s %9s %8s\n",
            "design", "arm", "param", "n", "cov50", "cov80", "cov95",
            "sd_ratio", "bias/se"))
thin <- 0L
for (d in unique(R$design)) for (p in unique(R$param)) for (a in ARMS) {
  x <- R[R$design == d & R$param == p & R$arm == a, , drop = FALSE]
  if (!nrow(x)) next
  # sd() needs >= 2 observations; with fewer there is no empirical spread to
  # compare the posterior SD against, so sd_ratio and bias/se are undefined
  # rather than zero. Reported as NA instead of crashing the summary.
  sd_emp <- if (nrow(x) >= 2L) stats::sd(x$post_mean) else NA_real_
  ok_sd  <- is.finite(sd_emp) && sd_emp > 0
  if (nrow(x) < 2L) thin <- thin + 1L
  bias   <- mean(x$post_mean) - x$truth[1]
  se     <- if (ok_sd) sd_emp / sqrt(nrow(x)) else NA_real_
  cat(sprintf("%-8s %-4s %-11s %4d %7.3f %7.3f %7.3f %9s %8s\n",
              d, a, p, nrow(x), mean(x$cov50), mean(x$cov80), mean(x$cov95),
              if (ok_sd) sprintf("%.3f", mean(x$post_sd) / sd_emp) else "-",
              if (ok_sd) sprintf("%.2f", bias / se) else "-"))
}
if (thin > 0L) {
  cat(sprintf("\n%d cell(s) had fewer than 2 replicates; sd_ratio and bias/se\n",
              thin))
  cat("need at least 2 and are shown as '-'. Coverage from a handful of\n")
  cat("replicates is not evidence either - this is a smoke test, not a result.\n")
}
cat(sprintf("\nMonte Carlo SE on 95%% coverage at these counts: ~%.3f\n",
            sqrt(0.95 * 0.05 / max(1, min(table(R$arm))))))
cat("\nRead this against arm A. A departure shared by A, D and E is a\n")
cat("property of the model or these settings, not of orthogonalization.\n")
cat("alpha is the negative control: no arm should move it.\n")
