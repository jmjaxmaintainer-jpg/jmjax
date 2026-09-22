# ==============================================================================
# jmjax vs JMbayes2: the recovered benchmark, with a pilot mode.
#
# PROVENANCE. This is a port of
#   dev/archive/benchmark_sweep_replicated_original.R
# recovered on 2026-09-21 from a test archive off the previous laptop. That
# script's own header states it produced the figure the validation vignette
# quotes: "267.7 (95% CI [207.7, 327.7]) vs JMbayes2's 19.9 (95% CI [15.8,
# 24.0]) result, 13.5x, paired t-test p=1.8e-7".
#
# Read that claim with one caveat the archive itself supplies: the recovered
# script's header also says "(Reconstructed from earlier session history -
# this is the exact script that produced the 13.5x ESS/sec result reported
# earlier.)" So it is a reconstruction asserting fidelity, not a literal
# original artifact. Its quoted numbers match the vignette exactly, which is
# good evidence, but it is not proof.
#
# WHY RE-RUN AT ALL. Three things have changed since it was last run:
#
#   1. float64 is now the backend default (it was float32), measured faster
#      at every size tested.
#   2. mcmc_warm_start exists and now seeds the baseline hazard, alpha and
#      gamma - which it did not when this benchmark was written.
#   3. The timing metric was FIXED. Section 3.1 of
#      inst/doc-notes/mcmc-performance-investigation.md documents that
#      `sampling_time_sec` captured only JAX *dispatch* time, not compute,
#      because mcmc.run() returns as soon as work is queued. The shipped
#      backend now calls jax.block_until_ready() before stopping the timer
#      (mcmc_model.py:1259).
#
# That third point matters most, because this script divides jmjax's ESS by
# exactly that field. The obvious inference is that 13.5x was inflated by a
# broken timer - but section 3.1 ALSO reports that after the fix, n=200 gave
# 164.9 ESS/sec against "the original benchmark's 171.9 for the same
# scenario", a 4% difference. So the inflation was not uniform and may have
# been small here. This script therefore records BOTH jmjax timings - the
# backend's own sampling_time_sec and an R-side wall clock around the whole
# call - so the question is answered by measurement instead of argument.
#
# The report's own section 5 "Final Validated Performance Comparison" gives
# 2.4x wall-clock / 3.81x ESS/sec at n=500, against this benchmark's 13.5x.
# Reconciling those two is the point of re-running.
#
# ------------------------------------------------------------------------------
# HOW TO RUN
#
#   BENCH_PILOT=1 Rscript dev/bench_vs_jmbayes2.R    # ~2 min, proves it works
#   BENCH_PILOT=0 Rscript dev/bench_vs_jmbayes2.R    # the full design, hours
#
# PILOT MODE IS NOT A SMALL BENCHMARK. It runs one scenario, one replicate,
# and a sampling budget far too short for either package to converge. Its
# only job is to prove the machinery works end to end: both packages fit,
# every metric extracts from the objects at the paths this script expects,
# and the table assembles. Do not read its numbers as a comparison.
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax)
  library(nlme)
  library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 is not installed. install.packages('JMbayes2') first - ",
       "this benchmark has nothing to compare against without it.",
       call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

# Three modes. The distinction that matters is the SAMPLING BUDGET, not the
# number of replicates: jmjax pays a fixed XLA compilation cost per model
# shape, so a short budget makes that fixed cost dominate and the ratio
# comes out backwards. "midi" therefore uses the FULL budget and cuts only
# the replicate count - it is a real comparison on few datasets, where
# "pilot" is not a comparison at all.
MODE <- tolower(Sys.getenv("BENCH_MODE", ""))
if (!nzchar(MODE)) {
  # Backward compatibility with the original BENCH_PILOT switch.
  MODE <- if (identical(Sys.getenv("BENCH_PILOT", "1"), "0")) "full" else "pilot"
}
if (!MODE %in% c("pilot", "midi", "full")) {
  stop("BENCH_MODE must be pilot, midi or full (got '", MODE, "')", call. = FALSE)
}
PILOT <- identical(MODE, "pilot")
OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

PREC  <- tryCatch(jmjax:::.backend_precision(), error = function(e) NA_character_)
STAMP <- format(Sys.time(), "%Y%m%d_%H%M")
TAG   <- MODE
OUT   <- file.path(OUTDIR, sprintf("bench_vs_jmbayes2_%s_%s", TAG, STAMP))

.con <- file(paste0(OUT, ".log"), open = "wt")
sink(.con, split = TRUE)

# ------------------------------------------------------------------------------
# Generator: VERBATIM from the recovered script. Do not "improve" it - the
# whole value of re-running this is that the data-generating process is
# identical to the one behind the published figure. alpha = 0.6 is the truth
# that |alpha error| is measured against.
# ------------------------------------------------------------------------------
simulate_joint_data_re2 <- function(n = 300, seed = 1,
                                     beta0 = 2.0, beta1 = 0.5,
                                     sigma_b0 = 0.8, sigma_b1 = 0.2, rho = 0.3,
                                     sigma_e = 0.3,
                                     weibull_shape = 1.2, log_lambda0 = -2.0,
                                     alpha = 0.6, max_time = 5,
                                     visit_times = seq(0, 5, by = 0.5)) {
  set.seed(seed)
  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)

  cum_hazard <- function(t, b0_i, b1_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      m_s <- (beta0 + b0_i) + (beta1 + b1_i) * s
      h0 * exp(alpha * m_s)
    }
    integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(runif(1))
    ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i]) < target) return(Inf)
    uniroot(function(t) cum_hazard(t, b0[i], b1[i]) - target,
            lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  list(data_long = data_long, data_surv = data_surv,
       truth = list(beta0 = beta0, beta1 = beta1, sigma_b0 = sigma_b0,
                    sigma_b1 = sigma_b1, rho = rho, sigma_e = sigma_e,
                    alpha = alpha))
}

# ------------------------------------------------------------------------------
# Design. FULL is the recovered design unchanged. Basis counts are matched by
# the confirmed formulas: jmjax n_interior_knots + ord(4) gives 9 and 11;
# JMbayes2 base_hazard_segments 6 and 8 with Bsplines_degree = 3.
# ------------------------------------------------------------------------------
# The full budget, used by BOTH midi and full - recovered unchanged.
JX_WARMUP <- 500L; JX_SAMPLES <- 1000L; JX_CHAINS <- 1L
JB_ITER   <- 1500L; JB_BURNIN <- 500L;  JB_CHAINS <- 2L

if (MODE == "pilot") {
  N_REPS    <- 1L
  scenarios <- data.frame(n = 200L, n_interior_knots_jmjax = 5L)
  JX_WARMUP <- 100L; JX_SAMPLES <- 200L; JX_CHAINS <- 1L
  JB_ITER   <- 300L; JB_BURNIN  <- 100L; JB_CHAINS <- 1L
} else if (MODE == "midi") {
  # Both sample sizes, one knot setting, 2 replicates: 4 dataset-fits per
  # package at the real budget. Enough to see whether the compilation cost
  # amortizes, whether the backend timer converges on wall clock, and how
  # long the full run will actually take.
  N_REPS    <- 2L
  scenarios <- data.frame(n = c(200L, 500L), n_interior_knots_jmjax = 5L)
} else {
  N_REPS    <- 5L
  scenarios <- expand.grid(n = c(200L, 500L),
                           n_interior_knots_jmjax = c(5L, 7L),
                           stringsAsFactors = FALSE)
}
scenarios$base_hazard_segments_jmbayes2 <-
  ifelse(scenarios$n_interior_knots_jmjax == 5L, 6L, 8L)

# ---- chain-count matching ----------------------------------------------------
# The recovered design gives JMbayes2 2 chains and jmjax 1. That is not a
# choice anyone made on the merits: it PREDATES section 3.4 of the report,
# which documents that jmjax could not run genuinely parallel chains until
# set_host_device_count() was added - before that, num_chains > 1 queued
# sequentially and bought nothing. So the design measures jmjax with one
# hand tied, and re-running it unchanged reproduces that handicap.
#
# ESS/second is only chain-count-neutral when chains are SERIAL (2 chains
# gives 2x the ESS for 2x the time). Once they are parallel, more chains
# means more ESS at roughly the same wall time - so the chain count changes
# the metric, and an unmatched count makes the comparison meaningless in
# whichever direction it happens to fall.
#
# BENCH_MATCH_CHAINS=1 gives jmjax the same chain count as JMbayes2. Both
# variants are worth recording: unmatched reproduces the historical design,
# matched is the like-for-like comparison.
if (identical(Sys.getenv("BENCH_MATCH_CHAINS", "0"), "1")) {
  JX_CHAINS <- JB_CHAINS
}

# ---- overrides, so recollections about the original run are TESTABLE --------
# The recovered script says 2 chains and n_iter = 1500 over n in {200, 500}.
# That is what it says; it is not necessarily what produced the published
# 13.5x, since the script's own header admits to being reconstructed from
# session history. Three specific recollections are worth testing rather
# than accepting or dismissing:
#
#   - "chains used was like 4"           -> BENCH_JX_CHAINS / BENCH_JB_CHAINS
#   - "larger n would favour jmjax"      -> BENCH_N=200,500,2000
#   - "JMbayes2 needs >= 3500 samples
#      for a decent R-hat"               -> BENCH_JB_ITER=4000
#
# That last one is not a tuning preference. If JMbayes2 has not converged at
# n_iter = 1500, then ESS/second is comparing a converged jmjax fit against
# an unconverged JMbayes2 one, and rewards the latter for finishing early.
# The R-hat columns below are what make that visible.
.envi <- function(nm, default) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else default
}
JX_CHAINS <- .envi("BENCH_JX_CHAINS", JX_CHAINS)
JB_CHAINS <- .envi("BENCH_JB_CHAINS", JB_CHAINS)
JB_ITER   <- .envi("BENCH_JB_ITER",   JB_ITER)
JB_BURNIN <- .envi("BENCH_JB_BURNIN", JB_BURNIN)
JX_WARMUP <- .envi("BENCH_JX_WARMUP", JX_WARMUP)
JX_SAMPLES<- .envi("BENCH_JX_SAMPLES", JX_SAMPLES)

.envn <- Sys.getenv("BENCH_N", "")
if (nzchar(.envn)) {
  .nv <- suppressWarnings(as.integer(strsplit(.envn, "[,[:space:]]+")[[1]]))
  .nv <- .nv[is.finite(.nv) & .nv > 0]
  if (length(.nv)) {
    scenarios <- expand.grid(n = .nv,
                             n_interior_knots_jmjax =
                               unique(scenarios$n_interior_knots_jmjax),
                             stringsAsFactors = FALSE)
    scenarios$base_hazard_segments_jmbayes2 <-
      ifelse(scenarios$n_interior_knots_jmjax == 5L, 6L, 8L)
  }
}

cat("=============================================================\n")
cat("  jmjax vs JMbayes2  (", MODE, " mode)\n", sep = "")
cat("  precision in force : ", PREC, "\n", sep = "")
cat("  scenarios          : ", nrow(scenarios), " x ", N_REPS, " reps = ",
    nrow(scenarios) * N_REPS, " dataset-fits per package\n", sep = "")
cat("  jmjax budget       : warmup ", JX_WARMUP, " / samples ", JX_SAMPLES,
    " / chains ", JX_CHAINS, "\n", sep = "")
cat("  JMbayes2 budget    : iter ", JB_ITER, " / burnin ", JB_BURNIN,
    " / chains ", JB_CHAINS, "\n", sep = "")
if (MODE == "pilot") {
  cat("\n  PILOT: budgets are deliberately too short to converge, and too\n")
  cat("  short for jmjax's fixed XLA compilation cost to amortize - which\n")
  cat("  pushes the ratio the WRONG WAY. This checks that the machinery\n")
  cat("  works, NOT how the packages compare. Use BENCH_MODE=midi for a\n")
  cat("  small but real comparison.\n")
} else if (MODE == "midi") {
  cat("\n  MIDI: the real sampling budget on 4 datasets per package. The\n")
  cat("  ratio here is meaningful but noisy; the projection at the end\n")
  cat("  estimates what the full run costs.\n")
}
cat("=============================================================\n")

# One dataset per (n, replicate), reused across knot settings - recovered
# design choice, so a knot effect is not confounded with dataset noise.
datasets <- list()
for (n_val in unique(scenarios$n)) {
  for (rep_i in seq_len(N_REPS)) {
    datasets[[paste0("n", n_val, "_rep", rep_i)]] <-
      simulate_joint_data_re2(n = n_val, seed = 3000 + n_val * 100 + rep_i)
  }
}

.num <- function(x) { v <- suppressWarnings(as.numeric(x)); if (length(v)) v[1] else NA_real_ }

# JMbayes2's R-hat field name is not something I can verify without running
# it, so try the plausible paths and record NA rather than guessing one and
# silently reporting nothing. An NA here is itself informative: it means the
# convergence gate below cannot be applied.
.jb_rhat <- function(f) {
  if (is.null(f)) return(NA_real_)
  cands <- list(function() f$statistics$Rhat$alphas,
                function() f$statistics$Rhat[["alphas"]],
                function() f$Rhat$alphas,
                function() f$statistics$Rhat)
  for (g in cands) {
    v <- tryCatch(.num(g()), error = function(e) NA_real_)
    if (is.finite(v)) return(v)
  }
  NA_real_
}
.at  <- function(l, nm) if (!is.null(l) && nm %in% names(l)) .num(l[[nm]]) else NA_real_

results <- list(); k <- 1L
for (i in seq_len(nrow(scenarios))) {
  sc <- scenarios[i, ]
  for (rep_i in seq_len(N_REPS)) {
    cat(sprintf("\n=== n=%d, jmjax knots=%d, JMbayes2 segments=%d, rep=%d/%d ===\n",
                sc$n, sc$n_interior_knots_jmjax,
                sc$base_hazard_segments_jmbayes2, rep_i, N_REPS))
    sim <- datasets[[paste0("n", sc$n, "_rep", rep_i)]]

    # BOTH timings. wall is measured in R around the whole call, so it
    # includes reticulate marshalling and cannot be fooled by async
    # dispatch; sampling_time_sec is the backend's own figure, which is what
    # the published 13.5x divided by. Recording both is how the section-3.1
    # question gets settled rather than argued.
    tw <- system.time(fit_jx <- tryCatch(
      jm_fit(long_formula = y ~ time,
             surv_formula = Surv(time, event) ~ 1,
             data_long = sim$data_long, data_surv = sim$data_surv,
             id_var = "id", time_var = "time",
             method = "spline-PH-mcmc", random_effects = "intercept_slope",
             control = list(n_interior_knots = sc$n_interior_knots_jmjax,
                            num_warmup = JX_WARMUP, num_samples = JX_SAMPLES,
                            num_chains = JX_CHAINS, progress_bar = FALSE,
                            spline_prior = "penalized")),
      error = function(e) { message("jmjax failed: ", conditionMessage(e)); NULL }))

    lme_fit <- tryCatch(
      lme(y ~ time, random = ~ time | id, data = sim$data_long,
          control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100)),
      error = function(e) { message("lme() failed: ", conditionMessage(e)); NULL })
    cox_fit <- coxph(Surv(time, event) ~ 1, data = sim$data_surv)

    tb <- system.time(fit_jb <- if (is.null(lme_fit)) NULL else tryCatch(
      jm(Surv_object = cox_fit, Mixed_objects = lme_fit, time_var = "time",
         n_chains = JB_CHAINS, n_iter = JB_ITER, n_burnin = JB_BURNIN,
         control = list(Bsplines_degree = 3,
                        base_hazard_segments = sc$base_hazard_segments_jmbayes2)),
      error = function(e) { message("JMbayes2 failed: ", conditionMessage(e)); NULL }))

    row <- data.frame(
      n = sc$n, rep = rep_i, precision = PREC,
      n_basis_jmjax = sc$n_interior_knots_jmjax + 4L,
      segments_jmbayes2 = sc$base_hazard_segments_jmbayes2,
      true_alpha = sim$truth$alpha,
      jx_alpha = if (is.null(fit_jx)) NA_real_ else .at(fit_jx$estimates, "alpha"),
      jx_ess   = if (is.null(fit_jx)) NA_real_ else .at(fit_jx$diagnostics$ess, "alpha"),
      jx_rhat  = if (is.null(fit_jx)) NA_real_ else .at(fit_jx$diagnostics$rhat, "alpha"),
      jx_t_backend = if (is.null(fit_jx)) NA_real_ else .num(fit_jx$convergence$sampling_time_sec),
      jx_t_wall    = as.numeric(tw[["elapsed"]]),
      jb_alpha = if (is.null(fit_jb)) NA_real_ else .num(fit_jb$statistics$Mean$alphas),
      jb_ess   = if (is.null(fit_jb)) NA_real_ else .num(fit_jb$statistics$Effective_Size$alphas),
      jb_rhat  = .jb_rhat(fit_jb),
      jx_chains = JX_CHAINS, jb_chains = JB_CHAINS,
      jb_iter = JB_ITER, jb_burnin = JB_BURNIN,
      jx_warmup = JX_WARMUP, jx_samples = JX_SAMPLES,
      jb_t_backend = if (is.null(fit_jb)) NA_real_ else .num(unname(fit_jb$running_time["elapsed"])),
      jb_t_wall    = as.numeric(tb[["elapsed"]]),
      stringsAsFactors = FALSE)

    row$jx_err        <- row$jx_alpha - row$true_alpha
    row$jb_err        <- row$jb_alpha - row$true_alpha
    row$jx_ess_sec    <- row$jx_ess / row$jx_t_backend   # as published
    row$jx_ess_sec_w  <- row$jx_ess / row$jx_t_wall      # honest wall clock
    row$jb_ess_sec    <- row$jb_ess / row$jb_t_backend
    results[[k]] <- row; k <- k + 1L

    cat(sprintf("  jmjax    alpha %7.4f (err %+7.4f)  ESS %8.1f  backend %6.1fs  wall %6.1fs  -> %7.2f / %7.2f ESS/s\n",
                row$jx_alpha, row$jx_err, row$jx_ess,
                row$jx_t_backend, row$jx_t_wall, row$jx_ess_sec, row$jx_ess_sec_w))
    cat(sprintf("  JMbayes2 alpha %7.4f (err %+7.4f)  ESS %8.1f  backend %6.1fs  wall %6.1fs  -> %7.2f ESS/s\n",
                row$jb_alpha, row$jb_err, row$jb_ess,
                row$jb_t_backend, row$jb_t_wall, row$jb_ess_sec))
    cat(sprintf("  R-hat(alpha)   jmjax %7.4f   JMbayes2 %s\n",
                row$jx_rhat,
                if (is.finite(row$jb_rhat)) sprintf("%7.4f", row$jb_rhat)
                else "     n/a (field not found)"))
  }
}

res <- do.call(rbind, results)
write.csv(res, paste0(OUT, ".csv"), row.names = FALSE)
saveRDS(list(results = res, precision = PREC, pilot = PILOT,
             session = utils::sessionInfo()), paste0(OUT, ".rds"))

cat("\n================ summary ================\n")
ok <- is.finite(res$jx_ess_sec) & is.finite(res$jb_ess_sec)
cat("  usable paired rows: ", sum(ok), " of ", nrow(res), "\n", sep = "")

if (sum(ok) >= 1L) {
  ag <- function(x) sprintf("%.4g (SD %.3g)", mean(x), if (length(x) > 1) sd(x) else NA_real_)
  cat("  jmjax    ESS/s, backend timer : ", ag(res$jx_ess_sec[ok]), "\n", sep = "")
  cat("  jmjax    ESS/s, wall clock    : ", ag(res$jx_ess_sec_w[ok]), "\n", sep = "")
  cat("  JMbayes2 ESS/s                : ", ag(res$jb_ess_sec[ok]), "\n", sep = "")
  cat("  ratio, backend timer          : ",
      sprintf("%.2fx", mean(res$jx_ess_sec[ok]) / mean(res$jb_ess_sec[ok])), "\n", sep = "")
  cat("  ratio, wall clock             : ",
      sprintf("%.2fx", mean(res$jx_ess_sec_w[ok]) / mean(res$jb_ess_sec[ok])), "\n", sep = "")
  cat("  |alpha err| jmjax / JMbayes2  : ",
      ag(abs(res$jx_err[ok])), " / ", ag(abs(res$jb_err[ok])), "\n", sep = "")
  # CONVERGENCE GATE. ESS/second is only a fair comparison between fits that
  # both converged: a chain that stops early and mixes badly scores well on
  # ESS/second while being unusable. So report R-hat health on both sides and
  # say plainly when the comparison does not hold.
  .bad_jx <- sum(res$jx_rhat[ok] >= 1.05, na.rm = TRUE)
  .bad_jb <- sum(res$jb_rhat[ok] >= 1.05, na.rm = TRUE)
  .na_jb  <- sum(!is.finite(res$jb_rhat[ok]))
  cat("  max R-hat(alpha) jmjax / JMbayes2: ",
      sprintf("%.4f", suppressWarnings(max(res$jx_rhat[ok], na.rm = TRUE))), " / ",
      if (.na_jb == sum(ok)) "not recorded"
      else sprintf("%.4f", suppressWarnings(max(res$jb_rhat[ok], na.rm = TRUE))), "\n", sep = "")
  cat("  fits with R-hat >= 1.05         : jmjax ", .bad_jx,
      ", JMbayes2 ", .bad_jb, if (.na_jb) sprintf(" (%d not recorded)", .na_jb) else "",
      " of ", sum(ok), "\n", sep = "")
  if (.bad_jb > 0 && .bad_jx == 0) {
    cat("\n  !! JMbayes2 did not converge on ", .bad_jb, " fit(s) while jmjax\n", sep = "")
    cat("  !! converged on all of them. The ESS/second comparison above is\n")
    cat("  !! NOT valid in that case - it credits JMbayes2 for finishing a\n")
    cat("  !! chain that is not usable. Raise BENCH_JB_ITER until its R-hat\n")
    cat("  !! clears 1.05 and re-run before drawing any conclusion.\n\n")
  }

  # How far apart are the two jmjax timings? This is the section-3.1 question.
  cat("  backend timer / wall clock    : ",
      sprintf("%.3f", mean(res$jx_t_backend[ok] / res$jx_t_wall[ok])),
      "  (1.0 means the backend timer is honest)\n", sep = "")
}

if (sum(ok) >= 3L && MODE != "pilot") {
  cat("\n  paired t-tests across the ", sum(ok), " matched datasets:\n", sep = "")
  for (nm in c("jx_ess_sec_w", "jx_ess_sec")) {
    tt <- t.test(res[[nm]][ok], res$jb_ess_sec[ok], paired = TRUE)
    cat(sprintf("    ESS/s (%s): t = %.2f, df = %d, p = %.3g\n",
                if (nm == "jx_ess_sec_w") "wall" else "backend timer",
                tt$statistic, tt$parameter, tt$p.value))
  }
  et <- t.test(abs(res$jx_err[ok]), abs(res$jb_err[ok]), paired = TRUE)
  cat(sprintf("    |alpha error|: t = %.2f, df = %d, p = %.3g\n",
              et$statistic, et$parameter, et$p.value))
} else if (MODE == "pilot") {
  cat("\n  No tests in pilot mode - one replicate, and budgets too short to\n")
  cat("  converge. If the two lines above printed finite numbers for both\n")
  cat("  packages, the machinery works and the full run is worth starting.\n")
}

# ---- what will the full run cost? -------------------------------------------
# Asked here because my own estimate before the pilot was wrong by an order
# of magnitude: I guessed hours from the published figure of 19.9 ESS/sec,
# which implies ~50s JMbayes2 fits, and the pilot fitted in 0.2s wall.
if (MODE %in% c("pilot", "midi")) {
  per_ds <- res$jx_t_wall + res$jb_t_wall
  if (any(is.finite(per_ds))) {
    mean_ds <- mean(per_ds[is.finite(per_ds)])
    full_n  <- 4L * 5L   # 4 scenarios x 5 reps in the recovered design
    est     <- mean_ds * full_n
    cat("\n  full-run projection\n")
    cat(sprintf("    mean wall time per dataset (both packages): %.1fs\n", mean_ds))
    cat(sprintf("    full design is %d dataset-fits per package  -> ~%.0f min\n",
                full_n, est / 60))
    if (MODE == "pilot") {
      cat("    NOTE: from pilot budgets, so this UNDERSTATES the full run by\n")
      cat("    roughly the budget ratio (5x the sampling). Run BENCH_MODE=midi\n")
      cat("    for a projection at the real budget.\n")
    } else {
      cat("    Measured at the real budget. The full design also adds the\n")
      cat("    knots=7 scenarios, slightly more expensive than knots=5, so\n")
      cat("    treat this as a lower bound.\n")
    }
  }
}

cat("\nwrote ", OUT, ".{log,csv,rds}\n", sep = "")
sink(); close(.con)
