# ==============================================================================
# Does float64 pay for itself? Measure it rather than argue about it.
#
# WHY THIS EXISTS. jmjax ran in float32 for its entire benchmarked history
# without any line of code choosing that: JAX defaults to single precision
# and silently downcasts float64 arrays handed to it. The default has now
# been flipped to float64 (jmjax_setup(enable_x64 = TRUE), the default), and
# that flip needs evidence at BOTH ends of the size range, not just the one
# point where it was noticed.
#
# WHAT IS ALREADY KNOWN, and what is not:
#
#   n = 8,000, MCMC   float64 was 4.24x FASTER end-to-end, ESS for alpha
#                     4 -> 1188. Measured once, on one seed. The mechanism
#                     was step count: float32 gradient noise drove NUTS into
#                     roughly four times as many leapfrog steps.
#
#   small n, MCMC     UNMEASURED. This is the real open question. float64
#                     does more work per operation, so at sizes where
#                     gradient noise does not dominate it should be SLOWER.
#                     If it is slower by a lot at n = 200, a size-dependent
#                     default deserves discussion; if it is slower by a few
#                     percent, the simpler always-on default wins.
#
#   ML / aGH path     UNMEASURED, and arguably where precision matters most.
#                     common.py computes the gradient and Hessian in JAX and
#                     hands them to scipy's L-BFGS-B, whose default
#                     tolerances assume ~1e-15 relative accuracy; float32
#                     supplies ~1e-7. The Hessian is then INVERTED for the
#                     standard errors. Two documented pathologies in
#                     common.py's own docstring - a line search that
#                     "terminated ABNORMALLY at iteration 0 with a finite
#                     objective AND a finite gradient", and a run that
#                     "stopped after 5 iterations with converged = TRUE"
#                     having left 78 log-likelihood units on the table -
#                     were attributed to parameter scaling and fixed with
#                     parscale. Float32 gradients are a PLAUSIBLE SECOND
#                     CAUSE of the same symptom. This script does not assume
#                     that; it measures grad_max, which is exactly the
#                     signal that would move if it were true.
#
# PRE-REGISTERED CRITERIA, written before running:
#
#   FLIP IS JUSTIFIED if, at n >= 1000, float64 is faster or within 10% on
#   ESS/second, AND at n = 200 float64 costs less than 2x wall time.
#
#   FLIP NEEDS RETHINKING if float64 is more than 2x slower at n = 200 with
#   no compensating ESS gain - that would argue for a size-dependent default
#   rather than an unconditional one.
#
#   THE ML HYPOTHESIS IS SUPPORTED if float64 reaches a materially smaller
#   grad_max at the optimum, or a higher log-likelihood, on the same data
#   and the same starting values. It is REFUTED if those are indistinguishable,
#   in which case parscale was the whole story and this script says so.
#
# Standard errors are compared too, but note what that can and cannot show:
# a difference proves float32's SEs were wrong, while agreement does not
# prove they were right on other data - a well-conditioned Hessian inverts
# acceptably in single precision, an ill-conditioned one does not.
#
# ------------------------------------------------------------------------------
# HOW TO RUN. Precision is fixed at the first jax import in a process, so a
# single R session cannot measure both. Run the wrapper, which spawns one
# process per setting and then compares:
#
#     bash dev/bench_x64.sh
#
# Or by hand:
#
#     JMJAX_ENABLE_X64=1 Rscript dev/bench_x64.R
#     JMJAX_ENABLE_X64=0 Rscript dev/bench_x64.R
#     Rscript dev/bench_x64.R compare
#
# Sizes and seeds are configurable. n = 8,000 is opt-in because it is hours,
# not minutes:
#
#     BENCH_SIZES=200,1000,8000 BENCH_SEEDS=1,2,3 bash dev/bench_x64.sh
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax)
  library(survival)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

.env_nums <- function(nm, default) {
  raw <- Sys.getenv(nm, "")
  if (!nzchar(raw)) return(default)
  v <- suppressWarnings(as.numeric(strsplit(raw, "[,[:space:]]+")[[1]]))
  v <- v[is.finite(v)]
  if (!length(v)) default else v
}

SIZES <- as.integer(.env_nums("BENCH_SIZES", c(200, 1000)))
SEEDS <- as.integer(.env_nums("BENCH_SEEDS", c(1, 2, 3)))
ML_N  <- as.integer(.env_nums("BENCH_ML_N", 400)[1])

# ------------------------------------------------------------------------------
# Generator. This is sim_C from dev/test_n8000_rho.R, reproduced verbatim so
# that these numbers sit on the same scale as the existing corpus. A new
# generator here would make the float64 numbers incomparable to every
# float32 number already recorded, which would defeat the purpose.
# ------------------------------------------------------------------------------
BETA0 <- 2.0; BETA1 <- 0.5; BX <- 0.3
SD_B0 <- 0.8; SD_B1 <- 0.2; SIGMA_E <- 0.3
ALPHA <- 0.4; TMAX <- 10; NVISIT <- 8L; TARGET_EVENT <- 0.53
VISIT <- seq(0, TMAX, length.out = NVISIT)

.assemble <- function(b0, b1, x, ot, ev) {
  n <- length(b0)
  reps <- vapply(ot, function(o) max(1L, sum(VISIT <= o)), integer(1))
  dl <- data.frame(id = factor(rep(seq_len(n), reps), levels = seq_len(n)),
                   time = unlist(lapply(reps, function(k) VISIT[seq_len(k)])),
                   x = rep(x, reps))
  dl$y <- (BETA0 + BX*dl$x + rep(b0, reps)) +
          (BETA1 + rep(b1, reps))*dl$time + rnorm(nrow(dl), 0, SIGMA_E)
  ds <- data.frame(id = factor(seq_len(n)), time = ot,
                   event = as.integer(ev), x = x)
  list(l = dl, s = ds, ev = mean(ev), rho_emp = stats::cor(b0, b1))
}

.invert <- function(E, A, D, lam0) {
  aD <- ALPHA * D; base <- lam0 * exp(ALPHA * A)
  out <- numeric(length(E))
  lin <- abs(aD) < 1e-8
  out[lin] <- E[lin] / base[lin]
  g <- !lin
  arg <- 1 + E[g] * aD[g] / base[g]
  out[g] <- ifelse(arg > 0, log(arg) / aD[g], Inf)
  out
}

sim_C <- function(n, seed, rho = 0.3) {
  set.seed(seed)
  b0 <- rnorm(n, 0, SD_B0); z <- rnorm(n)
  b1 <- rho * (SD_B1/SD_B0) * b0 + sqrt(1 - rho^2) * SD_B1 * z
  x <- rnorm(n); E <- rexp(n)
  A <- BETA0 + BX*x + b0; D <- BETA1 + b1
  rate_at <- function(L) mean(.invert(E, A, D, exp(L)) <= TMAX)
  L <- tryCatch(stats::uniroot(function(z) rate_at(z) - TARGET_EVENT,
                               lower = -20, upper = 5, tol = 1e-6)$root,
                error = function(e) { warning("calibration failed"); -4.6 })
  Tt <- .invert(E, A, D, exp(L))
  d <- .assemble(b0, b1, x, pmin(Tt, TMAX), Tt <= TMAX)
  d$log_lam0 <- L
  d
}

fm <- function(v, d = 3) if (is.finite(v)) formatC(v, format = "f", digits = d) else "?"
pick <- function(l, nm) {
  if (is.null(l) || !(nm %in% names(l))) return(NA_real_)
  v <- suppressWarnings(as.numeric(l[[nm]]))
  if (length(v) && is.finite(v[1])) v[1] else NA_real_
}

# estimates/ess are NAMED NUMERIC VECTORS, and `[["rho"]]` on a name that
# is not there is an ERROR, not NA - rho is absent whenever the model has a
# single random effect, and a benchmark should not die on that.
.at <- function(v, nm) {
  if (is.null(v) || !(nm %in% names(v))) return(NA_real_)
  suppressWarnings(as.numeric(v[[nm]]))[1]
}

# ==============================================================================
# compare mode
# ==============================================================================
if (length(commandArgs(TRUE)) && commandArgs(TRUE)[1] == "compare") {
  newest <- function(tag) {
    f <- sort(list.files(getwd(), pattern = sprintf("^bench_x64_%s_.*\\.rds$", tag),
                         full.names = TRUE), decreasing = TRUE)
    if (!length(f)) NULL else readRDS(f[1])
  }
  a <- newest("float64"); b <- newest("float32")
  if (is.null(a) || is.null(b)) {
    cat("Need one run of each precision first. Found:",
        if (is.null(a)) "" else "float64", if (is.null(b)) "" else "float32", "\n")
    quit(status = 1)
  }

  cat("\n=================== MCMC: float64 vs float32 ===================\n")
  cat(sprintf("  %6s %5s | %9s %9s | %9s %9s | %8s %8s | %7s\n",
              "n", "seed", "t64", "t32", "ESSa/s64", "ESSa/s32",
              "steps64", "steps32", "speedup"))
  m64 <- a$mcmc; m32 <- b$mcmc
  key <- function(d) paste(d$n, d$seed)
  for (k in intersect(key(m64), key(m32))) {
    r <- m64[key(m64) == k, ][1, ]; s <- m32[key(m32) == k, ][1, ]
    cat(sprintf("  %6d %5d | %9s %9s | %9s %9s | %8s %8s | %7s\n",
                r$n, r$seed, fm(r$time, 1), fm(s$time, 1),
                fm(r$ess_alpha_per_sec, 1), fm(s$ess_alpha_per_sec, 1),
                fm(r$steps, 1), fm(s$steps, 1),
                paste0(fm(s$time / r$time, 2), "x")))
  }

  cat("\n  R-hat / divergences (a float64 win that breaks convergence is not a win)\n")
  cat(sprintf("  %6s %5s | %9s %9s | %6s %6s\n",
              "n", "seed", "rhat64", "rhat32", "div64", "div32"))
  for (k in intersect(key(m64), key(m32))) {
    r <- m64[key(m64) == k, ][1, ]; s <- m32[key(m32) == k, ][1, ]
    cat(sprintf("  %6d %5d | %9s %9s | %6s %6s\n", r$n, r$seed,
                fm(r$rhat, 4), fm(s$rhat, 4), fm(r$ndiv, 0), fm(s$ndiv, 0)))
  }

  cat("\n=============== Maximum likelihood (aGH) path ==================\n")
  l64 <- a$ml; l32 <- b$ml
  if (!is.null(l64) && !is.null(l32) && nrow(l64) && nrow(l32)) {
    if (!identical(l64$random_effects[1], l32$random_effects[1])) {
      cat("  !! The two arms fell back to DIFFERENT random-effects structures\n")
      cat("  !! (", l64$random_effects[1], " vs ", l32$random_effects[1],
          "). These are different models; the difference below is not a\n", sep = "")
      cat("  !! precision effect. Investigate the fallback before reading on.\n\n")
    }
    cat(sprintf("  %-18s %14s %14s\n", "", "float64", "float32"))
    for (f in c("loglik", "grad_max", "grad_norm", "n_iter", "time")) {
      cat(sprintf("  %-18s %14s %14s\n", f,
                  fm(l64[[f]][1], 4), fm(l32[[f]][1], 4)))
    }
    cat(sprintf("  %-18s %14s %14s\n", "converged",
                l64$converged[1], l32$converged[1]))
    cat("\n  Standard errors (a difference PROVES the float32 SEs were wrong;\n",
        "  agreement does not prove they are right on other data)\n", sep = "")
    se64 <- a$ml_se; se32 <- b$ml_se
    nm <- intersect(names(se64), names(se32))
    cat(sprintf("  %-14s %12s %12s %10s\n", "param", "se64", "se32", "rel.diff"))
    for (p in nm) {
      rel <- abs(se32[[p]] - se64[[p]]) / max(1e-12, abs(se64[[p]]))
      cat(sprintf("  %-14s %12s %12s %10s\n", p,
                  fm(se64[[p]], 5), fm(se32[[p]], 5), fm(rel, 4)))
    }
  } else {
    cat("  (ML arm missing from one or both runs)\n")
  }
  cat("\n")
  quit(status = 0)
}

# ==============================================================================
# measurement mode
# ==============================================================================
PREC  <- jmjax:::.backend_precision()
STAMP <- format(Sys.time(), "%Y%m%d_%H%M")
OUT   <- file.path(getwd(), sprintf("bench_x64_%s_%s", PREC, STAMP))

.con <- file(paste0(OUT, ".log"), open = "wt")
sink(.con, split = TRUE)

cat("=============================================================\n")
cat("  jmjax precision benchmark\n")
cat("  precision in force : ", PREC, "\n", sep = "")
cat("  JMJAX_ENABLE_X64   : ", Sys.getenv("JMJAX_ENABLE_X64", "<unset>"), "\n", sep = "")
cat("  sizes              : ", paste(SIZES, collapse = ", "), "\n", sep = "")
cat("  seeds              : ", paste(SEEDS, collapse = ", "), "\n", sep = "")
cat("  started            : ", format(Sys.time()), "\n", sep = "")
cat("=============================================================\n\n")

# A run that silently measured the wrong thing would be worse than no run.
if (!identical(PREC, if (identical(Sys.getenv("JMJAX_ENABLE_X64", "1"), "0"))
                       "float32" else "float64")) {
  cat("!! The backend is running in ", PREC, " but JMJAX_ENABLE_X64 asked for\n",
      "!! something else. Another package probably imported jax first.\n",
      "!! Results are labelled by what is RUNNING, which is correct, but the\n",
      "!! two arms may not differ the way you intended.\n\n", sep = "")
}

# ---- MCMC arm ---------------------------------------------------------------
mcmc_rows <- list()
for (n in SIZES) {
  d <- sim_C(n, seed = 1L)
  cat(sprintf("-- n = %d (event rate %.3f, empirical rho %.3f)\n",
              n, d$ev, d$rho_emp))
  for (s in SEEDS) {
    tt <- system.time(f <- tryCatch(
      jm_fit(long_formula = y ~ time + x,
             surv_formula = Surv(time, event) ~ 1,
             data_long = d$l, data_surv = d$s,
             id_var = "id", time_var = "time",
             method = "spline-PH-mcmc", random_effects = "intercept_slope",
             random_formula = ~ time,
             control = list(n_interior_knots = 5L, spline_prior = "penalized",
                            num_warmup = 500L, num_samples = 500L,
                            num_chains = 2L, seed = s, progress_bar = FALSE)),
      error = function(e) { cat("    FIT ERROR:", conditionMessage(e), "\n"); NULL }))
    if (is.null(f)) next

    rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
    es <- f$diagnostics$ess
    tsec <- pick(f$convergence, "sampling_time_sec") %||% as.numeric(tt[["elapsed"]])
    if (!is.finite(tsec)) tsec <- as.numeric(tt[["elapsed"]])
    ess_a <- .at(unlist(es), "alpha")

    row <- data.frame(
      n = n, seed = s, precision = PREC,
      time = tsec,
      rhat = if (length(rh)) max(rh) else NA_real_,
      ess_min = suppressWarnings(min(unlist(es), na.rm = TRUE)),
      ess_alpha = ess_a,
      ess_alpha_per_sec = ess_a / tsec,
      steps = pick(f$convergence, "mean_num_steps"),
      ndiv = pick(f$convergence, "n_divergences"),
      est_alpha = .at(f$estimates, "alpha"),
      est_rho = .at(f$estimates, "rho"),
      stringsAsFactors = FALSE)
    mcmc_rows[[length(mcmc_rows) + 1L]] <- row

    cat(sprintf("   seed %3d  %8.1fs  rhat %7.4f  ESSa %7.1f  %6.2f ESSa/s  steps %6.1f  div %3.0f  alpha %6.3f\n",
                s, row$time, row$rhat, row$ess_alpha, row$ess_alpha_per_sec,
                row$steps, row$ndiv, row$est_alpha))
  }
  cat("\n")
}
mcmc <- if (length(mcmc_rows)) do.call(rbind, mcmc_rows) else
        data.frame(n = integer(), seed = integer())

# ---- maximum-likelihood arm -------------------------------------------------
# The sharp test. grad_max is the gradient's largest component at the
# reported optimum: at a true stationary point it is ~0, and an optimizer
# that stalled on noisy gradients leaves it visibly larger. Same data, same
# starting values, only the arithmetic differs.
cat("-- maximum likelihood (spline-PH-aGH), n =", ML_N, "\n")
dm <- sim_C(ML_N, seed = 1L)
ml <- NULL; ml_se <- NULL

# Try the two-random-effect model first, fall back to intercept-only. The
# fallback is not padding: if intercept_slope is unsupported or unstable
# for this method, failing outright would lose the ML comparison entirely
# in BOTH arms, and the ML arm is the sharpest test in this script.
.ml_fit <- function(re, rf) {
  tryCatch(
    jm_fit(long_formula = y ~ time + x,
           surv_formula = Surv(time, event) ~ 1,
           data_long = dm$l, data_surv = dm$s,
           id_var = "id", time_var = "time",
           method = "spline-PH-aGH", random_effects = re,
           random_formula = rf,
           control = list(n_interior_knots = 5L)),
    error = function(e) {
      cat("    ML FIT ERROR (", re, "): ", conditionMessage(e), "\n", sep = "")
      NULL
    })
}

ML_RE <- "intercept_slope"
tt <- system.time(fml <- .ml_fit("intercept_slope", ~ time))
if (is.null(fml)) {
  cat("    retrying with random_effects = 'intercept'\n")
  ML_RE <- "intercept"
  tt <- system.time(fml <- .ml_fit("intercept", NULL))
}

if (!is.null(fml)) {
  ml <- data.frame(
    precision = PREC,
    random_effects = ML_RE,   # so the compare step cannot line up two
                              # different models and call it a precision effect
    loglik = if (is.numeric(fml$loglik)) fml$loglik else NA_real_,
    grad_max = pick(fml$convergence, "grad_max"),
    grad_norm = pick(fml$convergence, "grad_norm"),
    n_iter = pick(fml$convergence, "n_iter"),
    converged = isTRUE(fml$convergence$converged),
    time = as.numeric(tt[["elapsed"]]),
    stringsAsFactors = FALSE)
  ml_se <- as.list(fml$se)
  cat(sprintf("   loglik %12.4f  grad_max %10.3e  grad_norm %10.3e  iter %5.0f  converged %s  %6.1fs\n",
              ml$loglik, ml$grad_max, ml$grad_norm, ml$n_iter,
              ml$converged, ml$time))
}

saveRDS(list(precision = PREC, mcmc = mcmc, ml = ml, ml_se = ml_se,
             sizes = SIZES, seeds = SEEDS,
             session = utils::sessionInfo()),
        paste0(OUT, ".rds"))

cat("\nwrote ", OUT, ".rds\n", sep = "")
cat("finished ", format(Sys.time()), "\n", sep = "")
sink(); close(.con)
