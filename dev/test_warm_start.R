# ==============================================================================
# Does warm-starting NUTS from the lme() pre-fit help?
#
# THE DIAGNOSIS IT ADDRESSES. At n = 8,000 with q = 2 the posterior has
# ~16,000 random-effect dimensions. init_to_uniform draws every one of them
# uniformly on [-2, 2] in unconstrained space, so warm-up must travel to the
# typical set AND adapt a step size within 500 iterations. Under float64,
# two of three sampler seeds on identical data still failed - always on a
# variance or covariance component, never on beta or alpha.
#
# Radius is not the problem: U(-2,2) has variance 4/3, so a random point
# sits at radius ~146 against a typical-set shell at sqrt(16000) ~ 126. It
# is that every COORDINATE is wrong. The lme() BLUPs put each one
# approximately right, which is what JMbayes2 has always done (R/jm.R
# initialises betas, sigmas, D, the per-subject b and gammas from the
# lme/coxph objects it is handed; R/jm_fit.R jitters per chain).
#
# WHAT IS ALREADY VERIFIED, so this script does not re-litigate it:
#   - the Cholesky transform b_std = L^-1 b, with L = diag(sigma_b) L_corr:
#     500 random SPD matrices, q in [2,4], max relative error 3.8e-16 for
#     L L' = D and 3.1e-16 for the b_std round trip, with L_corr's rows unit
#     norm as numpyro's CorrCholesky support requires.
#   - jm_fit() re-checks both on the actual fit and abandons the warm start
#     rather than passing on a bad transform.
#   - the backend then compares the potential energy at the warm start
#     against a uniform start and REJECTS the warm start if it is not
#     better. So a scaling mismatch cannot silently degrade a fit; it
#     produces a warning and the default behaviour.
#
# WHAT THIS SCRIPT ACTUALLY TESTS, in order of cost:
#
#   STAGE 1  n = 300, one fit. Is the plumbing right? Reads
#            fit$convergence$warm_start to confirm the start was BUILT,
#            ACCEPTED by the self-check, and how much better its log-density
#            was. A wiring bug shows up here in a minute rather than an hour.
#
#   STAGE 2  n = 1,000, 5 seeds, warm vs cold, paired. This is the regime
#            every real joint-model dataset lives in (pbc2 312, aids 467,
#            prothro 488, heart.valve 256, epileptic 605), and the question
#            here is DO NO HARM: cold fits at this size are already healthy
#            at 63 leapfrog steps, so a warm start that degrades anything is
#            disqualified regardless of what it does at scale.
#
#   STAGE 3  n = 8,000, seeds 7 and 42, warm vs cold. The two that failed
#            under float64. This is where a warm start should earn its keep,
#            and where it is worth the ~20 minutes.
#
# HOW TO READ IT. R-hat and ESS are the outcome, but interpret them knowing
# that a warm start makes R-hat LESS sensitive - chains that start together
# agree more readily, which is a real cost, not a free win. That is why the
# step count and the divergence count matter here: those are within-chain
# measures that a shared starting point does not flatter.
#
# RUN IT:
#   JAX_ENABLE_X64=1 caffeinate -i Rscript dev/test_warm_start.R 2>&1 | tee warm.log
# x64 because float32 independently cost 4.2x speed and ESS_alpha 4 vs 1188
# at n = 8,000; leaving it off would confound two effects.
# ==============================================================================

library(nlme); library(survival); library(jmjax)

`%||%` <- function(a, b) if (is.null(a)) b else a

RUN_STAGE3 <- TRUE          # set FALSE to stop after n = 1,000
STAMP <- format(Sys.time(), "%Y%m%d_%H%M")
OUT   <- file.path(getwd(), sprintf("warm_start_%s", STAMP))
.con  <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)

TMAX <- 10; VISIT <- seq(0, TMAX, length.out = 8L)

sim <- function(n, seed) {
  set.seed(seed)
  b0 <- rnorm(n, 0, .8); b1 <- rnorm(n, 0, .2); x <- rnorm(n)
  eta <- -2 + .4 * (2 + .3 * x + b0)
  Tt <- rexp(n, rate = pmin(pmax(exp(eta), 1e-4), 5) / 4)
  ot <- pmin(Tt, TMAX)
  reps <- vapply(ot, function(o) max(1L, sum(VISIT <= o)), integer(1))
  dl <- data.frame(id = factor(rep(seq_len(n), reps), levels = seq_len(n)),
                   time = unlist(lapply(reps, function(k) VISIT[seq_len(k)])),
                   x = rep(x, reps))
  dl$y <- (2 + .3 * dl$x + rep(b0, reps)) + (.5 + rep(b1, reps)) * dl$time +
          rnorm(nrow(dl), 0, .3)
  list(l = dl,
       s = data.frame(id = factor(seq_len(n)), time = ot,
                      event = as.integer(Tt <= TMAX), x = x),
       ev = mean(Tt <= TMAX))
}

one <- function(d, sseed, warm, warmup = 500L, samples = 500L, chains = 2L) {
  t <- system.time(f <- tryCatch(
    jm_fit(long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
           data_long = d$l, data_surv = d$s, id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(n_interior_knots = 5L, spline_prior = "penalized",
                          num_warmup = warmup, num_samples = samples,
                          num_chains = chains, seed = sseed,
                          mcmc_warm_start = warm, progress_bar = FALSE)),
    error = function(e) { cat("      FIT ERROR:", conditionMessage(e), "\n"); NULL }))
  if (is.null(f)) return(NULL)
  rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
  es <- f$diagnostics$ess
  pick <- function(l, nm) { v <- suppressWarnings(as.numeric(l[[nm]]))
                            if (length(v) && is.finite(v[1])) v[1] else NA_real_ }
  ws <- f$convergence$warm_start
  list(time = as.numeric(t["elapsed"]),
       rhat = if (length(rh)) max(rh) else NA_real_,
       worst = if (length(rh)) names(rh)[which.max(rh)] else NA_character_,
       steps = suppressWarnings(as.numeric(f$convergence$mean_num_steps)),
       div = { v <- f$convergence$n_divergences; if (is.null(v)) NA_real_ else as.numeric(v) },
       ess_a = pick(es, "alpha"), ess_rho = pick(es, "rho"),
       ess_min = { e <- unlist(es); e <- e[is.finite(e)]; if (length(e)) min(e) else NA_real_ },
       est_alpha = suppressWarnings(as.numeric(f$estimates[["alpha"]])),
       ws_used = if (is.null(ws)) NA else isTRUE(as.logical(ws$used)),
       pe_warm = if (is.null(ws)) NA_real_ else suppressWarnings(as.numeric(ws$potential_warm %||% NA)),
       pe_unif = if (is.null(ws)) NA_real_ else suppressWarnings(as.numeric(ws$potential_uniform %||% NA)),
       ws_sites = if (is.null(ws)) NA_character_ else paste(unlist(ws$sites), collapse = ","))
}

fm <- function(v, d = 3) if (is.finite(v)) formatC(v, format = "f", digits = d) else "?"
ROWS <- list()
rec <- function(stage, n, sseed, warm, r) {
  ROWS[[length(ROWS) + 1]] <<- data.frame(
    stage = stage, n = n, samp_seed = sseed, warm = warm,
    time = r$time, rhat = r$rhat, worst = r$worst %||% NA, steps = r$steps,
    divergences = r$div, ess_alpha = r$ess_a, ess_rho = r$ess_rho,
    ess_min = r$ess_min, est_alpha = r$est_alpha, ws_used = r$ws_used,
    pe_warm = r$pe_warm, pe_unif = r$pe_unif, stringsAsFactors = FALSE)
  res <- do.call(rbind, ROWS)
  saveRDS(res, paste0(OUT, ".rds"))
  utils::write.csv(res, paste0(OUT, ".csv"), row.names = FALSE)
}

# ========================= STAGE 1: is it wired up? =========================
cat("################ STAGE 1 - plumbing, n = 300 ################\n\n")
d3 <- sim(300L, 1L)
cat(sprintf("  n = 300 | rows = %d | events %.0f%%\n\n", nrow(d3$l), 100 * d3$ev))
r1 <- one(d3, 1L, warm = TRUE, warmup = 200L, samples = 200L, chains = 2L)
if (is.null(r1)) { cat("\n  STAGE 1 FAILED - the fit errored.\n"); sink(); quit(save = "no") }
rec("plumbing", 300L, 1L, TRUE, r1)

cat(sprintf("  warm start BUILT and reported : %s\n",
            if (is.na(r1$ws_used)) "NO - convergence$warm_start is NULL" else "yes"))
if (is.na(r1$ws_used)) {
  cat("\n  The R block did not attach control$init_values, or the backend did\n")
  cat("  not see it. Nothing below would mean anything - stopping.\n")
  sink(); close(.con); quit(save = "no")
}
cat(sprintf("  sites supplied                : %s\n", r1$ws_sites))
cat(sprintf("  self-check ACCEPTED it        : %s\n", r1$ws_used))
cat(sprintf("  potential at warm start       : %.1f\n", r1$pe_warm))
cat(sprintf("  potential at uniform start    : %.1f   (lower is better)\n", r1$pe_unif))
if (isTRUE(r1$ws_used)) {
  cat(sprintf("  improvement                   : %.1f log-density units\n",
              r1$pe_unif - r1$pe_warm))
} else {
  cat("\n  The self-check REJECTED the warm start, so the fit ran from the\n")
  cat("  default. That means the supplied values are on a different scale\n")
  cat("  than the model expects - check standardize_covariates and\n")
  cat("  scale_time. Stages 2 and 3 would compare cold against cold.\n")
  sink(); close(.con); quit(save = "no")
}
cat(sprintf("\n  fit ran in %.1fs, R-hat %.3f, %s steps\n\n",
            r1$time, r1$rhat, fm(r1$steps, 0)))

# ===================== STAGE 2: do no harm at real sizes ====================
hdr <- function() cat(sprintf("  %6s %5s %8s %8s %12s %7s %6s %8s %8s\n",
                              "seed", "start", "time", "R-hat", "worst",
                              "steps", "div", "ESS_a", "ESS_rho"))
show <- function(sseed, lbl, r) {
  cat(sprintf("  %6d %5s %8.1f %8s %12s %7s %6s %8s %8s\n", sseed, lbl, r$time,
              fm(r$rhat), substr(r$worst %||% "?", 1, 12), fm(r$steps, 0),
              fm(r$div, 0), fm(r$ess_a, 0), fm(r$ess_rho, 0)))
}

cat("\n################ STAGE 2 - n = 1,000, warm vs cold ################\n")
cat("  Real datasets live here (250-600 subjects), and cold fits at this\n")
cat("  size are already healthy. The bar is DO NO HARM.\n\n")
d1 <- sim(1000L, 1L)
cat(sprintf("  n = 1000 | rows = %d | events %.0f%%\n\n", nrow(d1$l), 100 * d1$ev))
hdr()
for (ss in c(1L, 7L, 13L, 42L, 99L)) {
  rc <- one(d1, ss, warm = FALSE); if (!is.null(rc)) { show(ss, "cold", rc); rec("n1000", 1000L, ss, FALSE, rc) }
  rw <- one(d1, ss, warm = TRUE);  if (!is.null(rw)) { show(ss, "warm", rw); rec("n1000", 1000L, ss, TRUE,  rw) }
}

# ==================== STAGE 3: where it should earn its keep ================
if (RUN_STAGE3) {
  cat("\n\n################ STAGE 3 - n = 8,000, warm vs cold ################\n")
  cat("  Seeds 7 and 42 both failed here under float64 from a cold start.\n\n")
  d8 <- sim(8000L, 1L)
  cat(sprintf("  n = 8000 | rows = %d | events %.0f%%\n\n", nrow(d8$l), 100 * d8$ev))
  hdr()
  for (ss in c(7L, 42L)) {
    rc <- one(d8, ss, warm = FALSE); if (!is.null(rc)) { show(ss, "cold", rc); rec("n8000", 8000L, ss, FALSE, rc) }
    rw <- one(d8, ss, warm = TRUE);  if (!is.null(rw)) { show(ss, "warm", rw); rec("n8000", 8000L, ss, TRUE,  rw) }
  }
}

# ================================ summary ==================================
res <- do.call(rbind, ROWS)
cat("\n\n================ SUMMARY ================\n")
bad <- function(s) sum(s$steps > 150 | s$rhat >= 1.1, na.rm = TRUE)
for (st in setdiff(unique(res$stage), "plumbing")) {
  for (w in c(FALSE, TRUE)) {
    s <- res[res$stage == st & res$warm == w, ]
    if (!nrow(s)) next
    cat(sprintf("  %-7s %-5s : median steps %5.0f | median R-hat %7.3f | median ESS rho %6.0f | median div %5.0f | median s %7.1f | %d/%d bad\n",
                st, if (w) "warm" else "cold",
                stats::median(s$steps, na.rm = TRUE), stats::median(s$rhat, na.rm = TRUE),
                stats::median(s$ess_rho, na.rm = TRUE), stats::median(s$divergences, na.rm = TRUE),
                stats::median(s$time, na.rm = TRUE), bad(s), nrow(s)))
  }
}

cat("\n================ HOW TO READ IT ================\n")
cat("  MERGE IT if n = 1,000 is no worse on any measure AND n = 8,000\n")
cat("    improves - fewer bad fits, lower step counts, fewer divergences.\n")
cat("  DO NOT MERGE if n = 1,000 degrades at all. Every real dataset is\n")
cat("    that size or smaller, and a large-n gain does not pay for a\n")
cat("    small-n loss in the regime users actually occupy.\n")
cat("  IF n = 8,000 IS UNCHANGED, the starting point was not the binding\n")
cat("    constraint and the honest conclusion is that warm-starting is a\n")
cat("    parity feature with JMbayes2 rather than a fix.\n\n")
cat("  Weight the WITHIN-chain measures - steps and divergences - above\n")
cat("  R-hat here. A warm start makes chains agree more readily whether or\n")
cat("  not they have converged, so an R-hat improvement is partly an\n")
cat("  artefact of the change being tested.\n")
cat(sprintf("\n  results: %s.csv / .rds / .log\n", OUT))
sink(); close(.con)
