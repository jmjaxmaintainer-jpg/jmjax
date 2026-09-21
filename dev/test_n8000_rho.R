# ==============================================================================
# Does the n = 8,000 failure reproduce here, and does a genuine rho fix it?
#
# WHERE THIS COMES FROM. At n = 1,000, thirty fits across three generators
# showed:
#
#   A old (flat hazard, rho=0)  vs  B correct h(t), rho=0 : INDISTINGUISHABLE
#     max|cor| 0.881 vs 0.905, ESS alpha 1274 vs 1288, ESS rho 61 vs 61,
#     63 leapfrog steps in every fit of both arms.
#
#   B (rho=0)  vs  C (rho=0.3) : C CLEARLY BETTER ON rho
#     ESS rho 61 -> 129, ESS min 51 -> 103, R-hat rho 1.018 -> 1.007,
#     for 63 -> 89 steps and 25.0s -> 31.3s. That is 2.1x the effective
#     sample size for 1.4x the work, 1.69x better per second.
#
# So the misspecification was NOT the problem - that hypothesis is dead -
# but rho = 0 measurably is. And rho is already the worst-mixed parameter
# in the model by 20-50x at every size (ESS 27-75 against 950-1470 for
# alpha). Every R-hat-on-rho reading this project has taken, including the
# n = 8,000 failure, was made at the least favourable point in rho's
# parameter space, on the noisiest statistic in the fit.
#
# THE RECORDED n = 8,000 RESULTS, on the OLD machine and OLD stack:
#
#   sampler seed  1    677s   R-hat 1.012   127.0 steps
#   sampler seed 42    478s   R-hat 1.088    63.5 steps
#   sampler seed  7   1873s   R-hat 1.544   543.0 steps   <- the failure
#
# TWO REASONS NOT TO TRUST THOSE AS A BASELINE HERE. They were produced on
# a different Mac, and with rw2_implementation = "scan", which cannot run
# on the stack this package now pins (jax-ml/jax#22045). The vectorized
# branch that replaced it is algebraically identical - checked against the
# sequential recursion over 200 random cases, worst relative difference
# 4.9e-15 - but it does NOT consume the same random numbers. So seed 7
# here is not the same draw sequence as seed 7 there, and an exact
# reproduction of 1.544 is not expected or meaningful.
#
# That is why STAGE 1 runs THREE sampler seeds rather than betting on
# seed 7. The question is not "does this one number come back" but "does
# the pathology exist on this stack at all".
#
# PRE-REGISTERED CRITERION, written before running: a fit is PATHOLOGICAL
# if mean leapfrog steps > 150 or max R-hat >= 1.1. Both were the criteria
# used in the earlier sweeps.
#
# STAGE 2 ONLY RUNS IF STAGE 1 FINDS PATHOLOGY. If none of the three seeds
# misbehaves, the phenomenon does not reproduce on the current stack, there
# is nothing for a sampler default to fix, and spending another ninety
# minutes on arm C would be answering a question that has dissolved. Set
# FORCE_STAGE2 <- TRUE to run it regardless.
#
# WHY NO ARM B. A and B were indistinguishable on every measure at
# n = 1,000, including identical ESS rho and identical step counts. So a
# difference between A and C here is attributable to rho, not to the
# hazard. That is an inference from the smaller study rather than something
# this script establishes, and it is the reason B is omitted - each arm
# costs about ninety minutes at this size.
#
# RUNTIME: stage 0 about five minutes. Each n = 8,000 fit ran 8-31 minutes
# on the old machine. Stage 1 is three of them; stage 2, if it runs, three
# more. Results are written after EVERY fit, so a sleep or a crash loses at
# most the fit in flight.
#
# RUN IT FROM A TERMINAL, not as a background job:
#   caffeinate -i Rscript dev/test_n8000_rho.R 2>&1 | tee n8000.log
# caffeinate prevents the idle sleep that contaminated two wall-clock
# timings in the n = 1,000 run (675s and 1734s with entirely normal step
# counts and diagnostics).
# ==============================================================================

library(nlme); library(survival); library(jmjax)

`%||%` <- function(a, b) if (is.null(a)) b else a

N            <- 8000L
SAMP_SEEDS   <- c(7L, 1L, 42L)   # seed 7 first: it is the one that failed
DATA_SEED    <- 1L
FORCE_STAGE2 <- FALSE
STAMP        <- format(Sys.time(), "%Y%m%d_%H%M")
OUT          <- file.path(getwd(), sprintf("n8000_rho_%s", STAMP))

# Keep the console transcript even when this runs unattended. The n = 1,000
# study wrote its table to RDS but not its console output, and the run
# before that left its numbers nowhere at all - an evening was then spent
# reconstructing an n = 8,000 results table from comments in old scripts.
.con <- file(paste0(OUT, ".log"), open = "wt")
sink(.con, split = TRUE)

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

# ---- arm A: the generator every previous n = 8,000 run used, verbatim ----
sim_A <- function(n, seed) {
  set.seed(seed)
  b0 <- rnorm(n, 0, .8); b1 <- rnorm(n, 0, .2); x <- rnorm(n)
  eta <- -2 + .4*(2 + .3*x + b0)
  Tt <- rexp(n, rate = pmin(pmax(exp(eta), 1e-4), 5) / 4)
  .assemble(b0, b1, x, pmin(Tt, TMAX), Tt <= TMAX)
}

# ---- arm C: hazard depends on m_i(t), and rho is genuinely 0.3 ----------
# H_i(t) = lambda0 exp(alpha A_i) (exp(alpha D_i t) - 1)/(alpha D_i),
# inverted in closed form. Verified against integrate() in the n = 1,000
# script (agreement to 1e-8) and re-verified below before use.
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

one_fit <- function(d, sseed, warmup = 500L, samples = 500L, chains = 2L) {
  t <- system.time(f <- tryCatch(
    jm_fit(long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
           data_long = d$l, data_surv = d$s, id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(n_interior_knots = 5L, spline_prior = "penalized",
                          num_warmup = warmup, num_samples = samples,
                          num_chains = chains, seed = sseed,
                          progress_bar = FALSE)),
    error = function(e) { cat("    FIT ERROR:", conditionMessage(e), "\n"); NULL }))
  if (is.null(f)) return(NULL)
  rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
  es <- f$diagnostics$ess
  pick <- function(l, nm) { v <- suppressWarnings(as.numeric(l[[nm]]))
                            if (length(v) && is.finite(v[1])) v[1] else NA_real_ }
  list(time = as.numeric(t["elapsed"]),
       rhat_max = if (length(rh)) max(rh) else NA_real_,
       rhat_worst = if (length(rh)) names(rh)[which.max(rh)] else NA_character_,
       rhat_rho = pick(f$diagnostics$rhat, "rho"),
       steps = suppressWarnings(as.numeric(f$convergence$mean_num_steps)),
       div = { v <- f$convergence$n_divergences; if (is.null(v)) NA_real_ else as.numeric(v) },
       ess_alpha = pick(es, "alpha"), ess_rho = pick(es, "rho"),
       ess_min = { e <- unlist(es); e <- e[is.finite(e)]; if (length(e)) min(e) else NA_real_ },
       est_alpha = suppressWarnings(as.numeric(f$estimates[["alpha"]])),
       est_rho = pick(f$estimates, "rho"))
}

fm <- function(v, d = 3) if (is.finite(v)) formatC(v, format = "f", digits = d) else "?"
ROWS <- list()
record <- function(arm, sseed, r) {
  ROWS[[length(ROWS)+1]] <<- data.frame(
    arm = arm, n = N, data_seed = DATA_SEED, samp_seed = sseed,
    time = r$time, rhat_max = r$rhat_max, rhat_worst = r$rhat_worst %||% NA,
    rhat_rho = r$rhat_rho, steps = r$steps, divergences = r$div,
    ess_alpha = r$ess_alpha, ess_rho = r$ess_rho, ess_min = r$ess_min,
    est_alpha = r$est_alpha, est_rho = r$est_rho, stringsAsFactors = FALSE)
  res <- do.call(rbind, ROWS)
  saveRDS(res, paste0(OUT, ".rds")); utils::write.csv(res, paste0(OUT, ".csv"),
                                                      row.names = FALSE)
}

pathological <- function(r) {
  (is.finite(r$steps) && r$steps > 150) || (is.finite(r$rhat_max) && r$rhat_max >= 1.1)
}

hdr <- function() cat(sprintf("  %6s %9s %9s %12s %7s %6s %8s %8s\n",
                              "samp", "time", "R-hat", "worst", "steps", "div",
                              "ESS_a", "ESS_rho"))
show <- function(sseed, r) {
  cat(sprintf("  %6d %9.1f %9s %12s %7s %6s %8s %8s%s\n", sseed, r$time,
              fm(r$rhat_max), substr(r$rhat_worst %||% "?", 1, 12),
              fm(r$steps, 0), fm(r$div, 0), fm(r$ess_alpha, 0),
              fm(r$ess_rho, 0),
              if (pathological(r)) "   <- PATHOLOGICAL" else ""))
}

# ============================ STAGE 0: mechanics ============================
# Not a reduced-budget version of the experiment. The failure being chased is
# an ADAPTATION failure and adaptation happens during warm-up, so a short
# warm-up would change the very thing under test. This checks only that the
# generator and the pipeline work at this size before an hour is committed.
cat("################ STAGE 0 - mechanics ################\n\n")

cat("  re-verifying the closed-form inversion:\n")
set.seed(99); lam0 <- exp(-4.6); ok <- TRUE
for (i in 1:3) {
  A <- rnorm(1, 2, .8); D <- rnorm(1, .5, .2); t0 <- runif(1, .5, 9)
  num <- stats::integrate(function(s) lam0*exp(ALPHA*(A + D*s)), 0, t0)$value
  back <- .invert(num, A, D, lam0)
  cat(sprintf("    t %5.2f -> H %10.6f -> inverted %6.3f  %s\n", t0, num, back,
              if (abs(back - t0) < 1e-6) "ok" else "MISMATCH"))
  if (abs(back - t0) > 1e-6) ok <- FALSE
}
if (!ok) { sink(); stop("closed-form inversion failed - stopping") }

dA <- sim_A(N, DATA_SEED)
dC <- sim_C(N, DATA_SEED)
cat(sprintf("\n  arm A : rows %7d | events %4.1f%% | empirical rho %+.3f\n",
            nrow(dA$l), 100*dA$ev, dA$rho_emp))
cat(sprintf("  arm C : rows %7d | events %4.1f%% | empirical rho %+.3f | log lambda0 %.3f\n",
            nrow(dC$l), 100*dC$ev, dC$rho_emp, dC$log_lam0))
if (abs(dC$rho_emp - 0.3) > 0.05) cat("  WARNING: arm C's empirical rho is not near 0.3\n")
if (abs(dC$ev - dA$ev) > 0.05) cat("  WARNING: event rates differ by more than 5 points\n")

cat("\n  one tiny fit on arm C, purely to confirm it compiles and executes\n")
cat("  at this data shape. Its DIAGNOSTICS ARE MEANINGLESS - 50 warm-up\n")
cat("  draws cannot adapt anything - only 'did it return' matters.\n")
smoke <- one_fit(dC, 1L, warmup = 50L, samples = 50L, chains = 1L)
if (is.null(smoke)) {
  cat("\n  STAGE 0 FAILED - the fit errored. Nothing below would be meaningful.\n")
  sink(); close(.con); stop("smoke test failed")
}
cat(sprintf("  ran in %.1fs. Pipeline is sound; proceeding.\n\n", smoke$time))

# ======================= STAGE 1: does it reproduce? ========================
cat("\n################ STAGE 1 - arm A, the recorded configuration ################\n")
cat("  Old machine, old stack, for reference only (different RNG stream):\n")
cat("    seed  1   677s  R-hat 1.012  127 steps\n")
cat("    seed 42   478s  R-hat 1.088   64 steps\n")
cat("    seed  7  1873s  R-hat 1.544  543 steps  <- the failure\n\n")
hdr()
any_path <- FALSE
for (ss in SAMP_SEEDS) {
  r <- one_fit(dA, ss)
  if (is.null(r)) next
  show(ss, r); record("A", ss, r)
  if (pathological(r)) any_path <- TRUE
}

if (!any_path && !FORCE_STAGE2) {
  cat("\n\n================ VERDICT ================\n")
  cat("  NONE of the three seeds is pathological on the current stack.\n")
  cat("  The n = 8,000 failure does not reproduce here.\n\n")
  cat("  That is the answer, and it dissolves the question rather than\n")
  cat("  answering it: there is no failure for target_accept_prob = 0.9 or\n")
  cat("  any other default to repair. The recorded 543 steps and R-hat\n")
  cat("  1.544 belong to a machine and a Python stack this package can no\n")
  cat("  longer produce, and should be reported that way rather than as a\n")
  cat("  property of the sampler.\n\n")
  cat("  Stage 2 skipped - with nothing to fix, arm C would be answering a\n")
  cat("  question that no longer exists. Set FORCE_STAGE2 <- TRUE to run it\n")
  cat("  anyway, e.g. to measure rho's ESS gain at this size for its own\n")
  cat("  sake.\n")
  cat(sprintf("\n  results: %s.csv / .rds / .log\n", OUT))
  sink(); close(.con); quit(save = "no")
}

# ===================== STAGE 2: does a genuine rho help? ====================
cat("\n\n################ STAGE 2 - arm C, rho = 0.3 ################\n")
cat("  Same subjects, same sampler seeds, same budget. The ONLY changes are\n")
cat("  a hazard that depends on m_i(t) and a genuine rho of 0.3.\n")
cat("  At n = 1,000 this doubled rho's ESS (61 -> 129) and improved its\n")
cat("  R-hat (1.018 -> 1.007) for 1.4x the leapfrog work.\n\n")
hdr()
for (ss in SAMP_SEEDS) {
  r <- one_fit(dC, ss)
  if (is.null(r)) next
  show(ss, r); record("C", ss, r)
}

# ================================ summary ==================================
res <- do.call(rbind, ROWS)
cat("\n\n================ SUMMARY ================\n")
for (a in unique(res$arm)) {
  s <- res[res$arm == a, ]
  cat(sprintf("  arm %s : median steps %5.0f | median max R-hat %6.3f | median ESS rho %6.0f | median s %7.1f | %d/%d pathological\n",
              a, stats::median(s$steps, na.rm = TRUE),
              stats::median(s$rhat_max, na.rm = TRUE),
              stats::median(s$ess_rho, na.rm = TRUE),
              stats::median(s$time, na.rm = TRUE),
              sum(s$steps > 150 | s$rhat_max >= 1.1, na.rm = TRUE), nrow(s)))
}
cat("\n  truth recovery (arm C only - arm A's hazard has no time-varying\n")
cat("  term, so alpha there is a compromise with no true value):\n")
cc <- res[res$arm == "C", ]
if (nrow(cc)) cat(sprintf("    est_alpha median %.4f (true 0.40) | est_rho median %.4f (true 0.30)\n",
                          stats::median(cc$est_alpha, na.rm = TRUE),
                          stats::median(cc$est_rho, na.rm = TRUE)))

cat("\n================ HOW TO READ IT ================\n")
cat("  A PATHOLOGICAL, C CLEAN : the n = 8,000 failure is largely the\n")
cat("    simulation parking rho - already the worst-mixed parameter in the\n")
cat("    model - at its least favourable value. No sampler default needs\n")
cat("    changing; the simulation design does.\n")
cat("  BOTH PATHOLOGICAL : rho's true value is not the driver either. The\n")
cat("    failure is a genuine large-n sampler limitation and\n")
cat("    target_accept_prob = 0.9 has finally earned its expensive run.\n")
cat("  A CLEAN, C PATHOLOGICAL : unexpected, and worth stopping to\n")
cat("    understand before drawing anything from it.\n")
cat(sprintf("\n  results: %s.csv / .rds / .log\n", OUT))
sink(); close(.con)
