# ==============================================================================
# Why did the warm start start failing under float64?
#
# Four tests failed after float64 became the default. This script does not
# assume a cause - it distinguishes between the two explanations that fit
# the evidence, and they call for opposite responses.
#
# HYPOTHESIS A - THE TEST WAS ALWAYS FRAGILE, float32 WAS A LUCKY DRAW.
#   The warm start supplies beta, sigma_e, sigma_b, L_corr and b_std. It
#   does NOT supply the spline block, which under spline_prior =
#   "penalized" is W01, z_step and tau_w. Those fall back to
#   init_to_uniform. jax.random draws with the SAME key produce DIFFERENT
#   numbers at different dtypes, so switching to float64 re-rolled both the
#   warm start's unsupplied block and the three uniform starts it is
#   compared against. If acceptance flips from seed to seed, then the test
#   asserts something the design never guaranteed, and the fix is to the
#   test (or to supplying the spline block), not to the precision.
#
# HYPOTHESIS B - float64 EXPOSED A REAL DEFECT.
#   Something in the warm-start transform is wrong in a way float32's
#   coarseness was hiding. If acceptance is stable across seeds and the
#   potential is consistently ~1e9, that is not luck - it is a bug, and
#   the float64 default should not ship until it is understood.
#
# DISTINGUISHING TEST: run the same penalized fit across several seeds and
# look at whether `used` flips. A is seed-dependent; B is not.
#
# It also asks a second, independent question about the scale-time failure:
# is it downstream of the warm start? If scale_time = off accepts the warm
# start and scale_time = 5 rejects it, the two fits began from different
# points and the 3.7% gap in alpha is a confound rather than a scaling bug.
# That is the same confound that made test-prefit-interface.R need
# mcmc_warm_start = FALSE.
#
#   Rscript dev/diagnose_warm_x64.R              # float64 (the default)
#   JMJAX_ENABLE_X64=0 Rscript dev/diagnose_warm_x64.R   # float32, to compare
#
# Takes a few minutes. Small budgets throughout - this measures WHICH START
# WAS CHOSEN, not fit quality.
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(survival) })

PREC <- jmjax:::.backend_precision()
cat("\n=========================================================\n")
cat("  precision in force: ", PREC, "\n", sep = "")
cat("=========================================================\n")

fm <- function(v, d = 1) if (is.finite(v)) formatC(v, format = "f", digits = d) else "?"
num <- function(x) { v <- suppressWarnings(as.numeric(x)); if (length(v)) v[1] else NA_real_ }

# helper-simulate.R is not installed with the package, so source it from the
# test directory - these are the exact generators the failing tests use, and
# a different generator would not be reproducing the failure.
HELP <- file.path(path.expand("~/Documents/R/jmjax"),
                  "tests", "testthat", "helper-simulate.R")
if (!file.exists(HELP)) stop("cannot find ", HELP)
source(HELP)

# ------------------------------------------------------------------------------
# PART 1. Is penalized warm-start acceptance seed-dependent?
# ------------------------------------------------------------------------------
cat("\n-- PART 1: penalized warm start across seeds",
    "(A predicts `used` flips; B predicts it never does)\n\n")
cat(sprintf("  %6s %6s %16s %16s  %s\n",
            "seed", "used", "potential_warm", "potential_unif", "sites"))

sim <- simulate_joint_data_re2(n = 150, seed = 3)   # same data as the test
for (s in c(1L, 2L, 3L, 11L, 42L)) {
  f <- tryCatch(
    jm_fit(long_formula = y ~ time,
           surv_formula = Surv(time, event) ~ 1,
           data_long = sim$data_long, data_surv = sim$data_surv,
           id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(spline_prior = "penalized", mcmc_warm_start = TRUE,
                          num_warmup = 100, num_samples = 100, num_chains = 1,
                          seed = s, progress_bar = FALSE)),
    error = function(e) { cat("   seed", s, "ERROR:", conditionMessage(e), "\n"); NULL })
  if (is.null(f)) next
  ws <- f$convergence$warm_start
  cat(sprintf("  %6d %6s %16s %16s  %s\n", s,
              if (isTRUE(as.logical(ws$used))) "TRUE" else "FALSE",
              fm(num(ws$potential_warm)), fm(num(ws$potential_uniform)),
              paste(unlist(ws$sites), collapse = ",")))
}

# ------------------------------------------------------------------------------
# PART 2. Same question WITHOUT the penalized prior.
# ------------------------------------------------------------------------------
# If the independent prior accepts where penalized rejects, the unsupplied
# spline block is implicated directly: that is the only part of the model
# the two configurations disagree about.
cat("\n-- PART 2: same fits, spline_prior = 'independent'\n\n")
cat(sprintf("  %6s %6s %16s %16s\n", "seed", "used", "potential_warm", "potential_unif"))
for (s in c(1L, 2L, 3L, 11L, 42L)) {
  f <- tryCatch(
    jm_fit(long_formula = y ~ time,
           surv_formula = Surv(time, event) ~ 1,
           data_long = sim$data_long, data_surv = sim$data_surv,
           id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(spline_prior = "independent", mcmc_warm_start = TRUE,
                          num_warmup = 100, num_samples = 100, num_chains = 1,
                          seed = s, progress_bar = FALSE)),
    error = function(e) NULL)
  if (is.null(f)) next
  ws <- f$convergence$warm_start
  cat(sprintf("  %6d %6s %16s %16s\n", s,
              if (isTRUE(as.logical(ws$used))) "TRUE" else "FALSE",
              fm(num(ws$potential_warm)), fm(num(ws$potential_uniform))))
}

# ------------------------------------------------------------------------------
# PART 3. Is the scale-time failure downstream of the warm start?
# ------------------------------------------------------------------------------
cat("\n-- PART 3: scale_time, with the warm start ON then OFF\n")
cat("   If the two arms disagree about `used` with the warm start ON, and\n")
cat("   alpha agrees once it is OFF, the scale-time failure is a confound.\n\n")

set.seed(7)
n <- 60; visit <- c(0, 1, 2, 3)
b0 <- rnorm(n, 0, 0.8); b1 <- rnorm(n, 0, 0.2); xx <- rnorm(n)
dl <- do.call(rbind, lapply(seq_len(n), function(i)
  data.frame(id = i, time = visit, x = xx[i],
             y = 2 + b0[i] + 0.3 * xx[i] + (0.5 + b1[i]) * visit +
                 rnorm(length(visit), 0, 0.2))))
ds <- data.frame(id = seq_len(n), time = runif(n, 1.5, 3),
                 event = rbinom(n, 1, 0.6))
dl$id <- factor(dl$id); ds$id <- factor(ds$id)

run <- function(ctl) tryCatch(
  jm_fit(long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
         data_long = dl, data_surv = ds, id_var = "id", time_var = "time",
         method = "spline-PH-mcmc", random_effects = "intercept_slope",
         random_formula = ~ time,
         control = c(list(n_interior_knots = 5L, num_warmup = 250L,
                          num_samples = 250L, num_chains = 2L,
                          progress_bar = FALSE, seed = 11L), ctl)),
  error = function(e) { cat("   ERROR:", conditionMessage(e), "\n"); NULL })

for (ws_on in c(TRUE, FALSE)) {
  f0 <- run(list(mcmc_warm_start = ws_on))
  f1 <- run(list(mcmc_warm_start = ws_on, scale_time = 5))
  if (is.null(f0) || is.null(f1)) next
  u0 <- isTRUE(as.logical(f0$convergence$warm_start$used))
  u1 <- isTRUE(as.logical(f1$convergence$warm_start$used))
  a0 <- num(f0$estimates[["alpha"]]); a1 <- num(f1$estimates[["alpha"]])
  cat(sprintf("   warm_start=%-5s  used: off=%-5s on=%-5s | alpha %8.5f vs %8.5f  ratio %.4f%s\n",
              ws_on, if (ws_on) u0 else NA, if (ws_on) u1 else NA,
              a0, a1, a1 / a0,
              if (abs(a1/a0 - 1) < 0.02) "  <- within test tolerance" else "  <- FAILS test"))
}

cat("\nHOW TO READ THIS\n")
cat("  Part 1 `used` flips across seeds        -> hypothesis A, fix the test\n")
cat("  Part 1 `used` FALSE at every seed, ~1e9 -> hypothesis B, real defect\n")
cat("  Part 2 accepts where Part 1 rejects     -> the unsupplied spline block\n")
cat("  Part 3 agrees only with the start OFF   -> scale-time is a confound\n\n")
