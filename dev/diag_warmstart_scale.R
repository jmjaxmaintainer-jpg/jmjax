# ==============================================================================
# Is the warm-start rejection on simulated data a TIME-SCALE problem?
#
# Established by dev/diag_warmstart.R: both datasets seed the identical 14
# sites and leave the identical 2 deterministic ones uniform, so the seeding
# is not the difference. What differs is the data:
#
#   pbc2        event times to ~1.2 (years/12)   warm 1881 vs uniform 8746  ACCEPTED
#   simulated   event times to 10                warm 167467 vs 50204      REJECTED
#
# and the potentials differ by 20-80x on only 1.4x the observations, which is
# a scale signature rather than a volume one. The baseline-hazard spline
# basis, the Gauss-Kronrod quadrature nodes and the RW2 penalty all live on
# the time axis, and the cumulative hazard integrates any seeding error over
# an 8x longer interval.
#
# PREDICTION, recorded before the run: dividing time by about 8-10 brings the
# simulated data onto pbc2's scale and the warm start is ACCEPTED. If it is
# still rejected at every scale, the time axis is not the cause and the
# spline seeding is wrong for some other reason.
#
# This is a diagnosis, not the fix. If scale_time rescues it, the real fix is
# to make the SEEDING scale-aware - users should not have to know to set
# scale_time, and the option's own documentation says it is a targeted
# treatment rather than a default.
#
#   Rscript dev/diag_warmstart_scale.R 2>&1 | tee ~/Documents/R/jmjax_results/diag_ws_scale.log
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

sim <- sim_joint(n = 300, seed = 3001L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
cat(sprintf("simulated: event times to %.2f | %.1f obs/subject | %.0f%% events\n",
            max(ds$time), sim$obs_per_subject, 100 * sim$event_rate))
cat("pbc2 for reference: event times to ~1.2 after years/12\n\n")

lme_fit <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)

cat(sprintf("%-12s %12s %12s %12s %9s\n",
            "scale_time", "warm", "uniform", "warm-unif", "used"))
for (sc in c(NA, 2, 5, 8, 10, 20)) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = 100, num_samples = 100, num_chains = 1,
              progress_bar = FALSE, seed = 1L)
  if (!is.na(sc)) ctl$scale_time <- sc
  f <- suppressWarnings(tryCatch(
    jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                  method = "spline-PH-mcmc", control = ctl),
    error = function(e) { cat(sprintf("%-12s  ERROR: %s\n",
                                      if (is.na(sc)) "none" else sc,
                                      conditionMessage(e))); NULL }))
  if (is.null(f)) next
  ws <- f$convergence$warm_start
  if (is.null(ws)) { cat(sprintf("%-12s  (no warm start attempted)\n",
                                 if (is.na(sc)) "none" else sc)); next }
  pw <- as.numeric(ws$potential_warm); pu <- as.numeric(ws$potential_uniform)
  cat(sprintf("%-12s %12.1f %12.1f %+12.1f %9s\n",
              if (is.na(sc)) "none" else sc, pw, pu, pw - pu,
              if (isTRUE(as.logical(ws$used))) "YES" else "no"))
}
cat("\nRead: if 'used' flips to YES as scale_time rises, the time axis is the\n")
cat("cause and the seeding must be made scale-aware. If it never flips, the\n")
cat("spline seeding is wrong for a reason the time scale does not explain.\n")
