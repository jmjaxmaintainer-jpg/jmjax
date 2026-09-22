# ==============================================================================
# diag_slope_basis.R (50 draws, n=200, 1 chain) said E pins BOTH columns to
# ~1e-17. verify_orth_slope.R (1000+1000 draws, 4 chains, n=500) said E does
# NOT pin column 1 (~0.02-0.04, not near zero). Both cannot be right for the
# same deterministic post-hoc projection - it is a per-draw subtraction that
# should not care how many draws or chains were run.
#
# This holds mode=E fixed and varies ONE factor at a time against the
# diag_slope_basis.R baseline, to localize which one flips the verdict:
# n (200 vs 500), num_chains (1 vs 4), draws (50+50 vs 1000+1000). All runs
# use k=3 (age+sex) except the isolated single-column-Z case is not needed
# here - verify's k=1 failure already shows covariates are not the cause.
#
#   Rscript dev/diag_slope_basis2.R
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

bmat <- function(ps, j, NS) t(vapply(ps[["b"]], function(d) {
  if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
  else { a <- as.matrix(d); as.numeric(a[, j]) }
}, numeric(NS)))

run_one <- function(label, n, chains, warmup, samples, seed_data = 7L, seed_sampler = 1L) {
  sim <- sim_joint(n = n, seed = seed_data, k_extra = 2L)
  dl <- sim$data_long; ds <- sim$data_surv
  lme_fit <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)
  NS <- length(unique(dl$id))

  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = warmup, num_samples = samples, num_chains = chains,
              progress_bar = FALSE, seed = seed_sampler, orthogonalize_b = TRUE)
  f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                     method = "spline-PH-mcmc", control = ctl)
  ps <- f$posterior_samples
  m0 <- max(abs(rowMeans(bmat(ps, 1, NS))))
  m1 <- max(abs(rowMeans(bmat(ps, 2, NS))))
  cat(sprintf("  %-38s n=%4d chains=%d draws=%d+%d  b0 %.3e  b1 %.3e  %s\n",
              label, n, chains, warmup, samples, m0, m1,
              if (m1 < 1e-10) "b1 PINNED" else "b1 FREE"))
  invisible(list(m0 = m0, m1 = m1))
}

cat("Isolating which factor flips E's slope-column pinning.\n")
cat("Baseline (diag_slope_basis.R's own config) first, then one change at a time.\n\n")

run_one("baseline (diag_slope_basis.R)",        n = 200, chains = 1, warmup = 50,   samples = 50)
run_one("+ num_chains 1->4",                     n = 200, chains = 4, warmup = 50,   samples = 50)
run_one("+ n 200->500",                          n = 500, chains = 1, warmup = 50,   samples = 50)
run_one("+ draws 50+50 -> 1000+1000",            n = 200, chains = 1, warmup = 1000, samples = 1000)
run_one("n=500, chains=4, draws=1000+1000 (verify's exact config)",
                                                  n = 500, chains = 4, warmup = 1000, samples = 1000)

cat("\nIf only the last row shows FREE, it takes the combination to trigger it -\n")
cat("check whichever single-factor row ALSO shows FREE for the actual cause.\n")
