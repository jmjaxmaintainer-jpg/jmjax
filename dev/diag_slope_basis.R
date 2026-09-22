# ==============================================================================
# Which b columns is orthogonalize_b ACTUALLY constraining?
#
# verify_orth_slope.R produced two findings that cannot both be right:
#   mean(b_i1)  A 5.971e-03   E 7.724e-03    - the slope constraint does NOT
#                                              bite (column 0 goes to 1e-19)
#   beta_1 ESS  A ~400        E ~5000-6300   - yet E does something D does not
#
# Either the basis built for the slope column is not the constant direction
# its design intends, or it is empty and the 13x improvement comes from
# somewhere else - in which case the mechanism written into mcmc_model.py is
# wrong even though the effect is real.
#
# This checks it three ways, cheaply, at 50 draws:
#   1. The backend now warns with the basis it built per column. Read it.
#   2. PER DRAW rather than averaged: max_draws |mean_i b_iq|. Averaging over
#      draws could in principle hide a per-draw constraint; this cannot.
#   3. The same for D, which is the positive control - column 0 must be
#      pinned and column 1 must not.
#
#   Rscript dev/diag_slope_basis.R
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

sim <- sim_joint(n = 200, seed = 7L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
lme_fit <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)
NS <- length(unique(dl$id))

bmat <- function(ps, j) t(vapply(ps[["b"]], function(d) {
  if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
  else { a <- as.matrix(d); as.numeric(a[, j]) }
}, numeric(NS)))

run <- function(mode) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = 50, num_samples = 50, num_chains = 1,
              progress_bar = FALSE, seed = 1L)
  if (mode == "D") ctl$orthogonalize_b0 <- TRUE
  if (mode == "E") ctl$orthogonalize_b  <- TRUE
  cat(sprintf("\n--- arm %s ---\n", mode))
  f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                     method = "spline-PH-mcmc", control = ctl)
  ps <- f$posterior_samples
  for (j in 1:2) {
    m <- rowMeans(bmat(ps, j))
    cat(sprintf("  b[,%d]  max over draws |mean_i b_iq| = %.3e   %s\n",
                j - 1L, max(abs(m)),
                if (max(abs(m)) < 1e-10) "<- PINNED" else "   free"))
  }
  invisible(NULL)
}
cat("Watch stderr for the backend's own report of the bases it built.\n")
for (mode in c("A", "D", "E")) run(mode)
cat("\nExpected if the design is doing what it claims:\n")
cat("  A  both free | D  b[,0] pinned, b[,1] free | E  BOTH pinned\n")
cat("If E leaves b[,1] free, the slope half is not being applied and the\n")
cat("13x on beta_1 has a different cause that is not yet identified.\n")
