# ==============================================================================
# What IS a good potential on this data? The reference measurement.
#
# Everything so far compares the warm start against a uniform start and finds
# it worse (167,467 vs 50,204). Neither number has ever been anchored: if the
# posterior itself sits at ~167,000, the warm start is FINE and the uniform
# start's 50,204 is the anomaly - a draw landing somewhere flat scores well
# without being anywhere useful, and the check takes min() over three such
# draws, which biases that baseline downward.
#
# Established so far, so this is the remaining gap and not a fishing trip:
#   - the seeded values match the generating truth to 2-3 decimals, and
#     reconstructing the linear predictor from them gives residuals of 0.37
#     against lme's 0.27;
#   - the longitudinal likelihood accounts for only ~14k of the 167k;
#   - the survival block is irrelevant (alpha = 0 plus a flat baseline moves
#     the potential by under 0.5%);
#   - ~153k is unaccounted for under every seeding variant tried.
#
# THE TRICK. No new plumbing: a fit is run to convergence, its posterior MEANS
# are fed back as control$init_values, and the self-check reports the
# potential at that point. User-supplied init_values are preserved rather than
# overwritten by the auto warm start, so what comes back is the potential at
# a point we KNOW is good.
#
#   potential(posterior mean) ~= 167k  -> the warm start was never bad; the
#       ACCEPTANCE TEST is wrong, and on long follow-up it has been rejecting
#       good starts all along. The fix is the test, not the seeding.
#   potential(posterior mean) ~= 2k    -> 167k really is a bad point, the
#       seeding is wrong somewhere the value checks do not reach, and the
#       search continues with a known target.
#
# standardize_covariates = FALSE throughout, so sampled space and reported
# space coincide and no back-transformation can contaminate the comparison.
#
#   Rscript dev/diag_reference_potential.R
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
suppressPackageStartupMessages(library(JMbayes2))
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")
`%||%` <- function(a, b) if (is.null(a)) b else a

CTL <- function(extra = list()) utils::modifyList(
  list(n_interior_knots = 5, spline_prior = "penalized",
       rw2_implementation = "vectorized", standardize_covariates = FALSE,
       num_warmup = 400, num_samples = 400, num_chains = 2,
       progress_bar = FALSE, seed = 1L), extra)

# Posterior mean of a site. vapply over a list of SCALARS returns a plain
# vector, not a 1-row matrix, so the obvious t()+colMeans collapses the wrong
# dimension and hands back all 800 draws - which is what made the first run
# die with "cannot reshape array of shape (800,) into shape (1,)".
pm <- function(ps, nm) {
  x <- ps[[nm]]
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    p <- length(unlist(x[[1]]))
    m <- vapply(x, function(z) as.numeric(unlist(z)), numeric(p))  # p x ndraws
    return(if (p == 1L) mean(m) else rowMeans(m))
  }
  a <- as.array(x)
  if (length(dim(a)) <= 1L) mean(a) else apply(a, seq_along(dim(a))[-1], mean)
}

# b_std needs its own extraction: it is [N_sub, q] per draw, and whether the
# per-draw element arrives as a matrix or as a list of per-subject vectors
# decides the flattening order. Pulling each column explicitly avoids
# guessing, the same way the other diagnostics in dev/ do it.
pm_bstd <- function(ps, nsub, q = 2L) {
  b <- ps[["b_std"]]
  if (is.null(b)) return(NULL)
  cols <- lapply(seq_len(q), function(j) {
    m <- vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
      else { a <- as.matrix(d); as.numeric(a[, j]) }
    }, numeric(nsub))
    rowMeans(m)
  })
  matrix(unlist(cols), ncol = q)
}

run_pair <- function(label, lme_fit, cox_fit, ds, tvar) {
  cat(sprintf("\n================ %s ================\n", label))
  f1 <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = tvar,
                      method = "spline-PH-mcmc", control = CTL())
  ps <- f1$posterior_samples
  cat(sprintf("  reference fit: max R-hat %.4f\n",
              max(unlist(f1$diagnostics$rhat), na.rm = TRUE)))

  # Posterior means of the SAMPLED sites. L_corr is rebuilt from mean rho
  # rather than averaged elementwise: the mean of Cholesky factors need not
  # be a valid Cholesky factor, and an invalid one would be a different bug
  # confounding this measurement.
  nsub <- length(unique(ds[[1]]))
  rho <- as.numeric(pm(ps, "rho"))[1]
  iv <- list(beta = as.numeric(pm(ps, "beta")),
             sigma_e = as.numeric(pm(ps, "sigma_e"))[1],
             sigma_b = as.numeric(pm(ps, "sigma_b")),
             L_corr = matrix(c(1, 0, rho, sqrt(max(1 - rho^2, 1e-8))),
                             nrow = 2, byrow = TRUE),
             b_std = pm_bstd(ps, nsub),
             alpha = as.numeric(pm(ps, "alpha"))[1],
             gamma = as.array(as.numeric(pm(ps, "gamma"))),
             tau_w = as.numeric(pm(ps, "tau_w"))[1],
             W01 = as.numeric(pm(ps, "W01")),
             z_step = as.numeric(pm(ps, "z_step")))
  iv <- iv[!vapply(iv, is.null, logical(1))]
  stopifnot(length(iv$sigma_e) == 1L, length(iv$alpha) == 1L,
            length(iv$tau_w) == 1L, length(iv$sigma_b) == 2L,
            length(iv$W01) == 2L, is.matrix(iv$b_std),
            nrow(iv$b_std) == nsub, ncol(iv$b_std) == 2L)
  cat(sprintf("  init: beta %d | b_std %dx%d | z_step %d | sigma_e %.4f | alpha %.4f\n",
              length(iv$beta), nrow(iv$b_std), ncol(iv$b_std),
              length(iv$z_step), iv$sigma_e, iv$alpha))

  f2 <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = tvar,
                      method = "spline-PH-mcmc",
                      control = CTL(list(init_values = iv)))
  w <- f2$convergence$warm_start
  cat(sprintf("  POTENTIAL AT THE POSTERIOR MEAN : %.1f\n",
              as.numeric(w$potential_warm)))
  cat(sprintf("  potential at a uniform start    : %.1f\n",
              as.numeric(w$potential_uniform)))
  cat(sprintf("  accepted: %s\n", isTRUE(as.logical(w$used))))
  invisible(NULL)
}

sim <- sim_joint(n = 300, seed = 3001L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
run_pair("SIMULATED (auto warm start scores 167,467; uniform 50,204)",
         lme(y ~ time + age + sex, random = ~ time | id, data = dl,
             control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100)),
         coxph(Surv(time, event) ~ trt, data = ds), ds, "time")

data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id)); pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12; pbc2.id$years2 <- pbc2.id$years / 12
pl <- pbc2[, c("id", "year2", "serBilir", "age")]; pl$log_serBilir <- log(pl$serBilir)
psv <- pbc2.id[, c("id", "years2", "status2", "drug")]
run_pair("PBC2 (auto warm start scores 1,881; uniform 8,746)",
         lme(log_serBilir ~ year2 + age, random = ~ year2 | id, data = pl,
             control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100)),
         coxph(Surv(years2, status2) ~ drug, data = psv), psv, "year2")

cat("\n---------------- read ----------------\n")
cat("  pbc2 anchors the scale: its posterior mean should score near or below\n")
cat("  its warm start's 1,881. If the SIMULATED posterior mean also lands\n")
cat("  near 167k, then 167k is simply what this data's potential IS, the\n")
cat("  seeding was never at fault, and the acceptance test - min() over three\n")
cat("  uniform draws - is what needs fixing.\n")
