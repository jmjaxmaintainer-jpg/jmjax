# ==============================================================================
# v3: the geometry of beta_0, measured in the space NUTS ACTUALLY EXPLORES.
#
# WHY v2 HAD TO BE REDONE. jm_fit() standardizes numeric covariates in
# long_formula by default and then back-transforms beta before returning it
# (jm_fit.R:2497, beta_original = A %*% beta_sampled). So
# fit$posterior_samples$beta is on the ORIGINAL scale - a linear map of the
# coordinates NUTS saw. v2 ran its correlation matrix and eigendecomposition
# on those back-transformed draws and reported a -0.9760 correlation between
# beta_0 and beta_2 with a worst eigendirection of 0.0224.
#
# That correlation is an ALGEBRAIC CONSEQUENCE of the back-transform, not a
# property of the posterior geometry. With only `age` standardized,
#
#     beta_0_orig = beta_0_std - (ctr/scl) * beta_age_std
#
# and for PBC2 ctr/scl ~ 4.7, so beta_0_orig is dominated by -4.7 *
# beta_age_std and corr(beta_0_orig, beta_age_orig) is forced toward -1 no
# matter what the sampler did. v2 measured the transform, not the target.
#
# Two conclusions built on v2 therefore have to be withdrawn and re-tested:
#   (a) "a near-singular beta_0/beta_2 ridge that a dense mass block should
#       fix" - if standardization already removed it in sampled space there
#       is no ridge to precondition, which would EXPLAIN the recorded null
#       result for dense_mass_beta rather than leaving it a mystery;
#   (b) "section 6.1's beta_0 <-> mean(b_i0) degeneracy is refuted on PBC2"
#       - that -0.1782 was also computed against beta_0_ORIG, so it says
#       nothing about the sampled coordinate either.
#
# WHAT THIS SCRIPT DOES. Recovers the standardized-space draws exactly by
# inverting the same map jm_fit applied, then measures the geometry in BOTH
# coordinate systems side by side. The reconstruction is self-validating:
# ESS(beta_0_std) must land near the value the BACKEND reported before any
# back-transformation (441.7 on PBC2 at these settings). Arm B refits with
# standardization off, where sampled space and original space coincide, as
# an independent cross-check that needs no algebra at all.
#
#   Rscript dev/diagnose_beta0_v3.R
#   BENCH_PBC2_DUP=2 Rscript dev/diagnose_beta0_v3.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 supplies the pbc2 data.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

# base R gained %||% only in 4.4.0, and jmjax does not export its internal
# copy, so define one rather than have the script die on a missing operator
# after the fits have already run.
`%||%` <- function(a, b) if (is.null(a)) b else a

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
DUP     <- .envi("BENCH_PBC2_DUP", 1L)
CHAINS  <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP  <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES <- .envi("BENCH_JX_SAMPLES", 1000L)

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("diagnose_beta0_v3_dup%d_%s", DUP,
                                 format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)
options(error = function() {
  try(while (sink.number() > 0) sink(), silent = TRUE)
})

# ---- data ---------------------------------------------------------------
data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id))
pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2$status2 <- as.integer(pbc2$status != "alive")
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12; pbc2.id$years2 <- pbc2.id$years / 12
data_long <- pbc2[, c("id", "year2", "serBilir", "age")]
data_long$log_serBilir <- log(data_long$serBilir)
data_surv <- pbc2.id[, c("id", "years2", "status2", "drug")]
if (DUP > 1L) {
  bm <- max(data_surv$id); dl <- list(); ds <- list()
  for (k in seq_len(DUP)) {
    a <- data_long; a$id <- a$id + (k - 1L) * bm
    b <- data_surv; b$id <- b$id + (k - 1L) * bm
    dl[[k]] <- a; ds[[k]] <- b
  }
  data_long <- do.call(rbind, dl); data_surv <- do.call(rbind, ds)
}

# The EXACT constants jm_fit will use: mean and sd of `age` over the rows of
# data_long (not over subjects - jm_fit.R:826 computes them on the long
# frame, and duplicating subjects leaves both unchanged anyway).
CTR <- mean(data_long$age); SCL <- stats::sd(data_long$age)

cat("=============================================================\n")
cat(sprintf("  beta_0 geometry in SAMPLED space | subjects %d | dup x%d\n",
            length(unique(data_long$id)), DUP))
cat(sprintf("  chains %d | warmup %d | samples %d\n", CHAINS, WARMUP, SAMPLES))
cat(sprintf("  age: centre %.4f  scale %.4f  =>  ctr/scl = %.3f\n",
            CTR, SCL, CTR / SCL))
cat(sprintf("  predicted corr(beta_0, beta_age) on the ORIGINAL scale,\n"))
cat(sprintf("  purely from the back-transform: %.4f  (v2 measured -0.9760)\n",
            -CTR / sqrt(CTR^2 + SCL^2)))
cat("=============================================================\n")

# posterior_samples$beta is a LIST of length n_draws, each element a
# numeric vector of length p - jm_fit.R:2505 rebuilds it that way after the
# back-transform. as.matrix() on that does NOT give a [draws, p] numeric
# matrix, which is what broke the first run of this script.
as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v)
  storage.mode(m) <- "double"
  m
}

ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE,
                   demean = TRUE)$acf[, 1, 1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) {
    pair <- ac[k] + ac[k + 1L]
    if (pair <= 0) break
    s <- s + pair; k <- k + 2L
  }
  tau <- -1 + 2 * s
  if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}

mean_b0_of <- function(ps) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(vapply(b, function(d) {
      m <- if (is.list(d)) vapply(d, function(r) as.numeric(r)[1], numeric(1))
           else as.numeric(d)
      mean(m)
    }, numeric(1)))
  }
  a <- as.array(b)
  if (length(dim(a)) == 3L) return(apply(a[, , 1, drop = FALSE], 1, mean))
  if (length(dim(a)) == 2L) return(apply(a, 1, mean))
  NULL
}

report_geometry <- function(label, B, mb0) {
  # B: [draws, p] beta in whatever space; column 1 is the intercept.
  cat(sprintf("\n---- %s ----\n", label))
  p <- ncol(B)
  nms <- c("beta0", paste0("beta", seq_len(p - 1L)))
  cm <- stats::cor(B)
  cat("  correlations with beta_0:\n")
  for (j in seq_len(p)[-1]) cat(sprintf("    %-8s %8.4f\n", nms[j], cm[1, j]))
  cat("  per-parameter ESS:\n")
  for (j in seq_len(p)) cat(sprintf("    %-8s %8.1f\n", nms[j], ess_geyer(B[, j])))
  if (p >= 2) {
    ev <- eigen(cm, symmetric = TRUE)
    k <- p
    z <- scale(B) %*% ev$vectors[, k]
    cat(sprintf("  tightest eigendirection: eigval %.4f  ESS %.1f\n",
                ev$values[k], ess_geyer(z)))
    z1 <- scale(B) %*% ev$vectors[, 1]
    cat(sprintf("  widest   eigendirection: eigval %.4f  ESS %.1f\n",
                ev$values[1], ess_geyer(z1)))
  }
  if (!is.null(mb0)) {
    cat(sprintf("  corr(beta_0, mean(b_i0))      %8.4f\n",
                stats::cor(B[, 1], mb0)))
    cat(sprintf("  ESS(beta_0)                   %8.1f\n", ess_geyer(B[, 1])))
    cat(sprintf("  ESS(beta_0 + mean(b_i0))      %8.1f\n", ess_geyer(B[, 1] + mb0)))
    cat(sprintf("  ESS(beta_0 - mean(b_i0))      %8.1f\n", ess_geyer(B[, 1] - mb0)))
  }
}

lme_fit <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id,
               data = data_long,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)

run_arm <- function(std) {
  jm_fit_prefit(
    lme_fit, cox_fit, data_surv = data_surv, time_var = "year2",
    method = "spline-PH-mcmc",
    control = list(n_interior_knots = 5, spline_prior = "penalized",
                   rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                   standardize_covariates = std,
                   num_warmup = WARMUP, num_samples = SAMPLES,
                   num_chains = CHAINS, progress_bar = FALSE))
}

# ============================ ARM A ======================================
cat("\n=== A. defaults (standardize_covariates = TRUE) ===\n")
tA <- system.time(fitA <- run_arm(TRUE))
cat(sprintf("  %.1fs  max R-hat %.4f  mean leapfrog steps %s\n",
            tA[["elapsed"]], max(unlist(fitA$diagnostics$rhat), na.rm = TRUE),
            format(fitA$convergence$mean_num_steps %||% NA)))

psA <- fitA$posterior_samples
Borig <- as_draw_matrix(psA[["beta"]])
mb0A  <- mean_b0_of(psA)

# Invert the map jm_fit applied. Only `age` is standardized; time_var is
# excluded by construction (jm_fit.R:815), so the columns are
# (Intercept, year2, age) and only two entries of A are not identity:
#   beta_0_orig   = beta_0_std - (CTR/SCL) * beta_age_std
#   beta_age_orig = beta_age_std / SCL
# hence
#   beta_age_std  = SCL * beta_age_orig
#   beta_0_std    = beta_0_orig + CTR * beta_age_orig
stopifnot(
  # (Intercept), year2, age - the reconstruction below indexes column 3 as
  # `age`, which is only correct for this exact long_formula.
  ncol(Borig) == 3L)
Bstd <- Borig
Bstd[, 1] <- Borig[, 1] + CTR * Borig[, 3]
Bstd[, 3] <- SCL * Borig[, 3]

report_geometry("A, ORIGINAL scale (what v2 measured)", Borig, mb0A)
report_geometry("A, SAMPLED scale (what NUTS explored)", Bstd, mb0A)

cat("\n  VALIDATION: ESS(beta_0) in sampled space should land near the\n")
cat("  backend's own pre-back-transform figure, 441.7 at these settings.\n")
cat(sprintf("  reconstructed %.1f   |   fit$diagnostics$ess$beta_0 = %s\n",
            ess_geyer(Bstd[, 1]),
            format(round(as.numeric(unlist(fitA$diagnostics$ess)[["beta_0"]]), 1))))

# ============================ ARM B ======================================
# No algebra at all: with standardization off, what comes back IS what NUTS
# sampled. If arm A's reconstruction is right, B's original-scale geometry
# should look like A's ORIGINAL-scale panel, not like A's sampled panel.
cat("\n=== B. standardize_covariates = FALSE (sampled == original) ===\n")
tB <- system.time(fitB <- run_arm(FALSE))
cat(sprintf("  %.1fs  max R-hat %.4f  mean leapfrog steps %s\n",
            tB[["elapsed"]], max(unlist(fitB$diagnostics$rhat), na.rm = TRUE),
            format(fitB$convergence$mean_num_steps %||% NA)))
psB <- fitB$posterior_samples
BB <- as_draw_matrix(psB[["beta"]])
report_geometry("B, sampled == original", BB, mean_b0_of(psB))

cat("\n==================== VERDICT ====================\n")
cat(sprintf("  ESS(beta_0)  A sampled-space  %8.1f\n", ess_geyer(Bstd[, 1])))
cat(sprintf("  ESS(beta_0)  A original-scale %8.1f\n", ess_geyer(Borig[, 1])))
cat(sprintf("  ESS(beta_0)  B unstandardized %8.1f\n", ess_geyer(BB[, 1])))
cat(sprintf("  leapfrog steps  A %s   B %s\n",
            format(fitA$convergence$mean_num_steps %||% NA),
            format(fitB$convergence$mean_num_steps %||% NA)))
cat("\n  If A-sampled corr(beta_0, beta_age) is near ZERO while A-original\n")
cat("  is near -0.98, then standardize_covariates already removed the ridge\n")
cat("  and there was never anything for dense_mass_beta to precondition -\n")
cat("  which retires that null result as EXPLAINED rather than mysterious.\n")
cat("  Whatever still limits beta_0 at ~440 is then NOT beta-block geometry.\n")

saveRDS(list(dup = DUP, ctr = CTR, scl = SCL,
             beta_orig = Borig, beta_std = Bstd, beta_unstd = BB,
             mean_b0 = mb0A,
             ess_A_std = ess_geyer(Bstd[, 1]), ess_A_orig = ess_geyer(Borig[, 1]),
             ess_B = ess_geyer(BB[, 1]),
             steps_A = fitA$convergence$mean_num_steps,
             steps_B = fitB$convergence$mean_num_steps,
             elapsed_A = as.numeric(tA[["elapsed"]]),
             elapsed_B = as.numeric(tB[["elapsed"]])), paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
