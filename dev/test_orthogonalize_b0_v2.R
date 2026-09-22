# ==============================================================================
# The v1 test flagged beta_0 (z = -14.80) and beta_2 (z = +15.18) as moving
# beyond Monte Carlo error, and passed everything else - including sigma_b0
# (z = 0.48), the failure the singular prior was supposed to produce.
#
# THE TEST CRITERION CONTRADICTED THE CORRECTNESS ARGUMENT IT WAS MEANT TO
# CHECK. The implementation note says, in as many words, that for any draw
# there is a shift of beta giving identical fitted values - which is a
# PREDICTION that beta_0 and beta_2 will differ between the arms. Then v1
# went and tested beta_0 and beta_2 for equality. Those two parameters are
# precisely the ones that should NOT be invariant; asserting on them was a
# test-design error, and "the flagged parameters are exactly the ones the
# constraint acts on" is not a defence unless the invariant they belong to
# is checked directly. That is what this script does.
#
# THE ALGEBRA. Writing b_i0 = c0 + c1*(age_i - abar) + r_i, the part of the
# linear predictor built from subject-constant terms is
#
#   beta_0 + beta_2*age_i + b_i0
#     = [beta_0 + c0 - c1*abar]  +  [beta_2 + c1]*age_i  +  r_i
#
# so the identified quantities are
#
#   I1 = beta_0 + c0 - c1*abar          (the level)
#   I2 = beta_2 + c1                    (the age effect)
#
# Orthogonalization forces c0 = c1 = 0, so I1 and I2 collapse onto beta_0
# and beta_2 themselves. If the reparameterization is correct, I1 and I2
# agree ACROSS ARMS even though beta_0 and beta_2 do not - and if it is
# wrong, they will not. Both are on the ORIGINAL covariate scale, since
# age_i is raw.
#
# THREE CHECKS, in increasing order of how hard they are to fool:
#   1. I1 and I2 agree across arms, on the MCSE scale.
#   2. Every FITTED VALUE agrees. This needs no algebra to be right: if the
#      arms describe the same model, X beta + Z b must match observation by
#      observation, and no relabelling of parameters can fake it.
#   3. lme()'s own fixed effects as an external referee. lme's BLUPs are
#      shrunk toward zero and very nearly sum to zero, so lme's intercept
#      should sit near the CONSTRAINED arm. If the constrained estimate is
#      the one that matches standard mixed-model convention, that is an
#      argument the constraint improves interpretability rather than
#      damaging it - but it is the weakest of the three and is reported,
#      not relied on.
#
#   Rscript dev/test_orthogonalize_b0_v2.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 supplies the pbc2 data.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

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
OUT <- file.path(OUTDIR, sprintf("test_orth_b0_v2_dup%d_%s", DUP,
                                 format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)
options(error = function() {
  try(while (sink.number() > 0) sink(), silent = TRUE)
})

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

.sub  <- data_long[!duplicated(data_long$id), c("id", "age")]
.sub  <- .sub[order(.sub$id), ]
AGE_I <- as.numeric(.sub$age); N_SUB <- nrow(.sub); ABAR <- mean(AGE_I)

cat("=============================================================\n")
cat(sprintf("  orthogonalize_b0, INVARIANCE test | subjects %d | dup x%d\n",
            N_SUB, DUP))
cat(sprintf("  chains %d | warmup %d | samples %d\n", CHAINS, WARMUP, SAMPLES))
cat("=============================================================\n")

as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
b0_matrix_of <- function(ps) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(t(vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[1], numeric(1))
      else { a <- as.matrix(d); as.numeric(a[, 1]) }
    }, numeric(N_SUB))))
  }
  a <- as.array(b)
  if (length(dim(a)) == 3L) a[, , 1] else as.matrix(a)
}
b1_matrix_of <- function(ps) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(t(vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[2], numeric(1))
      else { a <- as.matrix(d); as.numeric(a[, 2]) }
    }, numeric(N_SUB))))
  }
  a <- as.array(b)
  if (length(dim(a)) == 3L) a[, , 2] else NULL
}

lme_fit <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id,
               data = data_long,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)

run_arm <- function(extra = list()) {
  ctl <- utils::modifyList(
    list(n_interior_knots = 5, spline_prior = "penalized",
         rw2_implementation = "vectorized", dense_mass_spline = TRUE,
         num_warmup = WARMUP, num_samples = SAMPLES,
         num_chains = CHAINS, progress_bar = FALSE), extra)
  jm_fit_prefit(lme_fit, cox_fit, data_surv = data_surv, time_var = "year2",
                method = "spline-PH-mcmc", control = ctl)
}

# Per draw: the two identified combinations, on the original covariate scale.
invariants <- function(fit) {
  ps <- fit$posterior_samples
  B  <- as_draw_matrix(ps[["beta"]])
  B0 <- b0_matrix_of(ps)
  stopifnot(ncol(B) == 3L, ncol(B0) == N_SUB, nrow(B0) == nrow(B))
  ac <- AGE_I - ABAR
  S  <- cbind(1, ac)
  CO <- B0 %*% t(solve(crossprod(S), t(S)))     # [draws, 2] = (c0, c1)
  list(beta = B, b0 = B0, c0 = CO[, 1], c1 = CO[, 2],
       I1 = B[, 1] + CO[, 1] - CO[, 2] * ABAR,
       I2 = B[, 3] + CO[, 2])
}

cat("\nfitting arm A (defaults)...\n")
tA <- system.time(fitA <- run_arm())
cat("fitting arm D (orthogonalize_b0)...\n")
tD <- system.time(fitD <- run_arm(list(orthogonalize_b0 = TRUE)))
IA <- invariants(fitA); ID <- invariants(fitD)

ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE,
                   demean = TRUE)$acf[, 1, 1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) {
    pair <- ac[k] + ac[k + 1L]; if (pair <= 0) break
    s <- s + pair; k <- k + 2L
  }
  tau <- -1 + 2 * s; if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}
zdiff <- function(xa, xd) {
  mc <- sqrt(stats::var(xa) / ess_geyer(xa) + stats::var(xd) / ess_geyer(xd))
  (mean(xd) - mean(xa)) / mc
}

cat("\n---- did the constraint actually bite? ----\n")
cat(sprintf("  arm A   mean(c0) %10.5f   mean(c1) %10.6f\n",
            mean(IA$c0), mean(IA$c1)))
cat(sprintf("  arm D   mean(c0) %10.5f   mean(c1) %10.6f   (both must be ~0)\n",
            mean(ID$c0), mean(ID$c1)))

cat("\n---- CHECK 1: the identified combinations ----\n")
cat(sprintf("  %-28s %11s %11s %9s\n", "", "arm A", "arm D", "z"))
cat(sprintf("  %-28s %11.5f %11.5f %9.2f\n", "beta_0            (moves)",
            mean(IA$beta[, 1]), mean(ID$beta[, 1]),
            zdiff(IA$beta[, 1], ID$beta[, 1])))
cat(sprintf("  %-28s %11.5f %11.5f %9.2f\n", "I1 = b0 + c0 - c1*abar",
            mean(IA$I1), mean(ID$I1), zdiff(IA$I1, ID$I1)))
cat(sprintf("  %-28s %11.5f %11.5f %9.2f\n", "beta_2            (moves)",
            mean(IA$beta[, 3]), mean(ID$beta[, 3]),
            zdiff(IA$beta[, 3], ID$beta[, 3])))
cat(sprintf("  %-28s %11.5f %11.5f %9.2f\n", "I2 = b2 + c1",
            mean(IA$I2), mean(ID$I2), zdiff(IA$I2, ID$I2)))
cat("\n  The two rows that should move do, and the two that should not\n")
cat("  should now sit inside Monte Carlo error.\n")

cat("\n---- CHECK 2: fitted values, observation by observation ----\n")
# mu_io = beta_0 + beta_1*t_io + beta_2*age_i + b_i0 + b_i1*t_io, posterior
# mean per observation. No algebra involved: if the arms are the same model
# this matches, and nothing about how the parameters are labelled can help.
fitted_mean <- function(I, fit) {
  B <- I$beta; B0 <- I$b0; B1 <- b1_matrix_of(fit$posterior_samples)
  stopifnot(!is.null(B1))
  idx <- match(data_long$id, .sub$id)
  tt  <- data_long$year2; ag <- data_long$age
  # average over draws first, then evaluate - linear in every term, so the
  # posterior mean of mu is mu at the posterior means
  bm <- colMeans(B); b0m <- colMeans(B0); b1m <- colMeans(B1)
  bm[1] + bm[2] * tt + bm[3] * ag + b0m[idx] + b1m[idx] * tt
}
muA <- fitted_mean(IA, fitA); muD <- fitted_mean(ID, fitD)
d <- muD - muA
cat(sprintf("  n obs %d   mean|diff| %.3e   max|diff| %.3e\n",
            length(d), mean(abs(d)), max(abs(d))))
cat(sprintf("  residual sd of the model (sigma_e) is about %.3f, so the\n",
            mean(as.numeric(unlist(fitA$estimates)[["sigma_e"]]))))
cat(sprintf("  discrepancy is %.2e of one residual sd.\n",
            max(abs(d)) / as.numeric(unlist(fitA$estimates)[["sigma_e"]])))

cat("\n---- CHECK 3: lme() as an external referee ----\n")
lf <- nlme::fixef(lme_fit)
cat(sprintf("  lme intercept  %10.5f\n", lf[["(Intercept)"]]))
cat(sprintf("  lme age        %10.6f\n", lf[["age"]]))
cat(sprintf("  arm A beta_0   %10.5f   beta_2 %10.6f\n",
            mean(IA$beta[, 1]), mean(IA$beta[, 3])))
cat(sprintf("  arm D beta_0   %10.5f   beta_2 %10.6f\n",
            mean(ID$beta[, 1]), mean(ID$beta[, 3])))
cat("  lme's BLUPs very nearly sum to zero, so lme's fixed effects follow\n")
cat("  the same convention the CONSTRAINED arm enforces exactly.\n")

cat("\n==================== VERDICT ====================\n")
z1 <- zdiff(IA$I1, ID$I1); z2 <- zdiff(IA$I2, ID$I2)
if (abs(z1) < 3 && abs(z2) < 3 && max(abs(d)) < 0.01) {
  cat("  PASS. beta_0 and beta_2 are relocated, not biased: the quantities\n")
  cat("  the data identify agree across arms, and the fitted values agree\n")
  cat("  observation by observation. v1's REJECT was a test-design error -\n")
  cat("  it asserted invariance of the two parameters the reparameterization\n")
  cat("  is defined to move.\n")
} else {
  cat("  FAIL. The invariants do NOT agree, so this is a real change to the\n")
  cat("  model and not a relocation. The option should not ship.\n")
  cat(sprintf("  z(I1) = %.2f   z(I2) = %.2f   max|fitted diff| = %.3e\n",
              z1, z2, max(abs(d))))
}
cat("\n  Still one dataset, one seed.\n")

saveRDS(list(dup = DUP,
             mean_c0_A = mean(IA$c0), mean_c1_A = mean(IA$c1),
             mean_c0_D = mean(ID$c0), mean_c1_D = mean(ID$c1),
             I1 = c(A = mean(IA$I1), D = mean(ID$I1)),
             I2 = c(A = mean(IA$I2), D = mean(ID$I2)),
             z_I1 = z1, z_I2 = z2,
             max_fitted_diff = max(abs(d)), mean_fitted_diff = mean(abs(d)),
             lme_fixef = lf,
             elapsed = c(A = tA[["elapsed"]], D = tD[["elapsed"]])),
        paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
