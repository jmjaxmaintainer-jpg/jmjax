# ==============================================================================
# v4: the location degeneracy is REAL and it is not only about beta_0.
#
# WHAT v3 ESTABLISHED (sampled space, PBC2, n=312, validated against the
# backend's own pre-back-transform ESS to within 0.3%):
#
#     corr(beta_0, mean(b_i0))   -0.9709
#     ESS(beta_0)                   440.5
#     ESS(beta_0 + mean(b_i0))     2893.1     <- the identified direction
#     ESS(beta_0 - mean(b_i0))      428.7     <- the ridge
#     recoverable factor             6.57x
#
# So section 6.1's mechanism holds after all. The earlier "refutation"
# (-0.1782, ratio 1.02x) was computed against back-transformed draws and
# measured the transform, not the posterior.
#
# THE GENERALIZATION THIS SCRIPT TESTS. mean(b_i0) is one direction in the
# 312-dimensional random-intercept space: the one that trades against
# beta_0. It is not the only one. `age` is SUBJECT-CONSTANT, so beta_2
# trades against the AGE-PROJECTED component of b_0 in exactly the same
# way - the data constrain beta_2 + slope(b_i0 ~ age_i), not either alone.
# That would explain why beta_2 (ESS 526) is the second-slowest parameter
# and why sigma_b0 (~595) is third, while beta_1 - a WITHIN-subject
# contrast - sits at 1234 and alpha at 5106.
#
# If that holds, the right fix is not a sum-to-zero constraint on b_0 (which
# removes one direction) but ORTHOGONALIZING b_0 against the full column
# space of the subject-level design, which removes all of them at once.
#
# TWO THINGS MEASURED HERE:
#   1. Per draw, regress b_i0 on [1, age_i] and correlate the two
#      coefficients against beta_0 and beta_2. Predicts ~-0.97 for BOTH if
#      the generalization is right, and ~-0.97 / ~0 if it is beta_0 only.
#   2. An arm using random_effects_method = "wishart_gibbs_centered", which
#      already implements the sum-to-zero constraint on b_0 (mcmc_model.py
#      :397). It should move beta_0 and NOT beta_2 - a direct test that
#      separates "one direction" from "the whole subspace".
#
# NOTE plain "wishart_gibbs" is already centered (b sampled directly) and
# made beta_0 WORSE (441.7 -> 379.3), which is consistent: centering alone
# does not remove a location degeneracy, the constraint does.
#
#   Rscript dev/diagnose_beta0_v4.R
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
# The constrained arm uses HMCGibbs, which has hung at n=624 in earlier
# testing. Default it ON at dup1 and OFF above that rather than risk a
# wedged run; BENCH_RUN_CENTERED=1 forces it.
RUN_CENTERED <- if (nzchar(Sys.getenv("BENCH_RUN_CENTERED")))
  .envi("BENCH_RUN_CENTERED", 1L) == 1L else DUP == 1L

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("diagnose_beta0_v4_dup%d_%s", DUP,
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
CTR <- mean(data_long$age); SCL <- stats::sd(data_long$age)

# Subject-level age, in ascending id order. Which order the backend stacks
# b in is ASSERTED below rather than assumed: if it were wrong, the
# age-projection would be meaningless.
.sub <- data_long[!duplicated(data_long$id), c("id", "age")]
.sub <- .sub[order(.sub$id), ]
AGE_I <- as.numeric(.sub$age)
N_SUB <- nrow(.sub)

cat("=============================================================\n")
cat(sprintf("  v4: is the degeneracy beta_0 ONLY, or the whole subject-\n"))
cat(sprintf("      constant subspace? | subjects %d | dup x%d\n", N_SUB, DUP))
cat(sprintf("  chains %d | warmup %d | samples %d | centered arm: %s\n",
            CHAINS, WARMUP, SAMPLES, if (RUN_CENTERED) "yes" else "skipped"))
cat("=============================================================\n")

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

as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}

# [draws, N_sub] of the random INTERCEPT column.
b0_matrix_of <- function(ps) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(t(vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[1], numeric(1))
      else {
        a <- as.matrix(d)
        if (ncol(a) >= 1L) as.numeric(a[, 1]) else as.numeric(a)
      }
    }, numeric(N_SUB))))
  }
  a <- as.array(b)
  if (length(dim(a)) == 3L) return(a[, , 1])
  if (length(dim(a)) == 2L) return(a)
  NULL
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

# ============================ ARM A ======================================
cat("\n=== A. defaults ===\n")
tA <- system.time(fitA <- run_arm())
cat(sprintf("  %.1fs  max R-hat %.4f  leapfrog %s\n", tA[["elapsed"]],
            max(unlist(fitA$diagnostics$rhat), na.rm = TRUE),
            format(fitA$convergence$mean_num_steps %||% NA)))

psA   <- fitA$posterior_samples
Borig <- as_draw_matrix(psA[["beta"]])
stopifnot(ncol(Borig) == 3L)
# sampled space, inverting jm_fit.R:2497 as validated in v3
Bstd <- Borig
Bstd[, 1] <- Borig[, 1] + CTR * Borig[, 3]
Bstd[, 3] <- SCL * Borig[, 3]

B0 <- b0_matrix_of(psA)
if (is.null(B0)) stop("no per-subject random intercepts returned", call. = FALSE)
stopifnot(ncol(B0) == N_SUB, nrow(B0) == nrow(Bstd))

# ---- project b_0 onto the subject-level design [1, age_i] ---------------
# Per draw: b_i0 ~ c0 + c1 * age_i. c0 is the direction beta_0 trades
# against (equivalently mean(b_i0) once age is centred); c1 is the one
# beta_2 trades against. Closed form, applied to all draws at once.
ac <- AGE_I - mean(AGE_I)
S  <- cbind(1, ac)
P  <- solve(crossprod(S), t(S))        # [2, N_sub]
CO <- B0 %*% t(P)                      # [draws, 2]: c0, c1
c0 <- CO[, 1]; c1 <- CO[, 2]

# ORDER CHECK. If b's columns were not in ascending-id order, c1 would be
# projecting onto a shuffled age vector and the correlation below would be
# noise. The residual SD of b_i0 about the fitted line must be smaller than
# its raw SD by a non-trivial margin for the projection to mean anything;
# more decisively, mean(b_i0) must reproduce c0 exactly (it does by
# construction only if the design column of 1s lines up with b's columns).
mb0 <- rowMeans(B0)
cat(sprintf("\n  order check: max|c0 - mean(b_i0)| = %.3e  (must be ~0)\n",
            max(abs(c0 - mb0))))

cat("\n---------------- the two degeneracies ----------------\n")
cat(sprintf("  corr(beta_0, c0)   [c0 = mean level of b_i0]   %8.4f\n",
            stats::cor(Bstd[, 1], c0)))
cat(sprintf("  corr(beta_2, c1)   [c1 = age-slope of b_i0]    %8.4f\n",
            stats::cor(Bstd[, 3], c1)))
cat(sprintf("  corr(beta_1, c1)   [control: within-subject]   %8.4f\n",
            stats::cor(Bstd[, 2], c1)))

# beta_2 is on the STANDARDIZED age scale, c1 on the raw age scale, so the
# identified combination is beta_2/SCL + c1. Scale-matched explicitly
# rather than added blind.
b2_raw <- Bstd[, 3] / SCL
cat("\n---------------- recoverable factors ----------------\n")
tbl <- rbind(
  c(ess_geyer(Bstd[, 1]), ess_geyer(Bstd[, 1] + c0), ess_geyer(Bstd[, 1] - c0)),
  c(ess_geyer(b2_raw),    ess_geyer(b2_raw + c1),    ess_geyer(b2_raw - c1)),
  c(ess_geyer(Bstd[, 2]), NA, NA))
rownames(tbl) <- c("beta_0", "beta_2 (age)", "beta_1 (time, control)")
colnames(tbl) <- c("ESS(param)", "ESS(param+c)", "ESS(param-c)")
print(round(tbl, 1))
cat(sprintf("\n  recoverable  beta_0 : %.2fx      beta_2 : %.2fx\n",
            tbl[1, 2] / tbl[1, 1], tbl[2, 2] / tbl[2, 1]))
cat("\n  Both >> 1  -> the whole subject-constant subspace is degenerate and\n")
cat("               a sum-to-zero constraint on b_0 fixes only HALF of it.\n")
cat("  Only beta_0 -> the existing wishart_gibbs_centered constraint is the\n")
cat("               right shape of fix and beta_2 is slow for another reason.\n")

# ============================ ARM C ======================================
if (RUN_CENTERED) {
  cat("\n=== C. wishart_gibbs_centered (sum-to-zero on b_0) ===\n")
  tC <- system.time(fitC <- tryCatch(
    run_arm(list(random_effects_method = "wishart_gibbs_centered")),
    error = function(e) { cat("  FAILED: ", conditionMessage(e), "\n"); NULL }))
  if (!is.null(fitC)) {
    cat(sprintf("  %.1fs  max R-hat %.4f\n", tC[["elapsed"]],
                max(unlist(fitC$diagnostics$rhat), na.rm = TRUE)))
    BC <- as_draw_matrix(fitC$posterior_samples[["beta"]])
    BCs <- BC; BCs[, 1] <- BC[, 1] + CTR * BC[, 3]; BCs[, 3] <- SCL * BC[, 3]
    B0C <- b0_matrix_of(fitC$posterior_samples)
    cat(sprintf("  mean(b_i0) sd across draws: %.3e  (0 if the constraint bites)\n",
                if (!is.null(B0C)) stats::sd(rowMeans(B0C)) else NA))
    cat(sprintf("  ESS beta_0 %8.1f   (A: %.1f)\n",
                ess_geyer(BCs[, 1]), ess_geyer(Bstd[, 1])))
    cat(sprintf("  ESS beta_2 %8.1f   (A: %.1f)\n",
                ess_geyer(BCs[, 3]), ess_geyer(Bstd[, 3])))
    cat(sprintf("  ESS alpha  %8.1f\n",
                as.numeric(unlist(fitC$diagnostics$ess)[["alpha"]])))
  }
} else {
  fitC <- NULL; tC <- NULL
}

cat("\n==================== WHAT THIS DECIDES ====================\n")
cat("  If beta_2 shows the same recoverable factor as beta_0, the fix worth\n")
cat("  building is orthogonalizing b_0 against ALL subject-constant design\n")
cat("  columns, not just its mean - and arm C should move beta_0 alone,\n")
cat("  leaving beta_2 where it was. That pattern would confirm both halves\n")
cat("  at once: the mechanism, and the insufficiency of the existing option.\n")

saveRDS(list(dup = DUP, ctr = CTR, scl = SCL, age_i = AGE_I,
             beta_std = Bstd, c0 = c0, c1 = c1, tbl = tbl,
             corr_b0_c0 = stats::cor(Bstd[, 1], c0),
             corr_b2_c1 = stats::cor(Bstd[, 3], c1),
             ess_A = unlist(fitA$diagnostics$ess),
             steps_A = fitA$convergence$mean_num_steps,
             elapsed_A = as.numeric(tA[["elapsed"]]),
             centered_ran = !is.null(fitC)), paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
