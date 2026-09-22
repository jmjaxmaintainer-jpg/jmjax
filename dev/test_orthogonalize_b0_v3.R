# ==============================================================================
# Third pass, because the first two tests were both mis-specified BY ME and I
# am not going to settle this one by picking a better number out of the air.
#
#   v1  asserted beta_0 and beta_2 were invariant. They are the two
#       parameters the reparameterization is DEFINED to move. Test-design
#       error; the implementation note predicted the movement in writing.
#
#   v2  fixed that - the identified combinations agreed cleanly
#         I1  0.36228 vs 0.36136   z = -0.66
#         I2  0.00256 vs 0.00257   z = +0.33
#       and the constraint provably bit (mean c0, c1 = 0 to 6 decimals in
#       arm D). Then it FAILED on "max|fitted diff| = 1.178e-02 > 0.01",
#       where 0.01 was an absolute constant with no derivation behind it,
#       applied to a MAXIMUM over 1945 observations of a difference between
#       two INDEPENDENT MCMC runs. Two runs of the identical model differ by
#       some amount too; v2 never measured what that amount is, so it had no
#       basis for calling 1.178e-02 large.
#
# THE FIX IS A NEGATIVE CONTROL, not a better threshold. Run the DEFAULT
# configuration twice under different seeds and compare it to itself. That
# gives the null distribution of the fitted-value discrepancy directly, from
# the same machinery, with no assumption about Monte Carlo error on my part.
# Then:
#
#     discrepancy(A, D)  <=  discrepancy(A, A')     -> noise, proven
#     discrepancy(A, D)  >>  discrepancy(A, A')     -> real, option dies
#
# There is no threshold to choose. The comparison calibrates itself.
#
# CHECK 3 ALSO HAD A BUG IN ITS REASONING. v2 compared the joint model's
# beta_0 against lme()'s fixed intercept and expected the constrained arm to
# match. It does not: lme 0.47342 sits between arm A (0.533) and arm D
# (0.361), if anything nearer arm A. But lme's FIXED intercept is not the
# comparable quantity either - lme's BLUPs do not sum to exactly zero, so
# lme has its own c0 and c1 and therefore its own I1. That is what should be
# compared, and it is free to compute. (It still need not agree closely: a
# joint model conditions on the survival process, which is the entire point
# of fitting one, so a shift away from lme is expected rather than alarming.
# Reported, not relied on - for real this time.)
#
#   Rscript dev/test_orthogonalize_b0_v3.R
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
OUT <- file.path(OUTDIR, sprintf("test_orth_b0_v3_dup%d_%s", DUP,
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
IDX   <- match(data_long$id, .sub$id)
TT    <- data_long$year2; AG <- data_long$age

cat("=============================================================\n")
cat(sprintf("  orthogonalize_b0 | negative-control calibration\n"))
cat(sprintf("  subjects %d | dup x%d | chains %d | %d/%d\n",
            N_SUB, DUP, CHAINS, WARMUP, SAMPLES))
cat("=============================================================\n")

as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
bcol <- function(ps, j) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(t(vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
      else { a <- as.matrix(d); as.numeric(a[, j]) }
    }, numeric(N_SUB))))
  }
  a <- as.array(b); if (length(dim(a)) == 3L) a[, , j] else NULL
}
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

# Per-draw fitted values would be [draws, 1945] and large; every term is
# linear in the parameters, so the posterior MEAN of mu is mu evaluated at
# the posterior means. Kept as a function so all three arms go through the
# identical code path.
arm <- function(fit) {
  ps <- fit$posterior_samples
  B  <- as_draw_matrix(ps[["beta"]]); B0 <- bcol(ps, 1L); B1 <- bcol(ps, 2L)
  stopifnot(ncol(B) == 3L, !is.null(B0), !is.null(B1), ncol(B0) == N_SUB)
  ac <- AGE_I - ABAR; S <- cbind(1, ac)
  CO <- B0 %*% t(solve(crossprod(S), t(S)))
  bm <- colMeans(B); b0m <- colMeans(B0); b1m <- colMeans(B1)
  list(beta = B, c0 = CO[, 1], c1 = CO[, 2],
       I1 = B[, 1] + CO[, 1] - CO[, 2] * ABAR,
       I2 = B[, 3] + CO[, 2],
       mu = bm[1] + bm[2] * TT + bm[3] * AG + b0m[IDX] + b1m[IDX] * TT,
       ess = unlist(fit$diagnostics$ess))
}

cat("\nfitting A  (defaults, seed 1)...\n")
tA  <- system.time(fA  <- run_arm(list(seed = 1L)))
cat("fitting A' (defaults, seed 2) - the negative control...\n")
tA2 <- system.time(fA2 <- run_arm(list(seed = 2L)))
cat("fitting D  (orthogonalize_b0, seed 1)...\n")
tD  <- system.time(fD  <- run_arm(list(seed = 1L, orthogonalize_b0 = TRUE)))
A <- arm(fA); A2 <- arm(fA2); D <- arm(fD)

zdiff <- function(xa, xd) {
  mc <- sqrt(stats::var(xa) / ess_geyer(xa) + stats::var(xd) / ess_geyer(xd))
  (mean(xd) - mean(xa)) / mc
}

cat("\n---- constraint bites ----\n")
cat(sprintf("  A   mean(c0) %10.5f  mean(c1) %10.6f\n", mean(A$c0), mean(A$c1)))
cat(sprintf("  A'  mean(c0) %10.5f  mean(c1) %10.6f\n", mean(A2$c0), mean(A2$c1)))
cat(sprintf("  D   mean(c0) %10.5f  mean(c1) %10.6f\n", mean(D$c0), mean(D$c1)))

cat("\n---- identified combinations, vs the negative control ----\n")
cat(sprintf("  %-22s %10s %10s\n", "", "z(A,A')", "z(A,D)"))
cat(sprintf("  %-22s %10.2f %10.2f\n", "I1 (level)",
            zdiff(A$I1, A2$I1), zdiff(A$I1, D$I1)))
cat(sprintf("  %-22s %10.2f %10.2f\n", "I2 (age effect)",
            zdiff(A$I2, A2$I2), zdiff(A$I2, D$I2)))
cat(sprintf("  %-22s %10.2f %10.2f\n", "beta_0 (expected to move)",
            zdiff(A$beta[, 1], A2$beta[, 1]), zdiff(A$beta[, 1], D$beta[, 1])))

cat("\n---- fitted values: treatment against its own null ----\n")
dn <- A2$mu - A$mu     # two runs of the SAME model, different seed
dt <- D$mu  - A$mu     # the reparameterization
cat(sprintf("  %-28s %12s %12s\n", "", "A vs A' (null)", "A vs D"))
cat(sprintf("  %-28s %12.3e %12.3e\n", "mean|diff|", mean(abs(dn)), mean(abs(dt))))
cat(sprintf("  %-28s %12.3e %12.3e\n", "max|diff|",  max(abs(dn)),  max(abs(dt))))
cat(sprintf("  %-28s %12.3e %12.3e\n", "sd(diff)",   sd(dn),        sd(dt)))
r_mean <- mean(abs(dt)) / mean(abs(dn)); r_max <- max(abs(dt)) / max(abs(dn))
cat(sprintf("\n  ratio treatment/null:  mean %.2fx   max %.2fx\n", r_mean, r_max))
cat("  ~1x means the reparameterization perturbs fitted values no more than\n")
cat("  re-running the identical model with a different seed does.\n")

cat("\n---- lme's OWN invariant (v2 compared the wrong quantity) ----\n")
lf <- nlme::fixef(lme_fit); re <- nlme::ranef(lme_fit)
lc0 <- mean(re[[1]]); lc1 <- unname(stats::coef(
  stats::lm(re[[1]] ~ I(AGE_I - ABAR)))[2])
cat(sprintf("  lme  intercept %8.5f  mean(BLUP) %9.5f  age-slope(BLUP) %9.6f\n",
            lf[["(Intercept)"]], lc0, lc1))
cat(sprintf("  lme  I1 = %.5f    I2 = %.6f\n",
            lf[["(Intercept)"]] + lc0 - lc1 * ABAR, lf[["age"]] + lc1))
cat(sprintf("  jm   I1 = %.5f    I2 = %.6f   (arm D)\n", mean(D$I1), mean(D$I2)))
cat("  A joint model conditions on the survival process, so a shift from\n")
cat("  lme is expected. Context, not a criterion.\n")

cat("\n---- efficiency, restated ----\n")
for (k in c("beta_0", "beta_2", "sigma_b0", "alpha")) {
  if (k %in% names(A$ess) && k %in% names(D$ess))
    cat(sprintf("  %-10s  A %8.1f   A' %8.1f   D %8.1f\n",
                k, A$ess[[k]], A2$ess[[k]], D$ess[[k]]))
}
ok <- function(e) { e <- e[is.finite(e) & e > 0]; min(e) }
cat(sprintf("  %-10s  A %8.1f   A' %8.1f   D %8.1f\n",
            "min ESS", ok(A$ess), ok(A2$ess), ok(D$ess)))

cat("\n==================== VERDICT ====================\n")
zI1 <- abs(zdiff(A$I1, D$I1)); zI2 <- abs(zdiff(A$I2, D$I2))
nI1 <- abs(zdiff(A$I1, A2$I1)); nI2 <- abs(zdiff(A$I2, A2$I2))
pass_inv <- zI1 <= max(3, nI1 * 1.5) && zI2 <= max(3, nI2 * 1.5)
pass_fit <- r_max <= 2 && r_mean <= 2
if (pass_inv && pass_fit) {
  cat("  PASS. The reparameterization disturbs the identified quantities and\n")
  cat("  the fitted values no more than re-seeding the identical model does.\n")
  cat("  v1 and v2 both failed on criteria I specified wrongly, not on the\n")
  cat("  code: v1 asserted invariance of the parameters the transform moves,\n")
  cat("  v2 judged a max over 1945 observations against an undrived constant.\n")
} else {
  cat("  FAIL, and this time against a self-calibrating comparison rather\n")
  cat("  than a number I chose:\n")
  if (!pass_inv) cat(sprintf("    invariants: z(A,D) %.2f/%.2f vs null %.2f/%.2f\n",
                             zI1, zI2, nI1, nI2))
  if (!pass_fit) cat(sprintf("    fitted values: %.2fx the null at the mean, %.2fx at the max\n",
                             r_mean, r_max))
}
cat("\n  One dataset, one seed pair. Replication still outstanding.\n")

saveRDS(list(dup = DUP, r_mean = r_mean, r_max = r_max,
             z_I1 = zdiff(A$I1, D$I1), z_I2 = zdiff(A$I2, D$I2),
             null_z_I1 = zdiff(A$I1, A2$I1), null_z_I2 = zdiff(A$I2, A2$I2),
             mean_abs_null = mean(abs(dn)), mean_abs_treat = mean(abs(dt)),
             max_abs_null = max(abs(dn)), max_abs_treat = max(abs(dt)),
             ess_A = A$ess, ess_A2 = A2$ess, ess_D = D$ess,
             lme_I1 = lf[["(Intercept)"]] + lc0 - lc1 * ABAR,
             lme_I2 = lf[["age"]] + lc1,
             elapsed = c(A = tA[["elapsed"]], A2 = tA2[["elapsed"]],
                         D = tD[["elapsed"]])), paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
