# ==============================================================================
# The k=4 correctness failure: implementation bug, or my invariants again?
#
# WHAT THE REPLICATION SHOWED.
#   k=2 (n=312, n=624): 8/8 replicates clean, |z| <= 1.03.
#   k=4 (+ sex + drug):  4/4 replicates FAIL, z(I1) = +11.87, +12.57, +11.89,
#                        +11.95 and z(I2) = -7.35, -6.67, -8.11, -7.92.
#
# Same sign, same magnitude, every seed. That is deterministic, not noise -
# something is systematically wrong. The question is where.
#
# THE SPECIFIC SUSPICION, stated before the test so it can be refuted. The
# k=4 design has FOUR subject-constant columns - intercept, age, sex, drug -
# and the implementation orthogonalizes b_0 against all four. My invariants()
# function projects b_0 onto [1, age] ONLY. So in arm A it omits the sex and
# drug components of b_0 entirely, and worse, regressing on a subset gives
# biased c0 and c1 whenever sex and drug correlate with age. At k=2 there is
# no subset and no bias, which is exactly the pattern observed.
#
# THAT IS A PREDICTION, NOT A DEFENCE. It has failed twice already to assume
# my own test is the broken part:
#   v1 asserted invariance of the two parameters the transform is defined to
#      move;
#   v2 judged a maximum over 1945 observations against a constant I invented.
# Both times I was right about the cause and both times I only knew it after
# building a check that could have said otherwise. So this script does two
# things, and the second one is the one that counts.
#
# 1. THE GENERAL INVARIANT. Let S be the [N_sub, k] matrix of subject-level
#    values of every subject-constant design column (intercept included).
#    Writing b_i0 = S_i'c + r_i, the subject-constant part of the linear
#    predictor is
#
#        S_i'beta_S + b_i0  =  S_i'(beta_S + c) + r_i
#
#    so the identified vector is beta_S + c, componentwise, with c obtained
#    by regressing b_0 on S directly - no centring, no abar correction, and
#    correct for any k. Orthogonalization forces c = 0, collapsing it onto
#    beta_S. This is the k-general form my replication script got wrong.
#
# 2. FITTED VALUES AGAINST A RE-SEEDED NULL. No algebra at all: fit the
#    default twice under different seeds, and compare that discrepancy to
#    the default-vs-orthogonalized one. If the reparameterization changed
#    the model, its fitted values move MORE than re-seeding moves them, and
#    no amount of me being clever about projections can hide it. This check
#    existed in v3, passed at 0.77x/0.85x, and I then DROPPED it from the
#    replication script in favour of an algebraic test that does not
#    generalize. That omission is what produced this failure.
#
# If (1) clears and (2) shows ~1x, the option is sound at k=4 and the
# replication script's correctness column needs fixing. If (2) shows the
# treatment moving fitted values well beyond the null, the option is broken
# for k > 2 and does not ship, whatever (1) says.
#
#   Rscript dev/test_orth_b0_k4.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
suppressPackageStartupMessages(library(JMbayes2))

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
NSEED   <- .envi("BENCH_SEEDS", 2L)
CHAINS  <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP  <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES <- .envi("BENCH_JX_SAMPLES", 1000L)

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("test_orth_k4_%s", format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)
options(error = function() { try(while (sink.number() > 0) sink(), silent = TRUE) })

data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id))
pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2$status2 <- as.integer(pbc2$status != "alive")
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12; pbc2.id$years2 <- pbc2.id$years / 12

dl <- pbc2[, c("id", "year2", "serBilir", "age", "sex", "drug")]
dl$log_serBilir <- log(dl$serBilir)
dl <- dl[stats::complete.cases(dl), ]
ds <- pbc2.id[, c("id", "years2", "status2", "drug")]
ds <- ds[ds$id %in% unique(dl$id), ]; dl <- dl[dl$id %in% ds$id, ]
FORM <- log_serBilir ~ year2 + age + sex + drug

sub <- dl[!duplicated(dl$id), ]; sub <- sub[order(sub$id), ]
N_SUB <- nrow(sub); IDX <- match(dl$id, sub$id)

# The design exactly as model.matrix builds it, then the subject-constant
# columns identified the same way the backend does - by checking variation
# WITHIN subject rather than by assuming which ones they are.
X <- stats::model.matrix(FORM, data = dl)
const <- vapply(seq_len(ncol(X)), function(j)
  all(tapply(X[, j], dl$id, function(v) diff(range(v))) < 1e-8), logical(1))
S <- X[!duplicated(dl$id), , drop = FALSE][order(sub$id), const, drop = FALSE]
cat("=============================================================\n")
cat(sprintf("  k=4 correctness | subjects %d | design cols %d\n", N_SUB, ncol(X)))
cat(sprintf("  subject-constant: %s\n", paste(colnames(X)[const], collapse = ", ")))
cat(sprintf("  within-subject  : %s\n", paste(colnames(X)[!const], collapse = ", ")))
cat("=============================================================\n")
K <- which(const)

as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
bcol <- function(ps, j) {
  b <- ps[["b"]]
  t(vapply(b, function(d) {
    if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
    else { a <- as.matrix(d); as.numeric(a[, j]) }
  }, numeric(N_SUB)))
}
ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE, demean = TRUE)$acf[, 1, 1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) { pr <- ac[k] + ac[k + 1L]; if (pr <= 0) break
    s <- s + pr; k <- k + 2L }
  tau <- -1 + 2 * s; if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}
zdiff <- function(xa, xd) {
  mc <- sqrt(stats::var(xa) / ess_geyer(xa) + stats::var(xd) / ess_geyer(xd))
  if (!is.finite(mc) || mc <= 0) return(NA_real_)
  (mean(xd) - mean(xa)) / mc
}

lme_fit <- lme(FORM, random = ~ year2 | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(years2, status2) ~ drug, data = ds)
run <- function(seed, orth) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              progress_bar = FALSE, seed = seed)
  if (orth) ctl$orthogonalize_b0 <- TRUE
  jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "year2",
                method = "spline-PH-mcmc", control = ctl)
}
# c = (S'S)^-1 S' b_0, per draw. The k-GENERAL projection.
decomp <- function(fit) {
  ps <- fit$posterior_samples
  B  <- as_draw_matrix(ps[["beta"]]); B0 <- bcol(ps, 1L); B1 <- bcol(ps, 2L)
  stopifnot(ncol(B) == ncol(X))
  C <- B0 %*% t(solve(crossprod(S), t(S)))       # [draws, k]
  list(beta = B, C = C,
       inv = B[, K, drop = FALSE] + C,           # beta_S + c, componentwise
       mu = as.numeric(X %*% colMeans(B)) +
            colMeans(B0)[IDX] + colMeans(B1)[IDX] * dl$year2)
}

for (s in seq_len(NSEED)) {
  cat(sprintf("\n################ seed %d ################\n", s))
  A  <- decomp(run(s, FALSE))
  A2 <- decomp(run(s + 100L, FALSE))   # the null: same arm, different seed
  D  <- decomp(run(s, TRUE))

  cat("\n  c = projection of b_0 onto the subject-constant columns\n")
  cat(sprintf("    arm A  %s\n", paste(sprintf("%.5f", colMeans(A$C)), collapse = "  ")))
  cat(sprintf("    arm D  %s   (all ~0 if the constraint bites)\n",
              paste(sprintf("%.5f", colMeans(D$C)), collapse = "  ")))

  cat("\n  CHECK 1: beta_S + c, the k-general identified vector\n")
  cat(sprintf("    %-16s %11s %11s %9s %9s\n",
              "column", "arm A", "arm D", "z(A,A')", "z(A,D)"))
  for (i in seq_along(K)) {
    cat(sprintf("    %-16s %11.5f %11.5f %9.2f %9.2f\n",
                colnames(X)[K[i]], mean(A$inv[, i]), mean(D$inv[, i]),
                zdiff(A$inv[, i], A2$inv[, i]), zdiff(A$inv[, i], D$inv[, i])))
  }
  cat("    for contrast, the raw coefficients (expected to MOVE):\n")
  for (i in seq_along(K)) {
    cat(sprintf("    %-16s %11.5f %11.5f %9s %9.2f\n",
                paste0(colnames(X)[K[i]], " (raw)"),
                mean(A$beta[, K[i]]), mean(D$beta[, K[i]]), "",
                zdiff(A$beta[, K[i]], D$beta[, K[i]])))
  }

  cat("\n  CHECK 2: fitted values against a re-seeded null (no algebra)\n")
  dn <- A2$mu - A$mu; dt <- D$mu - A$mu
  cat(sprintf("    %-14s %14s %14s\n", "", "A vs A' (null)", "A vs D"))
  cat(sprintf("    %-14s %14.3e %14.3e\n", "mean|diff|", mean(abs(dn)), mean(abs(dt))))
  cat(sprintf("    %-14s %14.3e %14.3e\n", "max|diff|", max(abs(dn)), max(abs(dt))))
  cat(sprintf("    ratio treatment/null:  mean %.2fx   max %.2fx\n",
              mean(abs(dt)) / mean(abs(dn)), max(abs(dt)) / max(abs(dn))))
  rm <- mean(abs(dt)) / mean(abs(dn))
  cat(sprintf("    -> %s\n", if (rm <= 2)
    "within the null: the model is unchanged at k=4." else
    "BEYOND the null: the reparameterization changes the model. Option dies."))
}

cat("\n==================== READ ====================\n")
cat("  If check 1's z(A,D) column is now comparable to its z(A,A') null and\n")
cat("  check 2 sits near 1x, then the k=4 failure was the replication\n")
cat("  script projecting onto 2 of 4 subject-constant columns - a bug in the\n")
cat("  TEST, and the third such bug in this sequence. The fix then belongs in\n")
cat("  dev/replicate_orthogonalize_b0.R, which must adopt the k-general\n")
cat("  projection above AND restore the fitted-value null I removed from it.\n")
cat("  If check 2 is large, the implementation is wrong for k > 2 and the\n")
cat("  measured 4.91x min ESS/sec at k=4 is bought by changing the model.\n")
sink(); close(.con)
