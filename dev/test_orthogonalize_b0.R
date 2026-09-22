# ==============================================================================
# Does orthogonalize_b0 recover the 6.6x / 7.6x that v4 measured as available,
# and does it leave the inference unchanged?
#
# The second question is the one that decides whether the option ships. The
# transform is model-preserving in the sense that every direction swept out
# of b_0 lies in a column space beta already spans, so fitted values are
# reachable either way. But the implied prior on the constrained b_0 is
# SINGULAR while sigma_b still governs the unconstrained draws, and the
# swept directions of those draws are unidentified by the likelihood. That
# could move sigma_b0's posterior. Asserting it does not would be exactly
# the kind of claim this package has had to withdraw before, so it is
# measured here instead.
#
# PRE-REGISTERED READS, so the result cannot be rationalized afterwards:
#
#   ADOPT-WORTHY   min ESS rises materially AND every estimate agrees with
#                  the default arm inside Monte Carlo error (|diff| < 3 MCSE
#                  on the pooled scale), sigma_b0 included.
#
#   MECHANISM OK,  beta_0 and beta_2 both rise but some OTHER parameter
#   FIX INCOMPLETE becomes the new minimum. Worth recording, not shipping
#                  as a default.
#
#   REJECT         any estimate moves beyond Monte Carlo error - in
#                  particular sigma_b0 biased low, the failure mode the
#                  singular prior predicts - or min ESS does not improve.
#
# Note wishart_gibbs_centered already failed the second read in the v4 run:
# beta_0 441 -> 1490 but beta_2 526 -> 291, so min ESS went 442 -> 291.
#
#   Rscript dev/test_orthogonalize_b0.R
#   BENCH_PBC2_DUP=2 Rscript dev/test_orthogonalize_b0.R
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
OUT <- file.path(OUTDIR, sprintf("test_orth_b0_dup%d_%s", DUP,
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

cat("=============================================================\n")
cat(sprintf("  orthogonalize_b0 | subjects %d | dup x%d\n",
            length(unique(data_long$id)), DUP))
cat(sprintf("  chains %d | warmup %d | samples %d\n", CHAINS, WARMUP, SAMPLES))
cat("=============================================================\n")

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

summarise <- function(fit, label, elapsed) {
  ess  <- unlist(fit$diagnostics$ess)
  rhat <- unlist(fit$diagnostics$rhat)
  # Frailty-type slots with ESS 0 are not real parameters; drop by VALUE,
  # not by name, so the minimum is not silently an artifact.
  ess_ok <- ess[is.finite(ess) & ess > 0]
  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  %.1fs  max R-hat %.4f  leapfrog %s\n", elapsed,
              max(rhat, na.rm = TRUE),
              format(fit$convergence$mean_num_steps %||% NA)))
  cat(sprintf("  min ESS %8.1f  (%s)\n", min(ess_ok),
              names(ess_ok)[which.min(ess_ok)]))
  list(ess = ess, ess_ok = ess_ok, rhat = rhat,
       est = unlist(fit$estimates), se = unlist(fit$se),
       elapsed = elapsed,
       steps = fit$convergence$mean_num_steps %||% NA_real_)
}

tA <- system.time(fitA <- run_arm())
A <- summarise(fitA, "A. defaults", tA[["elapsed"]])

tD <- system.time(fitD <- run_arm(list(orthogonalize_b0 = TRUE)))
D <- summarise(fitD, "D. orthogonalize_b0 = TRUE", tD[["elapsed"]])

# ---- 1. EFFICIENCY -------------------------------------------------------
cat("\n---------------- ESS, arm by arm ----------------\n")
nm <- intersect(names(A$ess), names(D$ess))
nm <- nm[is.finite(A$ess[nm]) & A$ess[nm] > 0]
ord <- nm[order(A$ess[nm])]
cat(sprintf("  %-14s %10s %10s %8s\n", "parameter", "default", "orth", "ratio"))
for (k in ord) {
  cat(sprintf("  %-14s %10.1f %10.1f %8.2fx\n",
              k, A$ess[[k]], D$ess[[k]], D$ess[[k]] / A$ess[[k]]))
}
cat(sprintf("\n  min ESS      %10.1f %10.1f %8.2fx\n",
            min(A$ess_ok), min(D$ess_ok), min(D$ess_ok) / min(A$ess_ok)))
cat(sprintf("  min ESS/sec  %10.2f %10.2f %8.2fx\n",
            min(A$ess_ok) / A$elapsed, min(D$ess_ok) / D$elapsed,
            (min(D$ess_ok) / D$elapsed) / (min(A$ess_ok) / A$elapsed)))
cat(sprintf("  leapfrog     %10s %10s\n", format(A$steps), format(D$steps)))

# ---- 2. CORRECTNESS ------------------------------------------------------
# Two independent MCMC runs never agree exactly. The question is whether
# they agree to within Monte Carlo error, so the comparison is on the MCSE
# scale rather than against an arbitrary tolerance. MCSE = se / sqrt(ESS),
# pooled across the two arms.
cat("\n---------------- estimates, on the MCSE scale ----------------\n")
cat("  |difference| in units of pooled Monte Carlo standard error.\n")
cat("  Under a correct reparameterization these are ~N(0,1); > 3 is a flag.\n\n")
en <- intersect(names(A$est), names(D$est))
en <- en[en %in% names(A$se) & en %in% names(D$se)]
flag <- character(0)
cat(sprintf("  %-14s %12s %12s %10s %8s\n",
            "parameter", "default", "orth", "diff", "z"))
for (k in en) {
  ea <- A$est[[k]]; ed <- D$est[[k]]
  essA <- if (k %in% names(A$ess)) A$ess[[k]] else NA_real_
  essD <- if (k %in% names(D$ess)) D$ess[[k]] else NA_real_
  mc <- sqrt(ifelse(is.finite(essA) & essA > 0, A$se[[k]]^2 / essA, NA) +
             ifelse(is.finite(essD) & essD > 0, D$se[[k]]^2 / essD, NA))
  z <- if (is.finite(mc) && mc > 0) (ed - ea) / mc else NA_real_
  if (is.finite(z) && abs(z) > 3) flag <- c(flag, k)
  cat(sprintf("  %-14s %12.5f %12.5f %10.5f %8s\n", k, ea, ed, ed - ea,
              if (is.finite(z)) sprintf("%.2f", z) else "-"))
}

cat("\n==================== VERDICT ====================\n")
if (length(flag)) {
  cat("  REJECT as specified: estimate(s) moved beyond Monte Carlo error:\n")
  cat("    ", paste(flag, collapse = ", "), "\n")
  cat("  sigma_b0 among them would be the singular-prior failure mode the\n")
  cat("  implementation note predicts; anything else needs its own\n")
  cat("  explanation before this option goes further.\n")
} else {
  cat("  Estimates agree within Monte Carlo error on every parameter.\n")
  r <- min(D$ess_ok) / min(A$ess_ok)
  if (r > 1.15) {
    cat(sprintf("  min ESS %.2fx - the mechanism converts to the conservative\n", r))
    cat("  metric, not just to the parameter it targets.\n")
  } else {
    cat(sprintf("  min ESS only %.2fx: beta_0/beta_2 may have moved while\n", r))
    cat("  another parameter now binds. Check the per-parameter table above\n")
    cat("  for which one, and note it as an incomplete fix rather than a win.\n")
  }
}
cat("\n  One dataset, one seed. Nothing here is established until it\n")
cat("  replicates - the standard every other option in this package was\n")
cat("  held to, and the one dense_mass_beta failed.\n")

saveRDS(list(dup = DUP, A = A, D = D, flagged = flag), paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
