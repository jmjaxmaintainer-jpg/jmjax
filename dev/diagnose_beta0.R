# ==============================================================================
# The beta_0 bottleneck: is the location degeneracy the WHOLE story?
#
# beta_0 has the lowest ESS of any population parameter in every jmjax fit
# measured. On PBC2 it is ~442 against alpha's ~5100 - an 11x spread - and
# today's duplication test showed it DEGRADING with sample size (441.7 at
# n=312, 386.1 at n=624) while everything else held steady. It is the
# binding constraint on every performance claim this package makes.
#
# Section 6.1 of inst/doc-notes/mcmc-performance-investigation.md diagnosed
# the mechanism on simulated data: corr(beta_0, mean(b_i0)) = -0.9510 at
# n=500. The likelihood constrains only beta_0 + mean(b_i0); the split
# between them is pinned solely by the prior, at scale sigma_b0/sqrt(n).
#
# WHAT THIS SCRIPT ADDS. Section 6 established the correlation. It never
# measured the thing that decides whether a fix is worth building:
#
#     ESS(beta_0 + mean(b_i0))   vs   ESS(beta_0)
#
# The sum is the direction the data actually identify. If it mixes far
# better than beta_0 alone, then the degeneracy accounts for essentially
# all of the loss, and a reparameterisation that SAMPLES the sum would
# recover that factor. If the sum mixes no better, the problem is
# elsewhere and a reparameterisation would be wasted work.
#
# That ratio is the number worth having before writing any code, and it is
# cheap: it comes out of posterior samples already returned by a normal fit.
#
# It also measures the ANISOTROPY directly - SD along the identified
# direction against SD along the ridge. A diagonal mass matrix cannot
# represent a ridge that is diagonal in (beta_0, b_0) space, so that ratio
# is what NUTS is fighting, and the prediction is that it worsens with n.
#
#   Rscript dev/diagnose_beta0.R              # n = 312
#   BENCH_PBC2_DUP=2 Rscript dev/diagnose_beta0.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 supplies the pbc2 data.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

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
OUT <- file.path(OUTDIR, sprintf("diagnose_beta0_dup%d_%s", DUP,
                                 format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)

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

cat("=============================================================\n")
cat(sprintf("  beta_0 degeneracy diagnostic | subjects %d | dup x%d\n",
            length(unique(data_long$id)), DUP))
cat(sprintf("  chains %d | warmup %d | samples %d\n", CHAINS, WARMUP, SAMPLES))
cat("=============================================================\n")

# ---- an ESS estimator, so the comparison is like-for-like ---------------
# Geyer's initial positive sequence, the estimator Stan and NumPyro both
# build on. Applied to the SAME concatenated draws for every quantity, so
# the RATIO between quantities is the meaningful output even though the
# absolute values are approximate: with num_chains > 1 the draws come back
# concatenated, so chain boundaries add a little spurious autocorrelation.
# That contaminates every series identically and cannot manufacture a
# difference between beta_0 and the sum.
ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE,
                   demean = TRUE)$acf[, 1, 1]
  # pair consecutive lags; stop at the first non-positive pair sum
  s <- 0
  k <- 1L
  while (k + 1L <= length(ac)) {
    pair <- ac[k] + ac[k + 1L]
    if (pair <= 0) break
    s <- s + pair
    k <- k + 2L
  }
  tau <- -1 + 2 * s          # integrated autocorrelation time
  if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}

# ---- fit -----------------------------------------------------------------
cat("\nfitting...\n")
lme_fit <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id,
               data = data_long,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)

tt <- system.time(fit <- jm_fit_prefit(
  lme_fit, cox_fit, data_surv = data_surv, time_var = "year2",
  method = "spline-PH-mcmc",
  control = list(n_interior_knots = 5, spline_prior = "penalized",
                 rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                 num_warmup = WARMUP, num_samples = SAMPLES,
                 num_chains = CHAINS, progress_bar = FALSE)))
cat(sprintf("  %.1fs  max R-hat %.4f\n", tt[["elapsed"]],
            max(unlist(fit$diagnostics$rhat), na.rm = TRUE)))

ps <- fit$posterior_samples
if (is.null(ps)) stop("no posterior_samples returned", call. = FALSE)

# beta is [draws, p]; beta_0 is the intercept column.
beta <- ps[["beta"]]
if (is.list(beta)) beta <- do.call(rbind, lapply(beta, unlist))
beta <- as.matrix(beta)
b0_pop <- beta[, 1]

# The random intercepts. Prefer the deterministic "b" site; fall back to
# reconstructing from b_std if only the non-centred site is returned.
get_b0 <- function(ps) {
  if (!is.null(ps[["b"]])) {
    b <- ps[["b"]]
    if (is.list(b)) {
      # [draws][subject][q] or [draws][subject]
      return(vapply(b, function(d) {
        m <- if (is.list(d)) vapply(d, function(r) as.numeric(r)[1], numeric(1))
             else as.numeric(d)
        mean(m)
      }, numeric(1)))
    }
    a <- as.array(b)
    if (length(dim(a)) == 3L) return(apply(a[, , 1, drop = FALSE], 1, mean))
    if (length(dim(a)) == 2L) return(apply(a, 1, mean))
  }
  NULL
}
mean_b0 <- get_b0(ps)
if (is.null(mean_b0)) {
  stop("could not extract per-draw random intercepts from posterior_samples; ",
       "sites present: ", paste(names(ps), collapse = ", "), call. = FALSE)
}
stopifnot(length(mean_b0) == length(b0_pop))

# ---- the measurement -----------------------------------------------------
sum_dir  <- b0_pop + mean_b0     # what the data identify
ridge    <- b0_pop - mean_b0     # the degenerate direction
rho      <- stats::cor(b0_pop, mean_b0)

e_b0   <- ess_geyer(b0_pop)
e_mb0  <- ess_geyer(mean_b0)
e_sum  <- ess_geyer(sum_dir)
e_ridge<- ess_geyer(ridge)

cat("\n---------------- degeneracy ----------------\n")
cat(sprintf("  corr(beta_0, mean(b_i0))      %8.4f   (section 6.1 got -0.9510 at n=500)\n", rho))
cat(sprintf("  SD(beta_0)                    %8.5f\n", sd(b0_pop)))
cat(sprintf("  SD(mean(b_i0))                %8.5f\n", sd(mean_b0)))
cat(sprintf("  SD(beta_0 + mean(b_i0))       %8.5f   <- identified by data\n", sd(sum_dir)))
cat(sprintf("  SD(beta_0 - mean(b_i0))       %8.5f   <- the ridge\n", sd(ridge)))
cat(sprintf("  anisotropy ridge/identified   %8.2f   <- what a DIAGONAL mass matrix cannot represent\n",
            sd(ridge) / sd(sum_dir)))

cat("\n---------------- the decisive ratio ----------------\n")
cat(sprintf("  ESS(beta_0)                   %8.1f\n", e_b0))
cat(sprintf("  ESS(mean(b_i0))               %8.1f\n", e_mb0))
cat(sprintf("  ESS(beta_0 + mean(b_i0))      %8.1f\n", e_sum))
cat(sprintf("  ESS(beta_0 - mean(b_i0))      %8.1f\n", e_ridge))
cat(sprintf("\n  RECOVERABLE FACTOR  ESS(sum)/ESS(beta_0) = %.2fx\n", e_sum / e_b0))

cat("\n  How to read this:\n")
cat("    >> 1   the identified direction mixes fine and beta_0 does not, so\n")
cat("           the degeneracy IS the bottleneck and a reparameterisation\n")
cat("           that samples the sum should recover roughly this factor.\n")
cat("    ~= 1   the sum mixes no better; the problem is NOT the degeneracy\n")
cat("           and section 6's whole line of attack is misdirected.\n")

cat(sprintf("\n  For reference, this fit reported beta_0 ESS = %.1f and alpha ESS = %.1f\n",
            as.numeric(unlist(fit$diagnostics$ess)[["beta_0"]]),
            as.numeric(unlist(fit$diagnostics$ess)[["alpha"]])))

saveRDS(list(dup = DUP, rho = rho, sd_beta0 = sd(b0_pop),
             sd_mean_b0 = sd(mean_b0), sd_sum = sd(sum_dir),
             sd_ridge = sd(ridge), ess_beta0 = e_b0, ess_sum = e_sum,
             ess_mean_b0 = e_mb0, ess_ridge = e_ridge,
             reported_ess = unlist(fit$diagnostics$ess),
             elapsed = as.numeric(tt[["elapsed"]])), paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
