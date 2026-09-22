# ==============================================================================
# What is beta_0 ACTUALLY coupled to?
#
# v1 of this diagnostic tested section 6.1's hypothesis - that beta_0 is
# degenerate with mean(b_i0) - and REFUTED it for this model:
#
#     corr(beta_0, mean b_0)   -0.178 (n=312) / -0.204 (n=624)
#     anisotropy                1.08 / 1.09      (a ridge would be >> 1)
#     ESS(sum)/ESS(beta_0)      1.02x / 0.98x    (a ridge would be >> 1)
#
# Section 6.1 measured -0.9510 on SIMULATED data with no covariates and
# alpha = 0.6. PBC2 has covariates, standardize_covariates active (which
# already broke the beta_0/age coupling it reports as -0.98), and
# alpha = 1.25. The mechanism does not transfer.
#
# The arithmetic also rules it out independently: SD(beta_0) = 0.264 while
# SD(mean b_0) = 0.056, so the random-intercept mean accounts for at most a
# fifth of beta_0's width. In section 6's case the two were equal (0.037 vs
# 0.036), which is what a location degeneracy actually looks like.
#
# SO THIS SCRIPT DOES NOT TEST ANOTHER GUESS. Naming a mechanism and then
# measuring it is how v1's hypothesis survived as long as it did. Instead
# it takes the whole population-parameter correlation structure and asks
# the data which direction mixes worst:
#
#   1. correlations of beta_0 with every other population parameter
#   2. an eigendecomposition of the correlation matrix, then ESS of the
#      draws PROJECTED onto each eigenvector
#
# The worst-mixing eigenvector, and the parameters loading on it, is the
# bottleneck - whatever it turns out to be. If beta_0 loads alone on it,
# the problem is beta_0 itself and not a coupling at all.
#
#   Rscript dev/diagnose_beta0_v2.R
#   BENCH_PBC2_DUP=2 Rscript dev/diagnose_beta0_v2.R
# ==============================================================================

suppressPackageStartupMessages({library(jmjax); library(nlme); library(survival)})
if (!requireNamespace("JMbayes2", quietly = TRUE)) stop("need JMbayes2 for pbc2")
suppressPackageStartupMessages(library(JMbayes2))

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
DUP <- .envi("BENCH_PBC2_DUP", 1L)
OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("diagnose_beta0_v2_dup%d_%s", DUP,
                                 format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)

data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id)); pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2$status2 <- as.integer(pbc2$status != "alive")
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year/12; pbc2.id$years2 <- pbc2.id$years/12
data_long <- pbc2[, c("id","year2","serBilir","age")]
data_long$log_serBilir <- log(data_long$serBilir)
data_surv <- pbc2.id[, c("id","years2","status2","drug")]
if (DUP > 1L) {
  bm <- max(data_surv$id); dl <- list(); ds <- list()
  for (k in seq_len(DUP)) {
    a <- data_long; a$id <- a$id + (k-1L)*bm
    b <- data_surv; b$id <- b$id + (k-1L)*bm
    dl[[k]] <- a; ds[[k]] <- b
  }
  data_long <- do.call(rbind, dl); data_surv <- do.call(rbind, ds)
}

ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n-1L, 2000L), plot = FALSE, demean = TRUE)$acf[,1,1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) {
    p <- ac[k] + ac[k+1L]; if (p <= 0) break
    s <- s + p; k <- k + 2L
  }
  tau <- -1 + 2*s; if (!is.finite(tau) || tau < 1) tau <- 1
  n/tau
}

cat("=============================================================\n")
cat(sprintf("  beta_0 coupling structure | subjects %d | dup x%d\n",
            length(unique(data_long$id)), DUP))
cat("=============================================================\n\nfitting...\n")
lme_fit <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id, data = data_long,
               control = lmeControl(opt="optim", msMaxIter=200, niterEM=100))
cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)
fit <- jm_fit_prefit(lme_fit, cox_fit, data_surv = data_surv, time_var = "year2",
  method = "spline-PH-mcmc",
  control = list(n_interior_knots = 5, spline_prior = "penalized",
                 rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                 num_warmup = 1000, num_samples = 1000, num_chains = 4,
                 progress_bar = FALSE))
cat(sprintf("  max R-hat %.4f\n", max(unlist(fit$diagnostics$rhat), na.rm=TRUE)))

# ---- assemble a draws matrix over POPULATION parameters ---------------------
ps <- fit$posterior_samples
cols <- list()
addm <- function(nm, m) {
  m <- as.matrix(m)
  if (ncol(m) == 1L) cols[[nm]] <<- m[,1]
  else for (j in seq_len(ncol(m))) cols[[sprintf("%s%d", nm, j-1L)]] <<- m[,j]
}
for (site in c("beta","W","sigma_b","gamma")) {
  v <- ps[[site]]; if (is.null(v)) next
  if (is.list(v)) v <- do.call(rbind, lapply(v, unlist))
  addm(site, v)
}
for (site in c("alpha","sigma_e","rho","tau_w","log_lambda0","shape")) {
  v <- ps[[site]]; if (is.null(v)) next
  cols[[site]] <- as.numeric(unlist(v))
}
# mean of the random intercepts, kept in as a candidate like any other
bb <- ps[["b"]]
if (!is.null(bb)) {
  a <- try(as.array(bb), silent = TRUE)
  if (!inherits(a,"try-error") && length(dim(a)) == 3L)
    cols[["mean_b0"]] <- apply(a[,,1,drop=FALSE], 1, mean)
}
n_draw <- min(vapply(cols, length, integer(1)))
X <- do.call(cbind, lapply(cols, function(z) z[seq_len(n_draw)]))
keep <- apply(X, 2, function(z) is.finite(var(z)) && var(z) > 1e-14)
X <- X[, keep, drop = FALSE]
cat(sprintf("  %d draws x %d population parameters\n", nrow(X), ncol(X)))

# ---- 1. what is beta_0 correlated with? -------------------------------------
stopifnot("beta0" %in% colnames(X))
r <- stats::cor(X)[, "beta0"]
r <- r[names(r) != "beta0"]
o <- order(abs(r), decreasing = TRUE)
cat("\n---------------- beta_0's strongest couplings ----------------\n")
for (i in head(o, 8)) cat(sprintf("  %-14s %8.4f\n", names(r)[i], r[i]))

# ---- 2. which DIRECTION mixes worst? ----------------------------------------
# Eigenvectors of the correlation matrix, ESS of the draws projected onto
# each. This does not assume which parameters are involved.
Z <- scale(X)
eg <- eigen(stats::cor(X), symmetric = TRUE)
proj <- Z %*% eg$vectors
e <- vapply(seq_len(ncol(proj)), function(j) ess_geyer(proj[,j]), numeric(1))
ord <- order(e)
cat("\n---------------- worst-mixing directions ----------------\n")
cat(sprintf("  %-5s %9s %9s   %s\n", "dir", "ESS", "eigval", "top loadings"))
for (j in head(ord, 4)) {
  ld <- eg$vectors[, j]; names(ld) <- colnames(X)
  top <- head(sort(abs(ld), decreasing = TRUE), 4)
  cat(sprintf("  %-5d %9.1f %9.4f   %s\n", j, e[j], eg$values[j],
              paste(sprintf("%s(%+.2f)", names(top), ld[names(top)]), collapse=" ")))
}
cat("\n  per-parameter ESS, worst 8:\n")
ep <- vapply(seq_len(ncol(X)), function(j) ess_geyer(X[,j]), numeric(1))
names(ep) <- colnames(X)
for (nm in names(head(sort(ep), 8))) cat(sprintf("    %-14s %8.1f\n", nm, ep[nm]))

cat("\n  Read: if the worst direction loads mostly on ONE parameter, that\n")
cat("  parameter is intrinsically slow and no reparameterisation helps.\n")
cat("  If it loads on several, that combination is the ridge to remove.\n")

saveRDS(list(dup=DUP, cor_beta0=r, eig_ess=e, eig_values=eg$values,
             loadings=eg$vectors, params=colnames(X), per_param_ess=ep),
        paste0(OUT,".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep="")
sink(); close(.con)
