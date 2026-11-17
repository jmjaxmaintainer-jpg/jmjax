# ==============================================================================
# dev/bench_multi_sim.R - jmjax 0.5.0 multi-marker simulations (Milestone 1,
# Phase C): K = 2 and K = 3 markers, sparse and dense visits, rotated vs
# unrotated (JMbayes2 optional). Shared code: dev/bench_multi_common.R.
#
# DATA-GENERATING MODEL (n subjects, follow-up to 5 time units):
#   marker k: y_k(t) = (b_k0 + u_k0) + (b_k1 + u_k1) t [+ 0.3 x_i for k = 1] + e
#   u_i ~ N(0, D), D = diag(sd) R diag(sd) with a FULL correlation R
#   (within-marker 0.2, cross-marker intercepts 0.4, cross-marker slopes 0.3,
#   other cross pairs 0.1), so the markers are linked.
#   hazard: 1.2 t^0.2 exp(-2.2 - 0.3 w_i + sum_k alpha_k m_k(t)), w_i ~ Bern(0.5)
#   visits: dense  every 0.5 (marker k offset by 0.1 (k - 1))
#           sparse every 1.5 (marker k offset by 0.4 (k - 1)), +-0.1 jitter
#   so the markers are observed at different times in both designs.
#   Censoring: administrative at 5 plus uniform(2, 10).
#
# CONFIGS: K2_dense, K2_sparse (q = 4), K3_dense, K3_sparse (q = 6).
# ARMS:    A jmjax unrotated | R jmjax rotated + dense block | J JMbayes2 (opt-in)
#
# OUTPUT (in BM_OUTDIR, default ~/Documents/R/jmjax_results):
#   multi_sim.csv, multi_sim_draws/*.rds, multi_sim_summary.csv, multi_sim_tables.md
#   multi_sim_recovery.csv  (bias and 95% interval coverage per parameter)
#
# USAGE (resumable):
#   caffeinate -i Rscript dev/bench_multi_sim.R                 # A and R, 3 replicates
#   BM_ARMS=A,R,J Rscript dev/bench_multi_sim.R                 # with JMbayes2
#   BM_CONFIGS=K2_sparse BM_SEEDS=1 Rscript dev/bench_multi_sim.R   # quick look
# Defaults: n = 300, 3 replicates, 4 chains x (1000 + 1000).
# ==============================================================================

local({
  here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
  if (is.null(here)) {
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "dev"
  }
  source(file.path(here, "bench_multi_common.R"), local = globalenv())
})

CONFIGS <- .envl("BM_CONFIGS", c("K2_dense", "K2_sparse", "K3_dense", "K3_sparse"))
ARMS    <- .envl("BM_ARMS", c("A", "R"))
NSEED   <- .envi("BM_SEEDS", 3L)
NSUB    <- .envi("BM_N", 300L)
CHAINS  <- .envi("BM_CHAINS", 4L)
WARMUP  <- .envi("BM_WARMUP", 1000L)
SAMPLES <- .envi("BM_SAMPLES", 1000L)
JB_ITER <- .envi("BM_JB_ITER", 16000L)
JB_BURN <- .envi("BM_JB_BURN", 4000L)
OUTDIR  <- Sys.getenv("BM_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
DRAWDIR <- file.path(OUTDIR, "multi_sim_draws")
dir.create(DRAWDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, "multi_sim.csv")
if ("J" %in% ARMS && !requireNamespace("JMbayes2", quietly = TRUE))
  stop("arm J needs JMbayes2", call. = FALSE)

# ---- truth -------------------------------------------------------------------------
TRUTH <- list(
  beta = list(c(2.0, 0.4, 0.3), c(-1.0, 0.2), c(0.5, -0.3)),
  sd_b = list(c(0.7, 0.2), c(0.6, 0.15), c(0.5, 0.1)),
  sigma_e = c(0.3, 0.5, 0.4), alpha = c(0.5, -0.4, 0.3),
  gamma = -0.3, shape = 1.2, log_lambda0 = -2.2)
corr_K <- function(K) {
  Q <- 2L * K; R <- diag(Q)
  for (i in 1:Q) for (j in 1:Q) if (i != j) {
    ki <- (i - 1) %/% 2; kj <- (j - 1) %/% 2; ti <- (i - 1) %% 2; tj <- (j - 1) %% 2
    R[i, j] <- if (ki == kj) 0.2 else if (ti == 0 && tj == 0) 0.4 else if (ti == 1 && tj == 1) 0.3 else 0.1
  }
  stopifnot(min(eigen(R, symmetric = TRUE, only.values = TRUE)$values) > 0)
  R
}

simulate_K <- function(K, design, n, seed, max_time = 5) {
  set.seed(seed)
  mk <- paste0("y", seq_len(K))
  sd <- unlist(TRUTH$sd_b[seq_len(K)]); R <- corr_K(K)
  D <- diag(sd) %*% R %*% diag(sd)
  u <- matrix(stats::rnorm(n * 2 * K), n) %*% chol(D)
  x <- stats::rnorm(n); w <- stats::rbinom(n, 1, 0.5)
  mfun <- function(k, t, i) {
    b <- TRUTH$beta[[k]]
    (b[1] + u[i, 2 * k - 1]) + (b[2] + u[i, 2 * k]) * t + if (k == 1) b[3] * x[i] else 0
  }
  H <- function(t, i) {
    if (t <= 0) return(0)
    stats::integrate(function(s) {
      eta <- TRUTH$log_lambda0 + TRUTH$gamma * w[i]
      for (k in seq_len(K)) eta <- eta + TRUTH$alpha[k] * mfun(k, s, i)
      TRUTH$shape * s^(TRUTH$shape - 1) * exp(eta)
    }, 0, t)$value
  }
  Tt <- vapply(seq_len(n), function(i) {
    e <- -log(stats::runif(1))
    if (H(max_time * 4, i) < e) return(Inf)
    stats::uniroot(function(t) H(t, i) - e, c(1e-6, max_time * 4))$root
  }, numeric(1))
  cens <- pmin(max_time, stats::runif(n, 2, 10))
  obs <- pmin(Tt, cens); event <- as.integer(Tt <= cens)
  by <- if (design == "dense") 0.5 else 1.5
  dls <- lapply(seq_len(K), function(k) {
    off <- if (design == "dense") 0.1 * (k - 1) else 0.4 * (k - 1)
    do.call(rbind, lapply(seq_len(n), function(i) {
      t <- seq(off, max_time, by = by)
      if (design == "sparse") t <- pmax(0, t + stats::runif(length(t), -0.1, 0.1))
      t <- t[t <= obs[i]]
      if (k == 1 && !length(t)) t <- 0     # marker 1 carries x: every subject needs a row
      if (!length(t)) return(NULL)
      data.frame(id = i, time = t, x = x[i],
                 y = mfun(k, t, i) + stats::rnorm(length(t), 0, TRUTH$sigma_e[k]))
    }))
  })
  names(dls) <- mk
  forms <- c(list(y1 = y ~ time + x), stats::setNames(rep(list(y ~ time), K - 1L), mk[-1]))
  truth <- c(
    unlist(lapply(seq_len(K), function(k) {
      b <- TRUTH$beta[[k]][if (k == 1) 1:3 else 1:2]
      stats::setNames(b, sprintf("beta_%s_%d", mk[k], seq_along(b) - 1L)) })),
    stats::setNames(TRUTH$alpha[seq_len(K)], paste0("alpha_", mk)),
    stats::setNames(TRUTH$sigma_e[seq_len(K)], paste0("sigma_e_", mk)))
  Q <- 2L * K
  for (j in seq_len(Q)) for (i in j:Q) truth[sprintf("D[%d, %d]", i, j)] <- D[i, j]
  list(markers = mk, forms = forms, rforms = stats::setNames(rep(list(~ time), K), mk),
       dls = dls, ds = data.frame(id = seq_len(n), time = obs, event = event, w = w),
       truth = truth, q = Q, block_sizes = rep(2L, K), event_rate = mean(event))
}

# ---- run ----------------------------------------------------------------------------
surv_formula <- Surv(time, event) ~ w
done <- read_done(CSV)
have <- function(cf, s, a) !is.null(done) && any(done$config == cf & done$seed == s & done$arm == a)
for (cf in CONFIGS) {
  K <- as.integer(sub("^K([0-9]+)_.*$", "\\1", cf)); design <- sub("^K[0-9]+_", "", cf)
  comp <- list()
  for (s in seq_len(NSEED)) {
    if (all(vapply(ARMS, function(a) have(cf, s, a), logical(1)))) next
    sim <- simulate_K(K, design, NSUB, seed = 1000L * K + 10L * (design == "dense") + s)
    cat(sprintf("\n== %s replicate %d: %d subjects, %.0f%% events, rows per marker %s\n",
                cf, s, NSUB, 100 * sim$event_rate,
                paste(vapply(sim$dls, nrow, 1L), collapse = "/")))
    for (a in ARMS) {
      if (have(cf, s, a)) next
      if (a != "J" && is.null(comp[[a]])) {
        comp[[a]] <- compile_time_jx(sim$forms, sim$rforms, sim$dls, sim$ds, surv_formula,
                                     "time", a, CHAINS)
        cat(sprintf("   compilation estimate %s: %.1fs\n", a, comp[[a]]))
      }
      obj <- tryCatch({
        if (a == "J") {
          draws_jb(fit_jb(sim$forms, sim$rforms, sim$dls, sim$ds, surv_formula, "time", s,
                          CHAINS, JB_ITER, JB_BURN), sim$markers, sim$block_sizes)
        } else {
          f <- fit_jx(sim$forms, sim$rforms, sim$dls, sim$ds, surv_formula, "time", a, s,
                      CHAINS, WARMUP, SAMPLES)
          if (a == "R" && !isTRUE(f$convergence$orthogonalize$rotation$applied))
            stop("arm R: rotation not applied", call. = FALSE)
          draws_jx(f, sim$markers, sim$block_sizes)
        }
      }, error = function(e) {
        if (grepl("inspect|expected|rotation not applied", conditionMessage(e))) stop(e)
        cat("   ", cf, a, "replicate", s, "FAILED:", conditionMessage(e), "\n"); NULL })
      if (is.null(obj)) next
      saveRDS(c(obj, list(arm = a, config = cf, seed = s, truth = sim$truth)),
              file.path(DRAWDIR, sprintf("%s_%s_seed%d.rds", cf, a, s)))
      rows <- ess_rows(obj, "sim", cf, sim$q, s, a, truth = sim$truth,
                       sec_compile = if (a == "J") NA_real_ else comp[[a]])
      append_rows(rows, CSV)
      cat(sprintf("   %s rep %d %s: %7.1fs | min ESS/s beta %.2f alpha %.2f Sigma %.2f | max R-hat %.3f%s\n",
                  cf, s, a, obj$sec,
                  min(rows$ess_bulk[rows$block == "beta"]) / obj$sec,
                  min(rows$ess_bulk[rows$block == "alpha"]) / obj$sec,
                  min(rows$ess_bulk[rows$block == "Sigma"]) / obj$sec,
                  max(rows$rhat_posterior),
                  if (is.finite(obj$n_div)) sprintf(" | %d divergences", as.integer(obj$n_div)) else ""))
    }
  }
}

# ---- tables -------------------------------------------------------------------------
R <- read_done(CSV)
S <- fit_summary(R)
utils::write.csv(S, file.path(OUTDIR, "multi_sim_summary.csv"), row.names = FALSE)
tables_md(S, "Simulations, several markers: rotated vs unrotated",
          file.path(OUTDIR, "multi_sim_tables.md"))

# Recovery: bias and coverage of the central 95% interval (normal approximation
# from the posterior mean and SD), per config x arm x parameter block.
RR <- R[is.finite(R$truth), ]
RR$cover <- abs(RR$est - RR$truth) <= 1.96 * RR$sd
RR$bias <- RR$est - RR$truth
rec <- stats::aggregate(cbind(bias = abs(bias), cover) ~ config + arm + block, data = RR, FUN = mean)
names(rec)[names(rec) == "bias"] <- "mean_abs_bias"
utils::write.csv(rec, file.path(OUTDIR, "multi_sim_recovery.csv"), row.names = FALSE)
cat("\nRecovery (mean |bias|, share of 95% intervals covering the truth):\n")
print(rec, row.names = FALSE, digits = 3)
