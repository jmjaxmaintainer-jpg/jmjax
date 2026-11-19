# ==============================================================================
# dev/bench_m1_sim.R - paper-grade multi-marker simulations (Milestone 1,
# after 0.5.0). Shared code and conventions: dev/bench_m1_common.R (arms
# A R C [J], the R-hat/ESS bar, timing, hardware).
#
# DATA-GENERATING MODEL: as dev/bench_multi_sim.R (0.5.0), unchanged:
#   marker k: y_k(t) = (b_k0 + u_k0) + (b_k1 + u_k1) t [+ 0.3 x_i for k = 1] + e
#   u_i ~ N(0, D), D = diag(sd) R diag(sd), FULL R (within-marker 0.2,
#   cross-marker intercepts 0.4, slopes 0.3, other cross pairs 0.1)
#   hazard 1.2 t^0.2 exp(-3.7 - 0.3 w_i + sum_k alpha_k m_k(t)), w_i ~ Bern(0.5)
#   visits: dense every 0.5 (marker k offset 0.1 (k - 1)); sparse every 1.5
#   (offset 0.4 (k - 1)) +-0.1 jitter, so markers are seen at different times
#   censoring: administrative at 5 and uniform(2, 10); n = 300
# CONFIGS: K2_dense, K2_sparse (q = 4), K3_dense, K3_sparse (q = 6).
#
# TWO PARTS (BM_MODE = efficiency, coverage, or all = both, in that order):
#   efficiency  replicates 1..20 of every design, arms A R C (J with BM_ARMS)
#   coverage    replicates 1..100 of K2_dense and K3_sparse, arm R (the
#               default sampler); replicates 1..20 are shared with the
#               efficiency part. 95% equal-tailed credible intervals from
#               the draws.
# Replicate r of a design uses data seed 1000 K + 10 [dense] + r in both parts
# and MCMC seed r, so the parts share fits.
#
# OUTPUT (BM_OUTDIR, default ~/Documents/R/jmjax_results/m1_paper):
#   hardware.txt, sim.csv, sim_summary.csv, sim_tables.md, sim_coverage.csv
#
# USAGE (resumable):
#   JMJAX_NUM_DEVICES=4 caffeinate -i Rscript dev/bench_m1_sim.R
#   BM_MODE=efficiency BM_CONFIGS=K2_sparse BM_REPS=1 BM_SAMPLES=500 BM_WARMUP=500 \
#     BM_MAX_DOUBLINGS=0 BM_OUTDIR=/tmp/m1_smoke Rscript dev/bench_m1_sim.R   # quick look
# ==============================================================================

local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "dev"
  source(file.path(here, "bench_m1_common.R"), local = globalenv())
})

MODE     <- Sys.getenv("BM_MODE", "all")
CONFIGS  <- .envl("BM_CONFIGS", c("K2_dense", "K2_sparse", "K3_dense", "K3_sparse"))
COV_CFG  <- .envl("BM_COV_CONFIGS", c("K2_dense", "K3_sparse"))
ARMS     <- .envl("BM_ARMS", c("A", "R", "C"))
NREP     <- .envi("BM_REPS", 20L)
NREP_COV <- .envi("BM_REPS_COV", 100L)
NSUB     <- .envi("BM_N", 300L)
OUTDIR   <- Sys.getenv("BM_OUTDIR", path.expand("~/Documents/R/jmjax_results/m1_paper"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, "sim.csv")
if ("J" %in% ARMS && !requireNamespace("JMbayes2", quietly = TRUE))
  stop("arm J needs JMbayes2", call. = FALSE)
write_hardware(OUTDIR)

TRUTH <- list(
  beta = list(c(2.0, 0.4, 0.3), c(-1.0, 0.2), c(0.5, -0.3)),
  sd_b = list(c(0.7, 0.2), c(0.6, 0.15), c(0.5, 0.1)),
  sigma_e = c(0.3, 0.5, 0.4), alpha = c(0.5, -0.4, 0.3),
  gamma = -0.3, shape = 1.2, log_lambda0 = -3.7)
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
       truth = truth, q = Q, block_sizes = rep(2L, K), event_rate = mean(event),
       surv_formula = Surv(time, event) ~ w, time_var = "time")
}

# ---- run ---------------------------------------------------------------------------
done <- read_done(CSV)
have <- function(cf, r, a) !is.null(done) && any(done$config == cf & done$seed == r & done$arm == a)
run_design <- function(cf, reps, arms) {
  done <<- read_done(CSV)      # see fits written earlier in this session too
  K <- as.integer(sub("^K([0-9]+)_.*$", "\\1", cf)); design <- sub("^K[0-9]+_", "", cf)
  for (r in reps) {
    todo <- arms[!vapply(arms, function(a) have(cf, r, a), logical(1))]
    if (!length(todo)) next
    D <- simulate_K(K, design, NSUB, seed = 1000L * K + 10L * (design == "dense") + r)
    cat(sprintf("\n== %s replicate %d: %.0f%% events, rows per marker %s\n", cf, r,
                100 * D$event_rate, paste(vapply(D$dls, nrow, 1L), collapse = "/")))
    for (a in todo) {
      obj <- tryCatch(run_arm(D, a, r), error = function(e) {
        if (grepl("inspect|expected|not applied|reinstall", conditionMessage(e))) stop(e)
        cat("   ", cf, a, "replicate", r, "FAILED:", conditionMessage(e), "\n"); NULL })
      if (is.null(obj)) next
      sc <- if (a == "J") NA_real_ else compile_time_jx(D, a)
      rows <- rows_of(obj, "sim", cf, D$q, r, a, truth = D$truth, sec_compile = sc)
      append_rows(rows, CSV)
      progress_line(rows, sprintf("%s rep %3d %s", cf, r, a))
    }
  }
}
if (MODE %in% c("efficiency", "all")) for (cf in CONFIGS) run_design(cf, seq_len(NREP), ARMS)
if (MODE %in% c("coverage", "all")) for (cf in COV_CFG) run_design(cf, seq_len(NREP_COV), "R")

# ---- tables -------------------------------------------------------------------------
R <- read_done(CSV)
S <- fit_summary(R)
utils::write.csv(S, file.path(OUTDIR, "sim_summary.csv"), row.names = FALSE)
Seff <- S[S$seed <= NREP, ]
tables_md(Seff, sprintf("Simulations, several markers (%d replicates per design)", NREP),
          file.path(OUTDIR, "sim_tables.md"))

# Coverage of 95% equal-tailed credible intervals and bias, arm R, runs meeting
# the bar, per design x parameter block (and per parameter in the CSV).
C <- R[R$arm == "R" & R$meets_bar & is.finite(R$truth) & R$config %in% COV_CFG, ]
if (nrow(C)) {
  C$cover <- C$q025 <= C$truth & C$truth <= C$q975
  C$bias <- C$est - C$truth
  per_par <- stats::aggregate(cbind(cover, bias) ~ config + block + param, data = C, FUN = mean)
  n_rep <- stats::aggregate(seed ~ config + param, data = C, FUN = length)
  names(n_rep)[3] <- "replicates"
  per_par <- merge(per_par, n_rep, by = c("config", "param"))
  per_par$cover_mcse <- sqrt(per_par$cover * (1 - per_par$cover) / per_par$replicates)
  utils::write.csv(per_par, file.path(OUTDIR, "sim_coverage.csv"), row.names = FALSE)
  by_block <- stats::aggregate(cbind(cover, abs_bias = abs(bias)) ~ config + block, data = C, FUN = mean)
  cat("\nCoverage of 95% credible intervals (arm R, runs meeting the bar):\n")
  print(by_block, row.names = FALSE, digits = 3)
  L <- c("", sprintf("### Coverage (arm R; %s)", paste(COV_CFG, collapse = ", ")), "",
         "| design | block | replicates | coverage (range over parameters) | mean |bias| |",
         "|---|---|---:|---|---:|")
  for (cf in unique(per_par$config)) for (b in c("beta", "alpha", "Sigma", "sigma_e")) {
    X <- per_par[per_par$config == cf & per_par$block == b, ]; if (!nrow(X)) next
    L <- c(L, sprintf("| %s | %s | %d | %.3f (%.2f-%.2f) | %.4f |", cf, b, max(X$replicates),
                      mean(X$cover), min(X$cover), max(X$cover), mean(abs(X$bias))))
  }
  cat(L, sep = "\n"); write(L, file.path(OUTDIR, "sim_tables.md"), append = TRUE)
}
