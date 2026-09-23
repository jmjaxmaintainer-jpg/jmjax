# ==============================================================================
# dev/study_alpha_bias.R - where does the upward bias in alpha come from?
#
# THE FINDING. dev/study_calibration.R (R = 100, n = 300, "linear" design)
# found alpha's posterior mean 0.0142 above the truth of 0.5 in EVERY arm,
# including the unrotated fit: 3.0 Monte Carlo SEs, about 0.3 posterior SDs,
# with 95% coverage 0.93 and sd_ratio 0.92. The arms agree to the fourth
# decimal, so it has nothing to do with the reparameterization. The
# vignette's Section 9 says it must be understood before the table goes in
# front of a referee. This study separates three explanations:
#
#   1. Finite-sample bias. Maximum-likelihood and posterior-mean estimators
#      of a hazard-ratio-type parameter are biased away from zero at O(1/n)
#      (the familiar small-sample behaviour of Cox and joint-model
#      estimators). PREDICTION: the bias shrinks like 1/n - to about a half
#      at n = 600 and a quarter at n = 1200 - and sd_ratio moves toward 1.
#   2. The simulator's baseline constant. sim_joint() sets log_lambda0 PER
#      DATASET, as the quantile of that dataset's own event thresholds that
#      gives exactly a 50% event rate. The data are then not quite draws
#      from the fitted model with a fixed parameter. PREDICTION: fixing
#      log_lambda0 at its population value removes or changes the bias.
#   3. Baseline-hazard misspecification. The truth is Weibull with shape
#      1.3, whose hazard behaves like t^0.3 near zero; the fit is a penalized
#      B-spline with 5 interior knots. PREDICTION: fitting the correctly
#      specified Weibull model (method = "weibull-PH-mcmc") removes the bias.
#
# CELLS (each R = CAB_REPS replicates, same simulation seeds 7000 + rep as
# study_calibration.R at the same n, so cells at equal n are paired):
#   ref       n = 300, calibrated log_lambda0, spline fit   (the original)
#   fixed     n = 300, FIXED log_lambda0, spline fit        (explanation 2)
#   weibull   n = 300, calibrated log_lambda0, Weibull fit  (explanation 3)
#   n600      n = 600, calibrated, spline                   (explanation 1)
#   n1200     n = 1200, calibrated, spline                  (explanation 1)
#
# The population value of log_lambda0 for "fixed" is the calibrated value
# from one very large dataset (n = 20,000), computed once and printed.
#
# Fits use the package defaults (rotation on), 2 chains x (500 + 1000), as in
# study_calibration.R. Intervals are the 95% credible intervals summary()
# reports (CrI.lower/CrI.upper, straight posterior quantiles).
#
#   caffeinate -i Rscript dev/study_alpha_bias.R                 # all cells
#   CAB_CELLS=ref,fixed,weibull caffeinate -i Rscript dev/study_alpha_bias.R
#   Rscript dev/study_alpha_bias.R --summary                     # table only
#
# Cost (rough): n = 300 fits ~20 s, n = 600 ~40 s, n = 1200 ~80 s, so about
# 0.6 + 0.6 + 0.6 + 1.1 + 2.2 = 5 hours for all five cells at R = 100.
# Resumable: rows are appended as fits finish and finished (cell, rep) pairs
# are skipped on restart.
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(survival) })
`%||%` <- function(a, b) if (is.null(a)) b else a

.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.gen  <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
source(.gen)

.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }
.envc <- function(nm, d) { v <- Sys.getenv(nm, "")
  if (nzchar(v)) strsplit(v, "[,[:space:]]+")[[1]] else d }

SUMMARY_ONLY <- any(commandArgs(trailingOnly = TRUE) == "--summary")
REPS    <- .envi("CAB_REPS", 100L)
WARMUP  <- .envi("CAB_WARMUP", 500L)
SAMPLES <- .envi("CAB_SAMPLES", 1000L)
CHAINS  <- .envi("CAB_CHAINS", 2L)
K_EXTRA <- 2L
CELLS   <- .envc("CAB_CELLS", c("ref", "fixed", "weibull", "n600", "n1200"))
OUT     <- Sys.getenv("CAB_OUT", "dev/alpha_bias_results.csv")

CELL_SPEC <- list(
  ref     = list(n = 300L,  fixed = FALSE, method = "spline-PH-mcmc"),
  fixed   = list(n = 300L,  fixed = TRUE,  method = "spline-PH-mcmc"),
  weibull = list(n = 300L,  fixed = FALSE, method = "weibull-PH-mcmc"),
  n600    = list(n = 600L,  fixed = FALSE, method = "spline-PH-mcmc"),
  n1200   = list(n = 1200L, fixed = FALSE, method = "spline-PH-mcmc"))
bad <- setdiff(CELLS, names(CELL_SPEC))
if (length(bad)) stop("unknown cell(s): ", paste(bad, collapse = ", "), call. = FALSE)

COVS  <- c("age", "sex", "trt")[seq_len(K_EXTRA)]
LFORM <- stats::as.formula(paste("y ~ time +", paste(COVS, collapse = " + ")))

# Parameters recorded, with the truth each maps to. beta_1 is the fixed slope
# on time (column 2 of y ~ time + ...); gamma_0 is the trt effect on the
# hazard (the only baseline covariate in the survival formula).
truth_of <- function(tr) c(alpha = tr$alpha, gamma_0 = tr$gamma_trt,
                           beta_0 = tr$beta0, beta_1 = tr$beta1,
                           sigma_e = tr$sigma_e, sigma_b0 = tr$sigma_b0,
                           sigma_b1 = tr$sigma_b1, rho = tr$rho)

LOG_LAMBDA0_POP <- NULL
pop_log_lambda0 <- function() {
  if (is.null(LOG_LAMBDA0_POP)) {
    big <- sim_joint(n = 20000L, seed = 1L, k_extra = K_EXTRA)
    LOG_LAMBDA0_POP <<- big$log_lambda0
    cat(sprintf("population log_lambda0 (n = 20,000): %.4f (event rate %.3f)\n",
                big$log_lambda0, big$event_rate))
  }
  LOG_LAMBDA0_POP
}

fit_one <- function(cell, rep_id) {
  sp  <- CELL_SPEC[[cell]]
  sim <- sim_joint(n = sp$n, seed = 7000L + rep_id, k_extra = K_EXTRA,
                   log_lambda0 = if (sp$fixed) pop_log_lambda0() else NULL)
  ctl <- list(num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = rep_id, progress_bar = FALSE)
  if (sp$method == "spline-PH-mcmc")
    ctl <- c(ctl, list(n_interior_knots = 5, spline_prior = "penalized",
                       rw2_implementation = "vectorized", dense_mass_spline = TRUE))
  f <- jm_fit(long_formula = LFORM,
              surv_formula = survival::Surv(time, event) ~ trt,
              data_long = sim$data_long, data_surv = sim$data_surv,
              id_var = "id", time_var = "time",
              method = sp$method, random_effects = "intercept_slope",
              random_formula = ~ time, control = ctl)
  s   <- summary(f)
  tru <- truth_of(sim$truth)
  rh  <- unlist(f$diagnostics$rhat)
  pn  <- intersect(names(tru), rownames(s))
  data.frame(cell = cell, rep = rep_id, n_sub = sp$n, method = sp$method,
             fixed_lambda = sp$fixed, event_rate = sim$event_rate,
             param = pn, truth = unname(tru[pn]),
             post_mean = s[pn, "Estimate"], post_sd = s[pn, "Std.Err"],
             q025 = s[pn, "CrI.lower"], q975 = s[pn, "CrI.upper"],
             cov95 = as.integer(tru[pn] >= s[pn, "CrI.lower"] & tru[pn] <= s[pn, "CrI.upper"]),
             rhat = s[pn, "Rhat"], max_rhat_all = max(rh[is.finite(rh)]),
             sec = as.numeric(f$convergence$sampling_time_sec %||% NA_real_),
             stringsAsFactors = FALSE)
}

# ---- run (resumable) ---------------------------------------------------------
if (!SUMMARY_ONLY) {
  done <- if (file.exists(OUT)) utils::read.csv(OUT, stringsAsFactors = FALSE) else NULL
  have <- function(cell, r) !is.null(done) && any(done$cell == cell & done$rep == r)
  for (cell in CELLS) for (r in seq_len(REPS)) {
    if (have(cell, r)) next
    rows <- tryCatch(fit_one(cell, r), error = function(e) {
      cat(sprintf("   %-8s rep %3d FAILED: %s\n", cell, r, conditionMessage(e))); NULL })
    if (is.null(rows)) next
    utils::write.table(rows, OUT, sep = ",", row.names = FALSE,
                       col.names = !file.exists(OUT), append = file.exists(OUT))
    a <- rows[rows$param == "alpha", ]
    cat(sprintf("   %-8s rep %3d  n=%4d  %5.1fs  alpha %.4f [%.4f, %.4f]  max R-hat %.3f\n",
                cell, r, rows$n_sub[1], rows$sec[1], a$post_mean, a$q025, a$q975,
                rows$max_rhat_all[1]))
  }
}

# ---- summary -----------------------------------------------------------------
R <- utils::read.csv(OUT, stringsAsFactors = FALSE)
R <- R[R$cell %in% CELLS, ]
R <- R[R$max_rhat_all <= 1.05, ]   # same R-hat gate as study_calibration.R
cat(sprintf("\nReplicates kept (max R-hat <= 1.05): %s\n",
            paste(sprintf("%s %d", names(table(R$cell[R$param == "alpha"])),
                          table(R$cell[R$param == "alpha"])), collapse = ", ")))
summ <- function(X) {
  b <- mean(X$post_mean - X$truth); se <- stats::sd(X$post_mean) / sqrt(nrow(X))
  c(n = nrow(X), bias = b, z = b / se, rel = 100 * b / X$truth[1],
    bias_over_sd = b / mean(X$post_sd), cov95 = mean(X$cov95),
    sd_ratio = mean(X$post_sd) / stats::sd(X$post_mean))
}
for (p in c("alpha", "gamma_0", "beta_1", "sigma_b0", "sigma_b1")) {
  cat(sprintf("\n%s\n%-8s %5s %9s %7s %7s %9s %6s %8s\n", p, "cell", "R",
              "bias", "z", "rel%", "bias/SD", "cov95", "sd_ratio"))
  for (cell in CELLS) {
    X <- R[R$cell == cell & R$param == p, ]
    if (!nrow(X)) next
    v <- summ(X)
    cat(sprintf("%-8s %5d %9.4f %7.2f %7.2f %9.3f %6.2f %8.3f\n", cell, as.integer(v["n"]),
                v["bias"], v["z"], v["rel"], v["bias_over_sd"], v["cov95"], v["sd_ratio"]))
  }
}
A <- R[R$param == "alpha" & R$cell %in% c("ref", "n600", "n1200"), ]
if (length(unique(A$n_sub)) >= 2) {
  ag <- aggregate(cbind(err = post_mean - truth) ~ n_sub, A, mean)
  fit <- stats::lm(err ~ 0 + I(1 / n_sub), data = ag)
  cat(sprintf("\nalpha bias vs n (spline, calibrated): %s\n",
              paste(sprintf("n=%d %.4f", ag$n_sub, ag$err), collapse = ", ")))
  cat(sprintf("fit bias = c/n: c = %.2f; predicted at n = 600 / 1200: %.4f / %.4f\n",
              coef(fit)[[1]], coef(fit)[[1]] / 600, coef(fit)[[1]] / 1200))
  cat("If the bias is O(1/n) the measured values track the prediction;",
      "a bias that does not shrink points to explanation 2 or 3.\n")
}
cat(sprintf("\nWrote/read %s\n", OUT))
