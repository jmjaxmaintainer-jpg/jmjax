# ==============================================================================
# dev/study_q1_longrun.R - does the rotation's advantage persist when both
# fits have enough computation to converge? (methods paper, review
# experiment 4; prepared 24 Sep 2026, not yet run)
#
# WHY. In the random-intercept-only grid (ROT_GRID=q1 of
# dev/pilot_rotate_grid.R; paper Section 6.3) most UNROTATED fits had not
# converged at 2 chains x (500 + 1000), so the 6.5x-282x ESS/sec ratios
# compare a converged fit with an unconverged one. This script reruns a few
# representative q = 1 designs at 1x, 2x and 4x the sampling length and
# reports the ratios (a) over all seeds and (b) over the seeds where BOTH
# fits meet an explicit convergence criterion.
#
# CONVERGENCE CRITERION (fixed before running; override with env vars):
#   max split R-hat over every population parameter <= LONG_RHAT (1.01), AND
#   ESS >= LONG_MIN_ESS (400) for each reported quantity
#   (intercept, age, time, alpha).
# ESS and R-hat are NumPyro's estimators, via the package's own
# recompute_site_diagnostics(), exactly as in dev/pilot_rotate_grid.R.
#
# DESIGNS (LONG_CELLS; same simulator and settings as the q1 grid):
#   base_n300    ~9 visits,  n = 300, sigma_e = 0.3   (q1 grid: 129x)
#   dense_n300   ~17 visits, n = 300, sigma_e = 0.3   (100x)
#   vdense_n300  ~35 visits, n = 300, sigma_e = 0.3   (282x)
#   base_n100    ~9 visits,  n = 100, sigma_e = 0.3   (41x; optional)
# Seeds are the q1 grid's (simulated data seed 9000 + seed), so the 1x runs
# should reproduce that grid's numbers up to timing.
#
# USAGE (resumable; one row per fit x quantity):
#   caffeinate -i Rscript dev/study_q1_longrun.R
#   LONG_MULT=1,2 LONG_SEEDS=2 Rscript dev/study_q1_longrun.R     # smaller
#   LONG_CELLS=vdense_n300 LONG_MULT=4 Rscript dev/study_q1_longrun.R
#   LONG_ARMS=A LONG_MULT=4,8,16,32 LONG_SEEDS=2 Rscript dev/study_q1_longrun.R
#       extend ONLY the unrotated arm until it meets the criterion; the
#       summary then reports the computation each arm needed to get there.
# Memory: every draw of b is returned to R, so very long runs are heavy
# (about 300 x 2 x samples doubles per site); 32x is about 64,000 draws.
# Settings: LONG_WARMUP (500), LONG_BASE_SAMPLES (1000), LONG_CHAINS (2),
#   LONG_SCALE_WARMUP=1 also multiplies warm-up by the multiplier,
#   LONG_OUT (dev/study_q1_longrun.csv).
# Cost: the 4x runs dominate; roughly 3 cells x 3 seeds x 2 arms x (1+2+4)
# = 126 fit-units of the q1 grid's size, about 1.5-2.5 h in total.
# ==============================================================================

suppressPackageStartupMessages(library(jmjax))
`%||%` <- function(a, b) if (is.null(a)) b else a
.self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.gen <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
source(.gen)

.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }
.envn <- function(nm, d) { v <- suppressWarnings(as.numeric(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }
.envc <- function(nm, d) { v <- Sys.getenv(nm, "")
  if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d }

SEEDS    <- .envi("LONG_SEEDS", 3L)
WARMUP   <- .envi("LONG_WARMUP", 500L)
BASE_S   <- .envi("LONG_BASE_SAMPLES", 1000L)
CHAINS   <- .envi("LONG_CHAINS", 2L)
MULTS    <- as.integer(.envc("LONG_MULT", c("1", "2", "4")))
SCALE_WU <- identical(Sys.getenv("LONG_SCALE_WARMUP", "0"), "1")
RHAT_MAX <- .envn("LONG_RHAT", 1.01)
ESS_MIN  <- .envn("LONG_MIN_ESS", 400)
OUT      <- Sys.getenv("LONG_OUT", "dev/study_q1_longrun.csv")
ARMS     <- .envc("LONG_ARMS", c("A", "A_rotdense"))
stopifnot(all(ARMS %in% c("A", "A_rotdense")))
VISIT_GAP <- c(base = 1.0, dense = 0.5, vdense = 0.25)
CELLS <- .envc("LONG_CELLS", c("base_n300", "dense_n300", "vdense_n300"))
cells <- do.call(rbind, lapply(CELLS, function(cc) {
  p <- strsplit(cc, "_n", fixed = TRUE)[[1]]
  if (!p[1] %in% names(VISIT_GAP)) stop("unknown cell ", cc, call. = FALSE)
  data.frame(cell = cc, visits = p[1], n = as.integer(p[2]), sigma_e = 0.3,
             stringsAsFactors = FALSE)
}))
LFORM <- y ~ time + age + sex

as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) { k <- length(unlist(v[[1]]))
    return(matrix(vapply(v, function(z) as.numeric(unlist(z)), numeric(k)), ncol = k, byrow = TRUE)) }
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
site_diag <- function(M, nc) {
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  list(ess = as.numeric(unlist(rd$n_eff)), rhat = as.numeric(unlist(rd$r_hat)))
}
ctl_for <- function(arm, seed, mult) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = if (SCALE_WU) WARMUP * mult else WARMUP,
              num_samples = BASE_S * mult, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (arm == "A") ctl$rotate_absorbable <- FALSE
  if (arm == "A_rotdense") { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  ctl
}

COLS <- c("cell", "mult", "seed", "arm", "quantity", "est", "sd", "ess", "rhat",
          "n_draws", "ess_per_draw", "sec", "ess_per_sec", "mean_num_steps",
          "n_divergences", "max_rhat_all", "converged")

fit_one <- function(cl, arm, seed, mult) {
  sim <- sim_joint(n = cl$n, seed = 9000L + seed, k_extra = 2L, rho = 0,
                   sigma_e = cl$sigma_e, sigma_b1 = 0, visit_gap = VISIT_GAP[[cl$visits]])
  f <- jm_fit(long_formula = LFORM, surv_formula = survival::Surv(time, event) ~ trt,
              data_long = sim$data_long, data_surv = sim$data_surv,
              id_var = "id", time_var = "time", method = "spline-PH-mcmc",
              random_effects = "intercept", random_formula = ~ 1,
              control = ctl_for(arm, seed, mult))
  cv <- f$convergence
  if (arm == "A_rotdense" && !isTRUE(cv$orthogonalize$rotation$applied))
    stop("rotation not applied - is the installed backend current?", call. = FALSE)
  Xd <- stats::model.matrix(LFORM, sim$data_long)
  idx <- match(c("(Intercept)", "age", "time"), colnames(Xd))
  ps <- f$posterior_samples
  M <- as_draws(ps[["beta"]])[, idx, drop = FALSE]
  AL <- as_draws(ps[["alpha"]])
  d <- site_diag(cbind(M, AL[, 1]), CHAINS)
  secs <- as.numeric(cv$sampling_time_sec %||% NA_real_)
  mrh <- tryCatch(max(unlist(f$diagnostics$rhat), na.rm = TRUE), error = function(e) NA_real_)
  nd <- CHAINS * BASE_S * mult
  X <- cbind(M, AL[, 1])
  out <- data.frame(cell = cl$cell, mult = mult, seed = seed, arm = arm,
                    quantity = c("intercept", "age", "time", "alpha"),
                    est = colMeans(X), sd = apply(X, 2, stats::sd),
                    ess = d$ess, rhat = d$rhat, n_draws = nd, ess_per_draw = d$ess / nd,
                    sec = secs, ess_per_sec = d$ess / secs,
                    mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
                    n_divergences = as.numeric(cv$n_divergences %||% NA_real_),
                    max_rhat_all = mrh, stringsAsFactors = FALSE)
  out$converged <- is.finite(mrh) && mrh <= RHAT_MAX && all(out$ess >= ESS_MIN)
  out
}

done <- if (file.exists(OUT)) utils::read.csv(OUT, stringsAsFactors = FALSE) else NULL
if (!is.null(done) && !identical(names(done), COLS))
  stop(OUT, " has a different column layout; set LONG_OUT to a new file", call. = FALSE)
is_done <- function(cc, m, s, a) !is.null(done) &&
  any(done$cell == cc & done$mult == m & done$seed == s & done$arm == a)

jobs <- expand.grid(arm = ARMS, seed = seq_len(SEEDS), mult = MULTS,
                    i = seq_len(nrow(cells)), stringsAsFactors = FALSE)
cat(sprintf("q1 long runs: %d cells x mult {%s} x %d seeds x 2 arms = %d fits -> %s\n",
            nrow(cells), paste(MULTS, collapse = ","), SEEDS, nrow(jobs), OUT))
cat(sprintf("criterion: max R-hat <= %.2f and ESS >= %.0f on intercept, age, time, alpha\n",
            RHAT_MAX, ESS_MIN))
for (r in seq_len(nrow(jobs))) {
  cl <- cells[jobs$i[r], ]; a <- jobs$arm[r]; s <- jobs$seed[r]; m <- jobs$mult[r]
  if (is_done(cl$cell, m, s, a)) next
  rows <- tryCatch(fit_one(cl, a, s, m), error = function(e) {
    if (grepl("not applied", conditionMessage(e))) stop(e)
    cat(sprintf("  [%s x%d %s seed %d] FAILED: %s\n", cl$cell, m, a, s, conditionMessage(e))); NULL })
  if (is.null(rows)) next
  utils::write.table(rows[, COLS], OUT, sep = ",", row.names = FALSE,
                     col.names = !file.exists(OUT), append = file.exists(OUT))
  cat(sprintf("  [%3d/%d] %-12s x%d %-10s seed %d  %6.1fs  int ESS/draw %.3f  max R-hat %.3f  %s\n",
              r, nrow(jobs), cl$cell, m, a, s, rows$sec[1], rows$ess_per_draw[1],
              rows$max_rhat_all[1], if (rows$converged[1]) "converged" else "NOT converged"))
}

# ---- summary ------------------------------------------------------------------
R <- utils::read.csv(OUT, stringsAsFactors = FALSE)
R <- R[R$cell %in% cells$cell & R$mult %in% MULTS, ]
gm <- function(x) { x <- x[is.finite(x) & x > 0]; if (length(x)) exp(mean(log(x))) else NA_real_ }
cat("\nConverged fits (criterion above), by arm and run length\n")
cv <- unique(R[, c("cell", "mult", "seed", "arm", "converged")])
cv$converged <- as.integer(as.logical(cv$converged))
print(stats::xtabs(converged ~ cell + mult + arm, cv))
cat("\nRotated / unrotated, geometric mean over seeds: ESS/sec | ESS/draw   [seeds]\n")
cat("  'all' = every seed; 'both conv.' = seeds where both fits meet the criterion\n")
for (cc in cells$cell) for (m in MULTS) for (q in c("intercept", "age", "time", "alpha")) {
  A <- R[R$cell == cc & R$mult == m & R$arm == "A" & R$quantity == q, ]
  B <- R[R$cell == cc & R$mult == m & R$arm == "A_rotdense" & R$quantity == q, ]
  x <- merge(A, B, by = "seed"); if (!nrow(x)) next
  y <- x[as.logical(x$converged.x) & as.logical(x$converged.y), ]
  f <- function(z) if (nrow(z)) sprintf("%7.2fx | %6.2fx [%d]", gm(z$ess_per_sec.y / z$ess_per_sec.x),
                                         gm(z$ess_per_draw.y / z$ess_per_draw.x), nrow(z)) else "      -"
  cat(sprintf("  %-12s x%d %-9s all %s   both conv. %s   unrot. ESS/draw %.3f  R-hat %.3f\n",
              cc, m, q, f(x), f(y), mean(x$ess_per_draw.x), max(x$rhat.x)))
}
cat("\nComputation needed to MEET THE CRITERION (convergence-matched comparison):\n")
cat("  smallest run length at which each arm converged, its wall time and draws;\n")
cat("  ratio = unrotated seconds / rotated seconds, per seed\n")
cvr <- unique(R[, c("cell", "mult", "seed", "arm", "converged", "sec", "n_draws")])
cvr$converged <- as.logical(cvr$converged)
first_ok <- function(cc, s, a) {
  z <- cvr[cvr$cell == cc & cvr$seed == s & cvr$arm == a & cvr$converged, ]
  if (!nrow(z)) return(NULL); z[which.min(z$mult), ]
}
for (cc in cells$cell) for (s in sort(unique(cvr$seed[cvr$cell == cc]))) {
  a <- first_ok(cc, s, "A"); b <- first_ok(cc, s, "A_rotdense")
  tried <- max(cvr$mult[cvr$cell == cc & cvr$seed == s & cvr$arm == "A"], -Inf)
  cat(sprintf("  %-12s seed %d  unrotated: %s   rotated: %s   %s\n", cc, s,
              if (is.null(a)) sprintf("not met up to x%s", tried) else
                sprintf("x%-3d %7.1fs %7d draws", a$mult, a$sec, a$n_draws),
              if (is.null(b)) "not met" else sprintf("x%-3d %6.1fs %6d draws", b$mult, b$sec, b$n_draws),
              if (!is.null(a) && !is.null(b)) sprintf("ratio %.1fx", a$sec / b$sec) else ""))
}

cat("\nPosterior agreement between each arm's first converged fit (same cell and seed):\n")
cat("  |mean_rot - mean_unrot| / sd_rot and sd_rot / sd_unrot, over quantities\n")
agr <- NULL
for (cc in cells$cell) for (s in sort(unique(cvr$seed[cvr$cell == cc]))) {
  a <- first_ok(cc, s, "A"); b <- first_ok(cc, s, "A_rotdense")
  if (is.null(a) || is.null(b)) next
  x <- R[R$cell == cc & R$seed == s & R$arm == "A" & R$mult == a$mult, ]
  y <- R[R$cell == cc & R$seed == s & R$arm == "A_rotdense" & R$mult == b$mult, ]
  m <- merge(x, y, by = "quantity")
  agr <- rbind(agr, data.frame(d = abs(m$est.y - m$est.x) / m$sd.y, r = m$sd.y / m$sd.x))
}
if (!is.null(agr)) cat(sprintf("  median %.3f, max %.3f over %d pairs; SD ratio median %.3f (range %.3f-%.3f)\n",
                               stats::median(agr$d), max(agr$d), nrow(agr), stats::median(agr$r),
                               min(agr$r), max(agr$r))) else cat("  no cell/seed with both arms converged yet\n")
