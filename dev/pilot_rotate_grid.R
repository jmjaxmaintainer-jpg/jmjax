# ==============================================================================
# dev/pilot_rotate_grid.R - test the predictions of vignette Section 4.10
#
# WHAT THIS IS. Section 4.10 of vignette("jmjax-reparameterization") proves
# the rotation exact and, from a closed-form model (dev/theory_rotation_toy.py),
# predicts WHEN and roughly HOW MUCH it helps. This script checks those
# predictions with real NUTS fits of the full joint model, sampled variance
# components and survival submodel included - exactly what the closed-form
# check leaves out.
#
# ARMS (all with the penalized-spline baseline hazard used by every pilot):
#   A         default fit (reference; reports beta itself)
#   D         orthogonalize_b0
#   D_rot     D + orthogonalize_b0_rotate   (intercept-only, random completion)
#   D_rotall  D + orthogonalize_rotate_all  (every column, Householder)
#   E         orthogonalize_b
#   E_rot0    E + orthogonalize_b0_rotate
#   E_rotall  E + orthogonalize_rotate_all
#
# GRID. Cells vary what Section 4.10 says the gain depends on:
#   visits   - sim_joint's visit_gap: "base" = 1.0 (up to 11 visits, the
#              pilots' design), "dense" = 0.5 (up to 21), "sparse" = 2.0
#   sigma_e  - 0.3 (pilots) or 0.8 (noisier: less per-subject information)
#   rho      - generating intercept-slope correlation, 0 / 0.3 / 0.6 / 0.9
#   n        - 300, plus 150 and 600 in the N-invariance cells
# Default ("core") grid: visits in {base, dense} x sigma_e = 0.3 x all four
# rho at n = 300, plus n in {150, 600} at base / 0.3 / rho = 0.3 - 10 cells.
# ROT_GRID=full adds sparse visits and sigma_e = 0.8 (24 cells + 2).
#
# PREDICTIONS BEING TESTED (Section 4.10, under its table):
#   P1  beta_corrected[intercept]: D_rot and D_rotall >> D. D_rot's gain is
#       reduced at intermediate rho with dense visits; D_rotall's is not.
#   P2  beta_corrected[time] under orthogonalize_b: E_rot0 ~ E except at
#       large rho; E_rotall >> E at every rho.
#   P3  gains roughly independent of n.
#   P4  gains smaller with sparse visits / sigma_e = 0.8 (less per-subject
#       information), never below ~1x.
#   Not predicted by the theory, recorded here: wall time and mean
#   leapfrog steps per iteration (fit$convergence$mean_num_steps).
#
# USAGE (resumable; appends one row per fit x quantity to ROT_OUT):
#   Rscript dev/pilot_rotate_grid.R                          # core grid, 2 seeds
#   ROT_SEEDS=4 ROT_GRID=full caffeinate -i Rscript dev/pilot_rotate_grid.R
#   ROT_ARMS=D,D_rot,D_rotall Rscript dev/pilot_rotate_grid.R
# Rough cost at n = 300: ~15-25 s per fit; core grid, 7 arms, 2 seeds is
# 140 fits, about 45-60 min. Dense-visit and n = 600 cells cost more.
# ==============================================================================

suppressPackageStartupMessages(library(jmjax))
`%||%` <- function(a, b) if (is.null(a)) b else a

.self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.gen <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
if (!file.exists(.gen)) stop("cannot find sim_joint.R next to this script", call. = FALSE)
source(.gen)

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d
}
.envc <- function(nm, d) {
  v <- Sys.getenv(nm, "")
  if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d
}

SEEDS   <- .envi("ROT_SEEDS", 2L)
WARMUP  <- .envi("ROT_WARMUP", 500L)
SAMPLES <- .envi("ROT_SAMPLES", 1000L)
CHAINS  <- .envi("ROT_CHAINS", 2L)
K_EXTRA <- .envi("ROT_K_EXTRA", 2L)
GRID    <- Sys.getenv("ROT_GRID", "core")
OUT     <- Sys.getenv("ROT_OUT", "dev/pilot_rotate_grid.csv")
ARMS    <- .envc("ROT_ARMS", c("A", "D", "D_rot", "D_rotall", "E", "E_rot0", "E_rotall"))

VISIT_GAP <- c(sparse = 2.0, base = 1.0, dense = 0.5)

cells <- if (identical(GRID, "full")) {
  expand.grid(visits = c("sparse", "base", "dense"), sigma_e = c(0.3, 0.8),
              rho = c(0, 0.3, 0.6, 0.9), n = 300L, stringsAsFactors = FALSE)
} else {
  expand.grid(visits = c("base", "dense"), sigma_e = 0.3,
              rho = c(0, 0.3, 0.6, 0.9), n = 300L, stringsAsFactors = FALSE)
}
cells <- rbind(cells, data.frame(visits = "base", sigma_e = 0.3, rho = 0.3,
                                 n = c(150L, 600L), stringsAsFactors = FALSE))
cells$cell <- sprintf("%s_se%.1f_rho%.1f_n%d", cells$visits, cells$sigma_e,
                      cells$rho, cells$n)

COVS  <- c("age", "sex", "trt")[seq_len(K_EXTRA)]
LFORM <- stats::as.formula(paste("y ~ time +", paste(COVS, collapse = " + ")))

# draws x k matrix from a list-of-draws site. matrix(byrow = TRUE) rather
# than t(vapply(...)): for a scalar site (k = 1, e.g. alpha) vapply returns
# a plain vector and t() would give a 1 x draws matrix.
as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) {
    k <- length(unlist(v[[1]]))
    return(matrix(vapply(v, function(z) as.numeric(unlist(z)), numeric(k)),
                  ncol = k, byrow = TRUE))
  }
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
site_diag <- function(M, nc) {
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  list(ess = as.numeric(unlist(rd$n_eff)), rhat = as.numeric(unlist(rd$r_hat)))
}

arm_control <- function(arm, seed) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (startsWith(arm, "D")) ctl$orthogonalize_b0 <- TRUE
  if (startsWith(arm, "E")) ctl$orthogonalize_b  <- TRUE
  if (arm %in% c("D_rot", "E_rot0"))     ctl$orthogonalize_b0_rotate  <- TRUE
  if (arm %in% c("D_rotall", "E_rotall")) ctl$orthogonalize_rotate_all <- TRUE
  ctl
}

EXPECTED_COLS <- c("cell", "visits", "sigma_e", "rho", "n", "seed", "arm",
                   "quantity", "est", "sd", "ess", "rhat", "ess_per_sec",
                   "sec", "mean_num_steps", "n_divergences", "max_rhat_all",
                   "mean_visits")

fit_one <- function(cl, arm, seed) {
  sim <- sim_joint(n = cl$n, seed = 9000L + seed, k_extra = K_EXTRA,
                   rho = cl$rho, sigma_e = cl$sigma_e,
                   visit_gap = VISIT_GAP[[cl$visits]])
  f <- jm_fit(
    long_formula = LFORM,
    surv_formula = survival::Surv(time, event) ~ trt,
    data_long = sim$data_long, data_surv = sim$data_surv,
    id_var = "id", time_var = "time",
    method = "spline-PH-mcmc", random_effects = "intercept_slope",
    random_formula = ~ time, control = arm_control(arm, seed))

  cv   <- f$convergence
  secs <- as.numeric(cv$sampling_time_sec %||% NA_real_)
  mns  <- as.numeric(cv$mean_num_steps %||% NA_real_)
  ndiv <- as.numeric(cv$n_divergences %||% NA_real_)
  mrh  <- tryCatch(max(unlist(f$diagnostics$rhat), na.rm = TRUE), error = function(e) NA_real_)
  if (grepl("rot", arm)) {
    rot <- cv$orthogonalize$rotation
    if (is.null(rot) || !isTRUE(rot$applied))
      stop("arm ", arm, ": rotation not applied - is the installed backend current?",
           call. = FALSE)
  }
  mv <- mean(table(sim$data_long$id))

  Xd <- stats::model.matrix(LFORM, sim$data_long)
  i0 <- match("(Intercept)", colnames(Xd)); it <- match("time", colnames(Xd))
  ps <- f$posterior_samples
  # A reports beta itself; every other arm reports beta_corrected.
  M <- if (arm == "A") as_draws(ps[["beta"]]) else as_draws(ps[["beta_corrected"]])
  if (is.null(M)) stop("arm ", arm, " returned no reportable beta", call. = FALSE)
  d <- site_diag(M[, c(i0, it), drop = FALSE], CHAINS)
  AL <- as_draws(ps[["alpha"]])
  da <- site_diag(AL, CHAINS)

  mk <- function(q, x, ess, rhat) data.frame(
    cell = cl$cell, visits = cl$visits, sigma_e = cl$sigma_e, rho = cl$rho,
    n = cl$n, seed = seed, arm = arm, quantity = q,
    est = mean(x), sd = stats::sd(x), ess = ess, rhat = rhat,
    ess_per_sec = ess / secs, sec = secs, mean_num_steps = mns,
    n_divergences = ndiv, max_rhat_all = mrh, mean_visits = mv,
    stringsAsFactors = FALSE)
  rbind(mk("intercept", M[, i0], d$ess[1], d$rhat[1]),
        mk("time",      M[, it], d$ess[2], d$rhat[2]),
        mk("alpha",     AL[, 1], da$ess[1], da$rhat[1]))
}

# ---- resume -----------------------------------------------------------------
done <- if (file.exists(OUT)) utils::read.csv(OUT, stringsAsFactors = FALSE) else NULL
if (!is.null(done) && !identical(names(done), EXPECTED_COLS))
  stop(OUT, " exists with a different column layout; set ROT_OUT to a new file",
       call. = FALSE)
is_done <- function(cell, arm, seed) !is.null(done) &&
  any(done$cell == cell & done$arm == arm & done$seed == seed)

# arm varies fastest, so a partial run still has every arm for the cells
# it reached - ratios need matched arms, not a complete grid.
jobs <- expand.grid(arm = ARMS, seed = seq_len(SEEDS), i = seq_len(nrow(cells)),
                    stringsAsFactors = FALSE)
cat(sprintf("rotation grid: %d cells x %d arms x %d seeds = %d fits (%s grid) -> %s\n",
            nrow(cells), length(ARMS), SEEDS, nrow(jobs), GRID, OUT))
for (r in seq_len(nrow(jobs))) {
  cl <- cells[jobs$i[r], ]; a <- jobs$arm[r]; s <- jobs$seed[r]
  if (is_done(cl$cell, a, s)) next
  t0 <- Sys.time()
  rows <- tryCatch(fit_one(cl, a, s), error = function(e) {
    msg <- conditionMessage(e)
    if (grepl("not applied|installed backend", msg)) stop(msg, call. = FALSE)
    cat(sprintf("  [%s %s seed %d] FAILED: %s\n", cl$cell, a, s, msg)); NULL })
  if (is.null(rows)) next
  utils::write.table(rows[, EXPECTED_COLS], OUT, sep = ",", row.names = FALSE,
                     col.names = !file.exists(OUT), append = file.exists(OUT))
  cat(sprintf("  [%3d/%d] %-26s %-9s seed %d  %5.1fs  ess/s int %7.1f time %7.1f\n",
              r, nrow(jobs), cl$cell, a, s,
              as.numeric(difftime(Sys.time(), t0, units = "secs")),
              rows$ess_per_sec[1], rows$ess_per_sec[2]))
}

# ---- summary ----------------------------------------------------------------
R <- utils::read.csv(OUT, stringsAsFactors = FALSE)
R <- R[R$arm %in% ARMS & R$cell %in% cells$cell, , drop = FALSE]
bad <- is.na(R$max_rhat_all) | R$max_rhat_all > 1.05
if (any(bad)) cat(sprintf("\nR-hat gate (max over population sites <= 1.05): dropping %d of %d rows\n",
                          sum(bad), nrow(R)))
R <- R[!bad, , drop = FALSE]

agg <- stats::aggregate(cbind(ess_per_sec, sec, mean_num_steps) ~ cell + arm + quantity,
                        data = R, FUN = function(x) mean(x, na.rm = TRUE),
                        na.action = stats::na.pass)
ratio <- function(cell, q, num, den) {
  x <- agg$ess_per_sec[agg$cell == cell & agg$quantity == q & agg$arm == num]
  y <- agg$ess_per_sec[agg$cell == cell & agg$quantity == q & agg$arm == den]
  if (length(x) && length(y) && y > 0) x / y else NA_real_
}
show <- function(title, q, pairs) {
  cat("\n", title, "\n", sep = "")
  cat(sprintf("%-26s", "cell"), sprintf("%12s", names(pairs)), "\n", sep = "")
  for (cc in cells$cell) {
    v <- vapply(pairs, function(p) ratio(cc, q, p[1], p[2]), numeric(1))
    cat(sprintf("%-26s", cc), sprintf("%12s", ifelse(is.na(v), "-", sprintf("%.2fx", v))),
        "\n", sep = "")
  }
}
show("beta_corrected[intercept], ESS/sec ratios (P1, P3, P4)", "intercept",
     list("D/A" = c("D", "A"), "D_rot/D" = c("D_rot", "D"),
          "D_rotall/D" = c("D_rotall", "D"), "E_rotall/E" = c("E_rotall", "E")))
show("beta_corrected[time], ESS/sec ratios (P2)", "time",
     list("E/A" = c("E", "A"), "E_rot0/E" = c("E_rot0", "E"),
          "E_rotall/E" = c("E_rotall", "E")))
show("alpha (negative control: should stay near 1x)", "alpha",
     list("D_rotall/D" = c("D_rotall", "D"), "E_rotall/E" = c("E_rotall", "E")))

cat("\nCost: mean seconds and leapfrog steps per iteration (not predicted by the theory)\n")
cost <- agg[agg$quantity == "intercept", c("cell", "arm", "sec", "mean_num_steps")]
cost <- cost[order(cost$cell, match(cost$arm, ARMS)), ]
print(cost, row.names = FALSE, digits = 3)

cat("\nCorrectness: |mean(rotated) - mean(unrotated)| / sd, same cell and seed (should be small)\n")
for (pr in list(c("D_rot", "D"), c("D_rotall", "D"), c("E_rotall", "E"))) {
  x <- merge(R[R$arm == pr[1], c("cell", "seed", "quantity", "est", "sd")],
             R[R$arm == pr[2], c("cell", "seed", "quantity", "est", "sd")],
             by = c("cell", "seed", "quantity"))
  if (nrow(x)) cat(sprintf("  %-9s vs %-2s: median %.3f, max %.3f (%d pairs)\n", pr[1], pr[2],
                           stats::median(abs(x$est.x - x$est.y) / x$sd.y),
                           max(abs(x$est.x - x$est.y) / x$sd.y), nrow(x)))
}
