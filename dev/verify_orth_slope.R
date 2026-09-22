# ==============================================================================
# The last correctness gate before orthogonalize_b could be a default.
#
# WHY THIS IS NEEDED. Arm E (constrain EVERY random-effect column) produced
# 11.6-16.9x on regression-parameter ESS/sec across 4 cells, 12/12
# replicates, every CI far from 1. But its correctness evidence is thinner
# than arm D's, in two specific ways that are my instrumentation's fault:
#
#   1. The invariant check covers the INTERCEPT column only, so the slope
#      constraint - the half doing all the new work - has no algebraic
#      verification at all.
#   2. The fitted-value null was calibrated from ONE A-vs-A2 pair per cell.
#      E came in at 0.97x, 0.97x, 1.30x, 0.79x against it. With a single
#      pair the null's own scale is so noisy that 1.0x and 1.3x are not
#      distinguishable, so "1.30x" neither passes nor fails.
#
# Thin instrumentation is exactly what let the k=4 failure through and then
# raised two false alarms on my own thresholds. E is the arm that would
# become the default; it gets checked properly.
#
# WHAT THIS DOES DIFFERENTLY.
#
#   NULL FROM 4 SEEDS, NOT 1. Arm A is run at four sampler seeds on FIXED
#   data, giving six A-vs-A' pairs. The null then has a spread, not just a
#   point, and the treatment is judged against that spread - so the verdict
#   is "inside the range re-seeding produces" rather than a ratio against a
#   single noisy number.
#
#   THE INVARIANT, GENERALIZED TO EVERY RANDOM-EFFECT COLUMN. A shift of
#   b[, q] by a subject-constant s_i moves the linear predictor by
#   s_i * Z[, q], which beta absorbs exactly when some column j of X equals
#   s_i * Z[, q]. Writing b[, q] = S_q c_q + r, the identified quantities
#   are then
#
#       beta[kept_j] + c_q        componentwise, for every q
#
#   For q = 0 the absorbable columns are the subject-constant ones and this
#   is the check already used. For q = 1 (random slope on time) the only
#   absorbable column is `time` itself, so the identified quantity is
#   beta_1 + mean(b_i1) - the slope analogue, and the one that has never
#   been tested.
#
#   The detection is re-derived here in R rather than read out of the
#   backend. That is deliberate: a test that imports the implementation's
#   own notion of which columns are absorbable cannot catch the case where
#   that notion is wrong. The fitted-value null remains the check that
#   depends on no algebra at all, and it is the one that decides.
#
#   Rscript dev/verify_orth_slope.R
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
suppressPackageStartupMessages(library(JMbayes2))
`%||%` <- function(a, b) if (is.null(a)) b else a
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.gen  <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
source(.gen)

.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
                           if (length(v) && is.finite(v) && v > 0) v else d }
CHAINS <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES<- .envi("BENCH_JX_SAMPLES", 1000L)
NNULL  <- .envi("BENCH_NULL_SEEDS", 4L)

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("verify_orth_slope_%s", format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)
options(error = function() { try(while (sink.number() > 0) sink(), silent = TRUE) })

as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
bcol <- function(ps, j, nsub) {
  b <- ps[["b"]]
  t(vapply(b, function(d) {
    if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
    else { a <- as.matrix(d); as.numeric(a[, j]) }
  }, numeric(nsub)))
}

# Which columns of X can absorb a shift in b[, q]? Re-derived here rather
# than read from the backend, so a wrong notion there cannot pass unnoticed.
absorbable <- function(X, Zq, id, tol = 1e-8) {
  keep <- list()
  for (j in seq_len(ncol(X))) {
    nz <- abs(Zq) > tol
    if (any(!nz & abs(X[, j]) > tol)) next          # X must vanish where Z does
    r  <- rep(NA_real_, length(Zq)); r[nz] <- X[nz, j] / Zq[nz]
    sp <- tapply(r, id, function(v) { v <- v[is.finite(v)]
                                      if (!length(v)) 0 else diff(range(v)) })
    if (max(sp, na.rm = TRUE) > tol * max(abs(r), 1, na.rm = TRUE)) next
    si <- tapply(r, id, function(v) { v <- v[is.finite(v)]
                                      if (!length(v)) 0 else v[1] })
    keep[[colnames(X)[j]]] <- list(col = j, s = as.numeric(si))
  }
  keep
}

for (cfg in list(list(k = 3L, n = 500L), list(k = 1L, n = 500L))) {
  k_extra <- cfg$k - 1L
  sim <- sim_joint(n = cfg$n, seed = 1000L * cfg$k + 1L, k_extra = k_extra)
  dl <- sim$data_long; ds <- sim$data_surv
  covs  <- c("age", "sex", "trt")[seq_len(k_extra)]
  lform <- stats::as.formula(paste("y ~ time",
             if (length(covs)) paste("+", paste(covs, collapse = " + ")) else ""))
  lme_fit <- lme(lform, random = ~ time | id, data = dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)

  X  <- stats::model.matrix(lform, data = dl)
  sub<- dl[!duplicated(dl$id), ]; sub <- sub[order(sub$id), ]
  IDX<- match(dl$id, sub$id); NS <- nrow(sub)
  Zc <- list(rep(1, nrow(dl)), dl$time)             # Z = [1, time]
  ABS<- lapply(Zc, function(z) absorbable(X, z, dl$id))

  cat("\n=============================================================\n")
  cat(sprintf("  k = %d | n = %d | %.1f obs/subject | %.0f%% events\n",
              cfg$k, cfg$n, sim$obs_per_subject, 100 * sim$event_rate))
  for (q in seq_along(ABS))
    cat(sprintf("  b column %d absorbs: %s\n", q - 1L,
                if (length(ABS[[q]])) paste(names(ABS[[q]]), collapse = ", ") else "(nothing)"))
  cat("=============================================================\n")

  run <- function(mode, sseed) {
    ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
                rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                num_warmup = WARMUP, num_samples = SAMPLES,
                num_chains = CHAINS, progress_bar = FALSE, seed = sseed)
    if (mode == "D") ctl$orthogonalize_b0 <- TRUE
    if (mode == "E") ctl$orthogonalize_b  <- TRUE
    f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                       method = "spline-PH-mcmc", control = ctl)
    ps <- f$posterior_samples
    B  <- as_draw_matrix(ps[["beta"]])
    B0 <- bcol(ps, 1L, NS); B1 <- bcol(ps, 2L, NS)
    Bq <- list(B0, B1)
    # invariant: beta[kept] + c_q, for every random-effect column
    inv <- c()
    for (q in seq_along(ABS)) for (nm in names(ABS[[q]])) {
      S  <- matrix(ABS[[q]][[nm]]$s, ncol = 1)
      cq <- as.numeric(Bq[[q]] %*% S) / sum(S^2)
      inv[paste0("b", q - 1L, ":", nm)] <-
        mean(B[, ABS[[q]][[nm]]$col]) + mean(cq)
    }
    # "Pinned" is a PER-DRAW property: the constraint holds in every draw or
    # it does not hold. An average over draws is not a test of it, and the
    # first version of this script reported one and then printed a verdict
    # claiming the constraint had bitten - a claim the code never made.
    list(mu = as.numeric(X %*% colMeans(B)) +
              colMeans(B0)[IDX] + colMeans(B1)[IDX] * dl$time,
         inv = inv,
         pin0 = max(abs(rowMeans(B0))), pin1 = max(abs(rowMeans(B1))))
  }

  cat(sprintf("\n  fitting A at %d sampler seeds (the null), then D and E...\n", NNULL))
  As <- lapply(seq_len(NNULL), function(i) run("A", i))
  D  <- run("D", 1L); E <- run("E", 1L)

  nulls <- c()
  for (i in seq_len(NNULL)) for (j in seq_len(NNULL)) if (j > i)
    nulls <- c(nulls, mean(abs(As[[j]]$mu - As[[i]]$mu)))
  tD <- mean(abs(D$mu - As[[1]]$mu)); tE <- mean(abs(E$mu - As[[1]]$mu))

  cat("\n  ---- CHECK 0: does the constraint actually bite? ----\n")
  cat("    max OVER DRAWS of |mean_i b_iq|; pinned means < 1e-10.\n")
  cat(sprintf("    b[,0]   A %10.3e   D %10.3e   E %10.3e\n",
              As[[1]]$pin0, D$pin0, E$pin0))
  cat(sprintf("    b[,1]   A %10.3e   D %10.3e   E %10.3e\n",
              As[[1]]$pin1, D$pin1, E$pin1))
  pinD <- D$pin0 < 1e-10 && D$pin1 > 1e-10          # intercept only
  pinE <- E$pin0 < 1e-10 && E$pin1 < 1e-10          # both columns
  cat(sprintf("    D pins the intercept column alone: %s\n", if (pinD) "yes" else "NO"))
  cat(sprintf("    E pins both columns:               %s\n", if (pinE) "yes" else "NO"))

  cat("\n  ---- CHECK 1: identified quantities, every b column ----\n")
  cat(sprintf("    %-18s %12s %12s %12s\n", "invariant", "A", "D", "E"))
  for (nm in names(As[[1]]$inv))
    cat(sprintf("    %-18s %12.6f %12.6f %12.6f\n", nm,
                As[[1]]$inv[[nm]], D$inv[[nm]], E$inv[[nm]]))
  sprA <- vapply(names(As[[1]]$inv), function(nm)
    diff(range(vapply(As, function(a) a$inv[[nm]], numeric(1)))), numeric(1))
  cat("\n    spread across the four A seeds (the noise floor), vs |E - A|:\n")
  okinv <- TRUE
  for (nm in names(sprA)) {
    dE <- abs(E$inv[[nm]] - As[[1]]$inv[[nm]])
    if (dE > max(3 * sprA[[nm]], 1e-6)) okinv <- FALSE
    cat(sprintf("    %-18s null spread %9.2e   |E-A| %9.2e  %s\n",
                nm, sprA[[nm]], dE,
                if (dE <= max(3 * sprA[[nm]], 1e-6)) "ok" else "** EXCEEDS **"))
  }

  cat("\n  ---- CHECK 2: fitted values vs a null with a SPREAD ----\n")
  cat(sprintf("    null over %d pairs: min %.3e  median %.3e  max %.3e\n",
              length(nulls), min(nulls), stats::median(nulls), max(nulls)))
  cat(sprintf("    treatment D %.3e   (%.2fx the null median)\n", tD, tD / stats::median(nulls)))
  cat(sprintf("    treatment E %.3e   (%.2fx the null median)\n", tE, tE / stats::median(nulls)))
  inD <- tD <= max(nulls); inE <- tE <= max(nulls)
  cat(sprintf("    D inside the null range: %s      E inside: %s\n",
              if (inD) "yes" else "NO", if (inE) "yes" else "NO"))

  cat("\n  ---- VERDICT ----\n")
  if (okinv && inE && pinE)
    cat("    E PASSES. The slope constraint bites (mean(b_i1) -> 0), every\n",
        "   identified quantity including the slope invariant is unmoved\n",
        "   beyond the noise floor, and fitted values are perturbed no more\n",
        "   than re-seeding perturbs them.\n", sep = "")
  else
    cat("    E FAILS. ", if (!pinE) "The constraint did not bite. " else "",
        if (!okinv) "An invariant moved beyond the noise floor. " else "",
        if (!inE) "Fitted values fall outside the re-seeding null. " else "",
        "It must not become the default.\n", sep = "")
}
cat("\nwrote ", OUT, ".log\n", sep = "")
sink(); close(.con)
