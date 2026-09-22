# ==============================================================================
# Real-data pilot: jmjax (A / A2 / D / E) vs JMbayes2 (J), on the `aids`
# dataset bundled with JMbayes2 (Guo & Carlin 2004 / Goldman et al. 1996;
# n = 467 patients, CD4 count over time, Surv(Time, death)).
#
# WHY NOT PBC2. PBC2 is the only real dataset the orthogonalize_b/b0 work has
# ever touched - every degeneracy diagnostic, every warm-start investigation,
# every "measured motivation" number in mcmc_model.py cites it. A result that
# only ever shows up on the one dataset used to build the diagnostics that
# found it is not yet evidence the effect generalizes. aids is untouched by
# any of this project's history and is one of the more-cited benchmarks in
# the joint-modelling literature in its own right (Rizopoulos 2011 JM
# package; Guo & Carlin 2004), which also makes the result more legible to
# an outside reader than another pbc2 number would be.
#
# STRUCTURE, MATCHED TO THE SIMULATED STUDY.
#   longitudinal: sqrt(CD4) ~ obstime + <covariates>, random = ~ obstime |
#                 patient  (q = 2: intercept + slope on time, same shape as
#                 the simulated k=1/k=3 cells)
#   survival:     Surv(Time, death) ~ drug
#   covariates:   drug, gender, prevOI, AZT are all baseline (subject-
#                 constant) in the `aids` long-format data, so they are all
#                 candidate absorbable directions for b[,0] exactly as
#                 age/sex were in the simulated k=3 cell. BENCH_COVS controls
#                 how many are included - default "drug,gender" is the
#                 closest real-data analogue to the simulated k=3 pilot
#                 (intercept + 2 covariates = 3 absorbable directions).
#
# NO KNOWN TRUTH. Real data has no generating alpha/beta_1 to check recovery
# against, so the ACCURACY section reports each arm's point estimate for
# comparison across arms/packages (A and D must agree exactly - the
# reparameterization cannot move alpha or beta_1 - and A/D/E vs J is the
# cross-package check), not a truth-vs-estimate table. CORRECTNESS (the
# fitted-value null) and EFFICIENCY need no truth and are unchanged.
#
# SHARED PRE-FIT. lme()/coxph() are fitted ONCE (the data does not change
# across sampler seeds), matching dev/bench_pbc2.R's design: both packages
# start from the identical pre-fit, so the sampler comparison is sampler
# against sampler.
#
#   Rscript dev/study_realdata_aids.R
#   BENCH_SEEDS=2 BENCH_JB_ITER=3500 BENCH_JB_BURN=500 BENCH_TAG=aids_pilot \
#     Rscript dev/study_realdata_aids.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 is required for the comparison arm.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))
`%||%` <- function(a, b) if (is.null(a)) b else a

.envl <- function(nm, d) {
  v <- Sys.getenv(nm, "")
  if (!nzchar(v)) return(d)
  strsplit(v, ",", fixed = TRUE)[[1]]
}
.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}

COVS      <- .envl("BENCH_COVS", c("drug", "gender"))
NSEED     <- .envi("BENCH_SEEDS", 2L)
JB_SEEDS  <- .envi("BENCH_JB_SEEDS", 1L)
CHAINS    <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP    <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES   <- .envi("BENCH_JX_SAMPLES", 1000L)
JB_ITER   <- .envi("BENCH_JB_ITER", 3500L)
JB_BURN   <- .envi("BENCH_JB_BURN", 500L)
RHAT_GATE <- 1.05

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
TAG <- Sys.getenv("BENCH_TAG", "study_realdata_aids")
CSV <- file.path(OUTDIR, paste0(TAG, ".csv"))
MU  <- file.path(OUTDIR, paste0(TAG, "_mu.rds"))
LOG <- file.path(OUTDIR, paste0(TAG, "_", format(Sys.time(), "%Y%m%d_%H%M"), ".log"))
.con <- file(LOG, open = "wt"); sink(.con, split = TRUE)
options(error = function() { try(while (sink.number() > 0) sink(), silent = TRUE) })

REG_JX <- "^beta_|^alpha$|^gamma"
REG_JB <- "^betas|^alphas|^gammas"

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
minESS  <- function(e) if (length(e)) min(e, na.rm = TRUE) else NA_real_
whoMin  <- function(e) if (length(e)) names(e)[which.min(e)] else NA_character_
sub_ess <- function(e, pat) e[grepl(pat, names(e))]

done <- if (file.exists(CSV)) utils::read.csv(CSV, stringsAsFactors = FALSE) else NULL
mus  <- if (file.exists(MU)) readRDS(MU) else list()
have <- function(arm, s) {
  if (is.null(done)) return(FALSE)
  hit <- done$arm == arm & done$seed == s
  if (arm == "J" && "jb_iter" %in% names(done)) hit <- hit & done$jb_iter == JB_ITER
  any(hit)
}
append_row <- function(row) utils::write.table(row, CSV, sep = ",", row.names = FALSE,
                          col.names = !file.exists(CSV), append = file.exists(CSV))

cat("=============================================================\n")
cat("  aids real data: jmjax (A/A2/D/E) vs JMbayes2 (J)\n")
cat(sprintf("  covariates: %s | seeds %d (JMbayes2 %d) | chains %d\n",
            paste(COVS, collapse = ", "), NSEED, JB_SEEDS, CHAINS))
cat(sprintf("  jmjax %d/%d | JMbayes2 %d iter / %d burnin %s\n",
            WARMUP, SAMPLES, JB_ITER, JB_BURN,
            if (JB_ITER <= 3500L)
              "(JMbayes2's OWN DEFAULT - an at-defaults comparison, not an\n  efficiency one; expect the R-hat gate to fire)"
            else "(convergence-matched, above JMbayes2's 3500/500 default)"))
if (!is.null(done)) cat(sprintf("  resuming: %d rows already present\n", nrow(done)))
cat("=============================================================\n")

data("aids", package = "JMbayes2")
data("aids.id", package = "JMbayes2")
aids$sqrtCD4    <- sqrt(aids$CD4)
aids.id$sqrtCD4 <- sqrt(aids.id$CD4)

dl <- aids[, c("patient", "obstime", "sqrtCD4", COVS)]
ds <- aids.id[, c("patient", "Time", "death", "drug")]
names(dl)[1] <- "id"; names(ds)[1] <- "id"

lform <- stats::as.formula(paste("sqrtCD4 ~ obstime +", paste(COVS, collapse = " + ")))
sform <- survival::Surv(Time, death) ~ drug

cat(sprintf("\nn = %d patients | %d long obs | %.1f%% events | formula: %s\n",
            length(unique(dl$id)), nrow(dl), 100 * mean(ds$death),
            paste(deparse(lform), collapse = " ")))

lme_fit <- lme(lform, random = ~ obstime | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(sform, data = ds)

X <- stats::model.matrix(lform, data = dl)
cst <- vapply(seq_len(ncol(X)), function(j)
  all(tapply(X[, j], dl$id, function(v) diff(range(v))) < 1e-8), logical(1))
sub <- dl[!duplicated(dl$id), ]; sub <- sub[order(sub$id), ]
S   <- X[!duplicated(dl$id), , drop = FALSE][order(sub$id), cst, drop = FALSE]
IDX <- match(dl$id, sub$id)

jx <- function(orth, all_cols = FALSE, samp_seed) {
  base <- list(n_interior_knots = 5, spline_prior = "penalized",
               rw2_implementation = "vectorized", dense_mass_spline = TRUE,
               num_chains = CHAINS, progress_bar = FALSE, seed = samp_seed)
  if (orth) { if (all_cols) base$orthogonalize_b <- TRUE
              else          base$orthogonalize_b0 <- TRUE }
  ctl_w <- utils::modifyList(base, list(num_warmup = 5L, num_samples = 5L))
  t_cmp <- system.time(suppressWarnings(try(
    jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "obstime",
                  method = "spline-PH-mcmc", control = ctl_w), silent = TRUE)))
  ctl <- utils::modifyList(base, list(num_warmup = WARMUP, num_samples = SAMPLES))
  tm <- system.time(f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds,
                                       time_var = "obstime",
                                       method = "spline-PH-mcmc", control = ctl))
  ps <- f$posterior_samples
  B  <- as_draw_matrix(ps[["beta"]]); B0 <- bcol(ps, 1L, nrow(sub))
  B1 <- bcol(ps, 2L, nrow(sub))
  mu <- as.numeric(X %*% colMeans(B)) + colMeans(B0)[IDX] + colMeans(B1)[IDX] * dl$obstime
  C  <- B0 %*% t(solve(crossprod(S), t(S)))
  e  <- unlist(f$diagnostics$ess)
  list(fit = f, sec = as.numeric(tm[["elapsed"]]), mu = mu,
       compile_sec = as.numeric(t_cmp[["elapsed"]]),
       inv = colMeans(B[, which(cst), drop = FALSE] + C), ess = e)
}

for (s in seq_len(NSEED)) for (arm in c("A", "A2", "D", "E", "J")) {
  if (arm == "J"  && s > JB_SEEDS) next
  if (arm == "A2" && s > 1L) next
  if (have(arm, s)) { cat(sprintf("  seed %d %s: cached\n", s, arm)); next }

  if (arm %in% c("A", "A2", "D", "E")) {
    r <- jx(orth = (arm %in% c("D", "E")), all_cols = (arm == "E"),
            samp_seed = if (arm == "A2") s + 100L else s)
    e <- r$ess
    row <- data.frame(
      seed = s, arm = arm, sec = r$sec, min_ess = minESS(e), min_who = whoMin(e),
      min_ess_reg = minESS(sub_ess(e, REG_JX)), min_who_reg = whoMin(sub_ess(e, REG_JX)),
      beta_obstime_hat = tryCatch(as.numeric(unlist(r$fit$estimates)[["beta_1"]]),
                                  error = function(z) NA_real_),
      alpha_ess = if ("alpha" %in% names(e)) e[["alpha"]] else NA_real_,
      beta0_ess = if ("beta_0" %in% names(e)) e[["beta_0"]] else NA_real_,
      rhat = max(unlist(r$fit$diagnostics$rhat), na.rm = TRUE),
      rhat_reg = { .rh <- unlist(r$fit$diagnostics$rhat)
                   .rh <- .rh[grepl(REG_JX, names(.rh))]
                   if (length(.rh)) max(.rh, na.rm = TRUE) else NA_real_ },
      n_draws = CHAINS * SAMPLES, compile_sec = r$compile_sec, jb_iter = NA_integer_,
      alpha_hat = as.numeric(unlist(r$fit$estimates)[["alpha"]]),
      stringsAsFactors = FALSE)
    mus[[paste(s, arm, sep = "_")]] <- list(mu = r$mu, inv = r$inv, cols = colnames(X)[cst])
    saveRDS(mus, MU)
  } else {
    tm <- system.time(jb <- tryCatch(
      JMbayes2::jm(Surv_object = cox_fit, Mixed_objects = lme_fit,
                   time_var = "obstime", n_chains = CHAINS,
                   n_iter = JB_ITER, n_burnin = JB_BURN,
                   control = list(Bsplines_degree = 3, base_hazard_segments = 6, seed = s)),
      error = function(e) { cat("    JMbayes2 failed:", conditionMessage(e), "\n"); NULL }))
    if (is.null(jb)) next
    .da <- function(l) { if (is.null(l) || !length(l)) return(numeric(0))
                         v <- unlist(l); v[is.finite(v)] }
    eff <- .da(jb$statistics$Effective_Size); eff <- eff[eff > 0]
    eff <- eff[!grepl("^b\\.|frailty|alphaF|sigmaF", names(eff))]
    rh  <- .da(jb$statistics$Rhat); rh <- rh[!grepl("^b\\.|frailty|alphaF|sigmaF", names(rh))]
    row <- data.frame(
      seed = s, arm = "J", sec = as.numeric(tm[["elapsed"]]),
      min_ess = if (length(eff)) min(eff) else NA_real_,
      min_who = if (length(eff)) names(eff)[which.min(eff)] else NA_character_,
      min_ess_reg = { .r <- eff[grepl(REG_JB, names(eff))]; if (length(.r)) min(.r) else NA_real_ },
      min_who_reg = { .r <- eff[grepl(REG_JB, names(eff))]
                      if (length(.r)) names(.r)[which.min(.r)] else NA_character_ },
      beta_obstime_hat = NA_real_,
      alpha_ess = suppressWarnings(as.numeric(jb$statistics$Effective_Size$alphas)[1]),
      beta0_ess = NA_real_, rhat = if (length(rh)) max(rh) else NA_real_,
      rhat_reg = { .rr <- rh[grepl(REG_JB, names(rh))]
                   if (length(.rr)) max(.rr, na.rm = TRUE) else NA_real_ },
      n_draws = CHAINS * (JB_ITER - JB_BURN), compile_sec = NA_real_, jb_iter = JB_ITER,
      alpha_hat = suppressWarnings(as.numeric(jb$statistics$Mean$alphas)[1]),
      stringsAsFactors = FALSE)
  }
  append_row(row); done <- rbind(done, row)
  cat(sprintf("  seed %d %s: %6.1fs  minESS %7.1f (%s)  aESS %7.1f  R-hat %.4f\n",
              s, arm, row$sec, row$min_ess, row$min_who, row$alpha_ess, row$rhat))
}

# =========================== SUMMARY =======================================
R <- utils::read.csv(CSV, stringsAsFactors = FALSE)
R$minsec   <- R$min_ess / R$sec
R$minsecR  <- R$min_ess_reg / R$sec
R$essdrawR <- R$min_ess_reg / R$n_draws
R$essdraw  <- R$min_ess / R$n_draws
R$drawsec  <- R$n_draws / R$sec

cat("\n\n==================== CORRECTNESS (A vs D/E) ====================\n")
cat("  fitted values, treatment against a null built from A at other seeds.\n")
cat("  No algebra: if the arms are the same model this sits near 1x.\n\n")
ss <- sort(unique(R$seed[R$arm == "A"]))
null <- c(); trD <- c(); trE <- c(); zD <- c()
for (i in ss) {
  a  <- mus[[paste(i, "A",  sep = "_")]]; d  <- mus[[paste(i, "D",  sep = "_")]]
  e  <- mus[[paste(i, "E",  sep = "_")]]; a2 <- mus[[paste(i, "A2", sep = "_")]]
  if (is.null(a)) next
  if (!is.null(d)) { trD <- c(trD, mean(abs(d$mu - a$mu))); zD <- c(zD, max(abs(d$inv - a$inv))) }
  if (!is.null(e))   trE <- c(trE, mean(abs(e$mu - a$mu)))
  if (!is.null(a2)) null <- c(null, mean(abs(a2$mu - a$mu)))
}
if (length(trD) && length(null)) {
  cat(sprintf("  null %.3e | D %.3e (%.2fx)  E %s   max|d inv| %.2e\n",
              mean(null), mean(trD), mean(trD) / mean(null),
              if (length(trE)) sprintf("%.3e (%.2fx)", mean(trE), mean(trE) / mean(null)) else "-",
              max(zD, na.rm = TRUE)))
} else cat("  needs >= 2 seeds to build the null; skipped\n")

cat("\n==================== EFFICIENCY ====================\n")
gm <- function(v) { v <- v[is.finite(v) & v > 0]; if (!length(v)) NA_real_ else exp(mean(log(v))) }
A <- R[R$arm == "A", ]; D <- R[R$arm == "D", ]; E <- R[R$arm == "E", ]
J <- R[R$arm == "J", ]; if (nrow(J) && "jb_iter" %in% names(J)) J <- J[J$jb_iter == JB_ITER, ]
.cols <- c("seed", "minsec", "minsecR", "essdraw", "essdrawR", "drawsec",
           "beta0_ess", "sec", "rhat", "compile_sec")
m <- merge(A[, .cols], D[, .cols], by = "seed", suffixes = c("_A", "_D"))
if (nrow(E)) { .e <- E[, .cols]; names(.e)[-1] <- paste0(names(.e)[-1], "_E"); m <- merge(m, .e, by = "seed") }

.fail <- character(0)
if (any(c(A$rhat, D$rhat, E$rhat) >= RHAT_GATE, na.rm = TRUE)) .fail <- c(.fail, "jmjax")
if (nrow(J) && any(J$rhat >= RHAT_GATE, na.rm = TRUE)) .fail <- c(.fail, "JMbayes2")
if (length(.fail)) {
  cat(sprintf("  ** R-hat GATE (ALL PARAMS) FAILED: %s (max %.4f vs gate %.2f)\n",
              paste(.fail, collapse = " and "),
              max(c(A$rhat, D$rhat, E$rhat, J$rhat), na.rm = TRUE), RHAT_GATE))
  if (identical(.fail, "JMbayes2"))
    cat("  Overall ALL-PARAMS comparison is at-defaults, not efficiency, until\n",
        "  this clears. Checking separately whether it is a REGRESSION-parameter\n",
        "  problem or confined to nuisance parameters (covariance elements,\n",
        "  frailty, etc.) that nobody reports:\n", sep = "")
}
.failR <- character(0)
if ("rhat_reg" %in% names(A) && any(c(A$rhat_reg, D$rhat_reg, E$rhat_reg) >= RHAT_GATE, na.rm = TRUE))
  .failR <- c(.failR, "jmjax")
if (nrow(J) && "rhat_reg" %in% names(J) && any(J$rhat_reg >= RHAT_GATE, na.rm = TRUE))
  .failR <- c(.failR, "JMbayes2")
if (length(.failR)) {
  cat(sprintf("  ** R-hat GATE (REGRESSION ONLY) FAILED: %s (max %.4f vs gate %.2f)\n",
              paste(.failR, collapse = " and "),
              max(c(A$rhat_reg, D$rhat_reg, E$rhat_reg, J$rhat_reg), na.rm = TRUE), RHAT_GATE))
} else if (length(.fail)) {
  cat("  -- regression-only R-hat gate PASSES for every arm, including J: the\n",
      "     min-ESS-on-regression-parameters comparison below can be trusted\n",
      "     even though the ALL-PARAMS one above cannot. --\n", sep = "")
}
cat("  A = defaults | D = orth intercept | E = orth all b columns\n")
cat(sprintf("  binding(all)  A: %-14s D: %-14s E: %s\n",
            paste(unique(A$min_who), collapse = "/"), paste(unique(D$min_who), collapse = "/"),
            if (nrow(E)) paste(unique(E$min_who), collapse = "/") else "-"))
cat(sprintf("  binding(reg)  A: %-14s D: %-14s E: %s\n",
            paste(unique(A$min_who_reg), collapse = "/"), paste(unique(D$min_who_reg), collapse = "/"),
            if (nrow(E)) paste(unique(E$min_who_reg), collapse = "/") else "-"))
.line <- function(lab, va, vd, ve, fmt = "%8.2f") {
  cat(sprintf(paste0("  %-13s ", fmt, "  ", fmt, " (%.2fx)  %s\n"),
              lab, gm(va), gm(vd), gm(vd / va),
              if (!is.null(ve)) sprintf(paste0(fmt, " (%.2fx)"), gm(ve), gm(ve / va)) else "-"))
}
cat("  -- ALL PARAMETERS (the CONVERGENCE view) --\n")
.line("ESS/draw", m$essdraw_A, m$essdraw_D, if (nrow(E)) m$essdraw_E, "%8.4f")
.line("draws/sec", m$drawsec_A, m$drawsec_D, if (nrow(E)) m$drawsec_E, "%8.1f")
.line("ESS/sec", m$minsec_A, m$minsec_D, if (nrow(E)) m$minsec_E)
cat("  -- REGRESSION ONLY: beta, alpha, gamma (the REPORTING view) --\n")
.line("ESS/draw", m$essdrawR_A, m$essdrawR_D, if (nrow(E)) m$essdrawR_E, "%8.4f")
.line("ESS/sec", m$minsecR_A, m$minsecR_D, if (nrow(E)) m$minsecR_E)
if (nrow(m) >= 2 && nrow(E)) {
  h <- stats::t.test(log(m$minsecR_E / m$minsecR_A))
  cat(sprintf("    E vs A: p = %.4f  95%% CI [%.2fx, %.2fx]\n",
              h$p.value, exp(h$conf.int[1]), exp(h$conf.int[2])))
}
cat(sprintf("  wall (warm)   %8.1fs %8.1fs %s   + %.1fs JIT\n",
            mean(m$sec_A), mean(m$sec_D),
            if (nrow(E)) sprintf("%8.1fs", mean(m$sec_E)) else "-",
            mean(c(m$compile_sec_A, m$compile_sec_D), na.rm = TRUE)))
cat(sprintf("  max R-hat     %8.4f %8.4f %s\n", max(m$rhat_A), max(m$rhat_D),
            if (nrow(E)) sprintf("%8.4f", max(m$rhat_E)) else "-"))
if (nrow(J)) {
  cat(sprintf("  JMbayes2     %8.1fs  %d draws  R-hat %.4f\n",
              mean(J$sec), J$n_draws[1], max(J$rhat, na.rm = TRUE)))
  cat(sprintf("    min ESS/draw  J %8.4f   jmjax A %.2fx   D %.2fx  <- sampler\n",
              gm(J$essdraw), gm(m$essdraw_A) / gm(J$essdraw), gm(m$essdraw_D) / gm(J$essdraw)))
  cat(sprintf("    draws/sec     J %8.1f   jmjax A %.2fx   D %.2fx  <- machine\n",
              gm(J$drawsec), gm(m$drawsec_A) / gm(J$drawsec), gm(m$drawsec_D) / gm(J$drawsec)))
  cat(sprintf("    min ESS/sec   J %8.2f   jmjax A %.2fx   D %.2fx  = product\n",
              gm(J$minsec), gm(m$minsec_A) / gm(J$minsec), gm(m$minsec_D) / gm(J$minsec)))
  J$minsecR  <- J$min_ess_reg / J$sec; J$essdrawR <- J$min_ess_reg / J$n_draws
  cat(sprintf("    -- regression parameters only (J binds on %s) --\n",
              paste(unique(J$min_who_reg), collapse = "/")))
  cat(sprintf("    min ESS/draw  J %8.4f   jmjax A %.2fx   D %.2fx   E %s\n",
              gm(J$essdrawR), gm(m$essdrawR_A) / gm(J$essdrawR), gm(m$essdrawR_D) / gm(J$essdrawR),
              if (nrow(E)) sprintf("%.2fx", gm(m$essdrawR_E) / gm(J$essdrawR)) else "-"))
  cat(sprintf("    min ESS/sec   J %8.2f   jmjax A %.2fx   D %.2fx   E %s\n",
              gm(J$minsecR), gm(m$minsecR_A) / gm(J$minsecR), gm(m$minsecR_D) / gm(J$minsecR),
              if (nrow(E)) sprintf("%.2fx", gm(m$minsecR_E) / gm(J$minsecR)) else "-"))
}

cat("\n==================== ESTIMATES (no known truth - cross-arm/package agreement) ====================\n")
cat("  A and D must agree closely: the reparameterization cannot move alpha or beta_obstime.\n")
cat(sprintf("  %-6s %10s %16s\n", "arm", "alpha", "beta_obstime"))
for (a in c("A", "D", "E", "J")) {
  y <- R[R$arm == a & is.finite(R$alpha_hat), ]
  if (!nrow(y)) next
  bo <- R[R$arm == a & is.finite(R$beta_obstime_hat), ]
  cat(sprintf("  %-6s %10.4f %16s\n", a, mean(y$alpha_hat),
              if (nrow(bo)) sprintf("%.4f", mean(bo$beta_obstime_hat)) else "-"))
}

cat("\nwrote ", CSV, "\n", sep = "")
sink(); close(.con)
