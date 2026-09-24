# ==============================================================================
# The study that decides whether orthogonalize_b0 becomes the default, and
# what jmjax can honestly claim against JMbayes2 at realistic sample sizes.
#
# WHAT IS ALREADY ESTABLISHED (pbc2 only, n = 312 and a duplicate at 624):
#   correctness  k=2 over 8 replicates and k=4 over 2 seeds, verified by a
#                k-general invariant AND by fitted values against a re-seeded
#                null - the latter needing no algebra and so unable to be
#                fooled by a projection I got wrong (it caught me three times).
#   efficiency   beta_0 ESS 15-27x on 12 of 12 replicates; min ESS/sec 1.89x
#                (CI [0.91, 3.91] - INCLUDES 1), 2.06x and 4.91x; wall time
#                equal or better; max R-hat better in all three conditions.
#
# WHAT IS NOT:
#   - one dataset and its duplicate. dense_mass_beta looked this good too.
#   - k = 1 untested. That is the case the validation suite covers and the
#     case with least to gain, so it is where HARM would show up first. A
#     default must be safe there, not merely unhelpful.
#   - nothing above n = 624, and nothing at all on simulated data where the
#     truth is known.
#   - no JMbayes2 arm, so none of it says what a user actually gets.
#
# DESIGN. Three arms per cell, on simulated data with known truth:
#     A  jmjax, defaults
#     D  jmjax, orthogonalize_b0 = TRUE
#     J  JMbayes2, at a CONVERGENCE-MATCHED budget, not its default
#
# The budget point matters and is the reason section 5.3 of the white paper
# needs regenerating: pairing each package's own default compares a converged
# fit against an unconverged one. JMbayes2 measured max R-hat 1.068 at 8,000
# iterations on pbc2 and needed 16,000 to reach 1.025. Every cell here reports
# R-hat for both packages and the summary EXCLUDES any cell where either side
# failed the gate, rather than quoting a ratio across it.
#
# METRICS. min ESS/sec is the headline, not alpha. alpha is jmjax's best
# parameter and is structurally immune to the location degeneracy - it lives
# in the survival submodel and never touches the subject-constant column
# space - which is precisely why reporting it alone concealed this problem
# for so long. alpha is reported too, for continuity with published figures.
#
# RESUMABLE. Every row is appended to the CSV as it completes and re-reading
# that CSV on startup skips finished cells, so an interrupted run loses at
# most one fit.
#
#   caffeinate -i Rscript dev/study_orth_vs_jmbayes2.R
#   BENCH_K_LIST=3 BENCH_N_LIST=500 BENCH_SEEDS=1 Rscript dev/study_orth_vs_jmbayes2.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 is required for the comparison arm.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))
`%||%` <- function(a, b) if (is.null(a)) b else a

# Rscript gives no sys.frame()$ofile, so find this script via --file= and fall
# back to the conventional path when sourced interactively.
.self <- sub("^--file=", "",
             grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.gen <- if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R"
if (!file.exists(.gen)) .gen <- "dev/sim_joint.R"
if (!file.exists(.gen)) stop("cannot find sim_joint.R next to this script", call. = FALSE)
source(.gen)
.envl <- function(nm, d) {
  v <- Sys.getenv(nm, "")
  if (!nzchar(v)) return(d)
  as.integer(strsplit(v, "[,[:space:]]+")[[1]])
}
.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
K_LIST    <- .envl("BENCH_K_LIST", c(1L, 3L))
ARMS      <- { v <- Sys.getenv("BENCH_ARMS", "")
               if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]]
               else c("A", "A2", "D", "E", "J") }
N_LIST    <- .envl("BENCH_N_LIST", c(500L, 1000L, 1200L))
NSEED     <- .envi("BENCH_SEEDS", 3L)
JB_SEEDS  <- .envi("BENCH_JB_SEEDS", 2L)
CHAINS    <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP    <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES   <- .envi("BENCH_JX_SAMPLES", 1000L)
JB_ITER   <- .envi("BENCH_JB_ITER", 16000L)
JB_BURN   <- .envi("BENCH_JB_BURN", 4000L)
RHAT_GATE <- 1.05

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
TAG <- Sys.getenv("BENCH_TAG", "study_orth_vs_jmbayes2")
CSV <- file.path(OUTDIR, paste0(TAG, ".csv"))
MU  <- file.path(OUTDIR, paste0(TAG, "_mu.rds"))
LOG <- file.path(OUTDIR, paste0(TAG, "_", format(Sys.time(), "%Y%m%d_%H%M"), ".log"))
.con <- file(LOG, open = "wt"); sink(.con, split = TRUE)
options(error = function() { try(while (sink.number() > 0) sink(), silent = TRUE) })

done <- if (file.exists(CSV)) utils::read.csv(CSV, stringsAsFactors = FALSE) else NULL
mus  <- if (file.exists(MU)) readRDS(MU) else list()

cat("=============================================================\n")
cat("  orthogonalize_b0 vs JMbayes2, simulated data, known truth\n")
cat(sprintf("  k %s | n %s | seeds %d (JMbayes2 %d) | chains %d\n",
            paste(K_LIST, collapse = ","), paste(N_LIST, collapse = ","),
            NSEED, JB_SEEDS, CHAINS))
cat(sprintf("  jmjax %d/%d | JMbayes2 %d iter / %d burnin %s\n",
            WARMUP, SAMPLES, JB_ITER, JB_BURN,
            if (JB_ITER <= 3500L)
              "(JMbayes2's OWN DEFAULT - an at-defaults comparison, not an\n  efficiency one; expect the R-hat gate to fire)"
            else "(convergence-matched, above JMbayes2's 3500/500 default)"))
if (!is.null(done)) cat(sprintf("  resuming: %d rows already present\n", nrow(done)))
cat("=============================================================\n")

ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE, demean = TRUE)$acf[, 1, 1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) { pr <- ac[k] + ac[k + 1L]; if (pr <= 0) break
    s <- s + pr; k <- k + 2L }
  tau <- -1 + 2 * s; if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}
as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
bcol <- function(ps, j, nsub) {
  b <- ps[["b"]]; if (is.null(b)) return(NULL)
  if (is.list(b)) return(t(vapply(b, function(d) {
    if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
    else { a <- as.matrix(d); as.numeric(a[, j]) }
  }, numeric(nsub))))
  a <- as.array(b); if (length(dim(a)) == 3L) a[, , j] else if (j == 1L) as.matrix(a) else NULL
}
# ---- which parameters are ESTIMANDS and which are nuisance ---------------
# Declared here, before any result is seen, because choosing the subset
# afterwards is exactly what makes a headline figure indefensible - it is
# the same move as section 5.3 quoting alpha alone.
#
# REGRESSION: what a reader of the paper interprets - the longitudinal fixed
# effects, the association parameter, and the survival covariate effects.
# NUISANCE: the random-effects covariance, the residual scale, and the
# baseline-hazard spline. Called nuisance BY DECLARATION, not because it is
# obvious: sigma_b0 in particular is central to dynamic prediction, which is
# a main use of joint models. Anyone reporting predictions should read the
# all-parameter column instead.
#
# Both are reported. min over ALL parameters remains the CONVERGENCE
# criterion - if rho has not mixed, the chain has not explored the posterior
# and the beta draws are conditional on a badly-explored rho, so the betas
# are not trustworthy however well they mix on their own. min over
# REGRESSION is the reporting criterion. Neither replaces the other.
#
# The subset is not a jmjax flattery device: JMbayes2's binding parameter in
# the at-defaults run was bs_gammas.bs_gammas_2, a baseline-hazard spline
# coefficient, so restricting to estimands raises ITS number too.
REG_JX <- "^beta_|^alpha$|^gamma"
REG_JB <- "^betas|^alphas|^gammas"

ess_ok  <- function(e) { e <- e[is.finite(e) & e > 0]; if (!length(e)) NA_real_ else e }
sub_ess <- function(e, pat) { e <- ess_ok(e); e[grepl(pat, names(e))] }
minESS  <- function(e) { e <- ess_ok(e); if (all(is.na(e))) NA_real_ else min(e) }
whoMin  <- function(e) { e <- ess_ok(e); if (all(is.na(e))) NA_character_ else names(e)[which.min(e)] }

append_row <- function(r) {
  utils::write.table(r, CSV, sep = ",", row.names = FALSE,
                     col.names = !file.exists(CSV), append = file.exists(CSV))
}
have <- function(k, n, s, arm) {
  if (is.null(done)) return(FALSE)
  hit <- done$k == k & done$n == n & done$seed == s & done$arm == arm
  # A cached JMbayes2 row is only reusable at the SAME budget.
  if (arm == "J" && "jb_iter" %in% names(done)) hit <- hit & done$jb_iter == JB_ITER
  any(hit)
}

for (kk in K_LIST) for (nn in N_LIST) {
  k_extra <- kk - 1L
  cat(sprintf("\n=============================================================\n"))
  cat(sprintf("  k = %d | n = %d\n", kk, nn))
  cat(  "=============================================================\n")
  for (s in seq_len(NSEED)) {
    # DATA seed and SAMPLER seed are separate. Conflating them made the
    # correctness null compare fits on DIFFERENT DATASETS - different
    # subjects, different covariates - which reported a null of 1.6e+01
    # against a treatment of 1.1e-02 and a ratio of 0.00x. The null only
    # means anything when the data is held fixed and ONLY the sampler seed
    # moves, which is what arm A2 below provides.
    sim <- sim_joint(n = nn, seed = 1000L * kk + s, k_extra = k_extra)
    dl  <- sim$data_long; ds <- sim$data_surv
    if (s == 1L)
      cat(sprintf("  event rate %.1f%% | %.1f obs/subject | %d long obs\n",
                  100 * sim$event_rate, sim$obs_per_subject, sim$n_obs))

    covs <- if (k_extra <= 3L) c("age", "sex", "trt")[seq_len(k_extra)]
            else c("age", "sex", "trt", paste0("x", 4:k_extra))
    lform <- stats::as.formula(paste("y ~ time",
                                     if (length(covs)) paste("+", paste(covs, collapse = " + ")) else ""))
    sform <- survival::Surv(time, event) ~ trt

    lme_fit <- lme(lform, random = ~ time | id, data = dl,
                   control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
    cox_fit <- coxph(sform, data = ds)

    # subject-constant design columns, found the same way the backend does
    X <- stats::model.matrix(lform, data = dl)
    cst <- vapply(seq_len(ncol(X)), function(j)
      all(tapply(X[, j], dl$id, function(v) diff(range(v))) < 1e-8), logical(1))
    sub <- dl[!duplicated(dl$id), ]; sub <- sub[order(sub$id), ]
    S   <- X[!duplicated(dl$id), , drop = FALSE][order(sub$id), cst, drop = FALSE]
    IDX <- match(dl$id, sub$id)

    jx <- function(orth, all_cols = FALSE, samp_seed = s) {
      base <- list(n_interior_knots = 5, spline_prior = "penalized",
                   rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                    num_chains = CHAINS, progress_bar = FALSE,
                   seed = samp_seed)
      if (orth) {
        if (all_cols) base$orthogonalize_b <- TRUE
        else          base$orthogonalize_b0 <- TRUE
      }

      # ---- JIT warm-up -----------------------------------------------------
      # JAX compiles the leapfrog step for each DATA SHAPE on first call and
      # caches it. Compilation is therefore charged entirely to whichever fit
      # runs first and is free for every fit after - which makes a single
      # cold timing a measure of the compiler as much as of the sampler, and
      # makes it depend on the order the arms happen to run in.
      #
      # A 5+5 draw fit touches the same shapes and so triggers the same
      # compilation, at negligible sampling cost. Its elapsed time is an
      # upper bound on the compile overhead and is recorded rather than
      # discarded: a user's FIRST fit really does pay it, so it belongs in
      # the report, just not inside the steady-state throughput number.
      ctl_w <- utils::modifyList(base, list(num_warmup = 5L, num_samples = 5L))
      t_cmp <- system.time(suppressWarnings(try(
        jmjax:::jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                      method = "spline-PH-mcmc", control = ctl_w),
        silent = TRUE)))

      ctl <- utils::modifyList(base, list(num_warmup = WARMUP,
                                          num_samples = SAMPLES))
      tm <- system.time(f <- jmjax:::jm_fit_prefit(lme_fit, cox_fit, data_surv = ds,
                                           time_var = "time",
                                           method = "spline-PH-mcmc", control = ctl))
      ps <- f$posterior_samples
      B  <- as_draw_matrix(ps[["beta"]]); B0 <- bcol(ps, 1L, nrow(sub))
      B1 <- bcol(ps, 2L, nrow(sub))
      mu <- as.numeric(X %*% colMeans(B)) + colMeans(B0)[IDX] + colMeans(B1)[IDX] * dl$time
      C  <- B0 %*% t(solve(crossprod(S), t(S)))
      e  <- unlist(f$diagnostics$ess)
      list(fit = f, sec = as.numeric(tm[["elapsed"]]), mu = mu,
           compile_sec = as.numeric(t_cmp[["elapsed"]]),
           inv = colMeans(B[, which(cst), drop = FALSE] + C),
           ess = e)
    }

    # A2 is the correctness NULL: identical data and settings to A, only the
    # sampler seed differs. One per cell is enough to calibrate the scale, so
    # it runs on the first replicate only rather than costing a fit each time.
    for (arm in ARMS) {
      if (arm == "J"  && s > JB_SEEDS) next
      if (arm == "A2" && s > 1L) next
      if (have(kk, nn, s, arm)) { cat(sprintf("  seed %d %s: cached\n", s, arm)); next }

      if (arm %in% c("A", "A2", "D", "E")) {
        # D constrains the INTERCEPT column only; E constrains every
        # random-effect column, which is what beta_1 needs - beta_1 + mean(b_i1)
        # is identified exactly as beta_0 + mean(b_i0) is, and D leaves the
        # slope column untouched. Both arms are kept so the slope half can be
        # attributed separately from the intercept half.
        r <- jx(orth = (arm %in% c("D", "E")), all_cols = (arm == "E"),
                samp_seed = if (arm == "A2") s + 100L else s)
        e <- r$ess
        row <- data.frame(
          k = kk, n = nn, seed = s, arm = arm,
          sec = r$sec, min_ess = minESS(e), min_who = whoMin(e),
          min_ess_reg = minESS(sub_ess(e, REG_JX)),
          min_who_reg = whoMin(sub_ess(e, REG_JX)),
          beta1_hat = tryCatch(as.numeric(unlist(r$fit$estimates)[["beta_1"]]),
                               error = function(z) NA_real_),
          beta1_true = sim$truth$beta1,
          alpha_ess = if ("alpha" %in% names(e)) e[["alpha"]] else NA_real_,
          beta0_ess = if ("beta_0" %in% names(e)) e[["beta_0"]] else NA_real_,
          rhat = max(unlist(r$fit$diagnostics$rhat), na.rm = TRUE),
          rhat_reg = { .rh <- unlist(r$fit$diagnostics$rhat)
                       .rh <- .rh[grepl(REG_JX, names(.rh))]
                       if (length(.rh)) max(.rh, na.rm = TRUE) else NA_real_ },
          n_draws = CHAINS * SAMPLES, compile_sec = r$compile_sec,
          jb_iter = NA_integer_,
          alpha_hat = as.numeric(unlist(r$fit$estimates)[["alpha"]]),
          alpha_true = sim$truth$alpha,
          event_rate = sim$event_rate, n_obs = sim$n_obs,
          stringsAsFactors = FALSE)
        mus[[paste(kk, nn, s, arm, sep = "_")]] <-
          list(mu = r$mu, inv = r$inv, cols = colnames(X)[cst])
        saveRDS(mus, MU)
      } else {
        tm <- system.time(jb <- tryCatch(
          JMbayes2::jm(Surv_object = cox_fit, Mixed_objects = lme_fit,
                       time_var = "time", n_chains = CHAINS,
                       n_iter = JB_ITER, n_burnin = JB_BURN,
                       control = list(Bsplines_degree = 3,
                                      base_hazard_segments = 6, seed = s)),
          error = function(e) { cat("    JMbayes2 failed:", conditionMessage(e), "\n"); NULL }))
        if (is.null(jb)) next
        # Field names and the frailty filter come from dev/bench_pbc2.R, which
        # established them against a real fit. Never-sampled slots (alphaF,
        # frailty*, sigmaF on data with no frailty term) return ESS exactly 0
        # and a meaningless R-hat; dropping them by VALUE rather than by name
        # is what stops min ESS being reported as 0 for a parameter the model
        # does not contain.
        .da <- function(l) { if (is.null(l) || !length(l)) return(numeric(0))
                             v <- unlist(l); v[is.finite(v)] }
        eff <- .da(jb$statistics$Effective_Size)
        eff <- eff[eff > 0]
        eff <- eff[!grepl("^b\\.|frailty|alphaF|sigmaF", names(eff))]
        rh  <- .da(jb$statistics$Rhat)
        rh  <- rh[!grepl("^b\\.|frailty|alphaF|sigmaF", names(rh))]
        row <- data.frame(
          k = kk, n = nn, seed = s, arm = "J",
          sec = as.numeric(tm[["elapsed"]]),
          min_ess = if (length(eff)) min(eff) else NA_real_,
          min_who = if (length(eff)) names(eff)[which.min(eff)] else NA_character_,
          min_ess_reg = { .r <- eff[grepl(REG_JB, names(eff))]
                          if (length(.r)) min(.r) else NA_real_ },
          min_who_reg = { .r <- eff[grepl(REG_JB, names(eff))]
                          if (length(.r)) names(.r)[which.min(.r)] else NA_character_ },
          beta1_hat = NA_real_, beta1_true = sim$truth$beta1,
          alpha_ess = suppressWarnings(as.numeric(
            jb$statistics$Effective_Size$alphas)[1]),
          beta0_ess = NA_real_,
          rhat = if (length(rh)) max(rh) else NA_real_,
          rhat_reg = { .rr <- rh[grepl(REG_JB, names(rh))]
                       if (length(.rr)) max(.rr, na.rm = TRUE) else NA_real_ },
          n_draws = CHAINS * (JB_ITER - JB_BURN), compile_sec = NA_real_,
          jb_iter = JB_ITER,
          alpha_hat = suppressWarnings(as.numeric(jb$statistics$Mean$alphas)[1]),
          alpha_true = sim$truth$alpha,
          event_rate = sim$event_rate, n_obs = sim$n_obs,
          stringsAsFactors = FALSE)
      }
      append_row(row)
      done <- rbind(done, row)
      cat(sprintf("  seed %d %s: %6.1fs  minESS %7.1f (%s)  aESS %7.1f  R-hat %.4f\n",
                  s, arm, row$sec, row$min_ess, row$min_who, row$alpha_ess, row$rhat))
    }
  }
}

# =========================== SUMMARY =======================================
R <- utils::read.csv(CSV, stringsAsFactors = FALSE)
# ESS/sec is a PRODUCT of two quantities that behave quite differently:
#   ESS/draw   statistical efficiency - how much independent information the
#              sampler extracts per iteration. Hardware-free, so it is the
#              only one of the three that is a property of the SAMPLERS.
#   draws/sec  throughput - cores, threading, BLAS, JIT, thermal state. A
#              property of the machine on the afternoon it was run.
# Reporting only the product is how the earlier "12x" survived: it was about
# 6x draw count and 2x per-draw mixing, and the product hid that. Both are
# reported below, with their product, and none of the three is sufficient on
# its own - ESS/draw flatters NUTS (63 gradient evaluations per draw against
# a Gibbs/MH sweep's one or two), just as draws/sec flatters whoever got
# more cores.
R$minsec   <- R$min_ess / R$sec
R$minsecR  <- R$min_ess_reg / R$sec
R$essdrawR <- R$min_ess_reg / R$n_draws
R$essdraw  <- R$min_ess / R$n_draws
R$drawsec  <- R$n_draws / R$sec
cat("\n\n==================== CORRECTNESS (A vs D) ====================\n")
cat("  fitted values, treatment against a null built from A at other seeds.\n")
cat("  No algebra: if the arms are the same model this sits near 1x.\n\n")
for (kk in unique(R$k)) for (nn in unique(R$n[R$k == kk])) {
  ss <- sort(unique(R$seed[R$k == kk & R$n == nn & R$arm == "A"]))
  null <- c(); trD <- c(); trE <- c(); zD <- c(); zE <- c()
  for (i in ss) {
    a  <- mus[[paste(kk, nn, i, "A",  sep = "_")]]
    d  <- mus[[paste(kk, nn, i, "D",  sep = "_")]]
    e  <- mus[[paste(kk, nn, i, "E",  sep = "_")]]
    a2 <- mus[[paste(kk, nn, i, "A2", sep = "_")]]   # same data, other seed
    if (is.null(a)) next
    if (!is.null(d)) { trD <- c(trD, mean(abs(d$mu - a$mu)))
                       zD  <- c(zD,  max(abs(d$inv - a$inv))) }
    # E constrains the SLOPE column as well, which the intercept-only
    # invariant does not describe - so for E the fitted-value null is the
    # only check that applies, and it is the one that matters anyway.
    if (!is.null(e))   trE <- c(trE, mean(abs(e$mu - a$mu)))
    if (!is.null(a2)) null <- c(null, mean(abs(a2$mu - a$mu)))
  }
  treat <- trD; zinv <- if (length(zD)) zD else NA_real_
  if (!length(treat) || !length(null)) {
    cat(sprintf("  k=%d n=%4d   needs >= 2 seeds to build the null; skipped\n", kk, nn))
    next
  }
  cat(sprintf("  k=%d n=%4d   null %.3e | D %.3e (%.2fx)  E %s   max|d inv| %.2e\n",
              kk, nn, mean(null), mean(treat), mean(treat) / mean(null),
              if (length(trE)) sprintf("%.3e (%.2fx)", mean(trE),
                                       mean(trE) / mean(null)) else "-",
              max(zinv, na.rm = TRUE)))
}

cat("\n==================== EFFICIENCY ====================\n")
gm <- function(v) { v <- v[is.finite(v) & v > 0]; if (!length(v)) NA_real_ else exp(mean(log(v))) }
for (kk in unique(R$k)) for (nn in unique(R$n[R$k == kk])) {
  x <- R[R$k == kk & R$n == nn, ]
  A <- x[x$arm == "A", ]; D <- x[x$arm == "D", ]
  J <- x[x$arm == "J", ]
  # One CSV can now hold JMbayes2 at several budgets; report the one this
  # invocation asked for rather than mixing them into a single mean.
  if (nrow(J) && "jb_iter" %in% names(J)) J <- J[J$jb_iter == JB_ITER, ]
  # A2 exists only to calibrate the correctness null and must never enter a
  # ratio: it would double-count arm A at a different sampler seed.
  .cols <- c("seed", "minsec", "minsecR", "essdraw", "essdrawR", "drawsec",
             "beta0_ess", "sec", "rhat", "compile_sec")
  E <- x[x$arm == "E", ]
  m <- merge(A[, .cols], D[, .cols], by = "seed", suffixes = c("_A", "_D"))
  if (nrow(E)) {
    .e <- E[, .cols]; names(.e)[-1] <- paste0(names(.e)[-1], "_E")
    m <- merge(m, .e, by = "seed")
  }
  if (!nrow(m)) next
  .fail <- character(0)
  if (any(c(A$rhat, D$rhat, E$rhat) >= RHAT_GATE, na.rm = TRUE)) .fail <- c(.fail, "jmjax")
  if (nrow(J) && any(J$rhat >= RHAT_GATE, na.rm = TRUE))  .fail <- c(.fail, "JMbayes2")
  cat(sprintf("\n  k=%d  n=%4d\n", kk, nn))
  if (length(.fail)) {
    cat(sprintf("    ** R-hat GATE (ALL PARAMS) FAILED: %s (max %.4f vs gate %.2f)\n",
                paste(.fail, collapse = " and "),
                max(c(A$rhat, D$rhat, E$rhat, J$rhat), na.rm = TRUE), RHAT_GATE))
    if (identical(.fail, "JMbayes2"))
      cat("    Overall ALL-PARAMS comparison is at-defaults, not efficiency, until\n",
          "    this clears. Checking separately whether it is a REGRESSION-parameter\n",
          "    problem or confined to nuisance parameters (covariance elements,\n",
          "    frailty, etc.) that nobody reports:\n", sep = "")
  }
  .failR <- character(0)
  if ("rhat_reg" %in% names(A) && any(c(A$rhat_reg, D$rhat_reg, E$rhat_reg) >= RHAT_GATE, na.rm = TRUE))
    .failR <- c(.failR, "jmjax")
  if (nrow(J) && "rhat_reg" %in% names(J) && any(J$rhat_reg >= RHAT_GATE, na.rm = TRUE))
    .failR <- c(.failR, "JMbayes2")
  if (length(.failR)) {
    cat(sprintf("    ** R-hat GATE (REGRESSION ONLY) FAILED: %s (max %.4f vs gate %.2f)\n",
                paste(.failR, collapse = " and "),
                max(c(A$rhat_reg, D$rhat_reg, E$rhat_reg, J$rhat_reg), na.rm = TRUE), RHAT_GATE))
  } else if (length(.fail)) {
    cat("    -- regression-only R-hat gate PASSES for every arm, including J: the\n",
        "       min-ESS-on-regression-parameters comparison below can be trusted\n",
        "       even though the ALL-PARAMS one above cannot. --\n", sep = "")
  }
  cat(sprintf("    A = defaults | D = orth intercept | E = orth all b columns\n"))
  cat(sprintf("    binding(all)  A: %-14s D: %-14s E: %s\n",
              paste(unique(A$min_who), collapse = "/"),
              paste(unique(D$min_who), collapse = "/"),
              if (nrow(E)) paste(unique(E$min_who), collapse = "/") else "-"))
  cat(sprintf("    binding(reg)  A: %-14s D: %-14s E: %s\n",
              paste(unique(A$min_who_reg), collapse = "/"),
              paste(unique(D$min_who_reg), collapse = "/"),
              if (nrow(E)) paste(unique(E$min_who_reg), collapse = "/") else "-"))
  cat(sprintf("    beta_0 ESS    %8.1f  %8.1f (%.2fx)  %s\n",
              gm(m$beta0_ess_A), gm(m$beta0_ess_D),
              gm(m$beta0_ess_D / m$beta0_ess_A),
              if (nrow(E)) sprintf("%8.1f (%.2fx)", gm(m$beta0_ess_E),
                                   gm(m$beta0_ess_E / m$beta0_ess_A)) else "-"))
  .line <- function(lab, va, vd, ve, fmt = "%8.2f") {
    cat(sprintf(paste0("    %-13s ", fmt, "  ", fmt, " (%.2fx)  %s\n"),
                lab, gm(va), gm(vd), gm(vd / va),
                if (!is.null(ve)) sprintf(paste0(fmt, " (%.2fx)"),
                                          gm(ve), gm(ve / va)) else "-"))
  }
  cat("    -- ALL PARAMETERS (the CONVERGENCE view) --\n")
  .line("ESS/draw", m$essdraw_A, m$essdraw_D, if (nrow(E)) m$essdraw_E, "%8.4f")
  .line("draws/sec", m$drawsec_A, m$drawsec_D, if (nrow(E)) m$drawsec_E, "%8.1f")
  .line("ESS/sec", m$minsec_A, m$minsec_D, if (nrow(E)) m$minsec_E)
  if (nrow(m) >= 2 && nrow(E)) {
    h <- stats::t.test(log(m$minsec_E / m$minsec_A))
    cat(sprintf("      E vs A: p = %.4f  95%% CI [%.2fx, %.2fx]\n",
                h$p.value, exp(h$conf.int[1]), exp(h$conf.int[2])))
  }
  cat("    -- REGRESSION ONLY: beta, alpha, gamma (the REPORTING view) --\n")
  .line("ESS/draw", m$essdrawR_A, m$essdrawR_D, if (nrow(E)) m$essdrawR_E, "%8.4f")
  .line("ESS/sec", m$minsecR_A, m$minsecR_D, if (nrow(E)) m$minsecR_E)
  if (nrow(m) >= 2 && nrow(E)) {
    h <- stats::t.test(log(m$minsecR_E / m$minsecR_A))
    cat(sprintf("      E vs A: p = %.4f  95%% CI [%.2fx, %.2fx]\n",
                h$p.value, exp(h$conf.int[1]), exp(h$conf.int[2])))
  }
  cat(sprintf("    wall (warm)   %8.1fs %8.1fs %s   + %.1fs JIT\n",
              mean(m$sec_A), mean(m$sec_D),
              if (nrow(E)) sprintf("%8.1fs", mean(m$sec_E)) else "-",
              mean(c(m$compile_sec_A, m$compile_sec_D), na.rm = TRUE)))
  cat(sprintf("    max R-hat     %8.4f %8.4f %s\n",
              max(m$rhat_A), max(m$rhat_D),
              if (nrow(E)) sprintf("%8.4f", max(m$rhat_E)) else "-"))
  if (nrow(J)) {
    cat(sprintf("    JMbayes2     %8.1fs  %d draws  R-hat %.4f\n",
                mean(J$sec), J$n_draws[1], max(J$rhat, na.rm = TRUE)))
    cat(sprintf("      min ESS/draw  J %8.4f   jmjax A %.2fx   D %.2fx  <- sampler\n",
                gm(J$essdraw), gm(m$essdraw_A) / gm(J$essdraw),
                gm(m$essdraw_D) / gm(J$essdraw)))
    cat(sprintf("      draws/sec     J %8.1f   jmjax A %.2fx   D %.2fx  <- machine\n",
                gm(J$drawsec), gm(m$drawsec_A) / gm(J$drawsec),
                gm(m$drawsec_D) / gm(J$drawsec)))
    cat(sprintf("      min ESS/sec   J %8.2f   jmjax A %.2fx   D %.2fx  = product\n",
                gm(J$minsec), gm(m$minsec_A) / gm(J$minsec),
                gm(m$minsec_D) / gm(J$minsec)))
    J$minsecR  <- J$min_ess_reg / J$sec
    J$essdrawR <- J$min_ess_reg / J$n_draws
    cat(sprintf("      -- regression parameters only (J binds on %s) --\n",
                paste(unique(J$min_who_reg), collapse = "/")))
    cat(sprintf("      min ESS/draw  J %8.4f   jmjax A %.2fx   D %.2fx   E %s\n",
                gm(J$essdrawR), gm(m$essdrawR_A) / gm(J$essdrawR),
                gm(m$essdrawR_D) / gm(J$essdrawR),
                if (nrow(E)) sprintf("%.2fx", gm(m$essdrawR_E) / gm(J$essdrawR)) else "-"))
    cat(sprintf("      min ESS/sec   J %8.2f   jmjax A %.2fx   D %.2fx   E %s\n",
                gm(J$minsecR), gm(m$minsecR_A) / gm(J$minsecR),
                gm(m$minsecR_D) / gm(J$minsecR),
                if (nrow(E)) sprintf("%.2fx", gm(m$minsecR_E) / gm(J$minsecR)) else "-"))
  }
}

cat("\n==================== ACCURACY (alpha vs truth) ====================\n")
for (kk in unique(R$k)) for (nn in unique(R$n[R$k == kk])) {
  x <- R[R$k == kk & R$n == nn & is.finite(R$alpha_hat), ]
  if (!nrow(x)) next
  cat(sprintf("  k=%d n=%4d  truth %.3f | ", kk, nn, x$alpha_true[1]))
  for (a in c("A", "D", "J")) {
    y <- x[x$arm == a, ]
    if (nrow(y)) cat(sprintf("%s %.4f  ", a, mean(y$alpha_hat)))
  }
  cat("\n")
}
cat("\n  A and D must agree: the reparameterization cannot move alpha at all.\n")
cat("\n==================== ACCURACY (beta_1 vs truth) ====================\n")
cat("  beta_1 is the binding regression parameter in most cells, so whether\n")
cat("  it is RECOVERED matters as much as how fast it mixes.\n")
for (kk in unique(R$k)) for (nn in unique(R$n[R$k == kk])) {
  x <- R[R$k == kk & R$n == nn & is.finite(R$beta1_hat), ]
  if (!nrow(x)) next
  cat(sprintf("  k=%d n=%4d  truth %.3f | ", kk, nn, x$beta1_true[1]))
  for (a in c("A", "D", "E")) {
    y <- x[x$arm == a, ]
    if (nrow(y)) cat(sprintf("%s %.4f  ", a, mean(y$beta1_hat)))
  }
  cat("\n")
}
cat("  A/D vs J is the cross-package accuracy check.\n")

cat("\n==================== HARM CHECK, k = 1 ====================\n")
x1 <- R[R$k == 1L, ]
if (nrow(x1)) {
  .arm1 <- if (any(x1$arm == "E")) "E" else "D"
  m1 <- merge(x1[x1$arm == "A",     c("n", "seed", "minsec")],
              x1[x1$arm == .arm1,   c("n", "seed", "minsec")],
              by = c("n", "seed"), suffixes = c("_A", "_D"))
  cat(sprintf("  (judging arm %s, the one that would become the default)\n", .arm1))
  cat(sprintf("  k=1 is the least-to-gain case and where a DEFAULT would do harm\n"))
  cat(sprintf("  first. min ESS/sec ratio %.2fx over %d cells, worst cell %.2fx\n",
              gm(m1$minsec_D / m1$minsec_A), nrow(m1),
              min(m1$minsec_D / m1$minsec_A)))
  cat("  A default needs this to be >= 1 everywhere, not merely on average.\n")
} else cat("  k=1 not run.\n")

cat("\nwrote ", CSV, "\n", sep = "")
sink(); close(.con)
