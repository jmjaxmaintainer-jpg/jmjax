# ==============================================================================
# PBC2 real data: jmjax vs JMbayes2, with covariates.
#
# THREE DESIGN DECISIONS, all of which changed the numbers.
#
# 1. A SHARED PRE-FIT, TIMED SEPARATELY.
#    Earlier versions timed system.time(jm_fit(...)) against
#    system.time({lme(); coxph(); jm()}). Both included a pre-fit, so it was
#    not unfair exactly - but it compared jmjax's INTERNAL lme against
#    JMbayes2's external one, and charged each package for its own.
#
#    This version fits lme() and coxph() ONCE, times that once, and hands
#    the same two objects to both packages: JMbayes2 takes them directly,
#    and jmjax takes them through jm_fit_prefit(), which exists for exactly
#    this. Both then start from an identical point, and the sampler
#    comparison is sampler against sampler.
#
#    It also makes jmjax's warm start nearly free. Measured on the previous
#    design, the warm start cost 3.1s of a 14.97s fit - about 2.1s of that
#    was jmjax re-fitting an lme that had already been fitted for JMbayes2.
#
#    Both figures are reported: SAMPLER time (the like-for-like comparison)
#    and TOTAL including the shared pre-fit (what a user waits).
#
#    One asymmetry this introduces, stated rather than hidden:
#    jm_fit_prefit() reads coef(coxph) to centre the gamma prior, which
#    jm_fit() does not. JMbayes2 also uses its coxph, so this brings the
#    two closer together rather than further apart - but it is a real
#    difference from the plain jm_fit() path a formula-interface user takes.
#
# 2. SAMPLE-SIZE SCALING BY DUPLICATION.
#    BENCH_PBC2_DUP=k replicates every subject k times with fresh ids, the
#    standard trick for scaling a fixed real dataset. The data-generating
#    structure, covariate distributions and event rate are held exactly
#    fixed while n scales - which is the point, since a synthetic generator
#    at larger n changes more than the sample size.
#
#    Read the ESTIMATES from a duplicated run with care: duplicating data
#    is not the same as collecting more of it. Posterior SDs shrink like
#    1/sqrt(k) around the same point estimates, because every subject's
#    likelihood contribution is counted k times. It is a timing and mixing
#    experiment, not an inference experiment.
#
# 3. LIKE-FOR-LIKE min ESS.
#    alpha ESS flatters whichever sampler mixes well on that one parameter.
#    min ESS across population parameters answers "how long until the whole
#    fit is usable". Both are reported; the second is the one to quote.
#    JMbayes2 reports slots for model features this fit does not use
#    (frailty on PBC2), with ESS exactly 0 - those are dropped by VALUE.
#
#   Rscript dev/bench_pbc2.R
#   BENCH_PBC2_DUP=2 Rscript dev/bench_pbc2.R     # n = 624
#   BENCH_SKIP_WISHART=1 Rscript dev/bench_pbc2.R
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 is not installed - it supplies both the comparison AND the ",
       "pbc2 data.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
JX_WARMUP  <- .envi("BENCH_JX_WARMUP", 1000L)
JX_SAMPLES <- .envi("BENCH_JX_SAMPLES", 1000L)
JX_CHAINS  <- .envi("BENCH_JX_CHAINS", 4L)
JB_ITER    <- .envi("BENCH_JB_ITER", 8000L)
JB_BURNIN  <- .envi("BENCH_JB_BURNIN", 2000L)
JB_CHAINS  <- .envi("BENCH_JB_CHAINS", 4L)
DUP        <- .envi("BENCH_PBC2_DUP", 1L)
SKIP_WISH  <- identical(Sys.getenv("BENCH_SKIP_WISHART", "0"), "1")

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
PREC <- tryCatch(jmjax:::.backend_precision(), error = function(e) NA_character_)
OUT  <- file.path(OUTDIR, sprintf("bench_pbc2_dup%d_%s", DUP,
                                  format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)

# ---- data prep: verbatim, including the /12 rescaling -----------------------
data("pbc2", package = "JMbayes2")
data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id))
pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2$status2 <- as.integer(pbc2$status != "alive")
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12
pbc2.id$years2 <- pbc2.id$years / 12

data_long <- pbc2[, c("id", "year2", "serBilir", "age")]
data_long$log_serBilir <- log(data_long$serBilir)
data_surv <- pbc2.id[, c("id", "years2", "status2", "drug")]

if (DUP > 1L) {
  # Fresh ids per copy, so the two halves are distinct subjects rather than
  # repeated measurements on the same one - which would be a different model.
  base_max <- max(data_surv$id)
  dl <- vector("list", DUP); ds <- vector("list", DUP)
  for (k in seq_len(DUP)) {
    off <- (k - 1L) * base_max
    a <- data_long; a$id <- a$id + off
    b <- data_surv; b$id <- b$id + off
    dl[[k]] <- a; ds[[k]] <- b
  }
  data_long <- do.call(rbind, dl)
  data_surv <- do.call(rbind, ds)
  stopifnot(!anyDuplicated(data_surv$id))
}

cat("=============================================================\n")
cat("  PBC2, covariates on both sides", if (DUP > 1L)
      sprintf(" | DUPLICATED x%d", DUP) else "", "\n", sep = "")
cat("  precision in force : ", PREC, "\n", sep = "")
cat(sprintf("  subjects %d | long obs %d | events %d (%.1f%%)\n",
            length(unique(data_long$id)), nrow(data_long),
            sum(data_surv$status2), 100 * mean(data_surv$status2)))
cat("  jmjax    : warmup ", JX_WARMUP, " / samples ", JX_SAMPLES,
    " / chains ", JX_CHAINS, "\n", sep = "")
cat("  JMbayes2 : iter ", JB_ITER, " / burnin ", JB_BURNIN,
    " / chains ", JB_CHAINS, "\n", sep = "")
if (DUP > 1L) {
  cat("\n  NOTE: duplicated data. Posterior SDs shrink like 1/sqrt(k) because\n")
  cat("  each subject's likelihood is counted k times. Read the TIMING and\n")
  cat("  MIXING, not the inference.\n")
}
cat("=============================================================\n")

.num <- function(x) { v <- suppressWarnings(as.numeric(x)); if (length(v)) v[1] else NA_real_ }
.at  <- function(l, nm) if (!is.null(l) && nm %in% names(l)) .num(l[[nm]]) else NA_real_

# ---- the shared pre-fit, timed once -----------------------------------------
cat("\n=== shared pre-fit (lme + coxph), handed to BOTH packages ===\n")
t_pre <- system.time({
  lme_fit <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id,
                 data = data_long,
                 control = lmeControl(opt = "optim", msMaxIter = 200,
                                      niterEM = 100))
  cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)
})
PRE <- as.numeric(t_pre[["elapsed"]])
cat(sprintf("  %.2fs  (charged to neither package's sampler time)\n", PRE))

# ---- jmjax arms, from the SAME pre-fitted objects ----------------------------
run_jmjax <- function(label, extra = list()) {
  cat("\n=== ", label, " ===\n", sep = "")
  ctl <- c(list(n_interior_knots = 5, spline_prior = "penalized",
                rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                num_warmup = JX_WARMUP, num_samples = JX_SAMPLES,
                num_chains = JX_CHAINS, progress_bar = FALSE), extra)
  tt <- system.time(f <- tryCatch(
    jm_fit_prefit(lme_fit, cox_fit, data_surv = data_surv,
                  time_var = "year2", method = "spline-PH-mcmc",
                  control = ctl),
    error = function(e) { cat("  FAILED: ", conditionMessage(e), "\n", sep = ""); NULL }))
  if (is.null(f)) return(NULL)
  rh <- unlist(f$diagnostics$rhat); es <- unlist(f$diagnostics$ess)
  ws <- f$convergence$warm_start
  out <- list(label = label, sampler = as.numeric(tt[["elapsed"]]),
              backend = .num(f$convergence$sampling_time_sec),
              alpha = .at(f$estimates, "alpha"),
              alpha_ess = .at(as.list(es), "alpha"),
              min_ess = suppressWarnings(min(es, na.rm = TRUE)),
              min_ess_param = if (length(es)) names(es)[which.min(es)] else NA_character_,
              max_rhat = suppressWarnings(max(rh, na.rm = TRUE)),
              warm_used = isTRUE(as.logical(ws$used)),
              warm_why = if (is.null(ws)) "not attempted"
                         else if (isTRUE(as.logical(ws$used))) "accepted"
                         else if (!is.null(ws$error)) paste("error:", ws$error)
                         else sprintf("rejected (%.1f vs %.1f)",
                                      .num(ws$potential_warm), .num(ws$potential_uniform)))
  out$total <- out$sampler + PRE
  cat(sprintf("  sampler %7.2fs  (+ pre-fit = %7.2fs total)  max R-hat %.4f\n",
              out$sampler, out$total, out$max_rhat))
  cat(sprintf("  alpha %7.4f  alpha ESS %8.1f  min ESS %8.1f (%s)\n",
              out$alpha, out$alpha_ess, out$min_ess, out$min_ess_param))
  cat("  warm start: ", out$warm_why, "\n", sep = "")
  out
}

arms <- list()
arms$A <- run_jmjax("A. jmjax, DEFAULTS")
if (!SKIP_WISH) {
  arms$B <- run_jmjax("B. jmjax, wishart_gibbs",
                      list(random_effects_method = "wishart_gibbs"))
}

# ---- JMbayes2, from the same objects ----------------------------------------
cat("\n=== C. JMbayes2 ===\n")
tb <- system.time(fit_jb <- tryCatch(
  jm(Surv_object = cox_fit, Mixed_objects = lme_fit, time_var = "year2",
     n_chains = JB_CHAINS, n_iter = JB_ITER, n_burnin = JB_BURNIN,
     control = list(Bsplines_degree = 3, base_hazard_segments = 6)),
  error = function(e) { cat("  FAILED: ", conditionMessage(e), "\n", sep = ""); NULL }))

jb <- NULL
if (!is.null(fit_jb)) {
  # Drop never-sampled slots by VALUE. On PBC2 there is no frailty term, so
  # alphaF / frailty* / sigmaF come back with ESS exactly 0 and a
  # meaningless R-hat; an earlier version reported min ESS = 0, a min-ESS
  # ratio of "Inf", and a max R-hat of 1.0681 belonging to a parameter that
  # does not exist in the model.
  .drop_absent <- function(l) {
    if (is.null(l) || !length(l)) return(numeric(0))
    v <- unlist(l)
    v[is.finite(v)]
  }
  es_all <- .drop_absent(fit_jb$statistics$Effective_Size)
  es_all <- es_all[es_all > 0]
  es_all <- es_all[!grepl("^b\\.|frailty|alphaF|sigmaF", names(es_all))]
  # R-hat is filtered by the SAME surviving names where they match, and
  # otherwise by the same pattern - the two lists are not guaranteed to use
  # identical naming, and an earlier intersect() came back empty, silently
  # reporting "n/a" instead of a convergence check.
  rh_all <- .drop_absent(fit_jb$statistics$Rhat)
  rh_all <- rh_all[!grepl("^b\\.|frailty|alphaF|sigmaF", names(rh_all))]
  jb <- list(sampler = as.numeric(tb[["elapsed"]]),
             backend = .num(unname(fit_jb$running_time["elapsed"])),
             alpha = .num(fit_jb$statistics$Mean$alphas),
             alpha_ess = .num(fit_jb$statistics$Effective_Size$alphas),
             min_ess = if (length(es_all)) min(es_all) else NA_real_,
             min_ess_param = if (length(es_all)) names(es_all)[which.min(es_all)] else NA_character_,
             max_rhat = if (length(rh_all)) max(rh_all) else NA_real_,
             max_rhat_param = if (length(rh_all)) names(rh_all)[which.max(rh_all)] else NA_character_,
             n_params = length(es_all))
  jb$total <- jb$sampler + PRE
  cat(sprintf("  sampler %7.2fs  (+ pre-fit = %7.2fs total)\n", jb$sampler, jb$total))
  cat(sprintf("  alpha %7.4f  alpha ESS %8.1f  min ESS %8.1f (%s)\n",
              jb$alpha, jb$alpha_ess, jb$min_ess, jb$min_ess_param))
  cat(sprintf("  max R-hat %s (%s) over %d population params\n",
              if (is.finite(jb$max_rhat)) sprintf("%.4f", jb$max_rhat) else "n/a",
              jb$max_rhat_param, jb$n_params))
}

# ---- comparison --------------------------------------------------------------
cat("\n==================== COMPARISON ====================\n")
if (is.null(jb)) {
  cat("  JMbayes2 did not fit - nothing to compare against.\n")
} else {
  hdr <- function(w) cat(sprintf("  %-26s %9s %9s %10s %9s %10s\n",
                                 "arm", w, "aESS/s", "ratio(a)", "minESS/s", "ratio(min)"))
  body <- function(w_of) {
    for (a in c(arms, list(C = jb))) {
      if (is.null(a)) next
      lab <- if (is.null(a$label)) "C. JMbayes2" else substr(a$label, 1, 26)
      w <- w_of(a)
      cat(sprintf("  %-26s %9.2f %9.2f %10s %9.2f %10s\n", lab, w,
                  a$alpha_ess / w,
                  sprintf("%.2fx", (a$alpha_ess / w) / (jb$alpha_ess / w_of(jb))),
                  a$min_ess / w,
                  sprintf("%.2fx", (a$min_ess / w) / (jb$min_ess / w_of(jb)))))
    }
  }
  cat("\n  -- SAMPLER TIME (like-for-like: both start from the same lme/coxph)\n")
  hdr("sampler(s)"); body(function(a) a$sampler)
  cat("\n  -- TOTAL TIME (sampler + the ", sprintf("%.2f", PRE),
      "s shared pre-fit, what a user waits)\n", sep = "")
  hdr("total(s)"); body(function(a) a$total)

  if (!is.null(arms$A) && !is.null(arms$B)) {
    rA <- (arms$A$min_ess / arms$A$sampler) / (jb$min_ess / jb$sampler)
    rB <- (arms$B$min_ess / arms$B$sampler) / (jb$min_ess / jb$sampler)
    cat(sprintf("\n  wishart_gibbs on the min-ESS metric: %.2fx -> %.2fx (%s)\n",
                rA, rB, if (rB > rA) "helps" else "HURTS"))
  }

  bad <- c(vapply(arms, function(a) if (is.null(a)) FALSE else a$max_rhat >= 1.05,
                  logical(1)),
           isTRUE(is.finite(jb$max_rhat) && jb$max_rhat >= 1.05))
  if (any(bad)) {
    cat("\n  !! R-hat >= 1.05 somewhere. ESS/second is not a fair comparison\n")
    cat("  !! between a converged fit and an unconverged one. Raise the\n")
    cat("  !! budget on the offending side and re-run before quoting this.\n")
  }
}

saveRDS(list(arms = arms, jmbayes2 = jb, prefit_sec = PRE, dup = DUP,
             precision = PREC, session = utils::sessionInfo()),
        paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,rds}\n", sep = "")
sink(); close(.con)
