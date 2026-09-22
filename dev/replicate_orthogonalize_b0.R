# ==============================================================================
# Replication study for control$orthogonalize_b0.
#
# WHY THIS EXISTS. A single PBC2 run showed beta_0 ESS 519 -> 8219 (15.8x)
# and beta_2 527 -> 7853 (14.9x), with correctness established three ways
# (identified combinations agree; fitted values perturbed 0.77x as much as
# re-seeding the identical model perturbs them; lme's own invariant lines up
# once computed correctly). That is not enough to claim anything. The
# package's own record is the reason:
#
#   dense_mass_beta  targeted a correlation confirmed to -0.9764 against a
#                    first-principles prediction of -0.9802, and DIED at
#                    replication - 2 of 4 seeds worse in each condition.
#   seed_mass_matrix looked sound and failed catastrophically on 1 of 4.
#
# So the bar is the one dense_mass_alpha had to clear: multiple seeds, a
# PAIRED comparison, a t-test on the log ratio, and the count of seeds
# improved reported alongside the mean.
#
# ONE NUMBER FROM THE PILOT ALREADY NEEDS THIS. min ESS is the conservative
# metric and it moved 1.38x, 2.64x and 2.44x across three runs that differed
# only by seed - because sigma_b0 is the new binding parameter and its own
# ESS ranged 682-1092. beta_0's 15-16x was stable; the headline was not.
# Quoting either from one run would repeat the 13.5x mistake exactly.
#
# THREE CONDITIONS, chosen so the MECHANISM is tested and not just the
# effect. The degeneracy is k-dimensional in the number of subject-constant
# design columns, so the benefit should ORDER with k:
#
#   k2   pbc2, log_serBilir ~ year2 + age                  (intercept, age)
#   k2n  the same, subjects duplicated to n = 624          (does it hold in n)
#   k4   pbc2, + sex + drug                                (intercept + 3)
#
# If k4 > k2 the mechanism is confirmed, not merely reproduced. If the gain
# is flat in k, something else is doing the work and the explanation in
# mcmc_model.py is wrong even if the option helps.
#
# Correctness is re-checked on EVERY replicate, not once: the identified
# combinations I1 (level) and I2 (first covariate effect) must agree between
# arms within Monte Carlo error on all of them.
#
#   caffeinate -i Rscript dev/replicate_orthogonalize_b0.R
#   BENCH_SEEDS=2 Rscript dev/replicate_orthogonalize_b0.R    # quick pass
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) {
  stop("JMbayes2 supplies the pbc2 data.", call. = FALSE)
}
suppressPackageStartupMessages(library(JMbayes2))

`%||%` <- function(a, b) if (is.null(a)) b else a
.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) && is.finite(v) && v > 0) v else d
}
NSEED   <- .envi("BENCH_SEEDS", 4L)
CHAINS  <- .envi("BENCH_JX_CHAINS", 4L)
WARMUP  <- .envi("BENCH_JX_WARMUP", 1000L)
SAMPLES <- .envi("BENCH_JX_SAMPLES", 1000L)

OUTDIR <- Sys.getenv("BENCH_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(OUTDIR, sprintf("replicate_orth_b0_%s",
                                 format(Sys.time(), "%Y%m%d_%H%M")))
.con <- file(paste0(OUT, ".log"), open = "wt"); sink(.con, split = TRUE)
options(error = function() {
  try(while (sink.number() > 0) sink(), silent = TRUE)
})

# ---- data -----------------------------------------------------------------
data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id))
pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2$status2 <- as.integer(pbc2$status != "alive")
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12; pbc2.id$years2 <- pbc2.id$years / 12

make_data <- function(dup = 1L, extra = character(0)) {
  dl <- pbc2[, c("id", "year2", "serBilir", "age", extra)]
  dl$log_serBilir <- log(dl$serBilir)
  ds <- pbc2.id[, c("id", "years2", "status2", "drug")]
  dl <- dl[stats::complete.cases(dl), ]
  ds <- ds[ds$id %in% unique(dl$id), ]
  dl <- dl[dl$id %in% ds$id, ]
  if (dup > 1L) {
    bmax <- max(ds$id); a <- list(); b <- list()
    for (k in seq_len(dup)) {
      x <- dl; x$id <- x$id + (k - 1L) * bmax
      y <- ds; y$id <- y$id + (k - 1L) * bmax
      a[[k]] <- x; b[[k]] <- y
    }
    dl <- do.call(rbind, a); ds <- do.call(rbind, b)
  }
  list(long = dl, surv = ds)
}

CONDITIONS <- list(
  k2  = list(label = "k=2  pbc2 n=312          (intercept, age)",
             dup = 1L, extra = character(0),
             form = log_serBilir ~ year2 + age),
  k2n = list(label = "k=2  pbc2 n=624 (dup x2) (intercept, age)",
             dup = 2L, extra = character(0),
             form = log_serBilir ~ year2 + age),
  k4  = list(label = "k=4  pbc2 n=312          (intercept, age, sex, drug)",
             dup = 1L, extra = c("sex", "drug"),
             form = log_serBilir ~ year2 + age + sex + drug))

ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 10 || !is.finite(var(x)) || var(x) <= 0) return(NA_real_)
  ac <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE,
                   demean = TRUE)$acf[, 1, 1]
  s <- 0; k <- 1L
  while (k + 1L <= length(ac)) {
    pair <- ac[k] + ac[k + 1L]; if (pair <= 0) break
    s <- s + pair; k <- k + 2L
  }
  tau <- -1 + 2 * s; if (!is.finite(tau) || tau < 1) tau <- 1
  n / tau
}
as_draw_matrix <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) return(t(vapply(v, function(z) as.numeric(unlist(z)),
                                  numeric(length(unlist(v[[1]]))))))
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
bcol1 <- function(ps, nsub) {
  b <- ps[["b"]]
  if (is.null(b)) return(NULL)
  if (is.list(b)) {
    return(t(vapply(b, function(d) {
      if (is.list(d)) vapply(d, function(r) as.numeric(r)[1], numeric(1))
      else { a <- as.matrix(d); as.numeric(a[, 1]) }
    }, numeric(nsub))))
  }
  a <- as.array(b); if (length(dim(a)) == 3L) a[, , 1] else as.matrix(a)
}
min_ess <- function(e) { e <- e[is.finite(e) & e > 0]; if (!length(e)) NA_real_ else min(e) }
which_min_ess <- function(e) {
  e <- e[is.finite(e) & e > 0]; if (!length(e)) NA_character_ else names(e)[which.min(e)]
}

# I1 / I2 for one fit, using whichever subject-constant covariate comes
# first in the design (age in every condition here).
invariants <- function(fit, age_i, abar) {
  ps <- fit$posterior_samples
  B  <- as_draw_matrix(ps[["beta"]])
  B0 <- bcol1(ps, length(age_i))
  if (is.null(B) || is.null(B0)) return(list(I1 = NA_real_, I2 = NA_real_))
  ac <- age_i - abar; S <- cbind(1, ac)
  CO <- B0 %*% t(solve(crossprod(S), t(S)))
  list(I1 = B[, 1] + CO[, 1] - CO[, 2] * abar,
       I2 = B[, 3] + CO[, 2])   # column 3 is age in all three formulas
}
zdiff <- function(xa, xd) {
  if (!is.numeric(xa) || !is.numeric(xd) || length(xa) < 10) return(NA_real_)
  mc <- sqrt(stats::var(xa) / ess_geyer(xa) + stats::var(xd) / ess_geyer(xd))
  if (!is.finite(mc) || mc <= 0) return(NA_real_)
  (mean(xd) - mean(xa)) / mc
}

rows <- list()
for (cn in names(CONDITIONS)) {
  cfg <- CONDITIONS[[cn]]
  dd  <- make_data(cfg$dup, cfg$extra)
  dl  <- dd$long; ds <- dd$surv
  sub <- dl[!duplicated(dl$id), c("id", "age")]; sub <- sub[order(sub$id), ]
  age_i <- as.numeric(sub$age); abar <- mean(age_i)

  cat("\n=============================================================\n")
  cat(sprintf("  %s\n", cfg$label))
  cat(sprintf("  subjects %d | long obs %d | events %d\n",
              nrow(sub), nrow(dl), sum(ds$status2)))
  cat("=============================================================\n")

  lme_fit <- lme(cfg$form, random = ~ year2 | id, data = dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200,
                                      niterEM = 100))
  cox_fit <- coxph(Surv(years2, status2) ~ drug, data = ds)

  run <- function(seed, orth) {
    ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
                rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                num_warmup = WARMUP, num_samples = SAMPLES,
                num_chains = CHAINS, progress_bar = FALSE, seed = seed)
    if (orth) ctl$orthogonalize_b0 <- TRUE
    t <- system.time(f <- jm_fit_prefit(
      lme_fit, cox_fit, data_surv = ds, time_var = "year2",
      method = "spline-PH-mcmc", control = ctl))
    list(fit = f, elapsed = as.numeric(t[["elapsed"]]))
  }

  for (s in seq_len(NSEED)) {
    a <- run(s, FALSE); d <- run(s, TRUE)
    ea <- unlist(a$fit$diagnostics$ess); ed <- unlist(d$fit$diagnostics$ess)
    ia <- invariants(a$fit, age_i, abar); id <- invariants(d$fit, age_i, abar)
    rows[[length(rows) + 1L]] <- data.frame(
      cond = cn, seed = s,
      b0_A = ea[["beta_0"]], b0_D = ed[["beta_0"]],
      b2_A = ea[["beta_2"]], b2_D = ed[["beta_2"]],
      min_A = min_ess(ea),   min_D = min_ess(ed),
      minwho_A = which_min_ess(ea), minwho_D = which_min_ess(ed),
      sec_A = a$elapsed, sec_D = d$elapsed,
      rhat_A = max(unlist(a$fit$diagnostics$rhat), na.rm = TRUE),
      rhat_D = max(unlist(d$fit$diagnostics$rhat), na.rm = TRUE),
      z_I1 = zdiff(ia$I1, id$I1), z_I2 = zdiff(ia$I2, id$I2),
      stringsAsFactors = FALSE)
    r <- rows[[length(rows)]]
    cat(sprintf(paste0("  seed %d | beta_0 %7.1f -> %7.1f (%5.2fx) | ",
                       "min %6.1f -> %6.1f (%5.2fx, %s -> %s) | ",
                       "%.1fs -> %.1fs | z(I1,I2) %5.2f %5.2f\n"),
                s, r$b0_A, r$b0_D, r$b0_D / r$b0_A,
                r$min_A, r$min_D, r$min_D / r$min_A, r$minwho_A, r$minwho_D,
                r$sec_A, r$sec_D, r$z_I1, r$z_I2))
  }
}

R <- do.call(rbind, rows)
R$ratio_b0    <- R$b0_D / R$b0_A
R$ratio_min   <- R$min_D / R$min_A
R$minsec_A    <- R$min_A / R$sec_A
R$minsec_D    <- R$min_D / R$sec_D
R$ratio_minsec <- R$minsec_D / R$minsec_A

# ---- correctness across every replicate -----------------------------------
cat("\n==================== CORRECTNESS ====================\n")
bad <- R[is.finite(R$z_I1) & is.finite(R$z_I2) &
         (abs(R$z_I1) > 3 | abs(R$z_I2) > 3), ]
cat(sprintf("  replicates %d | max |z(I1)| %.2f | max |z(I2)| %.2f\n",
            nrow(R), max(abs(R$z_I1), na.rm = TRUE),
            max(abs(R$z_I2), na.rm = TRUE)))
if (nrow(bad)) {
  cat("  FAILED on:\n"); print(bad[, c("cond", "seed", "z_I1", "z_I2")])
  cat("  The identified quantities do not match on every replicate, so the\n")
  cat("  reparameterization is NOT inference-preserving in general. Stop.\n")
} else {
  cat("  Every replicate: the identified level and age effect agree between\n")
  cat("  arms within Monte Carlo error. Inference is preserved throughout.\n")
}

# ---- efficiency, per condition, paired ------------------------------------
tt <- function(x) {
  x <- x[is.finite(x) & x > 0]
  if (length(x) < 2) return(list(p = NA_real_, lo = NA_real_, hi = NA_real_))
  h <- stats::t.test(log(x))
  list(p = h$p.value, lo = exp(h$conf.int[1]), hi = exp(h$conf.int[2]))
}
cat("\n==================== EFFICIENCY ====================\n")
cat("  geometric means, paired within seed; t-test on the log ratio\n\n")
for (cn in names(CONDITIONS)) {
  x <- R[R$cond == cn, ]
  if (!nrow(x)) next
  gm <- function(v) exp(mean(log(v[is.finite(v) & v > 0])))
  h  <- tt(x$ratio_minsec)
  cat(sprintf("  %s\n", CONDITIONS[[cn]]$label))
  cat(sprintf("    beta_0 ESS      %6.2fx   (%d/%d seeds improved)\n",
              gm(x$ratio_b0), sum(x$ratio_b0 > 1), nrow(x)))
  cat(sprintf("    beta_2 ESS      %6.2fx\n", gm(x$b2_D / x$b2_A)))
  cat(sprintf("    min ESS         %6.2fx   (%d/%d)\n",
              gm(x$ratio_min), sum(x$ratio_min > 1), nrow(x)))
  cat(sprintf("    min ESS/sec     %6.2fx   (%d/%d)  p = %s  95%% CI [%.2fx, %.2fx]\n",
              gm(x$ratio_minsec), sum(x$ratio_minsec > 1), nrow(x),
              if (is.finite(h$p)) sprintf("%.4f", h$p) else "-",
              h$lo, h$hi))
  cat(sprintf("    wall            %6.2fx   (%.1fs -> %.1fs)\n",
              gm(x$sec_D / x$sec_A), mean(x$sec_A), mean(x$sec_D)))
  cat(sprintf("    binding param   %s  ->  %s\n",
              paste(unique(x$minwho_A), collapse = "/"),
              paste(unique(x$minwho_D), collapse = "/")))
  cat(sprintf("    max R-hat       %.4f -> %.4f\n",
              max(x$rhat_A), max(x$rhat_D)))
}

# ---- does the gain order with k? -----------------------------------------
cat("\n==================== DOES IT ORDER WITH k? ====================\n")
gm <- function(v) exp(mean(log(v[is.finite(v) & v > 0])))
g2  <- gm(R$ratio_b0[R$cond == "k2"])
g2n <- gm(R$ratio_b0[R$cond == "k2n"])
g4  <- gm(R$ratio_b0[R$cond == "k4"])
cat(sprintf("  beta_0 gain:  k=2 %.2fx   k=2 (n=624) %.2fx   k=4 %.2fx\n",
            g2, g2n, g4))
if (is.finite(g4) && is.finite(g2) && g4 > g2) {
  cat("  The gain GROWS with the number of subject-constant columns, which is\n")
  cat("  what a k-dimensional location degeneracy predicts. The mechanism in\n")
  cat("  mcmc_model.py is supported, not just the effect.\n")
} else {
  cat("  The gain does NOT grow with k. The option may still help, but the\n")
  cat("  k-dimensional explanation in mcmc_model.py is then not what is\n")
  cat("  doing the work and that comment needs rewriting before release.\n")
}

utils::write.csv(R, paste0(OUT, ".csv"), row.names = FALSE)
saveRDS(R, paste0(OUT, ".rds"))
cat("\nwrote ", OUT, ".{log,csv,rds}\n", sep = "")
sink(); close(.con)
