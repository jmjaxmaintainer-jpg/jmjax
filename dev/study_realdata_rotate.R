# ==============================================================================
# dev/study_realdata_rotate.R - real-data check of the rotation (vignette
# Section 7): default fit (arm A) vs control$rotate_absorbable +
# control$dense_mass_generator_beta (arm R), optionally JMbayes2 (arm J).
#
# DATASETS (both bundled with JMbayes2, same specifications as the earlier
# studies so numbers are comparable):
#   aids  sqrt(CD4) ~ obstime + drug + gender, random = ~ obstime | patient,
#         Surv(Time, death) ~ drug. n = 467. dev/study_realdata_aids.R's
#         design. Untouched by the diagnostics that found the degeneracy.
#   pbc2  log(serBilir) ~ year + age, random = ~ year | id,
#         Surv(years, status2) ~ drug. n = 312. dev/bench_pbc2.R's design;
#         the dataset the degeneracy was first measured on (vignette 3.5).
#
# WHAT IS RECORDED. One row per (dataset, seed, arm, parameter). A and R
# share the same lme()/coxph() pre-fit and the same sampler seed. Recorded:
#   - ESS and R-hat for every population parameter (beta, alpha, gamma,
#     sigma_e, sigma_b, rho);
#   - the posterior mean and SD, so A and R can be checked for agreement.
#     The rotation is exact, so they must agree to Monte Carlo error;
#   - NUTS sampling time, which excludes JIT compilation.
# Arm J (opt-in via REAL_ARMS) adds JMbayes2 on the same pre-fit, for the
# "rotated fit vs JMbayes2 on the regression coefficients" comparison of
# vignette Section 8.3. Its time includes JMbayes2's whole jm() call.
#
# USAGE (resumable; results go to REAL_OUTDIR, default ~/Documents/R/jmjax_results):
#   Rscript dev/study_realdata_rotate.R                      # aids + pbc2, A and R, 3 seeds
#   REAL_ARMS=A,R,J REAL_SEEDS=2 caffeinate -i Rscript dev/study_realdata_rotate.R
#   REAL_DATASETS=aids Rscript dev/study_realdata_rotate.R
# Cost: roughly 1-3 min per jmjax fit at 4 chains x (1000 + 1000); JMbayes2
# at REAL_JB_ITER = 16000 is several minutes per fit.
# ==============================================================================

suppressPackageStartupMessages({
  library(jmjax); library(nlme); library(survival)
})
if (!requireNamespace("JMbayes2", quietly = TRUE))
  stop("JMbayes2 is required (it provides the aids and pbc2 data).", call. = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a

.envl <- function(nm, d) { v <- Sys.getenv(nm, ""); if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d }
.envi <- function(nm, d) {
  v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d
}

DATASETS <- .envl("REAL_DATASETS", c("aids", "pbc2"))
ARMS     <- .envl("REAL_ARMS", c("A", "R"))
NSEED    <- .envi("REAL_SEEDS", 3L)
CHAINS   <- .envi("REAL_CHAINS", 4L)
WARMUP   <- .envi("REAL_WARMUP", 1000L)
SAMPLES  <- .envi("REAL_SAMPLES", 1000L)
JB_ITER  <- .envi("REAL_JB_ITER", 16000L)
JB_BURN  <- .envi("REAL_JB_BURN", 4000L)
OUTDIR   <- Sys.getenv("REAL_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, Sys.getenv("REAL_TAG", "study_realdata_rotate.csv"))

EXPECTED_COLS <- c("dataset", "seed", "arm", "param", "est", "sd", "ess",
                   "rhat", "sec", "n_draws", "mean_num_steps")

# ---- data --------------------------------------------------------------------
load_data <- function(ds) {
  if (ds == "aids") {
    utils::data("aids", package = "JMbayes2", envir = environment())
    utils::data("aids.id", package = "JMbayes2", envir = environment())
    dl <- aids[, c("patient", "obstime", "CD4", "drug", "gender")]
    dl$y <- sqrt(dl$CD4); names(dl)[1] <- "id"
    dsv <- aids.id[, c("patient", "Time", "death", "drug")]
    names(dsv) <- c("id", "stime", "event", "drug")
    list(dl = dl, ds = dsv, time_var = "obstime",
         lform = y ~ obstime + drug + gender, rform = ~ obstime | id)
  } else if (ds == "pbc2") {
    utils::data("pbc2", package = "JMbayes2", envir = environment())
    utils::data("pbc2.id", package = "JMbayes2", envir = environment())
    dl <- pbc2[, c("id", "year", "serBilir", "age")]
    dl$id <- as.integer(as.character(dl$id)); dl$y <- log(dl$serBilir)
    dsv <- pbc2.id[, c("id", "years", "status", "drug")]
    dsv$id <- as.integer(as.character(dsv$id))
    dsv$event <- as.integer(dsv$status != "alive")
    dsv <- data.frame(id = dsv$id, stime = dsv$years, event = dsv$event, drug = dsv$drug)
    list(dl = dl, ds = dsv, time_var = "year",
         lform = y ~ year + age, rform = ~ year | id)
  } else stop("unknown dataset ", ds, call. = FALSE)
}

prefit <- function(d) {
  lme_fit <- lme(d$lform, random = d$rform, data = d$dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- coxph(Surv(stime, event) ~ drug, data = d$ds)
  list(lme = lme_fit, cox = cox_fit)
}

jx_control <- function(arm, seed) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (arm == "R") { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  # A is the UNROTATED fit (the rotation is now the default; opt out so the
  # arm stays what it was).
  if (arm == "A") ctl$rotate_absorbable <- FALSE
  ctl
}

POP <- "^beta_|^alpha$|^gamma_|^sigma_e$|^sigma_b|^rho$"

fit_jx <- function(ds, d, pf, arm, seed) {
  f <- jm_fit_prefit(pf$lme, pf$cox, data_surv = d$ds, time_var = d$time_var,
                     method = "spline-PH-mcmc", control = jx_control(arm, seed))
  cv <- f$convergence
  if (arm == "R" && !isTRUE(cv$orthogonalize$rotation$applied))
    stop("arm R: rotation not applied - reinstall the backend (bash dev/install_jmjax.sh)",
         call. = FALSE)
  est <- unlist(f$estimates); se <- unlist(f$se)
  ess <- unlist(f$diagnostics$ess); rh <- unlist(f$diagnostics$rhat)
  nm <- names(est)[grepl(POP, names(est))]
  data.frame(dataset = ds, seed = seed, arm = arm, param = nm,
             est = as.numeric(est[nm]), sd = as.numeric(se[nm]),
             ess = as.numeric(ess[nm]), rhat = as.numeric(rh[nm]),
             sec = as.numeric(cv$sampling_time_sec %||% NA_real_),
             n_draws = CHAINS * SAMPLES,
             mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
             stringsAsFactors = FALSE)
}

fit_jb <- function(ds, d, pf, seed) {
  tm <- system.time(jb <- JMbayes2::jm(Surv_object = pf$cox, Mixed_objects = pf$lme,
                                       time_var = d$time_var, n_chains = CHAINS,
                                       n_iter = JB_ITER, n_burnin = JB_BURN,
                                       control = list(Bsplines_degree = 3,
                                                      base_hazard_segments = 6, seed = seed)))
  st <- jb$statistics
  pick <- function(block, prefix) {
    m <- st$Mean[[block]]; if (is.null(m)) return(NULL)
    s <- st$SD[[block]]; e <- st$Effective_Size[[block]]; r <- st$Rhat[[block]]
    r <- if (is.matrix(r)) r[, 1] else r
    data.frame(param = sprintf("%s%d", prefix, seq_along(m) - 1L),
               est = as.numeric(m), sd = as.numeric(s), ess = as.numeric(e),
               rhat = as.numeric(r), stringsAsFactors = FALSE)
  }
  out <- rbind(pick("betas1", "beta_"), pick("gammas", "gamma_"))
  a <- pick("alphas", "alpha"); if (!is.null(a)) { a$param <- "alpha"; out <- rbind(out, a[1, ]) }
  cbind(data.frame(dataset = ds, seed = seed, arm = "J", stringsAsFactors = FALSE),
        out, sec = as.numeric(tm[["elapsed"]]), n_draws = CHAINS * (JB_ITER - JB_BURN),
        mean_num_steps = NA_real_)
}

# ---- run (resumable) ----------------------------------------------------------
done <- if (file.exists(CSV)) utils::read.csv(CSV, stringsAsFactors = FALSE) else NULL
if (!is.null(done) && !identical(names(done), EXPECTED_COLS))
  stop(CSV, " has a different column layout; set REAL_TAG to a new file", call. = FALSE)
have <- function(ds, s, a) !is.null(done) && any(done$dataset == ds & done$seed == s & done$arm == a)

for (ds in DATASETS) {
  d <- load_data(ds); pf <- prefit(d)
  cat(sprintf("\n== %s: n = %d subjects, %d longitudinal obs, %.0f%% events\n", ds,
              length(unique(d$dl$id)), nrow(d$dl), 100 * mean(d$ds$event)))
  # Throwaway fit so the first timed fit does not carry JAX compilation.
  invisible(suppressWarnings(try(jm_fit_prefit(pf$lme, pf$cox, data_surv = d$ds,
    time_var = d$time_var, method = "spline-PH-mcmc",
    control = utils::modifyList(jx_control("R", 1L), list(num_warmup = 5L, num_samples = 5L))),
    silent = TRUE)))
  for (s in seq_len(NSEED)) for (a in ARMS) {
    if (have(ds, s, a)) next
    t0 <- Sys.time()
    rows <- tryCatch(if (a == "J") fit_jb(ds, d, pf, s) else fit_jx(ds, d, pf, a, s),
                     error = function(e) {
                       if (grepl("reinstall", conditionMessage(e))) stop(e)
                       cat("   ", ds, a, "seed", s, "FAILED:", conditionMessage(e), "\n"); NULL })
    if (is.null(rows)) next
    utils::write.table(rows[, EXPECTED_COLS], CSV, sep = ",", row.names = FALSE,
                       col.names = !file.exists(CSV), append = file.exists(CSV))
    b <- rows[grepl("^beta_", rows$param), ]
    cat(sprintf("   %s seed %d %s: %6.1fs sampling | min beta ESS %7.1f (%s) | alpha ESS %7.1f | max R-hat %.3f\n",
                ds, s, a, rows$sec[1], min(b$ess), b$param[which.min(b$ess)],
                rows$ess[rows$param == "alpha"][1] %||% NA, max(rows$rhat, na.rm = TRUE)))
  }
}

# ---- summary ------------------------------------------------------------------
R <- utils::read.csv(CSV, stringsAsFactors = FALSE)
R <- R[R$dataset %in% DATASETS & R$arm %in% ARMS, ]
R$ess_per_sec <- R$ess / R$sec
gm <- function(x) { x <- x[is.finite(x) & x > 0]; if (length(x)) exp(mean(log(x))) else NA_real_ }
for (ds in DATASETS) {
  X <- R[R$dataset == ds, ]; if (!nrow(X)) next
  cat(sprintf("\n==================== %s ====================\n", ds))
  cat(sprintf("%-10s %11s %11s %9s %9s %11s %9s\n", "param", "A ESS/s", "R ESS/s", "R/A",
              "z(A,R)", "J ESS/s", "R/J"))
  for (p in unique(X$param[X$arm == "A"])) {
    g <- function(a) X[X$arm == a & X$param == p, ]
    A <- g("A"); Rr <- g("R"); J <- g("J")
    pr <- merge(A, Rr, by = "seed")
    ratio <- if (nrow(pr)) gm(pr$ess_per_sec.y / pr$ess_per_sec.x) else NA
    # agreement: mean difference over seeds in units of its Monte Carlo SE
    z <- if (nrow(pr)) mean(pr$est.y - pr$est.x) /
           sqrt(mean(pr$sd.x^2 / pr$ess.x + pr$sd.y^2 / pr$ess.y) / nrow(pr)) else NA
    cat(sprintf("%-10s %11.1f %11.1f %8.2fx %9.2f %11s %9s\n", p,
                mean(A$ess_per_sec), mean(Rr$ess_per_sec %||% NA), ratio, z,
                if (nrow(J)) sprintf("%.1f", mean(J$ess_per_sec)) else "-",
                if (nrow(J) && nrow(Rr)) sprintf("%.2fx", mean(Rr$ess_per_sec) / mean(J$ess_per_sec)) else "-"))
  }
  cat(sprintf("sampling sec: A %.1f | R %.1f%s   max R-hat: A %.3f | R %.3f\n",
              mean(X$sec[X$arm == "A"]), mean(X$sec[X$arm == "R"]),
              if (any(X$arm == "J")) sprintf(" | J %.1f (whole jm() call)", mean(X$sec[X$arm == "J"])) else "",
              max(X$rhat[X$arm == "A"], na.rm = TRUE), max(X$rhat[X$arm == "R"], na.rm = TRUE)))
}
cat("\nz(A,R): mean difference of posterior means over seeds, in Monte Carlo SEs. The\n",
    "rotation is exact, so |z| should look like draws from N(0,1).\n", sep = "")
