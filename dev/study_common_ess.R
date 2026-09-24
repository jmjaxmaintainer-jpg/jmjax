# ==============================================================================
# dev/study_common_ess.R - the jmjax vs JMbayes2 comparison with ONE ESS
# estimator for both packages (methods paper, review experiment 3; prepared
# 24 Sep 2026, not yet run).
#
# WHY. dev/study_realdata_rotate.R (paper Tables 5-6) took each package's
# ESS from the package itself: NumPyro's estimator for jmjax, JMbayes2's own
# for JMbayes2. The ratios therefore mix two estimators. This script refits
# the same models, SAVES THE RAW DRAWS of every reported population
# parameter (beta, alpha, gamma) for all three arms, and computes ESS for
# all of them with the same estimators:
#   ess_numpyro  NumPyro's multi-chain estimator, through the package's own
#                recompute_site_diagnostics() (what the paper reports for jmjax)
#   ess_basic, ess_bulk, ess_tail  from the 'posterior' package, if installed
#                (basic = non-rank-normalized; bulk/tail = Vehtari et al. 2021)
#   ess_native   what each package itself reports (for reference)
# ESS per second uses jmjax's MCMC time (warm-up included) and JMbayes2's
# whole jm() call, as in the paper.
#
# ARMS (same pre-fit and settings as dev/study_realdata_rotate.R):
#   A  jmjax unrotated      R  jmjax rotated + dense block      J  JMbayes2
#
# OUTPUT (in CESS_OUTDIR, default ~/Documents/R/jmjax_results):
#   common_ess.csv                      one row per dataset x seed x arm x param
#   common_ess_draws/<ds>_<arm>_seed<s>.rds
#        list(draws = [n_draws x params] matrix with chains CONCATENATED,
#             chains, n_per_chain, params, sec, mean_num_steps, arm, dataset, seed)
#   so any other estimator can be applied later without refitting.
#
# USAGE (resumable):
#   caffeinate -i Rscript dev/study_common_ess.R                 # aids + pbc2, 3 seeds
#   CESS_DATASETS=aids CESS_SEEDS=1 Rscript dev/study_common_ess.R   # quick look
#   CESS_ARMS=J Rscript dev/study_common_ess.R                   # JMbayes2 only
# Cost: about the same as study_realdata_rotate.R with REAL_ARMS=A,R,J
# (JMbayes2 at 16,000 iterations dominates: several minutes per fit).
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
if (!requireNamespace("JMbayes2", quietly = TRUE)) stop("JMbayes2 is required", call. = FALSE)
HAVE_POSTERIOR <- requireNamespace("posterior", quietly = TRUE)
if (!HAVE_POSTERIOR) message("package 'posterior' not installed: ess_basic/bulk/tail will be NA ",
                             "(install.packages('posterior') to add them)")
`%||%` <- function(a, b) if (is.null(a)) b else a
.envl <- function(nm, d) { v <- Sys.getenv(nm, ""); if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d }
.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }

DATASETS <- .envl("CESS_DATASETS", c("aids", "pbc2"))
ARMS     <- .envl("CESS_ARMS", c("A", "R", "J"))
NSEED    <- .envi("CESS_SEEDS", 3L)
CHAINS   <- .envi("CESS_CHAINS", 4L)
WARMUP   <- .envi("CESS_WARMUP", 1000L)
SAMPLES  <- .envi("CESS_SAMPLES", 1000L)
JB_ITER  <- .envi("CESS_JB_ITER", 16000L)
JB_BURN  <- .envi("CESS_JB_BURN", 4000L)
OUTDIR   <- Sys.getenv("CESS_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
DRAWDIR  <- file.path(OUTDIR, "common_ess_draws")
dir.create(DRAWDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, "common_ess.csv")
COLS <- c("dataset", "seed", "arm", "param", "est", "sd", "n_draws", "chains", "sec",
          "mean_num_steps", "ess_numpyro", "rhat_numpyro", "ess_basic", "ess_bulk",
          "ess_tail", "ess_native")

# ---- data and settings: identical to dev/study_realdata_rotate.R -------------
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
    list(dl = dl, ds = dsv, time_var = "year", lform = y ~ year + age, rform = ~ year | id)
  } else stop("unknown dataset ", ds, call. = FALSE)
}
prefit <- function(d) {
  lme_fit <- lme(d$lform, random = d$rform, data = d$dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  list(lme = lme_fit, cox = coxph(Surv(stime, event) ~ drug, data = d$ds))
}
jx_control <- function(arm, seed) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
              rw2_implementation = "vectorized", dense_mass_spline = TRUE,
              num_warmup = WARMUP, num_samples = SAMPLES, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE)
  if (arm == "R") { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  if (arm == "A") ctl$rotate_absorbable <- FALSE
  ctl
}

# ---- draws -------------------------------------------------------------------
as_draws <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) { k <- length(unlist(v[[1]]))
    return(matrix(vapply(v, function(z) as.numeric(unlist(z)), numeric(k)), ncol = k, byrow = TRUE)) }
  m <- as.matrix(v); storage.mode(m) <- "double"; m
}
name_cols <- function(M, prefix) {
  if (is.null(M)) return(NULL)
  colnames(M) <- if (ncol(M) == 1 && prefix == "alpha") "alpha" else sprintf("%s_%d", prefix, seq_len(ncol(M)) - 1L)
  M
}
draws_jx <- function(ds, d, pf, arm, seed) {
  f <- jm_fit_prefit(pf$lme, pf$cox, data_surv = d$ds, time_var = d$time_var,
                     method = "spline-PH-mcmc", control = jx_control(arm, seed))
  cv <- f$convergence
  if (arm == "R" && !isTRUE(cv$orthogonalize$rotation$applied))
    stop("arm R: rotation not applied - reinstall the backend", call. = FALSE)
  ps <- f$posterior_samples
  M <- cbind(name_cols(as_draws(ps[["beta"]]), "beta"),
             name_cols(as_draws(ps[["alpha"]])[, 1, drop = FALSE], "alpha"),
             name_cols(as_draws(ps[["gamma"]]), "gamma"))
  native <- unlist(f$diagnostics$ess)
  list(draws = M, chains = CHAINS, n_per_chain = SAMPLES, params = colnames(M),
       sec = as.numeric(cv$sampling_time_sec %||% NA_real_),
       mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
       native = setNames(as.numeric(native[colnames(M)]), colnames(M)))
}
# JMbayes2 keeps its samples in jb$mcmc, one component per parameter block,
# each a list (mcmc.list) of per-chain matrices. Chains are concatenated in
# order, matching jmjax's layout.
jb_block <- function(jb, block) {
  x <- jb$mcmc[[block]]
  if (is.null(x)) return(NULL)
  ch <- if (is.matrix(x) || inherits(x, "mcmc")) list(as.matrix(x)) else lapply(x, as.matrix)
  n <- vapply(ch, nrow, integer(1))
  if (length(unique(n)) != 1L) stop("JMbayes2 chains of unequal length in ", block, call. = FALSE)
  list(M = do.call(rbind, ch), chains = length(ch), n = n[1])
}
draws_jb <- function(ds, d, pf, seed) {
  tm <- system.time(jb <- JMbayes2::jm(Surv_object = pf$cox, Mixed_objects = pf$lme,
                                       time_var = d$time_var, n_chains = CHAINS,
                                       n_iter = JB_ITER, n_burnin = JB_BURN,
                                       control = list(Bsplines_degree = 3,
                                                      base_hazard_segments = 6, seed = seed)))
  b <- jb_block(jb, "betas1"); a <- jb_block(jb, "alphas"); g <- jb_block(jb, "gammas")
  if (is.null(b) || is.null(a)) stop("JMbayes2 object has no betas1/alphas draws - ",
                                     "inspect names(jb$mcmc)", call. = FALSE)
  M <- cbind(name_cols(b$M, "beta"), name_cols(a$M[, 1, drop = FALSE], "alpha"),
             if (!is.null(g)) name_cols(g$M, "gamma"))
  st <- jb$statistics$Effective_Size
  native <- c(st$betas1, st$alphas[1], st$gammas)
  list(draws = M, chains = b$chains, n_per_chain = b$n, params = colnames(M),
       sec = as.numeric(tm[["elapsed"]]), mean_num_steps = NA_real_,
       native = setNames(as.numeric(native)[seq_len(ncol(M))], colnames(M)))
}

# ---- ESS ---------------------------------------------------------------------
ess_rows <- function(obj, ds, seed, arm) {
  M <- obj$draws; nc <- obj$chains; n <- obj$n_per_chain
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  e_np <- as.numeric(unlist(rd$n_eff)); r_np <- as.numeric(unlist(rd$r_hat))
  pstat <- function(fun) vapply(seq_len(ncol(M)), function(j) {
    if (!HAVE_POSTERIOR) return(NA_real_)
    fun(matrix(M[, j], nrow = n, ncol = nc))       # iterations x chains
  }, numeric(1))
  data.frame(dataset = ds, seed = seed, arm = arm, param = colnames(M),
             est = colMeans(M), sd = apply(M, 2, stats::sd), n_draws = nrow(M),
             chains = nc, sec = obj$sec, mean_num_steps = obj$mean_num_steps,
             ess_numpyro = e_np, rhat_numpyro = r_np,
             ess_basic = pstat(function(x) posterior::ess_basic(x)),
             ess_bulk  = pstat(function(x) posterior::ess_bulk(x)),
             ess_tail  = pstat(function(x) posterior::ess_tail(x)),
             ess_native = as.numeric(obj$native[colnames(M)]),
             stringsAsFactors = FALSE, row.names = NULL)
}

# ---- run (resumable) ---------------------------------------------------------
done <- if (file.exists(CSV)) utils::read.csv(CSV, stringsAsFactors = FALSE) else NULL
if (!is.null(done) && !identical(names(done), COLS))
  stop(CSV, " has a different column layout; move it aside", call. = FALSE)
have <- function(ds, s, a) !is.null(done) && any(done$dataset == ds & done$seed == s & done$arm == a)
for (ds in DATASETS) {
  d <- load_data(ds); pf <- prefit(d)
  cat(sprintf("\n== %s\n", ds))
  if (any(ARMS %in% c("A", "R")))   # throwaway fit so the first timed fit carries no compilation
    invisible(suppressWarnings(try(jm_fit_prefit(pf$lme, pf$cox, data_surv = d$ds,
      time_var = d$time_var, method = "spline-PH-mcmc",
      control = utils::modifyList(jx_control("R", 1L), list(num_warmup = 5L, num_samples = 5L))),
      silent = TRUE)))
  for (s in seq_len(NSEED)) for (a in ARMS) {
    if (have(ds, s, a)) next
    obj <- tryCatch(if (a == "J") draws_jb(ds, d, pf, s) else draws_jx(ds, d, pf, a, s),
                    error = function(e) { if (grepl("reinstall|inspect", conditionMessage(e))) stop(e)
                      cat("   ", ds, a, "seed", s, "FAILED:", conditionMessage(e), "\n"); NULL })
    if (is.null(obj)) next
    saveRDS(c(obj, list(arm = a, dataset = ds, seed = s)),
            file.path(DRAWDIR, sprintf("%s_%s_seed%d.rds", ds, a, s)))
    rows <- ess_rows(obj, ds, s, a)
    utils::write.table(rows[, COLS], CSV, sep = ",", row.names = FALSE,
                       col.names = !file.exists(CSV), append = file.exists(CSV))
    al <- rows[rows$param == "alpha", ]
    cat(sprintf("   %s seed %d %s: %7.1fs | alpha ESS numpyro %7.0f native %7.0f bulk %s | %d draws\n",
                ds, s, a, obj$sec, al$ess_numpyro, al$ess_native,
                if (is.na(al$ess_bulk)) "-" else sprintf("%.0f", al$ess_bulk), nrow(obj$draws)))
  }
}

# ---- summary -----------------------------------------------------------------
R <- utils::read.csv(CSV, stringsAsFactors = FALSE)
R <- R[R$dataset %in% DATASETS, ]
gm <- function(x) { x <- x[is.finite(x) & x > 0]; if (length(x)) exp(mean(log(x))) else NA_real_ }
est <- c("ess_numpyro", "ess_bulk", "ess_native")
for (ds in DATASETS) {
  X <- R[R$dataset == ds, ]; if (!nrow(X)) next
  cat(sprintf("\n==================== %s: ESS/sec ratios, geometric mean over seeds ====================\n", ds))
  cat(sprintf("%-9s %-9s %12s %12s %12s\n", "param", "ratio", "numpyro", "bulk", "native"))
  for (p in unique(X$param)) for (pr in list(c("R", "J"), c("A", "J"), c("R", "A"))) {
    num <- X[X$param == p & X$arm == pr[1], ]; den <- X[X$param == p & X$arm == pr[2], ]
    m <- merge(num, den, by = "seed"); if (!nrow(m)) next
    v <- vapply(est, function(e) gm((m[[paste0(e, ".x")]] / m$sec.x) / (m[[paste0(e, ".y")]] / m$sec.y)),
                numeric(1))
    cat(sprintf("%-9s %-9s %12s %12s %12s\n", p, paste0(pr[1], "/", pr[2]),
                ifelse(is.na(v), "-", sprintf("%.2fx", v))[1], ifelse(is.na(v), "-", sprintf("%.2fx", v))[2],
                ifelse(is.na(v), "-", sprintf("%.2fx", v))[3]))
  }
  cat("ESS per draw by arm (numpyro estimator), alpha: ",
      paste(sprintf("%s %.3f", c("A", "R", "J"),
                    vapply(c("A", "R", "J"), function(a) mean(X$ess_numpyro[X$arm == a & X$param == "alpha"] /
                                                          X$n_draws[X$arm == a & X$param == "alpha"]), numeric(1))),
            collapse = " | "), "\n", sep = "")
}
cat("\n'numpyro' and 'bulk' apply ONE estimator to every arm; 'native' reproduces the paper's\n",
    "mixed-estimator comparison. If the three columns agree, the estimator was not driving the ratios.\n", sep = "")
