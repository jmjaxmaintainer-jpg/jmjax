# ==============================================================================
# dev/alpha_missing_information.R - why alpha mixes slowly under a blocked
# (Gibbs / Metropolis-within-Gibbs) sampler, measured from jmjax's own
# posterior (vignette Section 8.1, Proposition 8).
#
# THE ARGUMENT. Consider a sampler that updates alpha from its full
# conditional given everything else R (random effects b, beta, gamma,
# baseline hazard, variance components), and updates R as one block. Its
# lag-1 autocorrelation for alpha is exactly
#     Var(E[alpha | R, y]) / Var(alpha | y)  =  1 - E[Var(alpha | R, y)] / Var(alpha | y)
# (Liu, Wong & Kong 1994). That is the fraction of alpha's posterior
# variance explained by the other parameters - its "fraction of missing
# information". This is the best case for a blocked scheme. Metropolis
# steps, or splitting R into more blocks, can only be expected to make it
# slower.
#
# The quantity is estimated here from jmjax's own draws: the R^2 of alpha
# regressed on every other parameter (the Gaussian approximation to
# 1 - E Var(alpha|R)/Var(alpha)), adjusted for the number of regressors.
# From it the script predicts the ESS per draw of that ideal blocked
# sampler, (1 - r)/(1 + r), and compares it with JMbayes2's measured ESS
# per draw on alpha, read from dev/study_realdata_rotate.R's results if
# present. The rotated jmjax fit is used because it mixes well, so the
# draws are close to independent; the target posterior is the same with or
# without the rotation.
#
# Also reported: R^2 given the random effects alone, and given the survival
# block alone (beta_0/baseline hazard, gamma), to show which conditioning
# carries the missing information.
#
#   Rscript dev/alpha_missing_information.R            # aids + pbc2, ~5 min
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
`%||%` <- function(a, b) if (is.null(a)) b else a
CHAINS  <- 4L; WARMUP <- 1000L
SAMPLES <- as.integer(Sys.getenv("AMI_SAMPLES", "5000"))
OUTDIR  <- Sys.getenv("REAL_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
JCSV    <- file.path(OUTDIR, "study_realdata_rotate.csv")

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
  } else {
    utils::data("pbc2", package = "JMbayes2", envir = environment())
    utils::data("pbc2.id", package = "JMbayes2", envir = environment())
    dl <- pbc2[, c("id", "year", "serBilir", "age")]
    dl$id <- as.integer(as.character(dl$id)); dl$y <- log(dl$serBilir)
    s <- pbc2.id
    dsv <- data.frame(id = as.integer(as.character(s$id)), stime = s$years,
                      event = as.integer(s$status != "alive"), drug = s$drug)
    list(dl = dl, ds = dsv, time_var = "year",
         lform = y ~ year + age, rform = ~ year | id)
  }
}

flat <- function(v) {                     # list-of-draws site -> draws x dim matrix
  if (is.null(v)) return(NULL)
  if (!is.list(v)) { m <- as.matrix(v); storage.mode(m) <- "double"; return(m) }
  k <- length(unlist(v[[1]]))
  matrix(vapply(v, function(z) as.numeric(unlist(z)), numeric(k)), ncol = k, byrow = TRUE)
}
adj_r2 <- function(y, X) {
  X <- X[, apply(X, 2, stats::sd) > 0, drop = FALSE]
  fit <- stats::lm.fit(cbind(1, X), y)
  r2 <- 1 - sum(fit$residuals^2) / sum((y - mean(y))^2)
  n <- length(y); p <- fit$rank - 1L
  c(r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1), p = p)
}

jb <- if (file.exists(JCSV)) utils::read.csv(JCSV, stringsAsFactors = FALSE) else NULL
SKIP <- c("alpha", "b_std", "b_gen_U", "b_gen_V", "b0_gen_u", "b0_gen_v",
          "b_std_rest", "beta_corrected", "b_raw", "L_corr")

cat("\nFraction of alpha's posterior variance explained by the other parameters\n")
cat("(adjusted R^2; = lag-1 autocorrelation of an ideal blocked sampler for alpha)\n\n")
for (ds in c("aids", "pbc2")) {
  d <- load_data(ds)
  lme_fit <- lme(d$lform, random = d$rform, data = d$dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- coxph(Surv(stime, event) ~ drug, data = d$ds)
  f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$ds, time_var = d$time_var,
                     method = "spline-PH-mcmc",
                     control = list(n_interior_knots = 5, spline_prior = "penalized",
                                    rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                                    rotate_absorbable = TRUE, dense_mass_generator_beta = TRUE,
                                    num_warmup = WARMUP, num_samples = SAMPLES,
                                    num_chains = CHAINS, seed = 7L, progress_bar = FALSE))
  ps <- f$posterior_samples
  a  <- flat(ps[["alpha"]])[, 1]
  sites <- setdiff(names(ps), SKIP)
  M <- lapply(sites, function(s) flat(ps[[s]])); names(M) <- sites
  M <- M[!vapply(M, is.null, logical(1))]
  M <- M[vapply(M, nrow, integer(1)) == length(a)]
  all_rest <- do.call(cbind, M)
  b_only   <- M[["b"]]
  surv     <- do.call(cbind, M[intersect(names(M), c("gamma", "W", "W01", "z_step", "tau_w"))])
  r_all <- adj_r2(a, all_rest); r_b <- adj_r2(a, b_only)
  r_s <- if (!is.null(surv)) adj_r2(a, surv) else c(r2 = NA, p = 0)
  pred <- (1 - r_all[["r2"]]) / (1 + r_all[["r2"]])
  jess <- if (!is.null(jb)) {
    x <- jb[jb$dataset == ds & jb$arm == "J" & jb$param == "alpha", ]
    if (nrow(x)) mean(x$ess / x$n_draws) else NA } else NA
  own <- unlist(f$diagnostics$ess)[["alpha"]] / (CHAINS * SAMPLES)
  cat(sprintf("%-5s draws %d | R^2(alpha | everything else) %.3f  [%d regressors]\n",
              ds, length(a), r_all[["r2"]], as.integer(r_all[["p"]])))
  cat(sprintf("      R^2(alpha | random effects only) %.3f | R^2(alpha | survival block only) %.3f\n",
              r_b[["r2"]], r_s[["r2"]]))
  cat(sprintf("      ideal blocked sampler, predicted alpha ESS/draw  %.3f\n", pred))
  cat(sprintf("      JMbayes2, measured alpha ESS/draw               %s\n",
              if (is.na(jess)) "- (run dev/study_realdata_rotate.R with REAL_ARMS=A,R,J)" else sprintf("%.3f", jess)))
  cat(sprintf("      jmjax NUTS (rotated), measured alpha ESS/draw   %.3f\n\n", own))
}
