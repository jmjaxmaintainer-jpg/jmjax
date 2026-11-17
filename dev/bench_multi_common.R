# ==============================================================================
# dev/bench_multi_common.R - shared code for the jmjax 0.5.0 multi-marker
# benchmarks (Milestone 1, Phase C):
#   dev/bench_multi_pbc2.R  JMbayes2 multi-outcome comparison on pbc2 (q = 4, 6)
#   dev/bench_multi_sim.R   simulations with K = 2, 3 markers, sparse / dense visits
# Sourced by both; not run on its own.
#
# ONE ESS ESTIMATOR. Every arm (jmjax unrotated "A", jmjax rotated "R",
# JMbayes2 "J") is reduced to a matrix of raw draws (chains concatenated) of
# the same quantities on the same scale:
#   beta_<marker>_<j>   fixed effects (original scale)
#   alpha_<marker>      association parameters
#   D[i, j]             random-effects covariance, lower triangle (i >= j),
#                       random effects stacked by marker (intercept, slope, ...)
#   sigma_e_<marker>    residual SDs
# and ESS is computed from those draws with posterior::ess_bulk (headline)
# and NumPyro's estimator via the package's recompute_site_diagnostics()
# (check). Native ESS values are not used.
#
# TIME. jmjax: NUTS wall time (warm-up + sampling, all chains), which includes
# JAX compilation; the compilation estimate (the wall time of a 5 + 5
# iteration fit of the same model) is stored as sec_compile so the tables can
# show ESS/s both ways. JMbayes2: the whole jm() call.
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
if (!requireNamespace("posterior", quietly = TRUE))
  stop("package 'posterior' is required (install.packages('posterior'))", call. = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a
.envl <- function(nm, d) { v <- Sys.getenv(nm, ""); if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d }
.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }

BM_COLS <- c("study", "config", "q", "seed", "arm", "block", "param", "truth", "est", "sd",
             "n_draws", "chains", "sec", "sec_compile", "mean_num_steps", "n_div",
             "ess_bulk", "ess_tail", "rhat_posterior", "ess_numpyro", "rhat_numpyro")

block_of <- function(p) ifelse(grepl("^beta_", p), "beta",
                        ifelse(grepl("^alpha_", p), "alpha",
                        ifelse(grepl("^D\\[", p), "Sigma",
                        ifelse(grepl("^sigma_e_", p), "sigma_e", "other"))))

# ---- jmjax ---------------------------------------------------------------------
jx_control <- function(arm, seed, chains, warmup, samples) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized", dense_mass_spline = TRUE,
              num_warmup = warmup, num_samples = samples, num_chains = chains,
              seed = seed, progress_bar = FALSE)
  if (arm == "R") { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  if (arm == "A") ctl$rotate_absorbable <- FALSE
  ctl
}

fit_jx <- function(forms, rforms, dls, ds, surv_formula, time_var, arm, seed,
                   chains, warmup, samples) {
  jm_bayes(long_formula = forms, surv_formula = surv_formula,
           data_long = dls, data_surv = ds, id_var = "id", time_var = time_var,
           random_formula = rforms,
           control = jx_control(arm, seed, chains, warmup, samples))
}

draws_jx <- function(f, markers, block_sizes) {
  ps <- f$posterior_samples
  v <- function(nm) as.numeric(unlist(ps[[nm]]))
  bn <- f$model_info$beta_map$name
  B <- sapply(bn, v)
  A <- sapply(f$model_info$alpha_names, v)
  S <- sapply(f$model_info$sigma_e_names, v)
  Q <- sum(block_sizes)
  sdb <- sapply(seq_len(Q) - 1L, function(i) v(sprintf("sigma_b%d", i)))
  Dm <- NULL; dn <- character(0)
  for (j in seq_len(Q)) for (i in j:Q) {
    r <- if (i == j) 1 else v(sprintf("rho_%d_%d", j - 1L, i - 1L))
    if (!length(r)) r <- 0        # block-diagonal fits: no cross-marker rho
    Dm <- cbind(Dm, sdb[, i] * sdb[, j] * r)
    dn <- c(dn, sprintf("D[%d, %d]", i, j))
  }
  colnames(Dm) <- dn
  M <- cbind(B, A, Dm, S)
  cv <- f$convergence
  list(draws = M, chains = f$mcmc_settings$num_chains, n_per_chain = f$mcmc_settings$num_samples,
       sec = as.numeric(cv$sampling_time_sec %||% NA_real_),
       mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
       n_div = as.numeric(cv$n_divergences %||% NA_real_),
       rotation = cv$orthogonalize$rotation)
}

# Wall time of a 5 + 5 fit of the same model: the compilation estimate.
compile_time_jx <- function(forms, rforms, dls, ds, surv_formula, time_var, arm, chains) {
  f <- suppressWarnings(suppressMessages(
    fit_jx(forms, rforms, dls, ds, surv_formula, time_var, arm, 1L, chains, 5L, 5L)))
  as.numeric(f$convergence$sampling_time_sec)
}

# ---- JMbayes2 ------------------------------------------------------------------
# JMbayes2 needs every lme() fitted to the SAME data frame. The markers are
# observed at different times, so build one frame over the union of (id,
# time) rows with one response column per marker (named after the marker,
# NA where that marker was not measured) and the subject-level covariates;
# each lme() drops its own NA rows (na.exclude), as in the JMbayes2 docs.
jb_common_data <- function(forms, dls, time_var) {
  W <- NULL; covs <- NULL
  for (m in names(forms)) {
    d <- dls[[m]]; resp <- all.vars(forms[[m]])[1]
    x <- data.frame(id = d$id, t = d[[time_var]], v = d[[resp]])
    names(x) <- c("id", time_var, m)
    W <- if (is.null(W)) x else merge(W, x, by = c("id", time_var), all = TRUE)
    cv <- setdiff(all.vars(forms[[m]][[3]]), time_var)
    if (length(cv)) {
      s <- d[!duplicated(d$id), c("id", cv), drop = FALSE]
      covs <- if (is.null(covs)) s else merge(covs, s[, c("id", setdiff(cv, names(covs))), drop = FALSE],
                                              by = "id", all = TRUE)
    }
  }
  if (!is.null(covs)) W <- merge(W, covs, by = "id", all.x = TRUE)
  W <- W[order(W$id, W[[time_var]]), , drop = FALSE]
  rownames(W) <- NULL
  W
}

# Mixed_objects: one lme() per marker on the common frame. Draws from
# jb$mcmc, chains concatenated.
fit_jb <- function(forms, rforms, dls, ds, surv_formula, time_var, seed, chains,
                   n_iter, n_burnin) {
  W <- jb_common_data(forms, dls, time_var)
  lmes <- lapply(names(forms), function(m) {
    rf <- stats::as.formula(paste("~", paste(deparse(rforms[[m]][[2]]), collapse = ""), "| id"))
    fx <- stats::as.formula(paste(m, "~", paste(deparse(forms[[m]][[3]]), collapse = "")))
    do.call(nlme::lme, list(fixed = fx, random = rf, data = W,
                            na.action = stats::na.exclude,
                            control = nlme::lmeControl(opt = "optim", msMaxIter = 200,
                                                       niterEM = 100)))
  })
  cox <- survival::coxph(surv_formula, data = ds, x = TRUE)
  tm <- system.time(jb <- JMbayes2::jm(Surv_object = cox, Mixed_objects = lmes,
                                       time_var = time_var, n_chains = chains,
                                       n_iter = n_iter, n_burnin = n_burnin,
                                       control = list(Bsplines_degree = 3,
                                                      base_hazard_segments = 6, seed = seed)))
  list(jb = jb, sec = as.numeric(tm[["elapsed"]]))
}

jb_block <- function(jb, block) {
  x <- jb$mcmc[[block]]
  if (is.null(x)) return(NULL)
  ch <- if (is.matrix(x) || inherits(x, "mcmc")) list(as.matrix(x)) else lapply(x, as.matrix)
  n <- vapply(ch, nrow, integer(1))
  if (length(unique(n)) != 1L) stop("JMbayes2 chains of unequal length in ", block, call. = FALSE)
  list(M = do.call(rbind, ch), chains = length(ch), n = n[1])
}

draws_jb <- function(fj, markers, block_sizes) {
  jb <- fj$jb
  B <- NULL; nchain <- NA; nper <- NA
  for (k in seq_along(markers)) {
    b <- jb_block(jb, paste0("betas", k))
    if (is.null(b)) stop("JMbayes2 object has no betas", k, " draws - inspect names(jb$mcmc)",
                         call. = FALSE)
    colnames(b$M) <- sprintf("beta_%s_%d", markers[k], seq_len(ncol(b$M)) - 1L)
    B <- cbind(B, b$M); nchain <- b$chains; nper <- b$n
  }
  a <- jb_block(jb, "alphas")$M
  if (ncol(a) != length(markers))
    stop("expected one JMbayes2 alpha per marker (value forms); got ", ncol(a), call. = FALSE)
  colnames(a) <- paste0("alpha_", markers)
  # D: lower triangle including the diagonal. Use the column names when they
  # say which element each column is; otherwise JMbayes2's vech order.
  Dm <- jb_block(jb, "D")$M
  Q <- sum(block_sizes)
  if (ncol(Dm) != Q * (Q + 1) / 2)
    stop("JMbayes2 D has ", ncol(Dm), " columns; expected ", Q * (Q + 1) / 2, call. = FALSE)
  cn <- colnames(Dm)
  ij <- if (!is.null(cn) && all(grepl("^D\\[[0-9]+, *[0-9]+\\]$", cn))) {
    t(vapply(regmatches(cn, gregexpr("[0-9]+", cn)), function(z) as.integer(z), integer(2)))
  } else {
    L <- which(lower.tri(diag(Q), diag = TRUE), arr.ind = TRUE); L[order(L[, 2], L[, 1]), ]
  }
  ij <- cbind(pmax(ij[, 1], ij[, 2]), pmin(ij[, 1], ij[, 2]))
  colnames(Dm) <- sprintf("D[%d, %d]", ij[, 1], ij[, 2])
  s <- jb_block(jb, "sigmas")$M
  if (!is.null(s)) { s <- s[, seq_along(markers), drop = FALSE]
    colnames(s) <- paste0("sigma_e_", markers) }
  list(draws = cbind(B, a, Dm, s), chains = nchain, n_per_chain = nper, sec = fj$sec,
       mean_num_steps = NA_real_, n_div = NA_real_, rotation = NULL)
}

# ---- ESS rows -------------------------------------------------------------------
ess_rows <- function(obj, study, config, q, seed, arm, truth = NULL, sec_compile = NA_real_) {
  M <- obj$draws; nc <- obj$chains; n <- obj$n_per_chain
  stopifnot(nrow(M) == nc * n)
  rd <- jmjax:::.get_backend()$mcmc_model$recompute_site_diagnostics(M, as.integer(nc))
  per <- function(fun) vapply(seq_len(ncol(M)), function(j) fun(matrix(M[, j], nrow = n, ncol = nc)),
                              numeric(1))
  tr <- if (is.null(truth)) rep(NA_real_, ncol(M)) else as.numeric(truth[colnames(M)])
  data.frame(study = study, config = config, q = q, seed = seed, arm = arm,
             block = block_of(colnames(M)), param = colnames(M), truth = tr,
             est = colMeans(M), sd = apply(M, 2, stats::sd), n_draws = nrow(M), chains = nc,
             sec = obj$sec, sec_compile = sec_compile, mean_num_steps = obj$mean_num_steps,
             n_div = obj$n_div,
             ess_bulk = per(posterior::ess_bulk), ess_tail = per(posterior::ess_tail),
             rhat_posterior = per(posterior::rhat),
             ess_numpyro = as.numeric(unlist(rd$n_eff)),
             rhat_numpyro = as.numeric(unlist(rd$r_hat)),
             stringsAsFactors = FALSE, row.names = NULL)
}

append_rows <- function(rows, csv) {
  utils::write.table(rows[, BM_COLS], csv, sep = ",", row.names = FALSE,
                     col.names = !file.exists(csv), append = file.exists(csv))
}
read_done <- function(csv) {
  if (!file.exists(csv)) return(NULL)
  d <- utils::read.csv(csv, stringsAsFactors = FALSE)
  if (!identical(names(d), BM_COLS)) stop(csv, " has a different column layout; move it aside",
                                          call. = FALSE)
  d
}

# ---- tables -----------------------------------------------------------------------
gm <- function(x) { x <- x[is.finite(x) & x > 0]; if (length(x)) exp(mean(log(x))) else NA_real_ }

# One row per config x arm x seed: time, ESS/s (min and median over each block,
# bulk estimator, with and without compilation), max R-hat, divergences.
fit_summary <- function(R) {
  keys <- unique(R[, c("study", "config", "q", "seed", "arm")])
  out <- lapply(seq_len(nrow(keys)), function(r) {
    k <- keys[r, ]
    X <- R[R$study == k$study & R$config == k$config & R$seed == k$seed & R$arm == k$arm, ]
    sec <- X$sec[1]; secn <- sec - ifelse(is.finite(X$sec_compile[1]), X$sec_compile[1], 0)
    row <- data.frame(k, sec = sec, sec_net = secn, n_div = X$n_div[1],
                      max_rhat = max(X$rhat_posterior, na.rm = TRUE),
                      stringsAsFactors = FALSE)
    for (b in c("beta", "alpha", "Sigma")) {
      e <- X$ess_bulk[X$block == b]; en <- X$ess_numpyro[X$block == b]
      row[[paste0(b, "_min")]] <- min(e) / sec
      row[[paste0(b, "_med")]] <- stats::median(e) / sec
      row[[paste0(b, "_min_net")]] <- min(e) / secn
      row[[paste0(b, "_min_np")]] <- min(en) / sec
    }
    row
  })
  do.call(rbind, out)
}

# Geometric mean over seeds per config x arm, and ratios R/A, R/J, A/J.
tables_md <- function(S, title, file) {
  arms <- intersect(c("A", "R", "J"), unique(S$arm))
  lab <- c(A = "jmjax unrotated", R = "jmjax rotated", J = "JMbayes2")
  met <- c("beta_min", "beta_med", "alpha_min", "alpha_med", "Sigma_min", "Sigma_med")
  L <- c(sprintf("## %s", title), "",
         "ESS/s: posterior::ess_bulk on raw draws (same estimator for every arm), divided by",
         "wall time (jmjax: NUTS warm-up + sampling incl. JAX compilation; JMbayes2: jm()).",
         "min / med = smallest / median over the parameters of the block; geometric mean over seeds.",
         "")
  for (cf in unique(S$config)) {
    X <- S[S$config == cf, ]
    L <- c(L, sprintf("### %s (q = %d, %d seed(s))", cf, X$q[1], length(unique(X$seed))), "",
           "| arm | sec | beta min | beta med | alpha min | alpha med | Sigma min | Sigma med | max R-hat | divergences |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (a in arms) {
      Y <- X[X$arm == a, ]; if (!nrow(Y)) next
      L <- c(L, sprintf("| %s | %.0f | %s | %.3f | %s |", lab[a], mean(Y$sec),
                        paste(sprintf("%.2f", vapply(met, function(m) gm(Y[[m]]), 1)), collapse = " | "),
                        max(Y$max_rhat),
                        if (all(is.na(Y$n_div))) "-" else as.character(sum(Y$n_div))))
    }
    pr <- list(c("R", "A"), c("R", "J"), c("A", "J"))
    rr <- character(0)
    for (p in pr) {
      if (!all(p %in% X$arm)) next
      m <- merge(X[X$arm == p[1], ], X[X$arm == p[2], ], by = "seed")
      v <- vapply(c("beta_min", "alpha_min", "Sigma_min", "beta_min_net", "alpha_min_net",
                    "Sigma_min_net"),
                  function(mm) gm(m[[paste0(mm, ".x")]] / m[[paste0(mm, ".y")]]), 1)
      rr <- c(rr, sprintf("| %s/%s | %s |", p[1], p[2], paste(sprintf("%.2fx", v), collapse = " | ")))
    }
    if (length(rr)) {
      L <- c(L, "", "Ratios of min ESS/s (geometric mean over seeds); 'net' excludes the jmjax compilation estimate.", "",
             "| ratio | beta | alpha | Sigma | beta (net) | alpha (net) | Sigma (net) |",
             "|---|---:|---:|---:|---:|---:|---:|", rr)
    }
    L <- c(L, "")
  }
  writeLines(L, file)
  cat(L, sep = "\n")
  invisible(L)
}
