# ==============================================================================
# dev/bench_m1_common.R - shared code for the paper-grade multi-marker
# benchmarks (Milestone 1 rerun, jmjax 0.5.1):
#   dev/bench_m1_pbc2.R  pbc2, q = 4 and q = 6, arms A R C J, 10 seeds
#   dev/bench_m1_sim.R   K = 2, 3 markers x sparse / dense visits, arms A R C,
#                        20 replicates for efficiency, 100 for coverage
# Sourced by both; not run on its own. Supersedes dev/bench_multi_*.R (the
# 0.5.0 development runs), which stay as they were.
#
# ARMS
#   A  jmjax, unrotated (rotate_absorbable = FALSE)
#   R  jmjax, rotated, dense (beta, U) block                (the default)
#   C  jmjax, rotated, absorbable block CENTRED: W = U L', dense (beta, W)
#      block (control$center_absorbable = TRUE)
#   J  JMbayes2
#
# PAPER-GRADE BAR. A run counts only if EVERY reported parameter (beta,
# alpha, the random-effects covariance D, sigma_e) has rank-normalized split
# R-hat < 1.01 and bulk and tail ESS > 400 (posterior::rhat, ess_bulk,
# ess_tail on the raw draws, chains kept apart). A run that misses it is
# repeated with twice the draws per chain (same seed, same warm-up), up to
# BM_MAX_DOUBLINGS times; the attempt that meets it is the one reported, and
# its wall time is the one used. Runs that never meet it are kept in the CSV
# (meets_bar = FALSE) and left out of every ESS/s table.
#
# ONE ESS ESTIMATOR for every arm: posterior::ess_bulk (and ess_tail) on raw
# draws of the same quantities on the same (reported) scale.
#
# TIME. jmjax: NUTS wall time (warm-up + sampling, all chains), JAX
# compilation INCLUDED; the compilation is also estimated separately for
# every fit as the wall time of a 5 + 5-iteration run of the same model on
# the same data (sec_compile). JMbayes2: the whole jm() call.
#
# HARDWARE. Both samplers run their chains in parallel, one core per chain
# (jmjax: one XLA CPU device per chain, JMJAX_NUM_DEVICES; JMbayes2:
# control$cores = chains). hardware.txt in the output folder records the
# machine, the core counts and every version.
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
if (!requireNamespace("posterior", quietly = TRUE))
  stop("package 'posterior' is required (install.packages('posterior'))", call. = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a
.envl <- function(nm, d) { v <- Sys.getenv(nm, ""); if (nzchar(v)) strsplit(v, ",", fixed = TRUE)[[1]] else d }
.envi <- function(nm, d) { v <- suppressWarnings(as.integer(Sys.getenv(nm, "")))
  if (length(v) == 1L && is.finite(v) && v > 0) v else d }

CHAINS   <- .envi("BM_CHAINS", 4L)
WARMUP   <- .envi("BM_WARMUP", 2000L)
SAMPLES  <- .envi("BM_SAMPLES", 4000L)     # per chain, first attempt
JB_KEEP  <- .envi("BM_JB_KEEP", 12000L)    # JMbayes2 kept iterations per chain, first attempt
JB_BURN  <- .envi("BM_JB_BURN", 4000L)
MAX_DBL  <- .envi("BM_MAX_DOUBLINGS", 2L)  # up to 4x the first attempt's draws
RHAT_MAX <- as.numeric(Sys.getenv("BM_RHAT_MAX", "1.01"))
ESS_MIN  <- as.numeric(Sys.getenv("BM_ESS_MIN", "400"))

BM_COLS <- c("study", "config", "q", "seed", "arm", "attempt", "meets_bar", "block", "param",
             "truth", "est", "sd", "q025", "q975", "n_draws", "chains", "sec", "sec_compile",
             "mean_num_steps", "n_div", "ess_bulk", "ess_tail", "rhat")

block_of <- function(p) ifelse(grepl("^beta_", p), "beta",
                        ifelse(grepl("^alpha_", p), "alpha",
                        ifelse(grepl("^D\\[", p), "Sigma",
                        ifelse(grepl("^sigma_e_", p), "sigma_e", "other"))))

# ---- hardware ------------------------------------------------------------------
write_hardware <- function(outdir) {
  be <- jmjax:::.get_backend()
  jx <- reticulate::import("jax"); np <- reticulate::import("numpyro")
  cpu <- tryCatch(system("sysctl -n machdep.cpu.brand_string", intern = TRUE),
                  error = function(e) NA_character_)
  pcores <- tryCatch(system("sysctl -n hw.perflevel0.physicalcpu", intern = TRUE),
                     error = function(e) NA_character_)
  ecores <- tryCatch(system("sysctl -n hw.perflevel1.physicalcpu", intern = TRUE),
                     error = function(e) NA_character_)
  si <- Sys.info()
  L <- c(sprintf("written:          %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
         sprintf("machine:          %s %s (%s)", si[["sysname"]], si[["release"]], si[["machine"]]),
         sprintf("cpu:              %s", paste(cpu, collapse = " ")),
         sprintf("cores:            %d physical, %d logical (performance %s, efficiency %s)",
                 parallel::detectCores(logical = FALSE), parallel::detectCores(logical = TRUE),
                 paste(pcores, collapse = ""), paste(ecores, collapse = "")),
         sprintf("R:                %s", R.version.string),
         sprintf("jmjax:            %s", as.character(utils::packageVersion("jmjax"))),
         sprintf("jax / numpyro:    %s / %s", jx$`__version__`, np$`__version__`),
         sprintf("jax CPU devices:  %d (JMJAX_NUM_DEVICES=%s); chains run in parallel, one device each",
                 as.integer(jx$local_device_count()), Sys.getenv("JMJAX_NUM_DEVICES", "<unset>")),
         sprintf("XLA_FLAGS:        %s", Sys.getenv("XLA_FLAGS", "<unset>")),
         sprintf("JMbayes2:         %s; jm(control = list(cores = %d))",
                 if (requireNamespace("JMbayes2", quietly = TRUE))
                   as.character(utils::packageVersion("JMbayes2")) else "not installed",
                 jb_cores()),
         sprintf("posterior:        %s", as.character(utils::packageVersion("posterior"))),
         sprintf("chains:           %d; jmjax warm-up %d, first attempt %d draws per chain;",
                 CHAINS, WARMUP, SAMPLES),
         sprintf("                  JMbayes2 burn-in %d, first attempt %d kept per chain;",
                 JB_BURN, JB_KEEP),
         sprintf("                  up to %d doublings; bar: R-hat < %.3f, bulk and tail ESS > %.0f",
                 MAX_DBL, RHAT_MAX, ESS_MIN))
  writeLines(L, file.path(outdir, "hardware.txt"))
  cat(L, sep = "\n"); cat("\n")
  if (as.integer(jx$local_device_count()) < CHAINS)
    warning("fewer JAX CPU devices than chains: jmjax chains will run sequentially. ",
            "Set JMJAX_NUM_DEVICES >= ", CHAINS, " before R starts.", call. = FALSE)
}
jb_cores <- function() as.integer(min(CHAINS, parallel::detectCores(logical = FALSE)))

# ---- jmjax ---------------------------------------------------------------------
jx_control <- function(arm, seed, warmup, samples) {
  ctl <- list(n_interior_knots = 5, spline_prior = "penalized", dense_mass_spline = TRUE,
              num_warmup = warmup, num_samples = samples, num_chains = CHAINS,
              seed = seed, progress_bar = FALSE, return_b_draws = FALSE)
  if (arm == "A") ctl$rotate_absorbable <- FALSE
  if (arm %in% c("R", "C")) { ctl$rotate_absorbable <- TRUE; ctl$dense_mass_generator_beta <- TRUE }
  if (arm == "C") ctl$center_absorbable <- TRUE
  ctl
}
fit_jx <- function(D, arm, seed, warmup, samples) {
  suppressMessages(jm_bayes(long_formula = D$forms, surv_formula = D$surv_formula,
           data_long = D$dls, data_surv = D$ds, id_var = "id", time_var = D$time_var,
           random_formula = D$rforms, control = jx_control(arm, seed, warmup, samples)))
}
draws_jx <- function(f, block_sizes) {
  ps <- f$posterior_samples
  v <- function(nm) as.numeric(unlist(ps[[nm]]))
  B <- sapply(f$model_info$beta_map$name, v)
  A <- sapply(f$model_info$alpha_names, v)
  S <- sapply(f$model_info$sigma_e_names, v)
  Q <- sum(block_sizes)
  sdb <- sapply(seq_len(Q) - 1L, function(i) v(sprintf("sigma_b%d", i)))
  Dm <- NULL; dn <- character(0)
  for (j in seq_len(Q)) for (i in j:Q) {
    r <- if (i == j) 1 else v(sprintf("rho_%d_%d", j - 1L, i - 1L))
    if (!length(r)) next                 # block-diagonal fits: structural zero, not reported
    Dm <- cbind(Dm, sdb[, i] * sdb[, j] * r); dn <- c(dn, sprintf("D[%d, %d]", i, j))
  }
  colnames(Dm) <- dn
  cv <- f$convergence
  list(draws = cbind(B, A, Dm, S), chains = f$mcmc_settings$num_chains,
       n_per_chain = f$mcmc_settings$num_samples,
       sec = as.numeric(cv$sampling_time_sec %||% NA_real_),
       mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
       n_div = as.numeric(cv$n_divergences %||% NA_real_),
       rotation = cv$orthogonalize$rotation, num_devices = cv$num_devices %||% NA)
}
compile_time_jx <- function(D, arm) {
  f <- suppressWarnings(fit_jx(D, arm, 1L, 5L, 5L))
  as.numeric(f$convergence$sampling_time_sec)
}

# ---- JMbayes2 ------------------------------------------------------------------
# JMbayes2 needs every lme() fitted to the SAME data frame: one frame over the
# union of (id, time) rows, one response column per marker (NA where not
# measured), subject-level covariates; each lme() drops its own NA rows.
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
  W <- W[order(W$id, W[[time_var]]), , drop = FALSE]; rownames(W) <- NULL
  W
}
fit_jb <- function(D, seed, keep) {
  W <- jb_common_data(D$forms, D$dls, D$time_var)
  lmes <- lapply(names(D$forms), function(m) {
    rf <- stats::as.formula(paste("~", paste(deparse(D$rforms[[m]][[2]]), collapse = ""), "| id"))
    fx <- stats::as.formula(paste(m, "~", paste(deparse(D$forms[[m]][[3]]), collapse = "")))
    do.call(nlme::lme, list(fixed = fx, random = rf, data = W, na.action = stats::na.exclude,
                            control = nlme::lmeControl(opt = "optim", msMaxIter = 200,
                                                       niterEM = 100)))
  })
  cox <- survival::coxph(D$surv_formula, data = D$ds, x = TRUE)
  tm <- system.time(jb <- JMbayes2::jm(Surv_object = cox, Mixed_objects = lmes,
                                       time_var = D$time_var, n_chains = CHAINS,
                                       n_iter = JB_BURN + keep, n_burnin = JB_BURN,
                                       control = list(Bsplines_degree = 3,
                                                      base_hazard_segments = 6,
                                                      cores = jb_cores(), seed = seed)))
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
  jb <- fj$jb; B <- NULL
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
  Dm <- jb_block(jb, "D")$M; Q <- sum(block_sizes)
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
  if (!is.null(s)) { s <- s[, seq_along(markers), drop = FALSE]; colnames(s) <- paste0("sigma_e_", markers) }
  list(draws = cbind(B, a, Dm, s), chains = nchain, n_per_chain = nper, sec = fj$sec,
       mean_num_steps = NA_real_, n_div = NA_real_, rotation = NULL, num_devices = NA)
}

# ---- convergence bar and rows ----------------------------------------------------
diag_draws <- function(obj) {
  M <- obj$draws; nc <- obj$chains; n <- obj$n_per_chain
  stopifnot(nrow(M) == nc * n)
  per <- function(fun) vapply(seq_len(ncol(M)), function(j)
    suppressWarnings(fun(matrix(M[, j], nrow = n, ncol = nc))), numeric(1))
  list(ess_bulk = per(posterior::ess_bulk), ess_tail = per(posterior::ess_tail),
       rhat = per(posterior::rhat))
}
meets_bar <- function(dg) all(is.finite(dg$rhat)) && max(dg$rhat) < RHAT_MAX &&
  min(dg$ess_bulk) > ESS_MIN && min(dg$ess_tail) > ESS_MIN

# Fit one arm, doubling the draws until the bar is met (or MAX_DBL doublings).
run_arm <- function(D, arm, seed) {
  for (att in 0:MAX_DBL) {
    obj <- if (arm == "J") {
      draws_jb(fit_jb(D, seed, JB_KEEP * 2^att), D$markers, D$block_sizes)
    } else {
      f <- fit_jx(D, arm, seed, WARMUP, SAMPLES * 2^att)
      if (arm %in% c("R", "C") && !isTRUE(f$convergence$orthogonalize$rotation$applied))
        stop("arm ", arm, ": rotation not applied", call. = FALSE)
      if (arm == "C" && !isTRUE(f$convergence$orthogonalize$rotation$center_absorbable))
        stop("arm C: center_absorbable not applied - reinstall jmjax", call. = FALSE)
      o <- draws_jx(f, D$block_sizes); rm(f); o
    }
    dg <- diag_draws(obj)
    obj$attempt <- att; obj$diag <- dg; obj$meets <- meets_bar(dg)
    if (obj$meets) break
  }
  obj
}

rows_of <- function(obj, study, config, q, seed, arm, truth = NULL, sec_compile = NA_real_) {
  M <- obj$draws; dg <- obj$diag
  tr <- if (is.null(truth)) rep(NA_real_, ncol(M)) else as.numeric(truth[colnames(M)])
  qs <- apply(M, 2, stats::quantile, probs = c(0.025, 0.975), names = FALSE)
  data.frame(study = study, config = config, q = q, seed = seed, arm = arm,
             attempt = obj$attempt, meets_bar = obj$meets,
             block = block_of(colnames(M)), param = colnames(M), truth = tr,
             est = colMeans(M), sd = apply(M, 2, stats::sd), q025 = qs[1, ], q975 = qs[2, ],
             n_draws = nrow(M), chains = obj$chains, sec = obj$sec, sec_compile = sec_compile,
             mean_num_steps = obj$mean_num_steps, n_div = obj$n_div,
             ess_bulk = dg$ess_bulk, ess_tail = dg$ess_tail, rhat = dg$rhat,
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
  # A fit written twice (same design, replicate, arm) keeps its first copy.
  # Fits are deterministic given the seed, so the copies differ only in wall
  # time; the first is the one run beside the other arms.
  d[!duplicated(d[, c("study", "config", "seed", "arm", "param")]), , drop = FALSE]
}
progress_line <- function(rows, label) {
  b <- function(bl) min(rows$ess_bulk[rows$block == bl]) / rows$sec[1]
  cat(sprintf("   %s: %s | %7.1fs (compile %.1fs) x%d draws | min ESS/s beta %.1f alpha %.1f Sigma %.1f | max R-hat %.4f | min ESS bulk %.0f tail %.0f%s\n",
              label, if (rows$meets_bar[1]) "BAR MET" else "bar NOT met",
              rows$sec[1], rows$sec_compile[1], 2^rows$attempt[1], b("beta"), b("alpha"), b("Sigma"),
              max(rows$rhat), min(rows$ess_bulk), min(rows$ess_tail),
              if (is.finite(rows$n_div[1])) sprintf(" | %d div", as.integer(rows$n_div[1])) else ""))
}

# ---- tables ----------------------------------------------------------------------
# Per fit: wall time, compile estimate, smallest bulk ESS/s per block (with and
# without compilation), divergences, max R-hat, min ESS.
fit_summary <- function(R) {
  keys <- unique(R[, c("study", "config", "q", "seed", "arm")])
  out <- lapply(seq_len(nrow(keys)), function(r) {
    k <- keys[r, ]
    X <- R[R$study == k$study & R$config == k$config & R$seed == k$seed & R$arm == k$arm, ]
    sec <- X$sec[1]; secn <- sec - ifelse(is.finite(X$sec_compile[1]), X$sec_compile[1], 0)
    row <- data.frame(k, meets_bar = X$meets_bar[1], attempt = X$attempt[1],
                      draws_per_chain = X$n_draws[1] / X$chains[1],
                      sec = sec, sec_compile = X$sec_compile[1], n_div = X$n_div[1],
                      max_rhat = max(X$rhat), min_ess_bulk = min(X$ess_bulk),
                      min_ess_tail = min(X$ess_tail), stringsAsFactors = FALSE)
    for (b in c("beta", "alpha", "Sigma")) {
      e <- X$ess_bulk[X$block == b]
      row[[b]] <- min(e) / sec
      row[[paste0(b, "_net")]] <- min(e) / secn
    }
    row
  })
  do.call(rbind, out)
}
mi <- function(x) { x <- x[is.finite(x)]
  if (!length(x)) return("-")
  q <- stats::quantile(x, c(0.25, 0.5, 0.75), names = FALSE)
  sprintf("%.2f [%.2f, %.2f]", q[2], q[1], q[3]) }

tables_md <- function(S, title, file, arms = c("A", "R", "C", "J")) {
  lab <- c(A = "jmjax unrotated", R = "jmjax rotated", C = "jmjax rotated, centred W",
           J = "JMbayes2")
  arms <- intersect(arms, unique(S$arm))
  L <- c(sprintf("## %s", title), "",
         "Cells: median [IQR] over replicates. ESS/s = smallest posterior::ess_bulk in the block",
         "divided by wall time (jmjax: warm-up + sampling, JAX compilation included; JMbayes2:",
         "the whole jm() call). Only runs meeting the bar (every reported parameter R-hat < 1.01,",
         "bulk and tail ESS > 400) are included; ratios are paired by replicate and need both",
         "runs to meet it. 'net' = jmjax time minus its compilation estimate.", "")
  for (cf in unique(S$config)) {
    X <- S[S$config == cf, ]
    ok <- X[X$meets_bar, ]
    L <- c(L, sprintf("### %s (q = %d)", cf, X$q[1]), "",
           "| arm | runs meeting bar | draws/chain | sec | compile sec | beta ESS/s | alpha ESS/s | Sigma ESS/s | divergent draws (runs with any) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (a in arms) {
      Y <- X[X$arm == a, ]; if (!nrow(Y)) next
      Z <- ok[ok$arm == a, ]
      L <- c(L, sprintf("| %s | %d / %d | %s | %s | %s | %s | %s | %s | %s |", lab[a],
                        nrow(Z), nrow(Y),
                        if (nrow(Z)) paste(sort(unique(Z$draws_per_chain)), collapse = "/") else "-",
                        mi(Z$sec), if (a == "J") "-" else mi(Z$sec_compile),
                        mi(Z$beta), mi(Z$alpha), mi(Z$Sigma),
                        if (a == "J") "-" else sprintf("%d (%d)", as.integer(sum(Y$n_div, na.rm = TRUE)),
                                                       sum(Y$n_div > 0, na.rm = TRUE))))
    }
    pr <- list(c("R", "A"), c("C", "A"), c("C", "R"), c("R", "J"), c("C", "J"), c("A", "J"))
    rr <- character(0)
    for (p in pr) {
      if (!all(p %in% ok$arm)) next
      m <- merge(ok[ok$arm == p[1], ], ok[ok$arm == p[2], ], by = "seed")
      if (!nrow(m)) next
      cell <- function(v) mi(m[[paste0(v, ".x")]] / m[[paste0(v, ".y")]])
      rr <- c(rr, sprintf("| %s/%s | %d | %s | %s | %s | %s | %s | %s |", p[1], p[2], nrow(m),
                          cell("beta"), cell("alpha"), cell("Sigma"),
                          cell("beta_net"), cell("alpha_net"), cell("Sigma_net")))
    }
    if (length(rr)) {
      L <- c(L, "", "Paired ratios of smallest ESS/s, median [IQR]:", "",
             "| ratio | pairs | beta | alpha | Sigma | beta (net) | alpha (net) | Sigma (net) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|", rr)
    }
    L <- c(L, "")
  }
  writeLines(L, file); cat(L, sep = "\n")
  invisible(L)
}
