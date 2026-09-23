# ==============================================================================
# dev/pilot_q1_intercept.R - does a random-intercept-only (q = 1) fit have
# the intercept-level location degeneracy badly enough to be worth rotating?
#
# The rotation (control$rotate_absorbable) is currently q >= 2 only. For
# q = 1 the degeneracy between beta_0 (and every subject-constant
# covariate) and the matching directions of b is the same one vignette
# Proposition 7 describes for the intercept column - that result does not
# involve q - so the theory predicts the same slow mixing. This pilot
# measures it before any code is written.
#
# For aids and pbc2 with random = ~ 1 | id (otherwise the Section 7
# designs), 2 seeds each, it records per population parameter:
#   - ESS per draw and ESS/sec (4 chains x 1000 + 1000);
#   - R^2 of the parameter on Q'b, the projection of the random effects on
#     the absorbable directions (Q = orthonormal basis of [1, subject-
#     constant covariates]). The ridge's signature is R^2 near 1 for
#     beta_0 and the subject-constant coefficients, and near 0 for the rest.
# and prints the q = 2 default fit's ESS per draw on the same coefficients
# from dev/study_realdata_rotate.R's results, for scale.
#
# DECISION RULE (fixed before the run): extend the rotation to q = 1 if
# beta_0 or a subject-constant coefficient has ESS per draw below 0.2 with
# R^2 above 0.9 on both datasets - i.e. the same picture as q = 2's default
# fit (0.03-0.06 on those coefficients). If they are already mixing well
# (ESS per draw above about 0.5), the extension is not worth it.
#
#   caffeinate -i Rscript dev/pilot_q1_intercept.R      # ~5-10 min
# ==============================================================================

suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
if (!requireNamespace("JMbayes2", quietly = TRUE))
  stop("JMbayes2 is required (it provides the aids and pbc2 data).", call. = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a
CHAINS <- 4L; WARMUP <- 1000L; SAMPLES <- 1000L
SEEDS  <- seq_len(as.integer(Sys.getenv("Q1_SEEDS", "2")))
OUTDIR <- Sys.getenv("REAL_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
CSV  <- file.path(OUTDIR, "pilot_q1_intercept.csv")
Q2CSV <- file.path(OUTDIR, "study_realdata_rotate.csv")

load_data <- function(ds) {
  if (ds == "aids") {
    utils::data("aids", package = "JMbayes2", envir = environment())
    utils::data("aids.id", package = "JMbayes2", envir = environment())
    dl <- aids[, c("patient", "obstime", "CD4", "drug", "gender")]
    dl$y <- sqrt(dl$CD4); names(dl)[1] <- "id"
    dsv <- aids.id[, c("patient", "Time", "death", "drug")]
    names(dsv) <- c("id", "stime", "event", "drug")
    list(dl = dl, ds = dsv, time_var = "obstime", lform = y ~ obstime + drug + gender)
  } else {
    utils::data("pbc2", package = "JMbayes2", envir = environment())
    utils::data("pbc2.id", package = "JMbayes2", envir = environment())
    dl <- pbc2[, c("id", "year", "serBilir", "age")]
    dl$id <- as.integer(as.character(dl$id)); dl$y <- log(dl$serBilir)
    s <- pbc2.id
    dsv <- data.frame(id = as.integer(as.character(s$id)), stime = s$years,
                      event = as.integer(s$status != "alive"), drug = s$drug)
    list(dl = dl, ds = dsv, time_var = "year", lform = y ~ year + age)
  }
}

flat <- function(v) {                     # list-of-draws site -> draws x dim matrix
  if (is.null(v)) return(NULL)
  if (!is.list(v)) { m <- as.matrix(v); storage.mode(m) <- "double"; return(m) }
  k <- length(unlist(v[[1]]))
  matrix(vapply(v, function(z) as.numeric(unlist(z)), numeric(k)), ncol = k, byrow = TRUE)
}
r2 <- function(y, X) {
  f <- stats::lm.fit(cbind(1, X), y)
  1 - sum(f$residuals^2) / sum((y - mean(y))^2)
}

rows <- list()
for (ds in c("aids", "pbc2")) {
  d <- load_data(ds)
  d$dl <- d$dl[order(d$dl$id, d$dl[[d$time_var]]), ]
  lme_fit <- lme(d$lform, random = ~ 1 | id, data = d$dl,
                 control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
  cox_fit <- coxph(Surv(stime, event) ~ drug, data = d$ds)

  # absorbable basis: [1, subject-constant columns of the fixed-effects design]
  X   <- stats::model.matrix(d$lform, d$dl)
  ids <- d$dl$id
  const <- vapply(seq_len(ncol(X)), function(j)
    all(tapply(X[, j], ids, function(z) length(unique(z)) == 1L)), logical(1))
  first <- !duplicated(ids)
  S <- X[first, const, drop = FALSE]
  Q <- qr.Q(qr(S))
  sc_names <- sprintf("beta_%d", which(const) - 1L)
  cat(sprintf("\n== %s: %d subjects, %.1f visits/subject; subject-constant: %s (%s)\n",
              ds, nrow(S), nrow(d$dl) / nrow(S), paste(colnames(X)[const], collapse = ", "),
              paste(sc_names, collapse = ", ")))

  for (s in SEEDS) {
    f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = d$ds, time_var = d$time_var,
                       method = "spline-PH-mcmc",
                       control = list(n_interior_knots = 5, spline_prior = "penalized",
                                      rw2_implementation = "vectorized", dense_mass_spline = TRUE,
                                      num_warmup = WARMUP, num_samples = SAMPLES,
                                      num_chains = CHAINS, seed = s, progress_bar = FALSE))
    cv <- f$convergence
    if (!is.null(cv$orthogonalize$rotation) && isTRUE(cv$orthogonalize$rotation$applied))
      stop("q = 1 fit reports a rotation - this pilot expects none", call. = FALSE)
    ps <- f$posterior_samples
    B  <- flat(ps$b)
    if (is.null(B) || ncol(B) != nrow(S))
      stop(sprintf("posterior b has %s columns, expected %d subjects",
                   if (is.null(B)) "no" else ncol(B), nrow(S)), call. = FALSE)
    # subject order of b: jmjax orders subjects by id (sorted unique ids)
    ord <- match(sort(unique(ids)), ids[first])
    U  <- B %*% Q[ord, , drop = FALSE]              # draws x k absorbable coordinates
    beta <- flat(ps$beta)
    est <- unlist(f$estimates); ess <- unlist(f$diagnostics$ess)
    nm  <- names(est)[grepl("^beta_|^alpha$|^gamma_|^sigma_e$|^sigma_b", names(est))]
    sec <- as.numeric(cv$sampling_time_sec %||% NA_real_)
    for (p in nm) {
      x <- if (grepl("^beta_", p)) beta[, as.integer(sub("beta_", "", p)) + 1L]
           else { v <- flat(ps[[p]]); if (is.null(v)) NULL else v[, 1] }
      rows[[length(rows) + 1L]] <- data.frame(
        dataset = ds, seed = s, param = p, subject_constant = p %in% sc_names,
        ess = as.numeric(ess[[p]]), ess_per_draw = as.numeric(ess[[p]]) / (CHAINS * SAMPLES),
        ess_per_sec = as.numeric(ess[[p]]) / sec,
        r2_absorbable = if (is.null(x)) NA_real_ else r2(x, U),
        sec = sec, mean_num_steps = as.numeric(cv$mean_num_steps %||% NA_real_),
        stringsAsFactors = FALSE)
    }
    cat(sprintf("   seed %d: %.1fs sampling, %.0f leapfrog steps/iteration\n", s, sec,
                as.numeric(cv$mean_num_steps %||% NA)))
  }
}
R <- do.call(rbind, rows)
utils::write.csv(R, CSV, row.names = FALSE)

q2 <- if (file.exists(Q2CSV)) utils::read.csv(Q2CSV, stringsAsFactors = FALSE) else NULL
cat("\nq = 1 default fit (mean over seeds). R^2 = share of the parameter's posterior\n")
cat("variance explained by the absorbable directions of b (ridge signature: near 1).\n")
for (ds in unique(R$dataset)) {
  X <- R[R$dataset == ds, ]
  cat(sprintf("\n%-5s %-10s %6s %10s %9s %8s %16s\n", ds, "param", "subj-c",
              "ESS/draw", "ESS/sec", "R^2", "q=2 A ESS/draw"))
  for (p in unique(X$param)) {
    Y <- X[X$param == p, ]
    q2v <- if (!is.null(q2)) { z <- q2[q2$dataset == ds & q2$arm == "A" & q2$param == p, ]
                               if (nrow(z)) sprintf("%.3f", mean(z$ess / z$n_draws)) else "-" } else "-"
    cat(sprintf("      %-10s %6s %10.3f %9.1f %8.3f %16s\n", p,
                if (Y$subject_constant[1]) "yes" else "", mean(Y$ess_per_draw),
                mean(Y$ess_per_sec), mean(Y$r2_absorbable), q2v))
  }
}
sc <- R[R$subject_constant, ]
worst <- tapply(sc$ess_per_draw, sc$dataset, min)
hi_r2 <- tapply(sc$r2_absorbable, sc$dataset, max)
cat("\nDecision rule: ESS/draw < 0.2 with R^2 > 0.9 on a subject-constant coefficient, both datasets\n")
for (ds in names(worst))
  cat(sprintf("  %-5s worst ESS/draw %.3f, max R^2 %.3f -> %s\n", ds, worst[[ds]], hi_r2[[ds]],
              if (worst[[ds]] < 0.2 && hi_r2[[ds]] > 0.9) "degenerate (rotation would help)"
              else if (worst[[ds]] > 0.5) "mixing well (not worth it)" else "in between"))
cat(sprintf("\nWrote %s\n", CSV))
