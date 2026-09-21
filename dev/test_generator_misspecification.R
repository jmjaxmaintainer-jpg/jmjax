# ==============================================================================
# Is the sampling difficulty ours, or the simulation's?
#
# WHAT PROMPTED THIS. Every n = 8,000 remedy script generates data with
#
#     eta <- -2 + .4*(2 + .3*x + b0)
#     Tt  <- rexp(n, rate = pmin(pmax(exp(eta), 1e-4), 5) / 4)
#
# That linear predictor is evaluated at t = 0 and never again. The hazard
# is CONSTANT in time and the random slope b1 does not enter it at all.
# b0 and b1 are drawn independently, so the true rho is 0.
#
# The package's own test helper refuses exactly this shortcut:
#
#   "Deliberately NOT a simplified/closed-form simulation... A simpler
#    simulation (e.g. alpha*b_i-only, which has a closed-form Weibull
#    inverse) would risk tests passing even if the time-dependent term
#    were implemented incorrectly."
#
# So the test suite inverts the real time-dependent hazard and the
# benchmark scripts use the version the test suite rejects.
#
# WHY IT MIGHT MATTER. jm_fit is asked for a hazard depending on
# m_i(t) = (beta0+b0) + (beta1+b1)t. In the data that term does not exist,
# so the likelihood has to drive its contribution to zero - and it can do
# that through alpha, beta1, sigma_b1 or rho, in combination. That is a
# ridge, and a ridge is what NUTS cannot fit a single step size to. The
# seed-7 failure was 543 leapfrog steps with the disagreement concentrated
# on rho, which is what a ridge looks like from the outside.
#
# WHAT THIS RUN CANNOT DO, and why it is still worth running. It cannot
# reproduce the seed-7 failure: that appeared only at n = 8,000, and at
# n <= 2,000 every one of the 30 fits in the previous sweep sat at exactly
# 63 leapfrog steps. Step count is quantised (63, 127, 255, 543) and has
# no variance to explain at this size.
#
# But a ridge is present at every n - it only becomes sharp enough to
# break adaptation when the posterior concentrates. At n = 1,000 it should
# still be visible in two CONTINUOUS measures:
#
#   1. POSTERIOR CORRELATION between alpha and the slope-related
#      parameters. This is the ridge measured directly, not inferred from
#      a symptom. If arm A shows a strong alpha-beta1 correlation that arm
#      B does not, the misspecification is manufacturing the geometry.
#   2. ESS, which a ridge costs even when adaptation succeeds.
#
# THE ARMS. Everything is held fixed except the hazard and rho:
#
#   A  old       hazard constant in t (rexp), b1 absent from it, rho = 0
#   B  correct   hazard depends on m_i(t), by exact inversion, rho = 0
#   C  correct   the same, with a genuine rho = 0.3
#
# A vs B isolates the misspecification. B vs C isolates rho's true value.
# The baseline hazard stays constant in all three, so the 9-coefficient
# spline is equally over-parameterised everywhere and cannot confound the
# A-vs-B contrast. Event rates are calibrated to match.
#
# ONE THING THAT CANNOT BE MATCHED, and should be read with the results.
# At the same marginal event rate, arm B's hazard is low early and high
# late, so its events fall later and its subjects reach more visits:
# roughly 6.2 observations per subject against arm A's 5.6, about 10% more
# longitudinal data. That is a consequence of a time-varying hazard, not a
# design flaw, but it mildly favours B. Row counts are printed per arm so
# the difference stays visible; 10% more data should not move a posterior
# correlation much, and if the A-vs-B gap is small it is the first thing
# to suspect.
#
# HOW TO READ IT
#   If A and B are indistinguishable, the misspecification is not the
#   driver, the n = 8,000 failure is a genuine sampler limitation, and the
#   expensive arms (target_accept_prob = 0.9) are worth running.
#   If A is measurably worse, several rounds of sampler tuning have been
#   chasing an artifact of the simulation, and the honest next step is to
#   re-do the n = 8,000 diagnosis on a correct generator before changing
#   any default.
#
# RUNTIME: 30 fits at n = 1,000, roughly 35-45 minutes. Set N <- 2000L for
# the larger size at roughly twice that.
#
# Results are written to RDS and CSV next to this script - previous runs
# left their numbers only in the console, and those are now unrecoverable.
# ==============================================================================

library(nlme); library(survival); library(jmjax)

`%||%` <- function(a, b) {
  if (is.null(a)) return(b)
  if (length(a) == 1L && is.na(a)) return(b)
  a
}

N          <- 1000L        # 1000L or 2000L
DATA_SEEDS <- c(1L, 2L)    # separates "this dataset is hard" from "this arm is hard"
SAMP_SEEDS <- c(1L, 7L, 13L, 42L, 99L)
OUT        <- file.path(getwd(), sprintf("generator_misspec_n%d", N))

# true generating values, shared by all three arms
BETA0 <- 2.0; BETA1 <- 0.5; BX <- 0.3
SD_B0 <- 0.8; SD_B1 <- 0.2; SIGMA_E <- 0.3
ALPHA <- 0.4
TMAX  <- 10; NVISIT <- 8L
TARGET_EVENT <- 0.53       # what arm A produces, so the arms are comparable

VISIT <- seq(0, TMAX, length.out = NVISIT)

# ------------------------------------------------------------------ helpers

# Assemble the longitudinal frame given per-subject effects and follow-up.
# Identical across arms, so any difference between them is the hazard.
.assemble <- function(b0, b1, x, ot, ev) {
  n <- length(b0)
  reps <- vapply(ot, function(o) max(1L, sum(VISIT <= o)), integer(1))
  dl <- data.frame(
    id   = factor(rep(seq_len(n), reps), levels = seq_len(n)),
    time = unlist(lapply(reps, function(k) VISIT[seq_len(k)])),
    x    = rep(x, reps))
  dl$y <- (BETA0 + BX*dl$x + rep(b0, reps)) +
          (BETA1 + rep(b1, reps))*dl$time +
          rnorm(nrow(dl), 0, SIGMA_E)
  ds <- data.frame(id = factor(seq_len(n)), time = ot,
                   event = as.integer(ev), x = x)
  list(l = dl, s = ds, ev = mean(ev))
}

# Draw (b0, b1) with a specified correlation. rho = 0 reproduces the
# independent draw the old generator used; the marginal SDs are unchanged
# either way, so only the correlation differs between arms B and C.
.effects <- function(n, rho) {
  b0 <- rnorm(n, 0, SD_B0)
  z  <- rnorm(n)
  b1 <- rho * (SD_B1/SD_B0) * b0 + sqrt(1 - rho^2) * SD_B1 * z
  list(b0 = b0, b1 = b1, x = rnorm(n))
}

# ---- arm A: the generator every remedy script used, verbatim -------------
sim_old <- function(n, seed) {
  set.seed(seed)
  b0 <- rnorm(n, 0, .8); b1 <- rnorm(n, 0, .2); x <- rnorm(n)
  eta <- -2 + .4*(2 + .3*x + b0)
  Tt <- rexp(n, rate = pmin(pmax(exp(eta), 1e-4), 5) / 4)
  ot <- pmin(Tt, TMAX)
  .assemble(b0, b1, x, ot, Tt <= TMAX)
}

# ---- arms B and C: the hazard jm_fit actually fits ------------------------
#
# h_i(t) = lambda0 * exp(alpha * m_i(t)),  m_i(t) = A_i + D_i t
#   A_i = beta0 + bx*x_i + b0_i      D_i = beta1 + b1_i
#
# H_i(t) = lambda0 exp(alpha A_i) * (exp(alpha D_i t) - 1) / (alpha D_i)
#
# so T solves H_i(T) = E, E ~ Exp(1), in closed form - no uniroot needed,
# and exact rather than tolerance-limited. Checked against integrate()
# below before anything is fitted.
.invert <- function(E, A, D, lam0) {
  aD <- ALPHA * D
  base <- lam0 * exp(ALPHA * A)
  out <- numeric(length(E))
  lin <- abs(aD) < 1e-8                       # D ~ 0: hazard constant, H = base*t
  out[lin] <- E[lin] / base[lin]
  g <- !lin
  arg <- 1 + E[g] * aD[g] / base[g]
  # arg <= 0 means H_i(inf) < E: a decaying hazard this subject outruns.
  # Those are genuine non-events, not a numerical failure.
  out[g] <- ifelse(arg > 0, log(arg) / aD[g], Inf)
  out
}

sim_correct <- function(n, seed, rho) {
  set.seed(seed)
  e <- .effects(n, rho)
  A <- BETA0 + BX*e$x + e$b0
  D <- BETA1 + e$b1
  E <- rexp(n)

  # Calibrate lambda0 so the event rate matches arm A's. Without this the
  # arms would differ in censoring as well as in the hazard, and the
  # comparison would not be clean.
  rate_at <- function(log_lam0) mean(.invert(E, A, D, exp(log_lam0)) <= TMAX)
  lo <- -20; hi <- 5
  log_lam0 <- tryCatch(
    stats::uniroot(function(z) rate_at(z) - TARGET_EVENT,
                   lower = lo, upper = hi, tol = 1e-6)$root,
    error = function(err) { warning("event-rate calibration failed; using -6"); -6 })

  Tt <- .invert(E, A, D, exp(log_lam0))
  ot <- pmin(Tt, TMAX)
  d <- .assemble(e$b0, e$b1, e$x, ot, Tt <= TMAX)
  d$log_lam0 <- log_lam0
  d
}

# ------------------------------------------------- verify the inversion
# The closed form is the one piece of new mathematics here. A wrong
# constant would silently produce a different model and invalidate the
# whole comparison, so it is checked against numerical integration of the
# same hazard before any fitting happens.
cat("================ checking the closed-form inversion ================\n")
set.seed(99)
{
  lam0 <- exp(-6); ok <- TRUE
  for (i in 1:5) {
    A <- rnorm(1, 2, .8); D <- rnorm(1, .5, .2); t0 <- runif(1, .5, 9)
    num <- stats::integrate(function(s) lam0*exp(ALPHA*(A + D*s)),
                            lower = 0, upper = t0)$value
    ana <- lam0*exp(ALPHA*A)*(exp(ALPHA*D*t0) - 1)/(ALPHA*D)
    back <- .invert(num, A, D, lam0)          # should return t0
    cat(sprintf("  A %6.3f D %6.3f t %5.2f | H numeric %10.6f analytic %10.6f | inverted t %6.3f\n",
                A, D, t0, num, ana, back))
    if (abs(ana - num) > 1e-6 * max(1, abs(num)) || abs(back - t0) > 1e-6) ok <- FALSE
  }
  if (!ok) stop("closed-form cumulative hazard does not match numerical integration - stopping")
  cat("  inversion verified.\n\n")
}

# ---------------------------------------------------------------- fitting

ARMS <- list(
  A = list(lbl = "A old (hazard flat in t, rho=0)",  gen = function(s) sim_old(N, s)),
  B = list(lbl = "B correct h(t), rho=0",            gen = function(s) sim_correct(N, s, 0.0)),
  C = list(lbl = "C correct h(t), rho=0.3",          gen = function(s) sim_correct(N, s, 0.3))
)

# Pull a named scalar out of a diagnostics list without assuming the name.
.pick <- function(lst, candidates) {
  if (is.null(lst)) return(NA_real_)
  for (nm in candidates) {
    if (!is.null(lst[[nm]])) {
      v <- suppressWarnings(as.numeric(lst[[nm]]))
      if (length(v) && is.finite(v[1])) return(v[1])
    }
  }
  NA_real_
}

# The ridge, measured directly: the largest absolute posterior correlation
# between alpha and any other scalar population parameter. A misspecified
# time-varying term should show up here as alpha trading off against the
# slope parameters.
.alpha_ridge <- function(ps) {
  empty <- list(max = NA_real_, with = NA_character_)
  if (is.null(ps) || is.null(ps[["alpha"]])) return(empty)
  a  <- suppressWarnings(as.numeric(unlist(ps[["alpha"]])))
  nd <- length(a)
  if (nd < 20 || !all(is.finite(a)) || stats::sd(a) == 0) return(empty)

  series <- list()
  for (nm in names(ps)) {
    if (identical(nm, "alpha")) next
    v <- suppressWarnings(as.numeric(unlist(ps[[nm]])))
    if (!length(v) || length(v) %% nd != 0) next
    k <- length(v) / nd
    # Skip the random effects: k would be n*q, thousands of columns, none
    # of them a population parameter. Vector sites like beta and the
    # spline block have k <= 32 and ARE kept - splitting them into columns
    # is the whole point, since beta1 is the parameter alpha is most
    # likely to trade off against.
    if (k > 32) next
    m <- matrix(v, nrow = nd, byrow = TRUE)   # numpy (n_draws, k) is row-major
    for (j in seq_len(k)) {
      col <- m[, j]
      if (all(is.finite(col)) && stats::sd(col) > 0) {
        series[[if (k == 1) nm else sprintf("%s[%d]", nm, j)]] <- col
      }
    }
  }
  if (!length(series)) return(empty)
  cr <- vapply(series, function(v) abs(stats::cor(a, v)), numeric(1))
  list(max = unname(max(cr)), with = names(which.max(cr)))
}

one_fit <- function(d, sseed, announce = FALSE) {
  t <- system.time(f <- tryCatch(
    jm_fit(long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
           data_long = d$l, data_surv = d$s, id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(n_interior_knots = 5L, spline_prior = "penalized",
                          num_warmup = 500L, num_samples = 500L,
                          num_chains = 2L, seed = sseed, progress_bar = FALSE)),
    error = function(e) { cat("    fit failed:", conditionMessage(e), "\n"); NULL }))
  if (is.null(f)) return(NULL)

  if (announce) {
    cat("  rhat sites      : ", paste(utils::head(names(f$diagnostics$rhat), 20),
                                       collapse = ", "), "\n")
    cat("  posterior sites : ", paste(utils::head(names(f$posterior_samples), 20),
                                       collapse = ", "), "\n\n")
  }

  rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
  es <- f$diagnostics$ess
  rg <- .alpha_ridge(f$posterior_samples)

  list(
    time      = as.numeric(t["elapsed"]),
    rhat_max  = if (length(rh)) max(rh) else NA_real_,
    rhat_rho  = .pick(f$diagnostics$rhat, c("rho", "rho_b", "rho_b01")),
    steps     = suppressWarnings(as.numeric(f$convergence$mean_num_steps)),
    div       = { v <- f$convergence$n_divergences; if (is.null(v)) NA_real_ else as.numeric(v) },
    ess_alpha = .pick(es, "alpha"),
    ess_rho   = .pick(es, c("rho", "rho_b", "rho_b01")),
    ess_min   = { e <- unlist(es); e <- e[is.finite(e)]; if (length(e)) min(e) else NA_real_ },
    ridge     = rg$max,
    ridge_with= rg$with,
    est_alpha = suppressWarnings(as.numeric(f$estimates[["alpha"]])),
    est_rho   = .pick(f$estimates, c("rho", "rho_b", "rho_b01"))
  )
}

fmt <- function(v, d = 3) if (is.finite(v)) formatC(v, format = "f", digits = d) else "?"

rows <- list(); first <- TRUE
for (arm in names(ARMS)) {
  cat(sprintf("\n################ %s ################\n", ARMS[[arm]]$lbl))
  for (dseed in DATA_SEEDS) {
    d <- ARMS[[arm]]$gen(dseed)
    cat(sprintf("  data seed %d | n = %d | rows = %d | events %.0f%%%s\n",
                dseed, N, nrow(d$l), 100*d$ev,
                if (!is.null(d$log_lam0)) sprintf(" | log lambda0 %.3f", d$log_lam0) else ""))
    cat(sprintf("  %6s %8s %8s %8s %7s %8s %8s %8s %-22s\n",
                "samp", "time", "R-hat", "R-hat_rho", "steps", "ESS_a", "ESS_rho",
                "|cor|a", "ridge partner"))
    for (ss in SAMP_SEEDS) {
      r <- one_fit(d, ss, announce = first); first <- FALSE
      if (is.null(r)) next
      rows[[length(rows)+1]] <- data.frame(
        arm = arm, label = ARMS[[arm]]$lbl, n = N, data_seed = dseed, samp_seed = ss,
        time = r$time, rhat_max = r$rhat_max, rhat_rho = r$rhat_rho,
        steps = r$steps, divergences = r$div,
        ess_alpha = r$ess_alpha, ess_rho = r$ess_rho, ess_min = r$ess_min,
        alpha_ridge = r$ridge, ridge_with = r$ridge_with %||% NA_character_,
        est_alpha = r$est_alpha, est_rho = r$est_rho,
        stringsAsFactors = FALSE)
      cat(sprintf("  %6d %8.1f %8s %8s %7s %8s %8s %8s %-22s\n",
                  ss, r$time, fmt(r$rhat_max), fmt(r$rhat_rho),
                  fmt(r$steps, 0), fmt(r$ess_alpha, 0), fmt(r$ess_rho, 0),
                  fmt(r$ridge), r$ridge_with %||% "?"))
    }
  }
}

`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1 && is.na(a))) b else a
res <- do.call(rbind, rows)

saveRDS(res, paste0(OUT, ".rds"))
utils::write.csv(res, paste0(OUT, ".csv"), row.names = FALSE)
cat(sprintf("\nresults written to %s.rds and %s.csv\n", OUT, OUT))

# ---------------------------------------------------------------- summary
cat("\n\n================ SUMMARY ================\n")
agg <- function(f, nm) {
  v <- vapply(split(res[[nm]], res$arm), function(z) f(z[is.finite(z)]), numeric(1))
  v[c("A","B","C")]
}
line <- function(nm, lab, d = 3) {
  m <- agg(stats::median, nm)
  cat(sprintf("  %-28s A %10s | B %10s | C %10s\n", lab,
              fmt(m[["A"]], d), fmt(m[["B"]], d), fmt(m[["C"]], d)))
}
cat("  (medians across 10 fits per arm)\n\n")
line("alpha_ridge", "max |cor| with alpha")
line("ess_alpha",   "ESS alpha", 0)
line("ess_rho",     "ESS rho", 0)
line("ess_min",     "ESS min", 0)
line("rhat_max",    "max R-hat")
line("rhat_rho",    "R-hat rho")
line("steps",       "leapfrog steps", 0)
line("time",        "seconds", 1)

cat("\n  most frequent ridge partner, by arm:\n")
for (a in c("A","B","C")) {
  w <- res$ridge_with[res$arm == a]; w <- w[!is.na(w)]
  if (length(w)) {
    tb <- sort(table(w), decreasing = TRUE)
    cat(sprintf("    %s : %s\n", a,
                paste(sprintf("%s (%d)", names(tb), as.integer(tb)),
                      collapse = ", ")))
  }
}

cat("\n================ HOW TO READ IT ================\n")
cat("THE PRIMARY NUMBER is 'max |cor| with alpha'. It measures the ridge\n")
cat("directly rather than inferring it from a symptom, and unlike the step\n")
cat("count it is not quantised, so it can discriminate at n = 1,000.\n\n")
cat("A MUCH HIGHER in arm A, especially paired with beta1 or a slope\n")
cat("  parameter: the misspecification is manufacturing the geometry. The\n")
cat("  n = 8,000 diagnosis should be redone on arm B before any sampler\n")
cat("  default is changed on its evidence.\n")
cat("A AND B ALIKE: the simulation is not the driver. The n = 8,000\n")
cat("  failure is a real sampler limitation and target_accept_prob = 0.9\n")
cat("  is worth the expensive run.\n\n")
cat("B vs C separates rho's TRUE VALUE from the hazard. If C is healthier\n")
cat("than B, rho = 0 was itself part of the difficulty and every previous\n")
cat("R-hat-on-rho reading was taken at the least favourable point.\n\n")
cat("TRUTH RECOVERY, available in B and C only: est_alpha should sit near\n")
cat(sprintf("%.2f and est_rho near 0.0 (B) / 0.3 (C). Arm A has no true alpha -\n", ALPHA))
cat("its hazard has no time-varying term for alpha to be the coefficient\n")
cat("of - which is itself the point.\n")
