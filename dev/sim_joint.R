# ==============================================================================
# A joint-model data generator with BASELINE COVARIATES, for studies that need
# to vary n and k independently.
#
# WHY NOT tests/testthat/helper-simulate.R. That generator has no covariates
# at all - the longitudinal model is y ~ time - so its design has exactly one
# subject-constant column, the intercept, and k = 1 always. It also has a
# random INTERCEPT only, while orthogonalize_b0 requires q >= 2. Neither is a
# defect for its purpose (unit tests), but both make it unusable for the
# k-scaling question, and changing it would touch the test suite.
#
# WHAT k MEANS HERE. k counts the SUBJECT-CONSTANT columns of the longitudinal
# design matrix, intercept included. y ~ time is k = 1; y ~ time + age is
# k = 2; y ~ time + age + sex is k = 3. Covariates are baseline (constant
# within subject), which is what makes them part of the location degeneracy -
# a time-varying covariate would not be.
#
# k_extra IS NOT CAPPED AT 3. age/sex/trt are the first three (named, kept
# for continuity with every study that already used them); anything past
# that is x4, x5, ... - standardized N(0,1) covariates with a fixed,
# decaying, alternating-sign coefficient (see EXTRA COVARIATES below) - so a
# k-sweep study can go past k = 4 to characterize whether jmjax's advantage
# over JMbayes2 grows with the number of adjustment covariates, which is the
# practical "many factors to adjust for" scenario, not just sample size.
#
# REALISM CHOICES, stated so they can be argued with:
#   - age ~ N(0, 10), CENTRED. An earlier version drew age ~ N(50, 10) on
#     purpose, to let a centred generator not hide the interaction between
#     uncentred covariates and standardize_covariates. In practice this
#     bought little - users are told to centre their own covariates before
#     fitting either package - and it confounded two different effects for
#     any study that varies covariate COUNT: more subject-constant columns
#     (bigger degenerate subspace for orthogonalize_b/b0 to remove) versus
#     one column happening to carry a large uncentred mean (its own,
#     unrelated conditioning problem for a standard sampler's mass matrix).
#     Centring removes the second effect so a k-sweep study isolates the
#     first. A separate, explicitly-uncentred cell is the right way to
#     study the scale-conditioning question if it is ever needed again.
#   - sex, trt binary at 0.5, the usual trial split.
#   - random intercept AND slope, correlated, because q = 2 is the realistic
#     joint-model case and the option requires it.
#   - sigma_b0 / sigma_e = 0.8 / 0.3 = 2.7, matching both the existing helper
#     and pbc2 (about 2.9), so the random intercept is neither unusually
#     strong nor unusually weak.
#   - Weibull baseline with shape != 1, so the fitted spline baseline has
#     something non-trivial to recover. The cumulative hazard is computed on
#     a grid rather than in closed form for exactly this reason; an
#     exponential baseline would be closed-form but unrealistically flat.
#   - visit times are regular with dropout at the event, which is how
#     longitudinal trial data actually arrives.
#
# The achieved event rate is RETURNED, not assumed, so any study using this
# can report it rather than claiming a target it did not hit.
# ==============================================================================

sim_joint <- function(n = 500, seed = 1L, k_extra = 2L,
                      beta0 = 2.0, beta1 = 0.5,
                      beta_cov = NULL,
                      sigma_b0 = 0.8, sigma_b1 = 0.3, rho = 0.3,
                      sigma_e = 0.3,
                      alpha = 0.5, gamma_trt = -0.3,
                      weibull_shape = 1.3, log_lambda0 = NULL,
                      target_event_rate = 0.5,
                      max_time = 10, visit_gap = 1.0) {
  set.seed(seed)
  k_extra <- as.integer(k_extra)
  stopifnot(k_extra >= 0L)

  # ---- baseline covariates ------------------------------------------------
  # age/sex/trt: the original three, kept for continuity with every existing
  # study/cache that names them explicitly.
  age <- stats::rnorm(n, 0, 10)
  sex <- stats::rbinom(n, 1, 0.5)
  trt <- stats::rbinom(n, 1, 0.5)
  named    <- cbind(age = age, sex = sex, trt = trt)
  n_named  <- min(k_extra, 3L)
  n_extra  <- max(k_extra - 3L, 0L)

  # EXTRA COVARIATES (x4, x5, ...), for k_extra > 3. Standardized N(0,1) -
  # deliberately, so a k-sweep study is not reintroducing the same scale-
  # conditioning confound that moving age off N(50,10) was meant to remove
  # (see the WHAT k MEANS HERE note above). The coefficient pattern is fixed
  # and decaying/alternating (0.25, -0.1875, 0.1406, ...) rather than drawn
  # at random, so a given k_extra/seed reproduces exactly and adding more
  # covariates to a sweep does not change the effect of the ones already
  # there - a realistic shape too, with most of the adjustment signal
  # concentrated in the first few covariates and a shrinking tail after.
  extra <- if (n_extra > 0L) {
    m <- matrix(stats::rnorm(n * n_extra), nrow = n, ncol = n_extra)
    colnames(m) <- paste0("x", 3L + seq_len(n_extra))
    m
  } else matrix(numeric(0), nrow = n, ncol = 0L)
  extra_beta <- if (n_extra > 0L) 0.25 * (-0.75)^(seq_len(n_extra) - 1L) else numeric(0)

  Zc <- cbind(named[, seq_len(n_named), drop = FALSE], extra)
  bc <- if (is.null(beta_cov)) c(c(0.02, -0.30, 0.15)[seq_len(n_named)], extra_beta)
        else beta_cov[seq_len(k_extra)]
  # subject-level covariate contribution to the longitudinal mean
  cov_eff <- if (k_extra > 0L) as.numeric(Zc %*% bc) else rep(0, n)

  # ---- correlated random intercept and slope ------------------------------
  L <- matrix(c(sigma_b0, 0, rho * sigma_b1, sigma_b1 * sqrt(1 - rho^2)),
              nrow = 2, byrow = TRUE)
  b <- matrix(stats::rnorm(2 * n), ncol = 2) %*% t(L)
  b0 <- b[, 1]; b1 <- b[, 2]

  # ---- event times by inverting the cumulative hazard on a grid -----------
  # h_i(t) = shape * exp(log_lambda0) * t^(shape-1)
  #          * exp(gamma*trt_i + alpha * m_i(t)),
  # m_i(t) = beta0 + cov_eff_i + b0_i + (beta1 + b1_i) * t.
  # Vectorised over subjects: no per-subject integrate()/uniroot(), so
  # n = 1200 costs the same as n = 200.
  tg  <- seq(0, max_time * 1.5, length.out = 600)
  dt  <- diff(tg)
  # Baseline computed at log_lambda0 = 0, so the whole cumulative hazard
  # scales as exp(log_lambda0) and the constant can be SOLVED for rather
  # than guessed (see below).
  h0u <- weibull_shape * pmax(tg, 1e-8)^(weibull_shape - 1)
  ci  <- gamma_trt * trt + alpha * (beta0 + cov_eff + b0)   # [n]
  di  <- alpha * (beta1 + b1)                               # [n]
  lin <- outer(ci, rep(1, length(tg))) + outer(di, tg)
  hz  <- sweep(exp(lin), 2, h0u, "*")
  H0  <- cbind(0, t(apply((hz[, -1, drop = FALSE] + hz[, -ncol(hz), drop = FALSE]) / 2, 1,
                          function(r) cumsum(r * dt))))
  target <- -log(stats::runif(n))

  # ---- calibrate the baseline constant to hit the target event rate -------
  # A GUESSED log_lambda0 does not survive a change to any other parameter.
  # The first version of this generator used -3.4 and produced a 97.8% event
  # rate at k=3, because uncentred age adds about 1.0 to the linear predictor
  # and alpha multiplies that inside the hazard - so subjects died almost
  # immediately and averaged 3.6 longitudinal observations each, which is not
  # a joint-model dataset in any useful sense.
  #
  # Since H_i(t) = exp(L) * H0_i(t), subject i has an event by max_time iff
  #     exp(L) * H0_i(max_time) >= target_i,  i.e.  L >= log(target_i / H0_i).
  # The event rate is therefore the fraction of subjects whose threshold lies
  # below L, so the L achieving any target rate is just the corresponding
  # QUANTILE of those thresholds. Exact, one line, no search - and it retunes
  # itself automatically whenever n, k, alpha or the covariate effects change.
  g_end <- which.min(abs(tg - max_time))
  thr   <- log(target) - log(pmax(H0[, g_end], 1e-12))
  if (is.null(log_lambda0)) {
    log_lambda0 <- unname(stats::quantile(thr, probs = target_event_rate,
                                          names = FALSE, type = 7))
  }
  H <- exp(log_lambda0) * H0
  T_true <- vapply(seq_len(n), function(i) {
    g <- which(H[i, ] >= target[i])[1]
    if (is.na(g) || g <= 1L) return(if (is.na(g)) Inf else tg[1])
    # linear interpolation between the bracketing grid points
    w <- (target[i] - H[i, g - 1L]) / max(H[i, g] - H[i, g - 1L], 1e-12)
    tg[g - 1L] + w * (tg[g] - tg[g - 1L])
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event    <- as.integer(T_true <= max_time)

  # ---- longitudinal observations up to the event/censoring ---------------
  visits <- seq(0, max_time, by = visit_gap)
  reps   <- vapply(obs_time, function(o) max(1L, sum(visits <= o)), integer(1))
  id_l   <- rep(seq_len(n), reps)
  t_l    <- unlist(lapply(reps, function(r) visits[seq_len(r)]), use.names = FALSE)
  y <- beta0 + cov_eff[id_l] + b0[id_l] + (beta1 + b1[id_l]) * t_l +
       stats::rnorm(length(t_l), 0, sigma_e)

  # NOTE: cbind(data.frame(...), NULL) is NOT a no-op - it still counts NULL
  # as a 0-row argument and errors ("differing number of rows: k, 0") once
  # any other argument is non-empty. So the extra columns are only cbind-ed
  # in when they actually exist, rather than always cbind-ing a possibly-
  # NULL frame - caught by k_extra = 0 failing this exact way.
  data_long <- data.frame(id = id_l, time = t_l, y = y,
                          age = age[id_l], sex = sex[id_l], trt = trt[id_l])
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event,
                          age = age, sex = sex, trt = trt)
  if (n_extra > 0L) {
    data_long <- cbind(data_long, as.data.frame(extra[id_l, , drop = FALSE]))
    data_surv <- cbind(data_surv, as.data.frame(extra))
  }

  list(data_long = data_long, data_surv = data_surv,
       event_rate = mean(event), n_obs = nrow(data_long),
       obs_per_subject = nrow(data_long) / n,
       log_lambda0 = log_lambda0,
       truth = list(beta0 = beta0, beta1 = beta1, log_lambda0 = log_lambda0,
                    beta_cov = if (k_extra > 0L) bc else numeric(0),
                    sigma_b0 = sigma_b0, sigma_b1 = sigma_b1, rho = rho,
                    sigma_e = sigma_e, alpha = alpha, gamma_trt = gamma_trt),
       k = 1L + k_extra)
}
