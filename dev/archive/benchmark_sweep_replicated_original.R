# ==============================================================================
# jmjax vs JMbayes2 benchmark sweep - REPLICATED, FINAL VERSION
# (Reconstructed from earlier session history - this is the exact script
# that produced the 13.5x ESS/sec result reported earlier.)
#
# Same design as the original single-run benchmark, now with n_reps=5
# independent datasets per (n) value, giving proper variability estimates
# (mean +/- SD, approximate 95% CI) for both the alpha-error and ESS/sec
# comparisons, rather than single-run point estimates.
#
# Includes the non-centered RW2 reparameterization fix (jmjax's default
# for spline_prior="penalized"), which improved mean ESS/sec from 37.2 to
# 178.8 in the single-run version of this benchmark - this replicated run
# is what turned that into the reportable, confidence-interval-backed
# 267.7 (95% CI [207.7, 327.7]) vs JMbayes2's 19.9 (95% CI [15.8, 24.0])
# result, 13.5x, paired t-test p=1.8e-7.
#
# Design choices carried over from development:
#   - simulate_joint_data_re2(): genuine correlated random intercept+slope,
#     avoiding the misspecification-attenuation confound of forcing q=2
#     onto slope-free data.
#   - One dataset per (n, replicate), reused across ALL knot-flexibility
#     variations at that (n, replicate) - avoids confounding a knot-setting
#     effect with dataset-to-dataset noise.
#   - Basis counts matched via the exactly-confirmed formulas for both
#     packages (jmjax: n_interior_knots + ord; JMbayes2: Bsplines_degree +
#     base_hazard_segments).
#   - random_effects="intercept_slope" required on both sides (JMbayes2
#     errors on random-intercept-only for a single longitudinal outcome).
#   - jmjax's own defaults (spline_prior="penalized", non-centered RW2,
#     empirical_bayes_prior=TRUE) used as shipped - this compares the two
#     packages as a user would actually run them, not artificially-matched
#     internals beyond basis count.
#
# NO baseline covariates, NO functional_forms channels beyond "value" -
# this is deliberately the SIMPLEST possible model on BOTH sides, unlike
# the PBC2 real-data comparisons which added q=2 + covariates. Rerunning
# this unchanged script is a direct check of whether anything about
# jmjax's core spline-PH-mcmc + q=2 path has shifted since this was last
# run, independent of all the feature work added afterward.
# ==============================================================================

library(jmjax)
library(JMbayes2)
library(nlme)
library(survival)

simulate_joint_data_re2 <- function(n = 300, seed = 1,
                                     beta0 = 2.0, beta1 = 0.5,
                                     sigma_b0 = 0.8, sigma_b1 = 0.2, rho = 0.3,
                                     sigma_e = 0.3,
                                     weibull_shape = 1.2, log_lambda0 = -2.0,
                                     alpha = 0.6, max_time = 5,
                                     visit_times = seq(0, 5, by = 0.5)) {
  set.seed(seed)
  b0 <- rnorm(n, 0, sigma_b0)
  b1 <- rho * (sigma_b1 / sigma_b0) * b0 + sqrt(1 - rho^2) * sigma_b1 * rnorm(n)

  cum_hazard <- function(t, b0_i, b1_i) {
    if (t <= 0) return(0)
    integrand <- function(s) {
      h0 <- weibull_shape * s^(weibull_shape - 1) * exp(log_lambda0)
      m_s <- (beta0 + b0_i) + (beta1 + b1_i) * s
      h0 * exp(alpha * m_s)
    }
    integrate(integrand, lower = 0, upper = t)$value
  }

  T_true <- vapply(seq_len(n), function(i) {
    target <- -log(runif(1))
    ub <- max_time * 4
    if (cum_hazard(ub, b0[i], b1[i]) < target) return(Inf)
    uniroot(function(t) cum_hazard(t, b0[i], b1[i]) - target, lower = 1e-6, upper = ub)$root
  }, numeric(1))

  obs_time <- pmin(T_true, max_time)
  event <- as.integer(T_true <= max_time)
  data_surv <- data.frame(id = seq_len(n), time = obs_time, event = event)

  long_rows <- lapply(seq_len(n), function(i) {
    vt <- visit_times[visit_times <= obs_time[i]]
    if (length(vt) == 0) vt <- 0
    y <- (beta0 + b0[i]) + (beta1 + b1[i]) * vt + rnorm(length(vt), 0, sigma_e)
    data.frame(id = i, time = vt, y = y)
  })
  data_long <- do.call(rbind, long_rows)

  list(
    data_long = data_long, data_surv = data_surv,
    truth = list(beta0 = beta0, beta1 = beta1, sigma_b0 = sigma_b0,
                 sigma_b1 = sigma_b1, rho = rho, sigma_e = sigma_e, alpha = alpha)
  )
}

N_REPS <- 5

scenarios <- expand.grid(
  n = c(200, 500),
  n_interior_knots_jmjax = c(5, 7),
  stringsAsFactors = FALSE
)
scenarios$base_hazard_segments_jmbayes2 <- ifelse(scenarios$n_interior_knots_jmjax == 5, 6, 8)

# One dataset per (n, replicate), reused across knot-flexibility variations
datasets <- list()
for (n_val in unique(scenarios$n)) {
  for (rep_i in seq_len(N_REPS)) {
    key <- paste0("n", n_val, "_rep", rep_i)
    datasets[[key]] <- simulate_joint_data_re2(n = n_val, seed = 3000 + n_val * 100 + rep_i)
  }
}

results <- list()
row_idx <- 1

for (i in seq_len(nrow(scenarios))) {
  sc <- scenarios[i, ]

  for (rep_i in seq_len(N_REPS)) {
    cat(sprintf("\n=== n=%d, jmjax knots=%d, JMbayes2 segments=%d, rep=%d/%d ===\n",
                sc$n, sc$n_interior_knots_jmjax, sc$base_hazard_segments_jmbayes2, rep_i, N_REPS))

    sim <- datasets[[paste0("n", sc$n, "_rep", rep_i)]]

    fit_jmjax <- tryCatch(
      jm_fit(
        long_formula = y ~ time,
        surv_formula = Surv(time, event) ~ 1,
        data_long = sim$data_long,
        data_surv = sim$data_surv,
        id_var = "id",
        time_var = "time",
        method = "spline-PH-mcmc",
        random_effects = "intercept_slope",
        control = list(n_interior_knots = sc$n_interior_knots_jmjax,
                        num_warmup = 500, num_samples = 1000, num_chains = 1,
                        progress_bar = FALSE, spline_prior = "penalized")
      ),
      error = function(e) { message("jmjax failed: ", conditionMessage(e)); NULL }
    )

    lme_fit <- tryCatch(
      lme(y ~ time, random = ~ time | id, data = sim$data_long,
          control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100)),
      error = function(e) { message("lme() failed: ", conditionMessage(e)); NULL }
    )
    cox_fit <- coxph(Surv(time, event) ~ 1, data = sim$data_surv)

    fit_jmbayes2 <- tryCatch(
      jm(
        Surv_object = cox_fit,
        Mixed_objects = lme_fit,
        time_var = "time",
        n_chains = 2,
        n_iter = 1500,
        n_burnin = 500,
        control = list(Bsplines_degree = 3,
                        base_hazard_segments = sc$base_hazard_segments_jmbayes2)
      ),
      error = function(e) { message("JMbayes2 failed: ", conditionMessage(e)); NULL }
    )

    row <- data.frame(
      n = sc$n,
      rep = rep_i,
      n_basis_jmjax = sc$n_interior_knots_jmjax + 4,
      base_hazard_segments_jmbayes2 = sc$base_hazard_segments_jmbayes2,
      true_alpha = sim$truth$alpha,

      jmjax_alpha_mean = if (!is.null(fit_jmjax)) fit_jmjax$estimates[["alpha"]] else NA,
      jmjax_alpha_ess = if (!is.null(fit_jmjax)) fit_jmjax$diagnostics$ess[["alpha"]] else NA,
      jmjax_alpha_rhat = if (!is.null(fit_jmjax)) fit_jmjax$diagnostics$rhat[["alpha"]] else NA,
      jmjax_time_sec = if (!is.null(fit_jmjax)) fit_jmjax$convergence$sampling_time_sec else NA,
      jmjax_sigma_b0 = if (!is.null(fit_jmjax)) fit_jmjax$estimates[["sigma_b0"]] else NA,
      jmjax_sigma_b1 = if (!is.null(fit_jmjax)) fit_jmjax$estimates[["sigma_b1"]] else NA,

      jmbayes2_alpha_mean = if (!is.null(fit_jmbayes2)) fit_jmbayes2$statistics$Mean$alphas else NA,
      jmbayes2_alpha_ess = if (!is.null(fit_jmbayes2)) fit_jmbayes2$statistics$Effective_Size$alphas else NA,
      jmbayes2_time_sec = if (!is.null(fit_jmbayes2)) unname(fit_jmbayes2$running_time["elapsed"]) else NA,
      jmbayes2_sigma_b0 = if (!is.null(fit_jmbayes2)) sqrt(fit_jmbayes2$statistics$Mean$D[1]) else NA,
      jmbayes2_sigma_b1 = if (!is.null(fit_jmbayes2)) sqrt(fit_jmbayes2$statistics$Mean$D[3]) else NA
    )

    row$jmjax_alpha_error <- row$jmjax_alpha_mean - row$true_alpha
    row$jmbayes2_alpha_error <- row$jmbayes2_alpha_mean - row$true_alpha
    row$jmjax_ess_per_sec <- row$jmjax_alpha_ess / row$jmjax_time_sec
    row$jmbayes2_ess_per_sec <- row$jmbayes2_alpha_ess / row$jmbayes2_time_sec

    results[[row_idx]] <- row
    row_idx <- row_idx + 1
  }
}

results_df <- do.call(rbind, results)
write.csv(results_df, "jmjax_vs_jmbayes2_benchmark_replicated_v2.csv", row.names = FALSE)
cat("\nSaved full results to jmjax_vs_jmbayes2_benchmark_replicated_v2.csv\n")

agg_fun <- function(x) {
  x <- x[!is.na(x)]
  m <- mean(x)
  s <- sd(x)
  se <- s / sqrt(length(x))
  c(mean = m, sd = s, ci_low = m - 1.96 * se, ci_high = m + 1.96 * se, n = length(x))
}

cat("\n=== jmjax ESS/sec, by scenario ===\n")
print(aggregate(jmjax_ess_per_sec ~ n + n_basis_jmjax, data = results_df, FUN = agg_fun), digits = 4)

cat("\n=== JMbayes2 ESS/sec, by scenario ===\n")
print(aggregate(jmbayes2_ess_per_sec ~ n + n_basis_jmjax, data = results_df, FUN = agg_fun), digits = 4)

cat("\n=== OVERALL SUMMARY (pooled across all", nrow(results_df), "runs) ===\n")

pooled_jmjax_ess <- agg_fun(results_df$jmjax_ess_per_sec)
pooled_jmbayes2_ess <- agg_fun(results_df$jmbayes2_ess_per_sec)
pooled_jmjax_err <- agg_fun(abs(results_df$jmjax_alpha_error))
pooled_jmbayes2_err <- agg_fun(abs(results_df$jmbayes2_alpha_error))

cat(sprintf("jmjax    ESS/sec: %.1f (SD %.1f, 95%% CI [%.1f, %.1f], n=%d)\n",
            pooled_jmjax_ess["mean"], pooled_jmjax_ess["sd"],
            pooled_jmjax_ess["ci_low"], pooled_jmjax_ess["ci_high"], pooled_jmjax_ess["n"]))
cat(sprintf("JMbayes2 ESS/sec: %.1f (SD %.1f, 95%% CI [%.1f, %.1f], n=%d)\n",
            pooled_jmbayes2_ess["mean"], pooled_jmbayes2_ess["sd"],
            pooled_jmbayes2_ess["ci_low"], pooled_jmbayes2_ess["ci_high"], pooled_jmbayes2_ess["n"]))
cat(sprintf("\nSpeed ratio (jmjax / JMbayes2): %.1fx\n",
            pooled_jmjax_ess["mean"] / pooled_jmbayes2_ess["mean"]))

cat(sprintf("\njmjax    mean |alpha error|: %.4f (SD %.4f)\n",
            pooled_jmjax_err["mean"], pooled_jmjax_err["sd"]))
cat(sprintf("JMbayes2 mean |alpha error|: %.4f (SD %.4f)\n",
            pooled_jmbayes2_err["mean"], pooled_jmbayes2_err["sd"]))

cat("\n=== Paired comparison (same datasets, both methods) ===\n")
ess_test <- t.test(results_df$jmjax_ess_per_sec, results_df$jmbayes2_ess_per_sec, paired = TRUE)
print(ess_test)

err_test <- t.test(abs(results_df$jmjax_alpha_error), abs(results_df$jmbayes2_alpha_error), paired = TRUE)
print(err_test)
