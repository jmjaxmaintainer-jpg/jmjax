# ==============================================================================
# jmjax vs JMbayes2 on REAL PBC2 data, updated with every fix validated
# during today's performance investigation:
#   - progress_bar = FALSE (was defaulting to TRUE - confirmed ~32x
#     overhead earlier today; this alone may be the single biggest
#     practical improvement for this real-data run)
#   - num_chains = 4 with genuine parallel execution (the
#     numpyro.set_host_device_count fix - chains actually run
#     simultaneously now, not queued sequentially)
#   - random_effects_method = "wishart_gibbs" (Wishart-conjugate Gibbs
#     update replacing LKJCholesky+HMC for q=2 correlation - validated
#     faster AND better-converging on every metric tested)
#   - rw2_implementation = "vectorized" (closed-form RW2 construction,
#     replacing the sequential scan - validated ~30% faster with
#     improved R-hat, mathematically identical model)
#   - dense_mass_spline = TRUE (targeted dense mass matrix for the
#     penalized spline coefficients specifically - validated to fix a
#     ~5x deep-tree slowdown for spline_prior="penalized")
#
# NOT included by default: random_effects_method = "wishart_gibbs_centered"
# (the beta_0/random-intercept decoupling fix) - this is flagged
# EXPERIMENTAL with a known credible-interval miscalibration at larger n
# (see the investigation report, Section 6). PBC2's n=312 is closer to
# the validated n=500 case than the flagged n=1500 case, but given this
# is real (not simulated) data where ground truth isn't available to
# double-check against, the safer default for now is the original,
# fully-validated "wishart_gibbs". A commented-out alternative call using
# the centered version is included below if you want to try it and
# compare directly.
# ==============================================================================

library(jmjax)
library(JMbayes2)
library(nlme)
library(survival)

# ------------------------------------------------------------------------
# Data prep - reconstructed from memory of the earlier setup this
# session. If you still have data_long/data_surv from before, skip this
# block entirely and use your existing objects - this is a best-effort
# fallback, not a verified copy of what you actually ran earlier.
# ------------------------------------------------------------------------
# data(pbc2, package = "JMbayes2")
# data_long <- pbc2
# data_long$log_serBilir <- log(data_long$serBilir)  # missing from the first
#                                                     # reconstruction attempt -
#                                                     # pbc2's raw column is
#                                                     # serBilir, not log_serBilir
# data_long$year2 <- data_long$year / 12          # rescaled, per the
# data_surv <- pbc2[!duplicated(pbc2$id), ]        # earlier time-scale
# data_surv$years2 <- data_surv$years / 12         # sensitivity finding
# data_surv$status2 <- as.numeric(data_surv$status != "alive")
# data_long$id <- factor(data_long$id)             # each gets its OWN
# data_surv$id <- factor(data_surv$id)             # factor() call - a
#                                                   # chained a<-b<-value
#                                                   # assignment here would
#                                                   # wrongly reuse the
#                                                   # longer vector for both

# --- jmjax: spline-PH-mcmc, q=2, baseline covariates, all fixes applied ---
cat("=== jmjax: spline-PH-mcmc (all validated fixes) ===\n")
t_jmjax <- system.time({
  fit_jmjax_spline <- jm_fit(
    long_formula = log_serBilir ~ year2 + age,
    surv_formula = Surv(years2, status2) ~ drug,
    data_long = data_long,
    data_surv = data_surv,
    id_var = "id",
    time_var = "year2",
    method = "spline-PH-mcmc",
    random_effects = "intercept_slope",
    random_formula = ~ year2,
    control = list(n_interior_knots = 5, spline_prior = "penalized",
                    rw2_implementation = "vectorized",
                    dense_mass_spline = TRUE,
                    random_effects_method = "wishart_gibbs",
                    num_warmup = 1000, num_samples = 1000, num_chains = 4,
                    progress_bar = FALSE)
  )
})
jmjax_rhat <- max(unlist(fit_jmjax_spline$diagnostics$rhat), na.rm = TRUE)
jmjax_min_ess <- min(unlist(fit_jmjax_spline$diagnostics$ess), na.rm = TRUE)
cat(sprintf("jmjax: %.2f sec | max R-hat: %.4f | min ESS: %.1f | ESS/sec: %.2f\n",
            t_jmjax["elapsed"], jmjax_rhat, jmjax_min_ess, jmjax_min_ess / t_jmjax["elapsed"]))
print(summary(fit_jmjax_spline))

# --- OPTIONAL: experimental wishart_gibbs_centered comparison ---
# Uncomment to also try the experimental beta_0-decoupling fix and
# compare directly against the run above. Flagged experimental due to a
# known credible-interval miscalibration for beta_0 at larger n (see
# investigation report Section 6) - treat beta_0's reported SE with
# extra caution if you use this, though point estimates were unbiased
# in all testing so far.
#
# t_jmjax_centered <- system.time({
#   fit_jmjax_spline_centered <- jm_fit(
#     long_formula = log_serBilir ~ year2 + age,
#     surv_formula = Surv(years2, status2) ~ drug,
#     data_long = data_long,
#     data_surv = data_surv,
#     id_var = "id",
#     time_var = "year2",
#     method = "spline-PH-mcmc",
#     random_effects = "intercept_slope",
#     random_formula = ~ year2,
#     control = list(n_interior_knots = 5, spline_prior = "penalized",
#                     rw2_implementation = "vectorized",
#                     dense_mass_spline = TRUE,
#                     random_effects_method = "wishart_gibbs_centered",
#                     num_warmup = 1000, num_samples = 1000, num_chains = 4,
#                     progress_bar = FALSE)
#   )
# })
# cat(sprintf("jmjax (centered, experimental): %.2f sec | beta_0 ESS: %.1f\n",
#             t_jmjax_centered["elapsed"], fit_jmjax_spline_centered$diagnostics$ess[["beta_0"]]))

# --- JMbayes2: spline baseline (its own default), same covariates ---
cat("\n=== JMbayes2: jm() ===\n")
t_jmbayes2 <- system.time({
  lme_fit <- lme(
    log_serBilir ~ year2 + age,
    random = ~ year2 | id,
    data = data_long,
    control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100)
  )
  cox_fit <- coxph(Surv(years2, status2) ~ drug, data = data_surv)

  fit_jmbayes2_spline <- jm(
    Surv_object = cox_fit,
    Mixed_objects = lme_fit,
    time_var = "year2",
    n_chains = 4,
    n_iter = 8000,
    n_burnin = 2000,
    control = list(Bsplines_degree = 3, base_hazard_segments = 6)
  )
})
cat(sprintf("JMbayes2: %.2f sec\n", t_jmbayes2["elapsed"]))
print(summary(fit_jmbayes2_spline))

# --- Comparison ---
jmbayes2_alpha_ess <- fit_jmbayes2_spline$statistics$Effective_Size$alphas
jmjax_alpha_ess <- fit_jmjax_spline$diagnostics$ess[["alpha"]]

cat("\n\n================ COMPARISON (alpha-specific) ================\n")
cat(sprintf("%-10s %10s %12s %12s\n", "", "time(s)", "alpha ESS", "ESS/sec"))
cat(sprintf("%-10s %10.2f %12.1f %12.2f\n", "jmjax",
            t_jmjax["elapsed"], jmjax_alpha_ess, jmjax_alpha_ess / t_jmjax["elapsed"]))
cat(sprintf("%-10s %10.2f %12.1f %12.2f\n", "JMbayes2",
            t_jmbayes2["elapsed"], jmbayes2_alpha_ess, jmbayes2_alpha_ess / t_jmbayes2["elapsed"]))
cat(sprintf("\nWall-clock ratio (jmjax/JMbayes2): %.2fx %s\n",
            t_jmjax["elapsed"] / t_jmbayes2["elapsed"],
            if (t_jmjax["elapsed"] < t_jmbayes2["elapsed"]) "(jmjax faster)" else "(jmjax slower)"))
cat(sprintf("ESS/sec ratio (jmjax/JMbayes2): %.2fx %s\n",
            (jmjax_alpha_ess / t_jmjax["elapsed"]) / (jmbayes2_alpha_ess / t_jmbayes2["elapsed"]),
            if ((jmjax_alpha_ess / t_jmjax["elapsed"]) > (jmbayes2_alpha_ess / t_jmbayes2["elapsed"])) "(jmjax more efficient)" else "(JMbayes2 more efficient)"))

cat("\n=== For reference: today's earlier PBC2 result on this SAME model, BEFORE any fixes ===\n")
cat("jmjax: near-parity with JMbayes2 (~0.95-1.09x wall-clock, ~1.4-2.0x ESS/sec)\n")
cat("This run should look substantially better if the fixes are working as validated.\n")
