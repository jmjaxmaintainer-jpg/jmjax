# Acceptance test for predict() (phase 1): conditional survival against
# JMbayes2 on pbc2.
#
# Criterion, fixed BEFORE any result was seen (agreed 2026-09-23):
#   for five censored subjects, at each prediction time u,
#     (a) |S_jmjax(u | t) - S_JMbayes2(u | t)| <= 0.03, and
#     (b) each package's estimate lies inside the other's 95% interval.
#   predict() ships in 0.3.0 only if every (subject, time) passes.
#
# Why censored subjects: jmjax's in-sample random effects condition on the
# subject's full data - all marker values and survival to the censoring
# time - which is exactly what JMbayes2's predict() conditions on when
# given that subject's rows with Tstart = the censoring time.
#
# The two models are matched: same longitudinal and survival submodels,
# and jmjax's baseline hazard set to JMbayes2's defaults (equally spaced
# knots over 10 segments = 9 interior knots, quadratic B-splines, and the
# second-order difference penalty). Priors still differ, so exact equality
# is not expected - hence the tolerance.
#
# Run from the package root:  Rscript dev/verify_predict_jmbayes2.R
# Takes a few minutes (two MCMC fits on 312 subjects).

suppressPackageStartupMessages({
  devtools::load_all(quiet = TRUE)
  library(JMbayes2); library(survival); library(nlme)
})

pbc2$lSB <- log(pbc2$serBilir)

cat("=== Fitting JMbayes2 ===\n")
lme_fit <- lme(lSB ~ year, random = ~ year | id, data = pbc2)
cox_fit <- coxph(Surv(years, status2) ~ drug, data = pbc2.id)
set.seed(2026)
jmb <- jm(cox_fit, lme_fit, time_var = "year",
          n_iter = 6500L, n_burnin = 1500L, n_chains = 3L)

cat("=== Fitting jmjax ===\n")
fit <- jm_fit(lSB ~ year, Surv(years, status2) ~ drug,
              data_long = pbc2, data_surv = pbc2.id,
              id_var = "id", time_var = "year", method = "spline-PH-mcmc",
              random_effects = "intercept_slope", random_formula = ~ year,
              control = list(knot_placement = "equal", n_interior_knots = 9L,
                             spline_order = 3L, spline_prior = "penalized",
                             num_warmup = 1000L, num_samples = 2000L,
                             num_chains = 2L, seed = 2026L))
cat(sprintf("jmjax: max R-hat %.3f, min ESS %.0f\n",
            max(unlist(fit$diagnostics$rhat), na.rm = TRUE),
            min(unlist(fit$diagnostics$ess), na.rm = TRUE)))
cat(sprintf("alpha: jmjax %.3f vs JMbayes2 %.3f\n", fit$estimates[["alpha"]],
            tryCatch(unname(coef(jmb)$association)[1], error = function(e) NA_real_)))

# Five censored subjects spread across follow-up, chosen by rule.
cens <- pbc2.id[pbc2.id$status2 == 0, c("id", "years")]
cens <- cens[order(cens$years), ]
pick <- cens$id[round(quantile(seq_len(nrow(cens)), c(0.2, 0.35, 0.5, 0.65, 0.8)))]
t_max <- max(pbc2.id$years)

rows <- list()
for (sid in pick) {
  nd <- pbc2[pbc2$id == sid, ]
  t0 <- nd$years[1]
  uu <- t0 + c(0.5, 1, 2, 3)
  uu <- uu[uu < t_max - 0.01]
  if (!length(uu)) next

  pj <- predict(fit, newdata = nd, process = "event", times = uu)
  pb <- predict(jmb, newdata = nd, process = "event", times = uu)
  # JMbayes2 returns cumulative risk P(T <= u | T > t0); convert.
  b <- data.frame(time = pb$times, S_jmb = 1 - pb$pred,
                  jmb_lower = 1 - pb$upp, jmb_upper = 1 - pb$low)
  b$time <- round(b$time, 8)
  m <- merge(data.frame(id = sid, t0 = t0, time = round(pj$time, 8),
                        S_jmjax = pj$estimate,
                        jmjax_lower = pj$lower, jmjax_upper = pj$upper),
             b, by = "time")
  rows[[length(rows) + 1]] <- m
}
res <- do.call(rbind, rows)
res$abs_diff <- abs(res$S_jmjax - res$S_jmb)
res$pass_diff <- res$abs_diff <= 0.03
res$pass_bands <- res$S_jmjax >= res$jmb_lower & res$S_jmjax <= res$jmb_upper &
                  res$S_jmb >= res$jmjax_lower & res$S_jmb <= res$jmjax_upper
res$pass <- res$pass_diff & res$pass_bands

print(res[, c("id", "t0", "time", "S_jmjax", "S_jmb", "abs_diff",
              "pass_diff", "pass_bands")], digits = 3, row.names = FALSE)
utils::write.csv(res, "dev/verify_predict_jmbayes2.csv", row.names = FALSE)

cat(sprintf("\n%d of %d (subject, time) pairs pass; max |diff| = %.4f\n",
            sum(res$pass), nrow(res), max(res$abs_diff)))
cat(if (all(res$pass)) "PASS: predict() meets the acceptance criterion.\n"
    else "FAIL: predict() does not meet the acceptance criterion - do not ship it.\n")
