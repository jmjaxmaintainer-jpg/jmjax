# ==============================================================================
# Why is the lme warm start REJECTED on simulated data but ACCEPTED on pbc2?
#
#   simulated  potential 71224.8 warm vs 30505.0 uniform  -> rejected, 3/3 arms
#   pbc2       accepted on every arm of every run today
#
# So every simulated-data fit in the whole study ran cold. That does not
# threaten any correctness result - it applied identically to arms A, D and
# E - but it means jmjax's throughput was measured UNSEEDED while JMbayes2
# was handed the same pre-fits and uses them. The draws/sec comparison is
# tilted by an unknown amount until this is understood.
#
# THE HYPOTHESIS, stated before looking. The warm start seeds SOME sites and
# leaves the rest at uniform, so it is a MIXTURE - and a mixture can score
# worse than a uniform start. The survival likelihood carries
# exp(alpha * m_i(t)); seeding beta and b at their MLE while alpha stays
# uniform on [-2, 2] gives m_i of about 2-3 and swings of exp(+-6). A fully
# uniform start avoids that because m_i is small and random too. If so, the
# culprit is a site that is NOT in the seeded list, and the fix is to seed
# it - not to loosen the acceptance test, which is doing its job.
#
# This prints, for both datasets:
#   - which sites the warm start seeded            (convergence$warm_start$sites)
#   - which sites the model actually samples       (names of posterior_samples)
#   - the difference: seeded nowhere, left uniform
# and the two potentials. If the simulated case has an unseeded site that
# pbc2 also leaves unseeded, the difference is elsewhere and the hypothesis
# is wrong.
#
#   Rscript dev/diag_warmstart.R 2>&1 | tee ~/Documents/R/jmjax_results/diag_warmstart.log
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
suppressPackageStartupMessages(library(JMbayes2))
`%||%` <- function(a, b) if (is.null(a)) b else a
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

# Rebuild W from the seeded sites exactly as the model does, so the SEEDED
# BASELINE LOG-HAZARD can be compared against the one the data was generated
# from. W01 and tau_w on their own do not say whether the target is sane:
# the RW2 reconstruction is exact, so a huge linear term is cancelled by an
# equally huge double-cumsum term, and reading catastrophe into W01 alone
# (as I did) is wrong. Only the reconstructed W settles it.
rebuild_W <- function(v) {
  if (is.null(v$W01) || is.null(v$z_step) || is.null(v$tau_w)) return(NULL)
  W01 <- as.numeric(unlist(v$W01)); z <- as.numeric(unlist(v$z_step))
  sw  <- 1 / sqrt(as.numeric(unlist(v$tau_w)))
  k   <- seq_along(z)
  c(W01, W01[2] + k * (W01[2] - W01[1]) + sw * cumsum(cumsum(z)))
}

report <- function(label, fit, truth = NULL) {
  ws <- fit$convergence$warm_start
  cat(sprintf("\n================ %s ================\n", label))
  if (is.null(ws)) { cat("  no warm start attempted\n"); return(invisible()) }
  cat(sprintf("  used: %s  tier: %s\n", isTRUE(as.logical(ws$used)),
              ws$tier %||% "(none - cold start)"))
  cat(sprintf("  potential   full %.1f | conservative %s | uniform %.1f\n",
              as.numeric(ws$potential_warm),
              if (is.null(ws$potential_conservative)) "n/a"
              else sprintf("%.1f", as.numeric(ws$potential_conservative)),
              as.numeric(ws$potential_uniform)))
  seeded  <- sort(as.character(unlist(ws$sites)))
  sampled <- sort(setdiff(names(fit$posterior_samples), "b"))
  cat(sprintf("  seeded  (%d): %s\n", length(seeded), paste(seeded, collapse = ", ")))
  cat(sprintf("  sampled (%d): %s\n", length(sampled), paste(sampled, collapse = ", ")))
  left <- setdiff(sampled, seeded)
  cat(sprintf("  LEFT UNIFORM (%d): %s\n", length(left),
              if (length(left)) paste(left, collapse = ", ") else "(none)"))
  v <- ws$values
  if (!is.null(v)) {
    cat("  seeded VALUES (truth for the simulated set: sigma_e 0.3,\n")
    cat("    sigma_b 0.8/0.3, rho 0.3, alpha 0.5, gamma -0.3, beta 2.0/0.5/0.02/-0.30):\n")
    for (nm in c("sigma_e", "sigma_b", "beta", "alpha", "gamma", "tau_w",
                 "W01", "z_step", "L_corr", "b_std")) {
      if (is.null(v[[nm]])) next
      x <- v[[nm]]
      if (is.list(x))
        cat(sprintf("    %-8s shape %s  mean %+.4f  sd %.4f  absmax %.4f\n", nm,
                    paste(unlist(x$shape), collapse = "x"), x$mean, x$sd, x$absmax))
      else {
        xv <- as.numeric(unlist(x))
        if (length(xv) > 12L)
          cat(sprintf("    %-8s n=%d  mean %+.4f  sd %.4f  absmax %.4f\n",
                      nm, length(xv), mean(xv), stats::sd(xv), max(abs(xv))))
        else
          cat(sprintf("    %-8s %s\n", nm,
                      paste(sprintf("%+.5f", xv), collapse = "  ")))
      }
    }
    W <- rebuild_W(v)
    if (!is.null(W)) {
      cat(sprintf("\n    RECONSTRUCTED W (the seeded log baseline hazard):\n      %s\n",
                  paste(sprintf("%+.2f", W), collapse = "  ")))
      cat(sprintf("      range %+.2f to %+.2f   =>  hazard multiplier %.3g to %.3g\n",
                  min(W), max(W), exp(min(W)), exp(max(W))))
      if (!is.null(truth)) {
        cat(sprintf("      TRUE log h0 over the follow-up: %+.2f to %+.2f\n",
                    truth[1], truth[2]))
        cat(sprintf("      -> seeded baseline is off by up to %.3g ORDERS OF MAGNITUDE\n",
                    max(abs(range(W) - truth)) / log(10)))
      }
    }
  }
}

CTL <- list(n_interior_knots = 5, spline_prior = "penalized",
            rw2_implementation = "vectorized", dense_mass_spline = TRUE,
            num_warmup = 100, num_samples = 100, num_chains = 1,
            progress_bar = FALSE, seed = 1L)

# ---- simulated (warm start rejected) ------------------------------------
sim <- sim_joint(n = 300, seed = 3001L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
lme_s <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
             control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_s <- coxph(Surv(time, event) ~ trt, data = ds)
fit_s <- jm_fit_prefit(lme_s, cox_s, data_surv = ds, time_var = "time",
                       method = "spline-PH-mcmc", control = CTL)
# True baseline for this generator: log h0(t) = log(shape) + (shape-1)log(t)
# + log_lambda0, evaluated across the observed follow-up.
.tt <- range(ds$time[ds$time > 0])
.tru <- log(1.3) + 0.3 * log(.tt) + sim$truth$log_lambda0
cat(sprintf("\ngenerator: log_lambda0 %.3f, follow-up %.2f to %.2f\n",
            sim$truth$log_lambda0, .tt[1], .tt[2]))
report("SIMULATED (rejection expected)", fit_s, truth = .tru)

# ---- pbc2 (warm start accepted) - the control ---------------------------
data("pbc2", package = "JMbayes2"); data("pbc2.id", package = "JMbayes2")
pbc2$id <- as.integer(as.character(pbc2$id)); pbc2.id$id <- as.integer(as.character(pbc2.id$id))
pbc2.id$status2 <- as.integer(pbc2.id$status != "alive")
pbc2$year2 <- pbc2$year / 12; pbc2.id$years2 <- pbc2.id$years / 12
pl <- pbc2[, c("id", "year2", "serBilir", "age")]; pl$log_serBilir <- log(pl$serBilir)
psv <- pbc2.id[, c("id", "years2", "status2", "drug")]
lme_p <- lme(log_serBilir ~ year2 + age, random = ~ year2 | id, data = pl,
             control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_p <- coxph(Surv(years2, status2) ~ drug, data = psv)
fit_p <- jm_fit_prefit(lme_p, cox_p, data_surv = psv, time_var = "year2",
                       method = "spline-PH-mcmc", control = CTL)
report("PBC2 (acceptance expected - the control)", fit_p)

cat("\n---------------- read ----------------\n")
cat("  If both datasets leave the SAME sites uniform, the mixture hypothesis\n")
cat("  does not explain the difference and the cause is in the data (event\n")
cat("  times to 10 vs pbc2's ~1.2 after the /12 rescale would put the spline\n")
cat("  knots and the quadrature on quite different scales).\n")
cat("  If the simulated fit leaves a site uniform that pbc2 seeds, that site\n")
cat("  is the bug and seeding it is the fix.\n")
