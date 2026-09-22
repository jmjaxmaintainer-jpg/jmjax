# ==============================================================================
# Do the SEEDED values reproduce lme's fit? A direct arithmetic check.
#
# The potential decomposes as:
#   full 167,466.5 | conservative (alpha=0, flat baseline) 168,213.6 | uniform 50,204.3
# so the survival block is irrelevant, and:
#   sigma_e held back -> 35,106.8, a 132,360 drop
# so the longitudinal likelihood carries it. Over ~2,700 observations that is
# 49 each, and 0.5*(r/0.305)^2 = 49 implies residuals near 3.0 where lme's are
# near 0.3. beta is correct to three decimals, so `b` would have to be wrong.
#
# The existing guards check b_std %*% t(L) == ranef() - an internal round
# trip. They do not check that row i of b_std is subject i in the MODEL's
# ordering. That is the only unguarded step, so it is tested here by
# rebuilding the linear predictor under two orderings and seeing which, if
# either, reproduces lme's residuals.
#
#   Rscript dev/diag_residuals.R
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

sim <- sim_joint(n = 300, seed = 3001L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
lme_fit <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)

cat("---- lme's own fit ----\n")
cat(sprintf("  sigma %.5f | sd(resid) %.5f | max|resid| %.5f\n",
            stats::sigma(lme_fit), stats::sd(resid(lme_fit)), max(abs(resid(lme_fit)))))

f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                   method = "spline-PH-mcmc",
                   control = list(n_interior_knots = 5, spline_prior = "penalized",
                                  rw2_implementation = "vectorized",
                                  num_warmup = 50, num_samples = 50,
                                  num_chains = 1, progress_bar = FALSE, seed = 1L))
v <- f$convergence$warm_start$values
if (is.null(v) || is.null(v$b_std)) stop("b_std not recorded verbatim", call. = FALSE)

bs <- matrix(as.numeric(unlist(v$b_std)), ncol = 2, byrow = TRUE)
sb <- as.numeric(unlist(v$sigma_b))
Lc <- matrix(as.numeric(unlist(v$L_corr)), nrow = 2, byrow = TRUE)
L  <- diag(sb) %*% Lc
b  <- bs %*% t(L)                       # model: b = b_std %*% t(L)
be <- as.numeric(unlist(v$beta))

# The design as the model builds it. standardize_covariates rewrites age/sex,
# and beta is seeded on THAT scale, so the check must use it too.
ctr <- c(age = mean(dl$age), sex = mean(dl$sex))
scl <- c(age = sd(dl$age),   sex = sd(dl$sex))
X <- cbind(1, dl$time, (dl$age - ctr[["age"]]) / scl[["age"]],
           (dl$sex - ctr[["sex"]]) / scl[["sex"]])

check <- function(ord, label) {
  idx <- match(dl$id, ord)
  if (anyNA(idx)) { cat(sprintf("  %-28s  ids do not match\n", label)); return(invisible()) }
  mu <- as.numeric(X %*% be) + b[idx, 1] + b[idx, 2] * dl$time
  r  <- dl$y - mu
  cat(sprintf("  %-28s  sd(resid) %8.4f   max %8.4f   implied potential/obs %8.2f\n",
              label, sd(r), max(abs(r)), mean(0.5 * (r / as.numeric(v$sigma_e))^2)))
}

cat("\n---- residuals from the SEEDED values, under two subject orderings ----\n")
cat(sprintf("  (seeded sigma_e = %.5f; lme's residual SD = %.5f)\n\n",
            as.numeric(v$sigma_e), stats::sd(resid(lme_fit))))
check(sort(unique(dl$id)),                         "ascending numeric id")
check(unique(dl$id),                               "first-appearance order")
check(as.integer(rownames(ranef(lme_fit))),        "nlme ranef() row order")

cat("\n---- read ----\n")
cat("  One of these should give sd(resid) ~ 0.31, matching lme. If ALL give\n")
cat("  ~3, the ordering is not the issue and `b` is wrong some other way.\n")
cat("  If the ascending-id row is the good one, the seeding is correct and\n")
cat("  the 167k has a different source - in which case the next thing to\n")
cat("  measure is the potential at the TRUE parameters, which would say\n")
cat("  whether 50,204 for a uniform start is itself the anomaly.\n")
