# ==============================================================================
# Ground truth from inside the model: b_raw_orth_debug (numpyro.deterministic,
# added temporarily) is b BEFORE the orthogonalizing projection. b is AFTER.
# For arm E on the n=500 config where verify_orth_slope.R found column 1 NOT
# pinned (~0.04-0.065 instead of ~1e-16), this checks, per draw:
#
#   mean_i(b_raw[,2])   - what the projection is supposed to remove entirely
#   mean_i(b[,2])       - what's left after
#   removed = mean_i(b_raw[,2]) - mean_i(b[,2])
#
# If removed == mean_i(b_raw[,2]) (to float64 precision), the subtraction IS
# doing exactly what it claims and the residual is coming from somewhere else
# entirely (e.g. Q itself is not what's assumed). If removed is systematically
# SMALLER than mean_i(b_raw[,2]), the projection itself is under-correcting.
#
#   Rscript dev/diag_slope_projection.R
# ==============================================================================
suppressPackageStartupMessages({ library(jmjax); library(nlme); library(survival) })
.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(if (length(.self)) file.path(dirname(.self[1]), "sim_joint.R") else "dev/sim_joint.R")

extract_col <- function(ps, site, j, NS) t(vapply(ps[[site]], function(d) {
  if (is.list(d)) vapply(d, function(r) as.numeric(r)[j], numeric(1))
  else { a <- as.matrix(d); as.numeric(a[, j]) }
}, numeric(NS)))

sim <- sim_joint(n = 500, seed = 7L, k_extra = 2L)
dl <- sim$data_long; ds <- sim$data_surv
lme_fit <- lme(y ~ time + age + sex, random = ~ time | id, data = dl,
               control = lmeControl(opt = "optim", msMaxIter = 200, niterEM = 100))
cox_fit <- coxph(Surv(time, event) ~ trt, data = ds)
NS <- length(unique(dl$id))
cat(sprintf("n = %d subjects\n", NS))

ctl <- list(n_interior_knots = 5, spline_prior = "penalized",
            rw2_implementation = "vectorized", dense_mass_spline = TRUE,
            num_warmup = 50, num_samples = 50, num_chains = 1,
            progress_bar = FALSE, seed = 1L, orthogonalize_b = TRUE)
f <- jm_fit_prefit(lme_fit, cox_fit, data_surv = ds, time_var = "time",
                   method = "spline-PH-mcmc", control = ctl)
ps <- f$posterior_samples
stopifnot("b_raw_orth_debug" %in% names(ps))

braw1 <- rowMeans(extract_col(ps, "b_raw_orth_debug", 2, NS))   # per-draw mean_i(b_raw[,2])
b1    <- rowMeans(extract_col(ps, "b", 2, NS))                  # per-draw mean_i(b[,2])
removed <- braw1 - b1

cat(sprintf("\n%-6s %14s %14s %14s %14s\n", "draw", "mean(b_raw1)", "mean(b1)", "removed", "removed/raw"))
for (i in c(1:5, (length(braw1)-4):length(braw1))) {
  cat(sprintf("%-6d %14.6e %14.6e %14.6e %14.4f\n",
              i, braw1[i], b1[i], removed[i], removed[i] / braw1[i]))
}

cat(sprintf("\nmax|mean(b_raw1)| over draws: %.6e\n", max(abs(braw1))))
cat(sprintf("max|mean(b1)|     over draws: %.6e\n", max(abs(b1))))
cat(sprintf("median(removed / mean_raw) over draws: %.6f  (1.0 = fully removed, 0 = nothing removed)\n",
            median(removed / braw1)))
cat(sprintf("range(removed / mean_raw): [%.6f, %.6f]\n",
            min(removed / braw1), max(removed / braw1)))
