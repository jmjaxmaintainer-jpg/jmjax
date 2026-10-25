# dev/verify_joint_vs_columnwise.R - does the joint absorbable construction
# (0.4.0) give the same subspace as the per-column construction (0.3.x) on
# the designs behind the methods paper and the test suite?
#
# WHY. Revised Section 3 of the methods paper defines the absorbable
# subspace jointly across random-effect columns, and says no reported number
# changes because, for every design in the paper, the joint subspace equals
# the per-column one. Each fit now records both in
# fit$convergence$orthogonalize$rotation (k, k_columnwise and
# columnwise_subspace_distance = ||(I - Q_old Q_old') Q||_2, the largest
# principal-angle sine; 1 when the dimensions differ). This script reads
# them off short fits. Only the design matters, so the MCMC is a token
# 5 + 5 draws.
#
# DESIGNS.
#   aids, pbc2  as dev/study_realdata_rotate.R (q = 2; paper Tables 3-4,
#               dev/study_common_ess.R) and dev/pilot_q1_intercept.R (q = 1)
#   sim_*       dev/sim_joint.R, k_extra = 3: y ~ time + age + sex + trt,
#               sparse/base/dense visits, q = 2 and q = 1 (Section 6,
#               dev/pilot_rotate_grid.R, dev/study_calibration.R)
#   test_*      tests/testthat/helper-simulate.R designs, including a
#               covariate-by-time interaction and the delta/area channels
#   coupled     y ~ time + I(x * (1 + time)): the paper's Example 2, where
#               the two constructions MUST differ (k = 2 vs 1)
# Section 7 (orthodont, toenail) uses dev/generality/mixed_models.py, a
# standalone mixed-model script; its designs were checked separately (same
# subspace, distance < 1e-14).
#
# RUN (from the package root; needs JMbayes2 for aids/pbc2):
#   Rscript dev/verify_joint_vs_columnwise.R
# Output: printed table, and ../jmjax_results/joint_vs_columnwise.csv
# Takes a few minutes (one JAX compile per design).

suppressPackageStartupMessages({
  if (requireNamespace("pkgload", quietly = TRUE) && file.exists("DESCRIPTION")) {
    pkgload::load_all(".", quiet = TRUE)
  } else {
    library(jmjax)
  }
  library(survival)
})
source("dev/sim_joint.R")
source("tests/testthat/helper-simulate.R")

CTL <- list(num_warmup = 5, num_samples = 5, num_chains = 1, seed = 1,
            progress_bar = FALSE)

one <- function(label, data_long, data_surv, lform, sform, time_var, q,
                ff = NULL, expect_same = TRUE) {
  args <- list(long_formula = lform, surv_formula = sform,
               data_long = data_long, data_surv = data_surv,
               id_var = "id", time_var = time_var,
               method = "spline-PH-mcmc",
               random_effects = if (q == 1) "intercept" else "intercept_slope",
               random_formula = if (q == 1) ~ 1 else stats::as.formula(paste("~", time_var)),
               control = CTL)
  if (!is.null(ff)) args$functional_forms <- ff
  t0 <- proc.time()[["elapsed"]]
  f <- tryCatch(do.call(jm_fit, args), error = function(e) e)
  if (inherits(f, "error")) {
    message(label, ": ", conditionMessage(f))
    return(data.frame(design = label, q = q, N = NA, baseline_only = NA,
                      k_joint = NA, k_columnwise = NA, distance = NA,
                      condition_s = NA, rejected_s = NA, dim_ker_M = NA,
                      expect_same = expect_same, ok = FALSE, sec = NA))
  }
  r <- f$convergence$orthogonalize$rotation
  nv <- table(data_long$id)
  same <- isTRUE(as.integer(r$k) == as.integer(r$k_columnwise)) &&
    isTRUE(r$columnwise_subspace_distance < 1e-8)
  data.frame(design = label, q = q, N = length(nv), baseline_only = sum(nv == 1L),
             k_joint = as.integer(r$k), k_columnwise = as.integer(r$k_columnwise),
             distance = signif(as.numeric(r$columnwise_subspace_distance), 3),
             condition_s = signif(as.numeric(r$condition_s_residual), 3),
             rejected_s = as.integer(r$n_rejected_condition_s),
             dim_ker_M = as.integer(r$dim_ker_M),
             expect_same = expect_same, ok = same == expect_same,
             sec = round(proc.time()[["elapsed"]] - t0, 1))
}

rows <- list()
add <- function(...) { rows[[length(rows) + 1L]] <<- one(...); print(utils::tail(rows[[length(rows)]], 1)) }

# ---- real data ---------------------------------------------------------------
if (requireNamespace("JMbayes2", quietly = TRUE)) {
  e <- new.env()
  utils::data("aids", "aids.id", "pbc2", "pbc2.id", package = "JMbayes2", envir = e)
  dl <- e$aids[, c("patient", "obstime", "CD4", "drug", "gender")]
  dl$y <- sqrt(dl$CD4); names(dl)[1] <- "id"
  ds <- e$aids.id[, c("patient", "Time", "death", "drug")]
  names(ds) <- c("id", "stime", "event", "drug")
  for (q in 2:1)
    add(sprintf("aids q=%d", q), dl, ds, y ~ obstime + drug + gender,
        Surv(stime, event) ~ drug, "obstime", q)

  dl <- e$pbc2[, c("id", "year", "serBilir", "age")]
  dl$id <- as.integer(as.character(dl$id)); dl$y <- log(dl$serBilir)
  ds <- e$pbc2.id[, c("id", "years", "status", "drug")]
  ds <- data.frame(id = as.integer(as.character(ds$id)), stime = ds$years,
                   event = as.integer(ds$status != "alive"), drug = ds$drug)
  for (q in 2:1)
    add(sprintf("pbc2 q=%d", q), dl, ds, y ~ year + age,
        Surv(stime, event) ~ drug, "year", q)
} else {
  message("JMbayes2 not installed: aids and pbc2 skipped")
}

# ---- simulation designs (Section 6) -----------------------------------------
for (vis in c(sparse = 2.0, base = 1.0, dense = 0.5)) {
  s <- sim_joint(n = 300, seed = 7001L, k_extra = 3L, visit_gap = vis)
  for (q in 2:1)
    add(sprintf("sim gap=%.1f q=%d", vis, q), s$data_long, s$data_surv,
        y ~ time + age + sex + trt, Surv(time, event) ~ trt, "time", q)
}

# ---- test-suite designs ------------------------------------------------------
s <- simulate_joint_data_re2(n = 200, seed = 21)
for (q in 2:1)
  add(sprintf("test y~time q=%d", q), s$data_long, s$data_surv, y ~ time,
      Surv(time, event) ~ 1, "time", q)
set.seed(5); x <- stats::rnorm(200); s$data_long$x <- x[s$data_long$id]
add("test y~time+x q=2", s$data_long, s$data_surv, y ~ time + x,
    Surv(time, event) ~ 1, "time", 2)
# Covariate-by-time interaction with baseline-only subjects: the per-column
# construction's consensus fill admits a non-absorbable direction here
# (see tests/testthat/python/check_joint_construction.py), so the two may
# differ. Not a design used in the paper.
add("test y~time*x q=2", s$data_long, s$data_surv, y ~ time * x,
    Surv(time, event) ~ 1, "time", 2, expect_same = NA)
add("coupled (Ex. 2) q=2", s$data_long, s$data_surv, y ~ time + I(x * (1 + time)),
    Surv(time, event) ~ 1, "time", 2, expect_same = FALSE)
for (ch in c("delta", "area")) {
  sc <- simulate_joint_data_re2_channel(ch, n = 200, seed = 31)
  add(sprintf("test %s channel q=2", ch), sc$data_long, sc$data_surv, y ~ time,
      Surv(time, event) ~ 1, "time", 2,
      ff = stats::as.formula(sprintf("~ value(y) + %s(y)", ch)))
}

res <- do.call(rbind, rows)
res$ok[is.na(res$expect_same)] <- NA
cat("\n"); print(res, row.names = FALSE)
out <- file.path("..", "jmjax_results", "joint_vs_columnwise.csv")
if (dir.exists(dirname(out))) {
  utils::write.csv(res, out, row.names = FALSE)
  cat("\nWrote", out, "\n")
}
bad <- res[!is.na(res$ok) & !res$ok, ]
if (nrow(bad)) {
  cat("\nUNEXPECTED:\n"); print(bad, row.names = FALSE)
  quit(status = 1)
}
cat("\nAll designs as expected.\n")
