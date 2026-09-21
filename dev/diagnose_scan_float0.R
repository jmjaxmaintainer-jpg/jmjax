# ==============================================================================
# Every penalized-spline MCMC fit now fails. Which layer is it?
#
# THE ERROR, on all 30 fits of the generator study including the arm using
# the verbatim old generator that has run many times before:
#
#   TypeError: body_fun output and input must have identical types, got
#   (ShapedArray(int32[], weak_type=True),
#    [ShapedArray(float32[5]), ShapedArray(float32[]),
#     DIFFERENT ShapedArray(int32[], weak_type=True) vs ShapedArray(float0[]),
#     ...], []).
#
# float0 is JAX's tangent type for an integer array - it appears when
# something integer-valued is differentiated. "body_fun output and input
# must have identical types" is a lax.scan / lax.while_loop carry check.
# So a scan carry holds an int32 where AD produced a float0.
#
# WHAT MAKES THIS REACHABLE. spline_prior="penalized" samples W01 and a
# z_step sequence, and rw2_implementation defaults to "scan" - i.e.
# numpyro.contrib.control_flow.scan. Every benchmark script sets
# spline_prior="penalized". The test suite sets it ZERO times, so
# tonight's 410 passing tests did not touch this path at all.
#
# WHY NOW. The r-jmjax virtualenv was rebuilt from scratch tonight.
# jmjax_setup() pins numpy/scipy/jax/jaxlib/numpyro but NOT their
# transitive dependencies, so pip resolved ml_dtypes to whatever is
# current - 0.5.4, a 2025 release - against jax 0.4.30 from June 2024.
# The previous venv was built when those resolved differently.
#
# If that is right, this is not a benchmark inconvenience: a fresh install
# of jmjax today cannot run spline_prior="penalized" at all, and no test
# would report it.
#
# FOUR CONFIGURATIONS, one small fit each. The point is to find the layer,
# and ideally a drop-in that works:
#
#   1 penalized + scan        the failing default
#   2 penalized + vectorized  same model, no numpyro scan. The package's
#                             own notes record both implementations
#                             reaching an identical 63.0 mean leapfrog
#                             steps, so if this works it is a workaround
#                             with no statistical cost.
#   3 independent prior       no RW2 at all - isolates the prior from scan
#   4 penalized + scan, dense_mass_spline = FALSE
#                             rules out the dense mass block as the source
#
# n = 300 and a short budget: this is looking for errors, not estimates.
# Two to three minutes.
# ==============================================================================

library(nlme); library(survival); library(jmjax)

cat("================ 1. THE PYTHON TRACEBACK ================\n")
cat("The R error names the exception; the traceback names the function.\n\n")
te <- tryCatch(reticulate::py_last_error(), error = function(e) NULL)
if (is.null(te)) {
  cat("  no Python error recorded in this session yet - the fits below\n")
  cat("  will produce one.\n")
} else {
  print(te)
}

cat("\n\n================ 2. WHAT IS ACTUALLY INSTALLED ================\n")
cat("jmjax_setup() pins five packages. Everything else pip chose freely,\n")
cat("and that is where a 2024 jax meets a 2025 dependency.\n\n")
invisible(tryCatch({
  reticulate::py_run_string("
import importlib, sys
print('  python      ', sys.version.split()[0])
for m in ('jax','jaxlib','numpyro','ml_dtypes','numpy','scipy','opt_einsum'):
    try:
        mod = importlib.import_module(m)
        print(f'  {m:12s}', getattr(mod, '__version__', '?'))
    except Exception as e:
        print(f'  {m:12s} NOT IMPORTABLE ({e})')
")
}, error = function(e) cat("  could not query versions:", conditionMessage(e), "\n")))

# ------------------------------------------------------------------ data
set.seed(1)
n <- 300L; TMAX <- 10; VISIT <- seq(0, TMAX, length.out = 8L)
b0 <- rnorm(n, 0, .8); b1 <- rnorm(n, 0, .2); x <- rnorm(n)
eta <- -2 + .4*(2 + .3*x + b0)
Tt <- rexp(n, rate = pmin(pmax(exp(eta), 1e-4), 5)/4)
ot <- pmin(Tt, TMAX)
ds <- data.frame(id = factor(seq_len(n)), time = ot,
                 event = as.integer(Tt <= TMAX), x = x)
reps <- vapply(ot, function(o) max(1L, sum(VISIT <= o)), integer(1))
dl <- data.frame(id = factor(rep(seq_len(n), reps), levels = seq_len(n)),
                 time = unlist(lapply(reps, function(k) VISIT[seq_len(k)])),
                 x = rep(x, reps))
dl$y <- (2 + .3*dl$x + rep(b0, reps)) + (.5 + rep(b1, reps))*dl$time +
        rnorm(nrow(dl), 0, .3)

cat(sprintf("\n  data: n = %d | rows = %d | events %.0f%%\n",
            n, nrow(dl), 100*mean(ds$event)))

# ---------------------------------------------------------------- the arms
try_fit <- function(lbl, extra) {
  cat(sprintf("\n  %-42s ", lbl))
  t <- system.time(f <- tryCatch(
    jm_fit(long_formula = y ~ time + x, surv_formula = Surv(time, event) ~ 1,
           data_long = dl, data_surv = ds, id_var = "id", time_var = "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = c(list(n_interior_knots = 5L, num_warmup = 200L,
                            num_samples = 200L, num_chains = 2L, seed = 1L,
                            progress_bar = FALSE), extra)),
    error = function(e) structure(conditionMessage(e), class = "failed")))
  if (inherits(f, "failed")) {
    cat("FAILED\n")
    msg <- strsplit(as.character(f), "\n")[[1]][1]
    cat("      ", substr(msg, 1, 150), "\n")
    return(FALSE)
  }
  rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
  cat(sprintf("OK  %5.1fs | max R-hat %.3f | steps %s\n", as.numeric(t["elapsed"]),
              max(rh),
              { s <- suppressWarnings(as.numeric(f$convergence$mean_num_steps))
                if (is.finite(s)) sprintf("%.0f", s) else "?" }))
  TRUE
}

cat("\n\n================ 3. WHICH CONFIGURATIONS RUN ================\n")
ok1 <- try_fit("1 penalized + scan   (the failing default)",
               list(spline_prior = "penalized"))
ok2 <- try_fit("2 penalized + vectorized  (no numpyro scan)",
               list(spline_prior = "penalized", rw2_implementation = "vectorized"))
ok3 <- try_fit("3 independent prior  (no RW2 at all)",
               list(spline_prior = "independent"))
ok4 <- try_fit("4 penalized + scan, dense_mass_spline = FALSE",
               list(spline_prior = "penalized", dense_mass_spline = FALSE))

cat("\n\n================ VERDICT ================\n")
if (!ok1 && ok2) {
  cat("  numpyro's scan is the broken layer, and rw2_implementation =\n")
  cat("  \"vectorized\" is a drop-in that avoids it. The package notes\n")
  cat("  record both reaching an identical 63.0 mean leapfrog steps, so\n")
  cat("  the benchmark corpus can be re-run by adding that one control\n")
  cat("  option - but the real fix is pinning the transitive dependencies\n")
  cat("  in jmjax_setup(), because a fresh install today is broken for\n")
  cat("  every user of spline_prior = \"penalized\".\n")
} else if (!ok1 && !ok2 && ok3) {
  cat("  The penalized prior itself is broken, not scan specifically -\n")
  cat("  both of its implementations fail while the independent prior\n")
  cat("  runs. Look at the RW2 block rather than the control flow.\n")
} else if (!ok1 && !ok2 && !ok3) {
  cat("  Every spline MCMC configuration fails, so this is not about the\n")
  cat("  prior at all - the environment is broken more broadly than the\n")
  cat("  penalized path and the version pins are the first suspect.\n")
} else if (ok1) {
  cat("  Configuration 1 ran here but failed in the generator study, so\n")
  cat("  the trigger is something that differs between them - sample\n")
  cat("  size, the longer budget, or the data itself. Re-run the\n")
  cat("  generator script's arm A alone at this smaller budget next.\n")
}
if (!ok4 && ok2) {
  cat("\n  (dense_mass_spline = FALSE did not help, which rules the dense\n")
  cat("  mass block out as the source.)\n")
}
cat("\n  Whatever the outcome: spline_prior = \"penalized\" has NO test\n")
cat("  coverage. It appears zero times in tests/testthat/, while every\n")
cat("  benchmark script uses it. That gap is why a 410-test green run\n")
cat("  and a totally broken benchmark corpus are consistent with each\n")
cat("  other, and it should be closed regardless of the fix.\n")
