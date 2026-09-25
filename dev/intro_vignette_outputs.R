# Runs the code shown in vignettes/jmjax-introduction.Rmd and writes its
# output to dev/intro_vignette_outputs.txt, so the vignette's "#>" lines
# come from a real run rather than being typed by hand. The vignette's
# chunks are eval = FALSE (a CRAN check machine has no Python backend).
#
# Run from the package root:  Rscript dev/intro_vignette_outputs.R

suppressPackageStartupMessages({
  devtools::load_all(quiet = TRUE)
  library(survival)
})
out <- file("dev/intro_vignette_outputs.txt", open = "w")
# Each chunk runs in the global environment, so objects made in one chunk
# (fit_w, fit_s) are there for the next, as in the vignette.
chunk <- function(label, expr) {
  e <- substitute(expr)
  # Split a { } block into its lines so each one's value is printed, as
  # at the console (withAutoprint only splits braces it quotes itself).
  e <- if (is.call(e) && identical(e[[1]], as.name("{"))) as.expression(as.list(e)[-1])
       else as.expression(list(e))
  txt <- utils::capture.output(withAutoprint(e, evaluated = TRUE,
                                             local = globalenv(), echo = TRUE,
                                             max.deparse.length = Inf))
  writeLines(c(paste0("===== ", label), txt, ""), out)
}

chunk("data", {
  data("pbc2", "pbc2.id", package = "JM")
  pbc2$log_bili <- log(pbc2$serBilir)
  head(pbc2[, c("id", "year", "log_bili", "drug")], 4)
  head(pbc2.id[, c("id", "years", "status2", "drug")], 4)
})

chunk("mle", {
  fit_w <- jm_mle(log_bili ~ year, Surv(years, status2) ~ drug,
                  data_long = pbc2, data_surv = pbc2.id,
                  id_var = "id", time_var = "year", baseline = "weibull")
  fit_w
})

chunk("bayes", {
  fit_s <- jm_bayes(log_bili ~ year, Surv(years, status2) ~ drug,
                    data_long = pbc2, data_surv = pbc2.id,
                    id_var = "id", time_var = "year",
                    random_effects = "intercept_slope", random_formula = ~ year,
                    chains = 2, warmup = 500, samples = 1000, seed = 1)
  fit_s
})

chunk("summary_bayes", summary(fit_s))

chunk("ranef", head(ranef(fit_s), 3))

chunk("methods", {
  coef(fit_s)
  confint(fit_s, "alpha")
  round(sqrt(diag(vcov(fit_s))), 4)
  logLik(fit_w); AIC(fit_w); BIC(fit_w)
  nobs(fit_w)
})

chunk("predict", {
  id_c <- pbc2.id$id[pbc2.id$status2 == 0 & pbc2.id$years < 8][1]
  nd <- pbc2[pbc2$id == id_c, ]
  predict(fit_s, newdata = nd, times = 0:4)
  predict(fit_s, newdata = nd, process = "event",
          times = nd$years[1] + 0:3)
})

chunk("warm_start", {
  fit_spline_mle <- jm_mle(log_bili ~ year, Surv(years, status2) ~ drug,
                           data_long = pbc2, data_surv = pbc2.id,
                           id_var = "id", time_var = "year",
                           baseline = "spline", init_theta = fit_w$estimates)
  coef(fit_spline_mle)[c("alpha", "gamma_0")]
  fit_spline_mle$convergence$converged
})

close(out)
cat("Wrote dev/intro_vignette_outputs.txt\n")
