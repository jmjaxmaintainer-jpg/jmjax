# Runs the two code examples added to vignettes/jmjax-introduction.Rmd for
# jmjax 0.5.0 (more than two random effects; several markers) and writes
# their output to dev/intro_vignette_outputs_050.txt, so the vignette's
# "#>" lines come from a real run. Same conventions as
# dev/intro_vignette_outputs.R.
#
# Run from the package root:  Rscript dev/intro_vignette_outputs_050.R

suppressPackageStartupMessages({
  devtools::load_all(quiet = TRUE)
  library(survival)
})
out <- file("dev/intro_vignette_outputs_050.txt", open = "w")
chunk <- function(label, expr) {
  e <- substitute(expr)
  e <- if (is.call(e) && identical(e[[1]], as.name("{"))) as.expression(as.list(e)[-1])
       else as.expression(list(e))
  txt <- utils::capture.output(withAutoprint(e, evaluated = TRUE,
                                             local = globalenv(), echo = TRUE,
                                             max.deparse.length = Inf))
  writeLines(c(paste0("===== ", label), txt, ""), out)
}

data("pbc2", "pbc2.id", package = "JM")
pbc2$log_bili <- log(pbc2$serBilir)

chunk("general-q", {
  fit_q3 <- jm_bayes(log_bili ~ year + I(year^2), Surv(years, status2) ~ drug,
                     data_long = pbc2, data_surv = pbc2.id,
                     id_var = "id", time_var = "year",
                     random_formula = ~ year + I(year^2),
                     chains = 2, warmup = 500, samples = 1000, seed = 1)
  fit_q3
  VarCorr(fit_q3)
})

chunk("multi", {
  fit_m <- jm_bayes(list(bili = log_bili ~ year, alb = albumin ~ year),
                    Surv(years, status2) ~ drug,
                    data_long = list(bili = pbc2, alb = pbc2), data_surv = pbc2.id,
                    id_var = "id", time_var = "year", random_formula = ~ year,
                    chains = 2, warmup = 500, samples = 1000, seed = 1)
  fit_m
  VarCorr(fit_m)
})

close(out)
cat("wrote dev/intro_vignette_outputs_050.txt\n")
