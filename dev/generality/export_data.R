# ==============================================================================
# dev/generality/export_data.R - write the two mixed-model datasets used by
# dev/generality/mixed_models.py to CSV (methods paper, Section 7:
# "Beyond joint models").
#
#   Orthodont (nlme): distance ~ age * Sex, random intercept and age slope
#     per child. 27 children x 4 visits. Sex is subject-constant.
#   toenail (HSAUR3): onychomycosis, binary outcome (moderate or severe vs
#     none or mild) ~ treatment * time, random intercept per patient.
#     294 patients, 1908 visits. Treatment is subject-constant.
#
#   Rscript dev/generality/export_data.R
# ==============================================================================
out <- "dev/generality"
suppressPackageStartupMessages(library(nlme))
o <- as.data.frame(nlme::Orthodont)
o <- data.frame(id = as.integer(factor(as.character(o$Subject))),
                y = o$distance, age_c = o$age - 11,
                male = as.integer(o$Sex == "Male"))
utils::write.csv(o, file.path(out, "orthodont.csv"), row.names = FALSE)
cat(sprintf("orthodont.csv: %d rows, %d subjects\n", nrow(o), length(unique(o$id))))

if (!requireNamespace("HSAUR3", quietly = TRUE))
  stop("the toenail data need HSAUR3: install.packages(\"HSAUR3\")", call. = FALSE)
utils::data("toenail", package = "HSAUR3", envir = environment())
t <- data.frame(id = as.integer(factor(as.character(toenail$patientID))),
                y = as.integer(toenail$outcome == "moderate or severe"),
                time = toenail$time,
                trt = as.integer(toenail$treatment == "terbinafine"))
utils::write.csv(t, file.path(out, "toenail.csv"), row.names = FALSE)
cat(sprintf("toenail.csv: %d rows, %d subjects, event rate %.3f\n",
            nrow(t), length(unique(t$id)), mean(t$y)))
