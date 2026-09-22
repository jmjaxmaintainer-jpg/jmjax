# Quick, non-fitting exploration of JMbayes2's other bundled longitudinal/
# survival datasets, to pick a pbc2 alternative for the real-data pilot.
# Candidates need: a longitudinal outcome with a plausible LINEAR time trend
# (so random = ~ time | id makes sense, q=2), at least one baseline
# (subject-constant) covariate for the longitudinal formula, and a survival
# outcome with a covariate - the same shape pbc2 has.
suppressPackageStartupMessages(library(JMbayes2))
cat("Datasets in JMbayes2:\n")
print(data(package = "JMbayes2")$results[, "Item"])

inspect <- function(long_name, id_name) {
  cat(sprintf("\n=== %s / %s ===\n", long_name, id_name))
  tryCatch({
    data(list = long_name, package = "JMbayes2")
    data(list = id_name,   package = "JMbayes2")
    dl <- get(long_name); di <- get(id_name)
    cat("long dims:", paste(dim(dl), collapse=" x "), " names:", paste(names(dl), collapse=", "), "\n")
    cat("id   dims:", paste(dim(di), collapse=" x "), " names:", paste(names(di), collapse=", "), "\n")
    cat("n subjects (long):", length(unique(dl[[1]])), "\n")
  }, error = function(e) cat("  not available / error:", conditionMessage(e), "\n"))
}

for (cand in list(c("prothro", "prothros"), c("liver", "liver.id"),
                   c("aids", "aids.id"), c("heart.valve", "heart.valve"),
                   c("epileptic", "epileptic"))) {
  inspect(cand[1], cand[2])
}
