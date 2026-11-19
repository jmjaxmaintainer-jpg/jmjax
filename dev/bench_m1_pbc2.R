# ==============================================================================
# dev/bench_m1_pbc2.R - paper-grade rerun of the pbc2 multi-marker comparison
# (Milestone 1, after 0.5.0). Shared code and conventions:
# dev/bench_m1_common.R (arms A R C J, the R-hat/ESS bar, timing, hardware).
#
# CONFIGS (random intercept and slope in year for every marker):
#   q4  log(serBilir) ~ year + age,  albumin ~ year
#   q6  log(serBilir) ~ year + age,  albumin ~ year,  prothrombin ~ year
# Survival: Surv(years, death or transplant) ~ drug; spline baseline (5
# interior knots, penalized; JMbayes2: cubic B-splines on 6 segments); value
# association per marker.
#
# OUTPUT (BM_OUTDIR, default ~/Documents/R/jmjax_results/m1_paper):
#   hardware.txt, pbc2.csv (one row per config x seed x arm x parameter),
#   pbc2_summary.csv (one row per fit), pbc2_tables.md, pbc2_estimates.csv
#
# USAGE (resumable: rerun the same command after an interruption):
#   JMJAX_NUM_DEVICES=4 caffeinate -i Rscript dev/bench_m1_pbc2.R
#   BM_SEEDS=1 BM_SAMPLES=500 BM_WARMUP=500 BM_JB_KEEP=2000 BM_MAX_DOUBLINGS=0 \
#     BM_OUTDIR=/tmp/m1_smoke Rscript dev/bench_m1_pbc2.R        # quick look
# ==============================================================================

local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "dev"
  source(file.path(here, "bench_m1_common.R"), local = globalenv())
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) stop("JMbayes2 is required", call. = FALSE)

CONFIGS <- .envl("BM_CONFIGS", c("q4", "q6"))
ARMS    <- .envl("BM_ARMS", c("A", "R", "C", "J"))
NSEED   <- .envi("BM_SEEDS", 10L)
OUTDIR  <- Sys.getenv("BM_OUTDIR", path.expand("~/Documents/R/jmjax_results/m1_paper"))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, "pbc2.csv")
write_hardware(OUTDIR)

utils::data("pbc2", package = "JMbayes2", envir = environment())
utils::data("pbc2.id", package = "JMbayes2", envir = environment())
pl <- data.frame(id = as.integer(as.character(pbc2$id)), year = pbc2$year,
                 age = pbc2$age, bili = log(pbc2$serBilir), alb = pbc2$albumin,
                 pro = pbc2$prothrombin)
ds <- data.frame(id = as.integer(as.character(pbc2.id$id)), stime = pbc2.id$years,
                 event = as.integer(pbc2.id$status != "alive"), drug = pbc2.id$drug)
all_forms <- list(bili = bili ~ year + age, alb = alb ~ year, pro = pro ~ year)
config_markers <- list(q4 = c("bili", "alb"), q6 = c("bili", "alb", "pro"))

make_config <- function(cf) {
  mk <- config_markers[[cf]]
  forms <- all_forms[mk]
  dls <- lapply(mk, function(m) {
    d <- pl[, unique(c("id", "year", all.vars(forms[[m]])))]
    d[stats::complete.cases(d), , drop = FALSE]
  })
  names(dls) <- mk
  list(markers = mk, forms = forms, rforms = stats::setNames(rep(list(~ year), length(mk)), mk),
       dls = dls, ds = ds, surv_formula = Surv(stime, event) ~ drug, time_var = "year",
       q = 2L * length(mk), block_sizes = rep(2L, length(mk)))
}

done <- read_done(CSV)
have <- function(cf, s, a) !is.null(done) && any(done$config == cf & done$seed == s & done$arm == a)
for (cf in CONFIGS) {
  D <- make_config(cf)
  cat(sprintf("\n== pbc2 %s: markers %s (q = %d)\n", cf, paste(D$markers, collapse = ", "), D$q))
  for (s in seq_len(NSEED)) for (a in ARMS) {
    if (have(cf, s, a)) next
    obj <- tryCatch(run_arm(D, a, s), error = function(e) {
      if (grepl("inspect|expected|not applied|reinstall", conditionMessage(e))) stop(e)
      cat("   ", cf, a, "seed", s, "FAILED:", conditionMessage(e), "\n"); NULL })
    if (is.null(obj)) next
    sc <- if (a == "J") NA_real_ else compile_time_jx(D, a)
    rows <- rows_of(obj, "pbc2", cf, D$q, s, a, sec_compile = sc)
    append_rows(rows, CSV)
    progress_line(rows, sprintf("%s seed %2d %s", cf, s, a))
  }
}

R <- read_done(CSV)
S <- fit_summary(R)
utils::write.csv(S, file.path(OUTDIR, "pbc2_summary.csv"), row.names = FALSE)
tables_md(S, "pbc2, several markers (paper-grade rerun)", file.path(OUTDIR, "pbc2_tables.md"))
P <- R[R$meets_bar & R$block %in% c("beta", "alpha", "sigma_e"), ]
if (nrow(P)) {
  E <- stats::aggregate(est ~ config + param + arm, data = P, FUN = stats::median)
  W <- stats::reshape(E, idvar = c("config", "param"), timevar = "arm", direction = "wide")
  utils::write.csv(W, file.path(OUTDIR, "pbc2_estimates.csv"), row.names = FALSE)
  cat("\nPosterior means (median over seeds; runs meeting the bar):\n")
  print(W, row.names = FALSE, digits = 4)
}
