# ==============================================================================
# dev/bench_multi_pbc2.R - jmjax 0.5.0 vs JMbayes2, several markers on pbc2
# (Milestone 1, Phase C). Shared code: dev/bench_multi_common.R.
#
# CONFIGS (random intercept and slope in year for every marker):
#   q4  log(serBilir) ~ year + age,  albumin ~ year
#   q6  log(serBilir) ~ year + age,  albumin ~ year,  prothrombin ~ year
# Survival: Surv(years, death or transplant) ~ drug, as in the paper's
# single-marker pbc2 study; spline baseline (5 interior knots, penalized,
# JMbayes2: cubic B-splines on 6 segments). Value association per marker.
# Each marker uses its own complete rows (prothrombin has missing values),
# so the markers are observed at different times.
#
# ARMS: A jmjax unrotated | R jmjax rotated + dense (beta, b_gen_U) block |
#       J JMbayes2 (full D across markers, its default).
#
# OUTPUT (in BM_OUTDIR, default ~/Documents/R/jmjax_results):
#   multi_pbc2.csv            one row per config x seed x arm x parameter
#   multi_pbc2_draws/*.rds    raw draws, so other estimators can be applied
#   multi_pbc2_summary.csv    one row per config x seed x arm
#   multi_pbc2_tables.md      the tables for the paper chat
#
# USAGE (resumable; rerun to continue after an interruption):
#   caffeinate -i Rscript dev/bench_multi_pbc2.R
#   BM_CONFIGS=q4 BM_SEEDS=1 BM_ARMS=A,R Rscript dev/bench_multi_pbc2.R   # quick look
# Defaults: 3 seeds, 4 chains x (1000 warm-up + 1000 draws); JMbayes2 16000
# iterations with 4000 burn-in per chain, as in the single-marker study.
# ==============================================================================

local({
  here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
  if (is.null(here)) {
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "dev"
  }
  source(file.path(here, "bench_multi_common.R"), local = globalenv())
})
if (!requireNamespace("JMbayes2", quietly = TRUE)) stop("JMbayes2 is required", call. = FALSE)

CONFIGS <- .envl("BM_CONFIGS", c("q4", "q6"))
ARMS    <- .envl("BM_ARMS", c("A", "R", "J"))
NSEED   <- .envi("BM_SEEDS", 3L)
CHAINS  <- .envi("BM_CHAINS", 4L)
WARMUP  <- .envi("BM_WARMUP", 1000L)
SAMPLES <- .envi("BM_SAMPLES", 1000L)
JB_ITER <- .envi("BM_JB_ITER", 16000L)
JB_BURN <- .envi("BM_JB_BURN", 4000L)
OUTDIR  <- Sys.getenv("BM_OUTDIR", path.expand("~/Documents/R/jmjax_results"))
DRAWDIR <- file.path(OUTDIR, "multi_pbc2_draws")
dir.create(DRAWDIR, showWarnings = FALSE, recursive = TRUE)
CSV <- file.path(OUTDIR, "multi_pbc2.csv")

# ---- data ----------------------------------------------------------------------
utils::data("pbc2", package = "JMbayes2", envir = environment())
utils::data("pbc2.id", package = "JMbayes2", envir = environment())
pl <- data.frame(id = as.integer(as.character(pbc2$id)), year = pbc2$year,
                 age = pbc2$age, bili = log(pbc2$serBilir), alb = pbc2$albumin,
                 pro = pbc2$prothrombin)
ds <- data.frame(id = as.integer(as.character(pbc2.id$id)), stime = pbc2.id$years,
                 event = as.integer(pbc2.id$status != "alive"), drug = pbc2.id$drug)
all_forms <- list(bili = bili ~ year + age, alb = alb ~ year, pro = pro ~ year)
config_markers <- list(q4 = c("bili", "alb"), q6 = c("bili", "alb", "pro"))
surv_formula <- Surv(stime, event) ~ drug

make_config <- function(cf) {
  mk <- config_markers[[cf]]
  forms <- all_forms[mk]
  rforms <- stats::setNames(rep(list(~ year), length(mk)), mk)
  dls <- lapply(mk, function(m) {
    v <- c("id", "year", all.vars(forms[[m]]))
    d <- pl[, unique(v)]; d[stats::complete.cases(d), , drop = FALSE]
  })
  names(dls) <- mk
  list(markers = mk, forms = forms, rforms = rforms, dls = dls, q = 2L * length(mk),
       block_sizes = rep(2L, length(mk)))
}

# ---- run -------------------------------------------------------------------------
done <- read_done(CSV)
have <- function(cf, s, a) !is.null(done) && any(done$config == cf & done$seed == s & done$arm == a)
for (cf in CONFIGS) {
  C <- make_config(cf)
  cat(sprintf("\n== pbc2 %s: markers %s (q = %d); rows per marker %s\n", cf,
              paste(C$markers, collapse = ", "), C$q,
              paste(vapply(C$dls, nrow, 1L), collapse = "/")))
  comp <- list()
  for (a in intersect(ARMS, c("A", "R"))) {
    comp[[a]] <- compile_time_jx(C$forms, C$rforms, C$dls, ds, surv_formula, "year", a, CHAINS)
    cat(sprintf("   compilation estimate %s: %.1fs\n", a, comp[[a]]))
  }
  for (s in seq_len(NSEED)) for (a in ARMS) {
    if (have(cf, s, a)) next
    obj <- tryCatch({
      if (a == "J") {
        draws_jb(fit_jb(C$forms, C$rforms, C$dls, ds, surv_formula, "year", s, CHAINS,
                        JB_ITER, JB_BURN), C$markers, C$block_sizes)
      } else {
        f <- fit_jx(C$forms, C$rforms, C$dls, ds, surv_formula, "year", a, s,
                    CHAINS, WARMUP, SAMPLES)
        if (a == "R" && !isTRUE(f$convergence$orthogonalize$rotation$applied))
          stop("arm R: rotation not applied", call. = FALSE)
        draws_jx(f, C$markers, C$block_sizes)
      }
    }, error = function(e) {
      if (grepl("inspect|expected|rotation not applied", conditionMessage(e))) stop(e)
      cat("   ", cf, a, "seed", s, "FAILED:", conditionMessage(e), "\n"); NULL })
    if (is.null(obj)) next
    saveRDS(c(obj, list(arm = a, config = cf, seed = s)),
            file.path(DRAWDIR, sprintf("%s_%s_seed%d.rds", cf, a, s)))
    rows <- ess_rows(obj, "pbc2", cf, C$q, s, a,
                     sec_compile = if (a == "J") NA_real_ else comp[[a]])
    append_rows(rows, CSV)
    rot <- obj$rotation
    cat(sprintf("   %s seed %d %s: %7.1fs | min ESS/s beta %.2f alpha %.2f Sigma %.2f | max R-hat %.3f%s%s\n",
                cf, s, a, obj$sec,
                min(rows$ess_bulk[rows$block == "beta"]) / obj$sec,
                min(rows$ess_bulk[rows$block == "alpha"]) / obj$sec,
                min(rows$ess_bulk[rows$block == "Sigma"]) / obj$sec,
                max(rows$rhat_posterior),
                if (is.finite(obj$n_div)) sprintf(" | %d divergences", as.integer(obj$n_div)) else "",
                if (!is.null(rot)) sprintf(" | k = %d, dense block %s", as.integer(rot$k),
                                           format(rot$dense_block_size %||% NA)) else ""))
  }
}

# ---- tables ------------------------------------------------------------------------
R <- read_done(CSV)
S <- fit_summary(R)
utils::write.csv(S, file.path(OUTDIR, "multi_pbc2_summary.csv"), row.names = FALSE)
tables_md(S, "pbc2, several markers: jmjax 0.5.0 vs JMbayes2",
          file.path(OUTDIR, "multi_pbc2_tables.md"))

# Posterior means side by side (seed 1), for a sanity check that the arms agree.
P1 <- R[R$seed == min(R$seed) & R$block %in% c("beta", "alpha", "sigma_e"), ]
if (nrow(P1)) {
  W <- stats::reshape(P1[, c("config", "param", "arm", "est")], idvar = c("config", "param"),
                      timevar = "arm", direction = "wide")
  utils::write.csv(W, file.path(OUTDIR, "multi_pbc2_estimates.csv"), row.names = FALSE)
  cat("\nPosterior means (first seed):\n"); print(W, row.names = FALSE, digits = 4)
}
