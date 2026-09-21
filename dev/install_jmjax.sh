#!/usr/bin/env bash
# =============================================================================
# Install jmjax from the source tree, and verify it actually works.
#
#   bash install_jmjax.sh                              # install what is already there
#   bash install_jmjax.sh path/to/some_delivery.zip    # unpack first, then install
#
# The source tree is ~/Documents/R/jmjax.
#
# UNPACKING IS NOW OPTIONAL. This script began as "unzip a delivery, then
# install it", and unpacking was mandatory. Once the package lives in git,
# that is the wrong default and actively dangerous: unzipping over a working
# tree silently overwrites whatever is checked out, including work not yet
# committed, and a stale zip can quietly revert files. With no argument it
# now installs the tree as it stands - so what gets installed is what
# `git status` says you have.
#
# WHY A SHELL SCRIPT AND NOT R. reticulate binds to one Python interpreter
# per R session, at the first Python call, and cannot be re-pointed
# afterwards. Python also caches imported modules in sys.modules, so an R
# session that has already imported jmjax_backend keeps the OLD
# mcmc_model.py no matter what you install over it - you reinstall, re-run,
# and see the identical bug.
#
# In RStudio that means restarting R twice, in the right places, every time.
# Each Rscript invocation below is a FRESH process - new R, new Python - so
# the problem cannot arise. That is the whole reason this is a shell script.
#
# It does NOT rebuild the Python virtualenv. Nothing about the dependencies
# has changed, and rebuilding costs a ~150MB download. Pass --venv if you
# genuinely need one (a new machine, or a broken environment).
# =============================================================================

set -euo pipefail

DEST="$HOME/Documents/R"
PKG="$DEST/jmjax"
REBUILD_VENV=0
ZIP=""
for a in "$@"; do
  case "$a" in
    --venv) REBUILD_VENV=1 ;;
    *.zip)  ZIP="$a" ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1/4  checking inputs"
command -v Rscript >/dev/null || { echo "   Rscript not on PATH."; exit 1; }
[ -d "$PKG" ] || { echo "   no source tree at $PKG"; exit 1; }
[ -f "$PKG/DESCRIPTION" ] || { echo "   $PKG has no DESCRIPTION"; exit 1; }
echo "   source : $PKG"
echo "   version: $(awk '/^Version:/{print $2}' "$PKG/DESCRIPTION")"

# Report the git state when there is one. Installing a tree with uncommitted
# changes is perfectly normal mid-development, but it should not be a
# surprise afterwards - "which code did I actually test" is not a question
# worth reconstructing later.
if git -C "$PKG" rev-parse --git-dir >/dev/null 2>&1; then
  echo "   branch : $(git -C "$PKG" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "   commit : $(git -C "$PKG" rev-parse --short HEAD 2>/dev/null)"
  _dirty="$(git -C "$PKG" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$_dirty" != "0" ]; then
    echo "   NOTE   : $_dirty file(s) modified but not committed - that is"
    echo "            what will be installed."
  fi
fi

if [ -n "$ZIP" ]; then
  say "2/4  unpacking $ZIP"
  [ -f "$ZIP" ] || { echo "   zip not found: $ZIP"; exit 1; }
  # Overwrites the working tree. Refuse when git says there is uncommitted
  # work to lose, since the zip cannot know about it.
  if git -C "$PKG" rev-parse --git-dir >/dev/null 2>&1 &&
     [ -n "$(git -C "$PKG" status --porcelain 2>/dev/null)" ]; then
    echo "   REFUSING: $PKG has uncommitted changes and unzipping would"
    echo "   overwrite them. Commit or stash first, or run without the zip"
    echo "   to install the tree as it stands."
    exit 1
  fi
  mkdir -p "$DEST"
  unzip -o -q "$ZIP" -d "$DEST"
  echo "   unpacked, $(find "$PKG" -name '*.R' | wc -l | tr -d ' ') R files"
else
  say "2/4  no zip given - installing the source tree as it stands"
fi

if [ "$REBUILD_VENV" = "1" ]; then
  say "2b/4  rebuilding the Python environment (slow, ~150MB)"
  Rscript -e 'jmjax::jmjax_setup(recreate = TRUE)' || {
    echo "   venv rebuild failed"; exit 1; }
fi

say "3/4  installing"
Rscript -e 'if (!requireNamespace("devtools", quietly = TRUE)) stop("install.packages(\"devtools\") first")' \
  || exit 1
# upgrade = FALSE, not "never": that string is remotes::'s convention and
# devtools::install() validates strictly against TRUE/FALSE/NA.
Rscript -e "devtools::install('$PKG', upgrade = FALSE)"

say "4/4  verifying"
# A FRESH process, so this imports the mcmc_model.py just installed.
#
# The delimiter is quoted ('REOF') so the shell leaves R's $ alone; the
# package path arrives as a command-line argument instead.
Rscript - "$PKG" <<'REOF'
suppressMessages({library(nlme); library(survival); library(jmjax)})
ok <- TRUE
fail <- function(...) { cat("   FAIL:", ..., "\n"); ok <<- FALSE }

# -- backend reachable. Deliberately the first Python touched: calling
# -- py_config() first would bind the interpreter and mask the very bug
# -- .get_backend() was rewritten to avoid.
b <- tryCatch({ jmjax:::.get_backend(); TRUE }, error = function(e) conditionMessage(e))
if (!isTRUE(b)) fail("Python backend unavailable -", b) else cat("   backend  : ok\n")

if (isTRUE(b)) {
  v <- reticulate::py_run_string("
import sys, jax, numpyro
_v = f\"python {sys.version.split()[0]} | jax {jax.__version__} | numpyro {numpyro.__version__} | x64 {jax.config.jax_enable_x64}\"
", convert = TRUE)
  cat("   versions :", v$`_v`, "\n")

  # -- precision. float64 is now the DEFAULT, so seeing float32 here means
  # -- something overrode it - JMJAX_ENABLE_X64=0 in the environment, or
  # -- another package importing jax before jmjax did. Either way the
  # -- numbers change, so this is a failure rather than a note.
  p <- tryCatch(jmjax:::.backend_precision(), error = function(e) NA_character_)
  if (identical(p, "float64")) {
    cat("   precision: ok (float64)\n")
  } else {
    fail("precision is", p, "- expected float64. Check JMJAX_ENABLE_X64 and",
         "whether another package imported jax first.")
  }

  # -- a real fit over the two paths that have broken before: the penalized
  # -- spline prior (no test coverage until recently, and unrunnable on the
  # -- pinned stack via numpyro's scan) and the lme warm start.
  set.seed(1); n <- 150L; vt <- seq(0, 10, length.out = 6L)
  b0 <- rnorm(n, 0, .8); x <- rnorm(n)
  Tt <- rexp(n, rate = pmin(pmax(exp(-2 + .4*(2 + .3*x + b0)), 1e-4), 5)/4)
  ot <- pmin(Tt, 10); r <- vapply(ot, function(o) max(1L, sum(vt <= o)), integer(1))
  ds <- data.frame(id = factor(1:n), time = ot, event = as.integer(Tt <= 10), x = x)
  dl <- data.frame(id = factor(rep(1:n, r), levels = 1:n),
                   time = unlist(lapply(r, function(k) vt[seq_len(k)])), x = rep(x, r))
  dl$y <- (2 + .3*dl$x + rep(b0, r)) + .5*dl$time + rnorm(nrow(dl), 0, .3)

  f <- tryCatch(
    jm_fit(y ~ time + x, Surv(time, event) ~ 1, dl, ds, "id", "time",
           method = "spline-PH-mcmc", random_effects = "intercept_slope",
           random_formula = ~ time,
           control = list(spline_prior = "penalized", num_warmup = 200,
                          num_samples = 200, num_chains = 2, seed = 1L,
                          progress_bar = FALSE)),
    error = function(e) e)

  if (inherits(f, "error")) {
    fail("penalized spline-PH-mcmc fit errored -", conditionMessage(f))
  } else {
    # tau_w exists ONLY under the penalized prior, so its presence proves
    # the path ran rather than silently falling back to the independent one.
    if (!"tau_w" %in% names(f$estimates)) fail("tau_w missing - penalized prior did not run")
    else cat("   penalized: ok (tau_w present)\n")

    ws <- f$convergence$warm_start
    if (is.null(ws)) {
      fail("warm start not reported - control$init_values never reached the backend")
    } else if (!isTRUE(as.logical(ws$used))) {
      fail("warm start built but REJECTED by the self-check (potential",
           signif(as.numeric(ws$potential_warm), 6), "vs",
           signif(as.numeric(ws$potential_uniform), 6), ")")
    } else {
      cat(sprintf("   warmstart: ok (%.0f log-density units better than a cold start)\n",
                  as.numeric(ws$potential_uniform) - as.numeric(ws$potential_warm)))
    }

    rh <- unlist(f$diagnostics$rhat); rh <- rh[is.finite(rh)]
    cat(sprintf("   fit      : max R-hat %.3f, %s leapfrog steps, %s divergences\n",
                max(rh),
                format(round(as.numeric(f$convergence$mean_num_steps))),
                { d <- f$convergence$n_divergences
                  if (is.null(d)) "?" else format(d) }))
  }
}

cat("\n")
if (ok) cat("   ALL CHECKS PASSED\n") else {
  cat("   SOME CHECKS FAILED - see above\n"); quit(status = 1)
}
REOF

say "done"
cat <<'EOT'
   Run the test suite with:
     Rscript -e 'devtools::test("~/Documents/R/jmjax")'

   float64 is now the DEFAULT - you no longer set JAX_ENABLE_X64 by hand.
   It measured 4.24x FASTER at n = 8,000, not slower, because float32
   gradient noise drove the sampler into four times as many leapfrog
   steps. To opt out (and accept approximate standard errors from the
   maximum-likelihood methods):
     Rscript -e 'jmjax::jmjax_setup(enable_x64 = FALSE)'   # before any fit
   or JMJAX_ENABLE_X64=0 in the environment.

   To measure the trade-off on this machine:
     bash dev/bench_x64.sh

   And for anything long-running, stop the Mac sleeping mid-run:
     caffeinate -i Rscript your_script.R 2>&1 | tee run.log
EOT
