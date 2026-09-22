#!/usr/bin/env bash
# =============================================================================
# Verify the absorbable-basis repair end to end.
#
#   bash dev/check_orth_repair.sh
#
# Checks, in increasing order of cost, so a failure stops you before the
# expensive part:
#
#   1. undefined names      (<1s, no R)      dev/check_undefined_names.py
#   2. the algebra          (~2s, no R)      dev/verify_absorbable_basis.py
#   3. the install carried  (~1s)            _absorbable_basis_exact present
#      it                                     in the INSTALLED copy
#   4. the R test suite     (~2-5 min)       tests/testthat/test-orthogonalize.R
#
# WHY EACH STEP IS ITS OWN PROCESS. Same reason install_jmjax.sh is a shell
# script: reticulate binds one Python interpreter per R session at the first
# Python call, and Python caches imported modules in sys.modules. An R
# session that has already imported jmjax_backend keeps the OLD mcmc_model.py
# no matter what you install over it. Every step below is a fresh process, so
# that cannot happen.
#
# WHY STEP 2 EXISTS AT ALL. inst/python/jmjax_backend/mcmc_model.py is COPIED
# into the installed package by devtools::install(). Editing the source tree
# changes nothing until you reinstall, and the failure is silent - the tests
# pass or fail against whatever was installed last, not against what you just
# wrote. Step 2 is the one-line check that catches it.
# =============================================================================

set -euo pipefail

# NOTE: step 2 shells out to dev/install_jmjax.sh, which targets
# ~/Documents/R/jmjax itself. Pointing PKG elsewhere changes which tree
# steps 1 and 3 use, not which one gets installed.
PKG="${1:-$HOME/Documents/R/jmjax}"
VENV="r-jmjax"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31m   FAILED: %s\033[0m\n\n' "$1"; exit 1; }

[ -d "$PKG" ] || die "no source tree at $PKG"
cd "$PKG"

# -----------------------------------------------------------------------
say "1/4  undefined names: dev/check_undefined_names.py"
# py_compile does NOT catch a name that is read but never bound - that is a
# runtime NameError. One shipped in commit b564889 (orthogonalize_report was
# computed in fit_nuts() but read in _package_result(), breaking EVERY MCMC
# fit) and survived a clean py_compile. This stage is <1s and would have
# caught it.
for _f in inst/python/jmjax_backend/*.py; do
  python3 dev/check_undefined_names.py "$_f" >/tmp/jmjax_undef.log 2>&1 || {
    cat /tmp/jmjax_undef.log
    die "undefined name(s) in $_f - this WILL raise NameError at runtime"
  }
done
echo "   no undefined names in any backend module"

say "2/4  algebra: dev/verify_absorbable_basis.py"

# Prefer the jmjax venv's interpreter: it has jax/numpyro, so the script
# imports the genuinely shipped module object rather than falling back to
# extracting the functions from source. Either mode is a real check; the
# import mode additionally proves the module still imports cleanly.
PY="$(Rscript -e 'cat(tryCatch(reticulate::virtualenv_python("r-jmjax"), error = function(e) ""))' 2>/dev/null || true)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || die "no usable Python found (tried the $VENV venv and python3)"
  echo "   python : $PY  (venv not found - will source-extract)"
else
  echo "   python : $PY"
fi

# A stale .pyc next to a modified .py is normally invalidated by mtime+size,
# but it costs nothing to remove the ambiguity.
rm -rf inst/python/jmjax_backend/__pycache__

"$PY" dev/verify_absorbable_basis.py || die "the basis construction is wrong - do NOT install this"

# -----------------------------------------------------------------------
say "3/4  installing, then checking the install actually carried the change"
bash dev/install_jmjax.sh >/tmp/jmjax_install.log 2>&1 || {
  tail -40 /tmp/jmjax_install.log
  die "install failed - full log in /tmp/jmjax_install.log"
}
# install_jmjax.sh ends with a long boilerplate footer, so `tail` shows the
# footer rather than the verification result. Pull the lines that matter.
grep -E "^   (backend|versions|precision|penalized|warmstart|fit|ALL CHECKS|SOME CHECKS)" \
  /tmp/jmjax_install.log || tail -5 /tmp/jmjax_install.log

INSTALLED="$(Rscript -e 'cat(system.file("python", "jmjax_backend", "mcmc_model.py", package = "jmjax"))')"
[ -n "$INSTALLED" ] && [ -f "$INSTALLED" ] || die "cannot locate the installed mcmc_model.py"
echo "   installed backend: $INSTALLED"
grep -q "_absorbable_basis_exact" "$INSTALLED" \
  || die "the installed copy has no _absorbable_basis_exact - the edit did not reach it"
grep -q "_structural_extension_residual" "$INSTALLED" \
  || die "the installed copy has no _structural_extension_residual"
echo "   repair present in the installed copy: ok"

# -----------------------------------------------------------------------
say "4/4  R test suite: test-orthogonalize.R"
# devtools::test() sets NOT_CRAN=true, so skip_if_slow_mcmc() does NOT skip
# here - these are real fits and take a few minutes. caffeinate keeps the
# Mac awake for them.
NOSLEEP=""
command -v caffeinate >/dev/null && NOSLEEP="caffeinate -i" || true
$NOSLEEP Rscript -e "devtools::test('$PKG', filter = 'orthogonalize')" \
  || die "test-orthogonalize.R failed - see output above"

printf '\n\033[1;32m   ALL FOUR STAGES PASSED\033[0m\n\n'
cat <<'EOT'
   What this did and did not establish:

     established  - the basis construction is algebraically sound and
                    complete, the repair reached the installed package, and
                    the R-visible contract (fit$convergence$orthogonalize,
                    estimate agreement with the default fit, the beta_1 ESS
                    guard) still holds.

     NOT established - posterior calibration. Nothing here checks whether
                    credible intervals under the reparameterized fit have
                    correct coverage; that is the open item in Section 9 of
                    vignette("jmjax-reparameterization"), and it is the
                    reason the option is still opt-in and experimental.
EOT
