# Code for "Rotating Away the Location Degeneracy"

This archive holds the code and results behind the article "Rotating Away
the Location Degeneracy: An Exact Reparameterization for Hamiltonian Monte
Carlo in Hierarchical and Joint Models".

- **`jmjax-0.3.0/`** is the source of the R package jmjax at release v0.3.0
  (DOI 10.5281/zenodo.22950084; https://github.com/jmjaxmaintainer-jpg/jmjax).
  It contains the package itself (`R/`, `tests/`, `vignettes/`) and, under
  `dev/`, the scripts and result files for every table and figure.
- **`results/`** holds two result files that the scripts write outside the
  repository.

## Setup

1. Install R and the R packages the scripts use: `reticulate`, `survival`,
   `nlme`, `JMbayes2`, `posterior` and `HSAUR3`.
2. Install jmjax from this folder: `R CMD INSTALL jmjax-0.3.0`.
3. Run `jmjax::jmjax_setup()` once in R. It creates a Python virtualenv,
   `r-jmjax`, with the pinned versions jax 0.4.30, jaxlib 0.4.30, numpyro
   0.15.0, numpy 1.26.4 and scipy 1.13.0.
4. For the Python scripts, use the same environment plus `matplotlib`.

Run every script from `jmjax-0.3.0/`. The R scripts write to
`~/Documents/R/jmjax_results` unless the output directory is set by an
environment variable. Each script's header gives its options and rough
cost.

## Where each result comes from

Paths are relative to `jmjax-0.3.0/`. Section, table and figure numbers
without an S are in the article; those with an S are in its supplement.

| Result | Script | Output used in the article |
|---|---|---|
| Figure 1; Section 4.4 numbers | `dev/theory_rotation_toy.py`, `dev/figures/make_figures.py` | printed; `dev/figures/` |
| Section S1 (Gaussian check) | `dev/theory_alpha_toy.py` | printed |
| Section 5, R² and its split | `dev/alpha_missing_information.R` | printed; uses `results/study_realdata_rotate.csv` |
| Figure 3 | `dev/figures/make_figures.py` | `dev/figures/` |
| Table 1 | `dev/pilot_rotate_grid.R` | `dev/pilot_rotate_grid.csv` |
| Section 6.3 (standard length) | `ROT_GRID=q1 Rscript dev/pilot_rotate_grid.R` | `dev/pilot_rotate_q1.csv` |
| Section 6.3 and S2 (runs to convergence) | `dev/study_q1_longrun.R` | `dev/study_q1_longrun.csv` |
| Section 6.4, Table S1 | `ROT_GRID=stress Rscript dev/pilot_rotate_grid.R` | `dev/pilot_rotate_stress.csv` |
| Figure 2 | `dev/figures/make_figures.py` | `dev/figures/` |
| Section 6.5, Table S2 | `dev/study_calibration.R` | `dev/calibration_results.csv` |
| α bias at N = 1200 (S4) | `dev/study_alpha_bias.R` | `dev/alpha_bias_results.csv` |
| Table 2 | `dev/generality/export_data.R`, then `dev/generality/mixed_models.py` | `dev/generality/results_orthodont_toenail_ext.csv` |
| Table 3 | `dev/study_realdata_rotate.R` | `results/study_realdata_rotate.csv` |
| Table 4; JMbayes2 in Figure 3 | `dev/study_common_ess.R` | `dev/common_ess.csv` |
| Section S5 | `dev/pilot_q1_intercept.R` | `results/pilot_q1_intercept.csv` |
| Section 5, α against JMbayes2 with 1–12 covariates | `dev/study_orth_vs_jmbayes2.R` | summarized in `vignettes/jmjax-reparameterization.Rmd`; the per-run file was not kept |
| Example 1, basis checks | `dev/verify_absorbable_basis.py` | printed |

## Notes

- **ESS and R-hat:** unless a script says otherwise, these come from
  NumPyro 0.15's estimators (multi-chain autocorrelation ESS with Geyer's
  initial monotone sequence and no rank normalization; split R-hat).
- **Timing:** wall-clock times depend on hardware. The article reports
  ratios between fits run on the same machine.
- **Fits from before the release:** the reported fits were run with the
  development versions that led to v0.3.0. The only sampler-side change
  since then adds a diagnostic check (Condition S on the rotation path) and
  does not alter any draws. Scripts that used the pre-release function
  `jm_fit_prefit()` now call it as `jmjax:::jm_fit_prefit()`, with the same
  arguments.
- **The other `dev/` scripts** are development diagnostics. They are not
  needed to reproduce the article.
