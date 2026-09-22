# jmjax Test, Simulation, and Exploratory Script Archive

This folder collects the exploratory R scripts generated across multiple development sessions on the `jmjax` package - a roadmap of what was tested, what was found, and why, organized roughly chronologically and thematically. These are working/diagnostic scripts, not a polished test suite - several were one-off checks that led nowhere, several were superseded by later, corrected versions, and several fed directly into the package's actual test suite (`tests/testthat/`) or into the technical report in `13_reports/`.

**Confidence note:** Folders 06-13 document a single, continuous investigation (this session) that I have complete, detailed context for. Folders 01-05 are from earlier development sessions (file dates Sept 5-10); descriptions there are reconstructed from filenames, script content, and general session context rather than a live memory of running them - treat those descriptions as a reasonable best-effort guide, not a certain record.

For the full narrative writeup of the most important findings (folders 06-12), see `13_reports/jmjax_mcmc_performance_investigation_report.md` (or the `.docx` version) - that document is the definitive account; this archive is the supporting evidence and code trail behind it.

---

## 01_early_development_and_validation

Early correctness-focused work: validating jmjax's spline knot placement, log-density computations, and MLE estimates directly against the reference `JM` package, and diagnosing specific numerical issues encountered during development.

- `extract_r_baseline_hazard.R` - extracts baseline hazard values from a reference `JM` fit for direct numerical comparison against jmjax's own computation.
- `knot_placement_replication_check.R`, `final_knot_resolution_test.R` - validate that jmjax's B-spline knot placement matches `JM`'s exactly (a knot-placement convention mismatch was an early, confirmed bug - these scripts document its resolution).
- `spline_q1_isolation_test.R` - isolates spline-baseline behavior at `q=1`, related to the confirmed finding that the `delta` functional form is non-identifiable at `q=1` with a flexible spline baseline.
- `rw2_noncentered_test.R` - tests the non-centered reparameterization of the RW2 (second-order random walk) spline-smoothing prior, addressing a funnel-geometry problem in the *centered* version.
- `rigorous_log_density_check.R` - a thorough, direct check of jmjax's log-density computation against an independent reference.
- `weibull_baseline_crosscheck.R`, `weibull_q2_mle_crosscheck.R`, `jm_mle_crosscheck.R` - direct MLE parameter cross-checks against `JM::jointModel()` for the Weibull baseline, including the `q=2` (random intercept + slope) case.
- `diagnose_intercept_slope_bias.R`, `diagnose_n1500_intercept_slope.R` - diagnose a suspected bias specific to the intercept+slope (`q=2`) parameterization, including at a specific larger sample size.
- `diagnose_mode_trapping.R` - diagnoses the MLE optimizer converging to a local rather than global mode.
- `rep6_fresh_check.R`, `rep6_profile_likelihood_check.R` - deep-dive on a specific anomalous replicate ("rep 6") from a validation sweep, using profile likelihood to understand its behavior.
- `n200_penalized_prior_check.R`, `n500_replication_check.R` - replication checks of core estimator behavior at specific sample sizes.
- `benchmark_step1_diagnostic.R` - an early diagnostic step in what became the benchmark development in folder 02.

## 02_original_benchmark_development

The iterative development of the simulated-data MCMC benchmark that ultimately produced the validated **"jmjax is ~12-13x more sample-efficient than JMbayes2"** result (spline baseline, `q=2`, no covariates) - later confirmed to reproduce cleanly (folder 07) and to be the foundation the whole efficiency investigation (folders 06-12) measured itself against.

- `benchmark_sweep.R` -> `benchmark_sweep_final.R` -> `benchmark_sweep_replicated.R` -> `benchmark_sweep_replicated_original.R` - successive refinements: adding proper replication (5 reps per scenario), confirming basis-function matching between packages, and arriving at the final, validated design. `benchmark_sweep_replicated_original.R` is the exact script later re-run (folder 07) to rule out a code regression as the explanation for an apparent efficiency loss.

## 03_functional_forms_testing

Validation of the `delta`/`area`/`area_avg` functional-form channels (alternative ways a longitudinal trajectory can enter the survival submodel, beyond its current value) against `JMbayes2`.

- `area_avg_vs_jmbayes2_benchmark.R`, `area_avg_q2_vs_jmbayes2_benchmark.R` - validate the `area_avg` channel at `q=1` and `q=2`.
- `area_quick_check.R` - a faster, lighter-weight check of the `area` channel.
- `spline_delta_debug.R` - debugging session for the `delta` channel under a spline baseline (related to the `q=1` non-identifiability finding in folder 01).

## 04_pbc2_real_data_examples

Early, exploratory use of the real PBC2 clinical dataset (from `JMbayes2`) - establishing the data-prep pattern (rescaling `year`/`years` by 12 for numerical stability, defining the composite death-or-transplant event) and basic model-fitting examples, predating the later performance investigation.

- `jmjax_pbc2_examples.R`, `jmjax_usage_examples.R` - general usage/worked examples.
- `jmjax_pbc2_covariates_v2.R` -> `jmjax_pbc2_covariates_v3_q2.R` - iterative development of a PBC2 model including baseline covariates (`age`, `drug`) and `q=2` random effects - `v3_q2` is the version whose exact model specification the later performance comparisons (folders 06, 12) replicate.
- `jmbayes2_pbc2_q2_crosscheck.R` - the corresponding `JMbayes2` cross-check on the same data/model.

## 05_original_performance_comparisons

The original jmjax-vs-JMbayes2 real-data speed comparisons that first suggested a possible performance regression - the trigger for the entire investigation in folders 06-12.

- `jmjax_vs_jmbayes2_performance.R` - Weibull-baseline PBC2 comparison.
- `jmjax_vs_jmbayes2_spline_performance.R` - spline-baseline PBC2 comparison (this is the specific comparison later found to show near-parity with `JMbayes2`, motivating the deep-dive investigation).
- `jmjax_no_eb_prior_comparison.R` - checks whether jmjax's empirical-Bayes prior centering feature itself affects performance.

---

## 06_pbc2_slowdown_investigation

**The core investigation**, prompted by real PBC2 data showing near-parity with `JMbayes2` where simulated data had shown a large jmjax advantage. Each script isolates one candidate explanation, tested and either confirmed or ruled out:

- `reproduce_pbc2_slowdown.R` - first attempt to reproduce the real-data slowdown using realistic (PBC2-fitted) parameter values in simulation.
- `covariate_ab_benchmark.R` - does adding *any* covariate slow things down? (Modest effect confirmed, ~7-17%.)
- `covariate_scale_benchmark.R` - does the *raw scale* of a covariate (e.g., `age`'s mean~50 vs a standardized variable) matter, independent of just having one? (Initially found no effect - later revisited and reversed once other fixes were in place; see folder 11.)
- `sigma_b1_ratio_sweep.R`, `beta1_magnitude_sweep.R` - test whether the ratio of random-slope variability to the population slope (`sigma_b1/beta1`), or `beta1`'s absolute magnitude, drives the slowdown. (Ratio conjecture not supported by a clean trend; magnitude sweep found a real but partial effect.)
- `time_scale_rescale_test.R` - tests whether the compressed time range in real data (year/12 rescaling) versus the wider range used in simulation explains the gap. (Modest effect, not the primary driver.)
- `effect_size_no_covariates_test.R` - isolates whether realistic *effect sizes* alone (without covariates) reproduce the slowdown. (They did not - pointed the investigation toward measurement rather than modeling.)

## 07_timing_bug_diagnosis

**The critical discovery**: the apparent slowdown was substantially a measurement artifact, not a real modeling effect. These scripts trace the diagnosis from first suspicion to root cause and fix.

- `alpha_vs_min_ess_diagnostic.R`, `alpha_vs_min_ess_diagnostic_v2.R` - first evidence that "min ESS across all parameters" and "alpha-specific ESS" (the metric the original 12-13x benchmark actually used) give very different numbers - a metric-inconsistency confound.
- `alpha_corrected_sampling_time.R` - corrects the metric but still finds a large, unexplained gap - escalating the investigation.
- `chase_compile_overhead.R`, `verify_compilation_overhead.R` - identifies that `sampling_time_sec` was being measured before JAX's asynchronous computation actually finished (the true root cause: **`jax.block_until_ready()`** was missing before the timer stopped).
- `test_lme_prefit_timing.R` - rules out the internal empirical-Bayes `lme()` pre-fit as a timing confound (it's fast, ~0.08s).
- `test_summary_jit_hypothesis.R` - tests (and refutes) a JIT-recompilation-per-shape hypothesis for a separate, related slowdown in `numpyro_summary()`.
- `verify_numpyro_summary_fix.R` - confirms the fix (excluding per-subject `b`/`b_std` sites from summary diagnostics, plus the `block_until_ready()` fix) restores the original benchmark's numbers.

## 08_mle_scaling_tests

Separate from the MCMC timing investigation: characterizes how jmjax's **maximum-likelihood** (adaptive Gauss-Hermite) methods scale with sample size relative to `JM`, finding a favorable and *growing* advantage as `n` increases (up to ~12x faster at `n=20,000`) - a structurally different result from the MCMC scaling story, attributable to full marginalization of random effects.

- `jmjax_vs_JM_spline_mle.R` - initial small-`n` spline MLE comparison.
- `scaling_test_weibull_spline.R` - the `n = 1,000` to `20,000` scaling sweep for both baseline hazards.
- `mcmc_scaling_test.R` - the analogous scaling test for MCMC, finding a stable (not widening) efficiency multiplier instead.

## 09_random_effects_and_rw2_optimization

**Major fixes**: replacing the LKJCholesky-based random-effects correlation structure with a Wishart-conjugate Gibbs update (matching `JMbayes2`'s own documented approach), and replacing the sequential RW2 spline-smoothing construction with a mathematically-equivalent, vectorized closed-form version.

- `dense_mass_comparison.R` - tests (and finds counterproductive) a *global* dense mass matrix for NUTS.
- `correlated_vs_independent_re.R` - first evidence that a simpler, uncorrelated random-effects model is both faster and better-converging than the correlated one.
- `wishart_gibbs_truth_recovery.R` - validates the new Wishart-Gibbs implementation recovers correct parameters before trusting any speed claim.
- `three_way_random_effects_comparison.R` - correlated vs. independent vs. Wishart-Gibbs, head to head.
- `jmjax_vs_jmbayes2_wishart_gibbs_single_run.R` - first real jmjax-vs-JMbayes2 comparison using the new method.
- `vectorized_rw2_comparison.R` - validates the closed-form RW2 construction against the original sequential scan.
- `dense_mass_spline_diagnosis.R` - diagnoses (via direct leapfrog step-count measurement) that the penalized spline prior needs a *targeted*, not global, dense mass matrix - and confirms the fix.
- `final_penalized_comparison.R` - the fully-fixed penalized-spline jmjax vs. `JMbayes2` comparison.

## 10_beta0_coupling_investigation

Diagnoses and experimentally addresses a residual inefficiency (the population intercept `beta_0` consistently mixing worse than any other parameter) left over after the Section 09 fixes - implemented as a fully separate, opt-in experimental feature (`wishart_gibbs_centered`) given a real limitation was found during validation.

- `beta0_random_effects_coupling_check.R` - confirms a near-perfect (-0.95) posterior correlation between `beta_0` and the mean random intercept, diagnosing a classic location-degeneracy.
- `wishart_gibbs_centered_validation.R` - validates the sum-to-zero-constrained fix at `n=500` (successful: `beta_0` ESS improved ~4.8x).
- `wishart_gibbs_centered_n1500_check.R` - the same validation at `n=1500`, which surfaced a credible-interval miscalibration.
- `beta0_bias_multiseed_check.R` - 5-seed replication confirming the miscalibration is real (not single-seed noise) but that point estimates remain unbiased.
- `beta0_multichain_rhat_check.R` - rules out insufficient MCMC exploration as the cause (multi-chain R-hat is excellent); the miscalibration is a genuine statistical property of the hard constraint, not a sampling artifact.

## 11_covariate_scale_investigation

A second, later covariate-scale test - re-running the earlier (folder 06) null result with all of folder 09's fixes applied, confirming that a real effect had been masked by the larger bottlenecks that existed at the time of the original test. Also revealed that jmjax's efficiency advantage over `JMbayes2` **depends on** covariate scale, reversing for unstandardized covariates - traced to `JMbayes2`'s MLE-pre-fit-informed proposal covariances (see the report, Section 7).

- `covariate_scale_benchmark_v2_with_fixes.R` - the confirming rerun (paired t-test now significant, `p<0.02`, where the original was `p=0.16`).

## 12_final_validated_comparisons

The final, fixed-configuration comparisons against real PBC2 data and against `JM`'s MLE.

- `pbc2_real_data_updated_comparison.R` - real PBC2 data, spline-PH-mcmc, all validated fixes applied (`wishart_gibbs`, vectorized RW2, targeted dense mass, genuine parallel chains, `progress_bar=FALSE`) - the headline confirmed result (~4.8x more sample-efficient than `JMbayes2`).
- `pbc2_mle_comparison.R` - the companion MLE (`weibull-PH-aGH`) comparison against `JM` on the same real data.

## 13_reports

- `jmjax_mcmc_performance_investigation_report.md` / `.docx` - the full technical writeup of folders 06-12's findings: seven confirmed root causes, one methodological correction, and two follow-up investigations (the `beta_0` coupling and covariate-scale findings), with recommendations for future work.
- `README.md`, `NEWS.md` - the package's own top-level documentation, included here for reference/context.
