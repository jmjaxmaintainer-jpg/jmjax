# ==============================================================================
# Main user-facing fitting function.
# ==============================================================================

#' Fit a shared-parameter joint model for longitudinal and survival data
#'
#' Fits a linear mixed-effects longitudinal submodel jointly with a
#' relative-risk survival submodel (Weibull or spline-approximated baseline
#' hazard), linked through a shared random effect. Four fitting methods are
#' available, combining two baseline hazard choices (Weibull, spline) with
#' two estimation approaches (adaptive Gauss-Hermite MLE, full Bayesian
#' NUTS) - see \code{method} below.
#'
#' @param long_formula Fixed-effects formula for the longitudinal submodel,
#'   e.g. \code{y ~ time}, optionally including time-constant baseline
#'   covariates such as \code{y ~ time + age}. Longitudinal-side covariates
#'   are supported for all four methods; for the two MLE methods
#'   (\code{"weibull-PH-aGH"}, \code{"spline-PH-aGH"}) they cannot yet be
#'   combined with \code{functional_forms} beyond the default value-only
#'   association, and at \code{q = 2} an explicit \code{random_formula} is
#'   required so that the random-effects design does not depend on column
#'   order.
#' @param surv_formula A \code{survival::Surv()} formula, e.g.
#'   \code{Surv(time, event) ~ 1} (no baseline covariates) or
#'   \code{Surv(time, event) ~ age + sex} (with baseline covariates -
#'   currently supported for \code{method = "weibull-PH-aGH"} or
#'   \code{"spline-PH-aGH"}, either with \code{random_effects} of
#'   \code{"intercept"} or \code{"intercept_slope"}, with no
#'   \code{functional_forms} beyond the default value-only association;
#'   other combinations raise a clear error rather than silently dropping
#'   the covariates - both MCMC methods support baseline covariates at any
#'   \code{q} and with any \code{functional_forms} channel). Must be
#'   fit-able by \code{coxph()} - see Details for why \code{survreg()}-
#'   style AFT specs are not supported.
#' @param data_long Long-format data frame for the longitudinal submodel.
#' @param data_surv One-row-per-subject data frame for the survival submodel.
#' @param id_var Name of the subject id column, present in both data frames.
#' @param time_var Name of the time variable used in long_formula (needed to
#'   evaluate the fixed-effects design at survival/quadrature times for the
#'   shared m_i(t) term - see \code{build_time_design()}).
#' @param method One of \code{"weibull-PH-aGH"} (fast, closed-parametric
#'   baseline hazard, adaptive Gauss-Hermite MLE - good for initial values
#'   and quick benchmarking), \code{"spline-PH-aGH"} (flexible
#'   spline-approximated baseline hazard, adaptive Gauss-Hermite MLE),
#'   \code{"spline-PH-mcmc"} (full Bayesian NUTS fit of the spline model;
#'   supports multivariate random effects, unlike the two MLE methods), or
#'   \code{"weibull-PH-mcmc"} (full Bayesian NUTS fit with the SAME
#'   closed-form Weibull baseline hazard as \code{"weibull-PH-aGH"}, added
#'   to isolate whether a discrepancy is about the spline/quadrature
#'   machinery specifically or something more fundamental, by removing
#'   splines from the picture entirely).
#' @param random_effects \code{"intercept"} (default) for a single random
#'   intercept, or \code{"intercept_slope"} for a correlated random
#'   intercept and slope on \code{time_var}. Ignored (with the actual
#'   random-effects dimension taken from \code{random_formula} instead) if
#'   \code{random_formula} is supplied. \code{"spline-PH-aGH"} supports
#'   both.
#' @param random_formula Optional one-sided formula for an INDEPENDENT
#'   random-effects design, e.g. \code{~ time} or \code{~ 1}. If
#'   \code{NULL} (default), the random-effects design \code{Z} is derived
#'   by taking the first \code{q} columns of the fixed-effects design
#'   built from \code{long_formula} (\code{q} determined by
#'   \code{random_effects}) - this is only correct when \code{long_formula}
#'   is itself linear in \code{time_var} (e.g. \code{y ~ time}), since it
#'   silently assumes the first columns of \code{X} ARE "intercept" and
#'   "slope". For any other \code{long_formula} (e.g. \code{y ~ time +
#'   I(time^2)} or \code{y ~ splines::ns(time, 3)}), supply
#'   \code{random_formula} explicitly so \code{Z} is evaluated
#'   independently and correctly, the same way \code{X} is evaluated from
#'   \code{long_formula}.
#' @param init_theta Optional named list of initial values, e.g. from a
#'   prior \code{method = "weibull-PH-aGH"} fit's \code{$estimates}, used to
#'   warm-start a spline or MCMC fit.
#' @param functional_forms Optional one-sided formula selecting which
#'   features of the longitudinal trajectory enter the hazard, e.g.
#'   \code{~ value(y)} (the default when \code{NULL}),
#'   \code{~ value(y) + delta(y)}, \code{~ value(y) + area(y)} or
#'   \code{~ value(y) + area_avg(y)}. Each term contributes its own
#'   association coefficient.
#' @param control List of backend-specific control arguments (n_interior_knots,
#'   n_gh_nodes, maxiter, num_warmup, num_samples, num_chains, seed, ...).
#'   For spline methods, also \code{knot_placement}: \code{"quantile"}
#'   (default) matches \code{JM}'s scheme (knots at quantiles of observed
#'   times); \code{"equal"} matches \code{JMbayes2}'s scheme (equally
#'   spaced knots across a padded time range) - see
#'   \code{build_spline_knots()} for the exact confirmed formulas.
#'   For \code{method = "spline-PH-mcmc"}, also: \code{spline_prior}
#'   (\code{"independent"} [default] or \code{"penalized"} - a P-spline/RW2
#'   prior on the baseline hazard spline coefficients, matching JMbayes2's
#'   approach); \code{empirical_bayes_prior} (default \code{TRUE} - centers
#'   the priors for \code{beta} and the random-effects SDs on an internally
#'   fit \code{nlme::lme()} model's MLEs, matching JMbayes2's confirmed
#'   default behavior; set \code{FALSE} for jmjax's original uninformative
#'   priors instead).
#'
#' @section Performance-tuning options:
#'   For \code{method = "spline-PH-mcmc"}/\code{"weibull-PH-mcmc"} at
#'   \code{q = 2} only. A set of options added following a thorough
#'   performance investigation - see \code{vignette("jmjax-validation")},
#'   section "Performance tuning", for the full derivations, benchmarks,
#'   and honest caveats behind each one. None change the statistical
#'   model being fit (truth-recovery validated for each); they change
#'   only how efficiently NUTS explores it.
#'   \describe{
#'     \item{\code{progress_bar}}{Default \code{TRUE} (NumPyro's own
#'       default). \strong{Recommend setting \code{FALSE}} for any
#'       performance-sensitive fit: the console progress bar was found to
#'       add roughly 30x overhead under R/reticulate relative to native
#'       Python execution. This affects every fit regardless of the other
#'       options below and is usually the single largest lever available.}
#'     \item{\code{random_effects_method}}{One of \code{"nuts"} (default -
#'       the random-effects correlation is explored jointly with everything
#'       else via an LKJCholesky-parameterized Cholesky factor and full
#'       gradient-based HMC) or \code{"wishart_gibbs"} (replaces this with
#'       a closed-form Wishart-conjugate Gibbs update for the
#'       random-effects covariance matrix, hybridized with NUTS for all
#'       other parameters via \code{numpyro.infer.HMCGibbs}; validated
#'       faster and better-converging on the configurations tested).
#'       \strong{Important:} \code{"wishart_gibbs"} also CHANGES THE PRIOR
#'       on the random-effects covariance - it replaces the default's
#'       separate Gamma priors on the standard deviations (centered on an
#'       internal \code{lme()} fit) and LKJ prior on the correlation with a
#'       single Wishart prior on the precision matrix, whose default scale
#'       (\code{S0 = I}) does NOT adapt to the data. The inverse-Wishart
#'       family is documented to bias variances upward and correlations
#'       toward zero when the true variance is small relative to the prior
#'       mean, with the bias persisting at large \code{n}. Set
#'       \code{wishart_eb_scale = TRUE} (below) to remove that mismatch.
#'       Note that jmjax's DEFAULT (\code{"nuts"}) path is the one that
#'       closely matches \code{JMbayes2}'s own prior specification, not
#'       this option. A third, \strong{experimental} value,
#'       \code{"wishart_gibbs_centered"}, additionally enforces an exact
#'       sum-to-zero constraint on the random intercepts to address a
#'       separate, smaller mixing inefficiency in the population intercept
#'       (\code{beta_0}); this was found to introduce a real,
#'       quantitatively-characterized credible-interval miscalibration for
#'       \code{beta_0} at larger sample sizes (point estimates remained
#'       unbiased) and should currently be used only for exploratory
#'       comparison, not for inference you intend to report as final.}
#'     \item{\code{mcmc_warm_start}}{Logical, \strong{default \code{TRUE}}.
#'       Starts the NUTS chains from the internal \code{lme()} pre-fit
#'       rather than NumPyro's \code{init_to_uniform}, supplying
#'       \code{beta}, \code{sigma_e}, \code{sigma_b}, \code{L_corr} and the
#'       per-subject random effects (as \code{b_std}, obtained from the
#'       BLUPs through the model's own Cholesky factor). This is what
#'       \pkg{JMbayes2} has always done with the \code{lme}/\code{coxph}
#'       objects it is handed.
#'
#'       It matters at scale. With \code{q = 2} and 8,000 subjects the
#'       posterior has ~16,000 random-effect dimensions, and a uniform
#'       start puts every coordinate in the wrong place, leaving warm-up to
#'       travel to the typical set and adapt a step size in the same 500
#'       iterations. Measured on two sampler seeds that both failed from a
#'       cold start: bad fits 2/2 to 0/2, median divergences 98 to 0,
#'       median ESS for \code{rho} 4 to 101, median leapfrog steps 155 to
#'       111, and 28\% faster. At \code{n = 1,000} across five seeds it
#'       changed nothing measurable (63 steps, zero divergences, 0/5 bad
#'       either way).
#'
#'       \strong{The cost, which applies to \pkg{JMbayes2} equally:} chains
#'       that start together agree more readily, so R-hat becomes a less
#'       sensitive convergence diagnostic - Gelman-Rubin assumes
#'       overdispersed starting values. Set \code{FALSE} for a maximally
#'       stringent R-hat. Note that the evidence above does not rest on
#'       R-hat: ESS, divergences and step counts are all within-chain
#'       measures that a shared starting point cannot flatter.
#'
#'       Safe by construction. The starting values are checked twice -
#'       \code{jm_fit()} verifies that \code{tcrossprod(L)} reproduces the
#'       covariance matrix and that \code{b_std} round-trips back to the
#'       BLUPs, and the backend then compares the log-density at the warm
#'       start against a CONSERVATIVE one and uses whichever scores better.
#'       The conservative seed is JMbayes2's design - the longitudinal block
#'       and the survival covariates from the pre-fits, the association
#'       parameter at zero and a flat baseline hazard - which cannot blow up,
#'       since with \code{alpha = 0} the trajectory never enters the hazard.
#'       It is the floor: there is no fallback to a cold start.
#'
#'       This replaced a comparison against a \emph{uniform} start, which was
#'       shown to be unusable: on \code{pbc2} with unstandardized covariates
#'       the potential at the posterior mean - the centre of the distribution
#'       being sampled - scored 1,465,425 against 118,098 for a uniform draw,
#'       so the test ranked the best possible starting point below a random
#'       one. The bar was also \code{min()} over three draws, an order
#'       statistic rather than a typical value, and it moved 50,204 to 776,297
#'       on identical data across configurations. The uniform potential is
#'       still recorded in \code{fit$convergence$warm_start} (as a median) but
#'       decides nothing.
#'       behaviour, never a silently worse fit. The outcome is recorded in
#'       \code{fit$convergence$warm_start}.}
#'     \item{\code{warm_start_jitter}}{Numeric, default \code{0.1}. SD of
#'       the per-chain perturbation applied to the warm start, in
#'       \emph{unconstrained} space - so every constraint survives
#'       automatically (SDs stay positive, \code{L_corr} stays a valid
#'       Cholesky factor) with no special cases, unlike jittering
#'       constrained values. \code{0} starts every chain at the same point,
#'       which makes R-hat close to meaningless; larger values trade
#'       starting quality for between-chain dispersion.}
#'     \item{\code{wishart_eb_scale}}{Logical, default \code{FALSE}. Only
#'       relevant when \code{random_effects_method = "wishart_gibbs"} (or
#'       \code{"wishart_gibbs_centered"}). When \code{TRUE}, the Wishart
#'       prior's scale matrix is centered on the internal \code{lme()}
#'       pre-fit's variance estimates rather than the default fixed
#'       \code{S0 = I}, using \code{nu0 = q + 2} and
#'       \code{S0 = solve(diag(sigma_hat^2))} so that the implied prior
#'       mean of the covariance matrix equals the pre-fit estimate exactly
#'       (the default \code{nu0 = q + 1} sits at the boundary where that
#'       mean is undefined). This gives the Wishart-Gibbs path the same
#'       empirical-Bayes scaling the default path already has.
#'       \strong{New and not yet validated} - it has not been through the
#'       truth-recovery testing the other options received.}
#'     \item{\code{rw2_implementation}}{One of \code{"scan"} (default - the
#'       penalized-spline RW2 smoothing prior is constructed via a
#'       sequential JAX \code{scan}) or \code{"vectorized"} (an
#'       algebraically identical closed-form construction using two
#'       cumulative sums instead, verified to match the sequential version
#'       to floating-point precision). An initial single run suggested
#'       \code{"vectorized"} was ~1.4x faster, but \strong{replication
#'       across 3 seeds x 2 sample sizes did not support this} - at
#'       \code{n = 1000} it was consistently slightly WORSE (ESS/second
#'       ratio 0.92x) with no wall-clock saving, and R-hat was marginally
#'       worse on average. The original figure appears to have been noise.
#'       \code{"scan"} therefore remains the default. Note that
#'       \code{dense_mass_spline} works with either implementation (an
#'       earlier assumption that it required \code{"vectorized"} was
#'       tested and disproven - both reach an identical 63.0 mean leapfrog
#'       steps).}
#'     \item{\code{dense_mass_spline}}{Logical, \strong{default \code{TRUE}}
#'       (changed from \code{FALSE} following replication testing). When
#'       \code{spline_prior = "penalized"}, restricts NUTS's mass matrix to
#'       a dense block over just the spline-smoothing parameters (a small,
#'       cheap-to-invert ~7-9 dimensional block), rather than the default
#'       fully-diagonal treatment. The default diagonal mass matrix cannot
#'       represent the genuine correlation the RW2 prior induces among
#'       neighboring spline coefficients, which was found (via direct
#'       leapfrog-step-count measurement) to force NUTS into substantially
#'       deeper, more expensive exploratory trees - 255 steps per iteration
#'       versus 63 with this enabled. Replication across 3 seeds x 2 sample
#'       sizes gave ~2.1x faster wall-clock and ~2.3x ESS/second,
#'       consistent in direction on every single run, with R-hat and truth
#'       recovery unaffected. Works with either \code{rw2_implementation}.
#'       Set \code{FALSE} to restore the previous behavior.}
#'     \item{\code{dense_mass_alpha}}{Logical, default \code{FALSE}.
#'       \strong{Opt-in; helps \code{alpha} specifically, not overall
#'       inference.} Adds \code{alpha} to the spline dense mass block,
#'       capturing a measured \code{-0.73} to \code{-0.86} posterior
#'       correlation between \code{alpha} and the spline coefficients that
#'       the default block forces to zero. Across 10 replicate datasets
#'       with raw-scale covariates, \code{alpha}'s ESS/second improved on
#'       8 of 10 (paired t-test on the log ratio, \code{p = 0.007};
#'       95\% CI \code{[1.23x, 2.66x]}), with mean ESS/second rising from
#'       13.9 to 22.3. The effect is \strong{floor-raising rather than
#'       ceiling-raising}: the worst case improved from 4.8 to 14.0, while
#'       the two seeds that did worse were the two where the default
#'       already performed best.
#'       \strong{The important caveat}: minimum ESS across all parameters
#'       was essentially unchanged (279.9 vs 284.0 on average, mixed by
#'       seed). The gain is specific to \code{alpha}; overall inference
#'       quality is not improved, so this is not a general speedup and is
#'       not the default. Mean leapfrog step counts were also unchanged
#'       (210.3 vs 211.7), which is consistent with the association
#'       parameter decorrelating faster within trajectories whose length is
#'       set by other directions entirely - see
#'       \code{mass_matrix_investigation_negative_results.md}. Reasonable
#'       to enable when \code{alpha} is the parameter of primary interest,
#'       which is typical for joint models.}
#'     \item{\code{orthogonalize_b0}}{Logical, default \code{FALSE}.
#'       \strong{Experimental.} Sweeps the subject-constant columns of the
#'       longitudinal design out of the random intercepts, replacing
#'       \code{b_0} by its residual off that column space. With an
#'       intercept and one baseline covariate this removes a
#'       two-dimensional near-degeneracy; with an intercept alone it
#'       reduces to the familiar sum-to-zero constraint.
#'
#'       \strong{Why.} The likelihood constrains only the SUM of a
#'       subject-constant coefficient and the matching projection of the
#'       random intercepts; the split between them is pinned by the prior
#'       alone, at scale \code{sigma_b0/sqrt(N)}. Measured on \code{pbc2}
#'       (n = 312, 4 chains x 1000, in the space NUTS actually samples):
#'       \code{corr(beta_0, mean(b_i0)) = -0.9709} and
#'       \code{corr(beta_2, age-slope(b_i0)) = -0.9715}, against
#'       \code{+0.0543} for \code{beta_1}, a within-subject contrast. ESS
#'       for the identified sums was 2893 and >4000 against 441 and 526
#'       for the coefficients alone - a recoverable factor of 6.6x and
#'       >7.6x. These figures are only meaningful in sampled space;
#'       \code{posterior_samples$beta} is back-transformed when
#'       \code{standardize_covariates = TRUE}, and correlations computed
#'       on it measure the transform rather than the posterior.
#'
#'       \strong{Not the same as} \code{random_effects_method =
#'       "wishart_gibbs_centered"}, which pins \code{mean(b_i0)} at zero
#'       and so removes one direction of the degeneracy: that moved
#'       \code{beta_0} 441 -> 1490 but \code{beta_2} 526 -> 291, making
#'       the worst parameter worse.
#'
#'       \strong{The model is unchanged.} Every direction swept out of
#'       \code{b_0} lies in a column space \code{beta} already spans, so
#'       for any draw there is a shift of \code{beta} giving identical
#'       fitted values - the set of achievable mean structures is the
#'       same. Currently requires \code{q >= 2},
#'       \code{random_effects_corr = TRUE} and the default
#'       \code{random_effects_method = "nuts"}; other combinations raise
#'       an error rather than silently doing nothing.
#'
#'       \strong{But the reported intervals are NOT unchanged - read this
#'       before reporting a fit.} The implied prior on the constrained
#'       \code{b_0} is singular while \code{sigma_b} still governs the
#'       unconstrained draws, so the swept component is unidentified by
#'       the likelihood and samples its prior, and that uncertainty is
#'       absent from \code{beta}'s posterior. Measured (n = 300, 16
#'       replicates): for the swept coefficients, mean posterior SD
#'       divided by the across-replicate SD of posterior means falls to
#'       \strong{0.28} against \strong{1.18} for an unmodified fit, and
#'       95\% coverage to \strong{0.50} against \strong{1.00}. Point
#'       estimates stay unbiased, so a bias check does not catch this.
#'       \code{sigma_b0} and \code{alpha} are unaffected in every arm,
#'       which settles the empirical question earlier versions of this
#'       note left open: sweeping does not shift \code{sigma_b0}'s
#'       posterior.
#'
#'       \strong{Use \code{beta_corrected}, not \code{beta}, for swept
#'       parameters.} Because the sweep relocates the uncertainty rather
#'       than destroying it, it is exactly invertible. When either option
#'       actually sweeps a direction, the fit gains
#'       \code{posterior_samples$beta_corrected}: the same draws expressed
#'       in the identified parameterization, obtained by adding back the
#'       \code{beta}-shift corresponding to the swept component of
#'       \code{b_raw}. It is computed from draws the fit already produced
#'       - no refit, no extra sampling - and is absent (\code{NULL}) for
#'       fits where nothing was swept. Measured on the same replicates,
#'       it restores the SD ratio to 1.18-1.19 and 95\% coverage to 1.00,
#'       and tracks an unmodified fit's \code{beta_0} across replicates at
#'       r = 0.9999. So: sample with the option for the mixing gain, then
#'       report \code{beta_corrected}. \code{beta} itself is left
#'       untouched and remains the constrained estimand.
#'
#'       Two caveats. The correction relies on the same off-grid
#'       proportionality (Condition (S)) the sweep does, so a design where
#'       the fit warns about (S) breaks both. And the corrected draws are
#'       exact draws from the unmodified model's posterior only when the
#'       prior on \code{beta} is flat over the swept directions; under
#'       \code{jmjax}'s N(0, 5) prior they differ by a weight of order
#'       \code{sigma_b/(sigma_beta^2 sqrt(N))}, negligible at any usual
#'       sample size. See \code{vignette("jmjax-reparameterization")},
#'       Sections 4.9 and 9.1, for the derivation and the study.
#'
#'       \code{orthogonalize_b} sweeps every random-effect column (not
#'       just the intercept), which also removes a degeneracy between the
#'       fixed slope on time and the mean random slope - present as soon
#'       as there is any random slope on time, independently of how many
#'       other covariates are in the model. See
#'       \code{vignette("jmjax-reparameterization")} for the full
#'       mechanism and a replicated covariate-count study.
#'
#'       Whichever of the two is requested, what was actually swept (per
#'       column: whether a basis was found, how many directions, and
#'       which fixed-effects columns it came from) is reported in
#'       \code{fit$convergence$orthogonalize} - \code{NULL} unless one of
#'       these options was requested - rather than only in a
#'       \code{RuntimeWarning} at fit time. The warning is emitted by the
#'       Python backend and arrives on stderr, not as an R condition, so
#'       it is not visible to \code{tryCatch()}/\code{withCallingHandlers()}
#'       on the R side; \code{fit$convergence$orthogonalize} is the
#'       reliable way to check programmatically what happened.}
#'     \item{\code{dense_mass_beta}, \code{seed_mass_matrix}}{Both logical,
#'       both default \code{FALSE}. \strong{Tested and not adopted} -
#'       retained only to document the negative results.
#'       \code{seed_mass_matrix} seeds NUTS's diagonal inverse mass matrix
#'       for \code{beta} from the internal \code{lme()} pre-fit's standard
#'       errors; it failed catastrophically on 1 of 4 test datasets, and a
#'       diagnostic showed NUTS's own adaptation already recovers
#'       essentially those same scales unaided, so it could not have helped
#'       much regardless. \code{dense_mass_beta} adds a dense mass block
#'       over \code{beta}, targeting a confirmed \code{-0.98}
#'       intercept/covariate correlation; despite the diagnosis matching a
#'       first-principles derivation to four decimal places, it produced no
#'       consistent improvement (2 of 4 seeds worse in each condition).
#'       \strong{The recommended answer to the covariate-scale gap remains
#'       user-side standardization of continuous covariates}, which has a
#'       replicated 4-8x effect.}
#'     \item{\code{dense_mass}}{Logical, default \code{FALSE}. A
#'       \emph{global} dense mass matrix over every continuous parameter,
#'       including the (potentially large, per-subject) random effects.
#'       \strong{Not recommended} - tested and found to be dramatically
#'       counterproductive (roughly 12x slower in testing) due to the
#'       O(n^3) cost of estimating and inverting a mass matrix that scales
#'       with the number of subjects. \code{dense_mass_spline} above is the
#'       targeted alternative that captures the actual benefit.}
#'   }
#'   Also see \code{jmjax_setup(num_devices = ...)} for enabling genuinely
#'   parallel (rather than sequential) execution of multiple
#'   \code{num_chains}.
#'
#' @section Covariate standardization:
#'   \code{control$standardize_covariates} (\strong{default \code{TRUE}})
#'   centres and scales the baseline covariates in
#'   \code{long_formula} before any design matrix is built, then
#'   back-transforms \code{beta} so the reported coefficients are on the
#'   ORIGINAL scale. It is a pure reparameterization - \code{X \%*\% beta}
#'   is identical either way, so \code{alpha}, \code{sigma_b}, \code{rho}
#'   and \code{sigma_e} are unaffected.
#'
#'   \strong{Why it exists.} With an uncentred covariate the posterior
#'   correlation between the intercept and that covariate's coefficient is
#'   approximately \code{-xbar / sqrt(xbar^2 + s^2)}, measured at
#'   \code{-0.976} for \code{x ~ N(50, 10)} and ~0 once centred. That
#'   near-degenerate ridge is expensive for NUTS to traverse. Measured
#'   across 4 replicate datasets with raw-scale covariates: mean leapfrog
#'   steps fell from 228.3 to 49.6 (a 78\% reduction, in every seed) and
#'   sampling efficiency rose 2.31x on average (range 1.04x-3.11x, 4 of 4
#'   seeds favouring it), moving jmjax from below \code{JMbayes2}'s
#'   raw-scale range to above it. Truth recovery was unchanged.
#'
#'   \strong{Scope.} Only time-constant numeric covariates are touched;
#'   \code{time_var} and any variable in \code{random_formula} are excluded,
#'   since those columns appear in the random-effects design where centring
#'   would change what the random intercept means. For MCMC methods the
#'   back-transformation is applied to the posterior samples, so the
#'   reported SDs are exact. For MLE methods the transformation is linear,
#'   so \code{vcov} propagates exactly as \code{A \%*\% vcov \%*\% t(A)} -
#'   also exact, not a delta-method approximation.
#'
#'   \strong{On already-standardized data there is no measured benefit,
#'   and possibly a small cost.} Across 3 seeds with \code{x ~ N(0,1)} the
#'   efficiency ratio was 0.97x, 0.72x and 0.49x - though the last of those
#'   is inflated by a timing artifact (identical leapfrog step counts, 53\%
#'   more wall-clock), and the standardized-data baseline itself ranged
#'   40-92 ESS/second across seeds on identical settings, so this sits
#'   inside documented natural variability rather than being established
#'   harm. It remains the default because the users who benefit - those
#'   supplying raw-scale covariates - are also the least likely to know
#'   they need it, and because estimates are back-transformed exactly, so
#'   this affects sampling efficiency only and never the reported
#'   inference. Set \code{FALSE} to disable.
#'
#'   \strong{Reporting.} A message is emitted only when standardization is
#'   doing something that matters - specifically when the implied posterior
#'   correlation it removes between the intercept and a covariate's
#'   coefficient, \code{-xbar / sqrt(xbar^2 + s^2)}, exceeds 0.3 in
#'   magnitude. Covariates already close to centred produce a near-no-op
#'   transformation and are not reported, since coefficients are returned on
#'   the original scale and nothing the user reads is affected. The message
#'   is also deferred until after model validation, so a fit rejected for an
#'   unrelated reason does not first announce that covariates were
#'   standardized.
#'
#'   \strong{Caveats.} Validated on one simulation configuration with 4
#'   seeds on the raw-scale arm and 3 on the standardized arm. One
#'   raw-scale seed showed the geometry improvement (steps fell 5.7x)
#'   without a matching efficiency gain, because the shorter trajectories
#'   produced more autocorrelated draws. R-hat rose slightly but
#'   consistently (worst 1.0363 against 1.0069), consistent with cheaper,
#'   less exploratory iterations, and stays well inside conventional
#'   bounds.
#'
#' @section Time scaling (\code{scale_time}):
#'   \strong{Default: \code{"auto"}.} Time units are a computational
#'   artifact rather than a modelling choice - recording days instead of
#'   years does not change the inference, only whether NUTS can sample it.
#'   Requiring users to know that, and to know the remedy, asks them to
#'   understand a weakness of a sampler they did not choose.
#'
#'   \code{control$scale_time} substitutes
#'   \code{I(time/c)} for the time variable in \code{long_formula} and
#'   \code{random_formula} before the design matrices are built, with
#'   \code{c = max(abs(time))} or a numeric constant you supply. All
#'   coefficients are returned on the ORIGINAL scale.
#'
#'   \strong{What it fixes.} NUTS adapts a \emph{diagonal} mass matrix
#'   from warmup draws. When the random intercept and slope SDs differ by
#'   orders of magnitude - which happens whenever time is measured in small
#'   units over a long study, since a per-day slope is necessarily tiny -
#'   the sampler hits its maximum tree depth every iteration and never
#'   moves enough for adaptation to estimate the scales it needs. Bad
#'   geometry prevents the adaptation that would fix the geometry.
#'
#'   \strong{Measured.} On the \code{epileptic} data (SD ratio 796:1, time
#'   in days): 1023 leapfrog steps and 292s became 63 steps and 41s, with
#'   max R-hat falling from 22 to 1.01. On four datasets with ratios from
#'   1.9:1 to 8:1, step counts changed by 0-15\% - within noise, so
#'   enabling it by default costs well-conditioned models nothing.
#'
#'   \strong{Values.} \code{"auto"} (default) scales when it can verify
#'   the random-effects transformation is diagonal and DECLINES silently
#'   otherwise, so it cannot turn a working fit into a failure.
#'   \code{TRUE} or a numeric divisor scales and ERRORS if it cannot -
#'   use this when you want to be told. \code{FALSE} disables it.
#'   \code{fit$scale_time_status} always records what happened.
#'
#'   \strong{What it does not touch.} Survival times, spline knots and
#'   quadrature nodes stay on the raw scale. A three-arm experiment showed
#'   scaling only the design columns matched scaling the time variable
#'   everywhere, to the leapfrog step, so \code{alpha}, the baseline hazard
#'   and \code{rho} need no back-transformation at all. Only \code{beta}
#'   and \code{sigma_b} are corrected, through the same projection that
#'   handles \code{standardize_covariates}.
#'
#'   It \emph{scales} and never centres: \code{t/c} preserves
#'   \code{t = 0}, so the random intercept still means "value at baseline".
#'
#'   \strong{When to reach for it.} jmjax reports the random-effect SD
#'   ratio from its \code{lme} pre-fit when that ratio exceeds 50:1. The
#'   more definitive symptom is
#'   \code{fit$convergence$mean_num_steps} near \code{2^k - 1} for the
#'   sampler's maximum tree depth (1023 by default) - that means the
#'   trajectory budget was exhausted every iteration, and more draws will
#'   not help. No ratio threshold is documented because none has been
#'   established: 8:1 sampled fine in testing and 796:1 failed, with
#'   nothing measured between.
#'
#' @section EM warm start (\code{em_warm_start}):
#'   \code{control$em_warm_start = TRUE} runs a short joint EM phase
#'   before the optimizer: an E-step for the random effects, a closed-form
#'   M-step for the longitudinal block, and a short BFGS step on the
#'   survival block, following \code{JM}'s schedule (\code{maxit} 20 for
#'   five iterations, then 4) and its stopping criteria
#'   (\code{tol1 = 1e-3}, \code{tol2 = 1e-4}, \code{tol3 = sqrt(eps)}).
#'   Default \code{FALSE}.
#'
#'   \strong{When it helps.} EM improves the observed-data likelihood
#'   monotonically - it has no line search - so it can leave a region an
#'   optimizer cannot escape. On the \code{prothro} data with a spline
#'   baseline, where L-BFGS-B stops after 5 iterations 78 log-likelihood
#'   units short:
#'
#'   \tabular{lrr}{
#'     \strong{start} \tab \strong{loglik} \tab \strong{alpha} \cr
#'     cold L-BFGS-B \tab -14083.14 \tab -0.0008 \cr
#'     EM-warmed L-BFGS-B \tab -14004.49 \tab -0.0346 \cr
#'     cold BFGS \tab -14004.80 \tab -0.0391 \cr
#'     EM-warmed BFGS \tab -14001.88 \tab -0.0417
#'   }
#'
#'   with \code{JM} reaching \code{-0.0400} and jmjax's own MCMC
#'   \code{-0.0411}. 78.7 of the 78.3 missing units recovered.
#'
#'   \strong{Why it is not the default, unlike in \code{JM}.} On fits
#'   that already converge it costs about 2.2x the runtime and changes
#'   nothing - and it can leave the optimizer at a WORSE-conditioned point:
#'   EM stops on relative parameter change, so it may halt where the
#'   gradient is still substantial, and a quasi-Newton method started there
#'   can take zero steps. (\code{em_phase} now returns the better
#'   conditioned of its own endpoint and its starting values when the two
#'   likelihoods tie, but the cost remains.)
#'
#'   \code{JM} runs EM on every fit because it has no way to tell a stalled
#'   optimizer from a converged one. jmjax does:
#'   \code{fit$convergence$grad_max} against the threshold in the warning.
#'   So the recommended use is \emph{diagnostic-driven} - refit with
#'   \code{em_warm_start = TRUE} when a fit reports
#'   \code{converged = FALSE} - rather than unconditional.
#'
#'   \strong{Scope.} Currently wired into the base \code{q = 1} Weibull
#'   path only (no baseline covariates, no extra functional-form
#'   channels) - the one routed through the shared likelihood builder. On
#'   any other model the option is silently inert; check
#'   \code{fit$convergence$n_iter} against a run with it off to confirm
#'   whether it took effect.
#'
#' @section MLE optimizer settings:
#'   \code{control$parscale} (default \code{0.01}) rescales the parameter
#'   vector before optimization - jmjax optimizes \code{u = theta/s} and
#'   maps back, chain-ruling the gradient. The objective is mathematically
#'   unchanged; only the step geometry the line search works in differs.
#'   This mirrors what \code{optim()}'s argument of the same name does in
#'   R, which the \code{JM} package relies on
#'   (\code{control$parscale <- rep(0.01, length(thetas))}); scipy has no
#'   equivalent, so jmjax implements it explicitly.
#'
#'   \strong{Why it is needed.} jmjax optimizes \code{log_sigma_e},
#'   \code{log_sigma_b} and \code{atanh_rho}, so \code{d/d(log sigma_b)}
#'   carries a factor of \code{sigma_b}. On data where that is large - the
#'   \code{prothro} dataset has \code{sigma_b} near 18 - the gradient
#'   component for one parameter is an order of magnitude bigger than the
#'   rest, and a line search sized for it overshoots everything else.
#'   Without \code{parscale}, L-BFGS-B terminated at iteration 0 on that
#'   data with a finite objective AND a finite gradient, returning the
#'   starting values as if they were estimates.
#'
#'   Measured: \code{prothro} goes from a non-fit to converging in 54
#'   iterations at \code{alpha = -0.0384}, matching \code{JM}'s
#'   \code{-0.0384}. Across simulated data (3 seeds, both baseline
#'   hazards) and the AIDS data - all of which already converged -
#'   estimates shift by 1e-4 to 1e-3, i.e. optimizer tolerance, with no
#'   convergence regressions. Set \code{parscale = NULL} to disable.
#'
#'   \code{control$opt_method} (default \code{"BFGS"}) selects the scipy
#'   optimizer. \code{"L-BFGS-B"}, \code{"CG"}, \code{"Nelder-Mead"} and
#'   \code{"trust-constr"} are also available.
#'
#'   \strong{Why BFGS.} \code{"L-BFGS-B"} was the default and failed on
#'   the \code{prothro} data with a spline baseline: it stopped after 5
#'   iterations reporting \code{converged = TRUE} with no non-finite
#'   standard errors, 78 log-likelihood units short, returning an
#'   association parameter of \code{-0.0008} where \code{JM}, BFGS,
#'   \code{trust-constr} and jmjax's own MCMC all agree near
#'   \code{-0.04}. Tightening \code{ftol} changed nothing - a line search
#'   quit and was reported as convergence.
#'
#'   Across five dataset/method combinations, \code{"L-BFGS-B"} was short
#'   of the best log-likelihood twice and never won. The likely reason is
#'   that it keeps a \emph{limited-memory} curvature approximation (10
#'   correction pairs by default) for a problem with around 19 densely
#'   coupled parameters. \code{JM} uses \code{method = "BFGS"} in every
#'   one of its fitters and \code{"L-BFGS-B"} in none.
#'
#'   \code{control$parscale} still matters independently: without it BFGS
#'   and \code{"CG"} both diverged badly on \code{prothro}.
#'
#'   \strong{Not implemented:} \code{JM} additionally runs an EM phase
#'   before any quasi-Newton step, which improves the likelihood
#'   monotonically and so cannot fail a line search at all. That is a more
#'   robust approach than rescaling alone and remains a possible future
#'   addition.
#'
#' @section Prior specification:
#'   \code{control$prior_set} selects between two validated prior sets. It
#'   is a single switch because the three priors it covers were changed and
#'   validated together, and should be reverted together.
#'   \describe{
#'     \item{\code{"jmbayes2"} (default)}{Matches \code{JMbayes2}'s
#'       construction, read from that package's installed source and from a
#'       fitted \code{jm} object's \code{$priors} rather than inferred:
#'       \code{beta} prior SD \code{sqrt(pmin(14400 * V, 1000))} where
#'       \code{V} is the \code{lme()} sampling variance (the cap is what
#'       makes it weakly informative, and binds once a standard error
#'       exceeds about 0.264); \code{sigma_e ~ Gamma(5, 5 / residual SD)};
#'       and LKJ concentration 3. Validated by truth recovery across 3
#'       seeds x 2 covariate scales - all \code{|z| < 3} (max 2.62), all
#'       max R-hat < 1.05 (max 1.0297).}
#'     \item{\code{"legacy"}}{jmjax's previous, separately validated
#'       priors: \code{beta} prior SD \code{sqrt(10 * V)},
#'       \code{sigma_e ~ HalfNormal(2.0)}, LKJ concentration 2. Every
#'       benchmark and truth-recovery figure produced before the
#'       prior-matching work used these, so this setting keeps those
#'       results reproducible and provides a one-line fallback.}
#'   }
#'   The two \code{beta} priors differ by roughly 38x in SD terms, with
#'   \code{"legacy"} the \emph{tighter} of the two - despite an old code
#'   comment that wrongly claimed it matched \code{JMbayes2}. In practice
#'   the likelihood dominates either at moderate sample sizes (a measured
#'   \code{beta_0} posterior SD of 0.0365 against a likelihood-only 0.0358),
#'   so the choice matters mainly for small samples or weakly-identified
#'   coefficients. Individual components can still be overridden directly
#'   via \code{control$beta_prior_sd}, \code{control$sigma_e_prior_mean},
#'   and \code{control$lkj_concentration}, which take precedence.
#'
#' @section Adaptive-GH tuning options (MLE methods, \code{q = 2} only):
#'   These control the 2D adaptive Gauss-Hermite machinery used when
#'   \code{random_effects = "intercept_slope"} with \code{"weibull-PH-aGH"}
#'   or \code{"spline-PH-aGH"}. Defaults are deliberately conservative; see
#'   the notes below before changing them.
#'   \describe{
#'     \item{\code{n_newton_steps_q2}}{Default \code{20}. Number of Newton
#'       iterations used to locate each subject's random-effects mode.
#'       Setting \code{12} was measured to give roughly a 1.35-1.43x
#'       speedup with no detectable accuracy cost (identical log-likelihood
#'       to ~1e-2 on values in the thousands, and essentially unchanged
#'       parameter recovery) at \code{n = 500} and \code{n = 2000}.
#'       \strong{It is not the default deliberately}: this was validated on
#'       a single simulation configuration, and \code{8} steps was observed
#'       to FAIL to converge - so \code{12} sits between a known-good and a
#'       known-bad value with limited evidence in between. Newton
#'       convergence speed depends on the likelihood's curvature, which
#'       varies with the random-effects variances, event rate, and visit
#'       schedule, so a margin that is comfortable on one dataset may be
#'       thin on another. Users who want the speedup and can check
#'       \code{fit$convergence$converged} on their own data may set it
#'       explicitly.}
#'     \item{\code{n_gh_nodes_per_dim}}{Default \code{7} (a 7x7 = 49-point
#'       tensor-product quadrature grid). Testing found the grid size has
#'       almost no effect on runtime - 25 vs 81 points made no consistent
#'       difference, since \code{vmap} evaluates all nodes as one batched
#'       operation - and the log-likelihood was already fully converged at
#'       25 points. There is therefore little reason to lower it, and
#'       little cost to raising it if extra quadrature accuracy is wanted.}
#'     \item{\code{cholesky_first_newton}}{Default \code{FALSE}.
#'       \strong{Tested and rejected - not recommended.} This was an
#'       attempt to avoid the full eigendecomposition run on every Newton
#'       step by trying a (cheaper) Cholesky factorization first. It is
#'       preserved as an option only to document the negative result: under
#'       JAX's \code{jit}, a Python \code{try}/\code{except} cannot guard a
#'       traced operation, so the implementation must use \code{jnp.where},
#'       which forces BOTH branches to be computed. Enabling it therefore
#'       ADDS work rather than replacing it - measured 33\% slower at 20
#'       Newton steps and 10\% slower at 12. Anyone revisiting this idea
#'       will need a genuinely different approach (e.g. \code{lax.cond}, or
#'       dropping the safeguard for provably well-conditioned problems),
#'       not a retry of this one.}
#'   }
#'
#' @section Choosing a method: For a quick first look or to generate
#'   warm-start values, use \code{"weibull-PH-aGH"} - it is the fastest
#'   method and has no priors to configure. For a flexible baseline hazard
#'   with a single random effect, use \code{"spline-PH-aGH"}. For a
#'   correlated random intercept and slope, use one of the two MCMC
#'   methods; \code{"weibull-PH-mcmc"} is faster and simpler, while
#'   \code{"spline-PH-mcmc"} allows a flexible baseline hazard shape.
#'   Always inspect \code{fit$diagnostics} (R-hat, ESS) for MCMC fits before
#'   trusting the point estimates - see \code{vignette("jmjax-introduction")}.
#'
#' @return An object of class \code{"jmjax"} with components including
#'   \code{estimates}, \code{se}, \code{loglik} (MLE methods only),
#'   \code{diagnostics} (MCMC methods only: R-hat and ESS per parameter),
#'   and \code{random_effects} (MCMC methods only: per-subject posterior
#'   means, accessible via \code{ranef()}). See \code{print.jmjax} and
#'   \code{summary.jmjax}.
#'
#' @examples
#' \dontrun{
#' library(survival)
#'
#' # Fast Weibull MLE - good first look / warm-start source
#' fit_w <- jm_fit(
#'   long_formula = y ~ time,
#'   surv_formula = Surv(time, event) ~ 1,
#'   data_long = sim_long, data_surv = sim_surv,
#'   id_var = "id", time_var = "time",
#'   method = "weibull-PH-aGH"
#' )
#' summary(fit_w)
#'
#' # Flexible spline baseline hazard via full Bayesian MCMC, with a
#' # correlated random intercept and slope
#' fit_s <- jm_fit(
#'   long_formula = y ~ time,
#'   surv_formula = Surv(time, event) ~ 1,
#'   data_long = sim_long, data_surv = sim_surv,
#'   id_var = "id", time_var = "time",
#'   method = "spline-PH-mcmc",
#'   random_effects = "intercept_slope",
#'   control = list(num_warmup = 500, num_samples = 1000, num_chains = 2)
#' )
#' print(fit_s)                # shows max R-hat / min ESS
#' head(ranef(fit_s))          # per-subject random effects
#'
#' # Same model, with the validated performance-tuning options enabled -
#' # recommended over the defaults above for any real analysis. Pair with
#' # jmjax_setup(num_devices = 4) (called once, before this) for genuinely
#' # parallel chains rather than sequential execution.
#' fit_s_fast <- jm_fit(
#'   long_formula = y ~ time,
#'   surv_formula = Surv(time, event) ~ 1,
#'   data_long = sim_long, data_surv = sim_surv,
#'   id_var = "id", time_var = "time",
#'   method = "spline-PH-mcmc",
#'   random_effects = "intercept_slope",
#'   control = list(num_warmup = 500, num_samples = 1000, num_chains = 4,
#'                  progress_bar = FALSE,
#'                  random_effects_method = "wishart_gibbs",
#'                  spline_prior = "penalized",
#'                  rw2_implementation = "vectorized",
#'                  dense_mass_spline = TRUE)
#' )
#' }
#'
#' @export
jm_fit <- function(long_formula,
                    surv_formula,
                    data_long,
                    data_surv,
                    id_var,
                    time_var,
                    method = c("weibull-PH-aGH", "spline-PH-aGH", "spline-PH-mcmc", "weibull-PH-mcmc"),
                    random_effects = "intercept",
                    random_formula = NULL,
                    functional_forms = NULL,
                    init_theta = NULL,
                    control = list()) {

  method <- match.arg(method)

  response_var <- all.vars(long_formula)[1]
  assoc_types <- parse_functional_forms(functional_forms, response_var)

  if (!identical(assoc_types, "value")) {
    # SCOPE LIMIT: functional forms beyond "value" are currently supported
    # for method = 'weibull-PH-aGH' (delta, area, area_avg; both
    # random_effects options) and, as of the spline extension, method =
    # 'spline-PH-aGH' for 'delta' specifically (area/area_avg not yet
    # ported to the spline baseline - see vignette("jmjax-validation")
    # development notes for the incremental plan; note spline-PH-aGH is
    # separately restricted to random_effects = "intercept" regardless of
    # functional_forms, enforced above).
    if (method %in% c("spline-PH-aGH", "spline-PH-mcmc")) {
      # spline-PH-aGH (MLE): only 'delta' has been implemented so far -
      # a pure scope/implementation limit (area/area_avg's spline-
      # baseline MLE functions don't exist yet), NOT an identifiability
      # issue - both ARE supported for spline-PH-mcmc, see below.
      if (method == "spline-PH-aGH") {
        unsupported_for_spline_mle <- setdiff(assoc_types, c("value", "delta"))
        if (length(unsupported_for_spline_mle) > 0) {
          stop("functional_forms ", paste(unsupported_for_spline_mle, collapse = ", "),
               " not yet supported for method = 'spline-PH-aGH' (only 'delta' ",
               "has been ported to the spline baseline MLE backend so far; ",
               "area/area_avg remain weibull-PH-aGH-only there for MLE, though ",
               "both ARE supported for method = 'spline-PH-mcmc').")
        }
      }

      if ("delta" %in% assoc_types && random_effects == "intercept") {
        # NOT a code limitation - a confirmed IDENTIFIABILITY problem that
        # applies EQUALLY to spline-PH-aGH (MLE) and spline-PH-mcmc (MCMC),
        # since it's a property of the model class (flexible spline
        # baseline + a channel with zero subject-level variation at q=1),
        # not the estimation algorithm. Only restricted here (inside the
        # spline branch) - weibull-PH-aGH/weibull-PH-mcmc + delta remain
        # fully valid at q=1, confirmed by their own truth-recovery tests.
        #
        # For a pure random intercept (q=1), delta(t) = beta1*t is a
        # purely deterministic function of t with NO subject-level
        # variation (the intercept and its random effect both cancel in
        # the t=0 subtraction). alpha_delta * delta(t) is therefore just
        # another linear-in-time addition to the log-hazard, which a
        # flexible multi-parameter spline baseline can absorb directly
        # into its own coefficients - the likelihood cannot distinguish
        # "this baseline hazard shape" from "a slightly different shape
        # plus alpha_delta accounting for the difference". Confirmed
        # empirically for spline-PH-aGH during development:
        # beta/sigma_e/sigma_b/alpha_value recovered essentially
        # perfectly, but alpha_delta came out with the WRONG SIGN and a
        # singular (NaN) standard error, with 8 of 9 spline coefficients
        # also showing NaN SEs (the joint singularity signature of a
        # genuine likelihood ridge).
        #
        # AREA and AREA_AVG do NOT have this problem even at q=1: their
        # b-contributions (b_i*t and plain b_i respectively) ARE genuine
        # SUBJECT-level variation that a population-level baseline curve
        # cannot mimic (the curve is identical across every subject,
        # however flexible) - so only delta is restricted here, and this
        # reasoning is directly testable via delta's truth-recovery test
        # correctly failing/erroring while area/area_avg's succeed.
        #
        # A genuine random SLOPE (q=2) gives delta(t) real subject-level
        # variation ((beta1+b1_i)*t) that a population-level spline curve
        # cannot absorb, which resolves this (confirmed for spline-PH-aGH).
        stop("functional_forms = ~ ... + delta(...) with method = '", method,
             "' and random_effects = 'intercept' (q=1) is a confirmed NON-",
             "IDENTIFIABLE combination, not merely unimplemented: a flexible ",
             "spline baseline can fully absorb the purely deterministic ",
             "delta(t) signal that exists at q=1, leaving alpha_delta ",
             "unidentified (confirmed empirically for spline-PH-aGH - the ",
             "same mathematical argument applies identically to ",
             "spline-PH-mcmc, since it's a property of the model class, not ",
             "the estimation algorithm - see package development notes). Use ",
             "method = 'weibull-PH-aGH'/'weibull-PH-mcmc' instead (delta is ",
             "fully identified there), or use random_effects = ",
             "'intercept_slope' (q=2), where a genuine random slope gives ",
             "delta(t) real subject-level variation that resolves this. ",
             "area/area_avg do not have this identifiability problem even at ",
             "q=1 and remain fully supported.")
      }
      # NOTE: everything else - area/area_avg at q=1 (no identifiability
      # problem, per the reasoning above); delta/area/area_avg at q=2 (a
      # genuine random slope resolves delta's q=1 issue, confirmed for
      # spline-PH-aGH) - falls through to the normal channel-construction
      # code below rather than being blocked here.
    } else if (method == "weibull-PH-mcmc") {
      # SCOPE LIMIT: delta/area/area_avg supported at both q=1 and q=2
      # (mirroring the exact incremental discipline used for the MLE
      # extensions - q=2 uses the general Z_channel(t) @ b rule directly,
      # q=1 falls back to each channel's already-validated shortcut
      # formula, matching how the MLE side generalized too). (Combining
      # more than one of delta/area/area_avg simultaneously is already
      # rejected generically, regardless of method - see the
      # extra_channels check further below.)
    } else if (method != "weibull-PH-aGH") {
      stop("functional_forms beyond the default value-only association is ",
           "currently only supported for method = 'weibull-PH-aGH' (any ",
           "channel), 'spline-PH-aGH' (delta, both q=1 and q=2), ",
           "'weibull-PH-mcmc' (delta/area/area_avg, both q=1 and q=2), or ",
           "'spline-PH-mcmc' (area/area_avg any q; delta q=2 only - see the ",
           "identifiability note above for why delta is q=2-only there). ",
           "Requested: method = '", method, "'.")
    }
  }

  # --- 1. Build design arrays (R side) ---
  # ------------------------------------------------------------------
  # control$standardize_covariates (EXPERIMENTAL, opt-in, default FALSE)
  #
  # Centres and scales the baseline covariates in long_formula before any
  # design matrix is built, then back-transforms beta afterwards so the
  # reported coefficients are on the ORIGINAL scale. Purely a
  # reparameterisation - the fitted values X %*% beta are identical either
  # way, so alpha, sigma_b, rho and sigma_e are unaffected and need no
  # back-transformation.
  #
  # MOTIVATION: with an uncentred covariate, the posterior correlation
  # between the intercept and that covariate's coefficient is
  #     corr(beta_0, beta_x) ~= -xbar / sqrt(xbar^2 + s^2)
  # which was measured at -0.9764 for x ~ N(50, 10) - against a derived
  # prediction of -0.9802 - and ~0 once centred. That near-degenerate
  # ridge is a plausible cause of the measured covariate-scale gap (jmjax
  # 2-4x FASTER than JMbayes2 with standardized covariates, 1.4-2.8x
  # SLOWER with raw-scale ones). JMbayes2 avoids it structurally through
  # hierarchical centering; jmjax has no equivalent, so this is the
  # analogous fix.
  #
  # Three mass-matrix interventions previously failed to close this gap.
  # This is a REPARAMETERISATION rather than a preconditioner, so it is a
  # genuinely different mechanism - but that is a reason to test it, not
  # to assume it works.
  #
  # WHY data_long AND NOT the design matrices: extract_baseline_covariates_long()
  # derives its values from data_long, and build_time_design() evaluates
  # long_formula against those values at the survival and quadrature time
  # points. Transforming data_long once here therefore keeps X_long,
  # X_time_surv and X_time_quad automatically consistent. Transforming the
  # three design matrices separately would risk them drifting apart, which
  # would silently corrupt the model rather than error.
  #
  # Only time-constant covariates are touched: time_var is excluded, as is
  # anything appearing in random_formula (those columns are in Z, where
  # centring would change what the random intercept means). jmjax already
  # rejects genuinely time-varying longitudinal covariates elsewhere.
  # ------------------------------------------------------------------
  .std_info <- NULL
  # Shared by standardize_covariates and scale_time: whichever transforms
  # data_long FIRST takes the snapshot, and the back-transformation
  # recovers ONE projection covering both.
  .data_long_raw <- NULL
  if (isTRUE(control$standardize_covariates %||% TRUE)) {
    .lf_vars <- all.vars(long_formula)
    .response <- .lf_vars[1]
    .re_vars <- if (!is.null(random_formula)) all.vars(random_formula) else character(0)
    .cand <- setdiff(.lf_vars, c(.response, time_var, .re_vars, id_var))
    # numeric only - centring a factor makes no sense, and model.matrix()
    # expands factors into dummies downstream anyway
    .cand <- .cand[vapply(.cand, function(v) {
      v %in% names(data_long) && is.numeric(data_long[[v]])
    }, logical(1))]
    # a constant column has zero scale; leave it alone rather than divide by 0
    .cand <- .cand[vapply(.cand, function(v) stats::sd(data_long[[v]]) > 1e-12, logical(1))]

    if (length(.cand)) {
      # Keep the untransformed data: the back-transformation needs BOTH
      # design matrices to recover the projection A from X_o %*% A = X_s.
      .data_long_raw <- data_long
      .ctr <- vapply(.cand, function(v) mean(data_long[[v]]), numeric(1))
      .scl <- vapply(.cand, function(v) stats::sd(data_long[[v]]), numeric(1))
      for (v in .cand) {
        data_long[[v]] <- (data_long[[v]] - .ctr[[v]]) / .scl[[v]]
      }
      .std_info <- list(vars = .cand, center = .ctr, scale = .scl)

      # Whether to SAY anything is a separate question from whether to do
      # it. Standardization is now the default, so an unconditional message
      # fires on essentially every fit with a covariate - noise, especially
      # since coefficients come back on the original scale and nothing the
      # user reads is affected.
      #
      # Report only when the transformation is doing something that
      # matters. The relevant quantity is the posterior correlation it
      # removes between the intercept and the covariate's coefficient,
      #     corr(beta_0, beta_x) ~= -xbar / sqrt(xbar^2 + s^2)
      # derived and then confirmed empirically (predicted -0.9802, measured
      # -0.9764 for x ~ N(50,10)). Near zero, standardizing is a no-op and
      # there is nothing worth saying; near one, it is the difference
      # between a well-conditioned posterior and a near-degenerate ridge.
      .implied_corr <- abs(.ctr[.cand]) / sqrt(.ctr[.cand]^2 + .scl[.cand]^2)
      if (any(.implied_corr > 0.3)) {
        .std_info$message <- paste0(
          "standardize_covariates: standardized ", length(.cand),
          " covariate(s) in long_formula (",
          paste(sprintf("%s: centre %.4g, scale %.4g, implied intercept correlation %.2f",
                        .cand, .ctr[.cand], .scl[.cand], -.implied_corr),
                collapse = "; "),
          "). Coefficients are returned on the ORIGINAL scale. ",
          "Set control$standardize_covariates = FALSE to disable.")
      }
    }
  }

  # ---- Opt-in time scaling ----------------------------------------------
  # OPT-IN, default FALSE. Divides the time variable by a constant before
  # the design matrices are built, so every time-derived column in X and Z
  # is scaled. Coefficients come back on the ORIGINAL scale through the
  # same projection that handles standardization - time scaling is just
  # another linear map on the design.
  #
  # WHAT IT FIXES. NUTS adapts a DIAGONAL mass matrix from warmup draws.
  # When the random intercept and slope SDs differ by orders of magnitude -
  # epileptic's are 0.843 and 0.0011, a ratio of 796:1, because a per-DAY
  # slope is necessarily tiny - the sampler hits its maximum tree depth
  # every iteration and never moves enough for adaptation to estimate
  # anything useful. Bad geometry prevents the adaptation that would fix
  # the geometry.
  #
  # MEASURED, three arms on two datasets (raw / column-only / global):
  #   epileptic (796:1)  1023.0 -> 63.0 -> 63.0 leapfrog steps
  #                      R-hat 22.07 -> 1.0111 -> 1.0094
  #   prothro   (5.1:1)    60.7 -> 62.3 -> 62.9 steps, alpha identical
  # So it rescues a catastrophic case and costs a healthy one nothing. It
  # is opt-in rather than default because it is a targeted treatment: the
  # gain saturates well before a 1:1 ratio (122 -> 66 -> 63 steps for
  # 15:1 -> 8:1 -> 2:1), so most data sees no benefit at all.
  #
  # SCALING ONLY, NEVER CENTRING. t/c preserves t = 0, so the random
  # intercept still means "value at baseline". Centring time would change
  # what it means, which is why standardize_covariates excludes time_var.
  #
  # The survival side is deliberately LEFT ALONE - event times, spline
  # knots and quadrature nodes all stay on the raw scale. Column-only
  # scaling matched full variable scaling to the leapfrog step on both
  # datasets, so alpha, the baseline hazard and rho need no
  # back-transformation whatsoever.
  .time_scale <- NULL
  # Recorded unconditionally, not only on failure: a field that exists just
  # in the error case is one nobody remembers to check.
  # ---- Materialise a NULL random_formula -----------------------------
  # `NULL` used to mean TWO things at once: "build Z by slicing X's first
  # q columns" and "the user did not specify, so check for ambiguity".
  # That overloading is the root of every bug found while testing
  # scale_time: a substitution that missed the NULL branch, a fix that
  # defeated the ambiguity guard, and a test written against a
  # configuration the package rejects.
  #
  # Materialising it here separates the two. `Z` is now always built from
  # an explicit formula, so the slicing convention - correct only when
  # time_var happens to precede the covariates in long_formula - is no
  # longer load-bearing. `.rf_user_supplied` carries the other meaning
  # for the one guard that needs it.
  #
  # Verified before doing this: the slice and the formula path agree to
  # 1e-12 on Z_long, Z_time_surv and Z_time_quad across all 389 tests.
  .rf_user_supplied <- !is.null(random_formula)
  if (!.rf_user_supplied) {
    random_formula <- if (identical(random_effects, "intercept")) ~ 1 else
                      stats::as.formula(paste0("~", time_var))
  }

  .scale_time_status <- "off"
  # DEFAULT: "auto". Time units are a computational artifact, not a
  # modelling choice - whether someone records days or years should not
  # change their inference, and it does not. It changes only whether NUTS
  # can sample. Leaving this off put the burden on the people least able
  # to carry it: a user who does not know about the maximum-tree-depth
  # wall will not go looking for control$scale_time.
  #
  # The cost of NOT doing this is measured, not hypothetical. On the
  # epileptic data with time in days (random-effect SD ratio 796:1), an
  # unscaled fit returns max R-hat 22 and an association parameter of 7.1
  # where four independent routes agree on 0.12 - and nothing in the
  # output says the units caused it.
  #
  # "auto" DECLINES rather than errors when the random-effects
  # transformation is not diagonal (log(time), a fixed-knot spline), so
  # enabling it by default cannot turn a working fit into a failure.
  # scale_time = TRUE keeps the strict behaviour for anyone who wants to
  # be told.
  .st <- control$scale_time %||% "auto"
  .st_auto <- identical(.st, "auto")
  .st_explicit <- isTRUE(.st) || is.numeric(.st)

  # MCMC ONLY. Every measurement behind this feature is a NUTS concept -
  # leapfrog steps, the maximum-tree-depth wall, diagonal mass-matrix
  # adaptation, the bootstrap trap where warmup draws carry no scale
  # information. None of it describes an optimizer.
  #
  # The MLE paths have their OWN conditioning fix, control$parscale,
  # validated separately: it took prothro from a non-fit at iteration 0 to
  # alpha -0.0384, matching JM to four decimal places, and did the same on
  # liver. Applying scale_time on top changed what the optimizer sees and
  # those matches drifted - prothro -0.0384 -> -0.0395, liver -0.0385 ->
  # -0.0240, heart.valve 1.1856 -> 0.3830.
  #
  # And it did not rescue the MLE where it might have. On raw epileptic the
  # MLE used to fail loudly, with a non-finite gradient and a guard saying
  # so. With scaling it returned alpha 0.0106 against a cross-package
  # consensus near 0.21, with a singular Hessian - a loud failure turned
  # into a quiet wrong answer, which is worse.
  #
  # Each fix stays where its evidence is.
  .st_applies <- method %in% c("spline-PH-mcmc", "weibull-PH-mcmc")
  if (!.st_applies && .st_explicit) {
    message("scale_time applies to the MCMC methods only and has been ",
            "ignored for method = '", method, "'. The MLE paths use ",
            "control$parscale for conditioning instead (default 0.01).")
  }
  if ((.st_auto || .st_explicit) && .st_applies) {
    .tv <- suppressWarnings(as.numeric(data_long[[time_var]]))
    .c <- if (is.numeric(.st)) as.numeric(.st)[1] else max(abs(.tv), na.rm = TRUE)

    if (!is.finite(.c) || .c <= 0) {
      .scale_time_status <- "declined_no_constant"
      if (!.st_auto) {
        warning("scale_time: could not determine a positive scaling constant ",
                "from '", time_var, "'; no scaling applied.", call. = FALSE)
      }
    } else if (!isTRUE(all.equal(.c, 1))) {
      # REWRITE THE FORMULA, DO NOT REWRITE THE DATA.
      #
      # A first implementation divided data_long[[time_var]] directly. That
      # is wrong, and wrong in a way that looks plausible: `time_var` is
      # ALSO the slot into which build_time_design() substitutes the
      # SURVIVAL times and the quadrature nodes, to evaluate m_i(t) inside
      # the hazard. Scaling the column but not T_surv left the trajectory
      # evaluated at raw event times in a formula expecting time/c - a
      # scale mismatch between the two evaluations of the same formula,
      # structurally the same error as the AIDS time-unit bug.
      #
      # Measured damage: the optimizer stopped after 2 iterations reporting
      # convergence, with log_lambda0 -1.773 -> -0.587, shape 1.079 ->
      # 1.233 and alpha 0.473 -> -0.228 - survival parameters that scaling
      # is supposed to leave alone entirely.
      #
      # Substituting `I(time/c)` into the formulae instead means every
      # evaluation - X_long, X_time_surv, X_time_quad, Z and its survival
      # counterparts - divides by c consistently, while T_surv itself, the
      # spline knots and the quadrature limits stay raw. That is exactly
      # the construction validated by the three-arm benchmark, where
      # column-only scaling matched full variable scaling to the leapfrog
      # step on both epileptic (1023 -> 63) and prothro (60.7 -> 62.3).
      .sub_time <- function(f, tv, cc) {
        if (is.null(f)) return(NULL)
        repl <- str2lang(sprintf("I(%s/%.17g)", tv, cc))
        rec <- function(e) {
          if (is.name(e) && identical(as.character(e), tv)) return(repl)
          if (is.call(e)) { for (i in seq_along(e)[-1]) e[[i]] <- rec(e[[i]]); return(e) }
          e
        }
        f2 <- rec(f); environment(f2) <- environment(f); f2
      }
      # MATERIALISE A NULL random_formula FIRST.
      #
      # When random_formula is NULL and random_effects = "intercept_slope",
      # build_time_design() constructs Z from `time_var` internally rather
      # than from a formula object. An earlier version substituted only
      # when random_formula was non-NULL, so X got I(time/c) while Z stayed
      # raw - a mixed-scale model, and the two specifications of the SAME
      # model stopped agreeing. Caught by the backward-compatibility test
      # asserting that NULL and ~ time give identical estimates.
      #
      # Only for intercept_slope: "intercept" gives Z = ~1, which has no
      # time column and nothing to scale.
      # random_formula is materialised at entry now, so there is nothing
      # to fill in here - it is never NULL by this point.
      .lf_s <- .sub_time(long_formula, time_var, .c)
      .rf_s <- .sub_time(random_formula, time_var, .c)

      # Refuse rather than misreport: if time never appears as a bare
      # symbol in either formula, nothing was substituted and there is
      # nothing to back-transform - but the user asked for scaling and
      # would otherwise get silence.
      if (identical(deparse(.lf_s), deparse(long_formula)) &&
          (is.null(random_formula) ||
           identical(deparse(.rf_s), deparse(random_formula)))) {
        .scale_time_status <- "declined_no_plain_time"
        if (!.st_auto) {
          warning("scale_time: '", time_var, "' does not appear as a plain ",
                  "variable in long_formula or random_formula, so there is ",
                  "nothing to scale. If time enters through a transformation ",
                  "you have already chosen its scale; no scaling applied.",
                  call. = FALSE)
        }
      } else {
        # CHECK BEFORE COMMITTING, not after fitting. The random-effects
        # transformation A_z must be recoverable AND diagonal for sigma_b
        # to be correctable by a per-component factor. An earlier version
        # tested this in the back-transformation block, which runs AFTER
        # the backend call - far too late for "auto" to decline, since the
        # model had already been fitted on scaled designs.
        .rf_ref0 <- if (!is.null(random_formula)) random_formula else
                    if (random_effects == "intercept") ~ 1 else
                    stats::as.formula(paste0("~", time_var))
        .rf_new0 <- if (!is.null(random_formula)) .rf_s else
                    if (random_effects == "intercept") ~ 1 else
                    stats::as.formula(sprintf("~ I(%s/%.17g)", time_var, .c))

        .az0 <- tryCatch({
          Zo <- stats::model.matrix(.rf_ref0,
                  stats::model.frame(.rf_ref0, data = data_long))
          Zs <- stats::model.matrix(.rf_new0,
                  stats::model.frame(.rf_new0, data = data_long))
          M <- qr.solve(Zo, Zs)
          if (max(abs(Zo %*% M - Zs)) > 1e-6 * max(1, max(abs(Zs)))) NULL else M
        }, error = function(e) NULL)

        .prob <- if (is.null(.az0)) "not_recoverable" else
                 if (!.is_diagonal(.az0)) "non_diagonal" else
                 NULL

        if (!is.null(.prob) && .st_auto) {
          # Declined: never substitute, so the fit proceeds exactly as it
          # would have with scale_time = FALSE. No regression.
          .scale_time_status <- paste0("declined_", .prob)
        } else if (identical(.prob, "not_recoverable")) {
          stop("scale_time: the random-effects design cannot be recovered as ",
               "a linear map of the original, so sigma_b could not be ",
               "back-transformed. Returning beta on the original scale and ",
               "sigma_b on the scaled one would be internally inconsistent, ",
               "so the fit has been stopped.\n  Remedy: use ",
               "control$scale_time = \"auto\", which declines silently here, ",
               "or scale the time variable in your own data (e.g. data$",
               time_var, " <- data$", time_var, " / ", signif(.c, 6),
               ", and the same for the survival times) and refit with ",
               "control$scale_time = FALSE.", call. = FALSE)
        } else if (identical(.prob, "non_diagonal")) {
          stop("scale_time: the random-effects transformation is not ",
               "diagonal, so sigma_b would need a full congruence ",
               "D = A_z D A_z' rather than a per-component correction, and ",
               "the reported SDs and correlation could not both be right. ",
               "This happens when time enters random_formula through a term ",
               "that shifts rather than scales - log(time), or a spline with ",
               "FIXED knots.\n  Remedy: use control$scale_time = \"auto\", ",
               "which declines silently here, or scale the time variable in ",
               "your own data and refit with control$scale_time = FALSE.",
               call. = FALSE)
        } else {
          if (is.null(.data_long_raw)) .data_long_raw <- data_long
          .lf_orig <- long_formula; .rf_orig <- random_formula
          long_formula <- .lf_s
          if (!is.null(random_formula)) random_formula <- .rf_s
          .time_scale <- list(divisor = .c, time_var = time_var,
                               long_formula_orig = .lf_orig,
                               random_formula_orig = .rf_orig,
                               auto = .st_auto)
          .scale_time_status <- sprintf("applied (c = %.6g)", .c)
        }
      }
    }
  }

  long_arr <- build_long_arrays(long_formula, data_long, id_var)
  surv_arr <- build_surv_arrays(surv_formula, data_surv, long_arr$subj_ids, id_var,
                                 gk_order = control$gk_order %||% 10)

  # --- Baseline (time-constant) covariates in the survival submodel ---
  # e.g. surv_formula = Surv(time, event) ~ age + sex. IMPORTANT: prior to
  # this addition, build_surv_arrays() only ever read surv_formula's LHS
  # (the Surv() response) - any RHS covariates were SILENTLY DROPPED, not
  # rejected. Building this properly now closes that gap. The actual
  # scope-limit guard is placed after q is computed below.
  data_surv_ordered <- data_surv[match(long_arr$subj_ids, data_surv[[id_var]]), , drop = FALSE]
  W_surv <- build_baseline_covariates(surv_formula, data_surv_ordered)
  n_gamma <- ncol(W_surv)

  # --- Baseline (time-constant) covariates in the LONGITUDINAL submodel ---
  # e.g. long_formula = y ~ time + age. Extracted once here and threaded
  # through build_time_design() below so X(t) evaluated at a SYNTHETIC
  # time point (the survival time, quadrature nodes) correctly carries
  # forward each subject's actual covariate value - those synthetic
  # newdata frames otherwise only ever had time_var, silently EXCLUDING
  # any other covariate from the design (a latent gap this closes, the
  # longitudinal-submodel counterpart to the survival-submodel gap closed
  # by build_baseline_covariates() above).
  baseline_covariates_long <- extract_baseline_covariates_long(
    long_formula, time_var, data_long, id_var, long_arr$subj_ids
  )

  if (!is.null(baseline_covariates_long)) {
    # MLE (weibull-PH-aGH): only plain "value" - no combined "channel +
    # longitudinal covariates" MLE function was ever built.
    # MCMC (weibull-PH-mcmc/spline-PH-mcmc): fully supported INCLUDING
    # combined with any functional_forms channel - beta's length already
    # generalizes to any p automatically in build_model() regardless of
    # whether a channel is active (the channel logic's own X_extra@beta
    # term uses the SAME beta, with no separate restriction preventing
    # the two from coexisting).
    if (method %in% c("weibull-PH-aGH", "spline-PH-aGH")) {
      if (!identical(assoc_types, "value")) {
        stop("Baseline covariates in long_formula (found: ",
             paste(colnames(baseline_covariates_long), collapse = ", "), ") combined ",
             "with functional_forms beyond the default value-only association is ",
             "not yet supported for method = '", method, "' (though it IS ",
             "supported for 'weibull-PH-mcmc'/'spline-PH-mcmc').")
      }
    } else if (!(method %in% c("weibull-PH-mcmc", "spline-PH-mcmc"))) {
      stop("Baseline covariates in long_formula (found: ",
           paste(colnames(baseline_covariates_long), collapse = ", "), ") are ",
           "currently only supported for method = 'weibull-PH-aGH'/'spline-PH-aGH' ",
           "(value-only) or 'weibull-PH-mcmc'/'spline-PH-mcmc' (any ",
           "functional_forms channel). Requested: method = '", method, "'.")
    }
  }

  # Fixed-effects design evaluated AT the survival/quadrature times, needed
  # for the shared m_i(t) = X(t) %*% beta + b_i term (see build_time_design()
  # docs re: v1 scope - time_var-only formulas, now extended to also
  # support baseline covariates via baseline_covariates_long above).
  X_time_surv <- build_time_design(long_formula, time_var, surv_arr$T_surv,
                                    baseline_covariates_long)   # [N_sub, p]
  X_time_quad <- build_time_design(long_formula, time_var, surv_arr$t_quad,
                                    baseline_covariates_long)   # [N_sub, n_quad, p]

  # --- Centralized random-effects design (Z) construction ---
  # If random_formula is supplied, Z is evaluated INDEPENDENTLY from it -
  # exactly the same way X is evaluated from long_formula - fixing a
  # design assumption that only happened to be correct for
  # long_formula = y ~ time: previously Z was ALWAYS derived by slicing
  # X's first q columns (Z = X[, 1:q]), which silently breaks for any
  # long_formula where the first columns aren't simply "intercept" and
  # "slope" (e.g. y ~ time + I(time^2), or a spline-basis longitudinal
  # formula). `q` (the random-effects dimension used throughout the rest
  # of this function) now comes from whichever path was taken, rather
  # than directly from the random_effects string argument - decoupling
  # "how many random effects" from "how they're specified".
  if (!is.null(random_formula)) {
    Z_long <- build_random_long_array(random_formula, data_long, id_var,
                                       long_arr$subj_ids, max(long_arr$n_obs))
    Z_time_surv <- build_time_design(random_formula, time_var, surv_arr$T_surv)
    Z_time_quad <- build_time_design(random_formula, time_var, surv_arr$t_quad)
    q <- dim(Z_time_surv)[2]
  }
  # The `else` branch that took Z as X[, 1:q] is gone. random_formula is
  # materialised at entry, so Z is always built from an explicit formula
  # and the column-order dependence the slice introduced cannot arise.
  # Before removal, the two paths were asserted identical to 1e-12 on
  # Z_long, Z_time_surv and Z_time_quad across all 389 tests.

  if (q > 2L) {
    # The q=2 Python machinery (2D adaptive-GH tensor-product quadrature,
    # sigma_b0/sigma_b1/rho parameterization, 2x2 LKJCholesky/bivariate-
    # normal priors) is hardcoded for EXACTLY 2 random effects - it does
    # not generalize to q>=3 just because random_formula happens to
    # produce more columns. Guarded explicitly here rather than allowing
    # a confusing shape-mismatch failure deeper in the Python backend.
    stop("random_formula produced ", q, " random-effects columns - only ",
         "q=1 or q=2 are currently supported (the underlying adaptive-GH/",
         "NUTS machinery is specifically built for these two cases, not ",
         "arbitrary q). Simplify random_formula to at most 2 terms ",
         "(e.g. ~ 1 or ~ time).")
  }

  if (n_gamma > 0) {
    # Baseline covariates in the SURVIVAL submodel: gamma'W_i is a
    # time-constant, q-INDEPENDENT term (same formula regardless of
    # random-effects dimension).
    # MLE (weibull-PH-aGH): q=1 or q=2, plain "value" only - no combined
    # "channel + gamma" MLE function was ever built.
    # MCMC (weibull-PH-mcmc/spline-PH-mcmc): fully supported at any q
    # (q>2 guarded elsewhere) INCLUDING combined with any functional_forms
    # channel - build_model() adds gamma_term identically regardless of
    # which branch (channel present or not) computes log_hazard.
    if (method == "weibull-PH-aGH") {
      if (!(q %in% c(1L, 2L)) || !identical(assoc_types, "value")) {
        stop("Baseline covariates in surv_formula (found ", n_gamma, ": ",
             paste(colnames(W_surv), collapse = ", "), ") combined with q = ", q,
             " and/or functional_forms beyond value-only are not yet supported ",
             "for method = 'weibull-PH-aGH' (though any q/channel combination IS ",
             "supported for 'weibull-PH-mcmc'/'spline-PH-mcmc').")
      }
    } else if (method == "spline-PH-aGH") {
      # q=1 AND q=2 now supported (matching weibull-PH-aGH's own scope),
      # value-only only - not yet combined with functional_forms channels.
      if (!(q %in% c(1L, 2L)) || !identical(assoc_types, "value")) {
        stop("Baseline covariates in surv_formula (found ", n_gamma, ": ",
             paste(colnames(W_surv), collapse = ", "), ") combined with q = ", q,
             " and/or functional_forms beyond value-only are not yet supported ",
             "for method = 'spline-PH-aGH' (q=1/q=2, value-only, as of this ",
             "release - any q/channel combination IS supported for ",
             "'weibull-PH-mcmc'/'spline-PH-mcmc').")
      }
    } else if (!(method %in% c("weibull-PH-mcmc", "spline-PH-mcmc"))) {
      stop("Baseline covariates in surv_formula (found ", n_gamma, ": ",
           paste(colnames(W_surv), collapse = ", "), ") are currently only ",
           "supported for method = 'weibull-PH-aGH' (q=1/q=2, value-only), ",
           "'spline-PH-aGH' (q=1/q=2, value-only), or 'weibull-PH-mcmc'/",
           "'spline-PH-mcmc' (any q, any functional_forms channel). ",
           "Requested: method = '", method, "'.")
    }
  }

  # Keyed on `.rf_user_supplied`, not on is.null(random_formula):
  # random_formula is materialised at entry, so by here it is never NULL
  # and the question "did the USER specify Z?" needs its own flag.
  #
  # Now that Z is always built from an explicit formula, the ambiguity
  # this guards against - Z = X[, 1:q] picking columns by position, so
  # `y ~ x + time` silently puts a random effect on the covariate - can no
  # longer arise. The guard is retained for one release as a deliberate
  # behaviour change: users who were relying on the error will get it
  # until the NEWS entry has circulated. Removing it makes `y ~ x + time`
  # at q=2 fit correctly rather than error.
  if (!is.null(baseline_covariates_long) && q == 2L &&
      !isTRUE(.rf_user_supplied)) {
    # Baseline covariates in the LONGITUDINAL submodel at q=2: unlike the
    # survival-side case, this DOES interact with how Z is constructed.
    # The legacy default Z = X[:, :2] slicing only correctly EXCLUDES a
    # baseline covariate from Z if it happens to appear AFTER time_var in
    # long_formula's column order (e.g. y ~ time + age is fine, but
    # y ~ age + time would silently put age in Z instead of the intended
    # time slope) - rather than rely on that fragile convention,
    # random_formula must be given explicitly at q=2 so Z's structure is
    # unambiguous regardless of long_formula's column order.
    stop("Baseline covariates in long_formula (found: ",
         paste(colnames(baseline_covariates_long), collapse = ", "), ") combined ",
         "with q=2 (random_effects = 'intercept_slope') require random_formula ",
         "to be given explicitly (e.g. random_formula = ~ ", time_var, ") - the ",
         "default column-slicing convention for Z is ambiguous once long_formula ",
         "has covariates beyond time_var, depending on their column order.")
  }
  if (!is.null(baseline_covariates_long) && q > 2L) {
    stop("Baseline covariates in long_formula combined with q > 2 are not ",
         "supported (q = ", q, ") - the underlying adaptive-GH machinery is ",
         "hardcoded for q=1 or q=2.")
  }



  backend_args <- list(
    X_long = long_arr$X_long,
    y_long = long_arr$y_long,
    n_obs = long_arr$n_obs,
    X_time_surv = X_time_surv,
    X_time_quad = X_time_quad,
    T_surv = surv_arr$T_surv,
    event = surv_arr$event,
    t_quad = surv_arr$t_quad,
    gk_weights = surv_arr$gk_weights
  )

  if (n_gamma > 0) {
    backend_args$W_surv <- W_surv
  }

  # SCOPE LIMIT (implemented incrementally): delta, area, and area_avg are
  # each their own separate function on the Python side (mirroring how
  # q=1/q=2 and Weibull/spline were each kept separate throughout this
  # package's development) rather than a fully general combinable
  # N-channel system - combining more than one non-value channel at once
  # is a natural future generalization once each is independently
  # validated on its own.
  extra_channels <- intersect(assoc_types, c("delta", "area", "area_avg"))
  if (length(extra_channels) > 1) {
    stop("functional_forms combining more than one of delta/area/area_avg ",
         "simultaneously is not yet supported (requested: ",
         paste(extra_channels, collapse = ", "), ") - use just one for now.")
  }

  # General rule (derived from the q=1 formulas): each channel's random-
  # effect contribution is Z_channel(t) @ b, where Z_channel(t) is that
  # channel's own random-effects design (built the same way as its
  # fixed-effects design, just from Z_time_surv/Z_time_quad and whichever
  # formula produced them). At q=1 with the legacy X-slicing Z, this
  # reduces to the three formulas already validated separately (delta's Z
  # has a zero first column -> "no b term"; area's Z's first column is
  # t -> "b*t"; area_avg's is 1 -> "plain b") - confirming those were all
  # special cases of one general rule, not three different ones.
  if ("delta" %in% assoc_types) {
    delta_channel <- build_delta_channel(long_formula, time_var, X_time_surv, X_time_quad)
    backend_args$X_delta_surv <- delta_channel$X_delta_surv
    backend_args$X_delta_quad <- delta_channel$X_delta_quad
    if (q > 1L) {
      if (!is.null(random_formula)) {
        # Build Z's OWN delta channel independently - reusing
        # build_delta_channel() a second time, now against
        # random_formula/Z_time_surv/Z_time_quad instead of
        # long_formula/X_time_surv/X_time_quad. Slicing X's channel would
        # be WRONG here (X's zero-row has the wrong number of columns to
        # match a sliced Z, and more fundamentally Z's own basis functions
        # may differ entirely from X's).
        z_delta <- build_delta_channel(random_formula, time_var, Z_time_surv, Z_time_quad)
        backend_args$Z_delta_surv <- z_delta$X_delta_surv
        backend_args$Z_delta_quad <- z_delta$X_delta_quad
      }
    }
  }

  if ("area" %in% assoc_types) {
    area_channel <- build_area_channel(long_formula, time_var, surv_arr$T_surv,
                                        surv_arr$t_quad, surv_arr$gk_weights)
    backend_args$X_area_surv <- area_channel$X_area_surv
    backend_args$X_area_quad <- area_channel$X_area_quad
    if (q > 1L) {
      if (!is.null(random_formula)) {
        z_area <- build_area_channel(random_formula, time_var, surv_arr$T_surv,
                                      surv_arr$t_quad, surv_arr$gk_weights)
        backend_args$Z_area_surv <- z_area$X_area_surv
        backend_args$Z_area_quad <- z_area$X_area_quad
      }
    }
  }

  if ("area_avg" %in% assoc_types) {
    # area_avg(t) = area(t) / t - confirmed by direct inspection of a
    # fitted JMbayes2 object's model_data$X_h that this, not the raw
    # integral, is what JMbayes2::area() actually computes (see
    # build_design.R's parse_functional_forms docs for the full story).
    # Reuses build_area_channel()'s already-validated raw integral rather
    # than duplicating the nested-quadrature logic.
    raw_area <- build_area_channel(long_formula, time_var, surv_arr$T_surv,
                                    surv_arr$t_quad, surv_arr$gk_weights)
    X_area_avg_surv <- raw_area$X_area_surv / surv_arr$T_surv

    # Explicit per-column division loop, NOT a direct array/matrix divide -
    # t_quad's shape ([N_sub, n_quad]) doesn't broadcast automatically
    # against X_area_quad's shape ([N_sub, n_quad, p]) the way it would in
    # numpy; R has no such multi-dimensional broadcasting, and silently
    # getting this wrong is exactly the class of bug (mismatched implicit
    # array assumptions) that has bitten this package before.
    X_area_avg_quad <- array(0, dim = dim(raw_area$X_area_quad))
    for (pp in seq_len(dim(raw_area$X_area_quad)[3])) {
      X_area_avg_quad[, , pp] <- raw_area$X_area_quad[, , pp] / surv_arr$t_quad
    }

    backend_args$X_area_avg_surv <- X_area_avg_surv
    backend_args$X_area_avg_quad <- X_area_avg_quad
    if (q > 1L) {
      if (!is.null(random_formula)) {
        z_raw_area <- build_area_channel(random_formula, time_var, surv_arr$T_surv,
                                          surv_arr$t_quad, surv_arr$gk_weights)
        Z_area_avg_surv <- z_raw_area$X_area_surv / surv_arr$T_surv
        Z_area_avg_quad <- array(0, dim = dim(z_raw_area$X_area_quad))
        for (pp in seq_len(dim(z_raw_area$X_area_quad)[3])) {
          Z_area_avg_quad[, , pp] <- z_raw_area$X_area_quad[, , pp] / surv_arr$t_quad
        }
        backend_args$Z_area_avg_surv <- Z_area_avg_surv
        backend_args$Z_area_avg_quad <- Z_area_avg_quad
      }
    }
  }

  if (method %in% c("weibull-PH-aGH", "spline-PH-aGH")) {
    # Adaptive-GH-specific args: NUTS (spline-PH-mcmc) doesn't marginalize
    # random effects via quadrature at all, so fit_nuts() has no such
    # parameters - passing these unconditionally caused
    # "unexpected keyword argument 'n_gh_nodes'" for the MCMC method.
    backend_args$n_gh_nodes <- as.integer(control$n_gh_nodes %||% 15L)
    backend_args$n_newton_steps <- as.integer(control$n_newton_steps %||% 8L)
  }

  if (method == "weibull-PH-aGH" && q > 1L) {
    # 2D adaptive-GH path (tensor-product quadrature) - added specifically
    # to cross-validate against JM's weibull-PH-aGH with a real random
    # slope, as a third independent check (pure MLE, zero sampling
    # considerations) alongside jmjax's own weibull-PH-mcmc. The exact
    # string here doesn't matter to the Python side beyond "not equal to
    # 'intercept'" - "intercept_slope" used for clarity/consistency with
    # the random_effects argument's own vocabulary.
    backend_args$random_effects <- "intercept_slope"
    backend_args$Z_long <- Z_long
    backend_args$Z_time_surv <- Z_time_surv
    backend_args$Z_time_quad <- Z_time_quad
  }

  if (method == "spline-PH-aGH" && q > 1L) {
    # 2D adaptive-GH path combined with the spline basis - the extension
    # flagged as a "separate, bigger undertaking" in earlier development
    # notes. Mirrors the weibull-PH-aGH q=2 block above exactly (same
    # centralized Z), only the baseline hazard computation differs
    # (spline B(t) @ W vs Weibull's closed form) on the Python side.
    backend_args$random_effects <- "intercept_slope"
    backend_args$Z_long <- Z_long
    backend_args$Z_time_surv <- Z_time_surv
    backend_args$Z_time_quad <- Z_time_quad
  }

  # ---- Starting values for the MLE optimizers -------------------------
  # Deliberately a SEPARATE block from the empirical-Bayes prior setup
  # below, which is MCMC-only: priors are irrelevant to an MLE fit, but
  # STARTING VALUES are not, and an earlier attempt to piggyback on that
  # block silently did nothing for exactly this reason.
  #
  # Why it matters. jmjax's MLE paths otherwise begin from hard-coded
  # constants that ignore the data: beta_0 = 2.0 with every other beta at
  # 0, sigma_e = 0.5, sigma_b = 0.5, alpha = +0.5. On the AIDS data the
  # lme() fit gives beta = (2.71, -0.40, -0.54), sigma_e = 0.40,
  # sigma_b = (0.83, 0.13) - and the fitted alpha is about -0.48, so the
  # default starts it at the WRONG SIGN, roughly 1.0 away. L-BFGS-B is a
  # local optimizer: unlike NUTS it cannot explore out of a poor start,
  # and can stall or converge where the Hessian is singular, which shows
  # up as NaN standard errors.
  #
  # JM does the same thing, via initial.surv():
  #     cph <- coxph(form, data = DD); coefs <- cph$coefficients
  #     out <- list(alpha = coefs[1:k], gammas = coefs[-(1:k)])
  # i.e. a two-stage approximation - fit a Cox model against the mixed
  # model's fitted trajectory and read alpha and gamma off it.
  #
  # Entirely best-effort: any failure leaves the previous constants in
  # place. A starting value that cannot be computed must never break a fit.
  if (method %in% c("weibull-PH-aGH", "spline-PH-aGH") &&
      isTRUE(control$init_from_prefit %||% TRUE)) {
    # Reuse a caller-supplied lme object rather than refitting.
    # jm_fit_prefit() sets control$.lme_object to the model the USER
    # fitted. Refitting would be wasteful, would silently discard any
    # lmeControl settings they chose, and would defeat the point of an
    # interface that accepts a fitted model.
    #
    # The object is needed here for more than the coefficients: the
    # two-stage alpha below reads ranef() and fixef() off it, so simply
    # supplying init_beta etc. and skipping the fit would leave alpha at
    # its hard-coded default.
    #
    # Stripped from `control` further down, before anything is handed to
    # the Python backend - an lme object cannot be marshalled through
    # reticulate.
    .mle_init <- if (inherits(control$.lme_object, "lme")) control$.lme_object
                 else tryCatch({
      .rf <- if (!is.null(random_formula)) {
        stats::as.formula(paste(deparse(random_formula), "|", id_var))
      } else if (random_effects == "intercept") {
        stats::as.formula(paste0("~1 | ", id_var))
      } else {
        stats::as.formula(paste0("~", time_var, " | ", id_var))
      }
      nlme::lme(long_formula, random = .rf, data = data_long,
                control = nlme::lmeControl(opt = "optim", msMaxIter = 200,
                                            niterEM = 100, msMaxEval = 500,
                                            returnObject = TRUE))
    }, error = function(e) NULL, warning = function(w) NULL)

    if (!is.null(.mle_init)) {
      .Dm <- tryCatch(nlme::getVarCov(.mle_init), error = function(e) NULL)
      control$init_beta    <- control$init_beta    %||% unname(nlme::fixef(.mle_init))
      control$init_sigma_e <- control$init_sigma_e %||% stats::sigma(.mle_init)
      if (!is.null(.Dm)) {
        control$init_sigma_b <- control$init_sigma_b %||% sqrt(diag(.Dm))
      # Surface the random-effect SD ratio. This is the signal for whether
      # control$scale_time is worth enabling: NUTS adapts a DIAGONAL mass
      # matrix from warmup draws, and when the ratio is extreme the sampler
      # hits its maximum tree depth every iteration and never moves enough
      # for adaptation to estimate anything. Reported without a threshold -
      # 8:1 sampled fine in testing and 796:1 failed outright, with nothing
      # measured between, so any cut-off would be invented.
      if (nrow(.Dm) >= 2 && isTRUE(control$report_re_ratio %||% TRUE)) {
        .sdv <- sqrt(diag(.Dm))
        .rt <- .sdv[1] / max(.sdv[2], .Machine$double.eps)
        if (is.finite(.rt) && .rt > 50) {
          message(sprintf(paste0(
            "random-effect SD ratio is %.0f:1 (intercept %.4g, slope %.4g). ",
            "A ratio this large can stall NUTS at its maximum tree depth. ",
            "If sampling is slow or R-hat is poor, check ",
            "fit$convergence$mean_num_steps: a value near 2^k - 1 (1023 by ",
            "default) means the sampler exhausted its budget every ",
            "iteration, and control$scale_time = TRUE may help."),
            .rt, .sdv[1], .sdv[2]))
        }
      }
        if (nrow(.Dm) >= 2) {
          control$init_rho <- control$init_rho %||%
            (.Dm[1, 2] / sqrt(.Dm[1, 1] * .Dm[2, 2]))
        }
      }

      # Weibull baseline-hazard starting values, from a survreg() fit.
      # The spline path already gets a data-driven baseline (projected onto
      # B_T, further below); the Weibull path did not, leaving log_lambda0
      # at -2.0 and shape at 1.2 regardless of the data.
      #
      # That is not cosmetic. On the prothro data those two parameters, plus
      # alpha, were exactly the three with non-finite standard errors, and
      # the optimizer terminated at iteration 0 - alpha was data-derived but
      # the two parameters it multiplies against were not.
      #
      # survreg's AFT parameterisation converts to PH as:
      #     shape       = 1 / scale
      #     log_lambda0 = -shape * (Intercept)
      if (is.null(control$init_log_lambda0) || is.null(control$init_log_shape)) {
        .wb <- tryCatch({
          .sr <- survival::survreg(surv_formula, data = data_surv, dist = "weibull")
          .sh <- 1 / .sr$scale
          list(log_shape = log(.sh),
               log_lambda0 = -.sh * unname(stats::coef(.sr)[1]))
        }, error = function(e) NULL, warning = function(w) NULL)
        if (!is.null(.wb) && all(is.finite(unlist(.wb)))) {
          control$init_log_lambda0 <- control$init_log_lambda0 %||% .wb$log_lambda0
          control$init_log_shape   <- control$init_log_shape   %||% .wb$log_shape
        }
      }

      # Two-stage alpha/gamma, mirroring JM's initial.surv(). Subjects are
      # aligned BY NAME in both directions - nlme orders by its own
      # grouping levels, and a positional match would silently pair the
      # wrong subjects, an error invisible in the output.
      if (is.null(control$init_alpha)) {
        .ts <- tryCatch({
          .re <- as.matrix(nlme::ranef(.mle_init))
          .ord <- match(as.character(long_arr$subj_ids), rownames(.re))
          stopifnot(!anyNA(.ord))
          .re <- .re[.ord, , drop = FALSE]
          .mhat <- as.vector(X_time_surv %*% nlme::fixef(.mle_init)) +
            rowSums(Z_time_surv * .re[, seq_len(ncol(Z_time_surv)), drop = FALSE])

          .did <- match(as.character(long_arr$subj_ids),
                        as.character(data_surv[[id_var]]))
          stopifnot(!anyNA(.did))
          .dd <- data_surv[.did, , drop = FALSE]
          .dd$.jmjax_mhat <- .mhat

          .cf <- stats::coef(survival::coxph(
            stats::update(surv_formula, . ~ . + .jmjax_mhat), data = .dd))
          list(alpha = unname(.cf[".jmjax_mhat"]),
               gamma = unname(.cf[setdiff(names(.cf), ".jmjax_mhat")]))
        }, error = function(e) NULL, warning = function(w) NULL)

        if (!is.null(.ts) && length(.ts$alpha) == 1L && is.finite(.ts$alpha)) {
          control$init_alpha <- .ts$alpha
          if (length(.ts$gamma) && is.null(control$init_gamma)) {
            control$init_gamma <- .ts$gamma
          }
        }
      }
    }
  }

  if (method %in% c("spline-PH-mcmc", "weibull-PH-mcmc")) {
    # Previously validated (see the stop() check above) but never actually
    # forwarded to the backend - requesting "intercept_slope" would have
    # silently been ignored and always fit a random-intercept-only model.
    # Derived from q (the centralized single source of truth - see the
    # random-effects construction block above) rather than the raw
    # random_effects argument directly, since q may come from
    # random_formula instead when that's supplied. fit_nuts()'s own logic
    # only checks equality to "intercept", so any other string is fine.
    backend_args$random_effects <- if (q == 1L) "intercept" else "intercept_slope"

    # fit_nuts() already accepts Z_long/Z_time_surv/Z_time_quad as
    # optional overrides, falling back to its own internal X[:, :q]
    # slicing only when they're not provided - so passing our already-
    # centralized Z (whether independently built from random_formula, or
    # the legacy X-slice) is a purely mechanical wiring change requiring
    # NO Python-side modifications at all. Confirmed by reading
    # mcmc_model.py's fit_nuts() signature directly rather than assuming.
    backend_args$Z_long <- Z_long
    backend_args$Z_time_surv <- Z_time_surv
    backend_args$Z_time_quad <- Z_time_quad

    # Tell the Python model which baseline hazard to use - unrelated to
    # spline_prior below, which only matters when baseline_hazard="spline".
    control$baseline_hazard <- if (method == "weibull-PH-mcmc") "weibull" else "spline"

    # --- Empirical-Bayes prior centering, matching JMbayes2's confirmed approach ---
    # JMbayes2 centers its priors for beta and the random-effects SDs on the
    # MLEs from an initial frequentist lme() fit (confirmed directly from a
    # fitted JMbayes2 object's $priors$D_sds_mean, $gamma_prior_D_sds during
    # development), rather than using flat/vague priors - this stabilizes
    # estimation of the parameters most entangled with alpha (the random
    # slope variance in particular) especially at small-to-moderate n. This
    # replicates that same mechanism, but jm_fit() fits the lme() internally
    # so the user never needs to pre-fit or pass one themselves, unlike
    # JMbayes2's own interface. Set control$empirical_bayes_prior = FALSE to
    # use jmjax's original uninformative priors instead. Applies to both
    # MCMC methods regardless of baseline hazard choice.
    use_eb_prior <- isTRUE(control$empirical_bayes_prior %||% TRUE)

    # ------------------------------------------------------------------
    # control$prior_set: a SINGLE switch covering all three priors that
    # were changed together to match JMbayes2. Kept as one knob rather
    # than three because they were validated together and should be
    # reverted together if anything regresses.
    #
    #   "jmbayes2" (DEFAULT) - matches JMbayes2's verified construction:
    #       beta prior sd  = sqrt(pmin(14400 * V, 1000))
    #       sigma_e        = Gamma(5, 5 / lme residual sd)
    #       LKJ eta        = 3
    #     All three were read from JMbayes2's installed source or a
    #     fitted jm object's $priors, not inferred. Validated by truth
    #     recovery across 3 seeds x 2 covariate scales: all |z| < 3
    #     (max 2.62), all max R-hat < 1.05 (max 1.0297).
    #
    #   "legacy" - jmjax's previous, separately validated priors:
    #       beta prior sd  = sqrt(10 * V)
    #       sigma_e        = HalfNormal(2.0)
    #       LKJ eta        = 2
    #     Every benchmark and truth-recovery result produced before the
    #     prior-matching work used these. Retained so those results stay
    #     reproducible and so there is a one-line fallback if the new set
    #     turns out to misbehave on a configuration not yet tested.
    #
    # Note the beta priors differ by roughly 38x in SD terms - "legacy"
    # is the TIGHTER of the two, despite a code comment that used to
    # claim it matched JMbayes2. In practice the likelihood dominates
    # either one at moderate sample sizes (measured beta_0 posterior SD
    # 0.0365 against a likelihood-only 0.0358), so the difference bites
    # mainly in small samples or on weakly-identified coefficients.
    # ------------------------------------------------------------------
    prior_set <- match.arg(control$prior_set %||% "jmbayes2",
                            c("jmbayes2", "legacy"))
    # LKJ concentration is set here rather than inside the pre-fit block
    # below, because it applies whether or not the empirical-Bayes
    # pre-fit runs or succeeds.
    control$lkj_concentration <- control$lkj_concentration %||%
      if (prior_set == "jmbayes2") 3.0 else 2.0

    if (use_eb_prior) {
      # Combine random_formula's own terms with the id_var grouping,
      # matching nlme::lme()'s "~ terms | group" syntax - random_formula
      # itself has no grouping concept (jmjax already takes id_var as a
      # separate argument), so it needs to be appended here. Using
      # random_formula directly (when supplied) rather than assuming the
      # random slope is always on time_var, which was only ever correct
      # by coincidence for the common random_formula = ~ time case.
      random_form <- if (!is.null(random_formula)) {
        stats::as.formula(paste(deparse(random_formula), "|", id_var))
      } else if (random_effects == "intercept") {
        stats::as.formula(paste0("~1 | ", id_var))
      } else {
        stats::as.formula(paste0("~", time_var, " | ", id_var))
      }
      lme_prefit <- tryCatch(
        nlme::lme(long_formula, random = random_form, data = data_long,
                  control = nlme::lmeControl(opt = "optim")),
        error = function(e) NULL
      )
      if (!is.null(lme_prefit)) {
        control$beta_prior_mean <- unname(nlme::fixef(lme_prefit))
        # Matches JMbayes2's weak_informative_Tau(), read directly from the
        # installed package's source:
        #     V <- vcov_center(vcov2(model), Xbar)
        #     diags <- pmin(14400 * diag(V), 1000)
        #     diag(1/diags, nrow(V), ncol(V))
        # So the prior VARIANCE is 14400x the sampling variance, CAPPED at
        # 1000. The cap is what makes it weakly informative rather than
        # unboundedly diffuse; it binds once a standard error exceeds about
        # 0.264, which is entirely plausible for a coefficient on a rare
        # binary covariate.
        #
        # This replaces an earlier "prior variance = 10x the sampling
        # variance" which carried a comment claiming JMbayes2 parity. That
        # claim was WRONG - it made jmjax's beta prior roughly 38x TIGHTER
        # than JMbayes2's. Verified empirically first (ratio exactly
        # 14400.0 across n = 312/156/78 and p = 2/3), then confirmed in
        # source.
        #
        # NOTE on vcov_center: JMbayes2 applies that transform because it
        # works internally in a CENTRED-intercept parameterisation, where
        # the intercept means "linear predictor at mean covariate values".
        # jmjax samples beta directly in the original, uncentred
        # parameterisation, so copying the transform would produce a prior
        # for the wrong quantity. Plain vcov() is the correct analogue
        # here - matching the construction, each package in its own
        # parameterisation. (The transform is a no-op whenever Xbar is
        # ~zero anyway, which is why the empirical probes never saw it.)
        .v <- diag(stats::vcov(lme_prefit))

        # Supply beta_se for control$seed_mass_matrix. Without this the
        # backend reads control.get("beta_se") as None and the seeding is
        # SILENTLY SKIPPED - so seed_mass_matrix = TRUE has been a no-op
        # since it was written, and its earlier negative result (failing on
        # 1 of 4 seeds) was measured by passing beta_se by hand.
        #
        # This is the scale information NUTS cannot bootstrap when the
        # geometry is bad: on raw epileptic the lme pre-fit converges fine
        # (fixef 1.93, 0.0004, -0.087, 0.0006) even though the JOINT MLE
        # does not, so the input exists exactly where adaptation fails.
        # Whether seeding beta is ENOUGH is a separate question - the 796:1
        # imbalance lives in the random effects, and under a non-centred
        # parameterisation b_std is already unit-scale - but beta_1 at
        # ~4e-4 against beta_0 at ~2 is a real diagonal mismatch that
        # seeding addresses directly.
        if (isTRUE(control$seed_mass_matrix) && is.null(control$beta_se)) {
          .bse <- sqrt(.v)
          if (all(is.finite(.bse)) && all(.bse > 0)) control$beta_se <- unname(.bse)
        }
        if (prior_set == "jmbayes2") {
          control$beta_prior_sd <- sqrt(pmin(14400 * .v, 1000))
          # Residual SD, for the empirical-Bayes Gamma prior on sigma_e -
          # matches JMbayes2's sigmas_mean / sigmas_shape = 5.
          control$sigma_e_prior_mean <- stats::sigma(lme_prefit)
        } else {
          # legacy: jmjax's previous priors. beta 38x tighter; sigma_e
          # left NULL so the backend falls back to HalfNormal(2.0).
          control$beta_prior_sd <- sqrt(10 * .v)
          control$sigma_e_prior_mean <- NULL
        }
        control$sigma_b_prior_mean <- sqrt(diag(nlme::getVarCov(lme_prefit)))
      } else {
        warning("empirical-Bayes prior centering: the internal lme() pre-fit ",
                "failed, so uninformative priors are used instead. Set ",
                "control$empirical_bayes_prior = FALSE to suppress this warning.")
      }
    }
  }

  # ---- MCMC warm start from the lme() pre-fit (opt-in) -----------------
  #
  # WHY. NumPyro's default init_to_uniform draws every parameter uniformly
  # on [-2, 2] in unconstrained space. At n = 8,000 with q = 2 that is
  # 16,000 random effects each starting at an independent random point, and
  # warm-up has to travel to the typical set AND estimate a step size in
  # 500 iterations. Measured: identical data, three sampler seeds, 63 / 127
  # / 543 leapfrog steps, the last failing R-hat - and under float64 two of
  # three seeds still failed, always on a variance or covariance component.
  #
  # The radius is not the issue - U(-2,2) has variance 4/3, so a random
  # point sits at radius ~146 against a typical-set shell at ~126. Every
  # COORDINATE is wrong, which is what leaves adaptation reading a journey
  # rather than the posterior. The lme() BLUPs put each coordinate
  # approximately right.
  #
  # This is what JMbayes2 does (R/jm.R initialises betas, log_sigmas, D,
  # the per-subject b and gammas from the lme/coxph objects it is given,
  # and R/jm_fit.R jitters each chain). The cost, for both packages: chains
  # that start together make R-hat LESS sensitive, since Gelman-Rubin
  # assumes overdispersed starts. Opt-in for now, and documented rather
  # than quietly enjoyed.
  #
  # BETA IS SUPPLIED, and the first version of this block wrongly omitted
  # it on the reasoning that standardize_covariates rescales the design
  # matrix "after this point". That is backwards: it rewrites data_long IN
  # PLACE (line ~780) and leaves the formula alone, so the lme() fitted
  # here is already on the standardized scale and fixef() matches the site
  # exactly.
  #
  # Omitting it was not merely over-cautious, it was actively harmful.
  # Supplying a tight sigma_e (~0.3) while leaving beta uniform-random
  # divides residuals of ~3 by 0.3 and squares them over every row. The
  # backend's self-check measured that warm start at potential 306,548
  # against 39,669 for a cold one - 267,000 log-density units WORSE -
  # because a cold start's large random sigma_e cushions its equally bad
  # beta, and this start had the cushion removed while keeping the bad
  # beta. The parameters have to be supplied as a coherent set or not at
  # all.
  #
  # WHAT IS STILL NOT SUPPLIED. alpha, because it is one parameter out of
  # ~16,000 and contributes nothing to the travel problem - JMbayes2 starts
  # its alphas at 0 too. The spline block, likewise. Anything not supplied
  # falls back to init_to_uniform, which was never the problem.
  #
  # scale_time needs no special handling: it rewrites long_formula and
  # random_formula above (line ~1050), so the lme() fitted here is already
  # on the internal time scale and sigma_b/b_std come out in the units the
  # model samples in.
  #
  # The two checks below are PRE-REGISTERED, in the sense that they are the
  # reason this can be trusted at all. The sigma_b back-transformation in
  # this package went wrong three times in three different ways and was
  # only caught by a ratio diagnostic written before the fix; the same
  # discipline applies to an inverse Cholesky. A failure here abandons the
  # warm start rather than starting the chain somewhere worse than random.
  # The backend then runs its own check - comparing the potential energy at
  # this start against a uniform one - and rejects it if it is not actually
  # better.
  #
  # `.ws_auto` records whether the values in control$init_values are ours or
  # the caller's, which decides later whether we may add the spline block to
  # them. A caller who supplied their own starting values asked for those
  # values, not for ours quietly folded in alongside.
  .ws_auto <- FALSE
  # random_effects_method = "wishart_gibbs" samples D_inv - the whole q x q
  # covariance, drawn in closed form by a Wishart-conjugate Gibbs step - and
  # does NOT sample sigma_b or L_corr. The warm start supplies both of those
  # plus b_std, and b_std is only meaningful RELATIVE to the L it was derived
  # from. Under this method that L is silently dropped while b_std is kept,
  # so the start is subject-level values calibrated against a matrix the
  # model never receives. Measured on PBC2: potential 62129.4 against 21483.3
  # for a uniform start. The self-check rejected it correctly, but attempting
  # it costs four initialize_model() calls and emits an alarming warning for
  # a configuration where it cannot ever work. So do not attempt it.
  .ws_incompatible <- identical(control$random_effects_method, "wishart_gibbs")
  if (method %in% c("spline-PH-mcmc", "weibull-PH-mcmc") &&
      isTRUE(control$mcmc_warm_start %||% TRUE) &&
      !.ws_incompatible &&
      is.null(control$init_values)) {
    .ws <- tryCatch({
      .lme_ws <- if (inherits(control$.lme_object, "lme")) {
        control$.lme_object
      } else {
        .rfw <- if (!is.null(random_formula)) {
          stats::as.formula(paste(deparse(random_formula), "|", id_var))
        } else {
          stats::as.formula(paste0("~1 | ", id_var))
        }
        nlme::lme(long_formula, random = .rfw, data = data_long,
                  control = nlme::lmeControl(opt = "optim", msMaxIter = 200,
                                             niterEM = 100, msMaxEval = 500,
                                             returnObject = TRUE))
      }

      .Dw <- as.matrix(nlme::getVarCov(.lme_ws))
      .sdb <- sqrt(diag(.Dw))
      .re <- as.matrix(nlme::ranef(.lme_ws))
      # Align BY NAME. nlme orders by its own grouping levels; a positional
      # match would pair the wrong subjects, and nothing downstream would
      # show it.
      .ord <- match(as.character(long_arr$subj_ids), rownames(.re))
      if (anyNA(.ord)) stop("ranef() subject ids do not match the fitted data")
      .re <- .re[.ord, , drop = FALSE]
      .qw <- ncol(.re)

      # data_long is already standardized (and long_formula already carries
      # any scale_time rewrite), so these are on the model's own scale.
      .vals <- list(beta = unname(nlme::fixef(.lme_ws)),
                    sigma_e = unname(stats::sigma(.lme_ws)))

      if (.qw == 1L) {
        # q = 1 samples `b` directly (no Cholesky, no b_std), so the BLUPs
        # go in unchanged.
        .vals$sigma_b <- unname(.sdb[1])
        .vals$b <- as.numeric(.re[, 1])
      } else {
        # The model draws b_std ~ N(0, I) and sets b = b_std %*% t(L) with
        # L = diag(sigma_b) %*% L_corr, so b_i = L b_std_i and the inverse
        # is b_std_i = L^-1 b_i. numpyro's LKJCholesky support is a LOWER
        # triangular factor, hence t(chol(.)) rather than chol(.).
        .Rc <- .Dw / tcrossprod(.sdb)
        .Lc <- t(chol(.Rc))
        .Lw <- diag(.sdb, nrow = .qw) %*% .Lc

        .e1 <- max(abs(.Lw %*% t(.Lw) - .Dw)) / max(1, max(abs(.Dw)))
        if (!is.finite(.e1) || .e1 > 1e-8) {
          stop(sprintf("L L' does not reproduce D (relative error %.3g)", .e1))
        }
        .bstd <- t(solve(.Lw, t(.re)))
        .e2 <- max(abs(.bstd %*% t(.Lw) - .re)) / max(1, max(abs(.re)))
        if (!is.finite(.e2) || .e2 > 1e-8) {
          stop(sprintf("b_std round-trip failed (relative error %.3g)", .e2))
        }

        .vals$sigma_b <- unname(.sdb)
        .vals$L_corr <- .Lc
        .vals$b_std <- .bstd
      }
      .vals
    }, error = function(e) {
      warning("mcmc_warm_start: could not build starting values (",
              conditionMessage(e), "); using the default start.")
      NULL
    })
    # Previously this overwrote control$init_values unconditionally, so a
    # caller who supplied their own starting values had them silently
    # discarded and replaced by the lme warm start - with no warning, and
    # with fit$convergence$warm_start reporting on values they never
    # supplied. The `is.null(control$init_values)` guard above is the fix;
    # this assignment is now only ever reached when there was nothing to
    # overwrite.
    if (!is.null(.ws)) {
      control$init_values <- .ws
      .ws_auto <- TRUE
    }

    # ---- Seeds for the sites lme() cannot supply ------------------------
    # lme() knows about beta, sigma_e and the random effects. It knows
    # nothing about the baseline hazard, the association parameter, or the
    # survival submodel's own covariates - so those were left to
    # init_to_uniform, and a uniform draw of any of them multiplies a
    # correctly warm-started trajectory and goes through exp().
    #
    # Everything below is ALREADY COMPUTED by this package, at line ~1461,
    # gated to `method %in% c("weibull-PH-aGH", "spline-PH-aGH")`. That
    # gate was written when only the maximum-likelihood paths existed and
    # was never widened when mcmc_warm_start arrived. The consequence was
    # visible in the test log as three warm starts rejected on every run:
    #
    #   functional-forms (weibull)     9448.9 vs 5164.6   log_lambda0/log_shape
    #   baseline-covariates           38912.1 vs 15130.5  gamma
    #   standardize-interaction      189714.4 vs 59458.0  gamma
    #
    # Each block is its OWN tryCatch and runs AFTER control$init_values is
    # set, so a failure here costs only that one seed - it cannot lose the
    # beta/b_std warm start, which is the part that matters most.
    if (isTRUE(.ws_auto)) {
      # (a) Weibull baseline. survreg's AFT parameterisation converts to
      # the model's log h0(t) = log(shape) + (shape-1)*log(t) + log_lambda0
      # as shape = 1/scale and log_lambda0 = -shape * (Intercept). Left
      # uniform, log_shape ~ Normal(0, 1) is initialised on [-2, 2], so
      # shape reaches 7.4 and (shape-1)*log(t) reaches ~15 by t = 10 -
      # the same amplification-through-exp() as W01.
      if (method == "weibull-PH-mcmc") {
        .wbw <- tryCatch({
          .srw <- survival::survreg(surv_formula, data = data_surv,
                                    dist = "weibull")
          .shw <- 1 / .srw$scale
          list(log_shape = log(.shw),
               log_lambda0 = -.shw * unname(stats::coef(.srw)[1]))
        }, error = function(e) NULL, warning = function(w) NULL)
        if (!is.null(.wbw) && all(is.finite(unlist(.wbw)))) {
          control$init_values$log_shape   <- .wbw$log_shape
          control$init_values$log_lambda0 <- .wbw$log_lambda0
        }
      }

      # (b) alpha and gamma, two-stage, mirroring JM's initial.surv():
      # coxph of the event on the FITTED trajectory m_i(T_i). Reuses the
      # lme fit built above rather than fitting a second one. Subjects are
      # aligned BY NAME in both directions, as nlme orders by its own
      # grouping levels and a positional match would pair the wrong
      # subjects with nothing downstream to show it.
      .tsw <- tryCatch({
        .rew <- as.matrix(nlme::ranef(.lme_ws))
        .ordw <- match(as.character(long_arr$subj_ids), rownames(.rew))
        stopifnot(!anyNA(.ordw))
        .rew <- .rew[.ordw, , drop = FALSE]
        .mhw <- as.vector(X_time_surv %*% nlme::fixef(.lme_ws)) +
          rowSums(Z_time_surv *
                    .rew[, seq_len(ncol(Z_time_surv)), drop = FALSE])
        .didw <- match(as.character(long_arr$subj_ids),
                       as.character(data_surv[[id_var]]))
        stopifnot(!anyNA(.didw))
        .ddw <- data_surv[.didw, , drop = FALSE]
        .ddw$.jmjax_mhat <- .mhw
        .cfw <- stats::coef(survival::coxph(
          stats::update(surv_formula, . ~ . + .jmjax_mhat), data = .ddw))
        list(alpha = unname(.cfw[".jmjax_mhat"]),
             gamma = unname(.cfw[setdiff(names(.cfw), ".jmjax_mhat")]),
             # Kept for the spline seed below, which needs alpha AND the
             # trajectory to convert a marginal baseline into a conditional
             # one. Recomputing it there would refit lme() a third time.
             mhat = .mhw)
      }, error = function(e) NULL, warning = function(w) NULL)

      if (!is.null(.tsw) && length(.tsw$alpha) == 1L && is.finite(.tsw$alpha)) {
        # alpha when there is no functional-forms channel, alpha_value when
        # there is; the model samples one or the other, never both, and a
        # name it does not sample is simply never looked up.
        control$init_values$alpha       <- .tsw$alpha
        control$init_values$alpha_value <- .tsw$alpha
        if (length(.tsw$gamma) && all(is.finite(.tsw$gamma))) {
          # as.array(), because reticulate unwraps a length-1 R vector into
          # a Python scalar. The `gamma` site has shape [n_gamma], so with a
          # single baseline covariate a bare numeric arrived 0-dimensional
          # and the model's einsum over subscript "g" had no index to use.
          # The backend now reshapes defensively as well; this keeps the
          # intent visible at the point the value is built.
          control$init_values$gamma <- as.array(as.numeric(.tsw$gamma))
        }
      }

      # Neutral fallbacks. Zero is the null of no association and of no
      # covariate effect - not a guess, and immediately left by NUTS. It
      # applies to the channel extras always (no two-stage estimate exists
      # for delta/area) and to alpha/gamma only when the two-stage above
      # did not produce one.
      for (.anw in c("alpha", "alpha_value", "alpha_delta",
                     "alpha_area", "alpha_area_avg")) {
        if (is.null(control$init_values[[.anw]])) {
          control$init_values[[.anw]] <- 0
        }
      }
    }
  }
  # ---- Time-scale consistency check -----------------------------------
  # The joint model evaluates the longitudinal trajectory AT the survival
  # times, so data_long[[time_var]] and the Surv() time in data_surv must
  # be on the SAME scale. Getting this wrong is easy and silent: rescaling
  # one (e.g. months to years) without the other produces wild
  # extrapolation rather than an error.
  #
  # Observed in practice on the AIDS data, where `obstime` was converted to
  # years but `Time` was left in months: subjects were evaluated at up to
  # 18 "years" against a model fitted over 0-1.5 years, giving predicted
  # values in [-18.7, 14.8] against an observed range of [0, 4.9]. That
  # corrupts the likelihood surface for any sampler or optimizer.
  #
  # A warning rather than an error: a legitimate design can have survival
  # times extending beyond the last longitudinal visit, so this cannot be
  # decided with certainty. The threshold is deliberately loose.
  .tl <- suppressWarnings(range(as.numeric(data_long[[time_var]]), na.rm = TRUE))
  .ts_max <- suppressWarnings(max(as.numeric(surv_arr$T_surv), na.rm = TRUE))
  if (all(is.finite(.tl)) && is.finite(.ts_max) && .tl[2] > 0 &&
      .ts_max > 5 * .tl[2]) {
    warning("time-scale mismatch: the survival times reach ",
            signif(.ts_max, 4), " but the longitudinal '", time_var,
            "' only reaches ", signif(.tl[2], 4), ". The joint model ",
            "evaluates the longitudinal trajectory at the survival times, ",
            "so these must be on the same scale - check that both were ",
            "rescaled if either was (a common cause is converting one from ",
            "months to years but not the other). Extrapolating this far ",
            "beyond the observed range will distort the fit rather than ",
            "raise an error.", call. = FALSE)
  }

  spline_info <- NULL
  if (method %in% c("spline-PH-aGH", "spline-PH-mcmc")) {
    spline_info <- build_spline_knots(
      surv_arr$T_surv,
      n_interior = as.integer(control$n_interior_knots %||% 5L),
      ord = as.integer(control$spline_order %||% 4L),
      placement = control$knot_placement %||% "quantile",
      t_quad = surv_arr$t_quad
    )
    backend_args$B_T <- spline_info$basis(surv_arr$T_surv)

    # ---- Starting values for the spline baseline hazard ---------------
    # Placed here rather than with the other MLE starting values above,
    # because it needs B_T, which only exists once the knots are built.
    #
    # jmjax otherwise starts every spline coefficient at 0, i.e. a flat
    # baseline hazard of exp(0) = 1. JM does better, in initial.surv():
    #     init.fit <- survreg(Surv(Time, d) ~ ., data = dat)
    #     xi   <- 1/init.fit$scale
    #     phi  <- exp(coefs[1])
    #     logh <- -log(phi * xi * dat$Time^(xi - 1))
    #     out$gammas.bs <- as.vector(lm.fit(extra$W2[ind, ], logh)$coefficients)
    # i.e. fit a parametric Weibull, evaluate its log hazard at the event
    # times, and least-squares project that onto the spline basis. The
    # starting baseline is then Weibull-shaped rather than flat.
    #
    # The same idea transfers directly because jmjax's model is
    #     log h0(t) = B(t) %*% W
    # so lm.fit(B_T, log_h0) is the exact analogue. (Note the sign: JM's
    # negation converts survreg's AFT parameterisation to PH; the
    # conversion below does that explicitly instead.)
    #
    # This projection is now used by BOTH paths. The comment here used to
    # read "MLE only for now: ... seeding NUTS was separately measured to be
    # unhelpful - it already recovers scales unaided", and that was true of
    # the case it was measured on: NUTS starting from init_to_uniform
    # EVERYWHERE does recover the baseline scale by itself.
    #
    # It stopped being true when mcmc_warm_start arrived, because that
    # creates a case nobody measured - beta, sigma_e, sigma_b and b_std at
    # the lme optimum while the spline block is still a uniform draw. Under
    # spline_prior = "penalized" that combination is catastrophic, and the
    # arithmetic says why. W01 ~ Normal(0, 10) is initialised uniformly on
    # [-2, 2], so the implied slope W01[2] - W01[1] can be 4, and the RW2
    # construction extrapolates it linearly: w_rest[k] = W01[2] +
    # k*(W01[2] - W01[1]), reaching ~22 by the fifth coefficient. The
    # hazard is exp(B W), and exp(22) is 3.6e9.
    #
    # Measured, on the same data and the same supplied sites:
    #
    #   spline_prior   potential at warm start   at uniform start   used
    #   independent                      860.2             2087.5   TRUE
    #   penalized                1,533,123,067           10,763.9   FALSE
    #
    # The independent prior was fine because its W is sampled directly, so
    # a uniform draw of it is merely mediocre rather than explosive.
    .want_spline_init <- isTRUE(control$init_from_prefit %||% TRUE) &&
      (method == "spline-PH-aGH" ||
       (method == "spline-PH-mcmc" && isTRUE(.ws_auto)))

    if (.want_spline_init && is.null(control$init_spline)) {
      control$init_spline <- tryCatch({
        .sr <- survival::survreg(surv_formula, data = data_surv, dist = "weibull")
        # survreg fits log(T) = mu + sigma * W (extreme value), so in PH terms
        #   shape     = 1/sigma
        #   log h0(t) = log(shape) + (shape - 1) * log(t) - shape * mu
        .shape <- 1 / .sr$scale
        .mu <- unname(stats::coef(.sr)[1])
        .tt <- pmax(as.numeric(surv_arr$T_surv), 1e-8)
        .logh0 <- log(.shape) + (.shape - 1) * log(.tt) - .shape * .mu

        # NOT corrected from marginal to conditional, deliberately.
        # survreg() above omits the longitudinal trajectory, so .logh0 is the
        # MARGINAL log baseline and overstates the conditional one by
        # alpha * m_i(t). Subtracting that was tried and made things WORSE:
        # .logh0 stops being a smooth function of t and becomes a scattered
        # cloud over subjects, so lm.fit() chases the scatter. Measured, the
        # seeded W went from smooth to oscillating on BOTH datasets -
        # pbc2's -0.38..+0.25 became -2.24, -0.56, -2.19, -0.82, ... with
        # tau_w collapsing 60.7 -> 0.25 - while the potential barely moved
        # (167467 -> 166850), which also shows the baseline was never the
        # dominant term. Any future attempt must smooth in t BEFORE
        # projecting, not subtract per-subject values from a per-time curve.
        .co <- stats::lm.fit(as.matrix(backend_args$B_T), .logh0)$coefficients
        if (any(!is.finite(.co))) NULL else unname(.co)
      }, error = function(e) NULL, warning = function(w) NULL)
    }

    # ---- Hand those coefficients to the MCMC warm start -----------------
    # Only when a warm start was actually built above; without one the
    # spline block is left alone, which is the configuration the original
    # "unhelpful" measurement covered and which still holds.
    if (method == "spline-PH-mcmc" && isTRUE(.ws_auto) &&
        !is.null(control$init_spline)) {
      .w <- as.numeric(control$init_spline)
      .ns <- as.integer(spline_info$n_splines)
      .prior <- control$spline_prior %||% "independent"

      if (length(.w) == .ns && all(is.finite(.w))) {
        if (identical(.prior, "penalized")) {
          # Invert the RW2 construction so the sampled sites reproduce .w
          # EXACTLY, rather than approximating it. The model builds
          #   W = c(W01, W01[2] + k*(W01[2]-W01[1]) + sigma_w*cumsum(cumsum(z)))
          # so W01 is the first two coefficients, and z is recovered by
          # second-differencing what is left after the linear part.
          #
          # sigma_w is then chosen as the SD of those second differences,
          # which puts z on the unit scale its N(0,1) prior expects - the
          # alternative, fixing sigma_w and letting z absorb the scale,
          # starts the chain in the tail of z's own prior.
          .nr <- .ns - 2L
          if (.nr >= 1L) {
            .k <- seq_len(.nr)
            .lin <- .w[2] + .k * (.w[2] - .w[1])
            .res <- .w[-(1:2)] - .lin
            .c1 <- c(.res[1], diff(.res))
            .zr <- c(.c1[1], diff(.c1))
            .sw <- if (.nr > 1L) stats::sd(.zr) else 1
            if (!is.finite(.sw) || .sw < 1e-6) .sw <- 1
            .sw <- min(max(.sw, 1e-3), 1e3)
            control$init_values$W01 <- .w[1:2]
            control$init_values$z_step <- .zr / .sw
            control$init_values$tau_w <- 1 / .sw^2
          }
        } else {
          control$init_values$W <- .w
        }
      }

      # alpha and gamma are seeded where the warm start is built, for both
      # MCMC methods rather than only this one - see the two-stage block
      # there. An earlier version zeroed alpha here, which was neutral but
      # wasted an estimate the package already knew how to compute, and
      # left the weibull path unseeded entirely.
    }

    # IMPORTANT: build this with an explicit loop, not apply()+array()
    # reshape. apply(t_quad, 2, basis) flattens each column's [N_sub,
    # n_splines] result in (subject-fastest, then spline-index) order, but
    # wrapping that flat vector in array(dim = c(N_sub, n_quad, n_splines))
    # assumes (subject, quad_node, spline) order - those don't match, and
    # the mismatch silently scrambles which basis value is associated with
    # which (quad_node, spline) pair rather than raising an error. This was
    # caught by test-matches-JM.R and test-recovers-truth.R both showing a
    # badly wrong `alpha` and log-likelihood for the spline model while the
    # (unaffected, B_quad-free) Weibull model matched JM cleanly.
    n_sub <- nrow(surv_arr$t_quad)
    n_quad <- ncol(surv_arr$t_quad)
    B_quad <- array(0, dim = c(n_sub, n_quad, spline_info$n_splines))
    for (k in seq_len(n_quad)) {
      B_quad[, k, ] <- spline_info$basis(surv_arr$t_quad[, k])
    }
    backend_args$B_quad <- B_quad
    backend_args$n_splines <- as.integer(spline_info$n_splines)
  } else if (method == "weibull-PH-mcmc") {
    # The Weibull baseline hazard is computed directly from T_surv/t_quad
    # in Python (build_model()'s baseline_hazard="weibull" branch) and
    # completely ignores B_T/B_quad - these trivial placeholders exist only
    # to satisfy fit_nuts()'s required-argument signature without touching
    # its Python interface for this simpler, spline-free method.
    n_sub <- nrow(surv_arr$t_quad)
    n_quad <- ncol(surv_arr$t_quad)
    backend_args$B_T <- matrix(0, n_sub, 1)
    backend_args$B_quad <- array(0, dim = c(n_sub, n_quad, 1))
    backend_args$n_splines <- 1L
  }

  # ---- init_theta: accept a prior fit's $estimates ---------------------
  # The documentation promises this works ("e.g. from a prior
  # method = 'weibull-PH-aGH' fit's $estimates"), and until now it did
  # not: the list was passed straight through, and np.asarray() on a
  # named list raises "float() argument must be a string or a real
  # number".
  #
  # Two conversions are needed and neither is obvious from the outside.
  # $estimates are on the ORIGINAL scale (sigma_e, not log_sigma_e) and
  # in reporting order; theta is on the working scale and in the backend's
  # layout order. Assembling it is the caller's job only if they know
  # both, which is not a reasonable thing to expect.
  if (!is.null(init_theta)) {
    # A numeric vector is treated as a ready-made theta whether or not it
    # carries names. Requiring names to be NULL was too strict: building
    # one with c(fit$coefficients$betas, ...) propagates names from the
    # first block, and the vector then took the ESTIMATES branch, looked
    # for "beta_0"/"sigma_e", found none, and produced a non-finite theta
    # that was silently discarded.
    #
    # The estimates branch is for a LIST (or a vector whose names actually
    # look like estimate names), which is what the documentation describes.
    .looks_like_estimates <- !is.null(names(init_theta)) &&
      any(grepl("^(beta_[0-9]+|sigma_e|sigma_b[0-9]*|alpha)$", names(init_theta)))
    if (is.numeric(init_theta) && !is.list(init_theta) &&
        !.looks_like_estimates) {
      backend_args$init_theta <- unname(as.numeric(init_theta))
    } else {
      .e <- unlist(init_theta)
      .g <- function(nm, default = NULL) {
        if (nm %in% names(.e)) as.numeric(.e[[nm]]) else default
      }
      .p <- sum(grepl("^beta_[0-9]+$", names(.e)))
      .th <- as.numeric(.e[paste0("beta_", seq_len(.p) - 1L)])
      .th <- c(.th, log(max(.g("sigma_e", 0.5), 1e-8)))
      if (!is.null(.g("sigma_b"))) {
        .th <- c(.th, log(max(.g("sigma_b"), 1e-8)))
      } else {
        .th <- c(.th, log(max(.g("sigma_b0", 0.5), 1e-8)),
                       log(max(.g("sigma_b1", 0.2), 1e-8)),
                       atanh(max(min(.g("rho", 0), 0.95), -0.95)))
      }
      if (method %in% c("weibull-PH-aGH", "weibull-PH-mcmc")) {
        .th <- c(.th, .g("log_lambda0", -2), log(max(.g("shape", 1.2), 1e-8)))
      } else {
        .wn <- grep("^W[0-9]+$", names(.e), value = TRUE)
        .wn <- .wn[order(as.integer(sub("^W", "", .wn)))]
        if (length(.wn)) .th <- c(.th, as.numeric(.e[.wn]))
      }
      .gn <- grep("^gamma_[0-9]+$", names(.e), value = TRUE)
      .gn <- .gn[order(as.integer(sub("^gamma_", "", .gn)))]
      if (length(.gn)) .th <- c(.th, as.numeric(.e[.gn]))
      .th <- c(.th, .g("alpha", 0))

      if (any(!is.finite(.th))) {
        # Name the positions. "non-finite entries" alone sent a diagnostic
        # down the wrong path: the fit fell back to its defaults, reported
        # a log-likelihood at a crude starting point, and that was read as
        # a difference between two likelihood DEFINITIONS.
        warning("init_theta could not be assembled from the supplied ",
                "estimates: entries ", paste(which(!is.finite(.th)),
                collapse = ", "), " of ", length(.th), " are non-finite. ",
                "Common causes are a zero variance component (log(0)) or a ",
                "correlation at +/-1 (atanh). Ignoring init_theta - the fit ",
                "will start from its own defaults.", call. = FALSE)
      } else {
        backend_args$init_theta <- .th
      }
    }
  }
  backend_args$control <- control

  # --- 2. Cross the R -> Python boundary ---
  backend <- .get_backend()
  # Emitted here rather than where the transformation happens, so that a
  # fit rejected by a later validation check (e.g. an unsupported
  # method/covariate combination) does not first tell the user their
  # covariates were standardized and then error out.
  if (!is.null(.std_info) && !is.null(.std_info$message)) message(.std_info$message)

  # An lme object cannot cross into Python; remove it now that the
  # starting values have been taken from it.
  control$.lme_object <- NULL

  # JAX compiles the likelihood and its gradient on first use, and again
  # whenever the array shapes change. On a large problem that is tens of
  # seconds with no output, which reads as a hung session - so say so.
  # Measured: 20.9s at n = 1,000 and 12.3s at n = 8,000, which is 54% and
  # 16% of the respective first fits.
  # Once per session per shape, not once per fit. An earlier version fired
  # on every call, which in a benchmark loop is noise rather than
  # information - and the message is about a cost that is NOT paid again.
  if (isTRUE(control$verbose %||% TRUE) && !isTRUE(getOption("jmjax.quiet"))) {
    .nsub <- tryCatch(length(unique(data_surv[[id_var]])), error = function(e) NA)
    .key <- paste(method, .nsub, sep = "/")
    .seen <- getOption("jmjax.compiled_shapes", character(0))
    if (is.finite(.nsub) && .nsub >= 500 && !(.key %in% .seen)) {
      options(jmjax.compiled_shapes = c(.seen, .key))
      message("compiling the ", method, " model for this data shape ",
              "(", .nsub, " subjects) - paid once per size, not on refits.")
    }
  }

  py_result <- switch(method,
    "weibull-PH-aGH" = backend$weibull_model$fit_mle(!!!backend_args),
    "spline-PH-aGH"  = backend$spline_model$fit_mle(!!!backend_args),
    "spline-PH-mcmc" = backend$mcmc_model$fit_nuts(!!!backend_args),
    "weibull-PH-mcmc" = backend$mcmc_model$fit_nuts(!!!backend_args)
  )

  # reticulate converts a Python dict into a named R LIST, not a named
  # numeric vector - unlist() here so downstream arithmetic (e.g.
  # summary.jmjax's estimates/se) works, while preserving names.
  estimates <- unlist(py_result$estimates)
  se <- if (!is.null(py_result$se)) unlist(py_result$se) else NULL

  # --- 2b. Back-transform beta to the ORIGINAL covariate scale ---------
  # Only reached when control$standardize_covariates = TRUE. With
  #     x_j^c = (x_j - m_j) / s_j
  # the model X^c %*% beta^c equals X %*% beta exactly, provided
  #     beta_j     = beta_j^c / s_j                        (slopes)
  #     beta_0     = beta_0^c - sum_j beta_j^c * m_j / s_j (intercept)
  # Every other parameter is untouched: alpha multiplies the fitted value
  # m_i(t), which is invariant under this reparameterisation, and
  # sigma_b/rho/sigma_e never involve X at all.
  #
  # Applied to the POSTERIOR SAMPLES rather than to the point estimates,
  # so the reported SDs are exact rather than a delta-method
  # approximation. For MLE methods (no samples) the point estimates are
  # transformed and the SEs are flagged, since a correct SE there needs
  # the full vcov and the transformation is not diagonal.
  if (!is.null(.std_info) || !is.null(.time_scale)) {
    # build_long_arrays() constructs the design as
    #     mf <- model.frame(long_formula, data_long)
    #     X  <- model.matrix(long_formula, mf)
    # and X_long is a 3-D array carrying no dimnames, so the column names
    # are reproduced here the same way rather than read off the array.
    #
    # THE BACK-TRANSFORMATION IS A MATRIX OPERATION, NOT TWO ELEMENTWISE
    # ADJUSTMENTS. An earlier version corrected only the intercept and the
    # standardized column itself:
    #     bm[, 1] <- bm[, 1] - bm[, j] * m / s
    #     bm[, j] <- bm[, j] / s
    # That is correct for a purely additive formula and WRONG as soon as
    # the standardized variable appears in an interaction, because
    #     b3 * t * (x - m)/s  =  (b3/s) * t * x  -  (b3 * m/s) * t
    # so the interaction leaks into the TIME main effect - a column the old
    # code never touched. Measured on y ~ time * x with x ~ N(50, 10):
    # the time coefficient came back 0.700 against a true 0.500, and the
    # interaction 0.0399 against 0.004.
    #
    # The general fix: standardization is a linear map on the design, so
    #     X_std = X %*% A     exactly (verified to 5e-13)
    # and therefore beta_original = A %*% beta_sampled. A is recovered by
    # solving that system once, which handles arbitrary interactions,
    # multi-way terms and polynomials without any formula parsing.
    #
    # qr.solve, NOT the normal equations: solve(crossprod(X), ...) squares
    # the condition number, and an uncentred design is precisely the
    # ill-conditioned case that motivates standardizing in the first place
    # (measured cond(X) = 1.7e3 against cond(X'X) = 2.7e6 on the test case).
    #
    # TWO TRANSFORMATIONS, ONE PROJECTION. They act on different things:
    # standardize_covariates rewrites the DATA and leaves the formula
    # alone; scale_time rewrites the FORMULA and leaves the data alone.
    # Both are linear maps on the design, so the reference matrix is
    # always (ORIGINAL formula, ORIGINAL data) and the fitted one is
    # (current formula, current data). One qr.solve covers whichever
    # combination is active.
    .lf_ref <- if (!is.null(.time_scale)) .time_scale$long_formula_orig
               else long_formula
    .dl_ref <- if (!is.null(.data_long_raw)) .data_long_raw else data_long

    .X_o   <- stats::model.matrix(.lf_ref,
                stats::model.frame(.lf_ref, data = .dl_ref))
    .X_s   <- stats::model.matrix(long_formula,
                stats::model.frame(long_formula, data = data_long))
    .xnames <- colnames(.X_o)

    # ---- the random-effects side -------------------------------------
    # Only time scaling touches Z: standardize_covariates excludes any
    # variable appearing in random_formula, which is what keeps A_z
    # DIAGONAL and the sigma_b correction elementwise. That exclusion is
    # load-bearing, so it is asserted rather than assumed.
    #
    # A_z is recovered the same way as A, by solving Z_o %*% A_z = Z_s.
    # Matching Z's columns BY NAME against time_var would look simpler and
    # fails silently for `~ I(time/2)` or `~ poly(time, 2)`, leaving
    # sigma_b untransformed with no warning - the same shape of bug as the
    # standardization interaction leak.
    .Az <- NULL
    if (!is.null(.time_scale)) {
      # Same asymmetry as above: the reference is the ORIGINAL random
      # formula, the fitted one is the substituted version.
      .rf_now <- if (!is.null(random_formula)) random_formula else
                 if (random_effects == "intercept") ~ 1 else
                 stats::as.formula(paste0("~ I(", time_var, "/",
                                           sprintf("%.17g", .time_scale$divisor), ")"))
      .rf_ref <- if (!is.null(.time_scale$random_formula_orig))
                   .time_scale$random_formula_orig else
                 if (random_effects == "intercept") ~ 1 else
                 stats::as.formula(paste0("~", time_var))
      .Az <- tryCatch({
        Zo <- stats::model.matrix(.rf_ref, stats::model.frame(.rf_ref, data = .dl_ref))
        Zs <- stats::model.matrix(.rf_now, stats::model.frame(.rf_now, data = data_long))
        M <- qr.solve(Zo, Zs)
        if (max(abs(Zo %*% M - Zs)) > 1e-6 * max(1, max(abs(Zs)))) NULL else M
      }, error = function(e) NULL)

      # FAIL HARD, do not warn and continue. An earlier version set .Az to
      # NULL and carried on, which returned beta on the ORIGINAL scale and
      # sigma_b on the SCALED one - an internally inconsistent object. The
      # warning said so, but warnings scroll past in long MCMC runs and
      # vanish entirely in batch jobs, and a variance component silently
      # off by a factor of c is exactly the kind of error that survives
      # into a published table. Downstream consumers (broom, custom print
      # and plot methods, predictive utilities) read sigma_b without
      # knowing anything about scaling.
      #
      # Erroring is cheap here because scale_time is OPT-IN: only a user
      # who explicitly asked for scaling can hit it, and the remedy is a
      # single line in their own data.
      # Explicit request vs "auto" part ways here. An explicit
      # scale_time = TRUE or a numeric divisor STOPS: the user asked for
      # scaling, and returning beta on the original scale with sigma_b on
      # the scaled one would be internally inconsistent in a way warnings
      # do not reliably prevent - they scroll past in long MCMC runs and
      # vanish in batch jobs, and downstream consumers read sigma_b knowing
      # nothing about scaling.
      #
      # "auto" DECLINES instead. It was not asked for on this model, so
      # turning a fit that would have run into a hard failure is a
      # regression from the user's point of view, whatever the reasoning.
      # The reason is recorded in .scale_time_status either way.
      .az_problem <- NULL
      if (is.null(.Az)) {
        .az_problem <- "not_recoverable"
      } else if (!.is_diagonal(.Az)) {
        .az_problem <- "non_diagonal"
      }

      if (!is.null(.az_problem)) {
        if (isTRUE(.time_scale$auto)) {
          # Roll the substitution back and carry on unscaled.
          long_formula <- .time_scale$long_formula_orig
          if (!is.null(.time_scale$random_formula_orig))
            random_formula <- .time_scale$random_formula_orig
          .scale_time_status <- paste0("declined_", .az_problem)
          .time_scale <- NULL
          .Az <- NULL
          .A <- NULL
        } else if (identical(.az_problem, "not_recoverable")) {
          stop("scale_time: the random-effects design cannot be recovered as ",
               "a linear map of the original, so sigma_b could not be ",
               "back-transformed. Returning beta on the original scale and ",
               "sigma_b on the scaled one would be internally inconsistent, ",
               "so the fit has been stopped.\n  Remedy: use ",
               "control$scale_time = \"auto\", which declines silently in ",
               "this situation, or scale the time variable in your own data ",
               "(e.g. data$", time_var, " <- data$", time_var, " / ",
               signif(.c, 6), ", and the same for the survival times) and ",
               "refit with control$scale_time = FALSE.", call. = FALSE)
        } else {
          stop("scale_time: the random-effects transformation is not ",
               "diagonal, so sigma_b would need a full congruence ",
               "D = A_z D A_z' rather than a per-component correction, and ",
               "the reported SDs and correlation could not both be right. ",
               "This happens when time enters random_formula through a term ",
               "that shifts rather than scales - log(time), or a spline with ",
               "FIXED knots. The fit has been stopped rather than return ",
               "mismatched scales.\n  Remedy: use control$scale_time = ",
               "\"auto\", which declines silently here, or scale the time ",
               "variable in your own data and refit with ",
               "control$scale_time = FALSE.", call. = FALSE)
        }
      }
      if (!.is_diagonal(.Az)) {
        stop("scale_time: the random-effects transformation is not diagonal, ",
             "so sigma_b would need a full congruence D = A_z D A_z' rather ",
             "than a per-component correction, and the reported SDs and ",
             "correlation could not both be right. This happens when time ",
             "enters random_formula through a term that shifts rather than ",
             "scales - log(time), or a spline with FIXED knots. The fit has ",
             "been stopped rather than return mismatched scales.\n  Remedy: ",
             "scale the time variable in your own data and refit with ",
             "control$scale_time = FALSE.", call. = FALSE)
      }
    }

    .A <- tryCatch(qr.solve(.X_o, .X_s), error = function(e) NULL)
    if (is.null(.A) || !all(is.finite(.A)) ||
        max(abs(.X_o %*% .A - .X_s)) > 1e-6 * max(1, max(abs(.X_s)))) {
      warning("standardize_covariates: could not recover the back-transformation ",
              "for this design (the standardized design is not an exact linear ",
              "map of the original, or the solve failed). beta is being reported ",
              "on the STANDARDIZED scale. Set control$standardize_covariates = ",
              "FALSE to avoid this.")
      .A <- NULL
    }
    if (!is.null(.A)) {
      .ps <- py_result$posterior_samples
      if (!is.null(.ps) && !is.null(.ps$beta)) {
        # samples: list[n_draws] -> numeric[p]
        .bm <- t(sapply(.ps$beta, function(z) as.numeric(unlist(z))))
        # beta_original = A %*% beta_sampled, applied to every draw at once.
        .bm <- .bm %*% t(.A)

        for (j in seq_len(ncol(.bm))) {
          nm <- paste0("beta_", j - 1)
          if (nm %in% names(estimates)) estimates[[nm]] <- mean(.bm[, j])
          if (!is.null(se) && nm %in% names(se))  se[[nm]] <- stats::sd(.bm[, j])
        }
        py_result$posterior_samples$beta <-
          lapply(seq_len(nrow(.bm)), function(i) .bm[i, ])

        # beta_corrected (SEPARATE, OPT-IN TRACK - see the "beta_corrected"
        # block in mcmc_model.py's fit_nuts(), added for
        # dev/study_calibration.R's analytical-correction test) is exactly
        # the same kind of p-dimensional coefficient vector as beta, on the
        # same (standardized) scale it was sampled on - the identical
        # standardized-to-original map applies unchanged. Absent unless
        # control$orthogonalize_b/_b0 found at least one absorbable
        # direction, so this only ever ADDS a transform; it is never
        # required and never touches beta itself.
        if (!is.null(.ps$beta_corrected)) {
          .bmc <- t(sapply(.ps$beta_corrected, function(z) as.numeric(unlist(z))))
          .bmc <- .bmc %*% t(.A)
          py_result$posterior_samples$beta_corrected <-
            lapply(seq_len(nrow(.bmc)), function(i) .bmc[i, ])
        }

        # ---- and the DIAGNOSTICS, which were being left behind ----------
        # estimates, se and posterior_samples above are all rewritten to
        # the original covariate scale. diagnostics$ess and $rhat were not:
        # they are computed in Python during sampling, on the STANDARDIZED
        # parameters. So fit$estimates[["beta_0"]] and
        # fit$diagnostics$ess[["beta_0"]] described DIFFERENT QUANTITIES
        # whenever standardization was active - which is the default.
        #
        # It also quietly corrupted cross-package benchmarking: a min-ESS
        # comparison against another package reads this field, and was
        # therefore comparing jmjax's standardized-scale ESS against the
        # other package's natural-scale ESS.
        #
        # Recomputed with numpyro's own summary rather than a second
        # estimator written in R, so every ESS the package reports comes
        # from one implementation. Failure here leaves the old values and
        # warns rather than dropping the diagnostics entirely - a
        # mismatched ESS is bad, but a missing one is worse.
        .nc <- as.integer(control$num_chains %||% 1L)
        .rd <- tryCatch(
          .get_backend()$mcmc_model$recompute_site_diagnostics(.bm, .nc),
          error = function(e) { 
            warning("standardize_covariates: could not recompute beta ",
                    "diagnostics on the original scale (", conditionMessage(e),
                    "); fit$diagnostics for beta_* remain on the ",
                    "standardized scale and do not correspond to ",
                    "fit$estimates.", call. = FALSE)
            NULL
          })
        if (!is.null(.rd)) {
          .ne <- as.numeric(unlist(.rd$n_eff)); .rh <- as.numeric(unlist(.rd$r_hat))
          for (j in seq_len(ncol(.bm))) {
            nm <- paste0("beta_", j - 1)
            if (j <= length(.ne) && nm %in% names(py_result$diagnostics$ess)) {
              py_result$diagnostics$ess[[nm]] <- .ne[j]
            }
            if (j <= length(.rh) && nm %in% names(py_result$diagnostics$rhat)) {
              py_result$diagnostics$rhat[[nm]] <- .rh[j]
            }
          }
        }
      } else {
        # ---- MLE path: the same projection, applied exactly -------------
        # beta_original = A %*% beta_sampled is linear, so the sampling
        # covariance propagates exactly as A %*% vcov %*% t(A) - not a
        # delta-method approximation. The same A recovered above is used,
        # so interactions are handled identically on both paths.
        .p_beta <- sum(grepl("^beta_[0-9]+$", names(estimates)))
        if (.p_beta > 0 && nrow(.A) == .p_beta) {
          .bn <- paste0("beta_", seq_len(.p_beta) - 1L)
          estimates[.bn] <- as.numeric(.A %*% as.numeric(estimates[.bn]))

          if (!is.null(py_result$vcov)) {
            V <- tryCatch(as.matrix(do.call(rbind, lapply(py_result$vcov, unlist))),
                           error = function(e) NULL)
            if (!is.null(V) && nrow(V) >= .p_beta) {
              bi <- seq_len(.p_beta)
              V[bi, bi] <- .A %*% V[bi, bi, drop = FALSE] %*% t(.A)
              if (ncol(V) > .p_beta) {
                oi <- setdiff(seq_len(ncol(V)), bi)
                V[bi, oi] <- .A %*% V[bi, oi, drop = FALSE]
                V[oi, bi] <- t(V[bi, oi, drop = FALSE])
              }
              py_result$vcov <- V
              if (!is.null(se)) se[.bn] <- sqrt(pmax(diag(V)[bi], 0))
            } else if (!is.null(se)) {
              warning("standardize_covariates: could not reshape vcov to a ",
                      "matrix, so beta standard errors are left on the ",
                      "STANDARDIZED scale while the estimates are on the ",
                      "original scale. Do not read the two together.")
          }
        }
      }
    }
    }

  # sigma_b: Z_s %*% b_s = Z_o %*% b_o gives b_o = A_z %*% b_s, so
  # D_o = A_z D_s t(A_z). A_z is diagonal (asserted above), so that
  # reduces to dividing each SD by its own divisor, with rho untouched
  # since correlation is scale-invariant.
  #
  # OUTSIDE the MCMC branch deliberately. An earlier version had this
  # inside `if (!is.null(.ps$beta))`, so it never ran for the aGH
  # methods - sigma_b1 came back a clean factor of c wrong on every MLE
  # fit, which is exactly what the debug run showed (ratio 5.0757 for
  # c = 5).
  if (!is.null(.Az)) {
    .dz <- diag(.Az)
    .psb <- py_result$posterior_samples
    for (k2 in seq_along(.dz)) {
      for (nm in c(paste0("sigma_b", k2 - 1L),
                   if (length(.dz) == 1L) "sigma_b")) {
        # MULTIPLY, do not divide. Z_s = Z_o %*% A_z, so b_o = A_z %*% b_s
        # and sigma_b_o = diag(A_z)[k] * sigma_b_s. With divisor c the
        # substituted column is time/c, so diag(A_z)[k] = 1/c - dividing by
        # it MULTIPLIES by c and lands c^2 from the truth. Measured with
        # c = 5: sigma_b1 came back 24.99x the unscaled value, and 25 = c^2
        # named the direction error outright. beta was unaffected because
        # it uses the matrix itself (.bm %*% t(.A)), not its diagonal.
        if (nm %in% names(estimates)) estimates[[nm]] <- estimates[[nm]] * .dz[k2]
        if (!is.null(se) && nm %in% names(se)) se[[nm]] <- se[[nm]] * .dz[k2]
        if (!is.null(.psb) && !is.null(.psb[[nm]])) {
          py_result$posterior_samples[[nm]] <-
            lapply(.psb[[nm]], function(z) as.numeric(unlist(z)) * .dz[k2])
        }
      }
    }
  }
  }

  # ---- Did the optimizer actually reach a stationary point? ----------
  # scipy's `converged` flag says it stopped cleanly, not that it arrived.
  # On prothro with a spline baseline, L-BFGS-B reported converged = TRUE
  # after 5 iterations with zero non-finite standard errors - and had left
  # 78 log-likelihood units behind, returning alpha -0.0008 where JM, BFGS
  # and jmjax's own MCMC all agree near -0.04.
  #
  # A stationary point has a small gradient. The threshold below is
  # deliberately loose: it is meant to catch a line search that quit, not
  # to adjudicate fine convergence, and a false alarm costs a user one
  # extra fit while a missed one costs them a wrong estimate.
  .gm <- suppressWarnings(as.numeric(py_result$convergence$grad_max %||% NA))
  # The threshold is RELATIVE to the log-likelihood, not absolute. A first
  # version used 1e-3 and fired on every fit - including BFGS at 2.29 and
  # trust-constr at 8.0, which are the GOOD answers on prothro, against
  # L-BFGS-B's 31.9 which is the bad one. A warning that fires every time
  # is noise, and noise is what gets ignored when it matters.
  #
  # Scaling by |loglik| separates them: on prothro (loglik ~ -14000) the
  # bar lands near 14, so 31.9 warns and 2.29 does not.
  .llv <- suppressWarnings(as.numeric(py_result$loglik %||% NA))
  .gtol <- if (is.finite(.llv)) max(1e-3, 1e-3 * abs(.llv)) else 1e-3
  # `converged` is now the SAME gradient test, so this fires exactly when
  # the flag is FALSE - the message explains what the flag means rather
  # than adding a second, differently-calibrated opinion.
  if (is.finite(.gm) && .gm > .gtol) {
    warning("the fit did not reach a stationary point: the largest ",
            "gradient component is ", signif(.gm, 3),
            " against a threshold of ", signif(.gtol, 3),
            ". fit$convergence$converged is FALSE for this reason. Note ",
            "the optimizer's own status (fit$convergence$optimizer_success) ",
            "may say otherwise - it reports which stopping rule fired, not ",
            "whether an optimum was reached, and on this class of problem ",
            "it is wrong in both directions.\n  Try ",
            "control$opt_method = \"BFGS\", and compare fit$loglik between ",
            "the two: the run reaching the HIGHER log-likelihood is the ",
            "better fit.", call. = FALSE)
  }

  # --- 3. Wrap result for R-side summary/print/plot methods ---

  structure(
    list(
      call = match.call(),
      method = method,
      long_formula = long_formula,
      surv_formula = surv_formula,
      spline_info = spline_info,
      estimates = estimates,
      se = se,
      vcov = py_result$vcov,
      loglik = if (is.numeric(py_result$loglik)) py_result$loglik else NULL,
      # Precision is stamped here, at the single point every method's
      # result passes through, rather than inside each Python fitting
      # function - so MCMC and maximum-likelihood fits carry it alike and
      # there is one place for it to be right.
      #
      # This exists because of a concrete failure. jmjax's benchmark
      # corpus was produced under float32 while float64 later measured
      # 4.24x faster at n = 8,000, and nothing in any saved fit recorded
      # which arithmetic had produced it - so the whole corpus had to be
      # treated as suspect rather than re-checked selectively. A fit
      # should describe the conditions that produced it.
      convergence = local({
        .cv <- py_result$convergence
        if (is.null(.cv)) .cv <- list()
        .cv$precision <- .backend_precision()
        .cv
      }),
      # Opt-in and purely ADDITIVE: control$return_backend_data = TRUE
      # attaches the arrays handed to Python. Off by default because they
      # are large, and returning them unconditionally would bloat every
      # fit object.
      #
      # It exists so em.py can check its reimplementation of h(b) against
      # the backend's own likelihood on the SAME data. That module is kept
      # deliberately separate from the fitting paths, so it cannot import
      # the closures it mirrors - and an unverifiable reimplementation of
      # a model is worse than no reimplementation.
      backend_data = if (isTRUE(control$return_backend_data)) backend_args else NULL,
      # Always present, not only on failure. One of: "off",
      # "applied (c = ...)", "declined_non_diagonal",
      # "declined_not_recoverable", "declined_no_plain_time",
      # "declined_no_constant". Lets a user audit programmatically why the
      # scaling did or did not happen, which matters most under
      # scale_time = "auto", where declining is silent by design.
      scale_time_status = .scale_time_status,
      diagnostics = py_result$diagnostics,        # NULL for MLE methods; rhat/ess lists for MCMC
      random_effects = py_result$random_effects,  # NULL for MLE methods; per-subject b for MCMC
      n_subjects = length(long_arr$subj_ids),
      n_obs_long = sum(long_arr$n_obs),
      n_events = sum(surv_arr$event),
      posterior_samples = py_result$posterior_samples  # NULL unless MCMC
    ),
    class = "jmjax"
  )
}

# Small helper, avoids a hard dependency on rlang for one operator
`%||%` <- function(a, b) if (is.null(a)) b else a

# TRUE when M is (numerically) diagonal. A q=1 model gives a 1x1 A_z whose
# off-diagonal set is EMPTY, and max(numeric(0)) is -Inf with a warning -
# the comparison then happens to give the right answer for the wrong
# reason. Handled explicitly instead.
.is_diagonal <- function(M, tol = 1e-8) {
  if (is.null(M)) return(FALSE)
  M <- as.matrix(M)
  if (nrow(M) != ncol(M)) return(FALSE)
  if (ncol(M) < 2L) return(TRUE)
  max(abs(M[row(M) != col(M)])) <= tol
}

