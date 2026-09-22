"""
Full Bayesian (NUTS) fit of the spline-baseline joint model.

Unlike the two MLE backends, this supports multivariate random effects
(random intercept + slope) because NUTS samples the random effects directly
as parameters rather than marginalizing them via quadrature.

Return-value design (this is the piece that was refactored from the first
working version): population-level parameters (beta, sigma_e, sigma_b,
alpha, W) and per-subject random effects (b) are kept SEPARATE, rather than
flattened into one estimates vector - a fit with 300 subjects would
otherwise produce a 300+ element "estimates" table dominated by individual
subject effects, burying the parameters actually of interest. Convergence
is assessed via NumPyro's real split-R-hat and effective sample size
diagnostics (numpyro.diagnostics.summary), not a placeholder.
"""
import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCGibbs
from numpyro.infer.util import log_density as numpyro_log_density
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.contrib.control_flow import scan as numpyro_scan

RHAT_CONVERGED_THRESHOLD = 1.05


def _absorbable_basis(X_long, Z_long, n_obs, q_idx, tol=1e-8):
    """Basis for the directions of random-effect column `q_idx` that the
    fixed effects can absorb.

    A shift of b[:, q] by s_i (subject-constant) changes the linear
    predictor by s_i * Z[:, :, q]. That shift is absorbed by beta - leaving
    fitted values unchanged - exactly when some column j of X satisfies

        X[:, :, j]  ==  s_i * Z[:, :, q]      for a subject-constant s_i

    so the absorbable space is spanned by the s_i vectors of every such
    column. The data decide which columns qualify; nothing is assumed about
    the formula.

    For q = 0 the random-effect column is the constant 1, so the condition
    reduces to "column j is constant within subject" and this recovers the
    intercept case. For q = 1 (a random slope on time) it picks up `time`
    itself, giving the sum-to-zero constraint on b_1, and would also pick up
    any covariate-by-time interaction present in X.

    Returns (Q, kept) with Q an [N_sub, k] orthonormal basis, or (None, []).

    Computed in fit_nuts() where X_long and Z_long are concrete arrays:
    inside the traced model "which columns qualify" would be a tracer and
    could not select columns at all.
    """
    X = np.asarray(X_long, dtype=float)
    Z = np.asarray(Z_long, dtype=float)
    if X.ndim != 3 or Z.ndim != 3 or q_idx >= Z.shape[2]:
        return None, []
    N_sub, max_obs, p = X.shape
    n = np.asarray(n_obs).astype(int).reshape(-1)
    if n.shape[0] != N_sub or N_sub == 0 or p == 0:
        return None, []

    mask = np.arange(max_obs)[None, :] < n[:, None]          # [N_sub, max_obs]
    zq   = Z[:, :, q_idx]                                     # [N_sub, max_obs]
    nz   = mask & (np.abs(zq) > tol)                          # usable cells
    zsc  = max(float(np.nanmax(np.abs(zq[mask]))) if mask.any() else 1.0, 1.0)

    keep, svals = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(p):
            xj = X[:, :, j]
            # Wherever Z_q is ~0 the column must be ~0 too, or no constant
            # s_i can reproduce it.
            bad = mask & ~nz & (np.abs(xj) > tol * max(
                float(np.nanmax(np.abs(xj[mask]))) if mask.any() else 1.0, 1.0))
            if bad.any():
                continue
            r = np.where(nz, xj / np.where(np.abs(zq) > tol, zq, 1.0), np.nan)
            rmin = np.nanmin(r, axis=1); rmax = np.nanmax(r, axis=1)
            spread = np.nanmax(rmax - rmin)
            if not np.isfinite(spread):
                continue
            scale = max(float(np.nanmax(np.abs(rmax))), 1.0)
            if spread <= tol * scale * zsc:
                # BUG FIX: a subject with no cell where |Z_q| > tol (e.g.
                # the only observation is at Z_q = 0, such as a baseline-
                # only trajectory when q is the time-slope column) has no
                # row from which to determine s_i at all - rmin/rmax are
                # NaN for them, not "the ratio is 0". Their own data is
                # silent on s_i, not evidence for s_i = 0: with Z_iq = 0
                # everywhere for that subject, ANY s_i leaves their fitted
                # values unchanged, so imputing the consensus value from
                # the subjects whose data DOES determine it (already known,
                # by the spread check just above, to agree with each other
                # to within tol) keeps the basis column the single uniform
                # direction it is supposed to be. The old fallback (0.0)
                # silently broke that uniformity - see
                # dev/diag_slope_projection.R: at n=500 with several
                # baseline-only subjects, Q[:,0] for the slope column came
                # back non-uniform (sd 5.3e-3, some entries exactly 0)
                # instead of constant, so the projection removed a
                # subject-WEIGHTED quantity instead of the simple mean,
                # and mean_i(b_i1) was left as large as 0.04-0.07 instead
                # of pinned to ~1e-16 like the intercept column always was.
                #
                # THE STRONGER REASON (Proposition 5, Section 4.6 of
                # vignette("jmjax-reparameterization")). Restoring
                # uniformity is the visible symptom, not the actual
                # justification, and stating only the symptom invites a
                # future reader to "simplify" this back. Read through the
                # LONGITUDINAL submodel alone, a silent subject's s_i is
                # genuinely free - any value leaves their fitted values
                # unchanged, which is what the paragraph above says. But b
                # also reaches the likelihood through the shared trajectory
                # m_i(t) at the event time and the quadrature nodes, and
                # there Z_q is generally NOT zero for that subject. The
                # compensating beta shift is already pinned by the subjects
                # whose data does determine the ratio, so the silent
                # subject's entry is forced to that same consensus value -
                # it is not free at all in the JOINT model. The old 0.0
                # fallback therefore produced a direction that is not
                # absorbable, i.e. one the survival submodel does identify,
                # and sweeping it changed the model rather than only its
                # coordinates. The median over the determined subjects is
                # the unique choice consistent with joint-model invariance.
                det = np.isfinite(rmin)
                s_fill = float(np.nanmedian(rmin)) if det.any() else 0.0
                s_i = np.where(det, rmin, s_fill)
                keep.append(int(j)); svals.append(s_i)
    if not keep:
        return None, []

    S = np.stack(svals, axis=1)                               # [N_sub, k]
    Q, R = np.linalg.qr(S)
    r = np.abs(np.diag(R))
    if r.size == 0:
        return None, []
    good = r > tol * max(float(r.max()), 1.0)
    Q = np.ascontiguousarray(Q[:, good], dtype=float)
    if Q.shape[1] == 0:
        return None, []
    return Q, [keep[i] for i, g in enumerate(good) if g]


def _absorbable_basis_exact(X_long, Z_long, n_obs, q_idx, tol=1e-8):
    """Basis-INDEPENDENT absorbable basis for random-effect column `q_idx`.

    WHY THIS EXISTS ALONGSIDE _absorbable_basis(). The column-wise search
    above asks, of each fixed-effects column separately, "is this column a
    subject-constant multiple of Z_q?". But a shift in b[:, q] is absorbed
    by beta whenever SOME LINEAR COMBINATION of X's columns reproduces it -
    the absorbable set depends only on the column SPACE of X, not on which
    basis for that space model.matrix() happened to emit. Those are not the
    same question, and the gap is not academic: with

        X = [1, t + t^2/2, t^2]     (span{1, t, t^2}, i.e. what poly() or
                                     ns() emit for a smooth longitudinal
                                     mean)

    no single column is a subject-constant multiple of `time`, so the
    column-wise search returns NOTHING for the slope column - yet
    d = (0, 1, -1/2) gives X d = t exactly, so mean(b_i1) is absorbable and
    the beta_1 location degeneracy is fully present. Worse, the failure is
    silent: the intercept column still yields a basis, so the "no absorbable
    direction" warning does not fire either. See
    vignette("jmjax-reparameterization"), Section 4.5 (Example 1).

    HOW. v is absorbable iff there is a beta-shift d with A_i d = v_i z_i
    for every subject i, where A_i is subject i's fixed-effects block and
    z_i its Z_q column. Read as a condition on d, that says A_i d must lie
    in span(z_i) for every i - a condition in R^p, NOT in R^N_sub. So

        D = ker( stack_i [ (I - P_i) A_i ] ),   P_i = proj onto span(z_i)

    is the null space of a [sum_i n_i, p] matrix (p is small: one SVD, no
    loop over columns, 0.06s at N_sub = 8000), and each d in a basis of D
    induces the absorbable direction v_i = (z_i^T A_i d) / (z_i^T z_i).
    Subjects with z_i = 0 contribute the constraint A_i d = 0 (their own
    data cannot determine a ratio) and take the consensus ratio, for the
    reason given in Proposition 5 of the vignette - the survival submodel
    DOES identify their entry even when the longitudinal one does not.

    Every candidate direction is verified against the defining identity
    before being returned, so the result is sound by construction rather
    than by tolerance choice: a direction that only NEARLY satisfies
    A_i d = v_i z_i is discarded rather than swept.

    Returns (Q, gens) with Q an [N_sub, k] orthonormal basis, or (None, []).
    `gens` is the list of (v, d) generator pairs, used by
    _structural_extension_residual() to check Condition (S).
    """
    X = np.asarray(X_long, dtype=float)
    Z = np.asarray(Z_long, dtype=float)
    if X.ndim != 3 or Z.ndim != 3 or q_idx >= Z.shape[2]:
        return None, []
    N_sub, max_obs, p = X.shape
    n = np.asarray(n_obs).astype(int).reshape(-1)
    if n.shape[0] != N_sub or N_sub == 0 or p == 0:
        return None, []

    mask = (np.arange(max_obs)[None, :] < n[:, None]).astype(float)
    flat = mask.reshape(-1) > 0
    if int(flat.sum()) < p:
        return None, []                      # fewer observations than columns
    zq = Z[:, :, q_idx] * mask
    Xm = X * mask[:, :, None]

    c = np.einsum("no,no->n", zq, zq)                     # z_i^T z_i
    num = np.einsum("no,nop->np", zq, Xm)                 # z_i^T A_i
    cmax = max(float(c.max()) if c.size else 1.0, 1.0)
    det = c > (tol ** 2) * cmax              # subjects whose data fixes v_i
    ratio = np.zeros((N_sub, p))
    ratio[det] = num[det] / c[det][:, None]

    # (I - P_i) A_i, which reduces to A_i exactly where z_i = 0.
    C = (Xm - zq[:, :, None] * ratio[:, None, :]).reshape(N_sub * max_obs, p)[flat]
    X_f = Xm.reshape(N_sub * max_obs, p)[flat]
    colnorm = np.linalg.norm(X_f, axis=0)    # so the rank test is unit-free
    scale = np.where(colnorm > tol, colnorm, 1.0)

    # full_matrices=False keeps U at [M, p] rather than [M, M]; M can be
    # ~10^5, so the full form would allocate hundreds of GB.
    _, sv, Vt = np.linalg.svd(C / scale, full_matrices=False)
    smax = float(sv.max()) if sv.size else 0.0
    rank = int((sv > tol * max(smax, 1.0)).sum())
    if rank >= p:
        return None, []
    D = Vt[rank:].T / scale[:, None]                      # [p, dim D]

    zq_f = zq.reshape(-1)[flat]
    subj = np.repeat(np.arange(N_sub), max_obs)[flat]
    gens = []
    for k in range(D.shape[1]):
        d = D[:, k]
        v = ratio @ d
        if det.any() and not det.all():
            v = np.where(det, v, float(np.median(ratio[det] @ d)))
        if np.abs(v).max() <= tol:
            continue
        lhs = X_f @ d
        rhs = v[subj] * zq_f
        denom = max(float(np.abs(rhs).max()), float(np.abs(lhs).max()), 1.0)
        if float(np.abs(lhs - rhs).max()) / denom > 1e-7:
            continue                         # not genuinely absorbable
        gens.append((v, d))
    if not gens:
        return None, []

    S = np.stack([g[0] for g in gens], axis=1)
    Q, R = np.linalg.qr(S)
    r = np.abs(np.diag(R))
    good = r > tol * max(float(r.max()), 1.0)
    Q = np.ascontiguousarray(Q[:, good], dtype=float)
    if Q.shape[1] == 0:
        return None, []
    return Q, gens


def _structural_extension_residual(gens, q_idx, X_time_surv, Z_time_surv,
                                    X_time_quad, Z_time_quad):
    """Check Condition (S): does absorbability extend off the observed grid?

    The ratio test runs on the longitudinal grid {t_ij}, but `b` also
    reaches the likelihood through the shared trajectory m_i(t), evaluated
    at the event time T_i and at the quadrature nodes. Invariance there is
    a SEPARATE requirement: it needs f_j(t) = s_i g_q(t) as an identity in
    t, not merely at the observed times. It holds by construction for every
    column type this option targets (a subject-constant column, `time`
    itself, a covariate-by-time interaction), but "holds by construction"
    and "holds for the arrays actually passed in" are different claims, and
    checking costs one einsum over arrays that are already built.

    Returns the worst relative residual over the generators; the caller
    warns if it is not negligible.
    """
    if not gens:
        return 0.0
    worst = 0.0
    Xs = np.asarray(X_time_surv, dtype=float)
    Zs = np.asarray(Z_time_surv, dtype=float)
    Xq = np.asarray(X_time_quad, dtype=float)
    Zq = np.asarray(Z_time_quad, dtype=float)
    for v, d in gens:
        for lhs, rhs in ((Xs @ d, v * Zs[:, q_idx]),
                          (np.einsum("nkp,p->nk", Xq, d),
                           v[:, None] * Zq[:, :, q_idx])):
            denom = max(float(np.abs(rhs).max()), float(np.abs(lhs).max()), 1.0)
            worst = max(worst, float(np.abs(lhs - rhs).max()) / denom)
    return worst


def _absorbable_generator_matrix(Q, gens, tol=1e-8):
    """Beta-shift matrix matching Q's columns - the analytical-correction
    track's only new piece of linear algebra.

    SEPARATE, OPT-IN TRACK. Not used by orthogonalize_b/_b0's own sampling
    at all (that only ever needs Q, not this). It exists for the
    "beta_corrected" post-processing block in fit_nuts(), added after
    dev/study_calibration.R's pilot run showed that orthogonalize_b/_b0
    reproduce random_effects_method = "wishart_gibbs_centered"'s retracted
    failure mode: unbiased point estimates but too-narrow credible
    intervals for exactly the swept parameter(s), because the swept
    component of b_raw is unidentified under the likelihood and simply
    samples its prior, with nothing downstream reflecting that uncertainty.
    See vignette("jmjax-reparameterization"), Section 9.

    WHY THIS RECOVERS WHAT IS NEEDED. _absorbable_basis_exact() already
    verifies, for each raw generator pair (v, d) in `gens`, the identity

        X_i @ d  ==  v_i * Z_i,q(t)      for every subject i (all t)

    - that is what "absorbable" means, and it is exactly the identity
    orthogonalize_b/_b0 relies on to subtract Q's columns from b[:, q]
    without changing any fitted value. Q is then an ORTHONORMAL basis
    (via QR) of span(v_1, v_2, ...), so Q's own columns are themselves
    linear combinations of those v's - and because the map d -> v = ratio @ d
    is LINEAR, the SAME combination of the matching d's satisfies the
    identity for Q's columns instead of the raw generators':

        X_i @ Dmat[:, m]  ==  Q[:, m]_i * Z_i,q(t)

    which is precisely the beta-shift a caller needs: subtracting
    Q[:, m] * c_m from b[:, q] changes the fitted values by
    -c_m * Q[:, m]_i * Z_i,q(t) = -c_m * X_i @ Dmat[:, m], so adding
    c_m * Dmat[:, m] back to beta restores them. Summed over m via
    c = Q.T @ b_raw[:, q] (the actual swept component of a given draw),
    this is exactly what recovers an unconstrained-equivalent beta from an
    orthogonalized fit's own posterior draws, with no refit.

    HOW THE COMBINATION IS FOUND. Q lies in span(S) exactly by
    construction (S = the raw v's stacked as columns), so solving
    Q = S @ coeffs by least squares recovers `coeffs` to numerical
    precision; Dmat = Dg @ coeffs (Dg = the raw d's stacked as columns)
    applies that same combination to the d side. Verified before being
    returned, in the same spirit as _absorbable_basis_exact's own
    per-generator check - a silent near-miss here would surface as a
    biased, not just mis-scaled, beta_corrected.

    Returns a [p, k] array (k = Q.shape[1]), or None (with a warning) if
    the recovery does not check out - defensive; should not happen for any
    Q actually returned by _absorbable_basis_exact.
    """
    if Q is None or not gens:
        return None
    S = np.stack([g[0] for g in gens], axis=1)            # [N_sub, k0]
    Dg = np.stack([g[1] for g in gens], axis=1)            # [p, k0]
    coeffs, _, _, _ = np.linalg.lstsq(S, Q, rcond=None)    # [k0, k]
    resid = float(np.abs(S @ coeffs - Q).max())
    scale = max(float(np.abs(Q).max()), 1.0)
    if resid > 1e-6 * scale:
        warnings.warn(
            "orthogonalize_b: could not recover a matching beta-shift basis "
            "for the analytical-correction track (residual %.2e); "
            "beta_corrected will not be reported for this column." % resid,
            RuntimeWarning, stacklevel=2)
        return None
    return Dg @ coeffs                                     # [p, k]


def build_model(p, q, n_splines, max_obs, alpha_prior_sd=2.0,
                 sigma_e_prior_mean=None, sigma_e_prior_shape=5.0,
                 lkj_concentration=3.0,
                 gamma_prior_mean=None, gamma_prior_sd=2.0,
                 baseline_hazard="spline",
                 spline_prior="independent",
                 spline_penalty_shape=5.0, spline_penalty_rate=0.5,
                 beta_prior_mean=None, beta_prior_sd=None,
                 sigma_b_prior_mean=None, sigma_b_prior_shape=5.0,
                 random_effects_corr=True,
                 random_effects_method="nuts",
                 rw2_implementation="scan",
                 wishart_eb_scale=False,
                 b_orth_bases=None):
    """
    Factory for the joint-model log-density, shared by fit_nuts() (which
    runs NUTS against it) and evaluate_log_density() (which evaluates it at
    arbitrary parameter values, e.g. the true simulation parameters, for
    diagnostic purposes). Extracting this ensures both code paths use the
    EXACT same model - a hand-reimplemented formula for diagnostic purposes
    would risk a subtle mismatch that defeats the point of a rigorous check.

    baseline_hazard:
      "spline" (default) - B-spline approximated log baseline hazard, per
        spline_prior below. B_T/B_quad must be real spline basis matrices.
      "weibull" - closed-form Weibull baseline hazard (log(shape) +
        (shape-1)*log(t) + log_lambda0), matching the ALREADY-VALIDATED MLE
        Weibull backend's parameterization exactly (weibull_model.py) -
        added specifically to isolate whether an alpha discrepancy seen
        under the spline model is about spline/quadrature specifics or
        something more fundamental in the random-effects/association
        mechanism, by removing splines from the picture entirely. B_T/B_quad
        are accepted but IGNORED in this branch (pass trivial placeholders).

    spline_prior, spline_penalty_shape/rate: only used when
    baseline_hazard="spline" - see spline-specific docs below.
      "independent" - W ~ iid Normal(0, 3), no smoothing at all.
      "penalized" - RW2 + Gamma precision hyperprior, matching JMbayes2's
        P-spline approach (shape=5.0, rate=0.5 CONFIRMED to match JMbayes2's
        actual defaults via fit$priors$A_tau_bs_gammas/$B_tau_bs_gammas).

    beta_prior_mean/sd, sigma_b_prior_mean/shape: EMPIRICAL-BAYES prior
    centering, matching JMbayes2's confirmed approach (see fit$priors$
    D_sds_mean, $gamma_prior_D_sds - directly verified against a real
    JMbayes2 fit during development). Applies regardless of baseline_hazard.
    If None, falls back to jmjax's original uninformative priors
    (Normal(0,5) for beta, HalfNormal(2) for sigma_b).

    NOTE: evaluate_log_density()/find_map_b_std() (used in the profile-
    likelihood diagnostic work) currently only support baseline_hazard=
    "spline" with spline_prior="independent" and no empirical-Bayes
    centering - extend those functions if a similarly rigorous check is
    needed under other settings.
    """
    def model(X_long, y_long, n_obs, X_time_surv, X_time_quad,
              T_surv, event, t_quad, gk_weights, B_T, B_quad,
              Z_long, Z_time_surv, Z_time_quad, N_sub,
              X_delta_surv=None, X_delta_quad=None,
              X_area_surv=None, X_area_quad=None,
              X_area_avg_surv=None, X_area_avg_quad=None,
              Z_delta_surv=None, Z_delta_quad=None,
              Z_area_surv=None, Z_area_quad=None,
              Z_area_avg_surv=None, Z_area_avg_quad=None,
              W_surv=None):
        if beta_prior_mean is not None:
            beta = numpyro.sample("beta", dist.Normal(jnp.atleast_1d(jnp.array(beta_prior_mean)),
                                                        jnp.atleast_1d(jnp.array(beta_prior_sd))))
        else:
            beta = numpyro.sample("beta", dist.Normal(0.0, 5.0).expand([p]))
        # sigma_e: empirical-Bayes Gamma centred on the lme() pre-fit's
        # residual SD, matching JMbayes2's construction exactly. Verified
        # against a fitted jm object's $priors:
        #     gamma_prior_sigmas = TRUE, sigmas_shape = 5,
        #     sigmas_mean = 0.349  (= the lme residual SD)
        # A Gamma(shape=a, rate=a/m) has mean m and CV 1/sqrt(a), so
        # shape 5 gives a ~45% CV centred on the pre-fit estimate - the
        # same parameterisation already used for sigma_b below.
        #
        # Previously HalfNormal(2.0), which ignored the data scale
        # entirely. Falls back to that when no pre-fit value is available
        # (empirical_bayes_prior = FALSE, or the internal lme() failed).
        if sigma_e_prior_mean is not None:
            _se_mean = jnp.maximum(jnp.array(float(sigma_e_prior_mean)), 1e-8)
            sigma_e = numpyro.sample(
                "sigma_e",
                dist.Gamma(sigma_e_prior_shape, sigma_e_prior_shape / _se_mean))
        else:
            sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(2.0))

        # Split into alpha_value/alpha_<channel> when an extra channel is
        # present, matching the MLE backends' naming convention exactly -
        # a pure config-time check on whether real data was passed for
        # one of the X_*_surv arrays, not a per-element JAX tracing
        # conditional (each is either a real array or None for the WHOLE
        # call, decided once by the R side before tracing begins; the
        # R side guards against combining more than one channel at once,
        # so at most one of these three is ever non-None).
        if X_delta_surv is not None:
            channel_name = "delta"
        elif X_area_surv is not None:
            channel_name = "area"
        elif X_area_avg_surv is not None:
            channel_name = "area_avg"
        else:
            channel_name = None

        if channel_name is not None:
            alpha_value = numpyro.sample("alpha_value", dist.Normal(0.0, alpha_prior_sd))
            alpha_extra = numpyro.sample(f"alpha_{channel_name}", dist.Normal(0.0, alpha_prior_sd))
        else:
            alpha = numpyro.sample("alpha", dist.Normal(0.0, alpha_prior_sd))

        # Baseline (time-constant) covariates in the SURVIVAL submodel -
        # gamma'W_i, a per-subject CONSTANT contribution to the log-hazard,
        # added identically to the event-time hazard and every quadrature
        # node (same formula as the already-validated MLE backends'
        # baseline-covariates functions - see weibull_model.py's
        # _fit_mle_with_baseline_covariates). n_gamma is read directly off
        # W_surv's own (static, trace-time-known) shape rather than a
        # separate factory parameter, since nothing before this point
        # needs to know it.
        #
        # Two separate shapes are needed: gamma_term (flat [N_sub], added
        # to the event-time hazard) and gamma_term_quad ([N_sub, 1], added
        # to the [N_sub, n_quad]-shaped quadrature-node hazard - a genuine
        # bug caught after an initial version used gamma_term directly in
        # both places: [N_sub, n_quad] + [N_sub] fails to broadcast,
        # since alignment happens from the RIGHT, trying to match n_quad
        # against N_sub). When W_surv is None, gamma_term stays the plain
        # Python scalar 0.0, which broadcasts safely against ANY shape
        # with no reshaping needed at all - gamma_term_quad is set to the
        # SAME scalar 0.0 in that case (0.0[:, None] would itself error,
        # since a plain float isn't subscriptable).
        gamma_term = 0.0
        gamma_term_quad = 0.0
        if W_surv is not None:
            n_gamma = W_surv.shape[-1]
            # gamma_prior_mean centres the survival-covariate prior on the
            # coxph() pre-fit's coefficients, matching JMbayes2's
            # mean_gammas (verified: coef(coxph) = -0.06697697 against
            # JMbayes2's mean_gammas = -0.067 on the same fit). The SD of 2
            # already matches JMbayes2's Tau_gammas = 0.25. Supplied only by
            # jm_fit_prefit(), which has a coxph object to read; the formula
            # interface has none, so it falls back to zero-centred.
            if gamma_prior_mean is not None:
                # result_type(float), not float32: a float32 prior mean in
                # an otherwise float64 model silently truncates the
                # coxph coefficient it exists to carry.
                _gm = jnp.atleast_1d(jnp.array(gamma_prior_mean,
                                               dtype=jnp.result_type(float)))
                if _gm.shape[0] == n_gamma:
                    gamma = numpyro.sample("gamma", dist.Normal(_gm, gamma_prior_sd))
                else:
                    warnings.warn(
                        f"gamma_prior_mean has length {_gm.shape[0]} but the model "
                        f"has {n_gamma} baseline covariate(s); ignoring it and "
                        "using a zero-centred prior.", RuntimeWarning)
                    gamma = numpyro.sample("gamma", dist.Normal(0.0, gamma_prior_sd).expand([n_gamma]))
            else:
                gamma = numpyro.sample("gamma", dist.Normal(0.0, gamma_prior_sd).expand([n_gamma]))
            gamma_term = jnp.einsum("ng,g->n", W_surv, gamma)
            gamma_term_quad = gamma_term[:, None]

        if baseline_hazard == "weibull":
            # Matches weibull_model.py's MLE parameterization exactly:
            # log h0(t) = log(shape) + (shape-1)*log(t) + log_lambda0.
            log_lambda0 = numpyro.sample("log_lambda0", dist.Normal(-2.0, 2.0))
            log_shape = numpyro.sample("log_shape", dist.Normal(0.0, 1.0))
            shape = numpyro.deterministic("shape", jnp.exp(log_shape))
        elif spline_prior == "penalized":
            tau_w = numpyro.sample("tau_w", dist.Gamma(spline_penalty_shape, spline_penalty_rate))
            sigma_w = 1.0 / jnp.sqrt(tau_w)
            # First two coefficients are unpenalized (an order-2 difference
            # penalty has a 2-dimensional null space - constant and linear
            # trends in the coefficient sequence - matching how P-splines
            # are always specified: only *changes in slope* of the
            # coefficient sequence are penalized).
            W01 = numpyro.sample("W01", dist.Normal(0.0, 10.0).expand([2]))

            # NON-CENTERED reparameterization: sample a unit-scale z_step and
            # scale by sigma_w afterward, rather than sampling w_next
            # directly with mean/scale both depending on sigma_w. The
            # centered version creates classic "funnel" geometry (NUTS needs
            # a very different step size depending on whether tau_w is large
            # or small), which showed up empirically as unusually variable
            # ESS/second across benchmark scenarios (13.9 to 61.7 ESS/sec
            # for the same model on different datasets) - a hallmark of
            # funnel-driven sampling inefficiency. This is the standard fix
            # (Betancourt & Girolami; standard NumPyro/Stan practice for
            # hierarchical scale parameters).
            if rw2_implementation == "vectorized":
                # VECTORIZED closed-form RW2 construction, replacing the
                # sequential scan below. Derivation: the recursion
                #   W_k = 2*W_{k-1} - W_{k-2} + sigma_w*z_k   (k=3..K)
                # has closed-form solution (verified by direct
                # substitution/induction):
                #   W_k = W_2 + (k-2)*(W_2-W_1) + sigma_w * S2_k
                # where S2_k is the DOUBLE cumulative sum of the
                # innovations z_3,...,z_K (i.e. cumsum(cumsum(z))).
                # Mathematically IDENTICAL to the scan version - same
                # linear recursion, solved in closed form instead of
                # stepped through sequentially - but replaces an O(K)
                # sequential dependency chain (with per-step
                # numpyro.sample bookkeeping overhead) with two vectorized
                # jnp.cumsum calls, letting XLA use its optimized
                # (parallelizable, prefix-sum) execution instead of a
                # strictly sequential loop.
                n_rest = n_splines - 2
                z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0).expand([n_rest]))
                s2 = jnp.cumsum(jnp.cumsum(z_step))
                k_offset = jnp.arange(1, n_rest + 1)  # (k-2) for k=3..K -> 1,2,...,n_rest
                w_rest = W01[1] + k_offset * (W01[1] - W01[0]) + sigma_w * s2
            else:
                def rw2_step(carry, _):
                    w_prev2, w_prev1 = carry
                    z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0))
                    w_next = 2.0 * w_prev1 - w_prev2 + sigma_w * z_step
                    return (w_prev1, w_next), w_next

                _, w_rest = numpyro_scan(rw2_step, (W01[0], W01[1]), None, length=n_splines - 2)
            # Registered as a deterministic site so it appears in
            # get_samples() under "W" uniformly regardless of which prior
            # branch was used - _package_result()'s extraction logic and
            # everything downstream doesn't need to know which branch ran.
            W = numpyro.deterministic("W", jnp.concatenate([W01, w_rest]))
        else:
            W = numpyro.sample("W", dist.Normal(0.0, 3.0).expand([n_splines]))

        if q == 1:
            if sigma_b_prior_mean is not None:
                # jnp.atleast_1d guards against sigma_b_prior_mean arriving as
                # a 0-dimensional array - a length-1 R vector (the q=1 case,
                # e.g. sqrt(diag(getVarCov(lme_fit))) for a 1x1 matrix) can
                # convert via reticulate to a plain Python scalar rather than
                # a length-1 array, and jnp.array(scalar)[0] then fails with
                # "Too many indices: array is 0-dimensional".
                rate = sigma_b_prior_shape / jnp.atleast_1d(jnp.array(sigma_b_prior_mean))[0]
                sigma_b = numpyro.sample("sigma_b", dist.Gamma(sigma_b_prior_shape, rate))
            else:
                sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(2.0))
            with numpyro.plate("subjects", N_sub):
                b = numpyro.sample("b", dist.Normal(0.0, sigma_b))
            b = b[:, None]
        else:
            if random_effects_method == "wishart_gibbs":
                # NEW, EXPERIMENTAL: D (the full q x q random-effects
                # covariance matrix, jointly encoding sigma_b0, sigma_b1,
                # AND rho) is sampled via a Wishart-conjugate GIBBS update
                # (see the matching gibbs_fn in fit_nuts()) instead of
                # NUTS exploring an LKJCholesky-parameterized L_corr via
                # gradients. This mirrors JMbayes2's own documented
                # approach (Rizopoulos 2016, JSS: "for the random effects
                # precision matrix D^-1 ... the posterior conditional is
                # a Wishart distribution") - D_inv is a closed-form draw
                # from a known distribution given the current b's, with
                # ZERO gradient evaluations needed for this parameter
                # block, unlike the LKJCholesky+HMC approach.
                #
                # D_inv is still declared as a numpyro.sample site (with
                # the SAME Wishart prior used in the Gibbs update below)
                # so NUTS's conditional log-density for the OTHER sites
                # correctly includes its contribution - HMCGibbs
                # substitutes the Gibbs-drawn value here rather than
                # letting NUTS sample it via gradients.
                #
                # b is sampled DIRECTLY (centered parameterization) rather
                # than via a non-centered b_std @ L.T transform - safe
                # here specifically because D is held FIXED (substituted
                # by the Gibbs step) during each NUTS sub-step, so the
                # classic funnel-geometry problem (which requires JOINT
                # exploration of a variance parameter and the effects
                # that depend on it) doesn't arise the way it would if D
                # were being explored via HMC simultaneously.
                nu0 = float(q + 1)  # minimal weakly-informative Wishart df

                # S0 (the Wishart prior's scale matrix) controls WHERE the
                # prior on D sits. The original default is S0 = I, which
                # ignores the data scale entirely - a known problem: the
                # inverse-Wishart family is documented to bias variances
                # UPWARD and correlations toward zero when the true
                # variance is small relative to the prior mean, with the
                # bias persisting even at large n. For a random slope with
                # true sd ~0.2 (variance 0.04) against an S0=I prior mean
                # near 1, that is a ~25x mismatch.
                #
                # wishart_eb_scale = TRUE instead centers the prior on the
                # lme() pre-fit's own variance estimates, the same
                # empirical-Bayes information jmjax's DEFAULT path already
                # uses for its Gamma priors on sigma_b (and that JMbayes2
                # uses for its D_sds_mean - confirmed by direct inspection
                # of a fitted jm object's $priors).
                #
                # Math: for D^-1 ~ Wishart(nu0, S0), E[D^-1] = nu0 * S0, so
                # E[D] ~= S0^-1 / (nu0 - q - 1) for nu0 > q + 1. With the
                # minimal nu0 = q + 1 that expectation is undefined (the
                # boundary case), so for the EB-scaled version we use
                # nu0 = q + 2, the smallest df giving a finite prior mean,
                # and set S0 = (D_hat * (nu0 - q - 1))^-1 = D_hat^-1 so
                # that E[D] = D_hat exactly.
                # NOT control.get(...): `control` is a parameter of fit_nuts,
                # not of build_model, so referring to it here raised
                #   NameError: name 'control' is not defined
                # the moment this branch executed - i.e. every
                # random_effects_method = "wishart_gibbs" fit. The flag is now
                # threaded in as an argument. Found on 2026-09-21 by running
                # the PBC2 comparison the validation vignette's 4.8x figure
                # came from: that figure's exact configuration could not run.
                if bool(wishart_eb_scale) and sigma_b_prior_mean is not None:
                    nu0 = float(q + 2)  # smallest df with a finite prior mean for D
                    sd_hat = jnp.atleast_1d(jnp.array(sigma_b_prior_mean))
                    # Diagonal D_hat from the pre-fit SDs (correlation is
                    # deliberately NOT imposed here - the prior should be
                    # centered on the right SCALE without also asserting a
                    # correlation direction).
                    D_hat = jnp.diag(sd_hat ** 2)
                    S0 = jnp.linalg.inv(D_hat)
                else:
                    S0 = jnp.eye(q)     # original fixed prior scale (default)

                D_inv = numpyro.sample("D_inv", dist.Wishart(concentration=nu0, scale_matrix=S0))
                D = jnp.linalg.inv(D_inv)
                D_sym = 0.5 * (D + D.T)  # numerical symmetry safety
                with numpyro.plate("subjects", N_sub):
                    b = numpyro.sample("b", dist.MultivariateNormal(jnp.zeros(q), covariance_matrix=D_sym))
            elif random_effects_method == "wishart_gibbs_centered":
                # EXPERIMENTAL, ADD-ON VARIANT - fully separate from
                # "wishart_gibbs" above (which is left entirely untouched).
                # Addresses a diagnosed, near-perfect posterior anti-
                # correlation (-0.95, confirmed empirically) between beta_0
                # and mean(b_i0) across subjects - a classic location-
                # degeneracy: the data only constrain beta_0 + mean(b_i0),
                # not the two separately, so NUTS spends real exploration
                # effort tracing out a narrow ridge between them (the
                # mechanistic reason beta_0 consistently showed the lowest
                # ESS of any parameter across every random-effects
                # configuration tested).
                #
                # Fix: the RAW sampled random effects (b_raw) are exactly
                # what the Wishart-Gibbs update (see the matching gibbs_fn
                # in fit_nuts(), which reads hmc_sites["b_raw"] for this
                # branch specifically) needs - left unconstrained, exactly
                # as in the "wishart_gibbs" branch. But the value actually
                # used in the LIKELIHOOD is a deterministic transform that
                # subtracts the current sample's own mean from the
                # intercept column only, forcing mean(b_i0) = 0 EXACTLY at
                # every posterior draw - not just encouraged toward zero by
                # the prior. This removes the beta_0/mean(b_i0) degeneracy
                # by construction: with mean(b_i0) pinned at zero, all
                # population-level intercept information must flow through
                # beta_0 alone, with nothing left to trade off against.
                # The slope column (if q=2) is left unchanged, since the
                # diagnosed coupling was specific to the intercept.
                #
                # This is a deterministic, differentiable transform of an
                # already-valid latent variable - HMC gradients flow
                # through it without issue, and the (singular) implied
                # marginal prior on the centered b is a valid distribution
                # confined to the sum-to-zero hyperplane for its intercept
                # component.
                nu0 = float(q + 1)
                S0 = jnp.eye(q)
                D_inv = numpyro.sample("D_inv", dist.Wishart(concentration=nu0, scale_matrix=S0))
                D = jnp.linalg.inv(D_inv)
                D_sym = 0.5 * (D + D.T)
                with numpyro.plate("subjects", N_sub):
                    b_raw = numpyro.sample("b_raw", dist.MultivariateNormal(jnp.zeros(q), covariance_matrix=D_sym))
                b0_centered = b_raw[:, 0] - jnp.mean(b_raw[:, 0])
                b = numpyro.deterministic("b", jnp.concatenate([b0_centered[:, None], b_raw[:, 1:]], axis=1))
            else:
                if sigma_b_prior_mean is not None:
                    rate_vec = sigma_b_prior_shape / jnp.atleast_1d(jnp.array(sigma_b_prior_mean))
                    sigma_b = numpyro.sample("sigma_b", dist.Gamma(sigma_b_prior_shape, rate_vec))
                else:
                    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(2.0).expand([q]))
                if random_effects_corr:
                    # concentration=3.0 matches JMbayes2's D_L_etaLKJ = 3,
                    # verified against a fitted jm object's $priors. For
                    # q=2 the density is proportional to (1 - rho^2)^(eta-1),
                    # so eta=3 gives Var(rho) = 1/(2*eta - 1) = 0.2 against
                    # 0.333 at the previous eta=2 - modestly more shrinkage
                    # toward zero correlation.
                    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(q, concentration=lkj_concentration))
                    L = sigma_b[:, None] * L_corr
                    with numpyro.plate("subjects", N_sub):
                        b_std = numpyro.sample("b_std", dist.Normal(0.0, 1.0).expand([q]).to_event(1))
                    b_raw = b_std @ L.T

                    if b_orth_bases is None or not any(
                            q_ is not None for q_ in b_orth_bases):
                        b = numpyro.deterministic("b", b_raw)
                    else:
                        # ----------------------------------------------------
                        # EXPERIMENTAL, opt-in (control$orthogonalize_b0).
                        #
                        # MEASURED MOTIVATION (PBC2, n=312, 4x1000, sampled
                        # space - the ONLY space these numbers mean anything
                        # in; see dev/diagnose_beta0_v3.R for why):
                        #
                        #   corr(beta_0, mean level of b_i0)      -0.9709
                        #   corr(beta_2, age-slope of b_i0)       -0.9715
                        #   corr(beta_1, age-slope of b_i0)       +0.0543
                        #
                        #             ESS    ESS(+c)   recoverable
                        #   beta_0   440.5   2893.1        6.57x
                        #   beta_2   526.0   4000.0       >7.60x   (censored)
                        #   beta_1  1233.6       -            -
                        #
                        # Every coefficient on a SUBJECT-CONSTANT covariate
                        # is in a near-perfect location degeneracy with the
                        # matching projection of the random intercepts: the
                        # likelihood constrains only the SUM, and the split
                        # is pinned by the prior alone, at scale
                        # sigma_b0/sqrt(N). beta_1 is a WITHIN-subject
                        # contrast and is clean, which is the control that
                        # makes this a mechanism rather than a correlation.
                        #
                        # WHY A MEAN CONSTRAINT IS NOT ENOUGH. The existing
                        # random_effects_method="wishart_gibbs_centered"
                        # pins mean(b_i0) = 0 exactly, removing ONE direction
                        # of a k-dimensional degeneracy. Measured: beta_0
                        # 440.5 -> 1490.4, but beta_2 526.0 -> 291.2, so the
                        # WORST parameter got worse (442 -> 291) and the
                        # conservative min-ESS metric moved backwards. That
                        # is the whole case for generalizing from the mean to
                        # the full column space.
                        #
                        # WHAT THIS DOES. b0_orth_basis is an orthonormal
                        # basis Q [N_sub, k] for the subject-constant columns
                        # of the longitudinal design, built once from the
                        # concrete X_long in fit_nuts(). The intercept column
                        # of b is replaced by its residual off that space:
                        #
                        #     b_0  <-  b_0 - Q (Q' b_0)
                        #
                        # which is the k-dimensional generalization of
                        # subtracting the mean (k=1, Q = 1/sqrt(N)).
                        #
                        # WHY THE MODEL IS UNCHANGED. The removed directions
                        # lie in span(S), and S is by construction a subset
                        # of X's columns - so every direction swept out of
                        # b_0 is one beta already spans. For any b_raw there
                        # is a shift of beta giving IDENTICAL fitted values
                        # X beta + Z b, so the achievable mean structures are
                        # exactly the same set. This is a reparameterization
                        # that removes an unidentified direction, not a
                        # different model. The transform is deterministic and
                        # differentiable, so HMC gradients flow through it.
                        #
                        # WHAT DOES CHANGE, AND MUST BE CHECKED. The implied
                        # prior on the constrained b_0 is singular (confined
                        # to the orthogonal complement), and sigma_b still
                        # governs the UNCONSTRAINED b_raw. The k swept
                        # directions of b_raw are then unidentified by the
                        # likelihood and simply sample their prior. Whether
                        # that shifts sigma_b0's posterior is an empirical
                        # question, not something to assert - which is why
                        # this is opt-in and why the accompanying test
                        # compares estimates arm to arm, not just ESS.
                        # ----------------------------------------------------
                        _cols = []
                        for _q in range(q):
                            _bq = b_raw[:, _q]
                            _Qq = (b_orth_bases[_q]
                                   if _q < len(b_orth_bases) else None)
                            if _Qq is not None:
                                _Qq = jnp.asarray(_Qq)
                                _bq = _bq - _Qq @ (_Qq.T @ _bq)
                            _cols.append(_bq[:, None])
                        b = numpyro.deterministic(
                            "b", jnp.concatenate(_cols, axis=1))
                else:
                    # Simpler alternative: intercept and slope sampled
                    # INDEPENDENTLY (no LKJCholesky prior, no b_std @ L.T
                    # matrix multiply, no rho parameter at all - matching a
                    # common simpler mixed-model specification that assumes
                    # zero correlation between random intercept and slope).
                    # Found via a direct empirical comparison against an
                    # external NumPyro prototype using this simpler structure
                    # - genuinely worth offering as an option, not just a
                    # performance shortcut, since the correlated model's
                    # extra parameter (rho) and coupling between b0/b1's
                    # gradients can add real geometric difficulty for NUTS
                    # beyond just the matrix multiply's own compute cost, and
                    # a user with a genuine substantive reason to assume
                    # independence pays no cost for modeling a correlation
                    # they don't believe exists.
                    with numpyro.plate("subjects", N_sub):
                        b = numpyro.sample("b", dist.Normal(0.0, sigma_b).to_event(1))

        mu_long = jnp.einsum("nop,p->no", X_long, beta) + jnp.einsum("noq,nq->no", Z_long, b)
        mask = jnp.arange(max_obs)[None, :] < n_obs[:, None]
        # BUG FIX (see NEWS/commit history): previously used
        #   numpyro.sample("y_obs", dist.Normal(mu_long, sigma_e), obs=jnp.where(mask, y_long, y_long))
        # which was a no-op mask, scoring padded zero-slots as real
        # zero-residual observations and systematically deflating sigma_e.
        log_p_y = dist.Normal(mu_long, sigma_e).log_prob(y_long)
        numpyro.factor("y_obs", jnp.sum(jnp.where(mask, log_p_y, 0.0)))

        if baseline_hazard == "weibull":
            log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_surv) + log_lambda0
            log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad) + log_lambda0
        else:
            log_h0_T = jnp.dot(B_T, W)
            log_h0_quad = jnp.dot(B_quad, W)

        m_T = jnp.einsum("np,p->n", X_time_surv, beta) + jnp.einsum("nq,nq->n", Z_time_surv, b)
        m_quad = (jnp.einsum("nkp,p->nk", X_time_quad, beta)
                  + jnp.einsum("nkq,nq->nk", Z_time_quad, b))

        if channel_name is not None:
            if channel_name == "delta":
                X_extra_surv, X_extra_quad = X_delta_surv, X_delta_quad
                Z_extra_surv, Z_extra_quad = Z_delta_surv, Z_delta_quad
            elif channel_name == "area":
                X_extra_surv, X_extra_quad = X_area_surv, X_area_quad
                Z_extra_surv, Z_extra_quad = Z_area_surv, Z_area_quad
            else:  # area_avg
                X_extra_surv, X_extra_quad = X_area_avg_surv, X_area_avg_quad
                Z_extra_surv, Z_extra_quad = Z_area_avg_surv, Z_area_avg_quad

            if Z_extra_surv is not None:
                # GENERAL rule (q=2, or any q where the R side has
                # computed Z_extra explicitly): m_extra(t) = X_extra(t) @
                # beta + Z_extra(t) @ b - the SAME rule already validated
                # on the MLE side's q=2 extension
                # (weibull_model.py's _fit_mle_q2_extra_channel).
                m_extra_T = (jnp.einsum("np,p->n", X_extra_surv, beta)
                             + jnp.einsum("nq,nq->n", Z_extra_surv, b))
                m_extra_quad = (jnp.einsum("nkp,p->nk", X_extra_quad, beta)
                                 + jnp.einsum("nkq,nq->nk", Z_extra_quad, b))
            else:
                # q=1-specific shortcuts (Z_extra not computed/passed by R
                # for q=1 - see jm_fit.R's channel-construction code,
                # which only builds Z_delta/Z_area/Z_area_avg when q>1) -
                # each already validated against its MLE counterpart, and
                # each a special case of the general rule above: delta's Z
                # has a zero first column -> no b term at all; area's is
                # t -> b_i*t; area_avg's is 1 -> plain b_i.
                b_scalar = b[:, 0]
                if channel_name == "delta":
                    m_extra_T = jnp.einsum("np,p->n", X_extra_surv, beta)
                    m_extra_quad = jnp.einsum("nkp,p->nk", X_extra_quad, beta)
                elif channel_name == "area":
                    m_extra_T = jnp.einsum("np,p->n", X_extra_surv, beta) + b_scalar * T_surv
                    m_extra_quad = (jnp.einsum("nkp,p->nk", X_extra_quad, beta)
                                     + b_scalar[:, None] * t_quad)
                else:  # area_avg
                    m_extra_T = jnp.einsum("np,p->n", X_extra_surv, beta) + b_scalar
                    m_extra_quad = jnp.einsum("nkp,p->nk", X_extra_quad, beta) + b_scalar[:, None]

            log_hazard = log_h0_T + gamma_term + alpha_value * m_T + alpha_extra * m_extra_T
            hazard_quad = jnp.exp(log_h0_quad + gamma_term_quad + alpha_value * m_quad + alpha_extra * m_extra_quad)
        else:
            log_hazard = log_h0_T + gamma_term + alpha * m_T
            hazard_quad = jnp.exp(log_h0_quad + gamma_term_quad + alpha * m_quad)

        cum_hazard = T_surv * jnp.sum(hazard_quad * gk_weights[None, :], axis=1)

        surv_log_prob = event * log_hazard - cum_hazard
        numpyro.factor("surv_log_prob", surv_log_prob)

    return model


def find_map_b_std(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                    T_surv, event, t_quad, gk_weights,
                    B_T, B_quad, Z_long, Z_time_surv, Z_time_quad,
                    beta, sigma_e, alpha, W, sigma_b, L_corr,
                    n_newton_steps=20):
    """
    For FIXED population parameters theta, find the per-subject b_std_i (in
    the model's b = b_std @ L.T reparameterization) that maximizes the
    joint conditional log-density - i.e. the MAP random effect given theta.

    Why this exists: comparing log-density at (theta_true, b_true_simulated)
    vs (theta_hat, b_hat_posterior_mean) is NOT a fair comparison, even when
    theta_true is exactly correct - b_true_simulated is one random draw
    from its prior, while b_hat_posterior_mean is effectively tuned to fit
    the data well. A fair, theta-vs-theta comparison requires optimizing b
    EQUALLY for both candidates: profile_log_density(theta) =
    joint_log_density(theta, b = find_map_b_std(theta)). This function
    computes that optimal b_std via per-subject 2D Newton steps (a
    generalization of the scalar Newton mode-finding already used in the
    adaptive-GH MLE backend, common.py's find_mode_and_tau, to a correlated
    2D random intercept+slope).

    Uses symmetrized-and-eigenvalue-clipped Hessians for numerical safety
    (guards against a non-negative-definite Hessian derailing the Newton
    step, generalizing common.py's scalar clipping to the 2x2 case).
    """
    beta = jnp.array(beta)
    sigma_e = jnp.array(sigma_e)
    alpha = jnp.array(alpha)
    W = jnp.array(W)
    sigma_b = jnp.array(sigma_b)
    L_corr = jnp.array(L_corr)
    L = sigma_b[:, None] * L_corr
    q = L.shape[0]

    X_long = jnp.array(X_long)
    y_long = jnp.array(y_long)
    n_obs = jnp.array(n_obs)
    X_time_surv = jnp.array(X_time_surv)
    X_time_quad = jnp.array(X_time_quad)
    T_surv = jnp.array(T_surv)
    event = jnp.array(event)
    t_quad = jnp.array(t_quad)
    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)
    Z_long = jnp.array(Z_long)
    Z_time_surv = jnp.array(Z_time_surv)
    Z_time_quad = jnp.array(Z_time_quad)
    max_obs = X_long.shape[1]

    def h_subject(b_std_i, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                  T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                  Z_long_i, Z_time_surv_i, Z_time_quad_i):
        b_i = L @ b_std_i

        mu = X_long_i @ beta + Z_long_i @ b_i
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(max_obs) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.dot(B_T_i, W)
        m_T = X_time_surv_i @ beta + Z_time_surv_i @ b_i
        log_h_T = log_h0_T + alpha * m_T

        log_h0_quad = jnp.dot(B_quad_i, W)
        m_quad = X_time_quad_i @ beta + Z_time_quad_i @ b_i
        hazard_quad = jnp.exp(log_h0_quad + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        # Standard normal prior on b_std (matches the model exactly: the
        # theta-dependent covariance is entirely captured by L above).
        log_prior_bstd = -0.5 * jnp.sum(b_std_i ** 2) - 0.5 * q * jnp.log(2.0 * jnp.pi)

        return log_p_y + log_p_surv + log_prior_bstd

    grad_fn = jax.grad(h_subject, argnums=0)
    hess_fn = jax.hessian(h_subject, argnums=0)

    def newton_step(b_std_i, subj_args, eps=1e-6):
        g = grad_fn(b_std_i, *subj_args)
        H = hess_fn(b_std_i, *subj_args)
        H_sym = 0.5 * (H + H.T)
        eigvals, eigvecs = jnp.linalg.eigh(H_sym)
        eigvals_safe = jnp.minimum(eigvals, -eps)  # ensure negative definite
        H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
        delta = jnp.linalg.solve(H_safe, g)
        return b_std_i - delta

    def newton_solve(subj_args):
        def step(b, _):
            return newton_step(b, subj_args), None
        b_final, _ = jax.lax.scan(step, jnp.zeros(q), None, length=n_newton_steps)
        return b_final

    b_std_map = jax.vmap(lambda *args: newton_solve(args))(
        X_long, y_long, n_obs, X_time_surv, X_time_quad,
        T_surv, event, t_quad, B_T, B_quad, Z_long, Z_time_surv, Z_time_quad
    )

    return np.asarray(b_std_map)  # [N_sub, q]


def evaluate_log_density(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                          T_surv, event, t_quad, gk_weights,
                          B_T, B_quad, n_splines,
                          Z_long, Z_time_surv, Z_time_quad,
                          beta, sigma_e, alpha, W, sigma_b, b_std,
                          L_corr=None, random_effects="intercept_slope",
                          alpha_prior_sd=2.0):
    """
    Evaluate the EXACT joint log-density (log-likelihood + log-prior, same
    quantity NUTS's potential energy is built from) at a given, fully
    specified parameter point - via numpyro.infer.util.log_density, not a
    hand-reimplemented formula. Used to rigorously check whether a
    posterior that converged (good R-hat) to a value far from the true
    simulation parameters is because the data genuinely supports that value
    more than the truth (a real, if unlucky, statistical outcome - not a
    bug), or because something is wrong (the true parameters would score
    HIGHER density than what NUTS converged to, despite good diagnostics -
    which would point to a real remaining implementation issue).

    All arguments except b_std/L_corr/sigma_b/random_effects/alpha_prior_sd
    match fit_nuts()'s data arguments exactly. Parameter arguments (beta,
    sigma_e, alpha, W, sigma_b, b_std, L_corr) must be given as VALUES in
    the model's natural (constrained) parameter space - e.g. sigma_e > 0
    directly, not a log/unconstrained transform - matching what
    numpyro.infer.util.log_density expects.

    Returns the scalar joint log-density (float) - directly comparable
    between two calls with different parameter values on the SAME data.
    """
    N_sub, max_obs, p = np.asarray(X_long).shape
    q = 1 if random_effects == "intercept" else 2

    model = build_model(p, q, n_splines, max_obs, alpha_prior_sd=alpha_prior_sd)

    params = {
        "beta": jnp.array(beta),
        "sigma_e": jnp.array(sigma_e),
        "alpha": jnp.array(alpha),
        "W": jnp.array(W),
        "sigma_b": jnp.array(sigma_b),
    }
    if q == 1:
        params["b"] = jnp.array(b_std)  # for q=1, "b" is sampled directly
    else:
        if L_corr is None:
            raise ValueError("L_corr is required when random_effects != 'intercept'")
        params["L_corr"] = jnp.array(L_corr)
        params["b_std"] = jnp.array(b_std)

    model_kwargs = dict(
        X_long=jnp.array(X_long), y_long=jnp.array(y_long), n_obs=jnp.array(n_obs),
        X_time_surv=jnp.array(X_time_surv), X_time_quad=jnp.array(X_time_quad),
        T_surv=jnp.array(T_surv), event=jnp.array(event), t_quad=jnp.array(t_quad),
        gk_weights=jnp.array(gk_weights), B_T=jnp.array(B_T), B_quad=jnp.array(B_quad),
        Z_long=jnp.array(Z_long), Z_time_surv=jnp.array(Z_time_surv), Z_time_quad=jnp.array(Z_time_quad),
        N_sub=N_sub,
    )

    log_joint, _ = numpyro_log_density(model, (), model_kwargs, params)
    return float(log_joint)


def fit_nuts(X_long, y_long, n_obs, X_time_surv, X_time_quad,
             T_surv, event, t_quad, gk_weights,
             B_T, B_quad, n_splines,
             Z_long=None, Z_time_surv=None, Z_time_quad=None,
             X_delta_surv=None, X_delta_quad=None,
             X_area_surv=None, X_area_quad=None,
             X_area_avg_surv=None, X_area_avg_quad=None,
             Z_delta_surv=None, Z_delta_quad=None,
             Z_area_surv=None, Z_area_quad=None,
             Z_area_avg_surv=None, Z_area_avg_quad=None,
             W_surv=None,
             random_effects="intercept",
             init_theta=None, control=None):
    control = control or {}
    N_sub, max_obs, p = X_long.shape
    q = 1 if random_effects == "intercept" else 2

    if Z_long is None:
        Z_long = X_long[:, :, :q]
        Z_time_surv = X_time_surv[:, :q]
        Z_time_quad = X_time_quad[:, :, :q]

    alpha_prior_sd = float(control.get("alpha_prior_sd", 2.0))
    baseline_hazard = control.get("baseline_hazard", "spline")
    spline_prior = control.get("spline_prior", "independent")
    spline_penalty_shape = float(control.get("spline_penalty_shape", 5.0))
    spline_penalty_rate = float(control.get("spline_penalty_rate", 0.5))
    beta_prior_mean = control.get("beta_prior_mean", None)
    beta_prior_sd = control.get("beta_prior_sd", None)
    sigma_b_prior_mean = control.get("sigma_b_prior_mean", None)
    sigma_b_prior_shape = float(control.get("sigma_b_prior_shape", 5.0))
    sigma_e_prior_mean = control.get("sigma_e_prior_mean", None)
    sigma_e_prior_shape = float(control.get("sigma_e_prior_shape", 5.0))
    lkj_concentration = float(control.get("lkj_concentration", 3.0))
    gamma_prior_mean = control.get("gamma_prior_mean", None)
    gamma_prior_sd = float(control.get("gamma_prior_sd", 2.0))
    random_effects_corr = bool(control.get("random_effects_corr", True))
    random_effects_method = control.get("random_effects_method", "nuts")
    # DEFAULT CHANGED from "scan" to "vectorized".
    #
    # "scan" is UNUSABLE on the stack this package pins. numpyro's
    # scan_wrapper builds its carry with device_put((0, rng_key, init)) -
    # device_put applied to a TUPLE containing a Python int - which yields a
    # weakly-typed int32. From jax 0.4.30 onward that carry element's tangent
    # is float0 on one side of lax.scan's backward pass and int32 on the
    # other, so any gradient through the scan dies with
    #
    #   TypeError: body_fun output and input must have identical types
    #   ... ShapedArray(int32[], weak_type=True) vs ShapedArray(float0[])
    #
    # This is jax-ml/jax#22045, present in jax >= 0.4.30. jmjax pins jax
    # 0.4.30 and numpyro 0.15.0, so EVERY correct install of this package
    # hits it: spline_prior="penalized" could not run at all. numpyro fixed
    # its side in 4704656 ("remove unnecessary device_put"), first released
    # in numpyro 0.17.0, but that is a stack upgrade needing its own
    # revalidation rather than something to do quietly here.
    #
    # The vectorized branch is not a workaround with a cost attached. It
    # solves the SAME linear recursion in closed form - verified against the
    # sequential version over 200 random cases (K in [3,30]), worst relative
    # difference 4.9e-15 - and the replication that introduced it measured
    # both at an identical 63.0 mean leapfrog steps. It also avoids an O(K)
    # sequential dependency chain, so if anything it should be faster.
    #
    # "scan" remains selectable for anyone on numpyro >= 0.17.0 who wants it.
    rw2_implementation = control.get("rw2_implementation", "vectorized")

    # ------------------------------------------------------------------
    # control$orthogonalize_b0 (EXPERIMENTAL, opt-in, default False).
    # See the long note in build_model()'s matching branch for the measured
    # motivation. Restricted to the path it was measured on rather than
    # silently no-op'ing elsewhere: the other random-effects branches name
    # "b" as a SAMPLE site, so constraining it there would leave
    # posterior_samples$b holding the unconstrained draws while the
    # likelihood used the projected ones - a discrepancy that would quietly
    # corrupt exactly the diagnostics this option exists to improve.
    # ------------------------------------------------------------------
    # orthogonalize_b  : every random-effect column (intercept AND slope)
    # orthogonalize_b0 : the intercept column only - retained so the two can
    #                    be compared, since the slope extension is the newer
    #                    and less tested half.
    _orth_all  = bool(control.get("orthogonalize_b", False))
    _orth_int  = bool(control.get("orthogonalize_b0", False))
    b_orth_bases = None
    # SEPARATE, OPT-IN TRACK (see the "beta_corrected" block after
    # mcmc.get_samples() below, and _absorbable_generator_matrix() near
    # _absorbable_basis_exact()). Parallel to b_orth_bases: _correction_bases
    # holds, per random-effect column, the [p, k] matrix Dmat matching the
    # SAME Q that b_orth_bases[q] holds, i.e. the beta-shift that keeps
    # fitted values unchanged when Q's columns are subtracted from b[:, q].
    # This does not change what is sampled - it is read only to turn the
    # already-sampled b_std/L_corr/sigma_b/beta draws into a corrected
    # estimand, entirely after the fact.
    _correction_bases = None
    # Structured record of what orthogonalize_b/_b0 actually did, surfaced
    # to R as fit$convergence$orthogonalize (NULL unless the option was
    # requested). This exists for the same reason warm_start_check does:
    # the warnings.warn() calls below arrive on stderr, not as an R
    # condition (reticulate does not route them through R's condition
    # system), so expect_warning() cannot see them and a non-interactive
    # run would silently lose the report. A caller - or a test - that
    # needs to know which columns were actually constrained has to be able
    # to read it off the fitted object instead.
    orthogonalize_report = None
    if _orth_all or _orth_int:
        if not (q >= 2 and random_effects_corr
                and random_effects_method == "nuts"):
            raise ValueError(
                "orthogonalize_b/_b0 = True currently requires q >= 2, "
                "random_effects_corr = TRUE and the default "
                "random_effects_method = 'nuts' (got q = %d, corr = %s, "
                "method = '%s'). It is an experimental option; the other "
                "random-effects branches are not yet covered."
                % (q, bool(random_effects_corr), random_effects_method))
        _qmax = q if _orth_all else 1
        b_orth_bases = []
        _correction_bases = []
        _orth_report = []
        _orth_columns = []
        _s_resid = 0.0
        for _q in range(q):
            if _q < _qmax:
                # TWO constructions, deliberately. The per-column search is
                # kept for ATTRIBUTION - it names which X columns expose each
                # direction, which is what makes the report diagnostic rather
                # than just a dimension count. The exact construction is
                # AUTHORITATIVE: it depends only on X's column space, so it
                # cannot miss directions that a spline or orthogonal-
                # polynomial basis hides from the per-column test. See
                # _absorbable_basis_exact's docstring and Section 4.5 of
                # vignette("jmjax-reparameterization").
                _Qcol, _kept = _absorbable_basis(X_long, Z_long, n_obs, _q)
                _Q, _gens = _absorbable_basis_exact(X_long, Z_long, n_obs, _q)
                _ncol = 0 if _Qcol is None else _Qcol.shape[1]
                _nex = 0 if _Q is None else _Q.shape[1]
                if _nex > _ncol:
                    warnings.warn(
                        "orthogonalize_b: b[:,%d] - the per-column search "
                        "found %d absorbable direction(s)%s, but %d exist. "
                        "The extra direction(s) are absorbable only by a "
                        "COMBINATION of fixed-effects columns that no single "
                        "column exposes, which is what a spline or "
                        "orthogonal-polynomial longitudinal mean produces. "
                        "The exact (basis-independent) basis is being used, "
                        "so the degeneracy IS removed; this notice exists "
                        "because the per-column answer alone would have "
                        "silently left it in place. See "
                        "vignette('jmjax-reparameterization'), Section 4.5."
                        % (_q, _ncol,
                           "" if not _kept else " (from X columns %s)" % (_kept,),
                           _nex),
                        RuntimeWarning, stacklevel=2)
                elif _nex < _ncol:
                    # Proposition 2 says this cannot happen: every
                    # per-column direction is absorbable, so the exact basis
                    # must contain it. If it ever fires, one of the two
                    # constructions is wrong and the fit should not be
                    # quietly trusted.
                    warnings.warn(
                        "orthogonalize_b: b[:,%d] - the exact construction "
                        "found FEWER directions (%d) than the per-column "
                        "search (%d). This contradicts the soundness result "
                        "and should be reported as a bug; the exact basis is "
                        "being used." % (_q, _nex, _ncol),
                        RuntimeWarning, stacklevel=2)
                _s_resid = max(_s_resid, _structural_extension_residual(
                    _gens, _q, X_time_surv, Z_time_surv,
                    X_time_quad, Z_time_quad))
                # SEPARATE, OPT-IN TRACK: the beta-shift basis matching Q,
                # for the "beta_corrected" post-processing block below. Reuses
                # _Q/_gens already computed above at no extra cost (no second
                # call into _absorbable_basis_exact).
                _Dmat = _absorbable_generator_matrix(_Q, _gens) if _Q is not None else None
            else:
                _Q, _kept, _ncol, _nex = None, [], 0, 0
                _Dmat = None
            b_orth_bases.append(_Q)
            _correction_bases.append(_Dmat)
            if _Q is None:
                _desc = "no basis (unconstrained)"
            elif _nex == _ncol:
                # wording unchanged in the ordinary case, so existing logs
                # and the vignette's quoted excerpts stay comparable
                _desc = "%d direction(s) from X columns %s" % (_nex, _kept)
            else:
                _desc = ("%d direction(s) (%d from X columns %s, %d from "
                         "column combinations)" % (_nex, _ncol, _kept,
                                                    _nex - _ncol))
            _orth_report.append("b[:,%d] <- %s" % (_q, _desc))
            _orth_columns.append({
                "column": int(_q),
                "applied": _Q is not None,
                "n_directions": int(_nex),
                "source_X_columns": [int(j) for j in _kept],
                # exposes the per-column search's answer separately, so a
                # caller can detect the spline/poly case programmatically
                # rather than by reading the warning text
                "n_directions_column_search": int(_ncol),
            })
        # Say what was ACTUALLY constrained, rather than leaving it to be
        # inferred from downstream behaviour. A verification run found
        # mean(b_i1) NOT driven to zero while beta_1's ESS rose 13x, which is
        # a contradiction that could not be resolved from outside: either the
        # basis for the slope column was not what its design intended, or the
        # effect came from somewhere else. A sampler that reports its own
        # constraints makes that question answerable in one line instead of
        # by inference from a second experiment.
        warnings.warn("orthogonalize_b: " + "; ".join(_orth_report),
                      RuntimeWarning, stacklevel=2)
        if _s_resid > 1e-6:
            # Condition (S) of vignette("jmjax-reparameterization"),
            # Section 4.4: absorbability is established on the longitudinal
            # grid, but `b` also enters through m_i(t) at the event and
            # quadrature times. If the proportionality does not extend
            # there, the swept direction is NOT likelihood-invariant, and
            # the reparameterization would change the model rather than
            # only its coordinates.
            warnings.warn(
                "orthogonalize_b: the absorbable directions do not extend "
                "to the survival/quadrature time grid (worst relative "
                "residual %.2e). Condition (S) of "
                "vignette('jmjax-reparameterization') Section 4.4 fails, so "
                "the reparameterization is NOT guaranteed to leave the joint "
                "likelihood unchanged for this design. Treat results from "
                "this fit as unvalidated." % _s_resid,
                RuntimeWarning, stacklevel=2)
        _orth_any_applied = any(_Q is not None for _Q in b_orth_bases)
        orthogonalize_report = {
            "requested": "b" if _orth_all else "b0",
            "columns": _orth_columns,
            "any_applied": _orth_any_applied,
        }
        if not _orth_any_applied:
            warnings.warn(
                "orthogonalize_b/_b0 = True but no fixed-effect column can "
                "absorb a shift in any random-effect column, so there is no "
                "location degeneracy to remove and the option has no "
                "effect. Expected when the design has no intercept and no "
                "column matching a random-effect term.",
                RuntimeWarning)
            b_orth_bases = None
            _correction_bases = None

    model = build_model(p, q, n_splines, max_obs, alpha_prior_sd=alpha_prior_sd,
                         baseline_hazard=baseline_hazard,
                         spline_prior=spline_prior,
                         spline_penalty_shape=spline_penalty_shape,
                         spline_penalty_rate=spline_penalty_rate,
                         beta_prior_mean=beta_prior_mean, beta_prior_sd=beta_prior_sd,
                         sigma_b_prior_mean=sigma_b_prior_mean,
                         sigma_b_prior_shape=sigma_b_prior_shape,
                         sigma_e_prior_mean=sigma_e_prior_mean,
                         sigma_e_prior_shape=sigma_e_prior_shape,
                         lkj_concentration=lkj_concentration,
                         gamma_prior_mean=gamma_prior_mean,
                         gamma_prior_sd=gamma_prior_sd,
                         random_effects_corr=random_effects_corr,
                         random_effects_method=random_effects_method,
                         rw2_implementation=rw2_implementation,
                         wishart_eb_scale=bool(control.get("wishart_eb_scale", False)),
                         b_orth_bases=b_orth_bases)

    # Targeted dense_mass for the spline coefficients specifically -
    # distinct from the global control$dense_mass (which covers EVERY
    # continuous site including the ~1000-dimensional b_std and made
    # things dramatically worse, ~12.5x slower, when tried earlier).
    # This restricts the dense mass matrix to just the small (~7-9
    # dimensional) spline-innovation block, directly informed by the
    # non-diagonal RW2 penalty structure - cheap to invert at this size,
    # and targets specifically where a diagonal mass matrix can't
    # represent the known correlation between neighboring spline
    # coefficients that the RW2 prior implies.
    #
    # DEFAULT since replication testing: 3 seeds x 2 sample sizes showed
    # ~2.1x faster wall-clock and ~2.3x ESS/sec, consistent in direction
    # across every single run, with R-hat and truth recovery unaffected.
    # It was the only one of three performance candidates to pass those
    # pre-registered criteria. Set control$dense_mass_spline = FALSE to
    # restore the previous (diagonal mass matrix) behavior.
    #
    # Only meaningful when spline_prior="penalized" (there is no RW2
    # structure to exploit otherwise). NOTE: an earlier version of this
    # comment claimed it additionally required rw2_implementation=
    # "vectorized", on the assumption that scan produces many separate
    # scalar z_step sites that couldn't form one dense block. That was
    # WRONG - numpyro's scan COLLECTS per-iteration samples into a single
    # stacked site of the same shape the vectorized branch produces
    # directly. Directly tested: both implementations reach an identical
    # 63.0 mean leapfrog steps with this enabled (down from 255.0 and
    # 232.5 respectively), so the two options are independent.
    # The W01/z_step sites this targets only EXIST when the baseline is a
    # spline AND spline_prior="penalized" (see build_model: the weibull
    # branch samples log_lambda0/log_shape instead, and the independent
    # spline branch samples a plain "W" vector). Requesting a dense mass
    # block over sites that don't exist raises KeyError: 'W01' - which is
    # exactly what happened when this default was first flipped to TRUE
    # without this guard, breaking every weibull-PH-mcmc fit and every
    # spline fit using the default spline_prior="independent". The
    # replication study that justified the default change used
    # spline_prior="penalized" throughout, so it never exercised this path.
    dense_mass_arg = bool(control.get("dense_mass", False))
    _penalized_spline = (baseline_hazard != "weibull") and (spline_prior == "penalized")
    _blocks = []
    if _penalized_spline and control.get("dense_mass_spline", True):
        # ------------------------------------------------------------------
        # EXPERIMENTAL, opt-in: include `alpha` in the spline dense block.
        #
        # MOTIVATION: a cross-block correlation diagnostic found that alpha
        # is strongly correlated with the spline coefficients - alpha<->W01
        # measured -0.730 (standardized covariate) and -0.858 (raw-scale),
        # the largest cross-block correlation anywhere in the model. The
        # default block ("W01", "z_step") EXCLUDES alpha, so a diagonal
        # mass entry is used for it and that correlation is forced to zero
        # by construction.
        #
        # WHY alpha SPECIFICALLY: the log-hazard is
        #     log h_i(t) = B(t).W + alpha * m_i(t)
        # so B(t).W and alpha*m_i(t) compete additively. With m_i(t) ~ 2 on
        # average, raising alpha by d lifts the log-hazard by ~2d, which
        # lowering W offsets exactly - a direct ridge. Note beta_0 sits in
        # the same algebraic position but is NOT correlated with W in
        # practice (measured ~0.02): the longitudinal likelihood has
        # thousands of y observations pinning beta_0 down, leaving it no
        # freedom to trade against the baseline hazard, while alpha appears
        # only in the survival submodel and is free to move. An earlier
        # derivation predicted beta_0 <-> W and was wrong for exactly this
        # reason.
        #
        # WHY IT MIGHT MATTER MORE THAN THE COVARIATE-SCALE WORK: alpha is
        # the parameter ESS is measured on, and the correlation is strong
        # in BOTH covariate-scale conditions - so this is a general
        # inefficiency in every penalized-spline fit, not a raw-scale-only
        # problem.
        #
        # TEMPERED BY TRACK RECORD: dense_mass_beta was also well-motivated
        # (it targeted a confirmed -0.977 correlation) and produced a null
        # result, apparently because that geometry was not the bottleneck.
        # A large correlation is necessary but evidently not sufficient.
        # Opt-in pending the same replication discipline as everything else.
        # ------------------------------------------------------------------
        if control.get("dense_mass_alpha", False):
            _blocks.append(("W01", "z_step", "alpha"))
        else:
            _blocks.append(("W01", "z_step"))

    # ------------------------------------------------------------------
    # EXPERIMENTAL, opt-in: a dense mass block over `beta`.
    #
    # MOTIVATION: with an UNCENTERED covariate, the posterior correlation
    # between the intercept and that covariate's coefficient is severe. For
    # a design with an intercept and x ~ N(xbar, s):
    #
    #     corr(beta_0, beta_x)  ~=  -xbar / sqrt(xbar^2 + s^2)
    #
    # For x ~ N(50, 10) that is about -0.98; for standardized x (xbar = 0)
    # it is exactly 0. That single difference tracks the whole covariate-
    # scale effect measured earlier (jmjax 2-4x FASTER than JMbayes2 with
    # standardized covariates, 1.4-2.8x SLOWER with raw-scale ones).
    #
    # A DIAGONAL mass matrix cannot represent correlation at all, however
    # well-scaled its entries are, so NUTS is forced along a near-
    # degenerate ridge. This is the same pathology dense_mass_spline fixes
    # for the RW2 block, where neighboring spline coefficients are
    # correlated by construction - and that option cut leapfrog steps
    # 255 -> 63 and passed replication at ~2.1x.
    #
    # NOTE this supersedes an earlier attempt (control$seed_mass_matrix,
    # still present but default FALSE and NOT recommended) which seeded a
    # DIAGONAL mass matrix from the lme() pre-fit's standard errors. A
    # diagnostic showed NUTS's own adaptation already recovers essentially
    # those same scales unaided (adapted [0.00122, 0.00031, 0.00136] vs
    # lme se^2 [0.00147, 0.00026, 0.00144]), so seeding added nothing -
    # and it failed catastrophically on 1 of 4 test datasets. Scale was
    # never the problem; correlation is.
    #
    # Only meaningful when p >= 2 - a "dense" 1x1 block is just a diagonal
    # entry. Kept opt-in pending the same replication discipline applied
    # to every other performance option in this package.
    # ------------------------------------------------------------------
    if control.get("dense_mass_beta", False) and p >= 2:
        _blocks.append(("beta",))

    if _blocks:
        dense_mass_arg = _blocks

    # ------------------------------------------------------------------
    # EXPERIMENTAL, opt-in: seed NUTS's inverse mass matrix for `beta`
    # from the internal lme() pre-fit's standard errors.
    #
    # MOTIVATION: a replicated experiment (4 seeds, no overlap between
    # conditions) found jmjax is 1.4-2.8x SLOWER than JMbayes2 with
    # raw-scale covariates (ESS/sec ratio 0.36-0.70x) while being 2-4x
    # FASTER with standardized ones. JMbayes2 is nearly scale-invariant
    # because it seeds its MCMC proposals from the covariance matrix of
    # its MLE pre-fits; NUTS has no equivalent and must learn each
    # coordinate's scale during warmup, which evidently does not fully
    # succeed when scales differ by orders of magnitude.
    #
    # MECHANISM: for a coordinate with posterior sd s, a well-scaled
    # diagonal inverse mass matrix entry is s^2. The lme() pre-fit already
    # estimates exactly that for beta, as sqrt(diag(vcov(lme_prefit))) -
    # passed through as control$beta_se by the R side.
    #
    # ADAPTATION IS LEFT ON deliberately (adapt_mass_matrix defaults to
    # True). A probe confirmed numpyro honors a seed verbatim when
    # adaptation is disabled - but the lme() pre-fit knows nothing about
    # the survival submodel, the spline coefficients, or alpha, so
    # freezing its scales would likely help beta while hurting everything
    # else. Seeding-plus-adaptation is strictly more informed than the
    # current generic start, with no obvious downside.
    #
    # HONEST EXPECTATION: this is a PARTIAL fix at best. NUTS must still
    # adapt every other block, and beta is only part of the raw-scale
    # problem. Whether it meaningfully closes the 0.36-0.70x gap is an
    # empirical question this option exists to answer, not a claim.
    # ------------------------------------------------------------------
    nuts_kwargs = {"dense_mass": dense_mass_arg}

    # Sampler knobs, previously left at NumPyro's defaults (max_tree_depth
    # 10 -> 1023 steps, target_accept_prob 0.8). Exposed so the tree-depth
    # wall can be raised, and the step size allowed to grow, as an
    # ALTERNATIVE to reparameterizing the model.
    #
    # Worth testing against scale_time rather than assuming: raising the
    # depth to 12 permits 4095 steps per iteration, so if the sampler still
    # needs a deep tree it costs ~65x the 63 steps that scaling achieves.
    # Whether it converges at all on a badly conditioned target is the
    # actual question, and the answer decides whether the much simpler
    # sampler-level fix can replace the formula machinery.
    if control.get("max_tree_depth") is not None:
        # NumPyro accepts a tuple (d1, d2): d1 caps the tree depth during
        # WARM-UP, d2 afterwards. Worth having separately, because a poor
        # adaptation can spend the whole warm-up building 543-step
        # trajectories and never recover - capping the warm-up depth stops
        # it burning the budget there while leaving sampling unrestricted.
        _mtd = control["max_tree_depth"]
        if isinstance(_mtd, (list, tuple)) and len(_mtd) == 2:
            nuts_kwargs["max_tree_depth"] = (int(_mtd[0]), int(_mtd[1]))
        else:
            nuts_kwargs["max_tree_depth"] = int(_mtd)
    if control.get("target_accept_prob") is not None:
        nuts_kwargs["target_accept_prob"] = float(control["target_accept_prob"])

    # find_heuristic_step_size: NumPyro's pre-adaptation heuristic, which
    # picks a starting step size at the beginning of each adaptation window
    # rather than letting dual averaging search from an arbitrary one.
    #
    # Worth exposing because the step size that dual averaging settles on
    # is itself random - it depends on the draws the chain happens to take
    # during warm-up - and a poor one persists for the whole sampling
    # phase, since warm-up is over by then. Measured on one dataset at
    # n = 8,000, three sampler seeds gave 127, 63 and 543 leapfrog steps
    # per draw, with the last failing R-hat at 1.544 on rho and running
    # four times as long.
    # init_strategy: where the chains START.
    #
    # NumPyro's default is init_to_uniform, which draws each parameter
    # uniformly on [-2, 2] in unconstrained space. For a joint model that
    # means every one of several thousand random effects starts at an
    # independent random point, and the chain spends early warm-up draws
    # travelling from there to the typical set - draws that dual averaging
    # then uses to estimate the step size.
    #
    # That is a plausible root of the seed dependence observed here: three
    # chain seeds on identical data at n = 8,000 gave 63, 127 and 543
    # leapfrog steps per draw, with the last failing R-hat. A starting
    # point closer to the typical set should leave adaptation with better
    # information and less to vary over.
    #
    # init_to_median draws a few samples from the prior and takes their
    # median - more central, and much less variable between seeds than a
    # uniform draw. init_to_feasible only guarantees finite log-density.
    # init_to_median is NOT the fix it sounds like, and the geometry says
    # why. The dominant block of the unconstrained space is b_std, whose
    # prior is standard normal. In d dimensions a standard normal's typical
    # set is a thin shell at radius sqrt(d): at n = 8,000 with q = 2 that is
    # 16,000 dimensions and a shell at radius ~126, with essentially no mass
    # near the origin. init_to_median puts every coordinate at its prior
    # median - i.e. AT THE ORIGIN, radius ~0 - while init_to_uniform's
    # U(-2,2) has variance 4/3 and lands at radius ~146, only 15% outside
    # the shell. The default is accidentally well suited to this model and
    # "start at the median" is the intuition that fails hardest here.
    #
    # Measured, and consistent with that: heuristic + init_to_median
    # rescued two seeds at n = 2,000 (shell radius 45, where the origin is
    # survivable) and at n = 8,000 produced R-hat 32.7 with ESS 3, far
    # worse than the defaults' 1.544.
    _is = control.get("init_strategy")
    if _is is not None:
        from numpyro.infer import initialization as _init
        _map = {"uniform": _init.init_to_uniform,
                "median": _init.init_to_median,
                "feasible": _init.init_to_feasible,
                "sample": _init.init_to_sample}
        _fn = _map.get(str(_is))
        if _fn is None:
            warnings.warn(
                f"unknown init_strategy '{_is}'; using NumPyro's default. "
                f"Valid: {', '.join(sorted(_map))}.", RuntimeWarning)
        else:
            nuts_kwargs["init_strategy"] = _fn

    # The model's data arguments, built ONCE. mcmc.run() below consumes
    # this same dict, so the warm-start self-check cannot drift out of sync
    # with what is actually sampled - a duplicated copy would go stale the
    # first time an argument was added in one place and not the other.
    model_kwargs = dict(
        X_long=jnp.array(X_long), y_long=jnp.array(y_long), n_obs=jnp.array(n_obs),
        X_time_surv=jnp.array(X_time_surv), X_time_quad=jnp.array(X_time_quad),
        T_surv=jnp.array(T_surv), event=jnp.array(event), t_quad=jnp.array(t_quad),
        gk_weights=jnp.array(gk_weights), B_T=jnp.array(B_T), B_quad=jnp.array(B_quad),
        Z_long=jnp.array(Z_long), Z_time_surv=jnp.array(Z_time_surv),
        Z_time_quad=jnp.array(Z_time_quad), N_sub=N_sub,
        X_delta_surv=jnp.array(X_delta_surv) if X_delta_surv is not None else None,
        X_delta_quad=jnp.array(X_delta_quad) if X_delta_quad is not None else None,
        X_area_surv=jnp.array(X_area_surv) if X_area_surv is not None else None,
        X_area_quad=jnp.array(X_area_quad) if X_area_quad is not None else None,
        X_area_avg_surv=jnp.array(X_area_avg_surv) if X_area_avg_surv is not None else None,
        X_area_avg_quad=jnp.array(X_area_avg_quad) if X_area_avg_quad is not None else None,
        Z_delta_surv=jnp.array(Z_delta_surv) if Z_delta_surv is not None else None,
        Z_delta_quad=jnp.array(Z_delta_quad) if Z_delta_quad is not None else None,
        Z_area_surv=jnp.array(Z_area_surv) if Z_area_surv is not None else None,
        Z_area_quad=jnp.array(Z_area_quad) if Z_area_quad is not None else None,
        Z_area_avg_surv=jnp.array(Z_area_avg_surv) if Z_area_avg_surv is not None else None,
        Z_area_avg_quad=jnp.array(Z_area_avg_quad) if Z_area_avg_quad is not None else None,
        W_surv=jnp.array(W_surv) if W_surv is not None else None,
    )

    # ---- warm start from the R-side lme pre-fit -------------------------
    #
    # control["init_values"] carries constrained-space starting values for
    # whichever sample sites R could supply - typically beta, sigma_e,
    # sigma_b, L_corr, alpha and, the one that matters at scale, b_std
    # derived from the lme() BLUPs. Sites not supplied fall back to
    # init_to_uniform, which is fine: the ~20 population parameters were
    # never the problem, the 16,000 random effects were.
    #
    # This mirrors JMbayes2, which initialises betas, sigmas, D, the
    # per-subject b and gammas from the lme/coxph objects it is handed
    # (R/jm.R) and jitters per chain (R/jm_fit.R). Note the consequence,
    # which applies to both packages: chains that start together make R-hat
    # LESS sensitive, because Gelman-Rubin assumes overdispersed starts.
    # That is the price of a warm start, and it should be stated rather
    # than quietly enjoyed.
    #
    # THE JITTER IS APPLIED IN UNCONSTRAINED SPACE, unlike JMbayes2's,
    # which perturbs constrained values and must exclude D to avoid
    # breaking positive-definiteness. Going through biject_to means every
    # constraint survives automatically - sigmas stay positive, L_corr
    # stays a valid Cholesky factor - with no special cases.
    _init_vals = control.get("init_values")
    if _init_vals:
        from numpyro.infer import initialization as _init
        from numpyro.distributions.transforms import biject_to
        from functools import partial as _partial

        _jit_scale = float(control.get("warm_start_jitter", 0.1))
        _vals = {str(k): jnp.asarray(v) for k, v in dict(_init_vals).items()}

        def _init_to_value_jittered(site=None, values=None, scale=0.1):
            if site is None:
                return _partial(_init_to_value_jittered, values=values, scale=scale)
            if (site["type"] == "sample" and not site["is_observed"]
                    and not site["fn"].support.is_discrete):
                nm = site["name"]
                if values is not None and nm in values:
                    tf = biject_to(site["fn"].support)
                    # RESHAPE TO THE SITE'S OWN SHAPE before transforming.
                    # A length-1 value crosses the R/Python boundary as a
                    # SCALAR, not a 1-element array: reticulate unwraps
                    # length-1 vectors. So a model with exactly one baseline
                    # survival covariate got `gamma` as a 0-d array, and
                    # einsum("ng,g->n", W_surv, gamma) failed with
                    #   Einstein sum subscript 'g' does not contain the
                    #   correct number of indices for operand 1
                    # which surfaced only as "warm-start self-check failed"
                    # - so the warm start silently never ran on any model
                    # with a single baseline covariate, PBC2 included.
                    #
                    # Reshaping here fixes every site at once rather than
                    # patching each caller, and a genuinely wrong element
                    # count now raises immediately instead of becoming a
                    # confusing einsum error further downstream.
                    val = jnp.asarray(values[nm], dtype=jnp.result_type(float))
                    want = tuple(site["fn"].shape())
                    if jnp.shape(val) != want:
                        val = jnp.reshape(val, want)
                    u = tf.inv(val)
                    key = site["kwargs"].get("rng_key")
                    if key is not None and scale > 0:
                        u = u + scale * jax.random.normal(key, jnp.shape(u), dtype=u.dtype)
                    return tf(u)
            return _init.init_to_uniform(site)

        _warm = _init_to_value_jittered(values=_vals, scale=_jit_scale)

        # SELF-CHECK, and the reason this can be enabled without auditing
        # every scaling path by hand. Between R's lme() and the model as
        # sampled sit standardize_covariates and scale_time, either of
        # which can put the pre-fit's coefficients on a different scale
        # than the site expects; the non-centred b_std transform is another
        # chance to be wrong. Rather than reason through each, evaluate the
        # potential energy at the warm start and at a uniform start and
        # compare. A correct warm start is enormously better; a wrong one
        # is worse, and is discarded here rather than silently degrading
        # the fit.
        try:
            from numpyro.infer.util import initialize_model as _init_model

            def _pe(strategy, key_int):
                st = _init_model(jax.random.PRNGKey(key_int), model,
                                 model_args=(), model_kwargs=model_kwargs,
                                 init_strategy=strategy, dynamic_args=False)
                return float(st[1](st[0].z))

            _pe_warm = _pe(_warm, 0)

            # The uniform potential is RECORDED, never used to decide. It was
            # the gate until a reference measurement showed it cannot be:
            # on pbc2 with unstandardized covariates the potential at the
            # POSTERIOR MEAN - the centre of the distribution being sampled -
            # came out 1,465,425 against 118,098 for a uniform draw, so the
            # test ranked the best possible starting point 12x BELOW a random
            # one and rejected it. A quantity that does that cannot gate
            # anything, at any threshold.
            #
            # Two things were wrong with it. min() over three draws is an
            # order statistic, so the bar was the LUCKIEST of three rather
            # than a typical one. And a uniform draw can score well for a
            # reason unrelated to being a good start: a large sigma_e flattens
            # the longitudinal likelihood, lowering the potential without
            # improving anything. The baseline also swung 50,204 -> 776,297
            # on identical data across configurations, a 15x move in the bar.
            #
            # Recorded as a median rather than a min so the number in the
            # record is at least a typical draw rather than the best of three.
            _unifs = [_pe(_init.init_to_uniform, k) for k in (1, 2, 3)]
            _pe_unif = float(np.median([u for u in _unifs if np.isfinite(u)])
                             if any(np.isfinite(u) for u in _unifs) else np.inf)

            # ---- the conservative seed: JMbayes2's design ------------------
            # Built HERE, before the holdback tables and the decision, both of
            # which compare against it. It used to be constructed further down;
            # moving the gate to use it left _pe_safe referenced before
            # assignment, the whole self-check then raised, and every fit
            # silently started cold - caught by install_jmjax.sh's own check.
            # When the full warm start loses, the current behaviour throws
            # away ALL fourteen seeded values and starts cold - including
            # beta, sigma_e, sigma_b and b_std, which were verified correct
            # to 2-3 decimals against the generating truth. One bad site
            # discarding thirteen good ones is a worse outcome than either
            # extreme.
            #
            # JMbayes2 (R/jm.R, initial_values) does not face this because it
            # never seeds the risky sites:
            #     bs_gammas     <- rep(-0.1, ncol(W0_H))    # flat baseline
            #     alphas        <- rep(0.0, ...)            # zero association
            #     tau_bs_gammas <- 20                       # tight smoothing
            #     betas, sigmas, D, b <- from the mixed-model pre-fits
            #     gammas              <- coef(Surv_object)
            # That combination CANNOT blow up: with alpha = 0 the trajectory
            # does not enter the hazard, so a flat baseline has nothing to be
            # amplified by. It is strictly more informed than a cold start
            # and strictly safer than seeding alpha and the baseline jointly,
            # which is what couples two estimates through exp(alpha * m).
            #
            # A flat baseline in the RW2 parameterization is W01 = (c, c)
            # with z_step = 0, since W_k = W01[2] + k*(W01[2]-W01[1]) +
            # sigma_w * cumsum(cumsum(z)) collapses to c.
            _vals_safe = {k: v for k, v in _vals.items()
                          if not k.startswith("alpha")
                          and k not in ("W01", "z_step", "tau_w")}
            if "W01" in _vals:
                _vals_safe["W01"] = np.array([-0.1, -0.1], dtype=float)
            if "z_step" in _vals:
                _vals_safe["z_step"] = np.zeros_like(
                    np.asarray(_vals["z_step"], dtype=float))
            if "tau_w" in _vals:
                _vals_safe["tau_w"] = 20.0
            for _an in ("alpha", "alpha_value", "alpha_delta", "alpha_area",
                        "alpha_area_avg"):
                if _an in _vals:
                    _vals_safe[_an] = np.zeros_like(
                        np.asarray(_vals[_an], dtype=float))
            try:
                _safe = _init_to_value_jittered(values=_vals_safe,
                                                scale=_jit_scale)
                _pe_safe = _pe(_safe, 0)
            except Exception:
                _safe, _pe_safe = None, np.inf

            # ---- localize a rejection to a BLOCK ---------------------------
            # A single pair of numbers says the warm start is worse without
            # saying WHERE, which leaves the cause to be guessed at from
            # outside - and guessing produced three wrong hypotheses in a row
            # (a partial-seeding mixture, the time scale, and the b_std
            # inversion; the last is guarded by round-trip assertions and was
            # never a candidate). Re-evaluating with one block at a time held
            # back turns that into a measurement.
            #
            # LONGITUDINAL: beta, sigma_e, sigma_b, L_corr, b_std - what lme()
            # supplies directly. SURVIVAL: alpha*, gamma and the baseline
            # hazard spline (W01, z_step, tau_w) - seeded from a two-stage
            # coxph fit, and the ones that enter the likelihood through
            # exp(alpha * m_i(t)), where a modest error is amplified.
            _blocks = {
                "longitudinal": ("beta", "sigma_e", "sigma_b", "L_corr",
                                 "b_std", "b"),
                "survival": ("alpha", "alpha_value", "alpha_delta",
                             "alpha_area", "alpha_area_avg", "gamma",
                             "W01", "z_step", "tau_w"),
            }
            _pe_block = {}
            for _bn, _names in _blocks.items():
                _kept = {k: v for k, v in _vals.items() if k not in _names}
                if len(_kept) == len(_vals):
                    continue          # nothing from this block was seeded
                try:
                    _pe_block[_bn] = _pe(
                        _init_to_value_jittered(values=_kept, scale=_jit_scale), 0)
                except Exception:
                    pass
            warm_start_blocks = dict(_pe_block)

            # On REJECTION only, go one level finer: hold back each seeded
            # site on its own. Two block numbers say which half is wrong;
            # this says which VALUE is wrong, which is what a fix needs. It
            # costs one initialize_model call per site and runs only when the
            # warm start has already failed, so the normal path pays nothing.
            # Record the seeded VALUES for small sites, not just their
            # effect on the potential. Holding a site back is a differential
            # measurement and cannot separate "this value is wrong" from
            # "this value is right and tight, and is exposing an error
            # elsewhere" - sigma_e sits in front of a squared residual, so it
            # shows up either way. The value itself says which, and is the
            # cheapest check available; it should have been recorded from the
            # start rather than inferred through four rounds of holdback.
            warm_start_values = {}
            for _nm, _v in _vals.items():
                _a = np.asarray(_v)
                # b_std is recorded in full despite its size: it is the only
                # seeded site whose per-SUBJECT values are needed to check
                # that the warm start's subject ordering matches the model's,
                # and that alignment is the one step the existing round-trip
                # assertions do not cover.
                _cap = 4096 if _nm in ("b_std", "b") else 8
                if _a.size <= _cap:
                    warm_start_values[_nm] = (float(_a) if _a.size == 1
                                              else [float(x) for x in _a.ravel()])
                else:
                    warm_start_values[_nm] = {
                        "shape": list(_a.shape), "mean": float(np.mean(_a)),
                        "sd": float(np.std(_a)),
                        "absmax": float(np.max(np.abs(_a)))}

            # CAVEAT on both holdback tables: each variant re-initialises
            # the sites it holds back, and although PRNGKey(0) is fixed, the
            # set of sites drawing from it changes - so the numbers are NOT
            # a controlled comparison across rows. This confound is what made
            # them finger sigma_e, which a direct reconstruction later cleared
            # (seeded values reproduce lme's fit to 0.37 against 0.27). Read
            # them as a hint about where to look, never as evidence.
            warm_start_sites_pe = {}
            if not (np.isfinite(_pe_warm)
                    and (not np.isfinite(_pe_safe) or _pe_warm <= _pe_safe)):
                for _nm in sorted(_vals):
                    _kept = {k: v for k, v in _vals.items() if k != _nm}
                    try:
                        warm_start_sites_pe[_nm] = _pe(
                            _init_to_value_jittered(values=_kept,
                                                    scale=_jit_scale), 0)
                    except Exception:
                        pass

            # ---- the decision ---------------------------------------------
            # Both candidates are DELIBERATE points on the same scale, so the
            # comparison is meaningful in a way the uniform one was not:
            #
            #   full          everything the pre-fits estimate, including
            #                 alpha and a fitted baseline hazard
            #   conservative  JMbayes2's design: longitudinal block and the
            #                 survival covariates from the pre-fits, alpha at
            #                 zero, flat baseline. Structurally cannot blow
            #                 up - with alpha = 0 the trajectory never enters
            #                 the hazard, so nothing amplifies a bad baseline.
            #
            # The conservative seed is the FLOOR, not a fallback of last
            # resort. There is no path back to a cold start any more: it is
            # strictly less informed than the conservative seed on every
            # dataset, and discarding beta, sigma_e, sigma_b and b_std -
            # verified correct to 2-3 decimals against the generating truth -
            # because a lottery went the wrong way was never defensible.
            # The MEDIAN uniform potential is a SANITY FLOOR - not the gate it
            # used to be. Removing the uniform comparison altogether was a
            # mistake: it had two jobs, and only one of them was wrong.
            #
            # As a GATE it was indefensible, because min() over three draws is
            # a favourable order statistic and uniform potentials span nine
            # orders of magnitude (simulated: min 50,204 against a median of
            # 6.8e9), so a lucky draw rejected a warm start 40,000x better
            # than a typical cold one.
            #
            # As a FLOOR it was load-bearing, and dropping it let a
            # deliberately corrupt seed through - tests/testthat/
            # test-penalized-spline.R:226 supplies beta = c(500, 500) with
            # sigma_e = 0.01 and expects rejection. The conservative tier does
            # not catch that: it only neutralises alpha and the baseline, so
            # when a caller supplies just beta and sigma_e there is nothing to
            # strip and the "safe" seed IS the corrupt one.
            #
            # A median floor does both jobs correctly. Good warm starts beat a
            # typical draw by orders of magnitude (167,467 vs 6.8e9; 1,881 vs
            # 2.35e6), and catastrophic ones lose to it.
            _cand, _cand_pe, _cand_tier = None, np.inf, None
            if np.isfinite(_pe_warm) and (not np.isfinite(_pe_safe)
                                          or _pe_warm <= _pe_safe):
                _cand, _cand_pe, _cand_tier = _warm, _pe_warm, "full"
            elif _safe is not None and np.isfinite(_pe_safe):
                _cand, _cand_pe, _cand_tier = _safe, _pe_safe, "conservative"

            if _cand is not None and not (_cand_pe < _pe_unif):
                warnings.warn(
                    f"warm start REJECTED: the better of the two seeds scored "
                    f"{_cand_pe:.1f}, which is not better than a TYPICAL "
                    f"uniform start ({_pe_unif:.1f}). A seed this far off is "
                    "worse than no seed at all - check that the supplied "
                    "values are on the scale the model expects "
                    "(standardize_covariates, scale_time). Falling back to "
                    "the default start.",
                    RuntimeWarning)
                _cand, _cand_tier = None, None

            if _cand is not None and _cand_tier == "full":
                nuts_kwargs["init_strategy"] = _warm
                warm_start_check = {
                    "used": True, "tier": "full",
                    "potential_warm": _pe_warm,
                    "potential_conservative": _pe_safe,
                    "potential_uniform": _pe_unif,
                    "sites": sorted(_vals),
                    "potential_by_block_held_back": warm_start_blocks,
                    "potential_by_site_held_back": warm_start_sites_pe,
                    "values": warm_start_values,
                }
            elif _cand is not None:
                nuts_kwargs["init_strategy"] = _safe
                # Informational, not a warning: this fires on every fit
                # where the full seed's association parameter isn't yet
                # supported by the data as strongly as JMbayes2's flat/zero
                # starting point (a common, expected outcome - see the
                # comment above "the decision"). Using warnings.warn() here
                # made a routine event read like something had gone wrong.
                print(
                    "jmjax: warm start using the CONSERVATIVE seed "
                    f"(full seed scored {_pe_warm:.1f} vs {_pe_safe:.1f}). "
                    "The association parameter starts at zero with a flat "
                    "baseline hazard (the combination JMbayes2 uses), while "
                    "the longitudinal block and survival covariates are still "
                    "taken from the pre-fits - this is not a fallback to a "
                    "cold start, those values are kept.")
                warm_start_check = {
                    "used": True, "tier": "conservative",
                    "potential_warm": _pe_warm,
                    "potential_conservative": _pe_safe,
                    "potential_uniform": _pe_unif,
                    "sites": sorted(_vals_safe),
                    "potential_by_block_held_back": warm_start_blocks,
                    "potential_by_site_held_back": warm_start_sites_pe,
                    "values": warm_start_values,
                }
            else:
                # Reached only when BOTH deliberate points are non-finite,
                # which means the model cannot be evaluated there at all - a
                # structural problem, not a close call about start quality.
                warnings.warn(
                    "warm start UNUSABLE: neither the full seed "
                    f"({_pe_warm:.1f}) nor the conservative one ({_pe_safe:.1f}) "
                    "has a finite log-density, so the model cannot be evaluated "
                    "at either. This is a structural problem rather than a "
                    "judgement about start quality - check for non-finite "
                    "values in the pre-fits. "
                    + ("Holding one block back at a time gives: "
                       + ", ".join("%s -> %.1f" % (k, v)
                                   for k, v in warm_start_blocks.items())
                       + ". " if warm_start_blocks else "")
                    + ("Per site, the largest improvements from holding ONE "
                       "value back: "
                       + ", ".join(
                           "%s -> %.1f" % (k, v) for k, v in
                           sorted(warm_start_sites_pe.items(),
                                  key=lambda kv: kv[1])[:4])
                       + " (against %.1f with all of them). " % _pe_warm
                       if warm_start_sites_pe else "")
                    + "Falling back to the default start.",
                    RuntimeWarning)
                warm_start_check = {
                    "used": False,
                    "potential_warm": _pe_warm,
                    "potential_conservative": _pe_safe,
                    "potential_uniform": _pe_unif,
                    "sites": sorted(_vals),
                    "potential_by_block_held_back": warm_start_blocks,
                    "potential_by_site_held_back": warm_start_sites_pe,
                    "values": warm_start_values,
                }
        except (NameError, AttributeError) as _e:
            # A bug in THIS code, not a property of the model or the data.
            # Deliberately NARROW: TypeError/IndexError/KeyError are how a
            # seed reports that it does not fit this model configuration -
            # rw2_implementation="scan" gives TypeError("cannot reshape array
            # of shape (7,)") because z_step has a different shape there - and
            # those must warn and fall back rather than abort a fit that would
            # otherwise work.
            # It used to be folded into the generic message below, which reads
            # like a modelling judgement - so a NameError from a mis-ordered
            # variable silently disabled the warm start on every fit and was
            # caught only by install_jmjax.sh asserting warm_start$used.
            raise RuntimeError(
                "jmjax internal error in the warm-start self-check: %r. This "
                "is a bug in jmjax, not a problem with your data - please "
                "report it. The fit was stopped rather than silently running "
                "without the pre-fit starting values." % (_e,)) from _e
        except Exception as _e:                       # noqa: BLE001
            warnings.warn(f"warm-start self-check failed ({_e}); using the "
                          "default start.", RuntimeWarning)
            warm_start_check = {"used": False, "error": str(_e)}
    else:
        warm_start_check = None

    if control.get("find_heuristic_step_size") is not None:
        nuts_kwargs["find_heuristic_step_size"] = bool(
            control["find_heuristic_step_size"])
    beta_se = control.get("beta_se", None)
    if control.get("seed_mass_matrix", False) and beta_se is not None:
        try:
            beta_se_arr = jnp.atleast_1d(jnp.array(beta_se,
                                                   dtype=jnp.result_type(float)))
            # LENGTH CHECK: beta_se comes from the lme() pre-fit, which uses
            # long_formula - the same formula X_long is built from, so the
            # lengths SHOULD match. But a silent mismatch would produce a
            # wrongly-shaped mass matrix rather than an error, so verify
            # rather than assume. p is X_long.shape[-1], the beta dimension.
            if beta_se_arr.shape[0] != p:
                warnings.warn(
                    f"seed_mass_matrix: beta_se has length {beta_se_arr.shape[0]} "
                    f"but the model's beta has dimension {p} - skipping mass-matrix "
                    "seeding for this fit. The fit itself is unaffected.",
                    RuntimeWarning,
                )
            elif bool(jnp.all(jnp.isfinite(beta_se_arr))) and bool(jnp.all(beta_se_arr > 0)):
                nuts_kwargs["inverse_mass_matrix"] = {("beta",): beta_se_arr ** 2}
        except Exception:
            # Never let a seeding failure abort a fit - the unseeded
            # kernel is a perfectly valid fallback.
            pass

    inner_kernel = NUTS(model, **nuts_kwargs)

    if q == 2 and random_effects_method == "wishart_gibbs":
        # EXPERIMENTAL: hybridize a closed-form Wishart-conjugate Gibbs
        # update for D_inv (the random-effects precision matrix) with
        # NUTS for everything else - see build_model()'s matching branch
        # for the full derivation. Must use the EXACT SAME prior
        # (nu0, S0) as build_model()'s D_inv site above, or the two would
        # be sampling from inconsistent distributions - including the
        # wishart_eb_scale branch, mirrored here deliberately.
        #
        # NOTE: an earlier comment here credited this to "JMbayes2's
        # documented approach". That was WRONG - JMbayes2 uses a
        # separation strategy (Gamma priors on SDs centered on the lme()
        # pre-fit, plus a separate LKJ prior on correlation), confirmed by
        # direct inspection of a fitted jm object's $priors. The Wishart
        # conjugacy result comes from Rizopoulos 2016 JSS, which describes
        # the PREDECESSOR package JMbayes, and concerns conjugacy
        # mechanics rather than prior specification.
        if bool(control.get("wishart_eb_scale", False)) and sigma_b_prior_mean is not None:
            nu0 = float(q + 2)
            sd_hat = jnp.atleast_1d(jnp.array(sigma_b_prior_mean))
            D_hat = jnp.diag(sd_hat ** 2)
            S0 = jnp.linalg.inv(D_hat)
        else:
            nu0 = float(q + 1)
            S0 = jnp.eye(q)
        S0_inv = jnp.linalg.inv(S0)

        def _gibbs_fn(rng_key, gibbs_sites, hmc_sites):
            b_current = hmc_sites["b"]  # [N_sub, q]
            n_sub_current = b_current.shape[0]
            new_scale_inv = S0_inv + b_current.T @ b_current
            new_scale = jnp.linalg.inv(new_scale_inv)
            new_scale_sym = 0.5 * (new_scale + new_scale.T)
            new_df = nu0 + n_sub_current
            d_inv_new = dist.Wishart(concentration=new_df, scale_matrix=new_scale_sym).sample(rng_key)
            return {"D_inv": d_inv_new}

        kernel = HMCGibbs(inner_kernel, gibbs_fn=_gibbs_fn, gibbs_sites=["D_inv"])
    elif q == 2 and random_effects_method == "wishart_gibbs_centered":
        # EXPERIMENTAL, ADD-ON - fully separate from the "wishart_gibbs"
        # branch above (not modified). Identical Gibbs update, but reads
        # hmc_sites["b_raw"] (the RAW, uncentered latent variable that is
        # actually part of the HMC state) rather than hmc_sites["b"] (which
        # in this branch is a numpyro.deterministic transform of b_raw, not
        # itself a sampled site, and so is not exposed via hmc_sites at all).
        nu0 = float(q + 1)
        S0 = jnp.eye(q)
        S0_inv = jnp.linalg.inv(S0)

        def _gibbs_fn_centered(rng_key, gibbs_sites, hmc_sites):
            b_raw_current = hmc_sites["b_raw"]  # [N_sub, q] - the raw, uncentered latent
            n_sub_current = b_raw_current.shape[0]
            new_scale_inv = S0_inv + b_raw_current.T @ b_raw_current
            new_scale = jnp.linalg.inv(new_scale_inv)
            new_scale_sym = 0.5 * (new_scale + new_scale.T)
            new_df = nu0 + n_sub_current
            d_inv_new = dist.Wishart(concentration=new_df, scale_matrix=new_scale_sym).sample(rng_key)
            return {"D_inv": d_inv_new}

        kernel = HMCGibbs(inner_kernel, gibbs_fn=_gibbs_fn_centered, gibbs_sites=["D_inv"])
    else:
        kernel = inner_kernel

    mcmc = MCMC(
        kernel,
        num_warmup=int(control.get("num_warmup", 500)),
        num_samples=int(control.get("num_samples", 1000)),
        num_chains=int(control.get("num_chains", 1)),
        progress_bar=control.get("progress_bar", True),
    )

    seed = int(control.get("seed", 2026))
    start = time.perf_counter()
    # extra_fields=("num_steps",) is only requested for plain NUTS - under
    # HMCGibbs wrapping (random_effects_method="wishart_gibbs"), the
    # internal state structure differs and requesting this broke the
    # previously-validated wishart_gibbs path entirely (a real bug caught
    # by testing, not just theorized). This diagnostic is nice-to-have,
    # not essential - better to skip it than let it break core
    # functionality that's already been validated to work.
    using_hmc_gibbs = q == 2 and random_effects_method in ("wishart_gibbs", "wishart_gibbs_centered")
    # "diverging" alongside "num_steps". A divergence means the leapfrog
    # integrator lost energy conservation - the step size was too large
    # for the local curvature - and the sampler may be silently skipping
    # a region of the posterior rather than exploring it.
    #
    # This matters for any change to target_accept_prob. Lowering it grows
    # the adapted step size, which raises ESS by making consecutive draws
    # less correlated (measured: 0.8 -> 0.65 roughly DOUBLED alpha's ESS on
    # epileptic, 2260 -> 4385, at fewer steps). But a higher ESS from a
    # chain that is quietly under-exploring is worse than a lower one from
    # a chain that is not, and divergence counts are what distinguishes
    # them. Without this field the comparison cannot be made honestly.
    run_kwargs = {} if using_hmc_gibbs else {
        "extra_fields": ("num_steps", "diverging")}

    mcmc.run(
        jax.random.PRNGKey(seed),
        # Same dict the warm-start self-check above evaluated against.
        **model_kwargs,
        **run_kwargs,
    )
    # JAX dispatches computation ASYNCHRONOUSLY - mcmc.run() can return as
    # soon as the computation is dispatched to the device, without
    # waiting for it to actually finish executing. Measuring `elapsed`
    # immediately here without forcing completion first would capture
    # only DISPATCH time, not real completion time - the actual wait
    # would then show up LATER, wherever the code first tries to read a
    # value (confirmed via direct profiling during development:
    # numpyro's own diagnostic math takes milliseconds; the actual delay
    # was jax.device_get() blocking on the real mcmc.run() computation
    # finally finishing, whenever a value was first read). Forcing
    # completion here, before measuring elapsed, makes sampling_time_sec
    # accurate.
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - start

    # Diagnostic: mean/max leapfrog steps per iteration - lets us directly
    # confirm (rather than just theorize) whether "penalized" forces
    # deeper NUTS trees than "independent". Wrapped defensively since
    # extra_fields access under HMCGibbs wrapping is less thoroughly
    # exercised elsewhere in this codebase than plain NUTS.
    try:
        _xf = mcmc.get_extra_fields()
        num_steps = np.asarray(_xf["num_steps"])
        mean_num_steps = float(np.mean(num_steps))
        max_num_steps = float(np.max(num_steps))
        if "diverging" in _xf:
            _div = np.asarray(_xf["diverging"])
            n_divergences = int(_div.sum())
            divergence_rate = float(_div.mean())
        else:
            n_divergences = divergence_rate = None
    except Exception:
        mean_num_steps = None
        max_num_steps = None
        n_divergences = divergence_rate = None

    samples = mcmc.get_samples(group_by_chain=False)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)

    # Derive rho (and, for wishart_gibbs, sigma_b itself) from the raw
    # matrix-valued site actually sampled, injecting them as their own
    # named entries so they get PROPER R-hat/ESS diagnostics (computed on
    # the actual derived-quantity trace, not just inherited from the raw
    # matrix) and get picked up automatically by population_sites below -
    # "sigma_b" already has a size>1 naming rule (sigma_b0/sigma_b1);
    # "rho" is added there explicitly.
    if "D_inv" in samples:
        # wishart_gibbs: D_inv -> D -> sigma_b0, sigma_b1, rho. This also
        # fixes what was otherwise a silent reporting gap for this new
        # path - without this, sigma_b/rho would never appear in
        # fit$estimates despite being the entire point of q=2.
        D_inv_flat = np.asarray(samples["D_inv"])
        D_flat = np.linalg.inv(D_inv_flat)
        sigma_b0_flat = np.sqrt(D_flat[:, 0, 0])
        sigma_b1_flat = np.sqrt(D_flat[:, 1, 1])
        rho_flat = D_flat[:, 0, 1] / (sigma_b0_flat * sigma_b1_flat)
        samples["sigma_b"] = np.stack([sigma_b0_flat, sigma_b1_flat], axis=-1)
        samples["rho"] = rho_flat

        D_inv_bc = np.asarray(samples_by_chain["D_inv"])
        D_bc = np.linalg.inv(D_inv_bc)
        sigma_b0_bc = np.sqrt(D_bc[..., 0, 0])
        sigma_b1_bc = np.sqrt(D_bc[..., 1, 1])
        rho_bc = D_bc[..., 0, 1] / (sigma_b0_bc * sigma_b1_bc)
        samples_by_chain["sigma_b"] = jnp.stack([jnp.array(sigma_b0_bc), jnp.array(sigma_b1_bc)], axis=-1)
        samples_by_chain["rho"] = jnp.array(rho_bc)
    elif "L_corr" in samples:
        # Existing LKJCholesky-based correlated case - rho was previously
        # NEVER extracted/reported at all (a genuine pre-existing gap,
        # found and fixed here rather than left as-is): for a 2x2
        # correlation Cholesky factor, L_corr[1,0] IS rho directly (no
        # further transform needed - L_corr @ L_corr.T is the correlation
        # matrix itself, whose off-diagonal element is rho by definition).
        L_corr_flat = np.asarray(samples["L_corr"])
        samples["rho"] = L_corr_flat[:, 1, 0]
        L_corr_bc = np.asarray(samples_by_chain["L_corr"])
        samples_by_chain["rho"] = jnp.array(L_corr_bc[..., 1, 0])

    # ------------------------------------------------------------------
    # SEPARATE, OPT-IN TRACK: "beta_corrected", an analytically
    # un-orthogonalized beta.
    #
    # WHY THIS EXISTS. dev/study_calibration.R's pilot run found that
    # orthogonalize_b/_b0 reproduce random_effects_method =
    # "wishart_gibbs_centered"'s retracted failure mode: unbiased point
    # estimates for the swept parameter(s) (Corollary 1's model-invariance
    # holds), but posterior SD too small by close to the theoretically
    # predicted sigma_b/sqrt(N) - because the swept component of b_raw,
    # c = Q^T b_raw[:, q], is unidentified under the likelihood and simply
    # draws from its prior/conditional distribution, with that uncertainty
    # reflected nowhere in beta's own posterior. This refutes the escape
    # hypothesis raised in vignette("jmjax-reparameterization") Section 4.7;
    # see Section 9 for the numbers.
    #
    # WHAT IT COMPUTES. For every draw, beta_orth and b_raw are related to
    # beta_default (what an UNCONSTRAINED fit's beta would be, given the
    # SAME b_raw draw) by
    #
    #     beta_default = beta_orth - sum_q  Dmat_q @ (Q_q^T @ b_raw[:, q])
    #
    # (see _absorbable_generator_matrix()'s docstring for the derivation).
    # This is PURE POST-PROCESSING of draws this fit already produced - no
    # new numpyro site, no refit - computed whenever orthogonalize_b/_b0
    # found at least one absorbable direction, with no separate control
    # flag: it only adds a new "beta_corrected" key, so it cannot change
    # any existing output ("beta" itself is untouched).
    #
    # NOT YET HANDLED: control$standardize_covariates / control$scale_time
    # (R/jm_fit.R) back-transform posterior_samples$beta but do not know
    # about this key - a fit using either option will have beta_corrected
    # on the SAMPLED, not the back-transformed, scale. Not exercised by
    # dev/study_calibration.R, which uses neither; flagged here so a future
    # caller does not assume otherwise.
    # ------------------------------------------------------------------
    if (_correction_bases is not None
            and any(_D is not None for _D in _correction_bases)
            and "beta" in samples and "b_std" in samples
            and "L_corr" in samples and "sigma_b" in samples):
        _beta_d = np.asarray(samples["beta"])              # [n_draws, p]
        _bstd_d = np.asarray(samples["b_std"])              # [n_draws, N_sub, q]
        _Lcorr_d = np.asarray(samples["L_corr"])            # [n_draws, q, q]
        _sigb_d = np.asarray(samples["sigma_b"])            # [n_draws, q]
        _L_d = _sigb_d[:, :, None] * _Lcorr_d               # [n_draws, q, q]
        # b_raw = b_std @ L.T, batched over draws.
        _braw_d = np.einsum("dnq,dpq->dnp", _bstd_d, _L_d)  # [n_draws, N_sub, q]
        _delta = np.zeros_like(_beta_d)
        for _q_idx, _Dmat in enumerate(_correction_bases):
            if _Dmat is None:
                continue
            _Qb = b_orth_bases[_q_idx]                      # [N_sub, k]
            _c = np.einsum("nk,dn->dk", _Qb, _braw_d[:, :, _q_idx])  # [n_draws, k]
            _delta += _c @ _Dmat.T                           # [n_draws, p]
        samples["beta_corrected"] = _beta_d - _delta

    # "b" (q=1's directly-sampled site, or q=2's deterministic b_std @
    # L.T transform, or q=2 wishart_gibbs's directly-sampled site) and
    # "b_std" (q=2 LKJCholesky case's actual per-subject sampled site for
    # the non-centered parameterization) are both sampled/computed
    # per-subject ([N_sub] or [N_sub, q]) - excluded here since neither
    # is ever included in population_sites below, so computing their
    # full R-hat/ESS diagnostics would be wasted work. Both sites' raw
    # samples remain fully available via `samples` above for ranef().
    _per_subject_sites = {"b", "b_std", "b_raw", "z_step"}
    samples_by_chain_for_diag = {k: v for k, v in samples_by_chain.items() if k not in _per_subject_sites}
    diag = numpyro_summary(samples_by_chain_for_diag, group_by_chain=True)

    return _package_result(samples, diag, p, q, n_splines, N_sub, elapsed,
                            mean_num_steps=mean_num_steps, max_num_steps=max_num_steps,
                            n_divergences=n_divergences,
                            divergence_rate=divergence_rate,
                            warm_start_check=warm_start_check,
                            orthogonalize_report=orthogonalize_report)


# Explicit naming per site, matching the MLE backends' convention
# (beta_0/beta_1..., W0/W1..., sigma_e/sigma_b/alpha as bare scalars) so
# results are comparable/interchangeable across all three methods.
def _site_names(site, size):
    if site == "beta":
        return [f"beta_{i}" for i in range(size)]
    if site == "W":
        return [f"W{i}" for i in range(size)]
    if site == "sigma_b" and size > 1:
        return [f"sigma_b{i}" for i in range(size)]
    if site == "gamma":
        return [f"gamma_{i}" for i in range(size)]
    return [site]  # scalar sites: sigma_e, alpha, sigma_b (q=1), L_corr handled separately


def recompute_site_diagnostics(draws, num_chains=1):
    """R-hat and ESS for draws that R has transformed after sampling.

    WHY THIS EXISTS. When control$standardize_covariates is active, jm_fit()
    back-transforms beta to the ORIGINAL covariate scale after the fit -
    rewriting estimates, se and posterior_samples. It did NOT rewrite
    diagnostics$ess or diagnostics$rhat, which are computed here, during
    sampling, on the STANDARDIZED parameters. So a user comparing
    fit$estimates[["beta_0"]] with fit$diagnostics$ess[["beta_0"]] was
    reading a diagnostic for a different quantity than the estimate.

    Recomputing in R would mean a second, hand-rolled ESS estimator
    disagreeing with this one in the third digit for no good reason. This
    routine runs numpyro's own summary over the transformed draws instead,
    so every ESS the package reports comes from one implementation.

    draws: [n_draws, p] array, chains CONCATENATED (as get_samples returns
    them). Reshaped to [num_chains, n_per_chain, p] so R-hat is genuinely
    between-chain rather than split-within-one.
    """
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_total = arr.shape[0]
    nc = max(1, int(num_chains))
    if n_total % nc != 0:
        # Not divisible: fall back to one chain rather than silently
        # mis-grouping draws, which would corrupt R-hat.
        nc = 1
    grouped = arr.reshape(nc, n_total // nc, arr.shape[1])
    summ = numpyro_summary({"beta": grouped}, prob=0.9, group_by_chain=True)["beta"]
    return {"r_hat": np.atleast_1d(summ["r_hat"]).tolist(),
            "n_eff": np.atleast_1d(summ["n_eff"]).tolist()}


def _package_result(samples, diag, p, q, n_splines, N_sub, elapsed,
                     mean_num_steps=None, max_num_steps=None,
                     n_divergences=None, divergence_rate=None,
                     warm_start_check=None, orthogonalize_report=None):
    # gamma's size isn't a fixed factory parameter like p/q/n_splines (it
    # depends on how many baseline covariates were in surv_formula, which
    # build_model() never needed to know ahead of time - W_surv's shape
    # was read directly at trace time) - inferred here from the actual
    # samples instead, defaulting to 0 (site simply absent) when no
    # baseline covariates were used.
    n_gamma = np.asarray(samples["gamma"]).shape[-1] if "gamma" in samples else 0

    population_sites = [("beta", p), ("sigma_e", 1), ("alpha", 1), ("W", n_splines), ("sigma_b", q),
                         ("rho", 1),  # only present when q=2 (either correlated LKJ or wishart_gibbs)
                         ("tau_w", 1),        # only present when spline_prior="penalized"
                         ("log_lambda0", 1), ("shape", 1),  # only present when baseline_hazard="weibull"
                         ("alpha_value", 1), ("alpha_delta", 1),
                         ("alpha_area", 1), ("alpha_area_avg", 1),  # only present when the corresponding channel is used (alpha absent in that case)
                         ("gamma", n_gamma)]  # only present when surv_formula has baseline covariates

    estimates, se, rhat, ess = {}, {}, {}, {}

    for site, size in population_sites:
        if site not in samples:
            continue
        arr = np.asarray(samples[site])
        mean = np.atleast_1d(arr.mean(axis=0))
        sd = np.atleast_1d(arr.std(axis=0))
        site_diag = diag.get(site, {})
        r_hat = np.atleast_1d(np.asarray(site_diag.get("r_hat", np.full(size, np.nan))))
        n_eff = np.atleast_1d(np.asarray(site_diag.get("n_eff", np.full(size, np.nan))))

        names = _site_names(site, size)
        for i, name in enumerate(names):
            estimates[name] = float(mean[i])
            se[name] = float(sd[i])
            rhat[name] = float(r_hat[i]) if i < len(r_hat) else float("nan")
            ess[name] = float(n_eff[i]) if i < len(n_eff) else float("nan")

    # --- Per-subject random effects, kept separate from population estimates ---
    random_effects = {"subject_index": list(range(1, N_sub + 1))}
    if "b" in samples:
        b_arr = np.asarray(samples["b"])  # [n_samples, N_sub] or [n_samples, N_sub, q]
        b_mean = b_arr.mean(axis=0)
        b_sd = b_arr.std(axis=0)
        if b_mean.ndim == 1:
            random_effects["b_mean"] = b_mean.tolist()
            random_effects["b_sd"] = b_sd.tolist()
        else:
            for j in range(b_mean.shape[1]):
                random_effects[f"b{j}_mean"] = b_mean[:, j].tolist()
                random_effects[f"b{j}_sd"] = b_sd[:, j].tolist()

    max_rhat = float(np.nanmax(list(rhat.values()))) if rhat else float("nan")
    converged = bool(max_rhat < RHAT_CONVERGED_THRESHOLD) if not np.isnan(max_rhat) else True
    message = (f"NUTS completed in {elapsed:.2f}s; "
               f"max split R-hat across population parameters = {max_rhat:.4f}")
    if not converged:
        message += (f" (>= {RHAT_CONVERGED_THRESHOLD} threshold - consider more warmup/samples "
                     f"or checking for a multimodal/poorly-identified posterior)")

    return {
        "estimates": estimates,
        "se": se,
        "vcov": None,
        "loglik": None,  # would need a separate log-density evaluation at, e.g., posterior means
        "convergence": {"converged": converged, "message": message, "n_iter": None,
                         "sampling_time_sec": elapsed,  # NUTS-only time, comparable to
                                                          # JMbayes2's running_time["elapsed"] -
                                                          # excludes R-side data prep / reticulate
                                                          # marshaling on both sides for a fair
                                                          # timing comparison in the benchmark.
                         "mean_num_steps": mean_num_steps, "max_num_steps": max_num_steps,
                         # Divergences: a non-zero count means the leapfrog
                         # integrator lost energy conservation somewhere, and
                         # the chain may be skipping a region of the posterior
                         # rather than exploring it. R-hat and ESS will NOT
                         # necessarily show that - a chain can mix well across
                         # the part of the space it does reach.
                         "n_divergences": n_divergences,
                         "divergence_rate": divergence_rate,
                         # NULL unless control$init_values was supplied.
                         # "used" is FALSE when the self-check rejected the
                         # warm start, so a rejection is visible in the
                         # fitted object rather than only in a warning that
                         # a non-interactive run would swallow.
                         "warm_start": warm_start_check,
                         # NULL unless control$orthogonalize_b/_b0 was
                         # requested. See the construction site (above,
                         # near "control$orthogonalize_b0") for why this
                         # exists rather than relying on the warnings.warn()
                         # text.
                         "orthogonalize": orthogonalize_report},
        "diagnostics": {"rhat": rhat, "ess": ess},
        "random_effects": random_effects,
        "posterior_samples": {k: np.asarray(v).tolist() for k, v in samples.items()},
    }
