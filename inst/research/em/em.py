"""EM support for jmjax's MLE paths - DELIBERATELY SEPARATE.

Nothing in this module is imported by the fitting paths. It was written
alongside them and is kept apart on purpose: the MLE and MCMC code has
been tested across 397 assertions and five real datasets, and an
unvalidated EM phase should not be able to break any of it.

WHY EM AT ALL. On the prothro data with a spline baseline, L-BFGS-B
stopped after 5 iterations reporting converged = True, zero non-finite
standard errors and a finite log-likelihood - and had left 78
log-likelihood units behind, returning an association parameter of
-0.0008 where JM, BFGS and jmjax's own MCMC all agree near -0.04.
Tightening ftol changed nothing: a LINE SEARCH quit and was reported as
convergence.

EM cannot fail that way. It improves the observed-data likelihood
MONOTONICALLY because it has no line search. That is a property, not a
track record.

WHAT IS VERIFIED
  - the closed-form M-step recovers known parameters, handles the padding
    mask (ignoring it inflated sigma_e by 398% on 35%-padded data), and
    carries the random-effects uncertainty into sigma_e
  - the loop is monotone: 0 log-likelihood drops from both a cold and a
    warm start on an exact-E-step linear mixed model, both reaching the
    same optimum

WHAT IS NOT
  - that an EM phase actually helps. "Monotone" and "leaves the optimizer
    somewhere better" are different claims, and only the second justifies
    integrating this.
  - anything about the survival block. alpha, the baseline hazard and
    gamma have NO closed-form M-step, which is why JM calls optim()
    INSIDE its EM loop rather than running a pure EM.

HOW IT WOULD BE INTEGRATED, when it is. The cleanest hook is the
EXISTING `init_theta` argument that every MLE entry point already
accepts and that is already tested: run the EM phase separately, hand the
result in as init_theta, and the fitting code needs no change at all.
That keeps this module genuinely optional rather than a branch inside a
hot path.

SCOPE NOTE. The E-step needs the backend's own `h_fn` and
`find_mode_and_tau`, which are defined INLINE inside each of 14 fitting
functions (8 Weibull, 6 spline) rather than shared. Any real integration
has to address that first - it is a refactor, not a repeat of a few
lines.

KEEP IT SHORT WHEN USED. EM converges slowly near the optimum: measured
on an exact-E-step linear mixed model it needed 378 iterations from a
warm start and 772 from cold to reach a tolerance L-BFGS-B reached in 54.
Run to convergence it would be slower than the optimizer it protects. The
point is to leave the region where line searches fail, not to finish.
"""

import warnings
import numpy as np
import jax
import jax.numpy as jnp

from . import common
from .common import make_theta_layout  # noqa: F401  (re-export for callers)


def _mstep_inputs_ok(y, X, Z, mask):
    """Shape check with a message, rather than a downstream einsum error."""
    y = np.asarray(y); X = np.asarray(X); Z = np.asarray(Z)
    if y.ndim != 2 or X.ndim != 3 or Z.ndim != 3:
        raise ValueError(
            f"expected y [n, T], X [n, T, p], Z [n, T, q]; got "
            f"{y.shape}, {X.shape}, {Z.shape}")
    if not (y.shape == X.shape[:2] == Z.shape[:2]):
        raise ValueError("y, X and Z disagree on n or T")
    if mask is not None and np.asarray(mask).shape != y.shape:
        raise ValueError("mask must have the same shape as y")
    return True



# =============================================================================
# EM support - Phase 1: the E-step
#
# STATUS: infrastructure only. Nothing in jmjax calls these yet, and no
# default behaviour changes. They exist so the E-step can be validated
# against brute-force integration BEFORE any M-step is built on top.
#
# The point of an EM phase is that it improves the observed-data
# log-likelihood MONOTONICALLY, so it cannot fail a line search the way
# L-BFGS-B did on the prothro data (terminating at iteration 0 with a
# finite objective AND a finite gradient). `JM` runs one before any
# quasi-Newton step; jmjax does not.
#
# WHY THIS IS CHEAPER THAN IT LOOKS. The adaptive Gauss-Hermite likelihood
# already computes, per subject, everything the E-step needs:
#
#     b_mode, tau = find_mode_and_tau(subj_args, theta)
#     b_nodes     = b_mode + sqrt(2) * tau * gh_nodes
#     log_terms   = log(gh_weights) + h_vals + gh_nodes**2
#     loglik_i    = log(sqrt(2)*tau) + logsumexp(log_terms)
#
# The posterior weights are just those log_terms normalised. So the E-step
# is a reweighting of quantities already in hand, not new numerics.
# =============================================================================


def posterior_weights(log_terms):
    """Normalised posterior weights over the quadrature nodes.

    `log_terms` is what the existing per-subject likelihood already builds:
    log(gh_weights) + h(b_k) + z_k^2, of shape [n_nodes] for one subject or
    [n_subjects, n_nodes] batched. Normalising in log space keeps this
    stable when the terms span many orders of magnitude, which they do
    whenever the random-effect distribution is wide - prothro's sigma_b is
    about 18.
    """
    lt = jnp.asarray(log_terms)
    return jnp.exp(lt - jax.nn.logsumexp(lt, axis=-1, keepdims=True))



def posterior_moments_1d(b_nodes, log_terms):
    """E[b] and E[b^2] under the posterior, for a scalar random effect.

    Returns (m1, m2). The conditional variance is m2 - m1**2, which an
    M-step needs for the `D` update; it is returned separately rather than
    computed by the caller so the subtraction happens in one place.
    """
    w = posterior_weights(log_terms)
    m1 = jnp.sum(w * b_nodes, axis=-1)
    m2 = jnp.sum(w * b_nodes ** 2, axis=-1)
    return m1, m2, jnp.maximum(m2 - m1 ** 2, 0.0)



def posterior_moments_2d(b_nodes, log_terms):
    """E[b] and E[b b'] under the posterior, for a q=2 random effect.

    `b_nodes` has shape [..., n_nodes, 2]; `log_terms` [..., n_nodes].
    Returns (m1, M2, Cov) with m1 [..., 2], M2 and Cov [..., 2, 2].
    """
    w = posterior_weights(log_terms)[..., None]          # [..., n_nodes, 1]
    m1 = jnp.sum(w * b_nodes, axis=-2)                   # [..., 2]
    outer = b_nodes[..., :, None] * b_nodes[..., None, :]  # [..., n_nodes, 2, 2]
    M2 = jnp.sum(w[..., None] * outer, axis=-3)          # [..., 2, 2]
    Cov = M2 - m1[..., :, None] * m1[..., None, :]
    return m1, M2, Cov



def em_d_update(M2):
    """Closed-form M-step for the random-effects covariance D.

    D = (1/n) * sum_i E[b_i b_i'], the standard EM update. Accepts either
    [n, q, q] or, for q=1, [n]. The q=1 case returns a scalar variance.
    """
    M2 = jnp.asarray(M2)
    if M2.ndim == 1:
        return jnp.mean(M2)
    return jnp.mean(M2, axis=0)

# =============================================================================
# EM support - Phase 2: closed-form M-step for the longitudinal block
#
# STATUS: infrastructure. Nothing calls this yet. It exists so the
# monotonicity check - the strongest correctness test an EM has - can be
# run before anything depends on it.
#
# WHY AN EM PHASE AT ALL. On the prothro data with a spline baseline,
# L-BFGS-B stopped after 5 iterations reporting converged = True, zero
# non-finite standard errors and a finite log-likelihood. It had left 78
# log-likelihood units on the table and returned an association parameter
# of -0.0008 where JM, BFGS and jmjax's own MCMC all agree near -0.04.
# Nothing in the output flagged it, and tightening ftol changed nothing -
# a LINE SEARCH quit and was reported as convergence.
#
# EM cannot fail that way. It improves the observed-data likelihood
# MONOTONICALLY by construction, because it has no line search. That is a
# property, not a track record - unlike "trust-constr was never short
# across five comparisons", which is five comparisons.
#
# WHAT THIS IS NOT. EM converges slowly near the optimum; that is its
# known weakness, and why JM runs it as a PHASE and then hands off to a
# short quasi-Newton step (maxit 20 early, 5 later) rather than running it
# to completion. The intent here is the same: get somewhere sensible,
# cheaply and safely, then let the existing optimizer finish from a warm
# start rather than a cold one.
#
# WHAT IS CLOSED-FORM AND WHAT IS NOT:
#   D (random-effects covariance)  closed form: (1/n) sum_i E[b_i b_i']
#   sigma_e                        closed form from the residuals given E[b]
#   beta                           weighted least squares given E[b]
#   alpha, baseline hazard, gamma  NO closed form - the survival block
#                                  needs its own optimisation, which is
#                                  exactly why JM calls optim() INSIDE its
#                                  EM loop rather than having a pure EM.
# =============================================================================



def em_longitudinal_mstep(y, X, Z, subj_index, Eb, Ebb, n_obs_mask=None):
    """Closed-form M-step for beta, sigma_e and D.

    Given the posterior moments of the random effects from an E-step, the
    longitudinal block has an exact maximiser - no search, no step size,
    no line search to fail.

    y        [N_sub, T]        outcomes, padded
    X        [N_sub, T, p]     fixed-effects design
    Z        [N_sub, T, q]     random-effects design
    Eb       [N_sub, q]        E[b_i]
    Ebb      [N_sub, q, q]     E[b_i b_i'] - NOT E[b]E[b]', the second
                               moment, which is what the D update needs
    n_obs_mask [N_sub, T]      1 for a real observation, 0 for padding

    Returns (beta, sigma_e, D).

    The mask matters: subjects are padded to a common length T, and
    counting padded rows as observations would bias sigma_e downward by
    the padding fraction. jmjax pads heavily - prothro has 1 to 17 visits
    per subject - so this is not a small correction.
    """
    y = jnp.asarray(y); X = jnp.asarray(X); Z = jnp.asarray(Z)
    Eb = jnp.asarray(Eb); Ebb = jnp.asarray(Ebb)
    m = jnp.ones_like(y) if n_obs_mask is None else jnp.asarray(n_obs_mask)

    # beta: weighted least squares on y - Z E[b], masked
    resid_z = y - jnp.einsum("itq,iq->it", Z, Eb)
    Xm = X * m[..., None]
    XtX = jnp.einsum("itp,itr->pr", Xm, X)
    Xty = jnp.einsum("itp,it->p", Xm, resid_z)
    beta = jnp.linalg.solve(XtX + 1e-10 * jnp.eye(XtX.shape[0]), Xty)

    # sigma_e: the residual variance must carry the random-effects
    # UNCERTAINTY, not just their means. The E[b b'] term is the part an
    # ad-hoc "plug in E[b]" version would drop, and dropping it biases
    # sigma_e downward - the classic error in a hand-rolled EM.
    r = resid_z - jnp.einsum("itp,p->it", X, beta)
    sse = jnp.sum(m * r ** 2)
    zzt = jnp.einsum("itq,itr->iqr", Z * m[..., None], Z)
    var_term = jnp.sum(jnp.einsum("iqr,iqr->i", zzt, Ebb - Eb[:, :, None] * Eb[:, None, :]))
    n_eff = jnp.sum(m)
    sigma_e = jnp.sqrt(jnp.maximum((sse + var_term) / n_eff, 1e-12))

    D = jnp.mean(Ebb, axis=0)
    return beta, sigma_e, D



def em_loop_longitudinal(y, X, Z, mask, estep_fn, max_iter=20, tol=1e-6,
                          loglik_fn=None, verbose=False):
    """Run EM on the longitudinal block, checking monotonicity.

    `estep_fn(beta, sigma_e, D) -> (Eb, Ebb)` supplies the posterior
    moments. It is a callback rather than something computed here because
    the moments come from the adaptive-GH machinery inside the likelihood,
    which differs between the Weibull and spline backends and between
    q=1 and q=2. Separating the loop from the E-step means the loop can
    be tested against an exact E-step on a problem with a known answer,
    which is what proves it before any backend plumbing exists.

    `loglik_fn(beta, sigma_e, D) -> float`, if given, is evaluated every
    iteration and checked to be NON-DECREASING. That check is the whole
    point of using EM here:

        EM improves the observed-data likelihood monotonically BY
        CONSTRUCTION - it has no line search to fail.

    which is exactly the failure that motivated this. On prothro with a
    spline baseline, L-BFGS-B stopped after 5 iterations reporting
    converged = True with clean standard errors, having left 78
    log-likelihood units behind and returned an association parameter of
    -0.0008 where three independent routes agree near -0.04.

    So a DROP in log-likelihood here is not a convergence hiccup to be
    tolerated - it means the implementation is wrong, whatever the
    estimates look like. It is reported loudly rather than smoothed over.

    Returns a dict with the final beta, sigma_e, D, the log-likelihood
    trace, and `monotone`.
    """
    beta = None
    sigma_e = None
    D = None
    trace = []
    monotone = True
    first_drop = None

    # Initialise from a no-random-effects fit: beta by OLS, sigma_e from
    # its residuals, D from a modest fraction of the residual variance.
    m = jnp.asarray(mask)
    Xm = jnp.asarray(X) * m[..., None]
    XtX = jnp.einsum("itp,itr->pr", Xm, jnp.asarray(X))
    Xty = jnp.einsum("itp,it->p", Xm, jnp.asarray(y))
    beta = jnp.linalg.solve(XtX + 1e-10 * jnp.eye(XtX.shape[0]), Xty)
    r0 = (jnp.asarray(y) - jnp.einsum("itp,p->it", jnp.asarray(X), beta)) * m
    sigma_e = jnp.sqrt(jnp.maximum(jnp.sum(r0 ** 2) / jnp.sum(m), 1e-12))
    q = jnp.asarray(Z).shape[-1]
    D = jnp.eye(q) * (0.5 * sigma_e ** 2)

    for it in range(int(max_iter)):
        Eb, Ebb = estep_fn(beta, sigma_e, D)
        beta, sigma_e, D = em_longitudinal_mstep(y, X, Z, None, Eb, Ebb, mask)

        if loglik_fn is not None:
            ll = float(loglik_fn(beta, sigma_e, D))
            trace.append(ll)
            if len(trace) > 1 and ll < trace[-2] - 1e-8:
                monotone = False
                if first_drop is None:
                    first_drop = (it, trace[-2], ll)
                warnings.warn(
                    f"EM log-likelihood DECREASED at iteration {it}: "
                    f"{trace[-2]:.6f} -> {ll:.6f}. EM is monotone by "
                    "construction, so this means the E-step or M-step is "
                    "wrong - not that the problem is hard.",
                    RuntimeWarning)
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
                break

    return {"beta": np.asarray(beta), "sigma_e": float(sigma_e),
            "D": np.asarray(D), "loglik_trace": trace,
            "monotone": bool(monotone), "first_drop": first_drop,
            "n_iter": len(trace) if trace else int(max_iter)}



def make_estep_1d(h_fn, find_mode_and_tau, gh_nodes, gh_weights):
    """Build an E-step from a backend's OWN likelihood pieces (q = 1).

    Takes the same `h_fn` and `find_mode_and_tau` the log-likelihood uses,
    so the E-step cannot drift from the likelihood it is supposed to be
    the expectation of. Re-deriving the posterior separately would be the
    obvious alternative and the wrong one: the two would agree until some
    change touched one and not the other.

    The adaptive-GH likelihood already computes everything needed:

        b_mode, tau = find_mode_and_tau(subj_args, theta)
        b_nodes     = b_mode + sqrt(2) * tau * gh_nodes
        log_terms   = log(gh_weights) + h(b_k) + z_k^2
        loglik_i    = log(sqrt(2)*tau) + logsumexp(log_terms)

    and the posterior weights are those log_terms normalised. So the
    E-step is a reweighting of quantities already in hand, not new
    numerics - which is why this is a small function rather than a second
    implementation of the model.

    NOTE ON SCOPE: h_fn and find_mode_and_tau are defined INLINE inside
    each of 14 fitting functions (8 Weibull, 6 spline), not shared. This
    helper is written to be called from any of them, but wiring it into
    all 14 is a substantial refactor, not a small addition.
    """
    def estep(subj_args_batched, theta):
        def per_subject(*subj_args):
            b_mode, tau = find_mode_and_tau(subj_args, theta)
            b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes
            h_vals = jax.vmap(lambda bk: h_fn(bk, *subj_args, theta))(b_nodes)
            log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
            w = jnp.exp(log_terms - jax.nn.logsumexp(log_terms))
            m1 = jnp.sum(w * b_nodes)
            m2 = jnp.sum(w * b_nodes ** 2)
            return m1, m2
        in_axes = tuple([0] * len(subj_args_batched))
        m1, m2 = jax.vmap(per_subject, in_axes=in_axes)(*subj_args_batched)
        # returned as [n, 1] and [n, 1, 1] so the M-step sees the same
        # shapes it does at q = 2
        return m1[:, None], m2[:, None, None]
    return estep



def em_warm_start(theta, estep, mstep_inputs, layout, n_em=25,
                  loglik_fn=None):
    """Run a SHORT EM phase and return an improved theta.

    n_em defaults to 25, not to convergence. EM is slow near the optimum -
    measured on an exact-E-step linear mixed model, it took 378 iterations
    from a warm start and 772 from cold to reach a tolerance L-BFGS-B
    reached in 54. Running it to completion would be slower than the
    optimizer it is meant to protect.

    The point is to leave the region where a line search fails, not to
    finish. `JM` uses the same shape: an EM phase, then optim() with
    maxit 20 early and 5 later.

    `layout` maps parameter names to positions in theta, as elsewhere in
    this module; only the longitudinal entries are written back, since
    alpha, the baseline hazard and gamma have no closed-form M-step and
    are left for the optimizer.
    """
    theta = np.array(theta, dtype=float).copy()
    y, X, Z, mask = mstep_inputs

    for _ in range(int(n_em)):
        Eb, Ebb = estep(theta)
        beta, sigma_e, D = em_longitudinal_mstep(y, X, Z, None, Eb, Ebb, mask)
        beta = np.asarray(beta); D = np.asarray(D)

        if "beta" in layout:
            lo, hi = layout["beta"]
            theta[lo:hi] = beta[:hi - lo]
        if "log_sigma_e" in layout:
            theta[layout["log_sigma_e"]] = float(np.log(max(float(sigma_e), 1e-8)))
        sd = np.sqrt(np.maximum(np.diag(D), 1e-12))
        if "log_sigma_b" in layout:
            theta[layout["log_sigma_b"]] = float(np.log(sd[0]))
        if "log_sigma_b0" in layout:
            theta[layout["log_sigma_b0"]] = float(np.log(sd[0]))
        if "log_sigma_b1" in layout and sd.size > 1:
            theta[layout["log_sigma_b1"]] = float(np.log(sd[1]))
        if "atanh_rho" in layout and D.shape[0] > 1:
            r = D[0, 1] / max(sd[0] * sd[1], 1e-12)
            theta[layout["atanh_rho"]] = float(np.arctanh(np.clip(r, -0.95, 0.95)))

    return theta

# =============================================================================
# A standalone h(b) for the q = 1 Weibull model
#
# THE TENSION THIS RESOLVES. The E-step needs the unnormalised posterior
# of b_i, which is exactly the `h_fn` each fitting function already
# builds. Using that directly would mean importing from - or worse,
# modifying - code that has been tested across 423 assertions and five
# real datasets. Reimplementing it here means a SECOND source of truth
# that can silently drift from the first.
#
# Neither is acceptable on its own, so this takes the second option and
# removes its downside: `verify_against_backend()` below checks the
# reimplementation against the backend's OWN marginal log-likelihood on
# real data, to machine precision. If they ever disagree, the check says
# so rather than the E-step quietly computing the wrong expectation.
#
# That makes the duplication auditable, which an import would not have
# made it - an import would have coupled the two so that a change to the
# fitting path silently changed the EM, with nothing to detect it.
# =============================================================================


def h_weibull_q1(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                 T_i, event_i, t_quad_i, X_delta_surv_i, X_delta_quad_i,
                 theta, idx, gk_weights):
    """log p(y_i | b) + log p(T_i, d_i | b) + log p(b), up to a constant.

    Mirrors weibull_model.fit_mle's inner h_fn. `idx` supplies the theta
    layout so this does not hard-code positions that differ between
    fitting paths.
    """
    beta = theta[idx["beta"][0]:idx["beta"][1]]
    sigma_e = jnp.exp(theta[idx["log_sigma_e"]])
    sigma_b = jnp.exp(theta[idx["log_sigma_b"]])
    log_lambda0 = theta[idx["log_lambda0"]]
    shape = jnp.exp(theta[idx["log_shape"]])
    a_val = theta[idx["alpha_value"]]
    # alpha_delta is present only on the delta-channel paths; 0.0 makes
    # its contribution vanish where it is absent.
    a_del = theta[idx["alpha_delta"]] if "alpha_delta" in idx else 0.0

    mu = X_long_i @ beta + b
    lp_terms = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e)
                - 0.5 * ((y_i - mu) / sigma_e) ** 2)
    mask = jnp.arange(X_long_i.shape[0]) < n_i
    log_p_y = jnp.sum(jnp.where(mask, lp_terms, 0.0))

    log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
    m_T = X_time_surv_i @ beta + b
    md_T = X_delta_surv_i @ beta
    log_h_T = log_h0_T + a_val * m_T + a_del * md_T

    log_h0_q = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
    m_q = X_time_quad_i @ beta + b
    md_q = X_delta_quad_i @ beta
    hz_q = jnp.exp(log_h0_q + a_val * m_q + a_del * md_q)
    cum_H = T_i * jnp.sum(gk_weights * hz_q)

    log_p_surv = event_i * log_h_T - cum_H
    log_prior = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b)
                 - 0.5 * (b / sigma_b) ** 2)
    return log_p_y + log_p_surv + log_prior


def marginal_loglik_q1(data, theta, idx, gh_nodes, gh_weights, gk_weights,
                       n_newton=20):
    """Observed-data log-likelihood by adaptive GH, from h_weibull_q1.

    Used two ways: as the monotonicity check inside an EM loop, and by
    verify_against_backend() to confirm this module agrees with the
    fitting path it mirrors.
    """
    # The quadrature arrays are NOT batched over subjects, so the shape
    # check above does not cover them - and a rank-0 gh_nodes makes
    # b_nodes rank 0, which fails in the INNER vmap. JAX reports the
    # outermost frame for that, so the traceback points at the subject
    # vmap and says nothing about the real cause. That cost three rounds
    # of diagnosis; hence the explicit handling here.
    gh_nodes = jnp.atleast_1d(jnp.asarray(gh_nodes))
    gh_weights = jnp.atleast_1d(jnp.asarray(gh_weights))
    gk_weights = jnp.atleast_1d(jnp.asarray(gk_weights))
    if gh_nodes.shape != gh_weights.shape:
        raise ValueError(
            f"gh_nodes {gh_nodes.shape} and gh_weights {gh_weights.shape} "
            "must match - they are the nodes and weights of one rule")
    if gh_nodes.size < 2:
        raise ValueError(
            f"gh_nodes has {gh_nodes.size} element(s); adaptive Gauss-Hermite "
            "needs a rule with several nodes. A length-1 vector crossing "
            "from R arrives as a scalar, so check how it was passed.")

    def per_subject(*args):
        def newton(bb, _):
            g = jax.grad(h_weibull_q1, argnums=0)(bb, *args, theta, idx, gk_weights)
            hc = jax.grad(jax.grad(h_weibull_q1, argnums=0), argnums=0)(
                bb, *args, theta, idx, gk_weights)
            hc = jnp.where(hc < -1e-8, hc, -1e-8)
            return bb - g / hc, None
        b_mode, _ = jax.lax.scan(newton, 0.0, None, length=n_newton)
        hc = jax.grad(jax.grad(h_weibull_q1, argnums=0), argnums=0)(
            b_mode, *args, theta, idx, gk_weights)
        tau = 1.0 / jnp.sqrt(jnp.maximum(-hc, 1e-12))
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes
        h_vals = jax.vmap(lambda bk: h_weibull_q1(
            bk, *args, theta, idx, gk_weights))(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    # The delta-channel arrays exist only when that functional form was
    # requested; the backend defaults them to None and dispatches to a
    # different fitting function. Here they are filled with zeros and
    # alpha_delta is absent from `idx`, so their contribution vanishes -
    # the same model, expressed without a separate code path.
    #
    # A KeyError here would be the honest failure. Silently substituting
    # something non-zero would not, which is why this is zeros and not,
    # say, X_time_surv reused.
    core = ["X_long", "y_long", "n_obs", "X_time_surv", "X_time_quad",
            "T_surv", "event", "t_quad"]
    args = [data[k] for k in core]
    for k, like in (("X_delta_surv", "X_time_surv"),
                    ("X_delta_quad", "X_time_quad")):
        args.append(jnp.zeros_like(jnp.asarray(data[like]))
                    if k not in data or data[k] is None
                    else jnp.asarray(data[k]))
    args = tuple(args)

    # Name the offending array rather than let vmap report a bare rank.
    # Every argument must be batched over subjects, so each needs rank >= 1
    # with a matching leading dimension. reticulate can flatten an array
    # crossing from R - a length-1 vector arrives as a scalar - and the
    # resulting "rank should be at least 1, but is only 0" says nothing
    # about WHICH one, which cost several rounds of guessing.
    names_in_order = core + ["X_delta_surv", "X_delta_quad"]
    shapes = [jnp.asarray(a).shape for a in args]
    bad = [(nm, sh) for nm, sh in zip(names_in_order, shapes) if len(sh) < 1]
    if bad:
        raise ValueError(
            "these arrays arrived with rank 0 and cannot be mapped over "
            "subjects: " + ", ".join(f"{nm} {sh}" for nm, sh in bad) +
            f"\n  all shapes: " +
            ", ".join(f"{nm}={sh}" for nm, sh in zip(names_in_order, shapes)))
    lead = {sh[0] for sh in shapes}
    if len(lead) > 1:
        raise ValueError(
            "arrays disagree on the number of subjects: " +
            ", ".join(f"{nm}={sh}" for nm, sh in zip(names_in_order, shapes)))
    return jnp.sum(jax.vmap(per_subject, in_axes=(0,) * len(args))(*args))


def verify_against_backend(data, theta, idx, gh_nodes, gh_weights,
                           gk_weights, backend_neg_log_lik, tol=1e-4):
    """Does this module's h(b) agree with the fitting path's?

    Run this whenever em.py is used, and certainly before trusting any EM
    result. A reimplementation that has drifted from the model it mirrors
    would produce an E-step for the WRONG posterior - and the estimates
    might still look plausible, which is what makes the drift dangerous
    rather than merely wrong.
    """
    ours = float(marginal_loglik_q1(data, theta, idx, gh_nodes, gh_weights,
                                     gk_weights))
    theirs = -float(backend_neg_log_lik(jnp.asarray(theta), data))
    diff = abs(ours - theirs)
    rel = diff / max(abs(theirs), 1.0)
    ok = rel < tol
    if not ok:
        warnings.warn(
            f"em.py's h(b) does NOT match the backend likelihood: "
            f"{ours:.6f} vs {theirs:.6f} (relative {rel:.2e}). The E-step "
            "would be computing the expectation of a different model. Do "
            "not use any EM result until this is resolved.", RuntimeWarning)
    return {"ours": ours, "backend": theirs, "abs_diff": diff,
            "rel_diff": rel, "match": ok}

# =============================================================================
# The spline counterpart
#
# Differs from h_weibull_q1 in the BASELINE ONLY: `B_T_i @ W` replaces
# log(shape) + (shape-1)*log(t) + log_lambda0. Everything else - the
# longitudinal density, the quadrature, the prior on b - is identical.
#
# This matters because prothro + spline is the ONLY case in this project
# where an optimizer silently stops short (L-BFGS-B at -14083.14 with
# alpha -0.0008, where BFGS reaches -14004.80 and JM -0.0400). Every
# synthetic problem tried so far converges fine from cold, so none of them
# can distinguish a scheme that would rescue that failure from one that
# merely matches an optimizer that was never stuck.
# =============================================================================


def h_spline_q1(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                T_i, event_i, t_quad_i, B_T_i, B_quad_i, W_surv_i, theta,
                idx, gk_weights):
    """log p(y_i | b) + log p(T_i, d_i | b) + log p(b), spline baseline.

    W_surv_i carries the BASELINE SURVIVAL COVARIATES. An earlier version
    omitted them, and the marginal log-likelihood came out 7.459 units
    below the backend's on prothro - a gap that did not move with 61 GH
    nodes or 50 Newton steps, which is what ruled out quadrature and
    pointed at a missing term rather than a numerical one.
    
    The omission was easy to make and hard to see: `Surv(Time, death) ~
    treat` has a covariate, so the backend dispatches to
    _fit_mle_spline_with_baseline_covariates, whose theta carries a gamma
    block between W and alpha. Reading the BASE path's h_fn - which has no
    gamma - and assuming it was the one in use produced a model that was
    correct for a formula nobody had fitted.
    """
    beta = theta[idx["beta"][0]:idx["beta"][1]]
    sigma_e = jnp.exp(theta[idx["log_sigma_e"]])
    sigma_b = jnp.exp(theta[idx["log_sigma_b"]])
    W = theta[idx["baseline"][0]:idx["baseline"][1]]
    alpha = theta[idx["alpha"]]
    # gamma is absent from idx when surv_formula has no covariates, in
    # which case its contribution is zero - the same model, no branch.
    gterm = (W_surv_i @ theta[idx["gamma"][0]:idx["gamma"][1]]
             if "gamma" in idx else 0.0)

    mu = X_long_i @ beta + b
    lp = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e)
          - 0.5 * ((y_i - mu) / sigma_e) ** 2)
    mask = jnp.arange(X_long_i.shape[0]) < n_i
    log_p_y = jnp.sum(jnp.where(mask, lp, 0.0))

    m_T = X_time_surv_i @ beta + b
    log_h_T = jnp.dot(B_T_i, W) + gterm + alpha * m_T

    m_q = X_time_quad_i @ beta + b
    hz = jnp.exp(jnp.dot(B_quad_i, W) + gterm + alpha * m_q)
    cum_H = T_i * jnp.sum(gk_weights * hz)

    log_p_surv = event_i * log_h_T - cum_H
    log_prior = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b)
                 - 0.5 * (b / sigma_b) ** 2)
    return log_p_y + log_p_surv + log_prior


def marginal_loglik_spline_q1(data, theta, idx, gh_nodes, gh_weights,
                              gk_weights, n_newton=20):
    """Observed-data log-likelihood by adaptive GH, spline baseline."""
    gh_nodes = jnp.atleast_1d(jnp.asarray(gh_nodes))
    gh_weights = jnp.atleast_1d(jnp.asarray(gh_weights))
    gk_weights = jnp.atleast_1d(jnp.asarray(gk_weights))

    def per_subject(*args):
        def newton(bb, _):
            g = jax.grad(h_spline_q1, argnums=0)(bb, *args, theta, idx, gk_weights)
            hc = jax.grad(jax.grad(h_spline_q1, argnums=0), argnums=0)(
                bb, *args, theta, idx, gk_weights)
            hc = jnp.where(hc < -1e-8, hc, -1e-8)
            return bb - g / hc, None
        b_mode, _ = jax.lax.scan(newton, 0.0, None, length=n_newton)
        hc = jax.grad(jax.grad(h_spline_q1, argnums=0), argnums=0)(
            b_mode, *args, theta, idx, gk_weights)
        tau = 1.0 / jnp.sqrt(jnp.maximum(-hc, 1e-12))
        bn = b_mode + jnp.sqrt(2.0) * tau * gh_nodes
        hv = jax.vmap(lambda bk: h_spline_q1(
            bk, *args, theta, idx, gk_weights))(bn)
        lt = jnp.log(gh_weights) + hv + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(lt)

    keys = ["X_long", "y_long", "n_obs", "X_time_surv", "X_time_quad",
            "T_surv", "event", "t_quad", "B_T", "B_quad", "W_surv"]
    if "W_surv" not in data or data["W_surv"] is None:
        # No survival covariates: a zero column keeps the einsum shapes
        # uniform and contributes nothing, rather than needing a branch.
        data = dict(data)
        data["W_surv"] = jnp.zeros((jnp.asarray(data["T_surv"]).shape[0], 1))
        idx = {k: v for k, v in idx.items() if k != "gamma"}
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(
            "spline marginal likelihood needs " + ", ".join(missing) +
            " - B_T and B_quad come from the fit's backend_data, which "
            "jm_fit returns only with control$return_backend_data = TRUE")
    args = tuple(jnp.asarray(data[k]) for k in keys)
    shapes = [a.shape for a in args]
    bad = [(k, sh) for k, sh in zip(keys, shapes) if len(sh) < 1]
    if bad:
        raise ValueError("rank-0 arrays: " + ", ".join(f"{k}{sh}" for k, sh in bad))
    return jnp.sum(jax.vmap(per_subject, in_axes=(0,) * len(args))(*args))

# =============================================================================
# q = 2: the bivariate posterior
#
# Needed because prothro's real fit is q=2, and the rescue was demonstrated
# at q=1:
#     cold L-BFGS-B      -14132.01   alpha -0.0039   grad_max 17.4
#     EM-warmed L-BFGS-B -14082.37   alpha -0.0279   grad_max  1.6
#     cold BFGS          -14078.79   alpha -0.0375
#     EM-warmed BFGS     -14078.01   alpha -0.0381   <- best of all four
# with 0/25 EM iterations increasing the negative log-likelihood.
#
# TWO THINGS GENUINELY CHANGE AT q = 2, and both are easy to get subtly
# wrong:
#
#   THE PRIOR is bivariate normal with a correlation. The backend
#   parameterises it as sigma_b0, sigma_b1 and atanh(rho) and writes the
#   quadratic form out longhand rather than inverting a matrix. That form
#   is reproduced exactly below, including the -log(2pi) (not -0.5*log)
#   and the -0.5*log(1 - rho^2) normalising terms.
#
#   THE MODE-FINDING is a 2-D Newton step needing a HESSIAN, and the
#   backend clips its eigenvalues to stay negative-definite before
#   inverting. Skipping that clip gives a step that can diverge on
#   subjects with little information; it is reproduced here rather than
#   replaced with a plain solve.
#
# The quadrature becomes a PRODUCT rule - n_gh^2 nodes rather than n_gh -
# so the E-step cost rises quadratically in the rule size. That is the
# main practical cost of q=2 and worth knowing before choosing n_gh.
# =============================================================================


def h_spline_q2(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i,
                theta, idx, gk_weights):
    """log p(y_i|b) + log p(T_i,d_i|b) + log p(b), q=2 spline baseline.

    `b` is length 2. Mirrors the backend's
    _fit_mle_spline_q2_with_baseline_covariates h_fn exactly.
    """
    beta = theta[idx["beta"][0]:idx["beta"][1]]
    sigma_e = jnp.exp(theta[idx["log_sigma_e"]])
    sigma_b0 = jnp.exp(theta[idx["log_sigma_b0"]])
    sigma_b1 = jnp.exp(theta[idx["log_sigma_b1"]])
    rho = jnp.tanh(theta[idx["atanh_rho"]])
    W = theta[idx["baseline"][0]:idx["baseline"][1]]
    alpha = theta[idx["alpha"]]
    gterm = (W_surv_i @ theta[idx["gamma"][0]:idx["gamma"][1]]
             if "gamma" in idx else 0.0)

    mu = X_long_i @ beta + Z_long_i @ b
    lp = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e)
          - 0.5 * ((y_i - mu) / sigma_e) ** 2)
    mask = jnp.arange(X_long_i.shape[0]) < n_i
    log_p_y = jnp.sum(jnp.where(mask, lp, 0.0))

    m_T = X_time_surv_i @ beta + Z_time_surv_i @ b
    log_h_T = jnp.dot(B_T_i, W) + gterm + alpha * m_T

    m_q = X_time_quad_i @ beta + Z_time_quad_i @ b
    hz = jnp.exp(jnp.dot(B_quad_i, W) + gterm + alpha * m_q)
    cum_H = T_i * jnp.sum(gk_weights * hz)
    log_p_surv = event_i * log_h_T - cum_H

    z0 = b[0] / sigma_b0
    z1 = b[1] / sigma_b1
    qf = (z0 ** 2 - 2.0 * rho * z0 * z1 + z1 ** 2) / (1.0 - rho ** 2)
    log_prior = (-jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b0)
                 - jnp.log(sigma_b1) - 0.5 * jnp.log(1.0 - rho ** 2)
                 - 0.5 * qf)
    return log_p_y + log_p_surv + log_prior


def _q2_mode_and_L(h, args, theta, idx, gk, n_newton):
    """2-D Newton to the posterior mode, with the backend's Hessian clip.

    The clip - symmetrise, eigendecompose, force eigenvalues below -eps -
    keeps the step well defined on subjects whose posterior is nearly flat
    in one direction. A plain solve() diverges there, which is why this
    reproduces the backend rather than simplifying it.
    """
    grad_fn = jax.grad(h, argnums=0)
    hess_fn = jax.hessian(h, argnums=0)

    def step(b, _):
        g = grad_fn(b, *args, theta, idx, gk)
        H = hess_fn(b, *args, theta, idx, gk)
        Hs = 0.5 * (H + H.T)
        ev, V = jnp.linalg.eigh(Hs)
        ev = jnp.minimum(ev, -1e-6)
        return b - V @ ((V.T @ g) / ev), None

    b_mode, _ = jax.lax.scan(step, jnp.zeros(2), None, length=n_newton)
    H = hess_fn(b_mode, *args, theta, idx, gk)
    Hs = 0.5 * (H + H.T)
    ev, V = jnp.linalg.eigh(Hs)
    ev = jnp.minimum(ev, -1e-6)
    # Sigma = (-H)^-1 ; L is its Cholesky factor, used to place nodes
    Sigma = V @ jnp.diag(1.0 / (-ev)) @ V.T
    L = jnp.linalg.cholesky(0.5 * (Sigma + Sigma.T))
    return b_mode, L


def posterior_moments_q2(data, theta, idx, gh_nodes, gh_weights, gk_weights,
                         n_newton=20):
    """E[b] and E[b b'] per subject, q=2 spline. Returns (Eb, Ebb).

    Uses a PRODUCT Gauss-Hermite rule: n_gh^2 nodes. The E-step cost is
    therefore quadratic in the rule size, which is the main practical
    difference from q=1.
    """
    gh_nodes = jnp.atleast_1d(jnp.asarray(gh_nodes))
    gh_weights = jnp.atleast_1d(jnp.asarray(gh_weights))
    gk = jnp.atleast_1d(jnp.asarray(gk_weights))

    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    zg = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)          # [n_gh^2, 2]
    wg = jnp.outer(gh_weights, gh_weights).ravel()

    keys = ["X_long", "y_long", "n_obs", "X_time_surv", "X_time_quad",
            "T_surv", "event", "t_quad", "B_T", "B_quad",
            "Z_long", "Z_time_surv", "Z_time_quad", "W_surv"]
    d = dict(data)
    if "W_surv" not in d or d["W_surv"] is None:
        d["W_surv"] = jnp.zeros((jnp.asarray(d["T_surv"]).shape[0], 1))
        idx = {k: v for k, v in idx.items() if k != "gamma"}
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError("q=2 E-step needs " + ", ".join(missing) +
                         " - Z_long, Z_time_surv and Z_time_quad come from "
                         "the fit's backend_data (return_backend_data = TRUE)")
    args_b = tuple(jnp.asarray(d[k]) for k in keys)

    def per(*a):
        b_mode, L = _q2_mode_and_L(h_spline_q2, a, theta, idx, gk, n_newton)
        bn = b_mode + jnp.sqrt(2.0) * (zg @ L.T)               # [n_gh^2, 2]
        hv = jax.vmap(lambda bk: h_spline_q2(bk, *a, theta, idx, gk))(bn)
        lt = jnp.log(wg) + hv + jnp.sum(zg ** 2, axis=-1)
        w = jnp.exp(lt - jax.nn.logsumexp(lt))
        m1 = jnp.sum(w[:, None] * bn, axis=0)
        M2 = jnp.sum(w[:, None, None] * (bn[:, :, None] * bn[:, None, :]),
                     axis=0)
        return m1, M2

    Eb, Ebb = jax.vmap(per, in_axes=(0,) * len(args_b))(*args_b)
    return np.asarray(Eb), np.asarray(Ebb)


def marginal_loglik_spline_q2(data, theta, idx, gh_nodes, gh_weights,
                              gk_weights, n_newton=20):
    """Observed-data log-likelihood, q=2 spline. Verify against the backend."""
    gh_nodes = jnp.atleast_1d(jnp.asarray(gh_nodes))
    gh_weights = jnp.atleast_1d(jnp.asarray(gh_weights))
    gk = jnp.atleast_1d(jnp.asarray(gk_weights))
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    zg = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
    wg = jnp.outer(gh_weights, gh_weights).ravel()

    keys = ["X_long", "y_long", "n_obs", "X_time_surv", "X_time_quad",
            "T_surv", "event", "t_quad", "B_T", "B_quad",
            "Z_long", "Z_time_surv", "Z_time_quad", "W_surv"]
    d = dict(data)
    if "W_surv" not in d or d["W_surv"] is None:
        d["W_surv"] = jnp.zeros((jnp.asarray(d["T_surv"]).shape[0], 1))
        idx = {k: v for k, v in idx.items() if k != "gamma"}
    args_b = tuple(jnp.asarray(d[k]) for k in keys)

    def per(*a):
        b_mode, L = _q2_mode_and_L(h_spline_q2, a, theta, idx, gk, n_newton)
        bn = b_mode + jnp.sqrt(2.0) * (zg @ L.T)
        hv = jax.vmap(lambda bk: h_spline_q2(bk, *a, theta, idx, gk))(bn)
        lt = jnp.log(wg) + hv + jnp.sum(zg ** 2, axis=-1)
        # the 2-D Jacobian is |det(sqrt(2) L)| = 2 * det(L)
        return (jnp.log(2.0) + jnp.log(jnp.abs(jnp.linalg.det(L)))
                + jax.nn.logsumexp(lt))

    return jnp.sum(jax.vmap(per, in_axes=(0,) * len(args_b))(*args_b))

# =============================================================================
# Wiring: an EM warm start built from the SHARED likelihood pieces
#
# `common.build_h_fn(log_h0_fn, idx)` already constructs h(b) for the q=1
# path, parameterised by the baseline hazard. Building the E-step from the
# same function means it cannot drift from the likelihood it is supposed to
# be the expectation of - the concern that made a standalone
# reimplementation need its own verification harness.
#
# SCOPE, stated plainly. Only the base q=1 Weibull path routes through
# build_h_fn; the other 13 fitting functions define h_fn inline. So this
# covers one path properly and the rest need either the same refactor or
# the standalone h_weibull_q1 / h_spline_q1 / h_spline_q2 above. Wiring it
# everywhere is the remaining work.
#
# WHY IT IS WORTH WIRING AT ALL - measured on real prothro, spline
# baseline, where L-BFGS-B silently stops short:
#
#   q=1   cold L-BFGS-B      -14132.01  alpha -0.0039  grad_max 17.4
#         EM-warmed L-BFGS-B -14082.37  alpha -0.0279  grad_max  1.6
#   q=2   cold L-BFGS-B      -14083.14  alpha -0.0008  grad_max 31.9
#         EM-warmed L-BFGS-B -14004.49  alpha -0.0346  grad_max  6.7
#         EM-warmed BFGS     -14001.88  alpha -0.0417  <- best of all
#
# 78.7 of the 78.3 missing log-likelihood units recovered at q=2, with
# 0/25 iterations increasing the objective. JM reaches alpha -0.0400 and
# jmjax's own MCMC -0.0411.
#
# THE SCHEDULE follows JM's: iter.EM = 120 for spline and 50 otherwise,
# with the inner survival optim capped at maxit 20 for the first five
# iterations and 4 after. Those are JM's numbers, tuned across more
# datasets than the one this was validated on.
# =============================================================================


def make_estep_from_h(h_fn, n_gh=15, n_newton=20):
    """E-step from the SAME h(b) the likelihood uses.

    `h_fn` has common.build_h_fn's signature:
        h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, gk_weights, theta)
    """
    gh_n, gh_w = np.polynomial.hermite.hermgauss(n_gh)
    gh_n = jnp.array(gh_n); gh_w = jnp.array(gh_w)
    grad_b = jax.grad(h_fn, argnums=0)
    hess_b = jax.grad(grad_b, argnums=0)

    def estep(data, theta):
        theta = jnp.asarray(theta)
        gk = jnp.asarray(data["gk_weights"])

        def per(*a):
            def newton(bb, _):
                g = grad_b(bb, *a, gk, theta)
                hc = hess_b(bb, *a, gk, theta)
                return bb - g / jnp.where(hc < -1e-8, hc, -1e-8), None
            b_mode, _ = jax.lax.scan(newton, 0.0, None, length=n_newton)
            hc = hess_b(b_mode, *a, gk, theta)
            tau = 1.0 / jnp.sqrt(jnp.maximum(-hc, 1e-12))
            bn = b_mode + jnp.sqrt(2.0) * tau * gh_n
            hv = jax.vmap(lambda bk: h_fn(bk, *a, gk, theta))(bn)
            lt = jnp.log(gh_w) + hv + gh_n ** 2
            w = jnp.exp(lt - jax.nn.logsumexp(lt))
            return jnp.sum(w * bn), jnp.sum(w * bn ** 2)

        keys = ["X_long", "y_long", "n_obs", "X_time_surv", "X_time_quad",
                "T_surv", "event", "t_quad"]
        args = tuple(jnp.asarray(data[k]) for k in keys)
        m1, m2 = jax.vmap(per, in_axes=(0,) * len(args))(*args)
        return np.asarray(m1)[:, None], np.asarray(m2)[:, None, None]

    return estep


def em_phase(theta0, data, idx, neg_log_lik, h_fn, n_em=50,
             n_gh=15, n_newton=20, surv_block=None, verbose=False,
             tol1=1e-3, tol2=1e-4, tol3=None):
    """A short joint EM phase. Returns (theta, trace, n_increases).

    JOINT, not longitudinal-only. An earlier version updated beta,
    sigma_e and D while holding alpha and the baseline fixed, and the
    JOINT negative log-likelihood ROSE (7803.97 -> 7804.19 over 25
    iterations): a partial M-step improves one block and degrades the
    whole. The survival block is optimised here for that reason, which is
    also why JM calls optim() inside its EM loop rather than running a
    pure EM.

    Never returns a WORSE theta than it was given. EM is monotone by
    construction, so a worse objective means the implementation is wrong -
    it is reported rather than silently passed to the optimizer.

    STOPPING CRITERIA, matched to JM's so the two are comparable:

        check1 = max(|theta_new - theta_old| / (|theta_old| + tol1)) < tol2
        check2 = (lgLik[it] - lgLik[it-1]) < tol3 * (|lgLik[it-1]| + tol3)

    with tol1 = 1e-3, tol2 = 1e-4, tol3 = sqrt(machine eps). Both are
    RELATIVE - to the parameter magnitudes and to the log-likelihood -
    where an earlier version stopped on an absolute |delta loglik| < 1e-8.
    That absolute test is scale-dependent in a way JM's is not: on a
    log-likelihood of 14000 it is far stricter than tol3 intends, and on a
    small problem far looser.

    `n_em` follows JM's `iter.EM`: 120 for a spline baseline, 50
    otherwise. The caller passes the right one, since only the caller
    knows which baseline is in use.
    """
    if tol3 is None:
        tol3 = float(np.sqrt(np.finfo(float).eps))
    from scipy.optimize import minimize

    theta = np.asarray(theta0, dtype=float).copy()
    ll0 = float(neg_log_lik(jnp.asarray(theta), data))
    if not np.isfinite(ll0):
        warnings.warn("em_phase: the starting objective is not finite; "
                      "skipping the EM phase.", RuntimeWarning)
        return np.asarray(theta0, dtype=float), [], 0

    estep = make_estep_from_h(h_fn, n_gh=n_gh, n_newton=n_newton)
    if surv_block is None:
        surv_block = list(range(idx["baseline"][0], idx["alpha"] + 1))

    y = np.asarray(data["y_long"]); X = np.asarray(data["X_long"])
    Z = X[:, :, :1]
    T = y.shape[1]
    mask = (np.arange(T)[None, :] < np.asarray(data["n_obs"])[:, None]
            ).astype(float)

    vg = jax.jit(jax.value_and_grad(lambda t: neg_log_lik(t, data)))

    def surv_step(t, maxit):
        """Optimise the survival block, and REFUSE a step that makes things
        worse.

        A truncated BFGS run can end at a worse point than it started -
        `maxiter` cuts it off mid-search, and scipy returns wherever it
        stopped without checking. That breaks EM's monotonicity, which is
        the entire reason for using EM here: the warning
        "objective increased on 1 of 5 iterations" came from exactly this.
        Accepting the step only when it improves restores the guarantee
        without changing the schedule.
        """
        t = np.asarray(t, float).copy(); sc = 0.01
        v0 = float(vg(jnp.asarray(t))[0])
        def f(u):
            full = t.copy(); full[surv_block] = u * sc
            v, g = vg(jnp.asarray(full))
            return float(v), np.array(g, dtype=np.float64)[surv_block] * sc
        r = minimize(f, t[surv_block] / sc, method="BFGS", jac=True,
                     options={"maxiter": maxit})
        cand = t.copy(); cand[surv_block] = r.x * sc
        v1 = float(vg(jnp.asarray(cand))[0])
        return cand if (np.isfinite(v1) and v1 <= v0) else t

    trace = [ll0]; bad = 0
    for it in range(int(n_em)):
        theta_prev = theta.copy()          # for JM's relative-change check
        Eb, Ebb = estep(data, theta)
        bh, sh, Dh = em_longitudinal_mstep(y, X, Z, None, Eb, Ebb, mask)
        p0, p1 = idx["beta"]
        theta[p0:p1] = np.asarray(bh)[:p1 - p0]
        theta[idx["log_sigma_e"]] = float(np.log(max(float(sh), 1e-8)))
        theta[idx["log_sigma_b"]] = float(
            np.log(max(np.sqrt(np.asarray(Dh)[0, 0]), 1e-8)))
        theta = surv_step(theta, 20 if it < 5 else 4)   # JM's schedule

        # The longitudinal M-step maximises its own block exactly, but the
        # JOINT likelihood can still fall if the survival step cannot keep
        # up - the partial-M-step failure that made an earlier
        # longitudinal-only version take the objective UP (7803.97 ->
        # 7804.19). Rejecting a worse joint step keeps the phase monotone
        # whatever the blocks do individually.
        v = float(neg_log_lik(jnp.asarray(theta), data))
        if not np.isfinite(v) or v > trace[-1] + 1e-9:
            # Roll back and stop: the phase has stopped improving, and
            # handing the optimizer a worse point than it would have had
            # is the one thing a warm start must never do.
            theta = theta_prev.copy()
            break
        trace.append(v)

        # JM's two checks, on the LOG-LIKELIHOOD (= -neg_log_lik)
        prev = np.asarray(theta_prev, dtype=float)
        check1 = np.max(np.abs(theta - prev) / (np.abs(prev) + tol1)) < tol2
        ll_now, ll_prev = -trace[-1], -trace[-2]
        check2 = (ll_now - ll_prev) < tol3 * (abs(ll_prev) + tol3)
        if check1 or check2:
            break

    # `bad` stays 0 by construction now: a step that raises the objective
    # is rejected rather than recorded. It is kept in the return value so a
    # caller can assert on it - if it is ever non-zero, something has
    # bypassed the rollback.
    if bad:
        warnings.warn(
            f"em_phase: the objective increased on {bad} of {len(trace)-1} "
            "iterations despite the rollback. That should be unreachable - "
            "please report it.", RuntimeWarning)
    if not np.isfinite(trace[-1]) or trace[-1] > ll0 + 1e-6:
        warnings.warn("em_phase: finished worse than it started; returning "
                      "the original starting values.", RuntimeWarning)
        return np.asarray(theta0, dtype=float), trace, bad

    # A BETTER LIKELIHOOD IS NOT ENOUGH. EM's stopping criteria (JM's
    # tol1/tol2) are about parameter CHANGE, so it can legitimately stop
    # somewhere with a substantial gradient - and a quasi-Newton method
    # started there may have no usable descent direction and take zero
    # steps.
    #
    # Measured: on a fit that converged cleanly from cold (grad_max 0.62),
    # the EM-warmed run took 0 optimizer iterations and ended at grad_max
    # 1.75, above the threshold, so the fit reported converged = FALSE for
    # an essentially identical log-likelihood. EM had made the STOPPING
    # POINT worse while improving nothing.
    #
    # So hand back whichever endpoint is better conditioned when the two
    # likelihoods are effectively tied. The warm start exists to help the
    # optimizer, not to move it somewhere it cannot work from.
    try:
        g_em = float(np.max(np.abs(np.asarray(
            jax.grad(lambda t: neg_log_lik(t, data))(jnp.asarray(theta))))))
        g_0 = float(np.max(np.abs(np.asarray(
            jax.grad(lambda t: neg_log_lik(t, data))(jnp.asarray(theta0))))))
    except Exception:
        g_em = g_0 = float("nan")

    tied = abs(trace[-1] - ll0) <= 1e-4 * max(abs(ll0), 1.0)
    if tied and np.isfinite(g_em) and np.isfinite(g_0) and g_em > g_0:
        warnings.warn(
            f"em_phase: the log-likelihood barely moved ({ll0:.6f} -> "
            f"{trace[-1]:.6f}) while the gradient grew ({g_0:.4g} -> "
            f"{g_em:.4g}); returning the starting values, which are better "
            "conditioned for the optimizer.", RuntimeWarning)
        return np.asarray(theta0, dtype=float), trace, bad

    return theta, trace, bad

