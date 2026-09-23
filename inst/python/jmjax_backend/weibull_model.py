"""Weibull baseline hazard, adaptive-GH MLE. See common.py for the shared
machinery this plugs into."""
import jax
import warnings
import numpy as np
import jax.numpy as jnp
from scipy.stats import norm

from . import common


def _log_h0_weibull(t, baseline_params):
    log_lambda0, log_shape = baseline_params[0], baseline_params[1]
    shape = jnp.exp(log_shape)
    return jnp.log(shape) + (shape - 1.0) * jnp.log(t) + log_lambda0


def fit_mle(X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights,
            random_effects="intercept",
            Z_long=None, Z_time_surv=None, Z_time_quad=None,
            X_delta_surv=None, X_delta_quad=None, Z_delta_surv=None, Z_delta_quad=None,
            X_area_surv=None, X_area_quad=None, Z_area_surv=None, Z_area_quad=None,
            X_area_avg_surv=None, X_area_avg_quad=None, Z_area_avg_surv=None, Z_area_avg_quad=None,
            W_surv=None,
            n_gh_nodes=15, n_newton_steps=8, init_theta=None, control=None):
    control = control or {}

    if random_effects != "intercept":
        q2_kwargs = dict(
            # int() coercion is REQUIRED, not defensive - see the matching
            # comment in spline_model.py's fit_mle(): R passes numeric
            # literals as doubles via reticulate (7 -> 7.0), which
            # numpy.polynomial.hermite.hermgauss() rejects outright.
            n_gh_nodes=int(control.get("n_gh_nodes_per_dim", 7)),
            n_newton_steps=int(control.get("n_newton_steps_q2", 20)),
            init_theta=init_theta, control=control,
        )
        # q=2 + an extra channel: route to the GENERIC extended function
        # (works for delta/area/area_avg alike - see module docstring for
        # why these all reduce to the same "Z_channel(t) @ b" structure).
        if X_delta_surv is not None:
            return _fit_mle_q2_extra_channel(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights,
                Z_long, Z_time_surv, Z_time_quad,
                X_delta_surv, X_delta_quad, Z_delta_surv, Z_delta_quad,
                channel_name="delta", **q2_kwargs,
            )
        if X_area_surv is not None:
            return _fit_mle_q2_extra_channel(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights,
                Z_long, Z_time_surv, Z_time_quad,
                X_area_surv, X_area_quad, Z_area_surv, Z_area_quad,
                channel_name="area", **q2_kwargs,
            )
        if X_area_avg_surv is not None:
            return _fit_mle_q2_extra_channel(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights,
                Z_long, Z_time_surv, Z_time_quad,
                X_area_avg_surv, X_area_avg_quad, Z_area_avg_surv, Z_area_avg_quad,
                channel_name="area_avg", **q2_kwargs,
            )
        if W_surv is not None:
            # gamma'W_i at q=2 - same time-constant term as the q=1
            # baseline-covariates function, built on the q=2 scaffolding.
            # Longitudinal-side baseline covariates need NO separate
            # dispatch here - they're already handled for free by
            # _fit_mle_intercept_slope's existing p-generalization (see
            # that function's W_surv=None fallback below).
            return _fit_mle_q2_with_baseline_covariates(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights,
                Z_long, Z_time_surv, Z_time_quad, W_surv,
                **q2_kwargs,
            )
        # Plain q=2, value-only - the already-validated original path.
        # Also handles longitudinal-side baseline covariates transparently
        # (p generalizes automatically to X_long's actual column count).
        return _fit_mle_intercept_slope(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights,
            Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
            **q2_kwargs,
        )

    if W_surv is not None:
        # Routed to a SEPARATE function, same pattern as delta/area/
        # area_avg - keeps the already-validated plain q=1 path
        # completely unaffected. R-side guards ensure this is only
        # reached for q=1, plain "value" association (no functional_forms
        # channel combined with baseline covariates yet).
        return _fit_mle_with_baseline_covariates(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, W_surv,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    if X_delta_surv is not None:
        # Routed to a SEPARATE function (not a modification of the plain
        # q=1 path below) so the already-validated single-value-channel
        # code path is completely unaffected by this new capability.
        return _fit_mle_with_delta(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, X_delta_surv, X_delta_quad,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    if X_area_surv is not None:
        # Also a separate function, mirroring the delta case - kept
        # independent rather than generalized into a combinable N-channel
        # system for the same reason (each new channel validated on its
        # own first; R-side already rejects combining these together).
        return _fit_mle_with_area(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, X_area_surv, X_area_quad,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    if X_area_avg_surv is not None:
        # area_avg(t) = area(t)/t - JMbayes2's confirmed actual convention
        # for its area() functional form (confirmed by directly inspecting
        # a fitted JMbayes2 object's model_data$X_h during development -
        # NOT the raw integral, despite the name). Its random-intercept
        # contribution reduces to plain b (the time-average of a constant
        # is that constant), making this structurally closer to the
        # "value" channel's own "+b" term than to area's "+b*t" term.
        return _fit_mle_with_area_avg(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, X_area_avg_surv, X_area_avg_quad,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    p = X_long.shape[-1]
    n_baseline = 2  # log_lambda0, log_shape
    idx, n_theta = common.make_theta_layout(p, n_baseline)

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
    )

    neg_log_lik = common.build_neg_log_lik(
        _log_h0_weibull, idx, n_gh=n_gh_nodes, n_newton_steps=n_newton_steps
    )

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[idx["beta"][0]] = 2.0        # crude default intercept guess
        theta0[idx["log_sigma_e"]] = np.log(0.5)
        theta0[idx["log_sigma_b"]] = np.log(0.5)
        theta0[idx["baseline"][0]] = -2.0    # log_lambda0
        theta0[idx["baseline"][0] + 1] = np.log(1.2)  # log_shape
        theta0[idx["alpha"]] = 0.5
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": idx["alpha"], "beta": idx["beta"], "log_sigma_b": idx["log_sigma_b"], "log_sigma_e": idx["log_sigma_e"], "log_lambda0": idx["baseline"][0], "log_shape": idx["baseline"][0] + 1}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)

    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result(fit, idx, p)


# ==============================================================================
# value + delta association, q=1 (random intercept only).
#
# For a pure random INTERCEPT, delta(t) = m_i(t) - m_i(0) has NO random-
# effect contribution: the intercept doesn't depend on time, so it cancels
# exactly in the subtraction. This is what makes q=1 the natural starting
# point for functional forms beyond "value" - the delta channel only needs
# the fixed-effects design (X_delta_surv/X_delta_quad, precomputed on the R
# side as X(t) - X(0)), no new random-effects machinery at all. The
# adaptive-GH/Newton scaffolding below is otherwise IDENTICAL to the plain
# q=1 path (common.py's build_neg_log_lik/build_h_fn), duplicated here
# rather than generalizing that shared code, to keep the validated
# single-channel path completely unaffected by this new capability.
# ==============================================================================
def _fit_mle_with_delta(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                         T_surv, event, t_quad, gk_weights,
                         X_delta_surv, X_delta_quad,
                         n_gh_nodes=15, n_newton_steps=8,
                         init_theta=None, control=None):
    control = control or {}
    p = X_long.shape[-1]

    # Same fix as _fit_mle_intercept_slope's gk_weights bug: this is
    # captured by closure inside h_fn below (subject-invariant, so unlike
    # the other data arrays it isn't threaded through vmap args) - must be
    # a real jnp.array before that closure forms, not the raw
    # list/unconverted value arriving from R via reticulate.
    gk_weights = jnp.array(gk_weights)

    # theta = [beta(p), log_sigma_e, log_sigma_b, log_lambda0, log_shape,
    #          alpha_value, alpha_delta]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_LOG_LAMBDA0 = p + 2
    IDX_LOG_SHAPE = p + 3
    IDX_ALPHA_VALUE = p + 4
    IDX_ALPHA_DELTA = p + 5
    n_theta = p + 6

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        X_delta_surv=X_delta_surv, X_delta_quad=X_delta_quad,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, X_delta_surv_i, X_delta_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_delta = theta[IDX_ALPHA_DELTA]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_value_T = X_time_surv_i @ beta + b
        m_delta_T = X_delta_surv_i @ beta  # no b term - see module docstring
        log_h_T = log_h0_T + alpha_value * m_value_T + alpha_delta * m_delta_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_value_quad = X_time_quad_i @ beta + b
        m_delta_quad = X_delta_quad_i @ beta
        hazard_quad = jnp.exp(log_h0_quad + alpha_value * m_value_quad + alpha_delta * m_delta_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.grad(grad_fn, argnums=0)

    def find_mode_and_tau(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            hcurv = hess_fn(b, *subj_args, theta)
            hcurv_safe = jnp.where(hcurv < -1e-8, hcurv, -1e-8)
            return b - g / hcurv_safe, None

        b_mode, _ = jax.lax.scan(newton_step, 0.0, None, length=n_newton_steps)
        hess_final = hess_fn(b_mode, *subj_args, theta)
        hess_safe = jnp.where(hess_final < -1e-8, hess_final, -1e-8)
        tau = 1.0 / jnp.sqrt(-hess_safe)
        return b_mode, tau

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, X_delta_surv_i, X_delta_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, X_delta_surv_i, X_delta_quad_i)
        b_mode, tau = find_mode_and_tau(subj_args, theta)
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["X_delta_surv"], data["X_delta_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_DELTA] = 0.0
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA_VALUE, "beta": IDX_BETA, "log_sigma_b": IDX_LOG_SIGMA_B, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_delta(fit, p)


def _package_result_delta(fit, p):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1, p + 3]  # sigma_e, sigma_b, shape
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b", "log_lambda0", "shape", "alpha_value", "alpha_delta"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


# ==============================================================================
# Baseline (time-constant) covariates in the survival submodel, q=1
# (random intercept only), Weibull baseline hazard.
#
# Mathematically the SIMPLEST possible extension of the model: a baseline
# covariate contributes gamma' @ W_i, a per-subject CONSTANT (no time
# dependence at all) to the log-hazard - it factors identically into the
# event-time hazard AND every cumulative-hazard quadrature node, with no
# new quadrature machinery needed at all (unlike delta/area, which needed
# genuinely new R-side design arrays). This is the first implementation
# step (weibull-PH-aGH, q=1, plain "value" association) - matching the
# same "prove it on the simplest case first" discipline used for
# delta/area/random_formula. R-side guards ensure this function is only
# reached in that configuration for now.
# ==============================================================================
def _fit_mle_with_baseline_covariates(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                       T_surv, event, t_quad, gk_weights, W_surv,
                                       n_gh_nodes=15, n_newton_steps=8,
                                       init_theta=None, control=None):
    control = control or {}
    p = X_long.shape[-1]
    n_gamma = np.asarray(W_surv).shape[-1]

    # Same gk_weights closure fix as every other extended-channel function
    # in this codebase.
    gk_weights = jnp.array(gk_weights)

    # theta = [beta(p), log_sigma_e, log_sigma_b, log_lambda0, log_shape,
    #          gamma(n_gamma), alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_LOG_LAMBDA0 = p + 2
    IDX_LOG_SHAPE = p + 3
    IDX_GAMMA = (p + 4, p + 4 + n_gamma)
    IDX_ALPHA = p + 4 + n_gamma
    n_theta = p + 5 + n_gamma

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        W_surv=W_surv,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, W_surv_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        gamma = theta[IDX_GAMMA[0]:IDX_GAMMA[1]]
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        gamma_term = W_surv_i @ gamma  # scalar - time-constant, added identically below

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_T = X_time_surv_i @ beta + b
        log_h_T = log_h0_T + gamma_term + alpha * m_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_quad = X_time_quad_i @ beta + b
        hazard_quad = jnp.exp(log_h0_quad + gamma_term + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.grad(grad_fn, argnums=0)

    def find_mode_and_tau(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            hcurv = hess_fn(b, *subj_args, theta)
            hcurv_safe = jnp.where(hcurv < -1e-8, hcurv, -1e-8)
            return b - g / hcurv_safe, None

        b_mode, _ = jax.lax.scan(newton_step, 0.0, None, length=n_newton_steps)
        hess_final = hess_fn(b_mode, *subj_args, theta)
        hess_safe = jnp.where(hess_final < -1e-8, hess_final, -1e-8)
        tau = 1.0 / jnp.sqrt(-hess_safe)
        return b_mode, tau

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, W_surv_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, W_surv_i)
        b_mode, tau = find_mode_and_tau(subj_args, theta)
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["W_surv"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA] = 0.5
        # gamma left at 0 (no prior effect assumed)
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA, "beta": IDX_BETA, "log_sigma_b": IDX_LOG_SIGMA_B, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_baseline_covariates(fit, p, n_gamma)


def _package_result_baseline_covariates(fit, p, n_gamma):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1, p + 3]  # sigma_e, sigma_b, shape
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b", "log_lambda0", "shape"]
        + [f"gamma_{i}" for i in range(n_gamma)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


# ==============================================================================
# value + area association, q=1 (random intercept only).
#
# Unlike delta, the random-intercept contribution to area does NOT cancel:
# integral_0^t b_i ds = b_i * t (linear in t, not zero), since a constant
# integrates to something proportional to the integration length. This is
# the key mathematical difference from _fit_mle_with_delta above, and the
# reason area needed genuine nested quadrature on the R side
# (build_area_channel()) to evaluate the FIXED-EFFECTS part at each outer
# quadrature node, while delta needed none.
#
# Structurally otherwise identical to _fit_mle_with_delta: same adaptive-GH/
# Newton scaffolding, duplicated rather than shared, to keep each channel's
# implementation independently simple and to avoid risking the
# already-validated delta path while adding this one.
# ==============================================================================
def _fit_mle_with_area(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                        T_surv, event, t_quad, gk_weights,
                        X_area_surv, X_area_quad,
                        n_gh_nodes=15, n_newton_steps=8,
                        init_theta=None, control=None):
    control = control or {}
    p = X_long.shape[-1]

    # Same gk_weights closure fix as the other extended-channel functions:
    # must be a real jnp.array before h_fn's closure captures it.
    gk_weights = jnp.array(gk_weights)

    # theta = [beta(p), log_sigma_e, log_sigma_b, log_lambda0, log_shape,
    #          alpha_value, alpha_area]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_LOG_LAMBDA0 = p + 2
    IDX_LOG_SHAPE = p + 3
    IDX_ALPHA_VALUE = p + 4
    IDX_ALPHA_AREA = p + 5
    n_theta = p + 6

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        X_area_surv=X_area_surv, X_area_quad=X_area_quad,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, X_area_surv_i, X_area_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_area = theta[IDX_ALPHA_AREA]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_value_T = X_time_surv_i @ beta + b
        # area's random-effect contribution is b*T_i, NOT b (see module
        # docstring) - integral of a constant b over [0, T_i] is b*T_i.
        m_area_T = X_area_surv_i @ beta + b * T_i
        log_h_T = log_h0_T + alpha_value * m_value_T + alpha_area * m_area_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_value_quad = X_time_quad_i @ beta + b
        # t_quad_i broadcasts elementwise over the GK nodes, giving b*t_k
        # for each node k - the SAME b*t formula, just evaluated at each
        # outer quadrature time rather than at T_i.
        m_area_quad = X_area_quad_i @ beta + b * t_quad_i
        hazard_quad = jnp.exp(log_h0_quad + alpha_value * m_value_quad + alpha_area * m_area_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.grad(grad_fn, argnums=0)

    def find_mode_and_tau(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            hcurv = hess_fn(b, *subj_args, theta)
            hcurv_safe = jnp.where(hcurv < -1e-8, hcurv, -1e-8)
            return b - g / hcurv_safe, None

        b_mode, _ = jax.lax.scan(newton_step, 0.0, None, length=n_newton_steps)
        hess_final = hess_fn(b_mode, *subj_args, theta)
        hess_safe = jnp.where(hess_final < -1e-8, hess_final, -1e-8)
        tau = 1.0 / jnp.sqrt(-hess_safe)
        return b_mode, tau

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, X_area_surv_i, X_area_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, X_area_surv_i, X_area_quad_i)
        b_mode, tau = find_mode_and_tau(subj_args, theta)
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["X_area_surv"], data["X_area_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_AREA] = 0.0
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA_VALUE, "beta": IDX_BETA, "log_sigma_b": IDX_LOG_SIGMA_B, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_area(fit, p)


def _package_result_area(fit, p):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1, p + 3]  # sigma_e, sigma_b, shape
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b", "log_lambda0", "shape", "alpha_value", "alpha_area"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


# ==============================================================================
# value + area_avg association, q=1 (random intercept only).
#
# area_avg(t) = area(t)/t - JMbayes2's confirmed actual convention for its
# area() functional form (see build_design.R's parse_functional_forms docs
# for the full story of how this was confirmed). Its random-intercept
# contribution reduces to plain b: the time-average of a constant over
# [0,t] is that same constant, so unlike area's "+b*t" term, this channel
# uses "+b" - structurally the SAME random-effect handling as the "value"
# channel, just applied to a different (R-precomputed) fixed-effects
# design. Kept as its own function rather than merged with
# _fit_mle_with_delta or _fit_mle_with_area, for the same
# independent-validation reasons as those two.
# ==============================================================================
def _fit_mle_with_area_avg(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                            T_surv, event, t_quad, gk_weights,
                            X_area_avg_surv, X_area_avg_quad,
                            n_gh_nodes=15, n_newton_steps=8,
                            init_theta=None, control=None):
    control = control or {}
    p = X_long.shape[-1]

    # Same gk_weights closure fix as the other extended-channel functions.
    gk_weights = jnp.array(gk_weights)

    # theta = [beta(p), log_sigma_e, log_sigma_b, log_lambda0, log_shape,
    #          alpha_value, alpha_area_avg]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_LOG_LAMBDA0 = p + 2
    IDX_LOG_SHAPE = p + 3
    IDX_ALPHA_VALUE = p + 4
    IDX_ALPHA_AREA_AVG = p + 5
    n_theta = p + 6

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        X_area_avg_surv=X_area_avg_surv, X_area_avg_quad=X_area_avg_quad,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, X_area_avg_surv_i, X_area_avg_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_area_avg = theta[IDX_ALPHA_AREA_AVG]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_value_T = X_time_surv_i @ beta + b
        # area_avg's random-effect contribution is plain b (see module
        # docstring), same as value's.
        m_area_avg_T = X_area_avg_surv_i @ beta + b
        log_h_T = log_h0_T + alpha_value * m_value_T + alpha_area_avg * m_area_avg_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_value_quad = X_time_quad_i @ beta + b
        m_area_avg_quad = X_area_avg_quad_i @ beta + b
        hazard_quad = jnp.exp(log_h0_quad + alpha_value * m_value_quad + alpha_area_avg * m_area_avg_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.grad(grad_fn, argnums=0)

    def find_mode_and_tau(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            hcurv = hess_fn(b, *subj_args, theta)
            hcurv_safe = jnp.where(hcurv < -1e-8, hcurv, -1e-8)
            return b - g / hcurv_safe, None

        b_mode, _ = jax.lax.scan(newton_step, 0.0, None, length=n_newton_steps)
        hess_final = hess_fn(b_mode, *subj_args, theta)
        hess_safe = jnp.where(hess_final < -1e-8, hess_final, -1e-8)
        tau = 1.0 / jnp.sqrt(-hess_safe)
        return b_mode, tau

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, X_area_avg_surv_i, X_area_avg_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, X_area_avg_surv_i, X_area_avg_quad_i)
        b_mode, tau = find_mode_and_tau(subj_args, theta)
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["X_area_avg_surv"], data["X_area_avg_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_AREA_AVG] = 0.0
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA_VALUE, "beta": IDX_BETA, "log_sigma_b": IDX_LOG_SIGMA_B, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_area_avg(fit, p)


def _package_result_area_avg(fit, p):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1, p + 3]  # sigma_e, sigma_b, shape
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b", "log_lambda0", "shape", "alpha_value", "alpha_area_avg"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


# ==============================================================================
# q=2 (random intercept + slope) via 2D adaptive Gauss-Hermite.
#
# Added specifically to cross-validate against JM's weibull-PH-aGH with a
# real random slope (q=2), as a THIRD independent check (alongside jmjax's
# own weibull-PH-mcmc, which already matched JM closely: 0.5180 vs 0.5159 on
# a hard n=200 dataset) that the core Weibull+q=2 machinery is correct -
# this one has zero MCMC/sampling considerations at all, being a pure
# numerical optimization, so it isolates any remaining doubt about NUTS
# itself.
#
# Generalizes the existing scalar (q=1) adaptive-GH scheme in common.py to a
# bivariate random effect: per-subject 2D Newton mode-finding (reusing the
# same eigenvalue-clipped-Hessian approach validated in mcmc_model.py's
# find_map_b_std), then a TENSOR-PRODUCT Gauss-Hermite grid (K nodes per
# dimension -> K^2 combined nodes) centered and scaled via the local
# Cholesky factor at each subject's mode - the natural 2D generalization of
# the 1D "mode + sqrt(2)*tau*node" scheme. K defaults to 7 (K^2=49 nodes/
# subject), matching JM's own default GH-node reduction for 2D random
# effects (control$GHk auto-reduces to 5 for ncol(Z)<=3, n<2000 - JM source,
# confirmed during development).
# ==============================================================================
def _fit_mle_intercept_slope(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                              T_surv, event, t_quad, gk_weights,
                              Z_long=None, Z_time_surv=None, Z_time_quad=None,
                              n_gh_nodes=7, n_newton_steps=20,
                              init_theta=None, control=None):
    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape

    # IMPORTANT: gk_weights is captured by closure inside h_fn below (it's
    # subject-invariant, so unlike the other data arrays it isn't threaded
    # through vmap args) - must be a real jnp.array before that closure
    # forms, not the raw list/unconverted value arriving from R via
    # reticulate. Multiplying a raw Python list by a JAX tracer during
    # jax.grad/jax.hessian fails with "unsupported operand type(s) for *:
    # 'list' and 'LinearizeTracer'".
    gk_weights = jnp.array(gk_weights)

    if Z_long is None:
        Z_long = np.asarray(X_long)[:, :, :q]
        Z_time_surv = np.asarray(X_time_surv)[:, :q]
        Z_time_quad = np.asarray(X_time_quad)[:, :, :q]

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
    )

    # theta = [beta(p), log_sigma_e, log_sigma_b0, log_sigma_b1, atanh_rho,
    #          log_lambda0, log_shape, alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_LOG_LAMBDA0 = p + 4
    IDX_LOG_SHAPE = p + 5
    IDX_ALPHA = p + 6
    n_theta = p + 7

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)  # [K^2, 2]
    w_grid = (W1 * W2).ravel()                              # [K^2]

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        log_h_T = log_h0_T + alpha * m_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        hazard_quad = jnp.exp(log_h0_quad + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        b0, b1 = b[0], b[1]
        z0, z1 = b0 / sigma_b0, b1 / sigma_b1
        quad_form = (z0 ** 2 - 2.0 * rho * z0 * z1 + z1 ** 2) / (1.0 - rho ** 2)
        log_prior_b = (-jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b0) - jnp.log(sigma_b1)
                       - 0.5 * jnp.log(1.0 - rho ** 2) - 0.5 * quad_form)

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.hessian(h_fn, argnums=0)

    def _clip_hessian(H, eps=1e-6):
        H_sym = 0.5 * (H + H.T)
        eigvals, eigvecs = jnp.linalg.eigh(H_sym)
        eigvals_safe = jnp.minimum(eigvals, -eps)
        return eigvals_safe, eigvecs

    def find_mode_and_L(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            H = hess_fn(b, *subj_args, theta)
            eigvals_safe, eigvecs = _clip_hessian(H)
            H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
            delta = jnp.linalg.solve(H_safe, g)
            return b - delta, None

        b_final, _ = jax.lax.scan(newton_step, jnp.zeros(2), None, length=n_newton_steps)
        H_final = hess_fn(b_final, *subj_args, theta)
        eigvals_safe, eigvecs = _clip_hessian(H_final)
        cov = eigvecs @ jnp.diag(1.0 / (-eigvals_safe)) @ eigvecs.T
        L = jnp.linalg.cholesky(cov)
        return b_final, L

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i)
        b_mode, L = find_mode_and_L(subj_args, theta)
        b_nodes = b_mode[None, :] + jnp.sqrt(2.0) * (z_grid @ L.T)  # [K^2, 2]

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)          # [K^2]
        sq_norms = jnp.sum(z_grid ** 2, axis=1)         # [K^2]
        log_det_L = jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
        log_terms = jnp.log(w_grid) + h_vals + sq_norms
        # sqrt(2)^q with q=2 is exactly 2 - the tensor-product generalization
        # of the 1D case's log(sqrt(2)*tau) prefactor.
        return jnp.log(2.0) + log_det_L + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["Z_long"], data["Z_time_surv"], data["Z_time_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B0] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B1] = np.log(0.2)
        theta0[IDX_ATANH_RHO] = 0.0
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA] = 0.5
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA, "atanh_rho": IDX_ATANH_RHO, "beta": IDX_BETA, "log_sigma_b0": IDX_LOG_SIGMA_B0, "log_sigma_b1": IDX_LOG_SIGMA_B1, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_q2(fit, p)


# ==============================================================================
# q=2 (random intercept + slope) + baseline (time-constant) covariates in
# the SURVIVAL submodel (gamma' @ W_i). Longitudinal-side baseline
# covariates need NO separate function at q=2 either - beta's length
# already generalizes to any p automatically via _fit_mle_intercept_slope
# above (X_long.shape[-1]), exactly as it does at q=1. This function only
# adds what's genuinely new: gamma'W_i, a per-subject CONSTANT (q-
# independent, same formula as the q=1 baseline-covariates function) added
# on top of the already-validated q=2 scaffolding.
# ==============================================================================
def _fit_mle_q2_with_baseline_covariates(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                          T_surv, event, t_quad, gk_weights,
                                          Z_long, Z_time_surv, Z_time_quad, W_surv,
                                          n_gh_nodes=7, n_newton_steps=20,
                                          init_theta=None, control=None):
    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape
    n_gamma = np.asarray(W_surv).shape[-1]

    gk_weights = jnp.array(gk_weights)

    if Z_long is None:
        Z_long = np.asarray(X_long)[:, :, :q]
        Z_time_surv = np.asarray(X_time_surv)[:, :q]
        Z_time_quad = np.asarray(X_time_quad)[:, :, :q]

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
        W_surv=W_surv,
    )

    # theta = [beta(p), log_sigma_e, log_sigma_b0, log_sigma_b1, atanh_rho,
    #          log_lambda0, log_shape, gamma(n_gamma), alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_LOG_LAMBDA0 = p + 4
    IDX_LOG_SHAPE = p + 5
    IDX_GAMMA = (p + 6, p + 6 + n_gamma)
    IDX_ALPHA = p + 6 + n_gamma
    n_theta = p + 7 + n_gamma

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
    w_grid = (W1 * W2).ravel()

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        gamma = theta[IDX_GAMMA[0]:IDX_GAMMA[1]]
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        gamma_term = W_surv_i @ gamma  # scalar, time-constant

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        log_h_T = log_h0_T + gamma_term + alpha * m_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        hazard_quad = jnp.exp(log_h0_quad + gamma_term + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        b0, b1 = b[0], b[1]
        z0, z1 = b0 / sigma_b0, b1 / sigma_b1
        quad_form = (z0 ** 2 - 2.0 * rho * z0 * z1 + z1 ** 2) / (1.0 - rho ** 2)
        log_prior_b = (-jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b0) - jnp.log(sigma_b1)
                       - 0.5 * jnp.log(1.0 - rho ** 2) - 0.5 * quad_form)

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.hessian(h_fn, argnums=0)

    def _clip_hessian(H, eps=1e-6):
        H_sym = 0.5 * (H + H.T)
        eigvals, eigvecs = jnp.linalg.eigh(H_sym)
        eigvals_safe = jnp.minimum(eigvals, -eps)
        return eigvals_safe, eigvecs

    def find_mode_and_L(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            H = hess_fn(b, *subj_args, theta)
            eigvals_safe, eigvecs = _clip_hessian(H)
            H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
            delta = jnp.linalg.solve(H_safe, g)
            return b - delta, None

        b_final, _ = jax.lax.scan(newton_step, jnp.zeros(2), None, length=n_newton_steps)
        H_final = hess_fn(b_final, *subj_args, theta)
        eigvals_safe, eigvecs = _clip_hessian(H_final)
        cov = eigvecs @ jnp.diag(1.0 / (-eigvals_safe)) @ eigvecs.T
        L = jnp.linalg.cholesky(cov)
        return b_final, L

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i)
        b_mode, L = find_mode_and_L(subj_args, theta)
        b_nodes = b_mode[None, :] + jnp.sqrt(2.0) * (z_grid @ L.T)

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        sq_norms = jnp.sum(z_grid ** 2, axis=1)
        log_det_L = jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
        log_terms = jnp.log(w_grid) + h_vals + sq_norms
        return jnp.log(2.0) + log_det_L + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["Z_long"], data["Z_time_surv"], data["Z_time_quad"],
            data["W_surv"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B0] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B1] = np.log(0.2)
        theta0[IDX_ATANH_RHO] = 0.0
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA] = 0.5
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA, "atanh_rho": IDX_ATANH_RHO, "beta": IDX_BETA, "log_sigma_b0": IDX_LOG_SIGMA_B0, "log_sigma_b1": IDX_LOG_SIGMA_B1, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_q2_baseline_covariates(fit, p, n_gamma)


def _package_result_q2_baseline_covariates(fit, p, n_gamma):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1", p + 5: "shape"}
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho", "log_lambda0", "shape"]
        + [f"gamma_{i}" for i in range(n_gamma)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


def _package_result_q2(fit, p):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {
        p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1", p + 5: "shape",
    }
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    # rho = tanh(atanh_rho); delta method: d(tanh)/dx = 1 - tanh(x)^2
    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho", "log_lambda0", "shape", "alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


# ==============================================================================
# q=2 (random intercept + slope) + an EXTRA association channel (delta,
# area, or area_avg) - the general case underlying all three.
#
# Key insight from working out the q=1 formulas: each channel's random-
# effect contribution is Z_channel(t) @ b, where Z_channel(t) is that
# channel's own fixed-effects design restricted to its first q columns -
# the SAME convention already used for "value" (Z = X[:, :q]). At q=1 this
# reduces to the three DIFFERENT-LOOKING formulas already validated
# separately (delta: Z's first column is 0 -> "no b term"; area: Z's first
# column is t -> "b*t"; area_avg: Z's first column is 1 -> "plain b") -
# confirming those were all special cases of one general rule, not three
# unrelated ones. At q=2, Z_channel simply has 2 columns instead of 1, and
# the SAME general formula (Z_channel(t) @ b, a proper 2-vector dot
# product now) applies directly - no channel-specific formula needed here
# at all, unlike the three separate q=1 functions.
#
# `channel_name` is used only for output naming (e.g. "alpha_delta" vs
# "alpha_area") - the numerical code is identical regardless of which
# channel's precomputed X_extra/Z_extra arrays are passed in.
# ==============================================================================
def _fit_mle_q2_extra_channel(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                               T_surv, event, t_quad, gk_weights,
                               Z_long, Z_time_surv, Z_time_quad,
                               X_extra_surv, X_extra_quad, Z_extra_surv, Z_extra_quad,
                               channel_name,
                               n_gh_nodes=7, n_newton_steps=20,
                               init_theta=None, control=None):
    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape

    gk_weights = jnp.array(gk_weights)

    if Z_long is None:
        Z_long = np.asarray(X_long)[:, :, :q]
        Z_time_surv = np.asarray(X_time_surv)[:, :q]
        Z_time_quad = np.asarray(X_time_quad)[:, :, :q]

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
        X_extra_surv=X_extra_surv, X_extra_quad=X_extra_quad,
        Z_extra_surv=Z_extra_surv, Z_extra_quad=Z_extra_quad,
    )

    # theta = [beta(p), log_sigma_e, log_sigma_b0, log_sigma_b1, atanh_rho,
    #          log_lambda0, log_shape, alpha_value, alpha_extra]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_LOG_LAMBDA0 = p + 4
    IDX_LOG_SHAPE = p + 5
    IDX_ALPHA_VALUE = p + 6
    IDX_ALPHA_EXTRA = p + 7
    n_theta = p + 8

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)  # [K^2, 2]
    w_grid = (W1 * W2).ravel()                              # [K^2]

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i,
             X_extra_surv_i, X_extra_quad_i, Z_extra_surv_i, Z_extra_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        log_lambda0 = theta[IDX_LOG_LAMBDA0]
        shape = jnp.exp(theta[IDX_LOG_SHAPE])
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_extra = theta[IDX_ALPHA_EXTRA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_i) + log_lambda0
        m_value_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        m_extra_T = X_extra_surv_i @ beta + Z_extra_surv_i @ b
        log_h_T = log_h0_T + alpha_value * m_value_T + alpha_extra * m_extra_T

        log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad_i) + log_lambda0
        m_value_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        # Z_extra_quad_i has shape [n_quad, q]; b has shape [q] - matrix-
        # vector product gives [n_quad], matching m_value_quad's shape.
        m_extra_quad = X_extra_quad_i @ beta + Z_extra_quad_i @ b
        hazard_quad = jnp.exp(log_h0_quad + alpha_value * m_value_quad + alpha_extra * m_extra_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        b0, b1 = b[0], b[1]
        z0, z1 = b0 / sigma_b0, b1 / sigma_b1
        quad_form = (z0 ** 2 - 2.0 * rho * z0 * z1 + z1 ** 2) / (1.0 - rho ** 2)
        log_prior_b = (-jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b0) - jnp.log(sigma_b1)
                       - 0.5 * jnp.log(1.0 - rho ** 2) - 0.5 * quad_form)

        return log_p_y + log_p_surv + log_prior_b

    grad_fn = jax.grad(h_fn, argnums=0)
    hess_fn = jax.hessian(h_fn, argnums=0)

    def _clip_hessian(H, eps=1e-6):
        H_sym = 0.5 * (H + H.T)
        eigvals, eigvecs = jnp.linalg.eigh(H_sym)
        eigvals_safe = jnp.minimum(eigvals, -eps)
        return eigvals_safe, eigvecs

    def find_mode_and_L(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            H = hess_fn(b, *subj_args, theta)
            eigvals_safe, eigvecs = _clip_hessian(H)
            H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
            delta = jnp.linalg.solve(H_safe, g)
            return b - delta, None

        b_final, _ = jax.lax.scan(newton_step, jnp.zeros(2), None, length=n_newton_steps)
        H_final = hess_fn(b_final, *subj_args, theta)
        eigvals_safe, eigvecs = _clip_hessian(H_final)
        cov = eigvecs @ jnp.diag(1.0 / (-eigvals_safe)) @ eigvecs.T
        L = jnp.linalg.cholesky(cov)
        return b_final, L

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i,
                         X_extra_surv_i, X_extra_quad_i, Z_extra_surv_i, Z_extra_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, Z_long_i, Z_time_surv_i, Z_time_quad_i,
                     X_extra_surv_i, X_extra_quad_i, Z_extra_surv_i, Z_extra_quad_i)
        b_mode, L = find_mode_and_L(subj_args, theta)
        b_nodes = b_mode[None, :] + jnp.sqrt(2.0) * (z_grid @ L.T)  # [K^2, 2]

        def h_at_node(bk):
            return h_fn(bk, *subj_args, theta)

        h_vals = jax.vmap(h_at_node)(b_nodes)          # [K^2]
        sq_norms = jnp.sum(z_grid ** 2, axis=1)         # [K^2]
        log_det_L = jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
        log_terms = jnp.log(w_grid) + h_vals + sq_norms
        return jnp.log(2.0) + log_det_L + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["Z_long"], data["Z_time_surv"], data["Z_time_quad"],
            data["X_extra_surv"], data["X_extra_quad"],
            data["Z_extra_surv"], data["Z_extra_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B0] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B1] = np.log(0.2)
        theta0[IDX_ATANH_RHO] = 0.0
        theta0[IDX_LOG_LAMBDA0] = -2.0
        theta0[IDX_LOG_SHAPE] = np.log(1.2)
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_EXTRA] = 0.0
        # Overwrite the crude constants above with lme()/coxph()-derived
        # starting values when jm_fit() supplied them. Without this the
        # R side computes them and the backend silently ignores them -
        # which left alpha at 0.5 and overflowed exp(alpha * m) on data
        # whose longitudinal outcome is not O(1).
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta, {"alpha": IDX_ALPHA_VALUE, "atanh_rho": IDX_ATANH_RHO, "beta": IDX_BETA, "log_sigma_b0": IDX_LOG_SIGMA_B0, "log_sigma_b1": IDX_LOG_SIGMA_B1, "log_sigma_e": IDX_LOG_SIGMA_E, "log_lambda0": IDX_LOG_LAMBDA0, "log_shape": IDX_LOG_SHAPE}, control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_q2_extra_channel(fit, p, channel_name)


def _package_result_q2_extra_channel(fit, p, channel_name):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {
        p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1", p + 5: "shape",
    }
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho", "log_lambda0", "shape",
           "alpha_value", f"alpha_{channel_name}"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }


def _package_result(fit, idx, p):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    # delta-method transforms for log-scale params
    for log_i, name in [(idx["log_sigma_e"], None), (idx["log_sigma_b"], None)]:
        est[log_i] = np.exp(theta_opt[log_i])
        se_nat[log_i] = est[log_i] * se[log_i]
    log_shape_i = idx["baseline"][0] + 1
    est[log_shape_i] = np.exp(theta_opt[log_shape_i])
    se_nat[log_shape_i] = est[log_shape_i] * se[log_shape_i]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b", "log_lambda0", "shape", "alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": common.natural_scale_vcov(fit, est, se_nat),
        "loglik": fit["loglik"],
        "convergence": {"converged": fit["converged"], "message": fit["message"], "n_iter": fit["n_iter"],
                        # grad_max is what distinguishes "stopped cleanly"
                        # from "reached an optimum". Copied explicitly
                        # because this dict is built by hand - an earlier
                        # version listed only three keys, so the gradient
                        # warning shipped and was silently never able to
                        # fire.
                        "grad_norm": fit.get("grad_norm"),
                        "grad_max": fit.get("grad_max"),
                        # scipy's raw exit status, kept alongside the
                        # `converged` flag so the two can be compared -
                        # they differ exactly when a method's own
                        # stopping rule is stricter than a stationary
                        # point requires.
                        "optimizer_success": fit.get("optimizer_success")},
        "posterior_samples": None,
    }
