"""Spline-approximated baseline hazard, adaptive-GH MLE. Knots are computed
R-side (see R/knots.R, ported from JM's internal logic) and passed in as
pre-evaluated basis matrices B_T / B_quad - Python never sees raw knot
values, only the finished design."""
import numpy as np
import jax.numpy as jnp

from . import common


def fit_mle(X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights,
            B_T, B_quad, n_splines,
            random_effects="intercept",
            Z_long=None, Z_time_surv=None, Z_time_quad=None,
            X_delta_surv=None, X_delta_quad=None, Z_delta_surv=None, Z_delta_quad=None,
            W_surv=None,
            n_gh_nodes=15, n_newton_steps=8, init_theta=None, control=None):
    control = control or {}

    if random_effects != "intercept":
        q2_kwargs = dict(
            # int() coercion is REQUIRED, not defensive: R passes numeric
            # literals (e.g. control = list(n_gh_nodes_per_dim = 7)) as
            # DOUBLES, which reticulate forwards to Python as 7.0 -
            # numpy.polynomial.hermite.hermgauss() then rejects it with
            # "deg must be an integer, received 7.0". Without this, these
            # two control options were effectively unusable from R unless
            # the caller happened to write 7L instead of 7.
            n_gh_nodes=int(control.get("n_gh_nodes_per_dim", 7)),
            n_newton_steps=int(control.get("n_newton_steps_q2", 20)),
            init_theta=init_theta, control=control,
        )
        if X_delta_surv is not None:
            return _fit_mle_spline_q2_extra_channel(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights, B_T, B_quad, n_splines,
                Z_long, Z_time_surv, Z_time_quad,
                X_delta_surv, X_delta_quad, Z_delta_surv, Z_delta_quad,
                channel_name="delta", **q2_kwargs,
            )
        if W_surv is not None:
            # q=2 baseline covariates - combines the already-validated q=2
            # scaffolding above with the same gamma_term addition used at
            # q=1 (see _fit_mle_spline_with_baseline_covariates), mirroring
            # weibull_model.py's own q=1 -> q=2 staged rollout for this
            # exact feature.
            return _fit_mle_spline_q2_with_baseline_covariates(
                X_long, y_long, n_obs, X_time_surv, X_time_quad,
                T_surv, event, t_quad, gk_weights, B_T, B_quad, n_splines, W_surv,
                Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
                **q2_kwargs,
            )
        # Plain q=2, value-only - the already-validated original path.
        return _fit_mle_spline_intercept_slope(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, B_T, B_quad, n_splines,
            Z_long=Z_long, Z_time_surv=Z_time_surv, Z_time_quad=Z_time_quad,
            **q2_kwargs,
        )

    if W_surv is not None:
        # Routed to a SEPARATE function, mirroring weibull_model.py's
        # _fit_mle_with_baseline_covariates and this file's own
        # _fit_mle_spline_with_delta pattern - keeps the already-validated
        # plain spline path completely unaffected. Not yet combinable with
        # X_delta_surv (delta channel) in the same fit - first
        # implementation step only, matching weibull's own staged rollout.
        return _fit_mle_spline_with_baseline_covariates(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, B_T, B_quad, n_splines,
            W_surv,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    if X_delta_surv is not None:
        # Routed to a SEPARATE function (not a modification of the plain
        # value-only path below), mirroring weibull_model.py's pattern -
        # keeps the already-validated single-channel spline path
        # completely unaffected by this new capability.
        return _fit_mle_spline_with_delta(
            X_long, y_long, n_obs, X_time_surv, X_time_quad,
            T_surv, event, t_quad, gk_weights, B_T, B_quad, n_splines,
            X_delta_surv, X_delta_quad,
            n_gh_nodes=n_gh_nodes, n_newton_steps=n_newton_steps,
            init_theta=init_theta, control=control,
        )

    p = X_long.shape[-1]
    idx, n_theta = common.make_theta_layout(p, n_splines)

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
    )
    # B_T / B_quad are keyed per-subject just like X_time_surv / X_time_quad,
    # so they get bundled into the closure via a small wrapper rather than
    # common.py's generic data dict (which doesn't know about them).
    B_T_j = jnp.array(B_T)
    B_quad_j = jnp.array(B_quad)

    # NOTE: common.build_neg_log_lik() expects log_h0_fn(t, baseline_params)
    # with no subject index, but the spline basis (B_T_i, B_quad_i) varies
    # per subject - so this doesn't fit that generic interface cleanly.
    # build_neg_log_lik_spline() below duplicates the adaptive-GH/Newton
    # wiring with B_T/B_quad threaded through explicitly instead. If a third
    # baseline-hazard family is added later, that's the natural point to
    # generalize common.py's interface rather than duplicating a third time.
    neg_log_lik = build_neg_log_lik_spline(idx, B_T_j, B_quad_j,
                                            n_gh=n_gh_nodes, n_newton_steps=n_newton_steps)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[idx["beta"][0]] = 2.0
        theta0[idx["log_sigma_e"]] = np.log(0.5)
        theta0[idx["log_sigma_b"]] = np.log(0.5)
        theta0[idx["alpha"]] = 0.5
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta,
            {"beta": idx["beta"], "log_sigma_e": idx["log_sigma_e"],
             "log_sigma_b": idx["log_sigma_b"], "W": idx["baseline"],
             "alpha": idx["alpha"]},
            control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result(fit, idx, p, n_splines)


def build_neg_log_lik_spline(idx, B_T, B_quad, n_gh=15, n_newton_steps=8):
    """
    Spline-specific variant of common.build_neg_log_lik(): the baseline
    hazard depends on a per-subject basis matrix (B_T_i, B_quad_i), not just
    a scalar/vector time argument, so it doesn't fit common.py's
    `log_h0_fn(t, baseline_params)` signature cleanly. Duplicating the
    Newton/adaptive-GH wiring here (rather than contorting common.py's
    interface) keeps both readable; if a third baseline-hazard family is
    added later, this is the natural point to generalize the shared
    interface instead of duplicating a third time.
    """
    import jax

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, gk_weights, B_T_i, B_quad_i, theta):
        beta = theta[idx["beta"][0]:idx["beta"][1]]
        sigma_e = jnp.exp(theta[idx["log_sigma_e"]])
        sigma_b = jnp.exp(theta[idx["log_sigma_b"]])
        W = theta[idx["baseline"][0]:idx["baseline"][1]]
        alpha = theta[idx["alpha"]]

        mu = X_long_i @ beta + b
        log_p_y = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y_sum = jnp.sum(jnp.where(mask, log_p_y, 0.0))

        m_T = X_time_surv_i @ beta + b
        log_h_T = jnp.dot(B_T_i, W) + alpha * m_T

        m_quad = X_time_quad_i @ beta + b
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2
        return log_p_y_sum + log_p_surv + log_prior_b

    grad_b_fn = jax.grad(h_fn, argnums=0)
    hess_b_fn = jax.grad(grad_b_fn, argnums=0)

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def find_mode_and_tau(*args):
        def newton_step(b, _):
            g = grad_b_fn(b, *args)
            hcurv = hess_b_fn(b, *args)
            hcurv_safe = jnp.where(hcurv < -1e-8, hcurv, -1e-8)
            return b - g / hcurv_safe, None

        b_mode, _ = jax.lax.scan(newton_step, 0.0, None, length=n_newton_steps)
        hess_final = hess_b_fn(b_mode, *args)
        hess_safe = jnp.where(hess_final < -1e-8, hess_final, -1e-8)
        tau = 1.0 / jnp.sqrt(-hess_safe)
        return b_mode, tau

    def subj_log_lik_fn(*args):
        b_mode, tau = find_mode_and_tau(*args)
        b_nodes = b_mode + jnp.sqrt(2.0) * tau * gh_nodes

        def h_at_node(bk):
            return h_fn(bk, *args)

        h_vals = jax.vmap(h_at_node)(b_nodes)
        log_terms = jnp.log(gh_weights) + h_vals + gh_nodes ** 2
        return jnp.log(jnp.sqrt(2.0) * tau) + jax.nn.logsumexp(log_terms)

    def neg_log_lik(theta, data):
        all_ll = jax.vmap(
            subj_log_lik_fn,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["gk_weights"], B_T, B_quad, theta
        )
        return -jnp.sum(all_ll)

    return neg_log_lik


def _package_result(fit, idx, p, n_splines):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    est[idx["log_sigma_e"]] = np.exp(theta_opt[idx["log_sigma_e"]])
    se_nat[idx["log_sigma_e"]] = est[idx["log_sigma_e"]] * se[idx["log_sigma_e"]]
    est[idx["log_sigma_b"]] = np.exp(theta_opt[idx["log_sigma_b"]])
    se_nat[idx["log_sigma_b"]] = est[idx["log_sigma_b"]] * se[idx["log_sigma_b"]]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b"]
        + [f"W{i}" for i in range(n_splines)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
# value + delta association, spline baseline, q=1 (random intercept only).
#
# Mirrors weibull_model.py's _fit_mle_with_delta EXACTLY in how the delta
# channel enters the hazard (m_delta = X_delta(t) @ beta, NO random-effect
# term - the intercept cancels in the t=0 subtraction, same as the Weibull
# case since this is a property of the RANDOM EFFECTS STRUCTURE, not the
# baseline hazard choice). The only difference from the Weibull version is
# the baseline hazard itself: log_h0(t) = B(t) @ W (a spline basis
# combination) instead of a closed-form Weibull formula - the SAME swap
# already made once when comparing build_neg_log_lik_spline() against
# common.py's Weibull-specific build_neg_log_lik().
#
# X_delta_surv/X_delta_quad are the SAME R-computed arrays already used for
# the Weibull case (build_delta_channel() doesn't know or care about the
# baseline hazard choice) - no new R-side code needed for this extension.
# ==============================================================================
def _fit_mle_spline_with_delta(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                T_surv, event, t_quad, gk_weights,
                                B_T, B_quad, n_splines,
                                X_delta_surv, X_delta_quad,
                                n_gh_nodes=15, n_newton_steps=8,
                                init_theta=None, control=None):
    import jax

    control = control or {}
    p = X_long.shape[-1]

    # Same gk_weights closure fix as every other extended-channel function
    # in this codebase (weibull_model.py's _fit_mle_with_delta/_with_area/
    # _with_area_avg all needed this) - must be a real jnp.array before
    # h_fn's closure captures it, not the raw unconverted value arriving
    # from R via reticulate.
    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)

    # theta = [beta(p), log_sigma_e, log_sigma_b, W(n_splines),
    #          alpha_value, alpha_delta]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_W = (p + 2, p + 2 + n_splines)
    IDX_ALPHA_VALUE = p + 2 + n_splines
    IDX_ALPHA_DELTA = p + 3 + n_splines
    n_theta = p + 4 + n_splines

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        X_delta_surv=X_delta_surv, X_delta_quad=X_delta_quad,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, B_T_i, B_quad_i,
             X_delta_surv_i, X_delta_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        W = theta[IDX_W[0]:IDX_W[1]]
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_delta = theta[IDX_ALPHA_DELTA]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        m_value_T = X_time_surv_i @ beta + b
        m_delta_T = X_delta_surv_i @ beta  # no b term - see module docstring
        log_h_T = jnp.dot(B_T_i, W) + alpha_value * m_value_T + alpha_delta * m_delta_T

        m_value_quad = X_time_quad_i @ beta + b
        m_delta_quad = X_delta_quad_i @ beta
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + alpha_value * m_value_quad + alpha_delta * m_delta_quad)
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
                         T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                         X_delta_surv_i, X_delta_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                     X_delta_surv_i, X_delta_quad_i)
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            B_T, B_quad,
            data["X_delta_surv"], data["X_delta_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_DELTA] = 0.0
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_spline_delta(fit, p, n_splines)


def _package_result_spline_delta(fit, p, n_splines):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1]  # sigma_e, sigma_b
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b"]
        + [f"W{i}" for i in range(n_splines)]
        + ["alpha_value", "alpha_delta"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
# Baseline covariates (gamma/W_surv), spline baseline hazard, q=1 (random
# intercept only), plain "value" association.
#
# Closes a longstanding gap: weibull_model.py's
# _fit_mle_with_baseline_covariates supported this at q=1 and q=2, but the
# spline-baseline MLE path never got the equivalent. Mirrors that function
# EXACTLY in how the gamma/W_surv term enters the model - a baseline
# covariate contributes gamma' @ W_i, a per-subject CONSTANT (no time
# dependence), identically to the event-time hazard and every
# cumulative-hazard quadrature node. No new quadrature machinery needed,
# same as the Weibull case - this is a property of baseline covariates
# being time-constant, independent of which baseline hazard family is
# used. The only difference from weibull's version is the baseline hazard
# itself: log_h0(t) = B(t) @ W (spline basis) instead of a closed-form
# Weibull formula - the same swap _fit_mle_spline_with_delta already made
# relative to weibull_model.py's _fit_mle_with_delta.
#
# Not yet combinable with the delta channel in the same fit (see fit_mle's
# dispatch above) - first implementation step only, matching weibull's own
# staged rollout ("prove it on the simplest case first").
# ==============================================================================
def _fit_mle_spline_with_baseline_covariates(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                              T_surv, event, t_quad, gk_weights,
                                              B_T, B_quad, n_splines, W_surv,
                                              n_gh_nodes=15, n_newton_steps=8,
                                              init_theta=None, control=None):
    import jax

    control = control or {}
    p = X_long.shape[-1]
    n_gamma = np.asarray(W_surv).shape[-1]

    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)

    # theta = [beta(p), log_sigma_e, log_sigma_b, W(n_splines),
    #          gamma(n_gamma), alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B = p + 1
    IDX_W = (p + 2, p + 2 + n_splines)
    IDX_GAMMA = (p + 2 + n_splines, p + 2 + n_splines + n_gamma)
    IDX_ALPHA = p + 2 + n_splines + n_gamma
    n_theta = p + 3 + n_splines + n_gamma

    data = common.to_jax_data(
        X_long=X_long, y_long=y_long, n_obs=n_obs,
        X_time_surv=X_time_surv, X_time_quad=X_time_quad,
        T_surv=T_surv, event=event, t_quad=t_quad, gk_weights=gk_weights,
        W_surv=W_surv,
    )

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, B_T_i, B_quad_i, W_surv_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b = jnp.exp(theta[IDX_LOG_SIGMA_B])
        W = theta[IDX_W[0]:IDX_W[1]]
        gamma = theta[IDX_GAMMA[0]:IDX_GAMMA[1]]
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        gamma_term = W_surv_i @ gamma  # scalar - time-constant, added identically below

        m_T = X_time_surv_i @ beta + b
        log_h_T = jnp.dot(B_T_i, W) + gamma_term + alpha * m_T

        m_quad = X_time_quad_i @ beta + b
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + gamma_term + alpha * m_quad)
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
                         T_i, event_i, t_quad_i, B_T_i, B_quad_i, W_surv_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, B_T_i, B_quad_i, W_surv_i)
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            B_T, B_quad, data["W_surv"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B] = np.log(0.5)
        theta0[IDX_ALPHA] = 0.5
        # gamma left at 0 unless a coxph pre-fit supplies it below
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta,
            {"beta": IDX_BETA, "log_sigma_e": IDX_LOG_SIGMA_E,
             "log_sigma_b": IDX_LOG_SIGMA_B, "W": IDX_W, "gamma": IDX_GAMMA,
             "alpha": IDX_ALPHA},
            control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_spline_with_baseline_covariates(fit, p, n_splines, n_gamma)


def _package_result_spline_with_baseline_covariates(fit, p, n_splines, n_gamma):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = [p, p + 1]  # sigma_e, sigma_b
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b"]
        + [f"W{i}" for i in range(n_splines)]
        + [f"gamma_{i}" for i in range(n_gamma)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
# q=2 (random intercept + slope) via 2D adaptive Gauss-Hermite, spline
# baseline hazard, value-only association.
#
# Combines weibull_model.py's already-validated _fit_mle_intercept_slope
# structure (2D Newton mode-finding via eigenvalue-clipped Hessian,
# tensor-product Gauss-Hermite quadrature, bivariate-normal random-effects
# prior - see that function's docstring for the full derivation) with THIS
# module's spline baseline hazard (B(t) @ W instead of a closed-form
# Weibull formula) - the only substantive difference between the two
# functions is that one line.
#
# Previously flagged as "a separate, bigger undertaking not yet
# implemented" - this is that undertaking. Should be validated the same
# way weibull_model.py's q=2 extension was: truth recovery on simulated
# data, then cross-validated against R's JM package (which supports a
# spline baseline with q=2 via its own adaptive-GH implementation).
# ==============================================================================
def _fit_mle_spline_intercept_slope(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                     T_surv, event, t_quad, gk_weights,
                                     B_T, B_quad, n_splines,
                                     Z_long=None, Z_time_surv=None, Z_time_quad=None,
                                     n_gh_nodes=7, n_newton_steps=20,
                                     init_theta=None, control=None):
    import jax

    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape

    # Same gk_weights/B_T/B_quad closure fix as every other extended
    # function in this codebase - must be real jnp.arrays before h_fn's
    # closure captures them, not raw unconverted values arriving from R
    # via reticulate.
    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)

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
    #          W(n_splines), alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_W = (p + 4, p + 4 + n_splines)
    IDX_ALPHA = p + 4 + n_splines
    n_theta = p + 5 + n_splines

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)  # [K^2, 2]
    w_grid = (W1 * W2).ravel()                              # [K^2]

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, B_T_i, B_quad_i,
             Z_long_i, Z_time_surv_i, Z_time_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        W = theta[IDX_W[0]:IDX_W[1]]
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        m_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        log_h_T = jnp.dot(B_T_i, W) + alpha * m_T

        m_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + alpha * m_quad)
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
                         T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                         Z_long_i, Z_time_surv_i, Z_time_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                     Z_long_i, Z_time_surv_i, Z_time_quad_i)
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            B_T, B_quad,
            data["Z_long"], data["Z_time_surv"], data["Z_time_quad"], theta
        )
        return -jnp.sum(all_ll)

    if init_theta is None:
        # Hard-coded fallback, then overwritten by any lme()/coxph()
        # pre-fit values supplied through control - see
        # common.init_theta_from_prefit() for why that matters here.
        theta0 = np.zeros(n_theta)
        theta0[IDX_BETA[0]] = 2.0
        theta0[IDX_LOG_SIGMA_E] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B0] = np.log(0.5)
        theta0[IDX_LOG_SIGMA_B1] = np.log(0.2)
        theta0[IDX_ATANH_RHO] = 0.0
        theta0[IDX_ALPHA] = 0.5
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta,
            {"beta": IDX_BETA, "log_sigma_e": IDX_LOG_SIGMA_E,
             "log_sigma_b0": IDX_LOG_SIGMA_B0, "log_sigma_b1": IDX_LOG_SIGMA_B1,
             "atanh_rho": IDX_ATANH_RHO, "W": IDX_W, "alpha": IDX_ALPHA},
            control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_spline_q2(fit, p, n_splines)


def _package_result_spline_q2(fit, p, n_splines):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1"}
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho"]
        + [f"W{i}" for i in range(n_splines)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
# q=2 (random intercept + slope) + baseline (time-constant) covariates in
# the SURVIVAL submodel (gamma' @ W_i), spline baseline hazard.
#
# Combines TWO previously-separate, already-validated pieces: the 2D
# adaptive-GH scaffolding from _fit_mle_spline_intercept_slope (tensor-
# product Gauss-Hermite quadrature, eigenvalue-clipped Hessian Newton
# mode-finding, bivariate-normal random-effects prior) and the gamma_term
# addition from weibull_model.py's _fit_mle_q2_with_baseline_covariates
# (gamma'W_i, a per-subject constant, q-independent - the identical
# formula used at q=1, since baseline covariates don't interact with the
# random-effects dimension at all). Longitudinal-side baseline covariates
# need no separate handling here either, exactly as in the weibull q=2
# case - beta's length already generalizes via X_long.shape[-1].
#
# Not yet combinable with the delta channel in the same fit (mirrors the
# q=1 baseline-covariates function's own scope limit).
# ==============================================================================
def _fit_mle_spline_q2_with_baseline_covariates(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                                 T_surv, event, t_quad, gk_weights,
                                                 B_T, B_quad, n_splines, W_surv,
                                                 Z_long=None, Z_time_surv=None, Z_time_quad=None,
                                                 n_gh_nodes=7, n_newton_steps=20,
                                                 init_theta=None, control=None):
    import jax

    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape
    n_gamma = np.asarray(W_surv).shape[-1]

    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)

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
    #          W(n_splines), gamma(n_gamma), alpha]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_W = (p + 4, p + 4 + n_splines)
    IDX_GAMMA = (p + 4 + n_splines, p + 4 + n_splines + n_gamma)
    IDX_ALPHA = p + 4 + n_splines + n_gamma
    n_theta = p + 5 + n_splines + n_gamma

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)
    w_grid = (W1 * W2).ravel()

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, B_T_i, B_quad_i,
             Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        W = theta[IDX_W[0]:IDX_W[1]]
        gamma = theta[IDX_GAMMA[0]:IDX_GAMMA[1]]
        alpha = theta[IDX_ALPHA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        gamma_term = W_surv_i @ gamma  # scalar, time-constant, q-independent

        m_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        log_h_T = jnp.dot(B_T_i, W) + gamma_term + alpha * m_T

        m_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + gamma_term + alpha * m_quad)
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

    # EXPERIMENTAL, opt-in via control$cholesky_first_newton = TRUE.
    # Default (False) is the existing, fully-validated eigendecomposition
    # path - unchanged. See common.newton_step_cholesky_first()'s docstring
    # for the reasoning, and for why this may NOT actually be faster under
    # jit (both branches get traced). This function is the first place it's
    # wired in deliberately: prove or disprove the benefit on ONE function
    # before touching the other five that share this same pattern.
    use_cholesky_first = bool(control.get("cholesky_first_newton", False))

    def find_mode_and_L(subj_args, theta):
        def newton_step(b, _):
            g = grad_fn(b, *subj_args, theta)
            H = hess_fn(b, *subj_args, theta)
            if use_cholesky_first:
                # Python-level `if` on a STATIC flag (not a traced value) -
                # resolved at trace time, so only the selected branch ends
                # up in the compiled graph.
                delta = common.newton_step_cholesky_first(H, g)
            else:
                eigvals_safe, eigvecs = _clip_hessian(H)
                H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
                delta = jnp.linalg.solve(H_safe, g)
            return b - delta, None

        b_final, _ = jax.lax.scan(newton_step, jnp.zeros(2), None, length=n_newton_steps)
        # The FINAL covariance still uses the eigendecomposition path
        # regardless: it needs the eigenvalues themselves (for the clipped
        # inverse), not just a linear solve, and runs only ONCE per subject
        # rather than per Newton step - not a bottleneck, no reason to risk
        # changing it.
        H_final = hess_fn(b_final, *subj_args, theta)
        eigvals_safe, eigvecs = _clip_hessian(H_final)
        cov = eigvecs @ jnp.diag(1.0 / (-eigvals_safe)) @ eigvecs.T
        L = jnp.linalg.cholesky(cov)
        return b_final, L

    def subj_log_lik_fn(X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                         T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                         Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                     Z_long_i, Z_time_surv_i, Z_time_quad_i, W_surv_i)
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            B_T, B_quad,
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
        theta0[IDX_ALPHA] = 0.5
        # gamma left at 0 unless a coxph pre-fit supplies it below
        _theta_default = theta0.copy()
        theta0 = common.init_theta_from_prefit(
            n_theta,
            {"beta": IDX_BETA, "log_sigma_e": IDX_LOG_SIGMA_E,
             "log_sigma_b0": IDX_LOG_SIGMA_B0, "log_sigma_b1": IDX_LOG_SIGMA_B1,
             "atanh_rho": IDX_ATANH_RHO, "W": IDX_W, "gamma": IDX_GAMMA,
             "alpha": IDX_ALPHA},
            control, theta0)
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_spline_q2_with_baseline_covariates(fit, p, n_splines, n_gamma)


def _package_result_spline_q2_with_baseline_covariates(fit, p, n_splines, n_gamma):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1"}
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho"]
        + [f"W{i}" for i in range(n_splines)]
        + [f"gamma_{i}" for i in range(n_gamma)]
        + ["alpha"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
# and by the same generic pattern, area/area_avg once ported), spline
# baseline hazard.
#
# Combines THREE previously-separate pieces, each already independently
# validated: weibull_model.py's _fit_mle_q2_extra_channel's generic
# "Z_channel(t) @ b" extra-channel formula (derived while extending
# Weibull to q=2 - see that function's docstring for the full derivation
# of why delta/area/area_avg all reduce to one general rule), this
# module's spline baseline hazard (B(t) @ W), and the 2D adaptive-GH
# machinery from _fit_mle_spline_intercept_slope. `channel_name` is used
# only for output naming - the numerical code is identical regardless of
# which channel's precomputed arrays are passed in.
# ==============================================================================
def _fit_mle_spline_q2_extra_channel(X_long, y_long, n_obs, X_time_surv, X_time_quad,
                                      T_surv, event, t_quad, gk_weights,
                                      B_T, B_quad, n_splines,
                                      Z_long, Z_time_surv, Z_time_quad,
                                      X_extra_surv, X_extra_quad, Z_extra_surv, Z_extra_quad,
                                      channel_name,
                                      n_gh_nodes=7, n_newton_steps=20,
                                      init_theta=None, control=None):
    import jax

    control = control or {}
    q = 2
    N_sub, max_obs, p = np.asarray(X_long).shape

    gk_weights = jnp.array(gk_weights)
    B_T = jnp.array(B_T)
    B_quad = jnp.array(B_quad)

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
    #          W(n_splines), alpha_value, alpha_extra]
    IDX_BETA = (0, p)
    IDX_LOG_SIGMA_E = p
    IDX_LOG_SIGMA_B0 = p + 1
    IDX_LOG_SIGMA_B1 = p + 2
    IDX_ATANH_RHO = p + 3
    IDX_W = (p + 4, p + 4 + n_splines)
    IDX_ALPHA_VALUE = p + 4 + n_splines
    IDX_ALPHA_EXTRA = p + 5 + n_splines
    n_theta = p + 6 + n_splines

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh_nodes)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)
    Z1, Z2 = jnp.meshgrid(gh_nodes, gh_nodes, indexing="ij")
    W1, W2 = jnp.meshgrid(gh_weights, gh_weights, indexing="ij")
    z_grid = jnp.stack([Z1.ravel(), Z2.ravel()], axis=-1)  # [K^2, 2]
    w_grid = (W1 * W2).ravel()                              # [K^2]

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, B_T_i, B_quad_i,
             Z_long_i, Z_time_surv_i, Z_time_quad_i,
             X_extra_surv_i, X_extra_quad_i, Z_extra_surv_i, Z_extra_quad_i, theta):
        beta = theta[IDX_BETA[0]:IDX_BETA[1]]
        sigma_e = jnp.exp(theta[IDX_LOG_SIGMA_E])
        sigma_b0 = jnp.exp(theta[IDX_LOG_SIGMA_B0])
        sigma_b1 = jnp.exp(theta[IDX_LOG_SIGMA_B1])
        rho = jnp.tanh(theta[IDX_ATANH_RHO])
        W = theta[IDX_W[0]:IDX_W[1]]
        alpha_value = theta[IDX_ALPHA_VALUE]
        alpha_extra = theta[IDX_ALPHA_EXTRA]

        mu = X_long_i @ beta + Z_long_i @ b
        log_p_y_terms = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y = jnp.sum(jnp.where(mask, log_p_y_terms, 0.0))

        m_value_T = X_time_surv_i @ beta + Z_time_surv_i @ b
        m_extra_T = X_extra_surv_i @ beta + Z_extra_surv_i @ b
        log_h_T = jnp.dot(B_T_i, W) + alpha_value * m_value_T + alpha_extra * m_extra_T

        m_value_quad = X_time_quad_i @ beta + Z_time_quad_i @ b
        # Z_extra_quad_i has shape [n_quad, q]; b has shape [q] - matrix-
        # vector product gives [n_quad], matching m_value_quad's shape.
        m_extra_quad = X_extra_quad_i @ beta + Z_extra_quad_i @ b
        hazard_quad = jnp.exp(jnp.dot(B_quad_i, W) + alpha_value * m_value_quad + alpha_extra * m_extra_quad)
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
                         T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                         Z_long_i, Z_time_surv_i, Z_time_quad_i,
                         X_extra_surv_i, X_extra_quad_i, Z_extra_surv_i, Z_extra_quad_i, theta):
        subj_args = (X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
                     T_i, event_i, t_quad_i, B_T_i, B_quad_i,
                     Z_long_i, Z_time_surv_i, Z_time_quad_i,
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            B_T, B_quad,
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
        theta0[IDX_ALPHA_VALUE] = 0.5
        theta0[IDX_ALPHA_EXTRA] = 0.0
    else:
        theta0 = np.asarray(init_theta)

    # Reject a pre-fit start that gives a non-finite objective - see
    # common.guard_initial_theta(). locals() keeps this safe in the paths
    # that never build a pre-fit start.
    _td = locals().get("_theta_default")
    if _td is not None:
        theta0 = common.guard_initial_theta(neg_log_lik, data, theta0, _td)
    fit = common.fit_theta(neg_log_lik, data, theta0,
                           opt_method=str(control.get("opt_method", "BFGS")),
                           parscale=control.get("parscale", 0.01),
                            maxiter=control.get("maxiter", 1000),
                            ftol=control.get("ftol", 1e-7))

    return _package_result_spline_q2_extra_channel(fit, p, n_splines, channel_name)


def _package_result_spline_q2_extra_channel(fit, p, n_splines, channel_name):
    theta_opt, se = fit["theta_opt"], fit["se_theta"]
    est = np.copy(theta_opt)
    se_nat = np.copy(se)

    log_indices = {p: "sigma_e", p + 1: "sigma_b0", p + 2: "sigma_b1"}
    for idx in log_indices:
        est[idx] = np.exp(theta_opt[idx])
        se_nat[idx] = est[idx] * se[idx]

    rho_idx = p + 3
    est[rho_idx] = np.tanh(theta_opt[rho_idx])
    se_nat[rho_idx] = (1.0 - est[rho_idx] ** 2) * se[rho_idx]

    names = (
        [f"beta_{i}" for i in range(p)]
        + ["sigma_e", "sigma_b0", "sigma_b1", "rho"]
        + [f"W{i}" for i in range(n_splines)]
        + ["alpha_value", f"alpha_{channel_name}"]
    )

    return {
        "estimates": dict(zip(names, est.tolist())),
        "se": dict(zip(names, se_nat.tolist())),
        "vcov": fit["vcov"].tolist(),
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
