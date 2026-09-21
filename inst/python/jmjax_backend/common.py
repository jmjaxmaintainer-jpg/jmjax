"""
Shared machinery for MLE joint-model fitting via adaptive Gauss-Hermite
quadrature over a scalar random intercept.

This generalizes the prototyping scripts: instead of hardcoded
`beta0 + beta1*t`, the fixed-effects contribution is `X(t) @ beta` for an
arbitrary design matrix X(t), and the baseline hazard is a pluggable
function `log_h0_fn(t, baseline_params)` - Weibull and spline models both
just supply a different `log_h0_fn` and reuse everything else here.

Random-effects scope: 1D random intercept only (see package design notes -
this is a deliberate v1 limit; the MCMC backend in mcmc_model.py supports
random intercept + slope since NUTS doesn't have the same quadrature-
dimensionality problem).
"""
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import norm

N_GH_DEFAULT = 15
N_NEWTON_STEPS_DEFAULT = 8


def guard_initial_theta(neg_log_lik, data, theta_prefit, theta_default):
    """Reject a pre-fit starting point that gives a non-finite objective.

    The starting values derived from an `lme()`/`coxph()` pre-fit are
    normally a large improvement on the hard-coded defaults, but they are
    only as good as the pre-fit. A badly wrong one can put the optimizer
    somewhere `exp(alpha * m_i(t))` overflows, and the resulting non-finite
    gradient stops L-BFGS-B almost immediately - producing a confidently
    WRONG answer rather than an error.

    Observed with a deliberately absurd pre-fit (fixed effects around 70
    where the data supports 2): the fit terminated after 6 iterations with
    alpha = -0.0016 against a true 0.5, and beta = 2.82 against a true 0.3.

    One extra likelihood evaluation converts that silent failure into a
    correct fit from the defaults. Only NON-FINITE values trigger the
    fallback: a merely higher objective at the start says little about
    which basin the optimizer will reach, so using it to choose would be
    an unmeasured heuristic rather than a safeguard.
    """
    def _probe(th):
        """Objective AND gradient - L-BFGS-B needs both to be finite."""
        try:
            t = jnp.asarray(th, dtype=jnp.float32)
            v = float(neg_log_lik(t, data))
            g = np.asarray(jax.grad(lambda z: neg_log_lik(z, data))(t))
            return v, g
        except Exception:
            return float("nan"), np.array([np.nan])

    v_pre, g_pre = _probe(theta_prefit)
    ok_pre = np.isfinite(v_pre) and np.all(np.isfinite(g_pre))
    if ok_pre:
        return theta_prefit

    # Checking the GRADIENT as well as the objective matters: on the
    # prothro data the objective evaluated finite while the gradient did
    # not, so an objective-only check passed and L-BFGS-B then terminated
    # ABNORMALLY at iteration 0 with the estimates left exactly at their
    # starting values - a silent non-fit reported as a result.
    bad = "gradient" if np.isfinite(v_pre) else "objective"
    idx = np.where(~np.isfinite(g_pre))[0] if g_pre.size > 1 else []
    where = f" (non-finite gradient entries at theta positions {list(idx)})" if len(idx) else ""

    v_def, g_def = _probe(theta_default)
    if np.isfinite(v_def) and np.all(np.isfinite(g_def)):
        warnings.warn(
            f"the pre-fit starting values give a non-finite {bad}{where}, so "
            "jmjax has fallen back to its default starting values. That "
            "usually means the pre-fit does not describe these data well - "
            "check it before trusting the result.",
            RuntimeWarning,
        )
        return theta_default

    # Neither start is usable. Say so plainly rather than returning a
    # vector the optimizer cannot move from, which would be reported as a
    # converged-looking fit whose estimates are just the starting values.
    warnings.warn(
        f"the {bad} is non-finite at BOTH the pre-fit and the default "
        f"starting values{where}, so the optimizer cannot take a step and "
        "the returned estimates will simply be the starting values. This is "
        "a failure, not a fit. Common causes: a longitudinal outcome on a "
        "scale that makes exp(alpha * m) overflow, a time variable whose "
        "scale disagrees between the longitudinal and survival data, or a "
        "baseline hazard whose defaults are far from these data. Check "
        "summary(lme_object) and the time ranges of both data sets.",
        RuntimeWarning,
    )
    return theta_default


def init_theta_from_prefit(n_theta, layout, control, defaults):
    """Assemble a starting vector from `lme()`/`coxph()` pre-fit pieces.

    jmjax's MLE paths historically started from hard-coded constants that
    ignored the data: beta_0 = 2.0 with every other beta at 0, sigma_e =
    0.5, sigma_b = 0.5, alpha = +0.5, all spline coefficients at 0. On the
    AIDS data that puts alpha roughly 1.0 from the optimum AND ON THE
    WRONG SIDE OF ZERO (the fitted value is about -0.48), while the
    available lme() fit already knows beta = (2.71, -0.40, -0.54),
    sigma_e = 0.40 and sigma_b = (0.83, 0.13).

    That matters far more for MLE than for MCMC. NUTS explores and
    recovers from a poor start; L-BFGS-B is a LOCAL optimizer and can
    stall or converge to a point where the Hessian is singular - which is
    exactly the failure seen on AIDS at q=2 with a spline baseline (NaN
    standard errors, in one configuration alongside converged = True).

    `layout` maps a piece name to its position in theta:
        scalar entries -> int index
        vector entries -> (start, stop) tuple
    Recognised names: "beta", "log_sigma_e", "log_sigma_b",
    "log_sigma_b0", "log_sigma_b1", "atanh_rho", "alpha", "gamma".
    Anything absent from `layout`, or not supplied in `control`, keeps the
    value already in `defaults` - so a partial pre-fit still helps, and
    the spline/Weibull baseline coefficients (which no pre-fit estimates)
    are simply left alone.
    """
    theta0 = np.array(defaults, dtype=float).copy()

    def _put(name, value, transform=None):
        if name not in layout or value is None:
            return
        v = np.atleast_1d(np.asarray(value, dtype=float))
        if not np.all(np.isfinite(v)):
            return
        if transform is not None:
            v = transform(v)
            if not np.all(np.isfinite(v)):
                return
        slot = layout[name]
        if isinstance(slot, tuple):
            lo, hi = slot
            if v.shape[0] != hi - lo:
                return          # length mismatch: leave the default alone
            theta0[lo:hi] = v
        else:
            theta0[slot] = v[0]

    _log = lambda v: np.log(np.maximum(v, 1e-8))

    _put("beta",         control.get("init_beta"))
    # Spline baseline-hazard coefficients, from a Weibull survreg projected
    # onto the basis on the R side - see jm_fit.R. Without this they start
    # at zero, i.e. a flat baseline hazard of exp(0) = 1.
    _put("W",            control.get("init_spline"))
    # Weibull baseline: log_lambda0 and log_shape, from a survreg() fit on
    # the R side. Previously fixed at -2.0 and log(1.2) whatever the data.
    _put("log_lambda0",  control.get("init_log_lambda0"))
    _put("log_shape",    control.get("init_log_shape"))
    _put("gamma",        control.get("init_gamma"))
    _put("alpha",        control.get("init_alpha"))
    _put("log_sigma_e",  control.get("init_sigma_e"),  _log)

    sb = control.get("init_sigma_b")
    if sb is not None:
        sb = np.atleast_1d(np.asarray(sb, dtype=float))
        _put("log_sigma_b",  sb[:1], _log)                     # q=1
        if sb.shape[0] >= 1:
            _put("log_sigma_b0", sb[:1], _log)                 # q=2
        if sb.shape[0] >= 2:
            _put("log_sigma_b1", sb[1:2], _log)

    rho = control.get("init_rho")
    if rho is not None:
        # atanh is undefined at +/-1; clip well inside the boundary
        _put("atanh_rho", rho, lambda v: np.arctanh(np.clip(v, -0.95, 0.95)))

    return theta0



# ==============================================================================
# Cholesky-first Newton step for q=2 mode-finding (EXPERIMENTAL, opt-in).
#
# Background: the q=2 adaptive-GH path finds each subject's random-effects
# mode via Newton iteration, guarding against a non-negative-definite
# Hessian by running a FULL eigendecomposition (jnp.linalg.eigh), clipping
# eigenvalues, and reconstructing - on EVERY Newton step (20 by default),
# for every subject. Profiling-by-elimination during development narrowed
# q=2's cost to this per-step work specifically: a configuration
# experiment showed the tensor-product quadrature grid size is nearly
# irrelevant to runtime (25 vs 81 nodes made no consistent difference,
# since vmap batches them), while reducing Newton steps 20->12 gave a real
# ~1.4x speedup - implicating the per-step cost, not the node count.
#
# The idea: for a well-behaved log-likelihood near the mode, the Hessian
# is ALREADY negative-definite, so the eigendecomposition usually just
# confirms what's already true at significant cost. A Cholesky
# factorization of -H is much cheaper AND its success is itself proof of
# negative-definiteness - so try that first and fall back to the
# eigendecomposition only when it fails.
#
# IMPORTANT - why this is opt-in and not the default: under jax.jit (which
# this codebase relies on throughout), Python-level try/except around a
# traced operation does NOT work - JAX must trace BOTH branches
# unconditionally, so a naive "try Cholesky, except: eigh" would end up
# computing both every step and be SLOWER, not faster. The implementation
# below instead uses jnp.where on a NaN check, which is jit-compatible,
# but this means the eigendecomposition IS still computed on every step -
# the saving comes only from using the cheaper Cholesky RESULT when it's
# valid, not from skipping the eigh work.
#
# Whether that is actually faster in practice is an EMPIRICAL question
# this helper exists to answer, not a guarantee. If it turns out not to
# help (quite possible, given the above), the honest conclusion is that
# the eigendecomposition cost is unavoidable under jit without a more
# invasive restructuring (e.g. lax.cond, or dropping the safeguard
# entirely for well-conditioned problems), and the default path stays.
# ==============================================================================
def newton_step_cholesky_first(H, g, eps=1e-6):
    """Return the Newton step `delta` solving H_safe @ delta = g, using a
    Cholesky factorization of -H when it succeeds and falling back to the
    eigenvalue-clipping path otherwise. jit-compatible (no try/except)."""
    H_sym = 0.5 * (H + H.T)

    # Cholesky of -H: succeeds (no NaNs) iff -H is positive-definite,
    # i.e. iff H is negative-definite - exactly the condition the
    # eigenvalue clipping is guarding against.
    neg_H = -H_sym
    L_chol = jnp.linalg.cholesky(neg_H)
    chol_ok = jnp.all(jnp.isfinite(L_chol))

    # Cholesky path: solve -H delta = -g  =>  H delta = g
    delta_chol = jax.scipy.linalg.cho_solve((L_chol, True), -g)

    # Eigen fallback path (the existing, validated behavior)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    eigvals_safe = jnp.minimum(eigvals, -eps)
    H_safe = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T
    delta_eigh = jnp.linalg.solve(H_safe, g)

    # jnp.where (not if/else) so this traces cleanly under jit. Note this
    # means BOTH branches are computed - see the module comment above for
    # why, and why this may therefore not actually be faster.
    return jnp.where(chol_ok, delta_chol, delta_eigh)


def make_theta_layout(p, n_baseline_params):
    """theta = [beta_0..beta_{p-1}, log_sigma_e, log_sigma_b,
                baseline_0..baseline_{m-1}, alpha]"""
    idx = {
        "beta": (0, p),
        "log_sigma_e": p,
        "log_sigma_b": p + 1,
        "baseline": (p + 2, p + 2 + n_baseline_params),
        "alpha": p + 2 + n_baseline_params,
    }
    n_theta = p + 2 + n_baseline_params + 1
    return idx, n_theta


def build_h_fn(log_h0_fn, idx):
    """
    log_h0_fn(t, baseline_params) -> log baseline hazard at time(s) t.
    Must broadcast elementwise (works for scalar T_i or a [n_quad] vector).
    """

    def h_fn(b, X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i,
             T_i, event_i, t_quad_i, gk_weights, theta):
        beta = theta[idx["beta"][0]:idx["beta"][1]]
        sigma_e = jnp.exp(theta[idx["log_sigma_e"]])
        sigma_b = jnp.exp(theta[idx["log_sigma_b"]])
        baseline_params = theta[idx["baseline"][0]:idx["baseline"][1]]
        alpha = theta[idx["alpha"]]

        # --- Longitudinal contribution ---
        mu = X_long_i @ beta + b  # [max_obs]
        log_p_y = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_e) - 0.5 * ((y_i - mu) / sigma_e) ** 2
        mask = jnp.arange(X_long_i.shape[0]) < n_i
        log_p_y_sum = jnp.sum(jnp.where(mask, log_p_y, 0.0))

        # --- Survival contribution: pluggable baseline, time-dependent alpha*m_i(t) ---
        m_T = X_time_surv_i @ beta + b
        log_h_T = log_h0_fn(T_i, baseline_params) + alpha * m_T

        m_quad = X_time_quad_i @ beta + b  # [n_quad]
        hazard_quad = jnp.exp(log_h0_fn(t_quad_i, baseline_params) + alpha * m_quad)
        cum_H = T_i * jnp.sum(gk_weights * hazard_quad)

        log_p_surv = event_i * log_h_T - cum_H

        # --- Prior on random intercept ---
        log_prior_b = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_b) - 0.5 * (b / sigma_b) ** 2

        return log_p_y_sum + log_p_surv + log_prior_b

    return h_fn


def build_neg_log_lik(log_h0_fn, idx, n_gh=N_GH_DEFAULT, n_newton_steps=N_NEWTON_STEPS_DEFAULT):
    h_fn = build_h_fn(log_h0_fn, idx)
    grad_b_fn = jax.grad(h_fn, argnums=0)
    hess_b_fn = jax.grad(grad_b_fn, argnums=0)

    gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(n_gh)
    gh_nodes = jnp.array(gh_nodes_np)
    gh_weights = jnp.array(gh_weights_np)

    def find_mode_and_tau(*args):
        # args: X_long_i, y_i, n_i, X_time_surv_i, X_time_quad_i, T_i, event_i,
        #       t_quad_i, gk_weights, theta
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
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None)
        )(
            data["X_long"], data["y_long"], data["n_obs"],
            data["X_time_surv"], data["X_time_quad"],
            data["T_surv"], data["event"], data["t_quad"],
            data["gk_weights"], theta
        )
        return -jnp.sum(all_ll)

    return neg_log_lik


def fit_theta(neg_log_lik, data, init_theta, maxiter=1000, ftol=1e-7,
              opt_method="BFGS", parscale=None):
    """Shared optimizer driver + Hessian-based SEs, used by both backends.

    WHY BFGS AND NOT L-BFGS-B. L-BFGS-B was the default and it failed on
    the prothro data with a spline baseline: 5 iterations, converged =
    True, zero non-finite standard errors, and 78 log-likelihood units
    left behind (-14083.14 against BFGS's -14004.80), returning an
    association parameter of -0.0008 where JM, BFGS, trust-constr and
    jmjax's own MCMC all agree near -0.04.

    Three reasons to prefer BFGS here:

      1. MEASURED. Across five dataset/method combinations, L-BFGS-B was
         short of the best log-likelihood twice and never won; BFGS won
         twice. trust-constr was never short but is slower and emits a
         delta_grad warning on every fit.
      2. THE PROBLEM IS SMALL AND DENSE. L-BFGS-B keeps a limited-memory
         approximation (scipy's default is 10 correction pairs) - a rank-10
         model of curvature for a problem with 19 parameters, 9 of them
         strongly coupled spline coefficients. BFGS keeps the full dense
         inverse Hessian. The tell: handed an excellent EM-derived start,
         L-BFGS-B stopped after 2 iterations at grad_max 6.65 while BFGS
         ran 56 to 2.18 - a deficient curvature model quitting early even
         from a good point.
      3. JM DOES THE SAME. `method = "BFGS"` appears in every one of its
         fitters - splinePHGH, piecewisePHGH, piecewiseAFTGH, flexCPH -
         and L-BFGS-B appears nowhere. Two independent implementations
         reaching for dense BFGS on the same problem class is worth more
         than our five comparisons alone.

    L-BFGS-B remains available via control$opt_method for anyone who wants
    it, and is the better choice if the parameter count ever grows enough
    that a dense Hessian approximation is the bottleneck.

    `parscale` mirrors what `optim()`'s argument of the same name does in
    R, which JM relies on:

        if (is.null(control$parscale)) control$parscale <- rep(0.01, length(thetas))
        optim(thetas, LogLik.weibullGH, Score.weibullGH, method = "BFGS",
              control = list(maxit = control$iter.qN, parscale = control$parscale))

    scipy has no equivalent, so it is implemented here as an explicit
    change of variables: optimise u = theta / s and map back. The
    objective is unchanged, only the geometry the line search sees.

    Why it matters. jmjax optimises log_sigma_e, log_sigma_b and
    atanh_rho, so d/d(log sigma_b) carries a factor of sigma_b. On the
    prothro data sigma_b is about 18, making that gradient component an
    order of magnitude larger than the rest; a line search sized for it
    overshoots everything else, and L-BFGS-B terminated ABNORMALLY at
    iteration 0 with a finite objective AND a finite gradient.

    NOTE this is only part of what JM does. It also runs an EM phase
    before any quasi-Newton step, which cannot fail a line search the way
    this can. That is not reproduced here.
    """
    maxiter = int(maxiter)  # guard against an R double leaking through control$maxiter
    val_and_grad_fn = jax.jit(lambda th: jax.value_and_grad(neg_log_lik)(th, data))
    hessian_fn = jax.jit(lambda th: jax.hessian(neg_log_lik)(th, data))

    x0 = np.asarray(init_theta, dtype=np.float64)

    if parscale is None:
        scale = np.ones_like(x0)
    else:
        scale = np.atleast_1d(np.asarray(parscale, dtype=np.float64))
        if scale.size == 1:
            scale = np.full_like(x0, float(scale[0]))
        if scale.size != x0.size or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            warnings.warn(
                f"parscale must be positive, finite and of length {x0.size} "
                f"(got size {scale.size}); ignoring it.", RuntimeWarning)
            scale = np.ones_like(x0)

    def scipy_objective(u):
        theta_np = u * scale
        val, grad = val_and_grad_fn(jnp.array(theta_np))
        # chain rule: d/du = (d/dtheta) * scale
        return float(val), np.array(grad, dtype=np.float64) * scale

    opts = {"maxiter": maxiter}
    if opt_method in ("L-BFGS-B", "TNC"):
        opts["ftol"] = ftol           # BFGS/CG use `gtol` and reject `ftol`

    res = minimize(
        fun=scipy_objective,
        x0=x0 / scale,
        method=opt_method,
        jac=True,
        options=opts
    )

    theta_opt = res.x * scale

    # ---- Final gradient norm ------------------------------------------
    # At a stationary point the gradient is ~0. A failed line search is
    # not, and scipy reports success anyway - so the usual health signals
    # say nothing.
    #
    # Measured on prothro with a spline baseline: L-BFGS-B stopped after 5
    # iterations with converged = True, ZERO non-finite standard errors,
    # and a finite log-likelihood. It had left 78 log-likelihood units on
    # the table (-14083.14 against BFGS's -14004.80) and returned an
    # association parameter of -0.0008 where JM, BFGS and jmjax's own MCMC
    # all agree near -0.04. Nothing in the output flagged it.
    #
    # Reported in the ORIGINAL parameterisation (the chain rule divides by
    # `scale`), so the number means the same thing whatever parscale is
    # set to.
    try:
        _v, _g = val_and_grad_fn(jnp.array(theta_opt))
        _g = np.asarray(_g, dtype=np.float64)
        grad_norm = float(np.sqrt(np.sum(_g ** 2)))
        grad_max = float(np.max(np.abs(_g)))
    except Exception:
        grad_norm = grad_max = float("nan")
    H = np.array(hessian_fn(jnp.array(theta_opt)), dtype=np.float64)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))

    return {
        "theta_opt": theta_opt,
        "se_theta": se,
        "vcov": cov,
        "loglik": float(-res.fun),
        "grad_norm": grad_norm,
        "grad_max": grad_max,
        "optimizer_success": bool(res.success),
        # `converged` is a GRADIENT criterion, not scipy's exit code.
        #
        # scipy's `success` cannot be trusted either way here, and it fails
        # in BOTH directions on the same data:
        #
        #   prothro + spline, L-BFGS-B : success TRUE  at grad_max 31.9,
        #       having stopped after 5 iterations 78 log-likelihood units
        #       short with an association parameter of -0.0008 where four
        #       independent routes agree near -0.04. It met `ftol` - the
        #       relative function change was tiny because the line search
        #       had quit - and reported convergence.
        #   the same data, BFGS        : success FALSE at grad_max 2.29,
        #       at the good optimum. It stops on `gtol`, an ABSOLUTE
        #       gradient norm defaulting to 1e-5, which on a
        #       log-likelihood of 14000 is a bar no sensible fit clears.
        #
        # So a disjunction (success OR small gradient) passes the failure,
        # and a conjunction (success AND small gradient) flags the good
        # BFGS fit. Neither works, because `success` carries no consistent
        # meaning across methods - it reports which stopping rule fired,
        # not whether an optimum was reached.
        #
        # The gradient does carry that meaning. A stationary point has a
        # small one; a quit line search does not. Scaled to the
        # log-likelihood it separates these cases cleanly: 31.9 and 17.4
        # for the two failing fits against a threshold near 14, versus
        # 2.29 and 8.0 for the two good ones.
        #
        # An earlier version reported `success` verbatim, which turned
        # nine tests red when the default became BFGS - every failure the
        # FLAG, with the estimates still recovering truth.
        #
        # scipy's own status stays available as `optimizer_success` for
        # anyone who wants it.
        # With an EM phase, JM's own criterion is available and is used:
        #
        #     if ((conv <- out$convergence) == 0 || -out[[2]] > lgLik) {
        #
        # the optimizer succeeded OR it improved on what EM reached. That
        # asks whether the fit beat something REAL, rather than whether a
        # threshold calibrated on one dataset was met.
        #
        # Without an EM phase that clause has no referent - there is no
        # prior log-likelihood to beat - so the gradient rule applies. It
        # is the weaker of the two, which is a reason to run the EM phase
        # rather than a reason to trust the threshold.
        # A GRADIENT criterion, not scipy's exit code.
        #
        # scipy's `success` means different things per method and is wrong
        # in both directions on the same data:
        #   prothro + spline, L-BFGS-B : success TRUE at grad_max 31.9,
        #       stopped 78 log-likelihood units short with an association
        #       parameter of -0.0008 where four independent routes agree
        #       near -0.04. It met `ftol` because the line search had quit.
        #   the same data, BFGS        : success FALSE at grad_max 2.29, at
        #       the good optimum. `gtol` defaults to 1e-5, a bar nothing on
        #       a log-likelihood of 14000 clears.
        #
        # A stationary point has a small gradient; a quit line search does
        # not. Scaled to the log-likelihood this separates the cases
        # cleanly: 31.9 and 17.4 for the two failing fits against a
        # threshold near 14, versus 2.29 and 8.0 for the good ones.
        #
        # An EM phase would allow JM's own rule as well - "the optimizer
        # succeeded OR it improved on what EM reached" - but jmjax does not
        # run one by default. See inst/research/em/ for that work, which is
        # kept as reference rather than wired in: with BFGS as the default
        # there is no known case where jmjax's MLE fails and JM's succeeds.
        "converged": bool(
            np.isfinite(grad_max)
            and grad_max <= max(1e-3, 1e-3 * abs(float(res.fun)))
        ),
        "message": str(res.message),
        "n_iter": int(res.nit),
    }


def to_jax_data(**arrays):
    return {k: jnp.array(v) for k, v in arrays.items()}
