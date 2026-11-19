"""Joint models with several longitudinal markers (jmjax 0.5.0, MCMC only).

Model, for markers k = 1..K and subjects i = 1..N:

    y_ijk = x_ijk' beta_k + z_ijk' b_ik + e_ijk,   e_ijk ~ N(0, sigma_e[k]^2)
    b_i = (b_i1, ..., b_iK) ~ N(0, Sigma),          Sigma = diag(s) R diag(s)
    h_i(t) = h0(t) exp{gamma' w_i + sum_c alpha_c m_ic(t)}

Each hazard channel c belongs to one marker and carries one functional form
of its trajectory (value, delta, area, area_avg): m_ic(t) = x_c(t)' beta +
z_c(t)' b_i, with x_c, z_c nonzero only in that marker's block. The default
covariance is full (an LKJ prior on the Q x Q correlation, which is what links
the markers); control$re_structure = "block" makes it block-diagonal over
markers (independent markers).

DATA LAYOUT. The longitudinal data are flat (one row per observation of any
marker): y [n], X [n, P] and Z [n, Q] with each row nonzero only in its
marker's block of columns, subj [n] and marker [n] indices. The hazard
channels are stacked: XS [C, N, P], ZS [C, N, Q] at the event times and
XQ [C, N, n_quad, P], ZQ [C, N, n_quad, Q] at the quadrature nodes.

ROTATION. The fixed and random effects are block-diagonal over markers with
beta stacked, so the joint absorbable construction (absorbable.py) applies
unchanged: M stacks the rows of every marker, and (T) is imposed on every
hazard channel. ONE Q spans the union over markers. With a full cross-marker
covariance the standardization (L')^{-1} mixes the columns of different
markers, so a shift only marker k can absorb appears in the other markers'
standardized columns too; Lemma 1 needs it in span(Q) there, which holds
only for the union. (Per-marker Q_k would also be valid for the
block-diagonal covariance, with a smaller dense block; the union is used in
both cases, which is valid for any L.)

SCALES. R fits the model on rescaled data (time / c, standardized
covariates) and passes the exact linear maps back to the user's scale:
beta_orig = A beta (A = beta_transform, P x P) and b_orig = B b
(B = re_transform, Q x Q, block-diagonal), so Sigma_orig = B Sigma B'. Every
reported draw is transformed before summaries and diagnostics.

PARTIAL CENTERING (control$center_absorbable = TRUE; experimental, 0.5.1).
The rotated model samples b_std = H [U; V] with U [k, Q], V [N - k, Q] iid
N(0, 1) and b = b_std L'. In (beta, U) a shift d of the fixed effects is
offset by U -> U + Q' Delta(d) (L')^{-1}, which moves with the sampled L,
so a dense metric adapted at one L mismatches as L moves. Centering only
the absorbable block samples W = U L' instead (rows ~ N(0, L L')), keeping
V non-centred: b = H [W; V L'] = Q W + Q_perp V L'. The ridge in
(beta, W) is W -> W + Q' Delta(d), free of L, so one fixed dense block over
(beta, W) fits it at every L. The prior is written on W directly (a
multivariate normal with scale_tril L), so no Jacobian term is needed; the
funnel this brings back involves only the k rows of W. Same model, same
posterior of every model quantity; only the coordinates NUTS moves in
change. Sample site: b_gen_W; dense block ("beta", "b_gen_W").
"""
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from numpyro.diagnostics import summary as numpyro_summary

from .absorbable import (_absorbable_basis_multi, _householder_reflectors,
                         _apply_reflectors)

RHAT_CONVERGED_THRESHOLD = 1.05


def _blockdiag_corr(L_blocks, sizes):
    """Assemble a block-diagonal Cholesky factor from per-block factors."""
    Q = int(sum(sizes))
    L = jnp.zeros((Q, Q))
    start = 0
    for Lb, s in zip(L_blocks, sizes):
        L = L.at[start:start + s, start:start + s].set(Lb)
        start += s
    return L


def build_model_multi(P, Q, n_splines, block_sizes, baseline_hazard="spline",
                      spline_prior="independent", re_structure="full",
                      beta_prior_mean=None, beta_prior_sd=None,
                      sigma_e_prior_mean=None, sigma_e_prior_shape=5.0,
                      sigma_b_prior_mean=None, sigma_b_prior_shape=5.0,
                      lkj_concentration=3.0, alpha_prior_sd=2.0,
                      gamma_prior_sd=2.0, spline_penalty_shape=5.0,
                      spline_penalty_rate=0.5, gen_reflectors=None,
                      center_absorbable=False):
    """Return the NumPyro model for K = len(block_sizes) markers."""
    K = len(block_sizes)

    def model(y, X, Z, subj, marker, XS, ZS, XQ, ZQ, T_surv, event, t_quad,
              gk_weights, B_T, B_quad, N_sub, W_surv=None):
        C = XS.shape[0]
        # ---- fixed effects, residual SDs, association -----------------
        if beta_prior_mean is not None:
            beta = numpyro.sample("beta", dist.Normal(jnp.asarray(beta_prior_mean),
                                                      jnp.asarray(beta_prior_sd)))
        else:
            beta = numpyro.sample("beta", dist.Normal(0.0, 5.0).expand([P]))
        if sigma_e_prior_mean is not None:
            _m = jnp.maximum(jnp.asarray(sigma_e_prior_mean), 1e-8)
            sigma_e = numpyro.sample("sigma_e", dist.Gamma(sigma_e_prior_shape,
                                                           sigma_e_prior_shape / _m))
        else:
            sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(2.0).expand([K]))
        alpha = numpyro.sample("alpha", dist.Normal(0.0, alpha_prior_sd).expand([C]))

        gamma_term = 0.0
        if W_surv is not None:
            gamma = numpyro.sample("gamma", dist.Normal(0.0, gamma_prior_sd)
                                   .expand([W_surv.shape[-1]]))
            gamma_term = W_surv @ gamma

        # ---- baseline hazard (same parameterizations as mcmc_model) ---
        if baseline_hazard == "weibull":
            log_lambda0 = numpyro.sample("log_lambda0", dist.Normal(-2.0, 2.0))
            log_shape = numpyro.sample("log_shape", dist.Normal(0.0, 1.0))
            shape = numpyro.deterministic("shape", jnp.exp(log_shape))
            log_h0_T = jnp.log(shape) + (shape - 1.0) * jnp.log(T_surv) + log_lambda0
            log_h0_quad = jnp.log(shape) + (shape - 1.0) * jnp.log(t_quad) + log_lambda0
        else:
            if spline_prior == "penalized":
                tau_w = numpyro.sample("tau_w", dist.Gamma(spline_penalty_shape,
                                                           spline_penalty_rate))
                sigma_w = 1.0 / jnp.sqrt(tau_w)
                W01 = numpyro.sample("W01", dist.Normal(0.0, 10.0).expand([2]))
                n_rest = n_splines - 2
                z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0).expand([n_rest]))
                s2 = jnp.cumsum(jnp.cumsum(z_step))
                k_off = jnp.arange(1, n_rest + 1)
                w_rest = W01[1] + k_off * (W01[1] - W01[0]) + sigma_w * s2
                W = numpyro.deterministic("W", jnp.concatenate([W01, w_rest]))
            else:
                W = numpyro.sample("W", dist.Normal(0.0, 3.0).expand([n_splines]))
            log_h0_T = B_T @ W
            log_h0_quad = jnp.einsum("nks,s->nk", B_quad, W)

        # ---- random effects -------------------------------------------
        if sigma_b_prior_mean is not None:
            _sm = jnp.maximum(jnp.asarray(sigma_b_prior_mean), 1e-8)
            sigma_b = numpyro.sample("sigma_b", dist.Gamma(sigma_b_prior_shape,
                                                           sigma_b_prior_shape / _sm))
        else:
            sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(2.0).expand([Q]))
        if re_structure == "block":
            L_blocks = []
            for k, s in enumerate(block_sizes):
                if s >= 2:
                    L_blocks.append(numpyro.sample(
                        "L_corr_%d" % k, dist.LKJCholesky(s, concentration=lkj_concentration)))
                else:
                    L_blocks.append(jnp.ones((1, 1)))
            L_corr = numpyro.deterministic("L_corr", _blockdiag_corr(L_blocks, block_sizes))
        else:
            L_corr = numpyro.sample("L_corr", dist.LKJCholesky(Q, concentration=lkj_concentration))
        L = sigma_b[:, None] * L_corr
        if gen_reflectors is not None and center_absorbable:
            # b = H [W; V L'], W rows ~ N(0, L L') (centred), V iid N(0, 1)
            Vh = jnp.asarray(gen_reflectors)
            kg = int(gen_reflectors.shape[1])
            Wc = numpyro.sample("b_gen_W", dist.MultivariateNormal(
                jnp.zeros(Q), scale_tril=L).expand([kg]).to_event(1))
            Vr = numpyro.sample("b_gen_V", dist.Normal(0.0, 1.0)
                                .expand([int(gen_reflectors.shape[0]) - kg, Q]).to_event(2))
            b = numpyro.deterministic("b", _apply_reflectors(
                Vh, jnp.concatenate([Wc, Vr @ L.T], axis=0)))
        elif gen_reflectors is not None:
            Vh = jnp.asarray(gen_reflectors)
            kg = int(gen_reflectors.shape[1])
            U = numpyro.sample("b_gen_U", dist.Normal(0.0, 1.0).expand([kg, Q]).to_event(2))
            Vr = numpyro.sample("b_gen_V", dist.Normal(0.0, 1.0)
                                .expand([int(gen_reflectors.shape[0]) - kg, Q]).to_event(2))
            b_std = _apply_reflectors(Vh, jnp.concatenate([U, Vr], axis=0))
            b = numpyro.deterministic("b", b_std @ L.T)
        else:
            b_std = numpyro.sample("b_std", dist.Normal(0.0, 1.0).expand([N_sub, Q]).to_event(2))
            b = numpyro.deterministic("b", b_std @ L.T)

        # ---- longitudinal likelihood (flat over markers) --------------
        mu = X @ beta + jnp.sum(Z * b[subj], axis=-1)
        numpyro.factor("y_obs", jnp.sum(dist.Normal(mu, sigma_e[marker]).log_prob(y)))

        # ---- survival likelihood --------------------------------------
        m_T = jnp.einsum("cnp,p->cn", XS, beta) + jnp.einsum("cnq,nq->cn", ZS, b)
        m_Q = jnp.einsum("cnkp,p->cnk", XQ, beta) + jnp.einsum("cnkq,nq->cnk", ZQ, b)
        eta_T = gamma_term + jnp.einsum("c,cn->n", alpha, m_T)
        eta_Q = (gamma_term[:, None] if W_surv is not None else 0.0) \
            + jnp.einsum("c,cnk->nk", alpha, m_Q)
        log_hazard = log_h0_T + eta_T
        cum_hazard = T_surv * jnp.sum(jnp.exp(log_h0_quad + eta_Q) * gk_weights[None, :], axis=1)
        numpyro.factor("surv_log_prob", jnp.sum(event * log_hazard - cum_hazard))

    return model


def _transform_draws(samples, A, B, block_mode, block_sizes):
    """Map internal-scale draws to the user's scale and derive rho_i_j."""
    out = {}
    beta = np.asarray(samples["beta"]) @ A.T                       # [S, P]
    b = np.einsum("snq,rq->snr", np.asarray(samples["b"]), B)      # [S, N, Q]
    s = np.asarray(samples["sigma_b"]); Lc = np.asarray(samples["L_corr"])
    L = s[:, :, None] * Lc
    Sig = np.einsum("ij,sjk,slk,ml->sim", B, L, L, B)              # B L L' B'
    sd = np.sqrt(np.einsum("sii->si", Sig))
    R = Sig / (sd[:, :, None] * sd[:, None, :])
    out["beta"] = beta; out["b"] = b; out["sigma_b"] = sd; out["R"] = R
    Q = sd.shape[1]
    block_of = np.repeat(np.arange(len(block_sizes)), block_sizes)
    pairs = [(i, j) for i in range(Q) for j in range(i + 1, Q)
             if not block_mode or block_of[i] == block_of[j]]
    out["pairs"] = pairs
    return out


def fit_nuts_multi(y, X, Z, subj, marker, XS, ZS, XQ, ZQ, T_surv, event,
                   t_quad, gk_weights, B_T, B_quad, n_splines, block_sizes,
                   beta_names, sigma_e_names, alpha_names,
                   beta_transform=None, re_transform=None, W_surv=None,
                   control=None):
    """Fit the multi-marker joint model by NUTS; same result layout as
    mcmc_model.fit_nuts() (estimates, se, diagnostics, posterior_samples,
    random_effects, convergence), with every population parameter also
    stored under its own name in posterior_samples."""
    control = control or {}
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    subj = np.asarray(subj, dtype=int).reshape(-1)
    marker = np.asarray(marker, dtype=int).reshape(-1)
    XS = np.asarray(XS, dtype=float); ZS = np.asarray(ZS, dtype=float)
    XQ = np.asarray(XQ, dtype=float); ZQ = np.asarray(ZQ, dtype=float)
    block_sizes = [int(v) for v in np.atleast_1d(block_sizes)]
    # str(): numpy string scalars as dict keys do not convert back to R names
    beta_names = [str(v) for v in np.atleast_1d(beta_names)]
    sigma_e_names = [str(v) for v in np.atleast_1d(sigma_e_names)]
    alpha_names = [str(v) for v in np.atleast_1d(alpha_names)]
    N_sub = int(np.asarray(T_surv).shape[0])
    P, Q, C = X.shape[1], Z.shape[1], XS.shape[0]
    K = len(block_sizes)
    if sum(block_sizes) != Q:
        raise ValueError("block_sizes must sum to ncol(Z)")
    A = np.eye(P) if beta_transform is None else np.asarray(beta_transform, float).reshape(P, P)
    B = np.eye(Q) if re_transform is None else np.asarray(re_transform, float).reshape(Q, Q)
    re_structure = str(control.get("re_structure", "full"))
    if re_structure not in ("full", "block"):
        raise ValueError("control$re_structure must be 'full' or 'block'")
    baseline_hazard = str(control.get("baseline_hazard", "spline"))
    spline_prior = str(control.get("spline_prior", "independent"))
    center_absorbable = bool(control.get("center_absorbable", False))
    if center_absorbable and control.get("rotate_absorbable", True) is False:
        raise ValueError("control$center_absorbable = TRUE needs the rotation "
                         "(rotate_absorbable = FALSE was given)")

    # ---- rotation: one Q for the union over markers ------------------
    gen_reflectors = None
    rot_report = None
    if control.get("rotate_absorbable", True) is not False:
        Qb, info = _absorbable_basis_multi(X, Z, subj, N_sub, XS, ZS, XQ, ZQ)
        if info["condition_s_residual"] > 1e-6:
            warnings.warn(
                "rotate_absorbable: Condition (S) fails for %d of the %d "
                "candidate direction(s) (worst relative residual %.2e); they "
                "were left out of the rotation. The fit is exact either way."
                % (info["n_rejected_s"], info["dim_ker_M"],
                   info["condition_s_residual"]), RuntimeWarning, stacklevel=2)
        if Qb is not None:
            gen_reflectors = _householder_reflectors(Qb)
        rot_report = {
            "mode": "all_columns_no_sweep", "construction": "joint_multi_marker",
            "applied": gen_reflectors is not None,
            "k": 0 if gen_reflectors is None else int(gen_reflectors.shape[1]),
            "dim_ker_M": int(info["dim_ker_M"]),
            "condition_s_residual": float(info["condition_s_residual"]),
            "n_rejected_condition_s": int(info["n_rejected_s"]),
            "dense_block_size": None,
            "center_absorbable": bool(center_absorbable and gen_reflectors is not None),
        }

    def _opt(name):
        v = control.get(name)
        return None if v is None else np.atleast_1d(np.asarray(v, dtype=float))

    model = build_model_multi(
        P, Q, int(n_splines), block_sizes, baseline_hazard=baseline_hazard,
        spline_prior=spline_prior, re_structure=re_structure,
        beta_prior_mean=_opt("beta_prior_mean"), beta_prior_sd=_opt("beta_prior_sd"),
        sigma_e_prior_mean=_opt("sigma_e_prior_mean"),
        sigma_e_prior_shape=float(control.get("sigma_e_prior_shape", 5.0)),
        sigma_b_prior_mean=_opt("sigma_b_prior_mean"),
        sigma_b_prior_shape=float(control.get("sigma_b_prior_shape", 5.0)),
        lkj_concentration=float(control.get("lkj_concentration", 3.0)),
        alpha_prior_sd=float(control.get("alpha_prior_sd", 2.0)),
        gamma_prior_sd=float(control.get("gamma_prior_sd", 2.0)),
        spline_penalty_shape=float(control.get("spline_penalty_shape", 5.0)),
        spline_penalty_rate=float(control.get("spline_penalty_rate", 0.5)),
        gen_reflectors=gen_reflectors, center_absorbable=center_absorbable)

    dense_blocks = []
    if gen_reflectors is not None and control.get("dense_mass_generator_beta", True) is not False:
        dense_blocks.append(("beta", "b_gen_W" if center_absorbable else "b_gen_U"))
        if rot_report is not None:
            rot_report["dense_block_size"] = int(P + gen_reflectors.shape[1] * Q)
    if baseline_hazard == "spline" and spline_prior == "penalized" and \
            control.get("dense_mass_spline", True) is not False:
        dense_blocks.append(("W01", "z_step"))
    nuts_kwargs = {"init_strategy": init_to_median(num_samples=15)}
    if dense_blocks:
        nuts_kwargs["dense_mass"] = dense_blocks
    if control.get("target_accept_prob") is not None:
        nuts_kwargs["target_accept_prob"] = float(control["target_accept_prob"])
    if control.get("max_tree_depth") is not None:
        nuts_kwargs["max_tree_depth"] = int(control["max_tree_depth"])

    model_kwargs = dict(
        y=jnp.asarray(y), X=jnp.asarray(X), Z=jnp.asarray(Z),
        subj=jnp.asarray(subj), marker=jnp.asarray(marker),
        XS=jnp.asarray(XS), ZS=jnp.asarray(ZS), XQ=jnp.asarray(XQ), ZQ=jnp.asarray(ZQ),
        T_surv=jnp.asarray(np.asarray(T_surv, float)), event=jnp.asarray(np.asarray(event, float)),
        t_quad=jnp.asarray(np.asarray(t_quad, float)),
        gk_weights=jnp.asarray(np.asarray(gk_weights, float).reshape(-1)),
        B_T=jnp.asarray(np.asarray(B_T, float)), B_quad=jnp.asarray(np.asarray(B_quad, float)),
        N_sub=N_sub,
        W_surv=None if W_surv is None else jnp.asarray(np.asarray(W_surv, float)))

    mcmc = MCMC(NUTS(model, **nuts_kwargs),
                num_warmup=int(control.get("num_warmup", 500)),
                num_samples=int(control.get("num_samples", 1000)),
                num_chains=int(control.get("num_chains", 1)),
                progress_bar=bool(control.get("progress_bar", False)))
    start = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(control.get("seed", 2026))),
             extra_fields=("num_steps", "diverging"), **model_kwargs)
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - start

    try:
        xf = mcmc.get_extra_fields()
        steps = np.asarray(xf["num_steps"]); div = np.asarray(xf["diverging"])
        mean_steps, max_steps = float(steps.mean()), float(steps.max())
        n_div, div_rate = int(div.sum()), float(div.mean())
    except Exception:
        mean_steps = max_steps = None
        n_div = div_rate = None

    nc = int(control.get("num_chains", 1))
    raw = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=False).items()}
    block_mode = re_structure == "block"
    tr = _transform_draws(raw, A, B, block_mode, block_sizes)

    # ---- every population parameter under its own name ---------------
    named = {}
    for j, nm in enumerate(beta_names):
        named[nm] = tr["beta"][:, j]
    se_arr = np.asarray(raw["sigma_e"]).reshape(len(raw["sigma_e"]), -1)
    for k, nm in enumerate(sigma_e_names):
        named[nm] = se_arr[:, k]
    al = np.asarray(raw["alpha"]).reshape(len(raw["alpha"]), -1)
    for c, nm in enumerate(alpha_names):
        named[nm] = al[:, c]
    if "gamma" in raw:
        g = np.asarray(raw["gamma"]).reshape(len(raw["gamma"]), -1)
        for j in range(g.shape[1]):
            named["gamma_%d" % j] = g[:, j]
    if "W" in raw:
        Wd = np.asarray(raw["W"]).reshape(len(raw["W"]), -1)
        for j in range(Wd.shape[1]):
            named["W%d" % j] = Wd[:, j]
    for nm in ("tau_w", "log_lambda0", "shape"):
        if nm in raw:
            named[nm] = np.asarray(raw[nm]).reshape(-1)
    for i in range(Q):
        named["sigma_b%d" % i] = tr["sigma_b"][:, i]
    for (i, j) in tr["pairs"]:
        named["rho_%d_%d" % (i, j)] = tr["R"][:, i, j]

    estimates, se, rhat, ess = {}, {}, {}, {}
    for nm, d in named.items():
        d = np.asarray(d, dtype=float)
        estimates[nm] = float(d.mean()); se[nm] = float(d.std())
        n_total = d.shape[0]
        ncu = nc if n_total % max(nc, 1) == 0 else 1
        try:
            sm = numpyro_summary({"x": d.reshape(ncu, n_total // ncu)}, group_by_chain=True)["x"]
            rhat[nm] = float(np.asarray(sm["r_hat"]).reshape(-1)[0])
            ess[nm] = float(np.asarray(sm["n_eff"]).reshape(-1)[0])
        except Exception:
            rhat[nm] = ess[nm] = float("nan")

    random_effects = {"subject_index": list(range(1, N_sub + 1))}
    bm, bs = tr["b"].mean(axis=0), tr["b"].std(axis=0)
    for j in range(Q):
        random_effects["b%d_mean" % j] = bm[:, j].tolist()
        random_effects["b%d_sd" % j] = bs[:, j].tolist()

    vals = [v for v in rhat.values() if np.isfinite(v)]
    max_rhat = max(vals) if vals else float("nan")
    converged = bool(max_rhat < RHAT_CONVERGED_THRESHOLD) if vals else True
    message = ("NUTS completed in %.2fs; max split R-hat across population "
               "parameters = %.4f" % (elapsed, max_rhat))
    if not converged:
        message += (" (>= %.2f threshold - consider more warmup/samples)"
                    % RHAT_CONVERGED_THRESHOLD)

    posterior_samples = {nm: np.asarray(v).tolist() for nm, v in named.items()}
    posterior_samples["beta"] = tr["beta"].tolist()
    posterior_samples["sigma_b"] = tr["sigma_b"].tolist()
    # The per-subject draws [S, N, Q] are the bulk of the result; long
    # benchmark runs drop them (control$return_b_draws = FALSE).
    if control.get("return_b_draws", True) is not False:
        posterior_samples["b"] = tr["b"].tolist()
    posterior_samples["corr"] = tr["R"].tolist()
    for _g in ("b_gen_U", "b_gen_W"):
        if _g in raw:
            posterior_samples[_g] = np.asarray(raw[_g]).tolist()

    return {
        "estimates": estimates, "se": se, "vcov": None, "loglik": None,
        "convergence": {"converged": converged, "message": message, "n_iter": None,
                        "sampling_time_sec": elapsed, "mean_num_steps": mean_steps,
                        "max_num_steps": max_steps, "n_divergences": n_div,
                        "divergence_rate": div_rate, "warm_start": None,
                        "num_devices": int(jax.local_device_count()),
                        "chain_method": ("parallel" if jax.local_device_count() >= nc
                                         else "sequential"),
                        "orthogonalize": None if rot_report is None else
                        {"requested": "none", "columns": [], "any_applied": False,
                         "rotation": rot_report}},
        "diagnostics": {"rhat": rhat, "ess": ess},
        "random_effects": random_effects,
        "posterior_samples": posterior_samples,
    }


def evaluate_log_density_multi(y, X, Z, subj, marker, XS, ZS, XQ, ZQ, T_surv,
                               event, t_quad, gk_weights, B_T, B_quad,
                               n_splines, block_sizes, beta, sigma_e, alpha, W,
                               sigma_b, L_corr, b_std, gen_reflectors=None,
                               re_structure="full", center_absorbable=False):
    """Joint log density of the multi-marker model at one point: unrotated,
    rotated (gen_reflectors given; b_std mapped to H' b_std), or partially
    centred (also center_absorbable=True; W = U L', so b is the same).
    For the exactness tests the unrotated and rotated values must agree; the
    partially centred one is a density in (W, V) rather than (U, V), so it
    differs from them by exactly -k log|det L| (the Jacobian of U -> U L'),
    and nothing else. Default (data-free) priors."""
    from numpyro.infer.util import log_density
    block_sizes = [int(v) for v in np.atleast_1d(block_sizes)]
    X = np.asarray(X, float); Z = np.asarray(Z, float)
    P, Q = X.shape[1], Z.shape[1]
    N_sub = int(np.asarray(T_surv).shape[0])
    model = build_model_multi(P, Q, int(n_splines), block_sizes,
                              re_structure=re_structure, gen_reflectors=gen_reflectors,
                              center_absorbable=bool(center_absorbable))
    params = {"beta": jnp.asarray(beta, dtype=float),
              "sigma_e": jnp.atleast_1d(jnp.asarray(sigma_e, dtype=float)),
              "alpha": jnp.atleast_1d(jnp.asarray(alpha, dtype=float)),
              "W": jnp.asarray(W, dtype=float),
              "sigma_b": jnp.asarray(sigma_b, dtype=float)}
    Lc = np.asarray(L_corr, float)
    if re_structure == "block":
        start = 0
        for k, s in enumerate(block_sizes):
            if s >= 2:
                params["L_corr_%d" % k] = jnp.asarray(Lc[start:start + s, start:start + s])
            start += s
    else:
        params["L_corr"] = jnp.asarray(Lc)
    Bs = np.asarray(b_std, float).reshape(N_sub, Q)
    if gen_reflectors is None:
        params["b_std"] = jnp.asarray(Bs)
    else:
        Vg = np.asarray(gen_reflectors, float)
        kg = int(Vg.shape[1])
        UV = _apply_reflectors(Vg, Bs, transpose=True)
        if center_absorbable:
            Lfull = np.asarray(sigma_b, float)[:, None] * Lc
            params["b_gen_W"] = jnp.asarray(UV[:kg] @ Lfull.T)
        else:
            params["b_gen_U"] = jnp.asarray(UV[:kg])
        params["b_gen_V"] = jnp.asarray(UV[kg:])
    kw = dict(y=jnp.asarray(np.asarray(y, float)), X=jnp.asarray(X), Z=jnp.asarray(Z),
              subj=jnp.asarray(np.asarray(subj, int).reshape(-1)),
              marker=jnp.asarray(np.asarray(marker, int).reshape(-1)),
              XS=jnp.asarray(np.asarray(XS, float)), ZS=jnp.asarray(np.asarray(ZS, float)),
              XQ=jnp.asarray(np.asarray(XQ, float)), ZQ=jnp.asarray(np.asarray(ZQ, float)),
              T_surv=jnp.asarray(np.asarray(T_surv, float)),
              event=jnp.asarray(np.asarray(event, float)),
              t_quad=jnp.asarray(np.asarray(t_quad, float)),
              gk_weights=jnp.asarray(np.asarray(gk_weights, float).reshape(-1)),
              B_T=jnp.asarray(np.asarray(B_T, float)),
              B_quad=jnp.asarray(np.asarray(B_quad, float)), N_sub=N_sub)
    ld, _ = log_density(model, (), kw, params)
    return float(ld)
