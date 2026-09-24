"""Beyond joint models: the rotation on two ordinary mixed models.

Methods paper, Section 7. The location degeneracy is not specific to joint
models: any mixed model whose fixed effects include subject-constant
columns (the intercept, a treatment arm, sex) leaves the split between
those coefficients and the matching random-effect directions to the prior.
This script is STANDALONE - it does not import jmjax - so it shows that the
construction transfers to models jmjax does not fit.

MODELS (data written by dev/generality/export_data.R):
  orthodont  Gaussian LMM. distance ~ age_c * male + (1 + age_c | child).
             27 children x 4 visits; male is subject-constant. q = 2.
  toenail    Logistic GLMM. y ~ trt * time + (1 | patient).
             294 patients, 1908 visits; trt is subject-constant. q = 1.

Both use the NON-CENTRED parameterization (b = b_std L^T), which is what
brms and rstanarm do by default, with NUTS and an adapted diagonal metric.

ARMS (GEN_ARMS, comma-separated; default all five):
  U       unrotated - the default a user of brms/rstanarm/NumPyro gets
  R_diag  rotated, diagonal metric only
  R       rotated + one dense metric block over (beta, U)   [the method]
  U_dense unrotated, FULL dense metric over every sampled coordinate
          (paper review, experiment 1: "why not just adapt a dense metric
          in the original coordinates?")
  C       centred parameterization (b sampled directly, b_i ~ N(0, L L')),
          diagonal metric (review experiment 2: "why not just centre?")

The rotation: Q = orthonormal basis of the absorbable directions (union
over random-effect columns; constructed below from the design, not from
column names), H = an orthogonal N x N matrix whose first k columns span
Q, and b_std = H [U; V] with U [k, q], V [N-k, q] iid N(0, 1). Exact:
b_std is iid N(0, 1) either way.

The rotation is applied as k Householder reflections (O(Nk) per column),
as in jmjax, not as a dense N x N matrix.

METRICS per parameter: ESS per draw, ESS per 1000 gradient evaluations
(leapfrog steps; machine-independent), ESS/sec, R-hat, and the posterior
mean difference from U in posterior SDs (exactness check).

    python dev/generality/mixed_models.py              # 3 seeds, all arms
    GEN_SEEDS=1 python dev/generality/mixed_models.py  # quick look
    GEN_ARMS=U,R_diag,R python dev/generality/mixed_models.py   # original 3 arms

Output: results_<datasets>.csv for the original three arms, and
results_<datasets>_ext.csv whenever U_dense or C is included, so the
committed three-arm results are never overwritten.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- absorbable directions (numpy only) ----------------------------------
def absorbable_basis(X, Z, sid, N, tol=1e-9):
    """Orthonormal [N, k] basis of the subject-level directions v that some
    fixed-effect shift d absorbs in some random-effect column j:
    X_i d = v_i * Z_ij for every subject i. For each column j, d must lie
    in the null space of stack_i (I - P_i) X_i (P_i projects on z_ij; rows
    with z_ij = 0 for the whole subject force X_i d = 0), and then
    v_i = z_ij' X_i d / z_ij' z_ij. Every direction is checked against the
    defining identity before it is kept. Returns (Q, k, per-column info)."""
    X = np.asarray(X, float); Z = np.asarray(Z, float); sid = np.asarray(sid)
    p = X.shape[1]
    vecs, info = [], []
    for j in range(Z.shape[1]):
        rows = []
        for i in range(N):
            m = sid == i
            Xi, zi = X[m], Z[m, j]
            zz = zi @ zi
            if zz <= tol:
                rows.append(Xi)
            else:
                rows.append(Xi - np.outer(zi, zi @ Xi) / zz)
        A = np.vstack(rows)
        _, s, Vt = np.linalg.svd(A, full_matrices=True)
        rank = int((s > tol * max(s.max(), 1.0)).sum())
        D = Vt[rank:].T                                  # [p, dim ker]
        found = 0
        for c in range(D.shape[1]):
            d = D[:, c]
            v = np.zeros(N)
            for i in range(N):
                m = sid == i
                zi = Z[m, j]; zz = zi @ zi
                v[i] = (zi @ (X[m] @ d)) / zz if zz > tol else 0.0
            resid = np.abs(X @ d - v[sid] * Z[:, j]).max()
            if resid <= 1e-8 * max(1.0, np.abs(X @ d).max()) and np.abs(v).max() > tol:
                vecs.append(v); found += 1
        info.append((j, found))
    if not vecs:
        return None, 0, info
    M = np.column_stack(vecs)
    U_, s, _ = np.linalg.svd(M, full_matrices=False)
    k = int((s > 1e-8 * s.max()).sum())
    return U_[:, :k], k, info


def householder_reflectors(Q):
    """k unit Householder vectors [N, k] with H = H_1 ... H_k orthogonal and
    H[:, :k] spanning Q. Applying H costs O(N k) per column, against
    O(N^2) for a dense N x N matrix - the form jmjax uses."""
    A = np.array(Q, float); N, k = A.shape
    Vr = np.zeros((N, k))
    for j in range(k):
        x = A[j:, j]
        alpha = -np.copysign(np.linalg.norm(x), x[0] if x[0] != 0 else 1.0)
        v = x.copy(); v[0] -= alpha
        v /= np.linalg.norm(v)
        Vr[j:, j] = v
        A = A - 2.0 * np.outer(Vr[:, j], Vr[:, j] @ A)
    return Vr


def apply_reflectors(Vr, M, xp=np):
    """H @ M with H = H_1 ... H_k, H_j = I - 2 v_j v_j'."""
    for j in range(Vr.shape[1] - 1, -1, -1):
        v = Vr[:, j:j + 1]
        M = M - 2.0 * (v @ (v.T @ M))
    return M


def completion(Q):
    """Orthogonal [N, N] matrix whose first k columns span Q."""
    H, _ = np.linalg.qr(Q, mode="complete")
    assert np.allclose(H.T @ H, np.eye(H.shape[0]), atol=1e-10)
    k = Q.shape[1]
    assert np.allclose(H[:, :k] @ (H[:, :k].T @ Q), Q, atol=1e-10)   # same span
    return H


# ---- data -----------------------------------------------------------------
def load(name):
    import csv
    path = os.path.join(HERE, f"{name}.csv")
    if not os.path.exists(path):
        print(f"\n== {name}: {path} not found - run Rscript dev/generality/export_data.R; skipped")
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    col = lambda c: np.array([float(r[c]) for r in rows])
    ids = col("id").astype(int)
    uniq = np.unique(ids); sid = np.searchsorted(uniq, ids)
    y = col("y")
    if name == "orthodont":
        a, m = col("age_c"), col("male")
        X = np.column_stack([np.ones_like(a), a, m, a * m])
        Z = np.column_stack([np.ones_like(a), a])
        names = ["intercept", "age", "male", "age:male"]
        family = "gaussian"
    else:
        t, tr = col("time"), col("trt")
        X = np.column_stack([np.ones_like(t), tr, t, tr * t])
        Z = np.ones((len(t), 1))
        names = ["intercept", "trt", "time", "trt:time"]
        family = "bernoulli"
    return dict(X=X, Z=Z, y=y, sid=sid, N=len(uniq), names=names, family=family)


# ---- model ----------------------------------------------------------------
def make_model(d, H=None, k=0, centred=False):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    X, Z, y = jnp.asarray(d["X"]), jnp.asarray(d["Z"]), jnp.asarray(d["y"])
    sid = jnp.asarray(d["sid"]); N = d["N"]; p = X.shape[1]; q = Z.shape[1]
    Vr = None if H is None else jnp.asarray(H)      # Householder vectors [N, k]
    gaussian = d["family"] == "gaussian"

    def model():
        beta = numpyro.sample("beta", dist.Normal(0.0, 10.0).expand([p]).to_event(1))
        sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(5.0).expand([q]).to_event(1))
        if q > 1:
            Lc = numpyro.sample("L_corr", dist.LKJCholesky(q, 2.0))
            L = sigma_b[:, None] * Lc
        else:
            L = sigma_b.reshape(1, 1)
        if centred:
            # centred: the random effects themselves are the sampled coordinates
            b = numpyro.sample("b", dist.MultivariateNormal(jnp.zeros(q), scale_tril=L)
                               .expand([N]).to_event(1))
            b_std = None
        elif Vr is None:
            b_std = numpyro.sample("b_std", dist.Normal(0.0, 1.0).expand([N, q]).to_event(2))
        else:
            U = numpyro.sample("U", dist.Normal(0.0, 1.0).expand([k, q]).to_event(2))
            V = numpyro.sample("V", dist.Normal(0.0, 1.0).expand([N - k, q]).to_event(2))
            b_std = apply_reflectors(Vr, jnp.concatenate([U, V], axis=0))
        if b_std is not None:
            b = b_std @ L.T
        eta = X @ beta + jnp.sum(Z * b[sid], axis=1)
        if gaussian:
            sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(5.0))
            numpyro.sample("y", dist.Normal(eta, sigma_e), obs=y)
        else:
            numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)
    return model


def run_arm(d, arm, seed, H, k, warmup, samples, chains):
    import jax
    from numpyro.infer import MCMC, NUTS
    from numpyro.diagnostics import summary
    rot = arm in ("R", "R_diag")
    model = make_model(d, H if rot else None, k if rot else 0, centred=(arm == "C"))
    dense = [("beta", "U")] if arm == "R" else (True if arm == "U_dense" else False)
    kern = NUTS(model, dense_mass=dense, target_accept_prob=0.8)
    mcmc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("num_steps", "diverging"))
    sec = time.time() - t0
    ex = mcmc.get_extra_fields()
    steps = float(np.sum(np.asarray(ex["num_steps"])))
    try:   # adapted step size, averaged over chains (a direct view of the geometry)
        eps = float(np.mean(np.asarray(mcmc.last_state.adapt_state.step_size)))
    except Exception:
        eps = float("nan")
    ndiv = int(np.sum(np.asarray(ex["diverging"])))
    s = mcmc.get_samples(group_by_chain=True)
    keep = {k_: v for k_, v in s.items() if k_ in ("beta", "sigma_b", "sigma_e", "L_corr")}
    sm = summary(keep, group_by_chain=True)
    out = []
    def add(name, key, idx):
        dr = np.asarray(s[key])
        x = dr[(slice(None), slice(None)) + idx]
        st = sm[key]
        ess = float(np.asarray(st["n_eff"])[idx]); rh = float(np.asarray(st["r_hat"])[idx])
        out.append(dict(param=name, mean=float(x.mean()), sd=float(x.std()),
                        ess=ess, rhat=rh))
    for j, nm in enumerate(d["names"]):
        add(nm, "beta", (j,))
    for j in range(d["Z"].shape[1]):
        add(f"sigma_b{j}", "sigma_b", (j,))
    if "sigma_e" in s:
        add("sigma_e", "sigma_e", ())
    if "L_corr" in s:
        add("rho", "L_corr", (1, 0))
    n = chains * samples
    for r in out:
        r.update(arm=arm, seed=seed, sec=sec, grads=steps, ndiv=ndiv, step_size=eps,
                 steps_per_draw=steps / n,
                 ess_per_draw=r["ess"] / n, ess_per_kgrad=1000 * r["ess"] / steps,
                 ess_per_sec=r["ess"] / sec)
    return out


def main():
    try:
        import numpyro
    except ImportError:
        sys.exit("numpyro not found: run this with the Python environment jmjax uses")
    numpyro.enable_x64()
    seeds = int(os.environ.get("GEN_SEEDS", "3"))
    warmup = int(os.environ.get("GEN_WARMUP", "1000"))
    samples = int(os.environ.get("GEN_SAMPLES", "1000"))
    chains = int(os.environ.get("GEN_CHAINS", "4"))
    out_csv = None
    rows = []
    names = os.environ.get("GEN_DATA", "orthodont,toenail").split(",")
    arms = os.environ.get("GEN_ARMS", "U,R_diag,R,U_dense,C").split(",")
    assert "U" in arms, "the unrotated arm U is the reference for every ratio"
    for name in names:
        d = load(name)
        if d is None:
            continue
        Q, k, info = absorbable_basis(d["X"], d["Z"], d["sid"], d["N"])
        print(f"\n== {name}: N = {d['N']} subjects, {len(d['y'])} obs, q = {d['Z'].shape[1]}; "
              f"absorbable directions per RE column {info}, union k = {k}")
        if Q is None:
            print("   no absorbable direction - nothing to rotate"); continue
        Hd = completion(Q)                      # dense check of the construction
        H = householder_reflectors(Q)
        Hr = apply_reflectors(H, np.eye(d["N"]))
        assert np.allclose(Hr.T @ Hr, np.eye(d["N"]), atol=1e-10)
        assert np.allclose(Hr[:, :k] @ (Hr[:, :k].T @ Q), Q, atol=1e-10)
        for seed in range(1, seeds + 1):
            for arm in arms:
                res = run_arm(d, arm, seed, H, k, warmup, samples, chains)
                for r in res:
                    r["dataset"] = name
                rows += res
                i0 = res[0]
                print(f"   seed {seed} {arm:7s} {i0['sec']:6.1f}s  grads {i0['grads']:9.0f}  "
                      f"step {i0['step_size']:.4f}  div {i0['ndiv']:3d}  "
                      f"intercept ESS/draw {i0['ess_per_draw']:.3f}  "
                      f"max R-hat {max(r['rhat'] for r in res):.3f}")
    import csv
    tag = "_".join(sorted(set(r["dataset"] for r in rows))) or "none"
    ext = "_ext" if ({"U_dense", "C"} & set(arms)) else ""
    out_csv = os.path.join(HERE, f"results_{tag}{ext}.csv")
    keys = ["dataset", "seed", "arm", "param", "mean", "sd", "ess", "rhat",
            "ess_per_draw", "ess_per_kgrad", "ess_per_sec", "sec", "grads", "ndiv"]
    if ext:
        keys += ["step_size", "steps_per_draw"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k_: r[k_] for k_ in keys})
    summarize(rows)
    print(f"\nWrote {out_csv}")


def summarize(rows):
    gm = lambda v: float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")
    for name in sorted(set(r["dataset"] for r in rows)):
        R = [r for r in rows if r["dataset"] == name]
        params = list(dict.fromkeys(r["param"] for r in R))
        seeds = sorted(set(r["seed"] for r in R))
        arms = [a for a in dict.fromkeys(r["arm"] for r in R) if a != "U"]
        print(f"\n{name}: ratios over U, geometric mean over seeds "
              f"(per gradient | per second, sec = whole run incl. warm-up); U's ESS per draw")
        print(f"{'param':11s} {'U ESS/draw':>10s} " + " ".join(f"{a + '/U':>17s}" for a in arms))
        for p in params:
            g = lambda arm, key: {r["seed"]: r[key] for r in R if r["param"] == p and r["arm"] == arm}
            def ratio(arm, key):
                a, u = g(arm, key), g("U", key)
                return gm([a[s_] / u[s_] for s_ in seeds if s_ in a and s_ in u and u[s_] > 0])
            eu = g("U", "ess_per_draw")
            print(f"{p:11s} {np.mean(list(eu.values())):10.3f} " + " ".join(
                f"{ratio(a,'ess_per_kgrad'):7.2f}x|{ratio(a,'ess_per_sec'):7.2f}x" for a in arms))
        print("per arm: max R-hat | divergences | mean step size | leapfrog steps per draw | "
              "mean seconds | largest |mean - mean_U| in SDs")
        for arm in ["U"] + arms:
            A = [r for r in R if r["arm"] == arm]
            dm = 0.0
            for r in A:
                u = [x for x in R if x["arm"] == "U" and x["param"] == r["param"] and x["seed"] == r["seed"]]
                if u and r["sd"] > 0:
                    dm = max(dm, abs(r["mean"] - u[0]["mean"]) / r["sd"])
            per_run = {(r["seed"]): r for r in A}
            print(f"  {arm:7s} {max(r['rhat'] for r in A):6.3f} | {sum(r['ndiv'] for r in per_run.values()):4d} | "
                  f"{np.mean([r.get('step_size', np.nan) for r in per_run.values()]):.4f} | "
                  f"{np.mean([r.get('steps_per_draw', np.nan) for r in per_run.values()]):7.1f} | "
                  f"{np.mean([r['sec'] for r in per_run.values()]):6.1f} | {dm:6.3f}")


if __name__ == "__main__":
    main()
