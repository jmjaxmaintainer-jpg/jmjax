"""Verification of the "beta_corrected" analytical-correction track added
alongside control$orthogonalize_b0 / orthogonalize_b.

CONTEXT. dev/study_calibration.R's pilot run found that orthogonalize_b/_b0
reproduce random_effects_method = "wishart_gibbs_centered"'s retracted
failure mode: unbiased point estimates for the swept parameter(s), but
posterior SD too small by close to the theoretically predicted
sigma_b/sqrt(N). fit_nuts() now computes, as pure post-processing of draws
it already produces (no new numpyro site, no refit), a "beta_corrected" key
in posterior_samples using

    beta_corrected = beta_orth - sum_q  Dmat_q @ (Q_q^T @ b_raw[:, q])

This file checks the SHIPPED code (_absorbable_generator_matrix(), and the
identity it exists to provide) against designs whose absorbable set is
known, independent of any MCMC run:

  G. Dmat identity: X_i @ Dmat[:, m] == Q[:, m]_i * Z_i,q(t) for every
     subject/time -- the property _absorbable_generator_matrix()'s
     docstring claims, checked directly rather than trusting the residual
     check already inside the function.
  H. Fitted-value equivalence: for synthetic (beta_orth, b_raw) draws, the
     orthogonalized model's fitted values (beta_orth, b after subtracting
     Q's projection) exactly match the unconstrained model's fitted values
     (beta_corrected, b_raw itself) -- the actual claim beta_corrected
     relies on, not just the Dmat identity it is built from.
  I. Multi-column case (orthogonalize_b, k >= 2 swept columns at once):
     the same equivalence holds when BOTH the intercept and slope columns
     are swept, combined additively.
  J. Randomized sweep over both coordinate-aligned and spline/orthogonal-
     polynomial (Example 1 style) designs.

Run:  python3 dev/verify_beta_correction.py
Exits non-zero if any expectation fails.

NOTE this verifies the ALGEBRA is correctly implemented. It does NOT by
itself establish that beta_corrected restores calibration in a real
MCMC fit -- that is an empirical question, checked by re-running
dev/study_calibration.R's pilot with the corrected estimand (see NEWS.md
and vignette("jmjax-reparameterization"), Section 9).
"""
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PY = os.path.join(HERE, os.pardir, "inst", "python", "jmjax_backend",
                        "mcmc_model.py")
WANTED = ("_absorbable_basis_exact", "_absorbable_generator_matrix")


def load():
    """Load the two functions from the shipped backend.

    Prefers a normal import so the genuinely shipped objects are tested; if
    jax/numpyro are not importable here, exec's just these functions, which
    depend only on numpy and warnings.
    """
    try:
        sys.path.insert(0, os.path.join(HERE, os.pardir, "inst", "python"))
        from jmjax_backend import mcmc_model as m
        return tuple(getattr(m, w) for w in WANTED), "imported from jmjax_backend"
    except Exception:
        src = open(MODEL_PY, encoding="utf-8").read()
        ns = {"np": np, "warnings": warnings}
        for w in WANTED:
            mt = re.search(r"^def %s\(.*?(?=^def |\Z)" % w, src, flags=re.S | re.M)
            if mt is None:
                raise RuntimeError("could not locate %s in %s" % (w, MODEL_PY))
            exec(mt.group(0), ns)
        return tuple(ns[w] for w in WANTED), "source-extracted from mcmc_model.py"


(_absorbable_basis_exact, _absorbable_generator_matrix), how = load()

failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def dim(Q):
    return 0 if Q is None else Q.shape[1]


def grid(N, mo, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = np.full(N, mo)
    t = np.tile(np.linspace(0.5, 3.0, mo), (N, 1))
    return rng, n_obs, t


print("loaded: %s\n" % how)

# ------------------------------------------------------------------ G + H
print("G+H. Dmat identity and fitted-value equivalence, canonical designs")
rng, n_obs, t = grid(40, 5)
xc = rng.normal(size=40)[:, None] * np.ones((1, 5))
sex = rng.integers(0, 2, 40).astype(float)[:, None] * np.ones((1, 5))
Z = np.stack([np.ones((40, 5)), t], axis=2)
designs = {
    "X = [1, t]": [np.ones((40, 5)), t],
    "X = [1, t, x]": [np.ones((40, 5)), t, xc],
    "X = [1, t, sex, sex:t]": [np.ones((40, 5)), t, sex, sex * t],
    "X = [1, t + 0.5t^2, t^2]  (spline/poly style, Example 1)":
        [np.ones((40, 5)), t + 0.5 * t ** 2, t ** 2],
}
mask40 = np.arange(5)[None, :] < n_obs[:, None]

for name, cols in designs.items():
    X = np.stack(cols, axis=2)
    p = X.shape[2]
    for q in (0, 1):
        Q, gens = quiet(_absorbable_basis_exact, X, Z, n_obs, q)
        if Q is None:
            continue
        Dmat = quiet(_absorbable_generator_matrix, Q, gens)
        check(Dmat is not None, "%s  q=%d: Dmat recovered (dim %d)"
                                 % (name, q, dim(Q)))
        if Dmat is None:
            continue
        check(Dmat.shape == (p, Q.shape[1]),
              "%s  q=%d: Dmat shape [%d, %d] matches [p, k]"
              % (name, q, *Dmat.shape))

        worst_id = 0.0
        for m in range(Dmat.shape[1]):
            lhs = X @ Dmat[:, m]
            rhs = Q[:, m][:, None] * Z[:, :, q]
            d = np.abs(lhs - rhs)[mask40]
            denom = max(float(np.abs(rhs)[mask40].max()),
                        float(np.abs(lhs)[mask40].max()), 1.0)
            worst_id = max(worst_id, float(d.max()) / denom)
        check(worst_id < 1e-8,
              "%s  q=%d: Dmat identity X@Dmat == Q*Z holds (resid %.1e)"
              % (name, q, worst_id))

        r2 = np.random.default_rng(42)
        beta_orth = r2.normal(size=p)
        b_raw_q = r2.normal(size=40) * 2.0
        c = Q.T @ b_raw_q
        b_after_q = b_raw_q - Q @ c
        beta_corrected = beta_orth - Dmat @ c

        mu_orth = (np.einsum("nop,p->no", X, beta_orth)
                   + b_after_q[:, None] * Z[:, :, q])
        mu_default = (np.einsum("nop,p->no", X, beta_corrected)
                      + b_raw_q[:, None] * Z[:, :, q])
        rel = (float(np.abs(mu_orth - mu_default)[mask40].max())
               / max(float(np.abs(mu_orth)[mask40].max()), 1.0))
        check(rel < 1e-8,
              "%s  q=%d: beta_corrected reproduces the unconstrained "
              "model's fitted values (resid %.1e)" % (name, q, rel))

# ---------------------------------------------------------------------- I
print("\nI. orthogonalize_b: intercept AND slope swept together (additive)")
X7 = np.stack([np.ones((40, 5)), t, xc], axis=2)
p7 = X7.shape[2]
Q0, gens0 = quiet(_absorbable_basis_exact, X7, Z, n_obs, 0)
Q1, gens1 = quiet(_absorbable_basis_exact, X7, Z, n_obs, 1)
check(Q0 is not None and Q1 is not None,
      "both intercept and slope columns have an absorbable direction here")
if Q0 is not None and Q1 is not None:
    D0 = quiet(_absorbable_generator_matrix, Q0, gens0)
    D1 = quiet(_absorbable_generator_matrix, Q1, gens1)
    r3 = np.random.default_rng(7)
    beta_orth = r3.normal(size=p7)
    b_raw = r3.normal(size=(40, 2)) * 1.5
    c0 = Q0.T @ b_raw[:, 0]
    c1 = Q1.T @ b_raw[:, 1]
    b_after = np.stack([b_raw[:, 0] - Q0 @ c0, b_raw[:, 1] - Q1 @ c1], axis=1)
    beta_corrected = beta_orth - D0 @ c0 - D1 @ c1

    mu_orth = (np.einsum("nop,p->no", X7, beta_orth)
               + np.einsum("noq,nq->no", Z, b_after))
    mu_default = (np.einsum("nop,p->no", X7, beta_corrected)
                  + np.einsum("noq,nq->no", Z, b_raw))
    rel = (float(np.abs(mu_orth - mu_default)[mask40].max())
           / max(float(np.abs(mu_orth)[mask40].max()), 1.0))
    check(rel < 1e-8,
          "combined (q=0 and q=1) correction reproduces fitted values "
          "(resid %.1e)" % rel)

# ---------------------------------------------------------------------- J
print("\nJ. randomized sweep: coordinate-aligned and spline/poly designs")
rs = np.random.default_rng(23)
n_checked = n_id_fail = n_mu_fail = n_dmat_none = 0
worst_id_all = worst_mu_all = 0.0
for _ in range(150):
    Nr = int(rs.integers(15, 40))
    mor = int(rs.integers(2, 6))
    nr = rs.integers(1, mor + 1, size=Nr)
    tr = np.zeros((Nr, mor))
    for i in range(Nr):
        tr[i, :nr[i]] = np.sort(rs.uniform(0, 4, nr[i]))
    poly_style = bool(rs.integers(0, 2))
    if poly_style:
        cols = [np.ones((Nr, mor)), tr + rs.normal() * tr ** 2, tr ** 2]
    else:
        cols = [np.ones((Nr, mor)), tr]
        for _k in range(int(rs.integers(0, 2))):
            cols.append(rs.normal(size=Nr)[:, None] * np.ones((1, mor)))
    Xr = np.stack(cols, axis=2)
    Zr = np.stack([np.ones((Nr, mor)), tr], axis=2)
    maskr = np.arange(mor)[None, :] < nr[:, None]
    pr = Xr.shape[2]
    for q in (0, 1):
        Q, gens = quiet(_absorbable_basis_exact, Xr, Zr, nr, q)
        if Q is None:
            continue
        Dmat = quiet(_absorbable_generator_matrix, Q, gens)
        if Dmat is None:
            n_dmat_none += 1
            continue
        n_checked += 1
        for m in range(Dmat.shape[1]):
            lhs = Xr @ Dmat[:, m]
            rhs = Q[:, m][:, None] * Zr[:, :, q]
            d = np.abs(lhs - rhs)[maskr]
            denom = max(float(np.abs(rhs)[maskr].max()),
                        float(np.abs(lhs)[maskr].max()), 1.0)
            resid = float(d.max()) / denom
            worst_id_all = max(worst_id_all, resid)
            if resid > 1e-6:
                n_id_fail += 1

        r4 = np.random.default_rng(int(rs.integers(0, 1_000_000)))
        beta_orth = r4.normal(size=pr)
        b_raw_q = r4.normal(size=Nr) * 2.0
        c = Q.T @ b_raw_q
        b_after_q = b_raw_q - Q @ c
        beta_corrected = beta_orth - Dmat @ c
        mu_orth = (np.einsum("nop,p->no", Xr, beta_orth)
                   + b_after_q[:, None] * Zr[:, :, q])
        mu_default = (np.einsum("nop,p->no", Xr, beta_corrected)
                      + b_raw_q[:, None] * Zr[:, :, q])
        rel = (float(np.abs(mu_orth - mu_default)[maskr].max())
               / max(float(np.abs(mu_orth)[maskr].max()), 1.0))
        worst_mu_all = max(worst_mu_all, rel)
        if rel > 1e-6:
            n_mu_fail += 1

check(n_checked > 0, "at least one (design, column) pair produced a "
                      "checkable Dmat (checked %d)" % n_checked)
check(n_dmat_none == 0,
      "_absorbable_generator_matrix never refused a Q that "
      "_absorbable_basis_exact returned (%d refusals)" % n_dmat_none)
check(n_id_fail == 0,
      "Dmat identity holds on every case (worst residual %.1e)" % worst_id_all)
check(n_mu_fail == 0,
      "fitted-value equivalence holds on every case (worst residual %.1e)"
      % worst_mu_all)

# ------------------------------------------------------------------ done
print("\n%d check(s) FAILED." % len(failures) if failures
      else "\nAll checks passed.")
sys.exit(1 if failures else 0)
