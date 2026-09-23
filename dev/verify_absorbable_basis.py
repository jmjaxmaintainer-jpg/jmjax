"""Verification of the absorbable-basis construction used by
control$orthogonalize_b0 / orthogonalize_b.

Checks the SHIPPED code in inst/python/jmjax_backend/mcmc_model.py against
designs whose absorbable set is known in closed form, and confirms the
claims made in vignette("jmjax-reparameterization"), Section 4:

  A. Soundness (Propositions 2-3): every direction returned is genuinely
     absorbable -- there is a beta-shift reproducing it exactly.
  B. Completeness (Proposition 4, Example 1): the exact construction finds
     directions that the per-column search misses when the longitudinal
     mean uses a spline / orthogonal-polynomial basis.
  C. Equivalence regression: on every design spelled the way dev/sim_joint.R
     and the study scripts spell them, the exact construction returns the
     SAME subspace as the per-column search -- so the results reported in
     the vignette are unaffected by the repair.
  D. Proposition 5: subjects whose own data cannot determine their ratio
     take the consensus value, keeping the basis uniform.
  E. Condition (S), Section 4.4: absorbability established on the
     longitudinal grid is checked against the survival/quadrature grid, and
     a design that violates it is detected.

Run:  python3 dev/verify_absorbable_basis.py
Exits non-zero if any expectation fails.

NOTE this verifies an ALGEBRAIC property of the construction. It says
nothing about posterior calibration, which remains open (Section 9).
"""
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The functions moved out of mcmc_model.py into absorbable.py (basis) and
# sweep.py (legacy correction); mcmc_model still re-exports them.
MODEL_PYS = [os.path.join(HERE, os.pardir, "inst", "python", "jmjax_backend", f)
             for f in ['absorbable.py']]
WANTED = ("_absorbable_basis", "_absorbable_basis_exact",
          "_structural_extension_residual")


def load():
    """Load the three functions from the shipped backend.

    Prefers a normal import so the genuinely shipped objects are tested; if
    jax/numpyro are not importable here, exec's just these functions, which
    depend only on numpy and warnings.
    """
    try:
        sys.path.insert(0, os.path.join(HERE, os.pardir, "inst", "python"))
        from jmjax_backend import mcmc_model as m
        return tuple(getattr(m, w) for w in WANTED), "imported from jmjax_backend"
    except Exception:
        src = "\n".join(open(f, encoding="utf-8").read() for f in MODEL_PYS)
        ns = {"np": np, "warnings": warnings}
        for w in WANTED:
            mt = re.search(r"^def %s\(.*?(?=^def |\Z)" % w, src, flags=re.S | re.M)
            if mt is None:
                raise RuntimeError("could not locate %s in %s" % (w, MODEL_PYS))
            exec(mt.group(0), ns)
        return tuple(ns[w] for w in WANTED), "source-extracted from %s" % ", ".join(os.path.basename(f) for f in MODEL_PYS)


(_absorbable_basis, _absorbable_basis_exact,
 _structural_extension_residual), how = load()

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


def same_span(A, B, tol=1e-8):
    if A is None and B is None:
        return True
    if A is None or B is None or A.shape[1] != B.shape[1]:
        return False
    return float(np.abs(A @ (A.T @ B) - B).max()) < tol


def absorb_residual(X, Z, n_obs, q, v):
    """Ground truth (Proposition 1): relative residual of least-squares
    fitting the stacked (v_i z_i) by the columns of X."""
    N, mo, _ = X.shape
    mask = np.arange(mo)[None, :] < np.asarray(n_obs)[:, None]
    A = np.vstack([X[i][mask[i]] for i in range(N)])
    rhs = np.concatenate([v[i] * Z[i, :, q][mask[i]] for i in range(N)])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return float(np.abs(A @ sol - rhs).max()) / max(float(np.abs(rhs).max()), 1.0)


def grid(N, mo, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = np.full(N, mo)
    t = np.tile(np.linspace(0.5, 3.0, mo), (N, 1))
    return rng, n_obs, t


print("loaded: %s\n" % how)

# ------------------------------------------------------------------ A / C
print("A+C. canonical designs: sound, complete, and unchanged by the repair")
rng, n_obs, t = grid(40, 5)
xc = rng.normal(size=40)[:, None] * np.ones((1, 5))
sex = rng.integers(0, 2, 40).astype(float)[:, None] * np.ones((1, 5))
Z = np.stack([np.ones((40, 5)), t], axis=2)
designs = {
    "X = [1, t]": [np.ones((40, 5)), t],
    "X = [1, t, x]": [np.ones((40, 5)), t, xc],
    "X = [1, t, sex, sex:t]": [np.ones((40, 5)), t, sex, sex * t],
    "X = [1, t, x] with age-scale column": [np.ones((40, 5)), t, 50.0 * xc],
}
for name, cols in designs.items():
    X = np.stack(cols, axis=2)
    for q in (0, 1):
        Qc, kept = quiet(_absorbable_basis, X, Z, n_obs, q)
        Qe, gens = quiet(_absorbable_basis_exact, X, Z, n_obs, q)
        check(same_span(Qc, Qe),
              "%s  q=%d: exact basis == per-column basis (dim %d)"
              % (name, q, dim(Qe)))
        for k in range(dim(Qe)):
            r = absorb_residual(X, Z, n_obs, q, Qe[:, k])
            check(r < 1e-8, "%s  q=%d: direction %d absorbable (resid %.1e)"
                            % (name, q, k, r))

# -------------------------------------------------------------------- B
print("\nB. Example 1: spline / orthogonal-polynomial longitudinal mean")
print("   X = [1, t + 0.5 t^2, t^2]  -- same span as [1, t, t^2]")
X2 = np.stack([np.ones((40, 5)), t + 0.5 * t ** 2, t ** 2], axis=2)
Qc0, _ = quiet(_absorbable_basis, X2, Z, n_obs, 0)
Qc1, _ = quiet(_absorbable_basis, X2, Z, n_obs, 1)
Qe1, _ = quiet(_absorbable_basis_exact, X2, Z, n_obs, 1)
check(Qc1 is None, "per-column search finds NO slope direction (the defect)")
check(Qc0 is not None,
      "but still finds one for the intercept, so the old no-op warning "
      "would not have fired -- the defect was silent")
check(dim(Qe1) == 1, "exact construction recovers 1 slope direction (the fix)")
if Qe1 is not None:
    check(float(np.std(Qe1[:, 0])) < 1e-12,
          "and it is uniform, i.e. exactly mean(b_1) (sd %.1e)"
          % float(np.std(Qe1[:, 0])))
    r = absorb_residual(X2, Z, n_obs, 1, Qe1[:, 0])
    check(r < 1e-8, "recovered direction is genuinely absorbable (resid %.1e)" % r)

# -------------------------------------------------------------------- D
print("\nD. Proposition 5: subjects whose data cannot determine their ratio")
N4, mo4 = 25, 4
n4 = np.array([1] * 5 + [4] * 20)
t4 = np.zeros((N4, mo4))
for i in range(N4):
    t4[i, :n4[i]] = np.arange(n4[i], dtype=float)
X4 = np.stack([np.ones((N4, mo4)), t4], axis=2)
Z4 = X4.copy()
Qe4, _ = quiet(_absorbable_basis_exact, X4, Z4, n4, 1)
check(dim(Qe4) == 1, "slope column yields a basis despite 5 silent subjects")
if Qe4 is not None:
    check(float(np.std(Qe4[:, 0])) < 1e-12,
          "basis uniform to %.1e (consensus imputation, not 0)"
          % float(np.std(Qe4[:, 0])))

# -------------------------------------------------------------------- E
print("\nE. Condition (S): extension to the survival / quadrature grid")
# Consistent design: same formula evaluated at T_i and at quadrature nodes.
N5, mo5 = 30, 4
n5 = np.full(N5, mo5)
t5 = np.tile(np.linspace(0.5, 3.0, mo5), (N5, 1))
X5 = np.stack([np.ones((N5, mo5)), t5], axis=2)
Z5 = X5.copy()
T = np.linspace(3.5, 5.0, N5)
tq = T[:, None] * np.linspace(0.05, 1.0, 7)[None, :]
Xs = np.stack([np.ones(N5), T], axis=1)
Zs = Xs.copy()
Xq = np.stack([np.ones_like(tq), tq], axis=2)
Zq = Xq.copy()
_, gens5 = quiet(_absorbable_basis_exact, X5, Z5, n5, 1)
r5 = _structural_extension_residual(gens5, 1, Xs, Zs, Xq, Zq)
check(r5 < 1e-10, "consistent design satisfies (S) (residual %.1e)" % r5)

# Violating design: one observation per subject, all at t = 1, with a t^2
# column. On the grid t^2 and t coincide, so the ratio test accepts; off the
# grid they do not.
N6 = 20
n6 = np.ones(N6, dtype=int)
t6 = np.ones((N6, 1))
X6 = np.stack([np.ones((N6, 1)), t6 ** 2], axis=2)
Z6 = np.stack([np.ones((N6, 1)), t6], axis=2)
T6 = np.full(N6, 2.0)
tq6 = T6[:, None] * np.linspace(0.1, 1.0, 5)[None, :]
Xs6 = np.stack([np.ones(N6), T6 ** 2], axis=1)
Zs6 = np.stack([np.ones(N6), T6], axis=1)
Xq6 = np.stack([np.ones_like(tq6), tq6 ** 2], axis=2)
Zq6 = np.stack([np.ones_like(tq6), tq6], axis=2)
_, gens6 = quiet(_absorbable_basis_exact, X6, Z6, n6, 1)
r6 = _structural_extension_residual(gens6, 1, Xs6, Zs6, Xq6, Zq6)
check(len(gens6) > 0, "grid-only test accepts the violating design")
check(r6 > 1e-6, "(S) check detects the violation (residual %.2e)" % r6)

# -------------------------------------------------------------------- F
print("\nF. randomized sweep: soundness always, equivalence on "
      "coordinate-aligned designs")
rs = np.random.default_rng(11)
n_unsound = n_recovered = n_equal = n_shrunk = 0
worst = 0.0
for _ in range(200):
    Nr = int(rs.integers(12, 40))
    mor = int(rs.integers(2, 6))
    nr = rs.integers(1, mor + 1, size=Nr)
    tr = np.zeros((Nr, mor))
    for i in range(Nr):
        tr[i, :nr[i]] = np.sort(rs.uniform(0, 4, nr[i]))
    poly_style = bool(rs.integers(0, 2))
    if poly_style:                      # basis that mixes t and t^2
        cols = [np.ones((Nr, mor)), tr + rs.normal() * tr ** 2, tr ** 2]
    else:                               # each term its own column
        cols = [np.ones((Nr, mor)), tr]
        for _k in range(int(rs.integers(0, 3))):
            cols.append(rs.normal(size=Nr)[:, None] * np.ones((1, mor)))
        if rs.integers(0, 2):
            c = rs.normal(size=Nr)[:, None] * np.ones((1, mor))
            cols += [c, c * tr]
    Xr = np.stack([c * float(rs.choice([1.0, 1.0, 50.0, 0.02])) for c in cols],
                  axis=2)
    Zr = np.stack([np.ones((Nr, mor)), tr], axis=2)
    for q in (0, 1):
        Qc, _ = quiet(_absorbable_basis, Xr, Zr, nr, q)
        Qe, _ = quiet(_absorbable_basis_exact, Xr, Zr, nr, q)
        for k in range(dim(Qe)):
            r = absorb_residual(Xr, Zr, nr, q, Qe[:, k])
            worst = max(worst, r)
            if r > 1e-7:
                n_unsound += 1
        if dim(Qe) > dim(Qc):
            n_recovered += 1
        elif dim(Qe) < dim(Qc):
            n_shrunk += 1
        elif not poly_style:
            n_equal += 1
            if not same_span(Qc, Qe):
                n_shrunk += 1           # same dim but different span
check(n_unsound == 0,
      "soundness: 0 of the returned directions failed verification "
      "(worst residual %.1e)" % worst)
check(n_shrunk == 0,
      "the exact basis never loses a direction the per-column search found "
      "(Proposition 2)")
check(n_recovered > 0,
      "and recovers directions the per-column search missed in %d case(s)"
      % n_recovered)
check(n_equal > 0,
      "on %d coordinate-aligned design/column pairs the two agree exactly "
      "-- the study's results are unaffected" % n_equal)

# ------------------------------------------------------------------ done
print("\n%d check(s) FAILED." % len(failures) if failures
      else "\nAll checks passed.")
sys.exit(1 if failures else 0)
