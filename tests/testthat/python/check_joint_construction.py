"""Regression checks for the joint absorbable-direction construction
(absorbable._absorbable_basis_joint; methods paper, revised Section 3).

    python3 check_joint_construction.py <path-to-jmjax_backend-parent>

Run by tests/testthat/test-absorbable-joint.R through reticulate, and runnable
on its own (it needs only numpy). The reference implementation `ref_joint`
follows checks/verify_joint_construction.py in the paper repository: (T) is
solved on a dense grid on [0, tau], while the shipped code solves it on the
grid the hazard actually uses (observation times, event time, quadrature
nodes). Raises AssertionError listing the failures; prints one line per check.
"""
import sys
import warnings

import numpy as np

# Load absorbable.py by path, not through the jmjax_backend package, whose
# __init__ imports jax and numpyro: this check needs numpy only.
import importlib.util
import os
_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "inst", "python")
_spec = importlib.util.spec_from_file_location(
    "jmjax_absorbable_check", os.path.join(_root, "jmjax_backend", "absorbable.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_absorbable_basis_joint = _mod._absorbable_basis_joint
_absorbable_basis_exact = _mod._absorbable_basis_exact
_union_basis = _mod._union_basis

TAU = 10.0
GRID = np.linspace(0.0, TAU, 201)
GL = np.polynomial.legendre.leggauss(7)[0]
rng = np.random.default_rng(20261024)
failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# --------------------------------------------------------------- helpers
def subjects(N, xrow, zrow, baseline_only=0, nobs=4):
    out = []
    for i in range(N):
        t = np.array([0.0]) if i < baseline_only else np.sort(rng.uniform(0, 8, nobs))
        T = rng.uniform(max(t.max(), 1.0) + 0.1, TAU)
        out.append(dict(t=t, T=T, xrow=xrow(i), zrow=zrow(i)))
    return out


def pack(subj):
    """Padded arrays in the shapes jm_fit passes to the backend."""
    N, mo = len(subj), max(len(s["t"]) for s in subj)
    p = subj[0]["xrow"](np.zeros(1)).shape[1]
    q = subj[0]["zrow"](np.zeros(1)).shape[1]
    K = GL.size
    X, Z = np.zeros((N, mo, p)), np.zeros((N, mo, q))
    n = np.zeros(N, dtype=int)
    Xs, Zs = np.zeros((N, p)), np.zeros((N, q))
    Xq, Zq = np.zeros((N, K, p)), np.zeros((N, K, q))
    for i, s in enumerate(subj):
        k = len(s["t"]); n[i] = k
        X[i, :k], Z[i, :k] = s["xrow"](s["t"]), s["zrow"](s["t"])
        tq = s["T"] * (GL + 1.0) / 2.0
        Xs[i], Zs[i] = s["xrow"](np.array([s["T"]]))[0], s["zrow"](np.array([s["T"]]))[0]
        Xq[i], Zq[i] = s["xrow"](tq), s["zrow"](tq)
    return X, Z, n, Xs, Zs, Xq, Zq


def _null(A, tol=1e-9):
    _, s, vt = np.linalg.svd(A)
    return vt[int((s > tol * max(1.0, s.max())).sum()):].T


def _orth(A, tol=1e-9):
    if A is None or A.size == 0:
        return None
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int((s > tol * max(1.0, s.max())).sum())
    return u[:, :r] if r else None


def ref_joint(subj, survival=True):
    """The paper's reference construction, for any q."""
    M = np.vstack([(np.eye(len(s["t"])) - s["zrow"](s["t"]) @ np.linalg.pinv(s["zrow"](s["t"])))
                   @ s["xrow"](s["t"]) for s in subj])
    cols = []
    for d in _null(M).T:
        D = []
        for s in subj:
            Xg, Zg = ((s["xrow"](GRID), s["zrow"](GRID)) if survival
                      else (s["xrow"](s["t"]), s["zrow"](s["t"])))
            delta = -np.linalg.pinv(Zg) @ (Xg @ d)
            if survival and np.abs(Xg @ d + Zg @ delta).max() > 1e-8:
                break                                   # (T) fails: not a witness
            D.append(delta)
        else:
            D = np.array(D)
            cols += [D[:, c] / np.linalg.norm(D[:, c]) for c in range(D.shape[1])
                     if np.linalg.norm(D[:, c]) > 1e-9 * max(np.linalg.norm(D), 1e-300)]
    return _orth(np.array(cols).T) if cols else None


def joint(A):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _absorbable_basis_joint(*A)


def columnwise(A, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _union_basis([_absorbable_basis_exact(A[0], A[1], A[2], c)[0] for c in range(q)])


def same(A, B, tol=1e-8):
    if A is None or B is None:
        return A is None and B is None
    return A.shape == B.shape and float(np.abs(A @ A.T - B @ B.T).max()) < tol


def k_of(Q):
    return 0 if Q is None else Q.shape[1]


def in_span(Q, v):
    return Q is not None and np.linalg.norm(v - Q @ (Q.T @ v)) < 1e-8 * max(1.0, np.linalg.norm(v))


def generators_verify(A, info):
    """Independent check of (L) and (T) for every generator returned."""
    X, Z, n, Xs, Zs, Xq, Zq = A
    mask = np.arange(X.shape[1])[None, :] < n[:, None]
    worst = 0.0
    for d, D in info["generators"]:
        for lhs, rhs in ((np.einsum("nop,p->no", X, d)[mask], np.einsum("noq,nq->no", Z, D)[mask]),
                         (Xs @ d, np.einsum("nq,nq->n", Zs, D)),
                         (np.einsum("nkp,p->nk", Xq, d), np.einsum("nkq,nq->nk", Zq, D))):
            worst = max(worst, float(np.abs(lhs + rhs).max()) / max(float(np.abs(lhs).max()), 1e-300))
    return worst


one = lambda s: np.ones_like(s)
Z_int_slope = lambda i: (lambda s: np.c_[one(s), s])
N = 60
x, age = rng.normal(size=N), rng.normal(size=N)
sex = rng.integers(0, 2, N).astype(float)

# ---------------------------------------- 1. coupled shift (paper Ex. 2)
subj = subjects(N, lambda i: (lambda s: np.c_[one(s), s, x[i] * (1 + s)]), Z_int_slope)
A = pack(subj)
Q, info = joint(A)
check(k_of(Q) == 2 and in_span(Q, x) and in_span(Q, np.ones(N)),
      "coupled design: k = 2 and span contains 1 and x (k = %d)" % k_of(Q))
check(same(Q, ref_joint(subj)), "coupled design: equals the reference construction")
Qo = columnwise(A, 2)
check(k_of(Qo) == 1 and not in_span(Qo, x),
      "coupled design: the pre-0.4.0 per-column union finds only span{1}")
check(generators_verify(A, info) < 1e-8, "coupled design: every generator satisfies (L) and (T)")

# ------------------- 2. separate columns, baseline-only subjects: old == new
subj2 = subjects(N, lambda i: (lambda s: np.c_[one(s), s, age[i] * one(s), sex[i] * one(s)]),
                 Z_int_slope, baseline_only=6)
A2 = pack(subj2)
Q2, info2 = joint(A2)
check(k_of(Q2) == 3 and same(Q2, columnwise(A2, 2)) and same(Q2, ref_joint(subj2)),
      "simulation-type design with baseline-only subjects: QQ' equal for new, "
      "per-column and reference (k = %d)" % k_of(Q2))

# --------------------------- 3. baseline-only subject: (T), not min-norm (L)
s0 = A2[0][0, :1], A2[1][0, :1]
GX = np.concatenate([s0[0], A2[3][:1], A2[5][0]])
GZ = np.concatenate([s0[1], A2[4][:1], A2[6][0]])
d_time = np.array([0.0, 1.0, 0.0, 0.0])
minnorm = -np.linalg.pinv(s0[1]) @ (s0[0] @ d_time)
check(np.allclose(minnorm, [0, 0]), "baseline-only subject: min-norm (L) solution for d = e_time is (0, 0)")
okT = all(float(np.abs(GX @ d + GZ @ D[0]).max()) < 1e-8 for d, D in info2["generators"])
pinned = [D[0] for d, D in info2["generators"]
          if abs(d[1]) > 1e-8 and abs(d[0]) < 1e-12 and np.abs(d[2:]).max() < 1e-12]
check(okT, "baseline-only subject: every shipped generator satisfies (T) on its grid")
Qm, _ = _absorbable_basis_joint(A2[0], A2[1], A2[2])
check(k_of(Qm) == 4 and same(Qm, ref_joint(subj2, survival=False)),
      "mixed model (no survival arrays): uses min-norm (L), so the slope "
      "direction is not uniform and k = 4 (k = %d)" % k_of(Qm))
# The direction the trajectory forces is uniform in the slope column:
check(in_span(Q2, np.ones(N)) and k_of(Q2) == 3,
      "joint model: the slope direction is the uniform one, as (T) forces")

# ----------------------------------------------- 4. rank-deficient Z_i
w = rng.normal(size=N)
subj3 = subjects(N, lambda i: (lambda s: np.c_[one(s), s, w[i] * one(s)]),
                 lambda i: (lambda s: np.c_[one(s), w[i] * one(s)]))
A3 = pack(subj3)
Q3, info3 = joint(A3)
check(same(Q3, ref_joint(subj3)) and generators_verify(A3, info3) < 1e-8,
      "rank-deficient Z_i = [1, w_i] (K_i nonzero in the joint model): equals "
      "the reference, generators verify (k = %d)" % k_of(Q3))
# shift orthogonal to K_i = span{(w_i, -1)}
orthK = max(abs(D[i] @ np.array([w[i], -1.0])) for d, D in info3["generators"] for i in range(N))
check(orthK < 1e-8, "rank-deficient Z_i: every delta_i is orthogonal to K_i (%.1e)" % orthK)

# ------------------------------- 5. Condition (S) fails: warn, do not admit
subj4 = [dict(t=np.array([1.0]), T=2.0, xrow=(lambda s: np.c_[one(s), s ** 2]),
              zrow=(lambda s: np.c_[one(s), s])) for _ in range(20)]
A4 = pack(subj4)
Q4, info4 = joint(A4)
check(info4["condition_s_residual"] > 1e-6 and info4["n_rejected_s"] == 1,
      "Condition (S) fails (t^2 vs t, one visit at t = 1): residual %.2e "
      "reported, 1 direction rejected" % info4["condition_s_residual"])
check(k_of(Q4) == 1 and in_span(Q4, np.ones(20))
      and all(abs(d[1]) < 1e-10 for d, _ in info4["generators"]),
      "Condition (S) fails: the t^2 witness is not admitted; only the intercept direction is")

# ---------------------------------------------------------- 6. q = 3
Z3 = lambda i: (lambda s: np.c_[one(s), s, s ** 2])
subj5 = subjects(N, lambda i: (lambda s: np.c_[one(s), s, s ** 2, x[i] * one(s), age[i] * one(s)]),
                 Z3, baseline_only=4, nobs=5)
A5 = pack(subj5)
Q5, info5 = joint(A5)
check(k_of(Q5) == 3 and same(Q5, ref_joint(subj5)) and same(Q5, columnwise(A5, 3))
      and generators_verify(A5, info5) < 1e-8,
      "q = 3 (random quadratic), subject-constant covariates: k = 3, equals "
      "reference and per-column (k = %d)" % k_of(Q5))
subj5b = subjects(N, lambda i: (lambda s: np.c_[one(s), s, s ** 2, x[i] * one(s), x[i] * s]),
                  Z3, baseline_only=4, nobs=5)
A5b = pack(subj5b)
Q5b, info5b = joint(A5b)
Qo5b = columnwise(A5b, 3)
check(k_of(Q5b) == 2 and same(Q5b, ref_joint(subj5b)),
      "q = 3 with x:time and baseline-only subjects: k = 2, equals reference")
check(k_of(Qo5b) == 3 and not same(Qo5b, Q5b),
      "  ... where the per-column union (consensus fill) adds a 3rd, "
      "non-absorbable direction")

# ---------------------------- 7. column-space invariance and Example 1
subj6 = subjects(N, lambda i: (lambda s: np.c_[one(s), s + s ** 2 / 2, s ** 2]), Z_int_slope)
A6 = pack(subj6)
Q6, _ = joint(A6)
G = rng.normal(size=(3, 3))
A6g = (np.einsum("nop,pk->nok", A6[0], G), A6[1], A6[2], A6[3] @ G, A6[4],
       np.einsum("nop,pk->nok", A6[5], G), A6[6])
check(same(Q6, joint(A6g)[0]), "column-space invariance: X -> X G gives the same Q")
check(k_of(Q6) == 1 and in_span(Q6, np.ones(N)) and k_of(columnwise(A6, 2)) == 1,
      "Example 1 (X = [1, t + t^2/2, t^2]): k = 1 (span{1})")

# ------------------- 8. randomized: always equals the reference; old vs new
n_rand = n_same_old = n_old_missed = n_old_extra = 0
n_aligned_clean = n_aligned_clean_same = 0
for rep in range(80):
    q = int(rng.integers(1, 4))
    Nr = int(rng.integers(15, 40))
    bo = int(rng.integers(0, 4)) if rng.random() < 0.6 else 0
    cov = rng.normal(size=(Nr, 3)) * rng.choice([1.0, 50.0, 0.02])
    kind = rng.choice(["aligned", "aligned_inter", "coupled", "poly"])
    if q == 1:
        kind = rng.choice(["aligned", "aligned_inter"])
    nc = int(rng.integers(0, 3))

    def xrow(i, kind=kind, nc=nc, cov=cov):
        def f(s):
            cols = [one(s), s]
            cols += [cov[i, j] * one(s) for j in range(nc)]
            if kind == "aligned_inter":
                cols.append(cov[i, 0] * s)
            elif kind == "coupled":
                cols.append(cov[i, 1] * (1 + s))
            elif kind == "poly":
                cols = [one(s), s + 0.3 * s ** 2, s ** 2] + cols[2:]
            return np.stack(cols, axis=1)
        return f
    zr = {1: lambda i: (lambda s: one(s)[:, None]),
          2: Z_int_slope, 3: Z3}[q]
    sj = subjects(Nr, xrow, zr, baseline_only=bo, nobs=int(rng.integers(q + 1, q + 4)))
    Ar = pack(sj)
    Qr, ir = joint(Ar)
    Rr = ref_joint(sj)
    n_rand += 1
    if not same(Qr, Rr):
        failures.append("random design %d (q=%d, %s): new != reference" % (rep, q, kind))
    if ir["generators"] and generators_verify(Ar, ir) > 1e-7:
        failures.append("random design %d: a generator fails (L)/(T)" % rep)
    Qo = columnwise(Ar, q)
    if same(Qr, Qo):
        n_same_old += 1
    else:
        missed = Qr is not None and not all(in_span(Qo, Qr[:, j]) for j in range(k_of(Qr)))
        extra = Qo is not None and not all(in_span(Qr, Qo[:, j]) for j in range(k_of(Qo)))
        n_old_missed += missed
        n_old_extra += extra
    if kind == "aligned" or (kind == "aligned_inter" and bo == 0):
        n_aligned_clean += 1
        n_aligned_clean_same += same(Qr, Qo)
check(not any(f.startswith("random design") for f in failures),
      "%d random designs (q = 1-3; subject-constant, time-interaction, coupled "
      "and polynomial columns; some baseline-only subjects): new == reference "
      "and every generator verifies" % n_rand)
check(n_aligned_clean == n_aligned_clean_same,
      "coordinate-aligned designs (no time interaction, or no baseline-only "
      "subjects): per-column == joint in all %d" % n_aligned_clean)
print("      per-column vs joint: same in %d of %d; per-column missed a "
      "direction in %d, had a non-absorbable one in %d"
      % (n_same_old, n_rand, n_old_missed, n_old_extra))

# ------------------- 9. tolerances (found in an independent code review)
# A near-witness: x_i(1 + 1e-11 t^2) passes verification at relative
# residual ~1e-10, and its tiny non-absorbable slope column must not be
# promoted to a full direction.
a9 = rng.normal(size=200)
subj9 = subjects(200, lambda i: (lambda s: np.c_[one(s), s, a9[i] * (1 + 1e-11 * s ** 2)]),
                 Z_int_slope, nobs=4)
Q9, _ = joint(pack(subj9))
check(k_of(Q9) == 2 and in_span(Q9, np.ones(200)) and in_span(Q9, a9),
      "near-witness (1e-11 t^2 term): k = 2, no promoted rounding direction (k = %d)" % k_of(Q9))
# A Condition (S) failure in a single subject out of many must be caught by
# the restriction and reported as an (S) rejection, not a verification one.
N10 = 1000
u10 = rng.normal(size=N10)
eps10 = np.zeros(N10); eps10[0] = 1e-4
subj10 = []
for i in range(N10):
    t = np.sort(rng.uniform(0, 4.5, 4))
    T = 9.0 if i == 0 else rng.uniform(4.6, 4.99)
    subj10.append(dict(t=t, T=T, zrow=Z_int_slope(i),
                       xrow=(lambda s, i=i: np.c_[one(s), s, u10[i] + eps10[i] * np.maximum(s - 5.0, 0.0)])))
Q10, info10 = joint(pack(subj10))
check(info10["n_rejected_s"] >= 1 and info10["n_rejected_verify"] == 0
      and k_of(Q10) == 1 and in_span(Q10, np.ones(N10)),
      "(S) failing in 1 of %d subjects: rejected by the restriction, reported "
      "as an (S) rejection (residual %.1e)" % (N10, info10["condition_s_residual"]))

if failures:
    raise AssertionError("%d check(s) failed:\n  " % len(failures) + "\n  ".join(failures))
print("ALL PASS")
