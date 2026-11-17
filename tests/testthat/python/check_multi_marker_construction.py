"""Checks of the absorbable construction for several longitudinal markers
(absorbable._absorbable_basis_multi; jmjax 0.5.0, Phase B).

    python3 check_multi_marker_construction.py <path-to-inst/python>

numpy only (absorbable.py is loaded by path). With X_i and Z_i block-diagonal
over markers and beta stacked, the joint construction of the methods paper's
Section 3 applies unchanged; this checks
  1. the union over markers against a reference built from the paper's
     reference implementation (dense grid on [0, tau]), including the
     Milestone 1 spec's example: two markers, each (1, time, age, sex) with a
     random intercept and slope -> k = 3, q = 4, dense block 8 + 12 = 20;
  2. Lemma 1 (localization) under a random FULL cross-marker L: for every
     generator, the standardized shift Delta(d) (L')^{-1} lies in span(Q)
     for the union Q, column by column, and NOT for per-marker bases Q_k
     applied to each marker's own columns;
  3. that per-marker bases do satisfy Lemma 1 when L is block-diagonal;
  4. markers observed at different times (different visit grids).
Raises AssertionError on failure; prints one line per check.
"""
import importlib.util
import os
import sys
import warnings

import numpy as np

_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "inst", "python")
_spec = importlib.util.spec_from_file_location(
    "jmjax_absorbable_check_multi", os.path.join(_root, "jmjax_backend", "absorbable.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_absorbable_basis_multi = _mod._absorbable_basis_multi
_absorbable_basis_joint = _mod._absorbable_basis_joint

rng = np.random.default_rng(20261116)
TAU = 10.0
GRID = np.linspace(0.0, TAU, 201)
GL = np.polynomial.legendre.leggauss(7)[0]
failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def orth(A, tol=1e-9):
    if A is None or A.size == 0:
        return None
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int((s > tol * max(1.0, s.max())).sum())
    return u[:, :r] if r else None


def same(A, B, tol=1e-8):
    if A is None or B is None:
        return A is None and B is None
    return A.shape == B.shape and float(np.abs(A @ A.T - B @ B.T).max()) < tol


def resid(Qm, v):
    if Qm is None:
        return float(np.linalg.norm(v))
    return float(np.linalg.norm(v - Qm @ (Qm.T @ v)))


class Design:
    """K markers; marker k has xrow_k(i, t) [len(t), p_k], zrow_k(t) [len(t), q_k]
    and its own visit times per subject. Builds the flat block-diagonal layout
    the package uses and a dense-grid reference."""

    def __init__(self, N, markers, visits):
        self.N, self.markers, self.visits = N, markers, visits
        self.p = [m["p"] for m in markers]
        self.q = [m["q"] for m in markers]
        self.P, self.Q = sum(self.p), sum(self.q)
        self.po = np.concatenate([[0], np.cumsum(self.p)])
        self.qo = np.concatenate([[0], np.cumsum(self.q)])
        self.T = rng.uniform(6.0, TAU, N)

    def block(self, k, i, t):
        t = np.atleast_1d(t)
        Xb = np.zeros((len(t), self.P)); Zb = np.zeros((len(t), self.Q))
        Xb[:, self.po[k]:self.po[k + 1]] = self.markers[k]["x"](i, t)
        Zb[:, self.qo[k]:self.qo[k + 1]] = self.markers[k]["z"](t)
        return Xb, Zb

    def flat(self):
        X, Z, subj = [], [], []
        for i in range(self.N):
            for k in range(len(self.markers)):
                Xb, Zb = self.block(k, i, self.visits[k][i])
                X.append(Xb); Z.append(Zb); subj += [i] * len(Xb)
        return np.vstack(X), np.vstack(Z), np.asarray(subj)

    def channels(self):
        C = len(self.markers)
        XS = np.zeros((C, self.N, self.P)); ZS = np.zeros((C, self.N, self.Q))
        XQ = np.zeros((C, self.N, len(GL), self.P)); ZQ = np.zeros((C, self.N, len(GL), self.Q))
        for k in range(C):
            for i in range(self.N):
                xs, zs = self.block(k, i, self.T[i])
                tq = self.T[i] * (GL + 1) / 2
                xq, zq = self.block(k, i, tq)
                XS[k, i], ZS[k, i], XQ[k, i], ZQ[k, i] = xs[0], zs[0], xq, zq
        return XS, ZS, XQ, ZQ

    def reference(self):
        """Paper reference, block-diagonal: M over all markers' rows, (T) on
        a dense grid for every marker; returns Q and the generators."""
        rows = []
        for i in range(self.N):
            Xi = np.vstack([self.block(k, i, self.visits[k][i])[0] for k in range(len(self.markers))])
            Zi = np.vstack([self.block(k, i, self.visits[k][i])[1] for k in range(len(self.markers))])
            rows.append((np.eye(len(Zi)) - Zi @ np.linalg.pinv(Zi)) @ Xi)
        M = np.vstack(rows)
        _, s, vt = np.linalg.svd(M)
        D = vt[int((s > 1e-9 * max(1.0, s.max())).sum()):].T
        cols, gens = [], []
        for d in D.T:
            Delta = []
            for i in range(self.N):
                Xg = np.vstack([self.block(k, i, GRID)[0] for k in range(len(self.markers))])
                Zg = np.vstack([self.block(k, i, GRID)[1] for k in range(len(self.markers))])
                Delta.append(-np.linalg.pinv(Zg) @ (Xg @ d))
            Delta = np.array(Delta)
            gens.append((d, Delta))
            cols += [Delta[:, c] / np.linalg.norm(Delta[:, c]) for c in range(self.Q)
                     if np.linalg.norm(Delta[:, c]) > 1e-9 * max(np.linalg.norm(Delta), 1e-300)]
        return orth(np.array(cols).T) if cols else None, gens

    def package(self):
        X, Z, subj = self.flat()
        XS, ZS, XQ, ZQ = self.channels()
        return _absorbable_basis_multi(X, Z, subj, self.N, XS, ZS, XQ, ZQ)


def visits_regular(N, gap, offset=0.0):
    return [np.arange(offset, 6.0, gap) for _ in range(N)]


def visits_random(N, lo, hi):
    return [np.sort(rng.uniform(0, 6, int(rng.integers(lo, hi + 1)))) for _ in range(N)]


one = lambda t: np.ones_like(t)
N = 50
age, sex = rng.normal(size=N), rng.integers(0, 2, N).astype(float)
z_int_slope = lambda t: np.c_[one(t), t]

# ---- 1. the spec's example: two markers, each (1, time, age, sex) -------
mk = dict(p=4, q=2, x=lambda i, t: np.c_[one(t), t, age[i] * one(t), sex[i] * one(t)], z=z_int_slope)
des = Design(N, [mk, mk], [visits_regular(N, 1.0), visits_random(N, 2, 5)])
Qp, info = des.package()
Qr, gens_ref = des.reference()
k = 0 if Qp is None else Qp.shape[1]
check(k == 3 and des.Q == 4 and same(Qp, Qr),
      "two markers, each (1, time, age, sex) + random intercept/slope: union k = %d, "
      "q = %d, equals the reference" % (k, des.Q))
check(des.P + k * des.Q == 20, "dense block (beta, b_gen_U) has p + k q = %d + %d = 20 coordinates"
      % (des.P, k * des.Q))
check(info["condition_s_residual"] < 1e-8 and info["n_rejected_s"] == 0,
      "Condition (S) holds on every channel (residual %.1e)" % info["condition_s_residual"])

# ---- 2. Lemma 1 under a random FULL L: union yes, per-marker no ----------
# Marker 1 has age, marker 2 does not, so A_1 = span{1, age}, A_2 = span{1}.
mk1 = dict(p=3, q=2, x=lambda i, t: np.c_[one(t), t, age[i] * one(t)], z=z_int_slope)
mk2 = dict(p=2, q=2, x=lambda i, t: np.c_[one(t), t], z=z_int_slope)
des2 = Design(N, [mk1, mk2], [visits_regular(N, 1.0), visits_regular(N, 1.5, 0.25)])
Qu, info2 = des2.package()
Qr2, gens2 = des2.reference()
check(Qu is not None and Qu.shape[1] == 2 and same(Qu, Qr2),
      "markers with different covariates: union = span{1, age}, k = 2, equals the reference")
# Per-marker bases, each from a single-marker construction.
single = []
for kk, m in enumerate([mk1, mk2]):
    d1 = Design(N, [m], [des2.visits[kk]]); d1.T = des2.T
    single.append(d1.package()[0])
check(single[0].shape[1] == 2 and single[1].shape[1] == 1,
      "per-marker bases: Q_1 = span{1, age} (k = 2), Q_2 = span{1} (k = 1)")
A = rng.normal(size=(4, 4))
L_full = np.linalg.cholesky(A @ A.T + 4 * np.eye(4))
L_block = L_full.copy(); L_block[:2, 2:] = 0; L_block[2:, :2] = 0
L_block = np.linalg.cholesky(L_block @ L_block.T)
for Lname, L in (("full", L_full), ("block-diagonal", L_block)):
    worst_union = worst_per = 0.0
    for _, Delta in gens2:
        S = Delta @ np.linalg.inv(L.T)          # standardized shift [N, Q]
        for c in range(4):
            v = S[:, c]
            scale = max(np.linalg.norm(Delta), 1e-300)
            worst_union = max(worst_union, resid(Qu, v) / scale)
            worst_per = max(worst_per, resid(single[0 if c < 2 else 1], v) / scale)
    if Lname == "full":
        check(worst_union < 1e-8,
              "Lemma 1, full L: every standardized shift column lies in span(Q_union) "
              "(worst %.1e)" % worst_union)
        check(worst_per > 1e-3,
              "Lemma 1, full L: per-marker bases FAIL - a marker-1 shift appears in "
              "marker 2's standardized columns outside span(Q_2) (worst %.2f)" % worst_per)
    else:
        check(worst_union < 1e-8 and worst_per < 1e-8,
              "Lemma 1, block-diagonal L: both the union and the per-marker bases hold "
              "(worst %.1e, %.1e)" % (worst_union, worst_per))

# ---- 3. markers at different times, a quadratic marker, random designs ---
n_ok = n_all = 0
for rep in range(25):
    Nr = int(rng.integers(20, 40))
    cov = rng.normal(size=(Nr, 2)) * rng.choice([1.0, 30.0])
    Ks = int(rng.integers(2, 4))
    ms, vs = [], []
    for kk in range(Ks):
        kind = rng.choice(["lin", "quad", "cov", "coupled"])
        if kind == "quad":
            ms.append(dict(p=3, q=3, x=lambda i, t: np.c_[one(t), t, t ** 2],
                           z=lambda t: np.c_[one(t), t, t ** 2]))
        elif kind == "cov":
            ms.append(dict(p=3, q=2, x=lambda i, t, cov=cov: np.c_[one(t), t, cov[i, 0] * one(t)],
                           z=z_int_slope))
        elif kind == "coupled":
            ms.append(dict(p=3, q=2, x=lambda i, t, cov=cov: np.c_[one(t), t, cov[i, 1] * (1 + t)],
                           z=z_int_slope))
        else:
            ms.append(dict(p=2, q=1, x=lambda i, t: np.c_[one(t), t], z=lambda t: one(t)[:, None]))
        vs.append(visits_random(Nr, ms[-1]["q"] + 1, ms[-1]["q"] + 4))
    dr = Design(Nr, ms, vs)
    Qp_r, _ = dr.package()
    Qr_r, _ = dr.reference()
    n_all += 1
    n_ok += same(Qp_r, Qr_r)
check(n_ok == n_all, "%d random multi-marker designs (2-3 markers; linear, quadratic, covariate "
      "and coupled markers; different visit times): package == reference in %d" % (n_all, n_ok))

if failures:
    raise AssertionError("%d check(s) failed:\n  " % len(failures) + "\n  ".join(failures))
print("ALL PASS")
