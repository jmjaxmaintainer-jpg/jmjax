"""Exact-Gaussian check of the rotation theory (vignette Section 4.10).

Pure NumPy; no JAX, NumPyro or R needed. Runs in 10-40 s.

WHAT THIS IS. The longitudinal submodel of jmjax's joint model with the
variance components (sigma_b, rho, sigma_e) held FIXED, so that the
conditional posterior of (beta, b_std) is exactly Gaussian and every
quantity below is computed in closed form rather than estimated from a
chain:

    y_ij = x_ij' beta + b_i0 + b_i1 t_ij + e_ij,     x_ij = (1, t_ij, age_i, sex_i)
    (b_i0, b_i1)' = L b_std_i,   L = [[s0, 0], [rho s1, sqrt(1-rho^2) s1]]
    beta ~ N(0, 5^2 I),   b_std_i ~ N(0, I_2),   e_ij ~ N(0, se^2)

with the same defaults as dev/sim_joint.R (s0 = 0.8, s1 = 0.3, se = 0.3,
rho = 0.3, age ~ N(0, 10), sex ~ Bernoulli(0.5), N = 300). The survival
submodel is omitted: by Propositions 3 and 6 it does not change which
directions are flat, only how strongly the identified ones are pinned.

WHAT IT CHECKS.
  Part 1  Proposition 9: the flat coordinates xi have posterior exactly
          N(0, I) and are uncorrelated with every other coordinate.
  Part 2  Proposition 10: the posterior moments of the generator
          coordinates (u, w) and the whitened variances R(.) of u, w and the
          flat direction xi under the oracle diagonal metric (marginal
          posterior variances - what NUTS's diagonal adaptation estimates),
          for plain orthogonalize_b0, the current column-0 rotation, and
          rotating every column; checked against the closed forms (a)-(c).
  Part 3  A stylised HMC model (exact dynamics, integration time
          T ~ U(0, 2 T0) with T0 set by the bulk of the whitened spectrum,
          step size set by its smallest eigenvalue): integrated
          autocorrelation time of beta_corrected and ESS per gradient, for
          each parameterization. Part 3 is a MODEL of NUTS, not NUTS; its
          outputs are predictions for dev/pilot_b0_rotate.R-style runs, not
          substitutes for them.

PARAMETERIZATIONS.
  A         unswept (default fit); reportable intercept is beta_0 itself
  D         orthogonalize_b0 (sweep column 0)
  D+rot0    D + orthogonalize_b0_rotate as currently implemented
            (fixed rotation of column 0 only)
  E         orthogonalize_b (sweep columns 0 and 1)
  E+rot0    E + the current column-0 rotation
  E+rotall  E + the SAME fixed rotation applied to every column (proposed)
Completions of Q are Householder (np.linalg.qr(Q, "complete")); the
current implementation's random completion gives the same results to
within 1% here (checked in Part 4), so the choice between them is about
cost (O(N k) vs O(N^2)), not mixing.
"""
import itertools
import numpy as np


def design(N=300, seed=1, nmin=2, nmax=11):
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 10, N)
    sex = rng.binomial(1, .5, N).astype(float)
    n = rng.integers(nmin, nmax + 1, N)
    sid = np.repeat(np.arange(N), n)
    t = np.concatenate([np.arange(k, dtype=float) for k in n])
    X = np.column_stack([np.ones_like(t), t, age[sid], sex[sid]])
    return dict(N=N, sid=sid, t=t, X=X, age=age, sex=sex, n=n)


def intercept_basis(d):
    """Q0 (orthonormal basis of span{1, age, sex}) and Gamma0 with
    X_i Gamma0 c = (Q0 c)_i * 1 for every subject (Definition 2)."""
    N = d["N"]
    S = np.column_stack([np.ones(N), d["age"], d["sex"]])
    Q, R = np.linalg.qr(S)
    G = np.zeros((4, 3))
    G[[0, 2, 3], :] = np.linalg.inv(R)
    return Q, G


def complete(Q, how="householder", seed=0):
    N, k = Q.shape
    if how == "householder":
        return np.linalg.qr(Q, mode="complete")[0][:, k:]
    M = np.concatenate([Q, np.random.default_rng(seed).standard_normal((N, N - k))], 1)
    return np.linalg.qr(M)[0][:, k:]


def posterior(d, sweep0, sweep1, rot, rho=0.3, s0=0.8, s1=0.3, se=0.3, sb=5.0,
              completion="householder"):
    """Exact posterior covariance of theta and the two reportable
    functionals beta_corrected[intercept], beta_corrected[time] as vectors
    a with f = a' theta. theta = (beta[4], col0 block[N], col1 block[N]),
    where each column block is b_std[:, c] itself, or (U_c, V_c) =
    (Q0' b_std[:, c], Qperp' b_std[:, c]) when that column is rotated."""
    N, X, sid, t = d["N"], d["X"], d["sid"], d["t"]
    M = len(t)
    Q0, G0 = intercept_basis(d)
    Q1 = np.ones((N, 1)) / np.sqrt(N)           # slope column: only 'time' absorbs
    G1 = np.zeros((4, 1)); G1[1, 0] = 1 / np.sqrt(N)
    R = np.concatenate([Q0, complete(Q0, completion)], 1)
    s = np.sqrt(1 - rho ** 2)
    E = np.zeros((M, N)); E[np.arange(M), sid] = 1.0
    Z1 = E * t[:, None]
    P0 = np.eye(N) - Q0 @ Q0.T if sweep0 else np.eye(N)
    P1 = np.eye(N) - Q1 @ Q1.T if sweep1 else np.eye(N)
    # b_raw0 = s0 bstd0, b_raw1 = s1 (rho bstd0 + s bstd1); swept columns projected
    J = np.concatenate([X, E @ (s0 * P0) + Z1 @ (P1 * rho * s1), Z1 @ (P1 * s * s1)], 1)
    A = np.zeros((2, 4 + 2 * N)); A[0, 0] = A[1, 1] = 1.0
    if sweep0:                                   # Proposition 6's correction
        A[:, 4:4 + N] -= (G0 @ (s0 * Q0.T))[[0, 1]]
    if sweep1:
        C = G1 @ Q1.T * s1
        A[:, 4:4 + N] -= (C * rho)[[0, 1]]
        A[:, 4 + N:] -= (C * s)[[0, 1]]
    T = np.eye(4 + 2 * N)
    if rot in ("col0", "all"):
        T[4:4 + N, 4:4 + N] = R
    if rot == "all":
        T[4 + N:, 4 + N:] = R
    Jt = J @ T
    prec = np.diag(np.r_[np.full(4, sb ** -2), np.ones(2 * N)]) + Jt.T @ Jt / se ** 2
    Sig = np.linalg.inv(prec)
    return 0.5 * (Sig + Sig.T), T.T @ A[0], T.T @ A[1]


def flat_basis(N, k, rot, rho):
    """Orthonormal basis (theta coordinates) of the flat subspace F(theta)
    for the intercept sweep: per generator m, the unit direction
    s * (col0 along q_m) - rho * (col1 along q_m)  (Corollary 3)."""
    s = np.sqrt(1 - rho ** 2)
    F = np.zeros((4 + 2 * N, k))
    if rot == "all":
        F[4:4 + k, :] = s * np.eye(k)
        F[4 + N:4 + N + k, :] = -rho * np.eye(k)
        return F
    raise ValueError("flat_basis only needed for the rotate-all check")


def hmc_model(Sig, a, T_scale="bulk"):
    """Stylised HMC on N(0, Sig) with oracle diagonal mass diag(Sig).
    Returns IAT of f = a' theta, mean leapfrog steps per iteration, and
    ESS per 1000 gradients."""
    m = np.diag(Sig)
    S = Sig / np.sqrt(np.outer(m, m))            # whitened covariance
    lam, V = np.linalg.eigh(S)
    c = V.T @ (np.sqrt(m) * a)
    w = c ** 2 * lam
    T0 = (np.pi / 2) * np.sqrt(np.median(lam))
    x = 2 * T0 / np.sqrt(lam)
    r = np.sin(x) / x                            # E cos(omega T), T ~ U(0, 2 T0)
    tau = np.sum(w * (1 + r) / (1 - r)) / np.sum(w)
    steps = T0 / np.sqrt(lam.min())
    return tau, steps, 1000.0 / (tau * steps)


def R_(Sig, delta):
    """Whitened variance of one direction under the oracle diagonal metric."""
    m = np.diag(Sig)
    return (delta @ Sig @ delta) / (delta @ (m * delta))


def lam_F(Sig, F):
    """Lambda_F = max over directions delta in span(F) of
    (delta' Sig delta) / (delta' M delta), M = diag(Sig) - a generalized
    eigenvalue problem on the small (dim F) block."""
    m = np.diag(Sig)
    return np.max(np.real(np.linalg.eigvals(
        np.linalg.solve((F.T * m) @ F, F.T @ Sig @ F))))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    N, k = 300, 3

    print("PART 1 - Proposition 9: flat coordinates are exactly N(0, I), "
          "independent of the rest (rotate-all, intercept sweep)")
    for (nmin, nmax), rho in itertools.product([(2, 11), (8, 20)], [0.0, 0.3, 0.6, 0.9]):
        d = design(N=N, nmin=nmin, nmax=nmax)
        Sig, a0, _ = posterior(d, 1, 0, "all", rho=rho)
        F = flat_basis(N, k, "all", rho)
        rest = np.ones(Sig.shape[0], bool); rest[np.r_[4:4 + k, 4 + N:4 + N + k]] = False
        E_ = np.zeros((Sig.shape[0], k)); s = np.sqrt(1 - rho ** 2)
        E_[4:4 + k] = rho * np.eye(k); E_[4 + N:4 + N + k] = s * np.eye(k)   # eta
        print(f"  n={nmin:2d}-{nmax:2d} rho={rho:.1f}: max|Cov(xi)-I|={np.abs(F.T @ Sig @ F - np.eye(k)).max():.1e}"
              f"  max|Cov(xi, other coords)|={np.abs(F.T @ Sig[:, rest]).max():.1e}"
              f"  max|Cov(xi, eta)|={np.abs(F.T @ Sig @ E_).max():.1e}"
              f"  flat share of Var(beta_corrected[int])={np.sum((F.T @ a0) ** 2) / (a0 @ Sig @ a0):.3f}")

    print("\nPART 2 - Proposition 10: generator-plane moments and whitened "
          "variances R(.) under the oracle diagonal metric, vs closed forms")
    print("  (per generator direction g = Q0[:, m], worst m over age/sex/mean;"
          " 'err' = max abs deviation of the closed forms from the exact values)")
    print("  design          rho | R(u)  D    rot0  all | R(w)  D    rot0  all | R(xi) D    rot0 <=1/s^4  all   | plane eig max all  1+|r| | err")
    worst_err = 0.0
    for (nmin, nmax), se, rho in itertools.product([(2, 4), (2, 11), (8, 20)], [0.3, 0.8], [0.0, 0.3, 0.6, 0.9]):
        d = design(N=N, nmin=nmin, nmax=nmax)
        s = np.sqrt(1 - rho ** 2)
        Q0, _ = intercept_basis(d)
        rows = {}
        for rot in (None, "col0", "all"):
            Sig, _, _ = posterior(d, 1, 0, rot, rho=rho, se=se)
            rows[rot] = Sig
        res = []
        for mm in range(k):
            g = Q0[:, mm]
            vec = {}
            for rot, Sig in rows.items():
                dim = Sig.shape[0]; u = np.zeros(dim); w = np.zeros(dim)
                if rot is None:
                    u[4:4 + N] = g; w[4 + N:] = g
                elif rot == "col0":
                    u[4 + mm] = 1; w[4 + N:] = g
                else:
                    u[4 + mm] = 1; w[4 + N + mm] = 1
                vec[rot] = (Sig, u, w)
            SigD, uD, wD = vec[None]
            mD = np.diag(SigD)
            lam = (rho * uD + s * wD) @ SigD @ (rho * uD + s * wD)
            # exact moments vs closed forms
            e = max(abs(uD @ SigD @ uD - (s**2 + rho**2 * lam)),
                    abs(wD @ SigD @ wD - (rho**2 + s**2 * lam)),
                    abs(uD @ SigD @ wD + rho * s * (1 - lam)))
            gm0 = g @ (mD[4:4 + N] * g); gm1 = g @ (mD[4 + N:] * g)
            Ru = [R_(vec[r][0], vec[r][1]) for r in (None, "col0", "all")]
            Rw = [R_(vec[r][0], vec[r][2]) for r in (None, "col0", "all")]
            Rx = [R_(vec[r][0], s * vec[r][1] - rho * vec[r][2]) for r in (None, "col0", "all")]
            e = max(e, abs(Ru[0] - (s**2 + rho**2 * lam) / gm0), abs(Rw[0] - (rho**2 + s**2 * lam) / gm1),
                    abs(Rx[0] - 1 / (s**2 * gm0 + rho**2 * gm1)), abs(Ru[1] - 1), abs(Rw[1] - Rw[0]))
            SigA, uA, wA = vec["all"]
            C = np.array([[uA @ SigA @ uA, uA @ SigA @ wA], [uA @ SigA @ wA, wA @ SigA @ wA]])
            r = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
            rpred = -rho * s * (1 - lam) / np.sqrt((s**2 + rho**2 * lam) * (rho**2 + s**2 * lam))
            pe = np.linalg.eigvalsh(C / np.sqrt(np.outer(np.diag(C), np.diag(C)))).max()
            e = max(e, abs(r - rpred), abs(pe - (1 + abs(r))))
            assert Rx[1] <= 1 / s**4 + 1e-9
            res.append((Ru, Rw, Rx, pe, 1 + abs(r), e))
        worst_err = max(worst_err, max(x[5] for x in res))
        mx = lambda f: max(f(x) for x in res)
        print(f"  n={nmin:2d}-{nmax:2d} se={se} {rho:.1f} | {mx(lambda x: x[0][0]):6.2f} {mx(lambda x: x[0][1]):5.2f} {mx(lambda x: x[0][2]):4.2f}"
              f" | {mx(lambda x: x[1][0]):6.2f} {mx(lambda x: x[1][1]):5.2f} {mx(lambda x: x[1][2]):4.2f}"
              f" | {mx(lambda x: x[2][0]):6.2f} {mx(lambda x: x[2][1]):5.2f} {1 / s**4:6.2f} {mx(lambda x: x[2][2]):5.2f}"
              f" |      {mx(lambda x: x[3]):.3f}        {mx(lambda x: x[4]):.3f} | {max(x[5] for x in res):.1e}")
    print(f"  worst deviation of any closed form from the exact value: {worst_err:.1e}")

    print("\nPART 3 - stylised HMC: ESS per 1000 gradients (IAT in iterations)")
    arms = {"A": (0, 0, None), "D": (1, 0, None), "D+rot0": (1, 0, "col0"),
            "E": (1, 1, None), "E+rot0": (1, 1, "col0"), "E+rotall": (1, 1, "all")}
    for label, idx in (("beta_corrected[intercept]", 0), ("beta_corrected[time]", 1)):
        print(f"  {label}")
        print("  design        rho  " + "".join(f"{a:>15s}" for a in arms))
        for (nmin, nmax), rho in itertools.product([(2, 11), (8, 20)], [0.0, 0.3, 0.6, 0.9]):
            d = design(N=N, nmin=nmin, nmax=nmax)
            cells = []
            for nm, (s0_, s1_, r) in arms.items():
                Sig, a0, a1 = posterior(d, s0_, s1_, r, rho=rho)
                tau, steps, ess = hmc_model(Sig, a0 if idx == 0 else a1)
                cells.append(f"{ess:7.2f} ({tau:5.1f})")
            print(f"  n={nmin:2d}-{nmax:2d}  {rho:.1f}  " + "".join(f"{c:>15s}" for c in cells))

    print("\nPART 4 - N-invariance and completion choice (n=2-11, rho=0.3): D+rot0 vs D, ESS/grad ratio")
    for Nn in (150, 300, 600):
        d = design(N=Nn)
        _, _, eD = hmc_model(*posterior(d, 1, 0, None)[:2])
        for how in ("householder", "random"):
            _, _, eR = hmc_model(*posterior(d, 1, 0, "col0", completion=how)[:2])
            print(f"  N={Nn:4d} {how:12s}: {eR / eD:5.2f}x")
