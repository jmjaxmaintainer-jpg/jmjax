"""Absorbable directions of the random effects, and the rotation built on them.

The location degeneracy of vignette("jmjax-reparameterization"), Section 3:
a shift of a random-effect column along an ABSORBABLE direction v can be
undone exactly by a shift of the fixed effects, so the likelihood cannot
separate the two. This module finds those directions (Section 4: the
per-column search, the exact basis-independent construction, and the
Condition (S) check on the survival/quadrature grid), and builds the fixed
orthogonal rotation that jmjax uses to make them explicit sampling
coordinates (control$rotate_absorbable; Householder reflectors, O(N k)).

From 0.4.0 the rotation uses _absorbable_basis_joint(), which defines the
absorbable subspace jointly across random-effect columns (methods paper,
Section 3) and solves each subject's shift from the trajectory identity on
the survival/quadrature grid. The per-column functions below remain for the
legacy sweep (sweep.py) and as the comparison the fit report records.

Split out of mcmc_model.py so that the absorbable-basis machinery, which the
recommended rotation needs, is separate from the legacy sweep in sweep.py,
which it does not. Depends only on NumPy; _apply_reflectors also works on
JAX arrays unchanged.
"""
import warnings

import numpy as np


def _absorbable_basis(X_long, Z_long, n_obs, q_idx, tol=1e-8):
    """Basis for the directions of random-effect column `q_idx` that the
    fixed effects can absorb.

    A shift of b[:, q] by s_i (subject-constant) changes the linear
    predictor by s_i * Z[:, :, q]. That shift is absorbed by beta - leaving
    fitted values unchanged - exactly when some column j of X satisfies

        X[:, :, j]  ==  s_i * Z[:, :, q]      for a subject-constant s_i

    so the absorbable space is spanned by the s_i vectors of every such
    column. The data decide which columns qualify; nothing is assumed about
    the formula.

    For q = 0 the random-effect column is the constant 1, so the condition
    reduces to "column j is constant within subject" and this recovers the
    intercept case. For q = 1 (a random slope on time) it picks up `time`
    itself, giving the sum-to-zero constraint on b_1, and would also pick up
    any covariate-by-time interaction present in X.

    Returns (Q, kept) with Q an [N_sub, k] orthonormal basis, or (None, []).

    Computed in fit_nuts() where X_long and Z_long are concrete arrays:
    inside the traced model "which columns qualify" would be a tracer and
    could not select columns at all.
    """
    X = np.asarray(X_long, dtype=float)
    Z = np.asarray(Z_long, dtype=float)
    if X.ndim != 3 or Z.ndim != 3 or q_idx >= Z.shape[2]:
        return None, []
    N_sub, max_obs, p = X.shape
    n = np.asarray(n_obs).astype(int).reshape(-1)
    if n.shape[0] != N_sub or N_sub == 0 or p == 0:
        return None, []

    mask = np.arange(max_obs)[None, :] < n[:, None]          # [N_sub, max_obs]
    zq   = Z[:, :, q_idx]                                     # [N_sub, max_obs]
    nz   = mask & (np.abs(zq) > tol)                          # usable cells
    zsc  = max(float(np.nanmax(np.abs(zq[mask]))) if mask.any() else 1.0, 1.0)

    keep, svals = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(p):
            xj = X[:, :, j]
            # Wherever Z_q is ~0 the column must be ~0 too, or no constant
            # s_i can reproduce it.
            bad = mask & ~nz & (np.abs(xj) > tol * max(
                float(np.nanmax(np.abs(xj[mask]))) if mask.any() else 1.0, 1.0))
            if bad.any():
                continue
            r = np.where(nz, xj / np.where(np.abs(zq) > tol, zq, 1.0), np.nan)
            rmin = np.nanmin(r, axis=1); rmax = np.nanmax(r, axis=1)
            spread = np.nanmax(rmax - rmin)
            if not np.isfinite(spread):
                continue
            scale = max(float(np.nanmax(np.abs(rmax))), 1.0)
            if spread <= tol * scale * zsc:
                # BUG FIX: a subject with no cell where |Z_q| > tol (e.g.
                # the only observation is at Z_q = 0, such as a baseline-
                # only trajectory when q is the time-slope column) has no
                # row from which to determine s_i at all - rmin/rmax are
                # NaN for them, not "the ratio is 0". Their own data is
                # silent on s_i, not evidence for s_i = 0: with Z_iq = 0
                # everywhere for that subject, ANY s_i leaves their fitted
                # values unchanged, so imputing the consensus value from
                # the subjects whose data DOES determine it (already known,
                # by the spread check just above, to agree with each other
                # to within tol) keeps the basis column the single uniform
                # direction it is supposed to be. The old fallback (0.0)
                # silently broke that uniformity - see
                # dev/diag_slope_projection.R: at n=500 with several
                # baseline-only subjects, Q[:,0] for the slope column came
                # back non-uniform (sd 5.3e-3, some entries exactly 0)
                # instead of constant, so the projection removed a
                # subject-WEIGHTED quantity instead of the simple mean,
                # and mean_i(b_i1) was left as large as 0.04-0.07 instead
                # of pinned to ~1e-16 like the intercept column always was.
                #
                # THE STRONGER REASON (Proposition 5, Section 4.6 of
                # vignette("jmjax-reparameterization")). Restoring
                # uniformity is the visible symptom, not the actual
                # justification, and stating only the symptom invites a
                # future reader to "simplify" this back. Read through the
                # LONGITUDINAL submodel alone, a silent subject's s_i is
                # genuinely free - any value leaves their fitted values
                # unchanged, which is what the paragraph above says. But b
                # also reaches the likelihood through the shared trajectory
                # m_i(t) at the event time and the quadrature nodes, and
                # there Z_q is generally NOT zero for that subject. The
                # compensating beta shift is already pinned by the subjects
                # whose data does determine the ratio, so the silent
                # subject's entry is forced to that same consensus value -
                # it is not free at all in the JOINT model. The old 0.0
                # fallback therefore produced a direction that is not
                # absorbable, i.e. one the survival submodel does identify,
                # and sweeping it changed the model rather than only its
                # coordinates. The median over the determined subjects is
                # the unique choice consistent with joint-model invariance.
                det = np.isfinite(rmin)
                s_fill = float(np.nanmedian(rmin)) if det.any() else 0.0
                s_i = np.where(det, rmin, s_fill)
                keep.append(int(j)); svals.append(s_i)
    if not keep:
        return None, []

    S = np.stack(svals, axis=1)                               # [N_sub, k]
    Q, R = np.linalg.qr(S)
    r = np.abs(np.diag(R))
    if r.size == 0:
        return None, []
    good = r > tol * max(float(r.max()), 1.0)
    Q = np.ascontiguousarray(Q[:, good], dtype=float)
    if Q.shape[1] == 0:
        return None, []
    return Q, [keep[i] for i, g in enumerate(good) if g]


def _absorbable_basis_exact(X_long, Z_long, n_obs, q_idx, tol=1e-8):
    """Basis-INDEPENDENT absorbable basis for random-effect column `q_idx`.

    WHY THIS EXISTS ALONGSIDE _absorbable_basis(). The column-wise search
    above asks, of each fixed-effects column separately, "is this column a
    subject-constant multiple of Z_q?". But a shift in b[:, q] is absorbed
    by beta whenever SOME LINEAR COMBINATION of X's columns reproduces it -
    the absorbable set depends only on the column SPACE of X, not on which
    basis for that space model.matrix() happened to emit. Those are not the
    same question, and the gap is not academic: with

        X = [1, t + t^2/2, t^2]     (span{1, t, t^2}, i.e. what poly() or
                                     ns() emit for a smooth longitudinal
                                     mean)

    no single column is a subject-constant multiple of `time`, so the
    column-wise search returns NOTHING for the slope column - yet
    d = (0, 1, -1/2) gives X d = t exactly, so mean(b_i1) is absorbable and
    the beta_1 location degeneracy is fully present. Worse, the failure is
    silent: the intercept column still yields a basis, so the "no absorbable
    direction" warning does not fire either. See
    vignette("jmjax-reparameterization"), Section 4.5 (Example 1).

    HOW. v is absorbable iff there is a beta-shift d with A_i d = v_i z_i
    for every subject i, where A_i is subject i's fixed-effects block and
    z_i its Z_q column. Read as a condition on d, that says A_i d must lie
    in span(z_i) for every i - a condition in R^p, NOT in R^N_sub. So

        D = ker( stack_i [ (I - P_i) A_i ] ),   P_i = proj onto span(z_i)

    is the null space of a [sum_i n_i, p] matrix (p is small: one SVD, no
    loop over columns, 0.06s at N_sub = 8000), and each d in a basis of D
    induces the absorbable direction v_i = (z_i^T A_i d) / (z_i^T z_i).
    Subjects with z_i = 0 contribute the constraint A_i d = 0 (their own
    data cannot determine a ratio) and take the consensus ratio, for the
    reason given in Proposition 5 of the vignette - the survival submodel
    DOES identify their entry even when the longitudinal one does not.

    Every candidate direction is verified against the defining identity
    before being returned, so the result is sound by construction rather
    than by tolerance choice: a direction that only NEARLY satisfies
    A_i d = v_i z_i is discarded rather than swept.

    Returns (Q, gens) with Q an [N_sub, k] orthonormal basis, or (None, []).
    `gens` is the list of (v, d) generator pairs, used by
    _structural_extension_residual() to check Condition (S).
    """
    X = np.asarray(X_long, dtype=float)
    Z = np.asarray(Z_long, dtype=float)
    if X.ndim != 3 or Z.ndim != 3 or q_idx >= Z.shape[2]:
        return None, []
    N_sub, max_obs, p = X.shape
    n = np.asarray(n_obs).astype(int).reshape(-1)
    if n.shape[0] != N_sub or N_sub == 0 or p == 0:
        return None, []

    mask = (np.arange(max_obs)[None, :] < n[:, None]).astype(float)
    flat = mask.reshape(-1) > 0
    if int(flat.sum()) < p:
        return None, []                      # fewer observations than columns
    zq = Z[:, :, q_idx] * mask
    Xm = X * mask[:, :, None]

    c = np.einsum("no,no->n", zq, zq)                     # z_i^T z_i
    num = np.einsum("no,nop->np", zq, Xm)                 # z_i^T A_i
    cmax = max(float(c.max()) if c.size else 1.0, 1.0)
    det = c > (tol ** 2) * cmax              # subjects whose data fixes v_i
    ratio = np.zeros((N_sub, p))
    ratio[det] = num[det] / c[det][:, None]

    # (I - P_i) A_i, which reduces to A_i exactly where z_i = 0.
    C = (Xm - zq[:, :, None] * ratio[:, None, :]).reshape(N_sub * max_obs, p)[flat]
    X_f = Xm.reshape(N_sub * max_obs, p)[flat]
    colnorm = np.linalg.norm(X_f, axis=0)    # so the rank test is unit-free
    scale = np.where(colnorm > tol, colnorm, 1.0)

    # full_matrices=False keeps U at [M, p] rather than [M, M]; M can be
    # ~10^5, so the full form would allocate hundreds of GB.
    _, sv, Vt = np.linalg.svd(C / scale, full_matrices=False)
    smax = float(sv.max()) if sv.size else 0.0
    rank = int((sv > tol * max(smax, 1.0)).sum())
    if rank >= p:
        return None, []
    D = Vt[rank:].T / scale[:, None]                      # [p, dim D]

    zq_f = zq.reshape(-1)[flat]
    subj = np.repeat(np.arange(N_sub), max_obs)[flat]
    gens = []
    for k in range(D.shape[1]):
        d = D[:, k]
        v = ratio @ d
        if det.any() and not det.all():
            v = np.where(det, v, float(np.median(ratio[det] @ d)))
        if np.abs(v).max() <= tol:
            continue
        lhs = X_f @ d
        rhs = v[subj] * zq_f
        denom = max(float(np.abs(rhs).max()), float(np.abs(lhs).max()), 1.0)
        if float(np.abs(lhs - rhs).max()) / denom > 1e-7:
            continue                         # not genuinely absorbable
        gens.append((v, d))
    if not gens:
        return None, []

    S = np.stack([g[0] for g in gens], axis=1)
    Q, R = np.linalg.qr(S)
    r = np.abs(np.diag(R))
    good = r > tol * max(float(r.max()), 1.0)
    Q = np.ascontiguousarray(Q[:, good], dtype=float)
    if Q.shape[1] == 0:
        return None, []
    return Q, gens


def _structural_extension_residual(gens, q_idx, X_time_surv, Z_time_surv,
                                    X_time_quad, Z_time_quad):
    """Check Condition (S): does absorbability extend off the observed grid?

    The ratio test runs on the longitudinal grid {t_ij}, but `b` also
    reaches the likelihood through the shared trajectory m_i(t), evaluated
    at the event time T_i and at the quadrature nodes. Invariance there is
    a SEPARATE requirement: it needs f_j(t) = s_i g_q(t) as an identity in
    t, not merely at the observed times. It holds by construction for every
    column type this option targets (a subject-constant column, `time`
    itself, a covariate-by-time interaction), but "holds by construction"
    and "holds for the arrays actually passed in" are different claims, and
    checking costs one einsum over arrays that are already built.

    Returns the worst relative residual over the generators; the caller
    warns if it is not negligible.
    """
    if not gens:
        return 0.0
    worst = 0.0
    Xs = np.asarray(X_time_surv, dtype=float)
    Zs = np.asarray(Z_time_surv, dtype=float)
    Xq = np.asarray(X_time_quad, dtype=float)
    Zq = np.asarray(Z_time_quad, dtype=float)
    for v, d in gens:
        for lhs, rhs in ((Xs @ d, v * Zs[:, q_idx]),
                          (np.einsum("nkp,p->nk", Xq, d),
                           v[:, None] * Zq[:, :, q_idx])):
            denom = max(float(np.abs(rhs).max()), float(np.abs(lhs).max()), 1.0)
            worst = max(worst, float(np.abs(lhs - rhs).max()) / denom)
    return worst


def _absorbable_basis_joint(X_long, Z_long, n_obs,
                            X_time_surv=None, Z_time_surv=None,
                            X_time_quad=None, Z_time_quad=None,
                            extra_pairs=(), tol=1e-8, s_tol=1e-6,
                            verify_tol=1e-7, rcond=1e-10):
    """Absorbable subspace defined JOINTLY across random-effect columns.

    This is the construction the rotation (control$rotate_absorbable) uses
    from jmjax 0.4.0 on. It replaces the union of the per-column bases of
    _absorbable_basis_exact(), which misses fixed-effect shifts that must be
    offset in several random-effect columns at once. Example: with
    X_i = [1, t, x_i (1 + t)] and Z_i = [1, t], each column on its own gives
    only span{1}, but beta_x + c, b_i0 - c x_i, b_i1 - c x_i leaves every
    trajectory unchanged, so the absorbable subspace is span{1, x}.

    DEFINITION (methods paper, Section 3). (d, Delta) is an invariant shift
    if beta -> beta + d, b_i -> b_i + delta_i leaves every trajectory
    unchanged:
        (L)  X_i d + Z_i delta_i = 0              (observation rows)
        (T)  x_i(t)' d + z_i(t)' delta_i = 0      (every t the hazard sees)
    For a witness d, Delta(d) is the shift whose rows are orthogonal to
    K_i = {a : z_i(t)' a = 0 for all t}. The absorbable subspace is the span
    of the columns of Delta(d) over all witnesses d.

    CONSTRUCTION.
     1. M stacks (I - P_{Z_i}) X_i over subjects, P_{Z_i} = Z_i Z_i^+, on
        the observation rows. Witnesses lie in ker M; one SVD gives a basis.
     2. For each basis vector d, delta_i(d) solves (T) on subject i's grid:
        the observation times, the event time, the quadrature nodes, and
        any functional-form rows (extra_pairs). The pseudoinverse gives the
        solution orthogonal to the grid null space of Z, which is K_i.
        Solving (L) from the observation rows alone is NOT the same when
        Z_i is rank-deficient: for a baseline-only subject with d = e_time
        and Z_i = (1, 0), the minimum-norm solution of (L) is (0, 0), but
        the trajectory forces (0, -1).
        The grid is the set of times at which the
        likelihood uses m_i(t), which is what invariance requires. The
        paper's Definition 1 asks for (T) on all of [0, tau]; the two agree
        whenever the grid null space of Z_i equals K_i on [0, tau] (any
        polynomial basis of degree below the number of grid points, so
        every design jmjax fits with ~ 1 or ~ time). They can differ when a
        Z column vanishes on part of a subject's range (a truncated-power
        or spline basis with a knot beyond the subject's event time): there
        the grid leaves that coordinate free and the pseudoinverse sets it
        to zero, where [0, tau] would pin it. Without survival arrays (a mixed
        model), (L) is the whole requirement and delta_i = -Z_i^+ X_i d.
     3. Condition (S): if (T) cannot be solved for some d in ker M (the
        worst relative residual exceeds s_tol), that d is not a witness.
        The residual is reported, and only the subspace of ker M on which
        (T) holds is admitted (an SVD of the residual operator), so a
        failing direction is never rotated.
     4. Every admitted generator is verified against (L) and (T) before use,
        so a loose rank tolerance cannot admit a non-absorbable direction.
     5. The N x q matrices Delta(d) are collected into an N x (q dim ker M)
        matrix, and one more SVD gives the orthonormal Q (N x k). Then
        k <= q dim ker M <= q p, and Q depends on X only through its column
        space (replacing X by X G maps ker M to G^{-1} ker M and leaves
        every X_i d, hence every Delta(d), unchanged).

    The rotation built on Q is exact whatever Q is (any fixed orthogonal
    matrix is), so a missed or extra direction affects efficiency, not the
    posterior.

    Arguments: X_long [N, max_obs, p], Z_long [N, max_obs, q], n_obs [N];
    X_time_surv [N, p], Z_time_surv [N, q]; X_time_quad [N, K, p],
    Z_time_quad [N, K, q]; extra_pairs a sequence of (X_surv, Z_surv,
    X_quad, Z_quad) for slope/area functional forms (any element may be
    None).

    Returns (Q, info). Q is [N, k] orthonormal, or None. info has
    "dim_ker_M", "n_witnesses" (dimension of the admitted witness space),
    "k", "condition_s_residual" (worst relative residual of (T) over a
    basis of ker M, before restriction), "n_rejected_s" (dimensions of
    ker M dropped because (T) fails), "n_rejected_verify" (generators that
    failed the final check), "joint_model" (whether (T) was used) and
    "generators" (list of (d, Delta) pairs, in the coordinates of X_long).
    """
    info = {"dim_ker_M": 0, "n_witnesses": 0, "k": 0,
            "condition_s_residual": 0.0, "n_rejected_s": 0,
            "n_rejected_verify": 0, "joint_model": False, "generators": []}
    X = np.asarray(X_long, dtype=float)
    Z = np.asarray(Z_long, dtype=float)
    if X.ndim != 3 or Z.ndim != 3 or X.shape[:2] != Z.shape[:2]:
        return None, info
    N, mo, p = X.shape
    q = Z.shape[2]
    n = np.asarray(n_obs).astype(int).reshape(-1)
    if n.shape[0] != N or N == 0 or p == 0 or q == 0:
        return None, info

    mask = (np.arange(mo)[None, :] < n[:, None]).astype(float)   # [N, mo]
    Xo = X * mask[:, :, None]
    Zo = Z * mask[:, :, None]
    flat = mask.reshape(-1) > 0
    if int(flat.sum()) < p:
        return None, info

    # ---- 1. ker M ---------------------------------------------------------
    # Padded rows are zero in both Xo and Zo; pinv([A; 0]) = [A^+, 0], so
    # they change nothing.
    Zo_pinv = np.linalg.pinv(Zo, rcond=rcond)                     # [N, q, mo]
    Mrows = Xo - Zo @ (Zo_pinv @ Xo)                               # [N, mo, p]
    Mf = Mrows.reshape(N * mo, p)[flat]
    X_f = Xo.reshape(N * mo, p)[flat]
    colnorm = np.linalg.norm(X_f, axis=0)       # so the rank test is unit-free
    scale = np.where(colnorm > tol, colnorm, 1.0)
    # full_matrices=False: U stays [rows, p], not [rows, rows].
    _, sv, Vt = np.linalg.svd(Mf / scale, full_matrices=False)
    smax = float(sv.max()) if sv.size else 0.0
    rank = int((sv > tol * max(smax, 1.0)).sum())
    if rank >= p:
        return None, info
    D = Vt[rank:].T / scale[:, None]                               # [p, m]
    info["dim_ker_M"] = int(D.shape[1])

    # ---- 2. the grid on which delta_i(d) is solved -----------------------
    def _rows(Xs, Zs, Xq, Zq):
        out = []
        if Xs is not None and Zs is not None:
            out.append((np.asarray(Xs, float)[:, None, :],
                        np.asarray(Zs, float)[:, None, :]))
        if Xq is not None and Zq is not None:
            out.append((np.asarray(Xq, float), np.asarray(Zq, float)))
        return out

    blocks = [(Xo, Zo)]
    surv_blocks = _rows(X_time_surv, Z_time_surv, X_time_quad, Z_time_quad)
    for pair in (extra_pairs or ()):
        surv_blocks += _rows(*pair)
    joint = len(surv_blocks) > 0
    info["joint_model"] = bool(joint)
    for bx, bz in surv_blocks:
        if bx.shape[0] != N or bx.shape[2] != p or bz.shape[2] != q \
                or bz.shape[:2] != bx.shape[:2]:
            raise ValueError("absorbable: survival/quadrature design arrays "
                             "do not match X_long/Z_long (%s vs %s)"
                             % (bx.shape, bz.shape))
        blocks.append((bx, bz))
    GX = np.concatenate([b[0] for b in blocks], axis=1)           # [N, R, p]
    GZ = np.concatenate([b[1] for b in blocks], axis=1)           # [N, R, q]
    GZ_pinv = np.linalg.pinv(GZ, rcond=rcond)                     # [N, q, R]

    def _solve(Dm):
        """Delta for each column of Dm, and the relative (T) residuals."""
        XD = np.einsum("nrp,pm->nrm", GX, Dm)                     # [N, R, m]
        Delta = -GZ_pinv @ XD                                     # [N, q, m]
        res = XD + GZ @ Delta
        mag = np.abs(XD).reshape(-1, Dm.shape[1]).max(axis=0)
        rel = np.abs(res).reshape(-1, Dm.shape[1]).max(axis=0) / np.maximum(mag, 1e-300)
        return Delta, res, mag, rel

    # Re-express ker M so that the trajectory changes X d on the grid are
    # orthonormal (flattened over subjects and rows); directions with X d = 0
    # everywhere (collinear X columns) are null shifts with Delta = 0 and
    # are dropped. Orthonormality makes the residual operator below
    # comparable across directions.
    XD0 = np.einsum("nrp,pm->nrm", GX, D).reshape(-1, D.shape[1])
    _, xs, xVt = np.linalg.svd(XD0, full_matrices=False)
    live = xs > tol * max(float(xs.max()) if xs.size else 0.0, 1e-300)
    if not live.any():
        return None, info
    D = D @ (xVt[live].T / xs[live])
    n_rows = int(GX.shape[0] * GX.shape[1])

    # ---- 3. Condition (S): restrict to the d for which (T) is solvable ---
    _, res, _, rel = _solve(D)
    info["condition_s_residual"] = float(rel.max()) if joint else 0.0
    if joint and float(rel.max()) > s_tol:
        # (T) is linear in d, so the admissible d form a subspace: the
        # (near-)null space of the residual operator restricted to ker M.
        # Only the right singular vectors are needed: an [m, m] SVD of the
        # Gram matrix would square the condition number, so take the thin
        # SVD of the [rows, m] operator (U stays [rows, m]).
        # The columns of X D are orthonormal, so for a unit coefficient
        # vector c, max|X D c| >= 1/sqrt(rows). A direction whose worst
        # relative residual exceeds s_tol therefore has ||R c||_2 >
        # s_tol/sqrt(rows); cutting there excludes every such direction,
        # however few subjects it fails in, while genuine directions sit at
        # rounding level (~1e-14), far below it.
        R_op = res.reshape(-1, D.shape[1])
        _, rs, rVt = np.linalg.svd(R_op, full_matrices=False)
        rs_full = np.zeros(D.shape[1])
        rs_full[:rs.size] = rs
        keep = rs_full <= s_tol / np.sqrt(max(n_rows, 1))
        D_adm = D @ rVt[keep].T
        info["n_rejected_s"] = int(D.shape[1] - D_adm.shape[1])
        D = D_adm
        if D.shape[1] == 0:
            return None, info

    # ---- 4. verify every generator before use -----------------------------
    Delta, _, _, rel_T = _solve(D)
    gens = []
    for j in range(D.shape[1]):
        d = D[:, j]
        Dj = Delta[:, :, j]                                       # [N, q]
        lhs = np.einsum("nop,p->no", Xo, d)
        resL = lhs + np.einsum("noq,nq->no", Zo, Dj)
        magL = max(float(np.abs(lhs).max()), 1e-300)
        okL = float(np.abs(resL).max()) / magL <= verify_tol
        okT = (not joint) or float(rel_T[j]) <= s_tol
        if joint and not okT:
            info["n_rejected_s"] += 1            # a Condition (S) failure
            continue
        if not okL or float(np.abs(Dj).max()) <= 0.0:
            info["n_rejected_verify"] += 1
            continue
        gens.append((d, Dj))
    info["n_witnesses"] = len(gens)
    info["generators"] = gens
    if not gens:
        return None, info

    # ---- 5. Q from the columns of every Delta(d) -------------------------
    # Each generator is scaled to unit Frobenius norm, but its columns keep
    # their relative sizes, and the rank cut is tied to the verification
    # tolerance: a generator passes verification with a relative residual
    # up to s_tol, so anything smaller than that, relative to its
    # generator, cannot be told apart from a residual and is treated as
    # zero rather than promoted to a full direction.
    col_tol = max(s_tol, verify_tol)
    cols = []
    for _, Dj in gens:
        fro = float(np.linalg.norm(Dj))
        for c in range(q):
            v = Dj[:, c] / fro
            if float(np.linalg.norm(v)) > col_tol:
                cols.append(v)
    if not cols:
        return None, info
    S = np.stack(cols, axis=1)
    U, ssv, _ = np.linalg.svd(S, full_matrices=False)
    r = int((ssv > col_tol * float(ssv.max())).sum())
    if r == 0:
        return None, info
    Q = np.ascontiguousarray(U[:, :r])
    # Q spans exactly the Delta columns: everything in it is absorbable,
    # and nothing that is absorbable (per the generators) is left out.
    miss = float(np.abs(S - Q @ (Q.T @ S)).max())
    if miss > 10 * col_tol:
        warnings.warn("absorbable: the joint basis does not reproduce its "
                      "generators (residual %.1e); rank cut too aggressive."
                      % miss, RuntimeWarning, stacklevel=2)
    info["k"] = int(Q.shape[1])
    return Q, info


def _union_basis(bases, tol=1e-8):
    """Orthonormal basis of span(union of the non-None bases), or None.

    SEPARATE, OPT-IN TRACK (control$orthogonalize_rotate_all). Q_cup in
    vignette("jmjax-reparameterization"), Section 4.10: by Lemma 1 there,
    every flat direction of the swept model - for every value of the
    sampled correlation - has all of its random-effect columns inside
    span(Q_cup), so rotating EVERY column of b_std by one fixed orthogonal
    matrix whose first k columns span Q_cup puts the whole flat subspace
    inside an explicit [k, q] coordinate block. Computed by SVD with an
    explicit rank cut, because the per-column bases usually overlap (the
    intercept's span{1, age, sex} contains the slope's span{1}).
    """
    cols = [np.asarray(Qb, dtype=float) for Qb in (bases or []) if Qb is not None]
    if not cols:
        return None
    S = np.concatenate(cols, axis=1)
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    r = int((sv > tol * max(float(sv.max()), 1.0)).sum())
    if r == 0:
        return None
    return np.ascontiguousarray(U[:, :r])


def _householder_reflectors(Q, tol=1e-10):
    """Householder vectors for an orthogonal H whose first k columns span Q.

    SEPARATE, OPT-IN TRACK (control$orthogonalize_rotate_all). Returns
    V [N, k] with unit columns v_j (zero in rows < j) such that
    H = H_0 H_1 ... H_{k-1}, H_j = I - 2 v_j v_j^T, is orthogonal and
    H[:, :k] = Q @ diag(+-1). H is never formed: applying it costs O(N k)
    (see _apply_reflectors), against O(N^2) memory and work for the dense
    random completion _orthogonal_complement() builds for the older
    intercept-only rotation - about 0.5 GB in float64 at N = 8000. The
    Householder completion's columns k..N-1 are also close to the
    coordinate axes (each is e_j plus an O(1/sqrt(N)) perturbation), so
    the per-subject coordinates keep their own mass-matrix entries.

    Verified before return (defensive, like _orthogonal_complement): H^T Q
    must vanish below row k, and H must preserve norms on random probes.
    Returns None if the check fails.
    """
    if Q is None:
        return None
    A = np.array(Q, dtype=float, copy=True)
    N, k = A.shape
    if k == 0 or k >= N:
        return None
    V = np.zeros((N, k))
    for j in range(k):
        x = A[j:, j].copy()
        nx = float(np.linalg.norm(x))
        if nx <= tol:
            return None
        alpha = -nx if x[0] >= 0 else nx          # stable sign choice
        v = x
        v[0] -= alpha
        v /= float(np.linalg.norm(v))
        V[j:, j] = v
        A[j:, :] -= 2.0 * np.outer(v, v @ A[j:, :])
    # --- verification ---------------------------------------------------
    HtQ = _apply_reflectors(V, np.asarray(Q, dtype=float), transpose=True)
    below = float(np.abs(HtQ[k:]).max()) if N > k else 0.0
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, 3))
    HX = _apply_reflectors(V, X)
    norm_err = float(np.abs(np.linalg.norm(HX, axis=0) - np.linalg.norm(X, axis=0)).max())
    back_err = float(np.abs(_apply_reflectors(V, HX, transpose=True) - X).max())
    if below > 1e-8 or norm_err > 1e-8 * np.sqrt(N) or back_err > 1e-8 * np.sqrt(N):
        warnings.warn(
            "orthogonalize_rotate_all: the Householder construction did not "
            "verify (below=%.2e, norm_err=%.2e, back_err=%.2e); falling back "
            "to the unrotated sampling for this fit." % (below, norm_err, back_err),
            RuntimeWarning, stacklevel=2)
        return None
    return V


def _apply_reflectors(V, X, transpose=False):
    """H @ X (or H.T @ X) for H = H_0 ... H_{k-1} given by V; NumPy or JAX.

    H_j is symmetric, so H.T = H_{k-1} ... H_0: H @ X applies the
    reflectors last-to-first, H.T @ X first-to-last. O(N k m) for X [N, m].
    """
    k = V.shape[1]
    order = range(k) if transpose else range(k - 1, -1, -1)
    for j in order:
        v = V[:, j:j + 1]
        X = X - 2.0 * (v @ (v.T @ X))
    return X
