"""LEGACY: the location sweep (control$orthogonalize_b0 / orthogonalize_b).

NOT PART OF jmjax'S RECOMMENDED PATH. The sweep projects each random-effect
column off its absorbable directions (absorbable.py), which removes the
location degeneracy from the sampled coordinates. It then needs the
analytical correction "beta_corrected" to report calibrated coefficients.
The measured result: the swept coefficient mixes 9-15x faster but is not
reportable, and the corrected coefficient gains only 1.25-1.31x. The
rotation layered on top (orthogonalize_b0_rotate, orthogonalize_rotate_all)
recovered the efficiency. The rotation WITHOUT the sweep
(control$rotate_absorbable plus control$dense_mass_generator_beta, in
mcmc_model.py) matched that in dev/pilot_rotate_grid.R and calibrated like a
default fit in dev/study_calibration.R. It also needs no correction. That
no-sweep rotation is what jmjax uses. The full development record is in
dev/notes/sweep-reparameterization.md.

Everything here is kept so that the dev/ studies which compare against the
sweep (arms D, E, D_rot, E_rotall, ...) stay reproducible. The code is
moved unchanged from mcmc_model.py; only the function boundaries are new.
Every entry point is a no-op unless one of these control flags is set:
orthogonalize_b0, orthogonalize_b, orthogonalize_b0_rotate,
orthogonalize_rotate_all, dense_mass_b0_generator.
"""
import warnings

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .absorbable import (_absorbable_basis, _absorbable_basis_exact,
                         _structural_extension_residual, _union_basis,
                         _householder_reflectors)


def _absorbable_generator_matrix(Q, gens, tol=1e-8):
    """Beta-shift matrix matching Q's columns - the analytical-correction
    track's only new piece of linear algebra.

    SEPARATE, OPT-IN TRACK. Not used by orthogonalize_b/_b0's own sampling
    at all (that only ever needs Q, not this). It exists for the
    "beta_corrected" post-processing block in fit_nuts(), added after
    dev/study_calibration.R's pilot run showed that orthogonalize_b/_b0
    reproduce random_effects_method = "wishart_gibbs_centered"'s retracted
    failure mode: unbiased point estimates but too-narrow credible
    intervals for exactly the swept parameter(s), because the swept
    component of b_raw is unidentified under the likelihood and simply
    samples its prior, with nothing downstream reflecting that uncertainty.
    See vignette("jmjax-reparameterization"), Section 9.

    WHY THIS RECOVERS WHAT IS NEEDED. _absorbable_basis_exact() already
    verifies, for each raw generator pair (v, d) in `gens`, the identity

        X_i @ d  ==  v_i * Z_i,q(t)      for every subject i (all t)

    - that is what "absorbable" means, and it is exactly the identity
    orthogonalize_b/_b0 relies on to subtract Q's columns from b[:, q]
    without changing any fitted value. Q is then an ORTHONORMAL basis
    (via QR) of span(v_1, v_2, ...), so Q's own columns are themselves
    linear combinations of those v's - and because the map d -> v = ratio @ d
    is LINEAR, the SAME combination of the matching d's satisfies the
    identity for Q's columns instead of the raw generators':

        X_i @ Dmat[:, m]  ==  Q[:, m]_i * Z_i,q(t)

    which is precisely the beta-shift a caller needs: subtracting
    Q[:, m] * c_m from b[:, q] changes the fitted values by
    -c_m * Q[:, m]_i * Z_i,q(t) = -c_m * X_i @ Dmat[:, m], so adding
    c_m * Dmat[:, m] back to beta restores them. Summed over m via
    c = Q.T @ b_raw[:, q] (the actual swept component of a given draw),
    this is exactly what recovers an unconstrained-equivalent beta from an
    orthogonalized fit's own posterior draws, with no refit.

    HOW THE COMBINATION IS FOUND. Q lies in span(S) exactly by
    construction (S = the raw v's stacked as columns), so solving
    Q = S @ coeffs by least squares recovers `coeffs` to numerical
    precision; Dmat = Dg @ coeffs (Dg = the raw d's stacked as columns)
    applies that same combination to the d side. Verified before being
    returned, in the same spirit as _absorbable_basis_exact's own
    per-generator check - a silent near-miss here would surface as a
    biased, not just mis-scaled, beta_corrected.

    Returns a [p, k] array (k = Q.shape[1]), or None (with a warning) if
    the recovery does not check out - defensive; should not happen for any
    Q actually returned by _absorbable_basis_exact.
    """
    if Q is None or not gens:
        return None
    S = np.stack([g[0] for g in gens], axis=1)            # [N_sub, k0]
    Dg = np.stack([g[1] for g in gens], axis=1)            # [p, k0]
    coeffs, _, _, _ = np.linalg.lstsq(S, Q, rcond=None)    # [k0, k]
    resid = float(np.abs(S @ coeffs - Q).max())
    scale = max(float(np.abs(Q).max()), 1.0)
    if resid > 1e-6 * scale:
        warnings.warn(
            "orthogonalize_b: could not recover a matching beta-shift basis "
            "for the analytical-correction track (residual %.2e); "
            "beta_corrected will not be reported for this column." % resid,
            RuntimeWarning, stacklevel=2)
        return None
    return Dg @ coeffs                                     # [p, k]


def _orthogonal_complement(Q, seed=0):
    """Orthonormal basis of the orthogonal complement of span(Q).

    SEPARATE, OPT-IN TRACK (control$orthogonalize_b0_rotate). Not used by
    orthogonalize_b/_b0's own sweep, nor by the beta_corrected correction -
    both only ever need Q. This exists to let the intercept column's own
    sampling be reparameterized so the swept generator direction becomes an
    explicit, low-dimensional NUTS site (see the long comment at this
    option's control-flag check in fit_nuts()), instead of being buried
    inside the full N_sub-dimensional b_std[:, 0], where it cannot receive
    its own mass-matrix treatment.

    Q: [N, k] orthonormal (as returned by _absorbable_basis_exact). Returns
    Qperp: [N, N-k] orthonormal, with Q.T @ Qperp == 0 and [Q, Qperp]
    jointly spanning R^N (verified below to a strict numerical tolerance,
    not assumed) - i.e. [Q, Qperp] is an N x N orthogonal matrix. Computed
    via QR of [Q | a random completion], a standard, numerically stable
    construction (Householder QR). The completion's seed is fixed (derived
    from control$seed by the caller) so a fit stays exactly reproducible.

    WHY THIS IS THE RIGHT OBJECT. If u ~ N(0, I_k) and v ~ N(0, I_{N-k})
    independently, then b := Q @ u + Qperp @ v is EXACTLY N(0, I_N)
    distributed - a basic rotation-invariance fact (Q, Qperp orthonormal
    and jointly spanning R^N), not an approximation. So replacing the
    N-dimensional standard-normal sample b_std[:, 0] with this
    reconstruction changes nothing about the model; it only changes which
    coordinates NUTS actually adapts a mass matrix to. And because
    Q.T @ Qperp = 0, Q.T @ b = u exactly - so u IS the same generator
    coordinate orthogonalize_b0's sweep step (Q @ (Q.T @ b_raw[:, 0])) and
    beta_corrected's correction (_absorbable_generator_matrix's own
    Q.T @ b_raw[:, q] term) already compute from b_raw. Giving it its own
    name is what lets it receive its own dense-mass block
    (control$dense_mass_b0_generator) via NumPyro's ordinary block-diagonal
    mass-matrix mechanism, with no new sampler-level machinery required.

    NOT USED for the slope column (or any q >= 1 column when
    orthogonalize_b sweeps it): b_raw[:, 0] = sigma_b[0] * b_std[:, 0]
    always (a valid Cholesky factor of a correlation matrix has first row
    [1, 0, ..., 0]), so a FIXED rotation of b_std[:, 0] alone exactly
    isolates the intercept's generator direction. The analogous slope
    direction is sigma_b[1] * (rho * b_std[:, 0] + sqrt(1-rho^2) *
    b_std[:, 1]) - a combination that depends on the SAMPLED correlation
    rho, so a fixed, precomputed rotation cannot isolate it the same way.
    (Superseded for that purpose by control$orthogonalize_rotate_all - see
    _householder_reflectors(): rotating EVERY column by the same fixed
    basis of the union of the swept spaces contains the rho-dependent
    direction inside a small explicit block for every rho, by Lemma 1 of
    vignette("jmjax-reparameterization") Section 4.10. This function and
    the intercept-only option are kept unchanged for comparison.)

    Returns None if Q is None, if Q.shape[1] >= Q.shape[0] (nothing left
    to complement), or if the construction fails its own verification
    (defensive; should not happen for any well-conditioned Q).
    """
    if Q is None:
        return None
    N, k = Q.shape
    if k >= N:
        return None
    rng = np.random.default_rng(seed)
    M = np.concatenate([Q, rng.standard_normal((N, N - k))], axis=1)
    Qfull, _ = np.linalg.qr(M)
    Qperp = Qfull[:, k:]
    # Defensive check, not a formality: an (astronomically unlikely)
    # near-rank-deficient random completion, or an ill-conditioned Q,
    # would silently produce a Qperp that does not actually complement Q -
    # which would corrupt the model rather than merely under-perform it.
    ortho_err = float(np.abs(Q.T @ Qperp).max())
    orthonorm_err = float(np.abs(Qperp.T @ Qperp - np.eye(N - k)).max())
    if ortho_err > 1e-6 or orthonorm_err > 1e-6:
        warnings.warn(
            "orthogonalize_b0_rotate: the orthogonal-complement construction "
            "did not verify to tolerance (ortho_err=%.2e, orthonorm_err="
            "%.2e); falling back to the unrotated sampling for this fit." %
            (ortho_err, orthonorm_err), RuntimeWarning, stacklevel=2)
        return None
    return Qperp


def setup(control, q, random_effects_corr, random_effects_method,
          X_long, Z_long, n_obs, X_time_surv, Z_time_surv,
          X_time_quad, Z_time_quad, N_sub):
    """Everything fit_nuts() needs for the legacy sweep options.

    Handles control$orthogonalize_b0, orthogonalize_b,
    orthogonalize_b0_rotate and orthogonalize_rotate_all: it validates them,
    builds the swept bases and the matching beta-shift bases for
    beta_corrected, and builds the rotation data. Moved here unchanged from
    fit_nuts(). Returns a dict. With none of the options requested, every
    entry is None/False and fit_nuts() behaves exactly as a default fit.
    """
    # ------------------------------------------------------------------
    # control$orthogonalize_b0 (EXPERIMENTAL, opt-in, default False).
    # See the long note in build_model()'s matching branch for the measured
    # motivation. Restricted to the path it was measured on rather than
    # silently no-op'ing elsewhere: the other random-effects branches name
    # "b" as a SAMPLE site, so constraining it there would leave
    # posterior_samples$b holding the unconstrained draws while the
    # likelihood used the projected ones - a discrepancy that would quietly
    # corrupt exactly the diagnostics this option exists to improve.
    # ------------------------------------------------------------------
    # orthogonalize_b  : every random-effect column (intercept AND slope)
    # orthogonalize_b0 : the intercept column only - retained so the two can
    #                    be compared, since the slope extension is the newer
    #                    and less tested half.
    _orth_all  = bool(control.get("orthogonalize_b", False))
    _orth_int  = bool(control.get("orthogonalize_b0", False))
    # SEPARATE, OPT-IN TRACK, layered on top of orthogonalize_b/_b0 rather
    # than replacing it: changes HOW the intercept column's degeneracy is
    # SAMPLED (an explicit, low-dimensional rotation - see
    # _orthogonal_complement()'s docstring), not WHETHER it is swept. Valid
    # whenever column 0 is touched at all, i.e. whenever orthogonalize_b0
    # or orthogonalize_b is requested (both compute b_orth_bases[0]).
    # Intercept-only: the analogous slope direction depends on the sampled
    # correlation rho, so this does not extend to orthogonalize_b's slope
    # column - see the same docstring, and control$orthogonalize_rotate_all
    # just below for the construction that does.
    _orth_int_rotate = bool(control.get("orthogonalize_b0_rotate", False))
    # SEPARATE, OPT-IN TRACK (control$orthogonalize_rotate_all): the
    # generalization vignette("jmjax-reparameterization") Section 4.10
    # (Corollary 4) derives - every random-effect column rotated by one
    # fixed Householder matrix built from the UNION of the swept bases.
    # Mutually exclusive with the intercept-only rotation above, which it
    # supersedes; kept separate so the two can be compared on equal seeds.
    _rot_all = bool(control.get("orthogonalize_rotate_all", False))
    if _rot_all and not (_orth_all or _orth_int):
        raise ValueError(
            "control$orthogonalize_rotate_all = TRUE requires "
            "orthogonalize_b0 or orthogonalize_b to also be TRUE - it only "
            "changes how the swept degeneracy is sampled, not whether it is "
            "swept at all.")
    if _rot_all and _orth_int_rotate:
        raise ValueError(
            "control$orthogonalize_rotate_all and "
            "control$orthogonalize_b0_rotate are alternative constructions; "
            "request at most one.")
    if _orth_int_rotate and not (_orth_all or _orth_int):
        raise ValueError(
            "control$orthogonalize_b0_rotate = TRUE requires "
            "orthogonalize_b0 or orthogonalize_b to also be TRUE - it only "
            "changes how the intercept column's degeneracy is sampled, not "
            "whether it is swept at all.")
    b_orth_bases = None
    # SEPARATE, OPT-IN TRACK (see the "beta_corrected" block after
    # mcmc.get_samples() below, and _absorbable_generator_matrix() near
    # _absorbable_basis_exact()). Parallel to b_orth_bases: _correction_bases
    # holds, per random-effect column, the [p, k] matrix Dmat matching the
    # SAME Q that b_orth_bases[q] holds, i.e. the beta-shift that keeps
    # fitted values unchanged when Q's columns are subtracted from b[:, q].
    # This does not change what is sampled - it is read only to turn the
    # already-sampled b_std/L_corr/sigma_b/beta draws into a corrected
    # estimand, entirely after the fact.
    _correction_bases = None
    # Structured record of what orthogonalize_b/_b0 actually did, surfaced
    # to R as fit$convergence$orthogonalize (NULL unless the option was
    # requested). This exists for the same reason warm_start_check does:
    # the warnings.warn() calls below arrive on stderr, not as an R
    # condition (reticulate does not route them through R's condition
    # system), so expect_warning() cannot see them and a non-interactive
    # run would silently lose the report. A caller - or a test - that
    # needs to know which columns were actually constrained has to be able
    # to read it off the fitted object instead.
    orthogonalize_report = None
    # Initialized here, OUTSIDE the `if _orth_all or _orth_int:` block below,
    # because build_model() is called unconditionally further down with
    # b0_gen_perp=_b0_gen_perp for EVERY fit, including an ordinary default
    # fit that never sets orthogonalize_b0/_b at all. A first attempt at this
    # left the init inside the `if`, which raised UnboundLocalError on every
    # non-orthogonalized fit - caught by the install script's self-check
    # (dev/install_jmjax.sh step 4/4) on its plain penalized-spline fit.
    _b0_gen_perp = None
    # Same reasoning: build_model() receives gen_reflectors on every fit.
    _gen_reflectors = None
    if _orth_all or _orth_int:
        if not (q >= 2 and random_effects_corr
                and random_effects_method == "nuts"):
            raise ValueError(
                "orthogonalize_b/_b0 = True currently requires q >= 2, "
                "random_effects_corr = TRUE and the default "
                "random_effects_method = 'nuts' (got q = %d, corr = %s, "
                "method = '%s'). It is an experimental option; the other "
                "random-effects branches are not yet covered."
                % (q, bool(random_effects_corr), random_effects_method))
        _qmax = q if _orth_all else 1
        b_orth_bases = []
        _correction_bases = []
        _orth_report = []
        _orth_columns = []
        _s_resid = 0.0
        for _q in range(q):
            if _q < _qmax:
                # TWO constructions, deliberately. The per-column search is
                # kept for ATTRIBUTION - it names which X columns expose each
                # direction, which is what makes the report diagnostic rather
                # than just a dimension count. The exact construction is
                # AUTHORITATIVE: it depends only on X's column space, so it
                # cannot miss directions that a spline or orthogonal-
                # polynomial basis hides from the per-column test. See
                # _absorbable_basis_exact's docstring and Section 4.5 of
                # vignette("jmjax-reparameterization").
                _Qcol, _kept = _absorbable_basis(X_long, Z_long, n_obs, _q)
                _Q, _gens = _absorbable_basis_exact(X_long, Z_long, n_obs, _q)
                _ncol = 0 if _Qcol is None else _Qcol.shape[1]
                _nex = 0 if _Q is None else _Q.shape[1]
                if _nex > _ncol:
                    warnings.warn(
                        "orthogonalize_b: b[:,%d] - the per-column search "
                        "found %d absorbable direction(s)%s, but %d exist. "
                        "The extra direction(s) are absorbable only by a "
                        "COMBINATION of fixed-effects columns that no single "
                        "column exposes, which is what a spline or "
                        "orthogonal-polynomial longitudinal mean produces. "
                        "The exact (basis-independent) basis is being used, "
                        "so the degeneracy IS removed; this notice exists "
                        "because the per-column answer alone would have "
                        "silently left it in place. See "
                        "vignette('jmjax-reparameterization'), Section 4.5."
                        % (_q, _ncol,
                           "" if not _kept else " (from X columns %s)" % (_kept,),
                           _nex),
                        RuntimeWarning, stacklevel=2)
                elif _nex < _ncol:
                    # Proposition 2 says this cannot happen: every
                    # per-column direction is absorbable, so the exact basis
                    # must contain it. If it ever fires, one of the two
                    # constructions is wrong and the fit should not be
                    # quietly trusted.
                    warnings.warn(
                        "orthogonalize_b: b[:,%d] - the exact construction "
                        "found FEWER directions (%d) than the per-column "
                        "search (%d). This contradicts the soundness result "
                        "and should be reported as a bug; the exact basis is "
                        "being used." % (_q, _nex, _ncol),
                        RuntimeWarning, stacklevel=2)
                _s_resid = max(_s_resid, _structural_extension_residual(
                    _gens, _q, X_time_surv, Z_time_surv,
                    X_time_quad, Z_time_quad))
                # SEPARATE, OPT-IN TRACK: the beta-shift basis matching Q,
                # for the "beta_corrected" post-processing block below. Reuses
                # _Q/_gens already computed above at no extra cost (no second
                # call into _absorbable_basis_exact).
                _Dmat = _absorbable_generator_matrix(_Q, _gens) if _Q is not None else None
                if _q == 0 and _orth_int_rotate and _Q is not None:
                    # Reuses the fit's own seed so the (otherwise arbitrary)
                    # random completion in _orthogonal_complement is exactly
                    # reproducible given the same control$seed - it affects
                    # only WHICH orthonormal completion is used, never
                    # whether the construction is exact (verified inside
                    # _orthogonal_complement regardless of the seed drawn).
                    _b0_gen_perp = _orthogonal_complement(
                        _Q, seed=int(control.get("seed", 2026)) + 90210)
            else:
                _Q, _kept, _ncol, _nex = None, [], 0, 0
                _Dmat = None
            b_orth_bases.append(_Q)
            _correction_bases.append(_Dmat)
            if _Q is None:
                _desc = "no basis (unconstrained)"
            elif _nex == _ncol:
                # wording unchanged in the ordinary case, so existing logs
                # and the vignette's quoted excerpts stay comparable
                _desc = "%d direction(s) from X columns %s" % (_nex, _kept)
            else:
                _desc = ("%d direction(s) (%d from X columns %s, %d from "
                         "column combinations)" % (_nex, _ncol, _kept,
                                                    _nex - _ncol))
            _orth_report.append("b[:,%d] <- %s" % (_q, _desc))
            _orth_columns.append({
                "column": int(_q),
                "applied": _Q is not None,
                "n_directions": int(_nex),
                "source_X_columns": [int(j) for j in _kept],
                # exposes the per-column search's answer separately, so a
                # caller can detect the spline/poly case programmatically
                # rather than by reading the warning text
                "n_directions_column_search": int(_ncol),
            })
        # Say what was ACTUALLY constrained, rather than leaving it to be
        # inferred from downstream behaviour. A verification run found
        # mean(b_i1) NOT driven to zero while beta_1's ESS rose 13x, which is
        # a contradiction that could not be resolved from outside: either the
        # basis for the slope column was not what its design intended, or the
        # effect came from somewhere else. A sampler that reports its own
        # constraints makes that question answerable in one line instead of
        # by inference from a second experiment.
        warnings.warn("orthogonalize_b: " + "; ".join(_orth_report),
                      RuntimeWarning, stacklevel=2)
        if _s_resid > 1e-6:
            # Condition (S) of vignette("jmjax-reparameterization"),
            # Section 4.4: absorbability is established on the longitudinal
            # grid, but `b` also enters through m_i(t) at the event and
            # quadrature times. If the proportionality does not extend
            # there, the swept direction is NOT likelihood-invariant, and
            # the reparameterization would change the model rather than
            # only its coordinates.
            warnings.warn(
                "orthogonalize_b: the absorbable directions do not extend "
                "to the survival/quadrature time grid (worst relative "
                "residual %.2e). Condition (S) of "
                "vignette('jmjax-reparameterization') Section 4.4 fails, so "
                "the reparameterization is NOT guaranteed to leave the joint "
                "likelihood unchanged for this design. Treat results from "
                "this fit as unvalidated." % _s_resid,
                RuntimeWarning, stacklevel=2)
        _orth_any_applied = any(_Q is not None for _Q in b_orth_bases)
        orthogonalize_report = {
            "requested": "b" if _orth_all else "b0",
            "columns": _orth_columns,
            "any_applied": _orth_any_applied,
        }
        if not _orth_any_applied:
            warnings.warn(
                "orthogonalize_b/_b0 = True but no fixed-effect column can "
                "absorb a shift in any random-effect column, so there is no "
                "location degeneracy to remove and the option has no "
                "effect. Expected when the design has no intercept and no "
                "column matching a random-effect term.",
                RuntimeWarning)
            b_orth_bases = None
            _correction_bases = None
        if _rot_all:
            _Qu = _union_basis(b_orth_bases) if b_orth_bases is not None else None
            if _Qu is not None:
                _gen_reflectors = _householder_reflectors(_Qu)
            if _gen_reflectors is None:
                warnings.warn(
                    "control$orthogonalize_rotate_all = TRUE had no effect: "
                    "no swept direction to rotate (or the construction did "
                    "not verify). The fit uses the unrotated sweep.",
                    RuntimeWarning, stacklevel=2)
            orthogonalize_report["rotation"] = {
                "mode": "all_columns",
                "applied": _gen_reflectors is not None,
                "k": 0 if _gen_reflectors is None else int(_gen_reflectors.shape[1]),
            }
        elif _orth_int_rotate:
            orthogonalize_report["rotation"] = {
                "mode": "intercept",
                "applied": _b0_gen_perp is not None,
                "k": 0 if _b0_gen_perp is None else int(N_sub - _b0_gen_perp.shape[1]),
            }

    return {"b_orth_bases": b_orth_bases,
            "correction_bases": _correction_bases,
            "report": orthogonalize_report,
            "b0_gen_perp": _b0_gen_perp,
            "gen_reflectors": _gen_reflectors,
            "orth_all": _orth_all, "orth_int": _orth_int,
            "orth_int_rotate": _orth_int_rotate, "rot_all": _rot_all}


def sample_b_std_b0_rotated(b_orth_bases, b0_gen_perp, N_sub, q):
    """Model fragment: b_std with column 0 rotated (orthogonalize_b0_rotate).

    Called from inside build_model()'s model function, so the
    numpyro.sample sites it creates belong to that model.
    """
    # ------------------------------------------------
    # EXPERIMENTAL, opt-in (control$orthogonalize_b0_rotate).
    # See _orthogonal_complement()'s docstring for the
    # derivation and why this is exact and intercept-
    # column-only. b_raw[:, 0] = sigma_b[0] * b_std[:, 0]
    # always, so a FIXED rotation of b_std[:, 0] alone -
    # into an explicit k-dim "generator" coordinate
    # (b0_gen_u, exactly Q.T @ b_std[:, 0]) plus an
    # (N_sub - k)-dim residual (b0_gen_v) - changes
    # nothing about the model (it is a rotation of a
    # spherical Gaussian) but gives the swept direction
    # its own NUTS site, which control$dense_mass_b0_
    # generator can then give its own mass-matrix block.
    # Columns 1+ (e.g. the slope) are sampled exactly as
    # before; only column 0's construction changes.
    _Q0 = jnp.asarray(b_orth_bases[0])
    _Qp0 = jnp.asarray(b0_gen_perp)
    _k_gen = _Q0.shape[1]
    _n_res = _Qp0.shape[1]
    b0_gen_u = numpyro.sample(
        "b0_gen_u",
        dist.Normal(0.0, 1.0).expand([_k_gen]).to_event(1))
    b0_gen_v = numpyro.sample(
        "b0_gen_v",
        dist.Normal(0.0, 1.0).expand([_n_res]).to_event(1))
    b_std_col0 = _Q0 @ b0_gen_u + _Qp0 @ b0_gen_v
    with numpyro.plate("subjects", N_sub):
        b_std_rest = numpyro.sample(
            "b_std_rest",
            dist.Normal(0.0, 1.0).expand([q - 1]).to_event(1))
    b_std = numpyro.deterministic(
        "b_std",
        jnp.concatenate([b_std_col0[:, None], b_std_rest], axis=1))
    return b_std


def apply_sweep(b_raw, b_orth_bases, q):
    """Model fragment: project the swept columns of b_raw (the sweep itself)."""
    # ----------------------------------------------------
    # EXPERIMENTAL, opt-in (control$orthogonalize_b0).
    #
    # MEASURED MOTIVATION (PBC2, n=312, 4x1000, sampled
    # space - the ONLY space these numbers mean anything
    # in; see dev/diagnose_beta0_v3.R for why):
    #
    #   corr(beta_0, mean level of b_i0)      -0.9709
    #   corr(beta_2, age-slope of b_i0)       -0.9715
    #   corr(beta_1, age-slope of b_i0)       +0.0543
    #
    #             ESS    ESS(+c)   recoverable
    #   beta_0   440.5   2893.1        6.57x
    #   beta_2   526.0   4000.0       >7.60x   (censored)
    #   beta_1  1233.6       -            -
    #
    # Every coefficient on a SUBJECT-CONSTANT covariate
    # is in a near-perfect location degeneracy with the
    # matching projection of the random intercepts: the
    # likelihood constrains only the SUM, and the split
    # is pinned by the prior alone, at scale
    # sigma_b0/sqrt(N). beta_1 is a WITHIN-subject
    # contrast and is clean, which is the control that
    # makes this a mechanism rather than a correlation.
    #
    # WHY A MEAN CONSTRAINT IS NOT ENOUGH. The existing
    # random_effects_method="wishart_gibbs_centered"
    # pins mean(b_i0) = 0 exactly, removing ONE direction
    # of a k-dimensional degeneracy. Measured: beta_0
    # 440.5 -> 1490.4, but beta_2 526.0 -> 291.2, so the
    # WORST parameter got worse (442 -> 291) and the
    # conservative min-ESS metric moved backwards. That
    # is the whole case for generalizing from the mean to
    # the full column space.
    #
    # WHAT THIS DOES. b0_orth_basis is an orthonormal
    # basis Q [N_sub, k] for the subject-constant columns
    # of the longitudinal design, built once from the
    # concrete X_long in fit_nuts(). The intercept column
    # of b is replaced by its residual off that space:
    #
    #     b_0  <-  b_0 - Q (Q' b_0)
    #
    # which is the k-dimensional generalization of
    # subtracting the mean (k=1, Q = 1/sqrt(N)).
    #
    # WHY THE MODEL IS UNCHANGED. The removed directions
    # lie in span(S), and S is by construction a subset
    # of X's columns - so every direction swept out of
    # b_0 is one beta already spans. For any b_raw there
    # is a shift of beta giving IDENTICAL fitted values
    # X beta + Z b, so the achievable mean structures are
    # exactly the same set. This is a reparameterization
    # that removes an unidentified direction, not a
    # different model. The transform is deterministic and
    # differentiable, so HMC gradients flow through it.
    #
    # WHAT DOES CHANGE, AND MUST BE CHECKED. The implied
    # prior on the constrained b_0 is singular (confined
    # to the orthogonal complement), and sigma_b still
    # governs the UNCONSTRAINED b_raw. The k swept
    # directions of b_raw are then unidentified by the
    # likelihood and simply sample their prior. Whether
    # that shifts sigma_b0's posterior is an empirical
    # question, not something to assert - which is why
    # this is opt-in and why the accompanying test
    # compares estimates arm to arm, not just ESS.
    # ----------------------------------------------------
    _cols = []
    for _q in range(q):
        _bq = b_raw[:, _q]
        _Qq = (b_orth_bases[_q]
               if _q < len(b_orth_bases) else None)
        if _Qq is not None:
            _Qq = jnp.asarray(_Qq)
            _bq = _bq - _Qq @ (_Qq.T @ _bq)
        _cols.append(_bq[:, None])
    b = numpyro.deterministic(
        "b", jnp.concatenate(_cols, axis=1))
    return b


def add_beta_corrected(samples, _correction_bases, b_orth_bases):
    """Post-processing: add samples["beta_corrected"] (in place).

    A no-op unless the sweep found at least one direction.
    """
    # ------------------------------------------------------------------
    # SEPARATE, OPT-IN TRACK: "beta_corrected", an analytically
    # un-orthogonalized beta.
    #
    # WHY THIS EXISTS. dev/study_calibration.R's pilot run found that
    # orthogonalize_b/_b0 reproduce random_effects_method =
    # "wishart_gibbs_centered"'s retracted failure mode: unbiased point
    # estimates for the swept parameter(s) (Corollary 1's model-invariance
    # holds), but posterior SD too small by close to the theoretically
    # predicted sigma_b/sqrt(N) - because the swept component of b_raw,
    # c = Q^T b_raw[:, q], is unidentified under the likelihood and simply
    # draws from its prior/conditional distribution, with that uncertainty
    # reflected nowhere in beta's own posterior. This refutes the escape
    # hypothesis raised in vignette("jmjax-reparameterization") Section 4.7;
    # see Section 9 for the numbers.
    #
    # WHAT IT COMPUTES. For every draw, beta_orth and b_raw are related to
    # beta_default (what an UNCONSTRAINED fit's beta would be, given the
    # SAME b_raw draw) by
    #
    #     beta_default = beta_orth - sum_q  Dmat_q @ (Q_q^T @ b_raw[:, q])
    #
    # (see _absorbable_generator_matrix()'s docstring for the derivation).
    # This is PURE POST-PROCESSING of draws this fit already produced - no
    # new numpyro site, no refit - computed whenever orthogonalize_b/_b0
    # found at least one absorbable direction, with no separate control
    # flag: it only adds a new "beta_corrected" key, so it cannot change
    # any existing output ("beta" itself is untouched).
    #
    # NOT YET HANDLED: control$standardize_covariates / control$scale_time
    # (R/jm_fit.R) back-transform posterior_samples$beta but do not know
    # about this key - a fit using either option will have beta_corrected
    # on the SAMPLED, not the back-transformed, scale. Not exercised by
    # dev/study_calibration.R, which uses neither; flagged here so a future
    # caller does not assume otherwise.
    # ------------------------------------------------------------------
    if (_correction_bases is not None
            and any(_D is not None for _D in _correction_bases)
            and "beta" in samples and "b_std" in samples
            and "L_corr" in samples and "sigma_b" in samples):
        _beta_d = np.asarray(samples["beta"])              # [n_draws, p]
        _bstd_d = np.asarray(samples["b_std"])              # [n_draws, N_sub, q]
        _Lcorr_d = np.asarray(samples["L_corr"])            # [n_draws, q, q]
        _sigb_d = np.asarray(samples["sigma_b"])            # [n_draws, q]
        _L_d = _sigb_d[:, :, None] * _Lcorr_d               # [n_draws, q, q]
        # b_raw = b_std @ L.T, batched over draws.
        _braw_d = np.einsum("dnq,dpq->dnp", _bstd_d, _L_d)  # [n_draws, N_sub, q]
        _delta = np.zeros_like(_beta_d)
        for _q_idx, _Dmat in enumerate(_correction_bases):
            if _Dmat is None:
                continue
            _Qb = b_orth_bases[_q_idx]                      # [N_sub, k]
            _c = np.einsum("nk,dn->dk", _Qb, _braw_d[:, :, _q_idx])  # [n_draws, k]
            _delta += _c @ _Dmat.T                           # [n_draws, p]
        samples["beta_corrected"] = _beta_d - _delta


def warm_start_b0_rotated(_bs, b_orth_bases, _b0_gen_perp):
    """Warm-start values for the intercept-only rotation's sample sites."""
    _Q0w = np.asarray(b_orth_bases[0], dtype=float)
    return {"b0_gen_u": jnp.asarray(_Q0w.T @ _bs[:, 0]),
            "b0_gen_v": jnp.asarray(_b0_gen_perp.T @ _bs[:, 0]),
            "b_std_rest": jnp.asarray(_bs[:, 1:])}


def dense_blocks(control, _b0_gen_perp):
    """Dense mass-matrix blocks requested by the legacy options."""
    _blocks = []
    # ------------------------------------------------------------------
    # EXPERIMENTAL, opt-in: a dense mass block over the intercept column's
    # explicit generator coordinate, b0_gen_u (see _orthogonal_complement()
    # and control$orthogonalize_b0_rotate). Only meaningful when that
    # option actually created the site - guarded the same way dense_mass_
    # spline is guarded against sites that do not exist for the current
    # model configuration (requesting a dense block over a nonexistent
    # site raises KeyError, not a silent no-op).
    #
    # MOTIVATION. dense_mass_beta (above) targets a comparable, even more
    # strongly correlated, ridge (beta's own cross-coefficient correlation)
    # and was measured to produce a NULL result - a large correlation is
    # necessary but evidently not sufficient for a dense block to help,
    # which is a real prior finding this option's evaluation should be
    # read against, not a reason to skip trying it: b0_gen_u is a
    # DIFFERENT kind of block than dense_mass_beta's. dense_mass_beta gives
    # a dense matrix to a slice of an ALREADY well-conditioned coordinate
    # (beta, once orthogonalized); b0_gen_u is the coordinate the
    # degeneracy was relocated INTO (vignette("jmjax-reparameterization"),
    # Section 9.1.1) - a small (k = number of absorbable directions,
    # typically 1-4), previously-nonexistent site with no prior track
    # record at all. Whether isolating it this way closes the ESS gap
    # Section 9.1.1 measures for beta_corrected is exactly the open
    # question this option exists to test; a null result here would be
    # informative in the same way dense_mass_beta's was, not a failure of
    # the prototype.
    if control.get("dense_mass_b0_generator", False):
        if _b0_gen_perp is None:
            raise ValueError(
                "control$dense_mass_b0_generator = TRUE requires "
                "control$orthogonalize_b0_rotate = TRUE to have actually "
                "created the b0_gen_u site (it did not for this fit - "
                "either the option was not requested, or no absorbable "
                "intercept direction was found).")
        _blocks.append(("b0_gen_u",))
    return _blocks
