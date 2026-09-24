"""Figures 1-3 of the methods paper (dev/notes/paper-methods-outline.md, Sec 5).

Run from the package root:   python3 dev/figures/make_figures.py
Needs only NumPy and Matplotlib; no new model fits.

  Fig 1  fig1_ridge.{pdf,png}   exact Gaussian toy (dev/theory_rotation_toy.py)
  Fig 2  fig2_gain.{pdf,png}    dev/pilot_rotate_grid.csv (core q = 2 grid),
                                dev/pilot_rotate_stress.csv, dev/pilot_rotate_q1.csv
  Fig 3  fig3_alpha.{pdf,png}   numbers from vignette Section 8.1
                                (dev/alpha_missing_information.R and
                                dev/study_realdata_rotate.R output)

Every number plotted is also printed, so the figure can be checked against
the vignette tables without opening it.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
sys.path.insert(0, DEV)
from theory_rotation_toy import design, posterior, intercept_basis  # noqa: E402

# Reference categorical palette, first three slots (validated all-pairs,
# light mode); every series also has its own marker so grayscale works.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def _sym_inv_root(M):
    e, V = np.linalg.eigh(M)
    return V @ np.diag(e ** -0.5) @ V.T


def _slice(W, E):
    """2-D conditional covariance of the whitened posterior N(0, W) along
    the orthonormal directions E (columns): the contour a sampler sees
    through the mode when the other coordinates are held at theirs."""
    return np.linalg.inv(E.T @ np.linalg.inv(W) @ E)


def fig1(N=300, k=3, nmin=2, nmax=11, rho=0.3, se=0.3):
    d = design(N=N, nmin=nmin, nmax=nmax)
    g = intercept_basis(d)[0][:, 0]           # mean direction (Q0 column 1)
    sgn = np.sign(g.sum())                    # QR fixes Q0 only up to sign:
    g = sgn * g                               # orient u as +mean(b), so the
    panels = []                               # ridge shows its true (negative) sign

    # (a) unrotated, diagonal metric: u = g'b is spread over N coordinates
    S, _, _ = posterior(d, 0, 0, None, rho=rho, se=se)
    m = np.diag(S); W = S / np.sqrt(np.outer(m, m))
    e0 = np.zeros(len(m)); e0[0] = 1
    v = np.zeros(len(m)); v[4:4 + N] = g * np.sqrt(m[4:4 + N]); v /= np.linalg.norm(v)
    panels.append(("(a) unrotated, diagonal", W, np.column_stack([e0, v])))

    # (b) rotated, diagonal metric: u is one coordinate, ridge unchanged
    S, _, _ = posterior(d, 0, 0, "all", rho=rho, se=se)
    m = np.diag(S); W = S / np.sqrt(np.outer(m, m))
    E = np.zeros((len(m), 2)); E[0, 0] = 1; E[4, 1] = sgn
    panels.append(("(b) rotated, diagonal", W, E))

    # (c) rotated, dense block over (beta, U): symmetric-root whitening
    blk = list(range(4)) + list(range(4, 4 + k)) + list(range(4 + N, 4 + N + k))
    M = np.diag(m).copy(); M[np.ix_(blk, blk)] = S[np.ix_(blk, blk)]
    R = _sym_inv_root(M); W = R @ S @ R; W = 0.5 * (W + W.T)
    panels.append(("(c) rotated, dense block", W, E))

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharex=True, sharey=True)
    t = np.linspace(0, 2 * np.pi, 400)
    circ = np.vstack([np.cos(t), np.sin(t)])
    print("Figure 1  (design: visits %d-%d, N=%d, rho=%.1f, sigma_e=%.1f)" % (nmin, nmax, N, rho, se))
    for ax, (title, W, E) in zip(axes, panels):
        C = _slice(W, E)
        r = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
        ev = np.linalg.eigvalsh(W)
        kap = ev.max() / ev.min()
        L = np.linalg.cholesky(C)
        ax.plot(circ[0] * 2, circ[1] * 2, ls=(0, (3, 3)), color=INK2, lw=0.9)
        for s, a in ((1, 1.0), (2, 0.65), (3, 0.35)):
            xy = s * L @ circ
            ax.plot(xy[0], xy[1], color=C1, alpha=a, lw=1.6)
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
        ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"{title}\ncorr {r:+.2f},  cond. {kap:,.0f}", loc="left", color=INK)
        print(f"  {title.replace(chr(10), ' '):38s} slice corr {r:+.4f}  "
              f"slice sds {np.sqrt(np.diag(C)).round(3)}  whitened condition number {kap:,.0f}")
    axes[0].set_xlabel(r"$\beta_0$ (metric units)")
    axes[1].set_xlabel(r"$\beta_0$ (metric units)")
    axes[2].set_xlabel(r"$\beta_0$ (metric units)")
    axes[0].set_ylabel("mean-intercept direction $u$\n(metric units)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C1, lw=1.6), Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 3)))],
               ["posterior slice: 1, 2, 3 sd contours", "2 sd contour of a perfectly scaled target"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig1_ridge")


# ---------------------------------------------------------------- Figure 2
def _paired(path, qty, arm_a="A", arm_r="A_rotdense", sigma0=0.8):
    rows = [r for r in csv.DictReader(open(path)) if r["quantity"] == qty]
    by = defaultdict(dict)
    for r in rows:
        by[(r["cell"], r["seed"])][r["arm"]] = r
    cells = defaultdict(list)
    for (cell, seed), arms in by.items():
        if arm_a in arms and arm_r in arms:
            a, b = arms[arm_a], arms[arm_r]
            ratio = float(b["ess_per_sec"]) / float(a["ess_per_sec"])
            info = float(a["mean_visits"]) * sigma0 ** 2 / float(a["sigma_e"]) ** 2
            cells[cell].append((info, ratio))
    out = []
    for cell, v in sorted(cells.items()):
        info = np.mean([x[0] for x in v]); rs = np.array([x[1] for x in v])
        out.append((cell, info, float(np.exp(np.mean(np.log(rs)))), rs))
    return out


def fig2():
    src = {
        "core grid, q = 2": (os.path.join(DEV, "pilot_rotate_grid.csv"), C1, "o"),
        "stress grid, q = 2": (os.path.join(DEV, "pilot_rotate_stress.csv"), C2, "s"),
        "random intercept, q = 1": (os.path.join(DEV, "pilot_rotate_q1.csv"), C3, "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    print("\nFigure 2  (ESS/sec ratio rotated/unrotated; per cell: geometric mean over seeds)")
    allr = []
    for ax, qty, title in ((axes[0], "intercept", "(a) intercept"), (axes[1], "time", "(b) time slope")):
        for lab, (path, col, mk) in src.items():
            cells = _paired(path, qty)
            q1_slope = qty == "time" and "q = 1" in lab
            for cell, info, gm, rs in cells:
                ax.scatter(np.full(len(rs), info), rs, s=7, marker=mk,
                           color=col, lw=0, alpha=0.35, zorder=2)
                print(f"  {qty:9s} {lab:24s} {cell:28s} info {info:6.1f}  ratio {gm:7.2f}x  seeds {np.round(rs, 2)}")
                if not q1_slope:
                    allr.extend(rs)
            xs = [c[1] for c in cells]; ys = [c[2] for c in cells]
            ax.scatter(xs, ys, s=34, marker=mk, color="white" if q1_slope else col,
                       edgecolor=col, lw=1.2, zorder=3,
                       label=lab + (" (slope not rotated)" if q1_slope else ""))
        ax.axhline(1, color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
        ax.set_yticks([0.5, 1, 3, 10, 30, 100, 300])
        ax.set_yticklabels(["0.5", "1", "3", "10", "30", "100", "300"])
        ax.grid(True, which="major", axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(r"per-subject information  $\bar n\,\sigma_0^2/\sigma_e^2$")
    axes[0].set_ylabel("ESS/sec, rotated / unrotated")
    from matplotlib.lines import Line2D
    mk_ = lambda m, c, fill=True, ms=6: Line2D([], [], ls="", marker=m, ms=ms, color=c,
                                              markerfacecolor=c if fill else "white", markeredgewidth=1.2)
    h = [mk_("o", C1), mk_("s", C2), mk_("^", C3), mk_("^", C3, fill=False),
         Line2D([], [], ls="", marker="o", ms=3, color=INK2, alpha=0.5)]
    l = ["core grid, q = 2", "stress grid, q = 2", "random intercept, q = 1",
         "q = 1 time slope (no random slope, nothing rotated)", "single seed (large marker: geometric mean)"]
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.33, wspace=0.08)
    print(f"  range over rotated per-seed ratios plotted: {min(allr):.2f}x - {max(allr):.2f}x")
    save(fig, "fig2_gain")


# ---------------------------------------------------------------- Figure 3
# R^2 and jmjax's ESS/draw: dev/alpha_missing_information.R (long rotated
# fit, 4 chains x 5000). JMbayes2's ESS/draw: dev/study_common_ess.R, its
# draws analysed with NumPyro's estimator (3 seeds; JMbayes2's own
# estimator gives 0.035 and 0.028).
ALPHA = {
    "aids": dict(R2=0.873, R2_re=0.30, jmb=0.032, jmjax=1.05),
    "pbc2": dict(R2=0.853, R2_re=0.20, jmb=0.025, jmjax=1.72),
}


def fig3():
    ideal = lambda r2: (1 - r2) / (1 + r2)
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    x = np.linspace(0, 0.995, 400)
    ax.plot(x, ideal(x), color=INK, lw=1.6, label="ideal two-block sampler, $(1-R^2)/(1+R^2)$")
    ax.axhline(1, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.text(0.01, 1.08, "independent draws", fontsize=7.5, color=INK2, va="bottom")
    print("\nFigure 3  (alpha ESS per draw)")
    for ds, mk in (("aids", "o"), ("pbc2", "s")):
        a = ALPHA[ds]
        ax.scatter([a["R2"]], [a["jmjax"]], marker=mk, s=38, color=C1, zorder=3,
                   label="jmjax NUTS, measured" if ds == "aids" else None)
        ax.scatter([a["R2"]], [a["jmb"]], marker=mk, s=38, color=C2, zorder=3,
                   label="JMbayes2, measured" if ds == "aids" else None)
        ax.scatter([a["R2"]], [ideal(a["R2"])], marker=mk, s=38, facecolor="white",
                   edgecolor=INK, lw=1.1, zorder=3,
                   label="ideal two-block, at measured $R^2$" if ds == "aids" else None)
        ax.scatter([a["R2_re"]], [ideal(a["R2_re"])], marker=mk, s=38, facecolor="white",
                   edgecolor=C3, lw=1.3, zorder=3,
                   label="ideal, if alpha were blocked with\nthe survival parameters" if ds == "aids" else None)
        print(f"  {ds}: R2 {a['R2']:.3f} -> ideal {ideal(a['R2']):.3f}; JMbayes2 {a['jmb']:.3f}; "
              f"jmjax {a['jmjax']:.2f}; R2(random effects only) {a['R2_re']:.2f} -> ideal {ideal(a['R2_re']):.2f}")
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(0.01, 3)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
    ax.grid(True, which="major", axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(r"$R^2$ of alpha on the other parameters")
    ax.set_ylabel("alpha ESS per draw")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker="o", ms=5, color=INK2), Line2D([], [], ls="", marker="s", ms=5, color=INK2)]
    l += ["aids", "pbc2"]
    ax.legend(h, l, loc="lower left", frameon=False, fontsize=7)
    save(fig, "fig3_alpha")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
