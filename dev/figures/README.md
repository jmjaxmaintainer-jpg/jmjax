# Methods-paper figures

`python3 dev/figures/make_figures.py` (from the package root) rebuilds all
three as PDF and PNG and prints every plotted number. NumPy and Matplotlib
only; no model fits. Checked 23 Sep 2026: every per-cell ratio printed for
Fig 2 matches vignette Sections 5.3, 6.2 and 6.4, and every Fig 3 value
matches Section 8.1.

## Draft captions

**Figure 1. The location ridge and what removes it.** Exact conditional
posterior of the fixed intercept beta_0 and the mean-intercept direction u,
through the mode, in the units a HMC metric sees them (closed-form
Gaussian longitudinal submodel with variance components fixed at the
simulation values: N = 300, 2-11 visits, rho = 0.3, sigma_e = 0.3;
`dev/theory_rotation_toy.py`). Solid: 1, 2 and 3 sd contours; dashed: the
2 sd contour of a perfectly scaled target. (a) Unrotated coordinates with a
diagonal metric: u is spread over all N random intercepts and trades off
almost one-for-one with beta_0. (b) Rotation alone makes u a single
coordinate but leaves the ridge in place. (c) The small dense metric block
over (beta, U) removes it. kappa is the condition number of the whole
whitened posterior; what remains in (c) comes from directions outside the
ridge.

**Figure 2. The gain grows with how tightly each subject pins its own
level.** ESS per second of the reported coefficient, rotated fit over
unrotated fit on the same simulated data, against per-subject information
n-bar sigma_0^2 / sigma_e^2 (sigma_0 = 0.8). Large markers: geometric mean
over 3 seeds in a design cell; small: single seeds. Core q = 2 grid
(Section 6.2), stress q = 2 grid (Section 6.4) and the random-intercept
(q = 1) grid (Section 5.3). In (b), open triangles are the q = 1 time slope,
which has no random effect and hence nothing to rotate; it moves only
through the step size. The q = 1 intercept ratios compare against
unrotated fits that had mostly not converged at this run length, so only
their order of magnitude is meaningful.

**Figure 3. Why a blocked sampler mixes alpha slowly.** Curve: ESS per draw
of an ideal two-block sampler, (1 - R^2)/(1 + R^2) (Proposition 8). R^2 of
alpha on all other parameters, measured from a long NUTS fit, is 0.873
(aids) and 0.853 (pbc2); black open markers are the ideal sampler there.
JMbayes2 does 2-3 times worse than that ideal; jmjax's NUTS draws are
effectively independent. Green open markers: the ideal if alpha were
updated in one block with gamma and the baseline-hazard coefficients, so
that only the random effects are conditioned away (R^2 0.30 and 0.20).
