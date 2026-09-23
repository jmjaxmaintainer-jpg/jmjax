"""Gaussian check of vignette Proposition 8 (Section 8.1): why alpha mixes
slowly under a blocked sampler and not under HMC.

Target: N(0, S) over (alpha, R), unit marginal variances, alpha correlated
with one linear combination of R so that R^2(alpha | R) is set exactly.

(a) Two-block Gibbs (alpha | R, then R | alpha), simulated: the lag-1
    autocorrelation of alpha equals R^2 and the alpha chain is AR(1), so the
    integrated autocorrelation time is (1 + R^2) / (1 - R^2).
(b) HMC with the oracle diagonal metric, via the stylised exact-dynamics
    model of theory_rotation_toy.py (Part 3): alpha's IAT stays O(1); the
    correlation costs leapfrog steps per iteration, which grow like
    sqrt(kappa), i.e. like 1/sqrt(1 - R^2), not 1/(1 - R^2).
(c) The same with an unrelated, narrower ridge elsewhere in R (a pair at
    correlation 0.995): HMC's step size is already set by that ridge, so
    alpha's correlation adds nothing until it becomes the narrowest
    direction. A Gibbs chain for alpha pays (a) regardless.

    python3 dev/theory_alpha_toy.py        # < 10 s
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_rotation_toy import hmc_model


def target(R2, d=40, other=None, seed=3):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d); w[-2:] = 0.0; w /= np.linalg.norm(w)
    S = np.eye(d + 1)
    S[0, 1:] = np.sqrt(R2) * w; S[1:, 0] = S[0, 1:]
    if other is not None:                       # unrelated narrow ridge in R
        S[-1, -2] = S[-2, -1] = other
    return S


def gibbs_alpha(S, n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    Srr, c = S[1:, 1:], S[1:, 0]
    bcoef = np.linalg.solve(Srr, c)
    r2 = c @ bcoef
    Lr = np.linalg.cholesky(Srr - np.outer(c, c))
    x, a = 0.0, np.empty(n)
    for t in range(n):
        R = c * x + Lr @ rng.standard_normal(len(c))
        x = bcoef @ R + np.sqrt(1 - r2) * rng.standard_normal()
        a[t] = x
    return r2, np.corrcoef(a[:-1], a[1:])[0, 1]


if __name__ == "__main__":
    e = np.zeros(41); e[0] = 1.0
    print("(a)+(b)  alpha correlated with R only")
    print(f"{'R^2':>6} {'Gibbs lag1':>10} {'Gibbs IAT':>9} | {'HMC IAT':>7} {'steps':>6} {'grads/indep':>11}")
    for R2 in (0.5, 0.8, 0.93, 0.95, 0.99):
        S = target(R2)
        r2, lag1 = gibbs_alpha(S)
        tau, steps, _ = hmc_model(S, e)
        print(f"{r2:6.3f} {lag1:10.3f} {(1 + r2) / (1 - r2):9.1f} | {tau:7.2f} {steps:6.1f} {tau * steps:11.1f}")
    print("\n(c)  plus an unrelated ridge in R at correlation 0.995")
    print(f"{'R^2':>6} {'Gibbs IAT':>9} | {'HMC IAT':>7} {'steps':>6} {'grads/indep':>11}")
    for R2 in (0.0, 0.5, 0.8, 0.93, 0.95, 0.99):
        S = target(R2, other=0.995)
        tau, steps, _ = hmc_model(S, e)
        print(f"{R2:6.3f} {(1 + R2) / (1 - R2):9.1f} | {tau:7.2f} {steps:6.1f} {tau * steps:11.1f}")
