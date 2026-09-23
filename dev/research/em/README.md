# EM warm start for jmjax's MLE paths — research, not wired in

`em.py` is **not on the Python path** and nothing in jmjax imports it.
It lives here so the work and its measurements survive, not because it
runs.

## What it is

A joint EM phase following `JM`'s structure: an E-step for the random
effects, a closed-form M-step for the longitudinal block, and a short
BFGS step on the survival block, with JM's schedule (`maxit` 20 for five
iterations then 4) and its stopping criteria (`tol1 = 1e-3`,
`tol2 = 1e-4`, `tol3 = sqrt(eps)`).

## What was verified

Each piece was checked independently, and the checks are the reason the
negative conclusion below is trustworthy:

- **M-step** — recovers known parameters; handles the padding mask
  (ignoring it inflated `sigma_e` by 398% on 35%-padded data); carries the
  random-effects uncertainty into `sigma_e` via `E[bb'] - E[b]E[b]'`,
  which a "plug in the posterior mean" version drops.
- **Loop** — monotone, 0 log-likelihood drops from both a cold and a warm
  start on an exact-E-step linear mixed model, both reaching the same
  optimum.
- **`h(b)`** — matches the backend's own marginal log-likelihood exactly
  at q=1 and q=2 on real `prothro` data, and matches brute-force numerical
  integration to `4e-08` for the Weibull case.

## What it achieved

On real `prothro` with a spline baseline, where L-BFGS-B stopped after 5
iterations 78 log-likelihood units short:

| start | loglik | alpha |
|---|---|---|
| cold L-BFGS-B | -14083.14 | -0.0008 |
| EM-warmed L-BFGS-B | -14004.49 | -0.0346 |
| cold BFGS | -14004.80 | -0.0391 |
| EM-warmed BFGS | **-14001.88** | **-0.0417** |

against `JM`'s -0.0400 and jmjax's own MCMC -0.0411. 78.7 of the 78.3
missing units recovered, with no iteration increasing the objective.

## Why it is not wired in

**A one-line default change does the same job, more cheaply and
everywhere.** Switching `opt_method` to `BFGS` clears the same failure on
all fitting paths. EM was wired into 1 of 13; extending it means 12
per-path edits, each with its own argument list and its own verification.

**It costs 2.6x on fits that already work and changes nothing** — and can
leave the optimizer at a *worse-conditioned* point. EM stops on relative
parameter change, so it may halt where the gradient is still substantial:
measured on a fit that converged cleanly from cold (`grad_max` 0.62), the
EM-warmed run took 0 optimizer iterations and ended at `grad_max` 1.75,
above the threshold, for an essentially identical log-likelihood.

**There is no remaining case to justify it.** With BFGS as the default,
no dataset tried — simulated, AIDS, prothro, liver, mental, heart.valve —
shows jmjax's MLE failing where `JM`'s succeeds. The one case both fail
(`epileptic` with time in days, range [3, 2400]) is a units problem, fixed
by rescaling rather than by any optimizer.

`JM` runs EM unconditionally because it cannot distinguish a stalled
optimizer from a converged one. jmjax can, via
`fit$convergence$grad_max` — which is why detection replaced prevention
here.

## When to reach for this again

If a dataset appears where BFGS reports `converged = FALSE` with a large
`grad_max` and no optimizer setting fixes it. Then the case is concrete,
and this is ready.

## Using it

`em.py` needs `jmjax_backend` importable for `common.build_h_fn`:

```python
import sys; sys.path.insert(0, "<system.file('python', package='jmjax')>")
sys.path.insert(0, "<this directory>")
import em
```

`jm_fit(..., control = list(return_backend_data = TRUE))` returns the
arrays `em.py` needs, which is the one piece of plumbing kept in the
package — it is generally useful for inspection, not EM-specific.

## Scripts

Kept alongside: the verification harnesses (`verify_em_h.py`,
`test_em_estep_phase1.R`, `test_em_mstep_phase2.R`), the rescue
demonstrations (`test_joint_em_rescue.R`, `test_joint_em_q2_rescue.R`),
and the cost measurement (`test_em_always_cost.R`).
