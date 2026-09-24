# Methods paper: outline for review

Status: draft 1 of the paper is in `dev/paper/` (LaTeX, 23 Sep 2026;
`latexmk -pdf methods-paper.tex`). The outline below is kept for the record;
the draft supersedes it where they differ (e.g. Figure 1 has three panels). The
source for every number is `vignettes/jmjax-reparameterization.Rmd`
(the full technical record), which this paper condenses.

## 1. Target

- **Primary: Journal of Computational and Graphical Statistics (JCGS).**
  Computational-methods readership, and it requires code and data for
  reproducibility, which the `dev/` studies already provide.
- **Alternative: Statistics and Computing.**
- **Length:** aim for about 25 pages in the journal's format plus an online
  supplement. Check the current author guidelines for the exact limit and
  the reproducibility requirements before drafting.
- **Relation to the other two documents:** the vignette stays the complete
  record. The JSS paper (later, after 0.3.0 is on CRAN) cites this one for
  the method and is about the software; it must not reuse this paper's
  text or repeat its derivations.

## 2. Working title (pick one, or suggest)

1. *Rotating away the location degeneracy: an exact reparameterization for
   Hamiltonian Monte Carlo in hierarchical and joint models*
2. *Absorbable directions, exact rotations, and the limits of blocked
   samplers in Bayesian joint models*
3. *Efficient Hamiltonian Monte Carlo for joint longitudinal-survival
   models: an exact rotation for the location degeneracy*

Title 1 leads with the general idea, 3 with the application. The choice
depends on how strong the generality example in Section 7 turns out.

## 3. Claimed contributions

1. **Characterization of the absorbable directions.** The random-effect
   directions that fixed effects can absorb, for any design: a
   basis-independent exact construction (Props 1-4), an example where the
   natural column-by-column check silently misses them (Example 1), and,
   for joint models, the condition the survival submodel imposes
   (Condition (S), Prop 5).
2. **An exact rotation plus a small dense metric block.** One fixed
   Householder rotation of every random-effect column puts the degeneracy
   into a k-by-q block of sampling coordinates. It commutes with the
   random-effect covariance, so it is exact for every value of it
   (Prop 6, Lemma 1). The ridge's geometry (Prop 7) explains why a
   diagonal metric makes it narrow, why one small dense block fixes that,
   and where a fixed metric is weakest. The predicted weak spots were
   confirmed by a stress grid. The construction is O(Nk).
3. **Why it beats the alternatives.** Hierarchical centering fixes the
   coefficients but costs the variance components. Sweeping the directions
   out (constraining) miscalibrates the intervals without a correction.
   The rotation needs no correction and leaves every reported quantity
   unchanged.
4. **Blocked samplers and the association parameter.** For a sampler that
   updates alpha given the other parameters, the lag-1 autocorrelation
   is the fraction of alpha's posterior variance those parameters explain
   (Prop 8, via Liu, Wong & Kong 1994). This is an application, not new
   theory. The contribution is the diagnosis: in joint models alpha
   trades off against the level of the log hazard, so R^2 is 0.85-0.87 on
   real data, mostly from the survival block. That caps any blocked
   sampler at one effective draw in 12-15 and explains about three
   quarters of the measured gap to JMbayes2.
5. **Evidence.** Simulation grids (q = 2 core and stress, q = 1),
   a 100-replicate calibration study, two real datasets, and the
   comparison with JMbayes2.

## 4. Section plan (target pages)

1. **Introduction (2).** Joint models; alpha as the primary target; why
   MCMC mixing matters; the two findings; contributions list.
2. **Model and the location degeneracy (2).** Non-centred hierarchical
   model, general mixed model first, then the joint model. The degeneracy
   for the intercept and, independently of the covariates, for the time
   slope.
3. **Absorbable directions (3).** Definitions, Props 1-4, Example 1,
   Condition (S) and Prop 5. Proofs to the supplement.
4. **The rotation and the metric (3).** Construction, Prop 6, Lemma 1,
   Prop 7, the fixed-metric residual and what it predicts. Implementation
   and cost.
5. **Blocked samplers and the association parameter (2).** Prop 8, the
   Gaussian check, the measured R^2 and its split between random effects
   and survival block.
6. **Simulation (4).** Design; efficiency (q = 2 core grid, q = 1 grid);
   stress grid; calibration, with one sentence on alpha's small-sample
   bias.
7. **Beyond joint models (1.5).** The generality example (below).
8. **Real data and JMbayes2 (3).** aids and pbc2: rotated vs unrotated,
   and against JMbayes2 on the coefficients and on alpha.
9. **Discussion (1.5).** Known costs (alpha at sparse follow-up; the
   fixed metric vs the retired sweep in the densest designs), extensions
   (a metric that follows the sampled sigma_0), related work.

**Supplement:** proofs; full tables; the retired sweep and why it needs a
correction; the list of `dev/` scripts that reproduce each table.

## 5. Figures and tables (draft list)

- **Fig 1. The ridge, before and after.** Two-panel posterior contour of
  (beta_0, the mean-intercept direction): unrotated under a diagonal
  metric; rotated with the dense block. Built from the closed-form
  Gaussian toy, so it is exact rather than sampled.
- **Fig 2. Gain against information per subject.** ESS/sec ratio (log
  scale) against visits per subject, pooling the q = 2 core, stress and
  q = 1 grids, intercept and time slope marked separately.
- **Fig 3. Alpha under blocked sampling.** Predicted ESS per draw
  (1 - R^2)/(1 + R^2) as a curve, with the measured points for JMbayes2
  and jmjax on aids and pbc2.
- **Table 1.** Simulation efficiency summary (both grids).
- **Table 2.** Calibration: coverage and sd_ratio, rotated vs unrotated.
- **Table 3.** Real data: rotated vs unrotated, and both against
  JMbayes2.

## 6. Work still needed before drafting

1. **Generality example (Section 7): DONE 23 Sep.** Results in
   `dev/generality/results_orthodont_toenail.csv` (4 chains x (1000 + 1000),
   3 seeds). Rotated + dense block over unrotated, geometric mean over seeds,
   per 1000 gradient evaluations (per second in brackets; each run is 1-8 s
   including JIT compilation, so per gradient is the fair measure):

   | parameter | Orthodont (LMM, q = 2) | toenail (logistic GLMM, q = 1) |
   |---|---:|---:|
   | intercept | 7.1x (4.4x) | 5.5x (4.1x) |
   | subject-constant covariate | 7.1x (4.4x), `male` | 6.7x (5.1x), `trt` |
   | within-subject coefficients | 2.4-2.5x | 1.3x |
   | variance components | 1.3-1.6x | 1.1x |

   The rotation with a diagonal metric alone gives 2.3-2.8x on the
   targeted coefficients; the dense block supplies the rest. Posterior
   means agree to 0.14 posterior SDs; R-hat <= 1.011 everywhere; no
   divergences. The unrotated arm is the parameterization brms and
   rstanarm use by default. **Title 1 is supported.**

   Original plan, kept for the record: The strongest thing a referee will
   ask is whether this is a trick for joint models only. Plan: standalone
   NumPyro code, not jmjax, on two classic mixed-model datasets with a
   subject-constant covariate:
   - a linear mixed model: `nlme::Orthodont` (distance ~ age * Sex,
     random intercept and slope; Sex is subject-constant);
   - a logistic mixed model: the toenail onychomycosis data (treatment is
     subject-constant, binary outcome over visits).

   Measure ESS/sec for the intercept and the subject-constant covariate,
   unrotated vs rotated with the dense block, and report what Stan/brms's
   default (non-centred) gives on the same model as a reference point.
   Roughly one session to write, one short run.
2. **Literature check** before claiming novelty, specifically: interweaving
   of centred and non-centred parameterizations (ASIS; Yu & Meng 2011);
   partial non-centring (Papaspiliopoulos, Roberts & Skold 2007);
   complexity of Gibbs samplers for multilevel models (Zanella & Roberts);
   Stan's QR reparameterization and sum-to-zero constraints; restricted
   spatial regression. The vignette's Section 10 covers some of these.
3. **Figures 1-3.** Fig 1 from `dev/theory_rotation_toy.py`; Figs 2-3 from
   the committed result CSVs. No new fits needed.

## 7. Decisions

- **Journal (decided 23 Sep):** JCGS first; then Statistics and Computing,
  Communications in Statistics - Simulation and Computation, or similar.
- **Title (decided):** title 1 - the generality example (Section 6, item 1)
  reproduced the gains outside joint models.
- **Authors:** Changbin Guo, sole author for now.
- **JMbayes2 comparison (proposed, awaiting confirmation):** keep a narrow
  version here, framed as blocked Metropolis-within-Gibbs against HMC
  rather than package against package:
  - alpha: the ESS-per-draw decomposition and the R^2 measurement, which
    contribution 4 needs;
  - the regression coefficients: parity with hierarchical centering, as
    an external check that the rotation recovers what centering buys,
    without centering's cost on the variance components.

  Leave out JMbayes2's own degradation with the number of covariates
  (vignette 8.2): it is a property of one package's implementation, not
  of the method, and reads as criticism of another package. It stays in
  the vignette. The broad benchmark (JM, joineRML, JMbayes2, rstanarm's
  stan_jm), usability and prediction go in the JSS paper.
