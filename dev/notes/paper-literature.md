# Methods paper: literature check (23 Sep 2026)

Purpose: decide how the novelty claims in `paper-methods-outline.md` can
be worded. Status per item: **verified** = checked against the publisher,
arXiv or official documentation page; **to read** = existence and topic
confirmed, but the full text must be read before the paper cites it for a
specific claim.

## 1. Work the paper builds on or must distinguish itself from

### Centring and non-centring
- Gelfand, Sahu & Carlin (1995), *Biometrika* 82(3), 479-488: hierarchical
  centring for normal linear mixed models. Already cited. Their 1996 GLMM
  companion (*Bayesian Statistics 5*) should be added.
- Papaspiliopoulos, Roberts & Skold (2003, 2007 *Statistical Science*):
  non-centred and partially non-centred parameterizations. Already cited.
- Yu & Meng (2011), *JCGS* 20(3): ASIS - interweave centred and non-centred
  steps instead of choosing. **Verified.** Same journal as our target, so
  it must be cited and distinguished: ASIS alternates parameterizations
  inside a Gibbs-type sampler; ours is one fixed, exact parameterization
  for gradient-based sampling.
- Gorinova, Moore & Hoffman (2020), ICML: automatic reparameterization in
  probabilistic programs - searches over partial non-centring, including a
  variational criterion. **Verified** (arXiv 1906.03028). Distinguish: it
  interpolates centred vs non-centred per variable; it does not address
  the fixed-effect / random-effect location ridge.

### Sweeping, sum-to-zero and other constraints (the closest prior art)
- Vines, Gilks & Wild (1996), *Statistics and Computing* 6(4), 337-346:
  "Fitting Bayesian multiple random effects models" - reparameterizes
  random effects by sweeping out their means (sum-to-zero). **To read**
  (reference details confirmed from Zanella & Roberts' bibliography). Almost
  certainly the origin of the idea behind our retired sweep, and must be
  cited as such.
- Stan's `sum_to_zero_vector`: constrains a vector to sum to zero through
  an orthonormal (Helmert-type) basis, related to the isometric log-ratio
  transform. **Verified** (Stan Reference Manual, Constraint Transforms).
  This is the closest geometric relative of our rotation, and the paper
  must say how they differ (below).
- Ogle & Barber (2020), *Ecological Applications* 30(7), e02159: "post-
  sweeping" of random effects to recover identifiable quantities. Already
  cited as Ogle et al. (2020). **Verified.**
- Restricted spatial regression: Hodges & Reich (2010, *The American
  Statistician* 64(4)); Hanks et al. (2015); Khan & Calder (2022).
  Already cited. Same projection idea in the spatial setting, where it is
  known to change inference - the same lesson as our retired sweep.

### Multilevel samplers and reparameterization for MCMC
- Browne, Steele, Golalizadeh & Green (2009), *JRSS A* 172(3), 579-598.
  **Read (full text).** The three techniques are hierarchical centring
  (Gelfand et al. 1995), orthogonal predictors, and parameter expansion
  (Liu, Rubin & Wu 1998), run in MLwiN / WinBUGS with single-site
  Metropolis and Gibbs steps on discrete-time survival (random-intercept
  logit) models.
  - Their "orthogonal" option (Sec 3.2, 4.1, Appendix B) is Gram-Schmidt
    on the FIXED-EFFECT predictors among themselves (P* = WP, W lower
    triangular, order-dependent). Same family as Stan's QR trick; it does
    not orthogonalize against the random effects. So it is not our
    construction.
  - But they name our problem. Sec 4.1: predictors should ideally also be
    "close to orthogonal to the woman identifiers" (the 0-1 cluster
    indicator vectors), because a predictor correlated with them makes
    the fixed effect and the random effects "play similar roles", giving
    highly correlated, poorly mixing chains. Sec 5 lists this as future
    work: orthogonal predictors "also close to orthogonal to the dummy
    variables representing the level 2 unit identifiers would seem
    preferable", possibly combined with centring. Our absorbable
    subspace is exactly the span they point at, done exactly (the
    random effects are rotated, not the predictors) and for every
    random-effect column. Cite as the clearest statement of the problem
    and of the open question.
  - Hierarchical centring (an exact reparameterization, like ours) wins
    big when the cluster variance is large (mastitis: ESS gains >100x)
    and fails when it is small (Indonesia: ESS 1665 -> 28 for the
    intercept), citing Gelfand et al. (1995): small random effects make
    the centred effects strongly correlated with the predictors carrying
    their mean. Useful contrast: centring moves the ridge between the
    small- and large-variance regimes; the rotation removes the location
    directions for any variance (q=1 exactly; q>=2 up to the dense block).
  - Also relevant for Prop 8: the random-effect variance mixes worst
    (ESS 13-20 of 25,000) under all their Gibbs/Metropolis schemes until
    parameter expansion. Another blocked-sampler weak spot, consistent
    with our framing.
- Zanella & Roberts (2021), *Bayesian Analysis* 16(4), 1309-1391 (with
  discussion and rejoinder). **Read (full text incl. discussion).**
  - Multigrid decomposition: an ANALYSIS device for Gibbs samplers on
    nested and crossed Gaussian models without covariates. The chain
    splits into independent sub-chains on subspaces intrinsic to the
    model (Remark 1: not tied to the parameterization); the coarsest
    (global-mean) one is the slowest (Thms 1-2, 9-11).
  - Sec 4.2, Thm 6: imposing linear identifiability constraints (e.g.
    zero means of crossed factors) always improves Gibbs convergence.
    They sample the law conditioned on the constraint - a constraint,
    not an exact reparameterization of the original posterior.
  - Sec 5.1, Table 3 (Poisson crossed model, Stan): the constraints also
    help HMC/NUTS dramatically (min ESS 3.7 -> 4977; leapfrog steps per
    iteration 325 -> 20). This is prior evidence that the mean ridge hurts
    HMC and that removing it helps; the paper must credit it. Rejoinder
    caveats: the runtimes exclude warm-up, and Carpenter's tuned Stan code
    halves NUTS time without changing the relative conclusions.
  - Discussion, C. P. Robert: asks whether the impact of these
    parameterization choices "has been studied for HMC". Rejoinder: an
    HMC version of the multigrid analysis "would be a very interesting
    line of research". So as of 2021 the authors regard the HMC-side
    analysis as open - our dense-metric analysis (Lemma 1, Prop 7) is a
    contribution on exactly that side, for the absorbable directions.
  - Discussion, Yang & Liu, Sec 3 "Incorporating regression covariates":
    CLOSEST PRIOR ART FOR CLAIM 2. For y_ij = X_ij'beta + a_i + e_ij
    (random intercepts) they set P = (Xbar'Xbar)^{-1/2} Xbar' (Xbar =
    subject-level covariate means) with complement L, L'L + P'P = I;
    Thm 3.1: the Gibbs chain splits into {La} and {beta, Xbar'a}, with
    rate = squared maximal correlation of beta and Xbar'a. That is the
    covariate-weighted absorbable subspace for a random intercept, used to
    analyse Gibbs. Differences from ours: (i) analysis of Gibbs, not
    sampling coordinates for HMC; (ii) random intercept only, with Xbar as
    subject means (exact only when the absorbed columns are subject-
    constant), whereas we build the subspace for every random-effect
    column from the design, including slopes and time-varying designs
    (Example 1); (iii) they note (Secs 3.2-3.3) that the clean split fails
    for general X; (iv) no joint-model condition. Sec 4 of their
    discussion: partial centring with exact one-step sampling.
  - Discussion, Zhou & Zhou: numerical CP / PNCP / NCP study for an LMM
    with covariates (Gibbs). Discussion, Irie & Sugasawa: plug-in, adaptive
    and ASIS choices of parameterization. Neither touches HMC.
- Gelfand & Sahu (1999), *JASA* 94(445), 247-253: identifiability,
  improper priors and Gibbs sampling for GLMs (cited by Z&R for
  constraints). To add.
- Xie & Carlin (2006), *JSPI* 136(10), 3458-3477: measures of Bayesian
  learning and identifiability in hierarchical models. To add.
- Browne (2004), *Multilevel Modelling Newsletter* 16(1), 13-25:
  centring and parameter expansion for crossed random effects. To add
  (background only).
- Papaspiliopoulos, Roberts & Zanella (2020), *Biometrika* 107(1):
  scalable inference for crossed random effects (collapsed Gibbs).
  **Verified.** Background for the blocked-sampler discussion.
- Parameter expansion: Liu, Rubin & Wu (1998); Gelman et al. (2008,
  "Using redundant parameterizations to fit hierarchical models", JCGS).
  To add.

### HMC geometry and preconditioning
- Betancourt & Girolami (2013); Neal (2011); Hoffman & Gelman (2014).
  Already cited.
- Hird & Livingstone (2025), *JMLR*: effectiveness of linear
  preconditioning. Already cited; supports the dense-block argument.
- Stan User's Guide, QR reparameterization of the fixed-effect design.
  Already cited. Distinguish: it decorrelates the columns of X from each
  other, not X from the random effects.
- Marginalizing random effects for HMC in linear mixed models (recent
  work, found by title only). **To read.** An alternative that only
  applies when the random effects can be integrated out (Gaussian
  outcomes); not available for joint models or GLMMs.

### Blocked samplers and the association parameter
- Liu, Wong & Kong (1994), *Biometrika* 81(1), 27-40. Already cited
  (Proposition 8 rests on it).
- Roberts & Sahu (1997), *JRSS B*. Already cited.

### Joint-model software (context, and for the JSS paper later)
- JMbayes2 (Rizopoulos, Papageorgiou & Miranda Afonso; CRAN): Metropolis-
  within-Gibbs. Already used as the blocked-sampler comparator.
- rstanarm `stan_jm` (Brilleman et al. 2018): joint models in Stan with
  NUTS. **Verified** (rstanarm vignette). Relevant to the methods paper
  too: it is an HMC joint-model implementation, so the paper should say
  whether it suffers the same location ridge (it uses the non-centred
  parameterization, so it should).
- INLAjoint (Rustand et al. 2024, *Biostatistics* 25(2), 429ff; R package
  paper arXiv 2402.08335): fast approximate inference by INLA. **Verified.**
  Context only: a different inferential approach, not MCMC.

## 2. What the paper can claim, and how to word it

Established, not ours:
- that centring/non-centring choice drives mixing, and that sweeping or
  constraining random effects is one way to remove the location ridge;
- that blocked samplers slow down with between-block correlation, and the
  lag-1 identity behind Proposition 8.

Also established, and to be credited explicitly (from the reading):
- that the mean ridge hurts HMC/NUTS and that removing it by constraint
  helps (Zanella & Roberts 2021, Sec 5.1);
- the covariate-weighted absorbable direction for a random intercept, as
  a device for analysing Gibbs (Yang & Liu, discussion of Z&R, Sec 3);
- that predictors ideally should be orthogonal to the cluster indicators,
  posed as open (Browne et al. 2009, Secs 4.1 and 5).

Ours, worded "to our knowledge" (never "first"); items 1 and 2 of section
3 below are now read and did not change these, but narrowed claim 2:
1. **Rotate, don't constrain.** Stan's sum-to-zero type, sweeping (Vines
   et al.) and restricted spatial regression all REMOVE the mean or
   absorbable directions, which changes the model or needs a correction
   (our retired sweep miscalibrated without one). The rotation KEEPS them
   as explicit sampled coordinates, so the model, prior and every reported
   quantity are unchanged, for every value of the random-effect
   covariance.
2. **The absorbable subspace in general, as sampling coordinates.** Yang
   & Liu have the random-intercept case with subject-mean covariates, for
   Gibbs analysis. Ours: the exact construction for every random-effect
   column including slopes, from the design (no reliance on column names
   or subject-constant covariates), with an example where the obvious
   check fails, and used as the coordinates HMC samples in. Word it as
   generalising and operationalising their decomposition, and cite it.
3. **The metric.** The analysis of the ridge under a diagonal HMC metric
   and the small dense block that removes it, with the predicted weak
   spots confirmed. Frame it as answering the HMC question Robert raised
   and Zanella & Roberts called open in their rejoinder.
4. **Joint models.** The condition the survival submodel imposes, and the
   diagnosis of alpha under blocked samplers with the measured R^2.

## 3. To read before drafting (in this order)

1. ~~Browne et al. (2009)~~ - READ 23 Sep: orthogonal = Gram-Schmidt
   among fixed-effect predictors; orthogonality to cluster indicators
   posed as future work (see section 1).
2. Vines, Gilks & Wild (1996) - exact form of their sweeping. Still
   needed.
3. ~~Zanella & Roberts (2021)~~ - READ 23 Sep incl. discussion: see
   section 1; Yang & Liu discussion is the closest prior art for claim 2.
4. Gelfand, Sahu & Carlin (1996) GLMM paper; Gelman et al. (2008).
5. The marginalized-LMM HMC paper, to cite correctly.

Item 2 can still change claim 1; the rest affect citations only.
These need the full texts, which should come through your library access.
