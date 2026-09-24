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
- Vines, Gilks & Wild (1996), *Statistics and Computing*: "Fitting
  Bayesian multiple random effects models" - reparameterizes random
  effects by sweeping out their means (sum-to-zero). **To read.** Almost
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
- Browne, Steele, Golalizadeh & Green (2009), *JRSS A* 172(3), 579-598:
  "three reparameterization techniques" for MCMC in multilevel models
  (MLwiN), applied to discrete-time survival. **To read - priority.** The
  abstract does not name the three; from MLwiN they are believed to be
  hierarchical centring, an orthogonal parameterization of the fixed
  effects, and parameter expansion. If their orthogonal option
  orthogonalizes predictors against the random effects rather than among
  themselves, it is closer to our construction than anything else here.
- Zanella & Roberts (2021), *Bayesian Analysis* 16(4): "Multilevel linear
  models, Gibbs samplers and multigrid decompositions" (with discussion).
  **Verified.** Reparameterizes nested random effects into group means and
  deviations and proves Gibbs complexity bounds. Related in spirit (it
  separates the mean directions); distinguish: Gibbs samplers, nested
  factors without covariates, versus our HMC and covariate-defined
  subspaces.
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

Ours, worded "to our knowledge" (never "first") until items 3.1-3.3 below
are read:
1. **Rotate, don't constrain.** Stan's sum-to-zero type, sweeping (Vines
   et al.) and restricted spatial regression all REMOVE the mean or
   absorbable directions, which changes the model or needs a correction
   (our retired sweep miscalibrated without one). The rotation KEEPS them
   as explicit sampled coordinates, so the model, prior and every reported
   quantity are unchanged, for every value of the random-effect
   covariance.
2. **The absorbable subspace in general.** Not just the mean of the random
   intercepts: the covariate-weighted directions that any subject-constant
   column absorbs, for every random-effect column including slopes, built
   from the design without relying on column names, with an example where
   the obvious check fails.
3. **The metric.** The analysis of the ridge under a diagonal HMC metric
   and the small dense block that removes it, with the predicted weak
   spots confirmed.
4. **Joint models.** The condition the survival submodel imposes, and the
   diagnosis of alpha under blocked samplers with the measured R^2.

## 3. To read before drafting (in this order)

1. Browne et al. (2009) - what their "orthogonal" parameterization is.
2. Vines, Gilks & Wild (1996) - exact form of their sweeping.
3. Zanella & Roberts (2021) - how close multigrid decomposition is to
   isolating the absorbable directions.
4. Gelfand, Sahu & Carlin (1996) GLMM paper; Gelman et al. (2008).
5. The marginalized-LMM HMC paper, to cite correctly.

Items 1-3 can change claim 1 or 2 above; the rest affect citations only.
These need the full texts, which should come through your library access.
