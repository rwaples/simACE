# ADR 0025: Vertical transmission of the common-environment component

## Status

Accepted and implemented 2026-10-14. Amends
[0009](0009-relationship-semantics-home.md): the shared-environment
coefficient of a relationship type now depends on the transmission
coefficients and is undefined for lineage-mixed types.

## Context

simACE draws the common-environment component $C$ fresh for each maternal
household in every generation (`reproduce()` in
`simace/simulation/simulate.py`). Full sibs, maternal half-sibs and MZ twins
share all of $C$; parent–offspring pairs, paternal half-sibs, grandparents,
avuncular pairs and cousins share none of it. Methods that pool relationship
types (a planned minimum-distance estimator over EPIMIGHT's per-relationship
liability correlations) need test pedigrees in which parent and offspring
share environment, so that $h^2$ estimated from parent–offspring pairs is
biased by $C$ the way it is in real data.

## Decision

1. **Model.** For each trait $t$,
   $C_{t,child} = b_{m,t} C_{t,mother} + b_{f,t} C_{t,father} + u_{t,household}$.
   Liability stays $A + C + E$; the pedigree gains no columns.
2. **Coefficients.** $b_m$ and $b_f$ are separate, per trait, scalar (not
   per generation), default 0, each in $[0, 1]$. Config:
   `pedigree.trait{t}.C_transmission.{maternal,paternal}`, flat keys
   `Ctm_maternal{t}` and `Ctm_paternal{t}`. Transmission is within a trait
   only. Config-load rejects $b_m^2 + b_f^2 > 1$. This is a scope
   restriction, not a mathematical necessity: the bound is necessary under
   random or positive assortative mating with constant $C$, but
   disassortative mating or growing $C$ could make larger values feasible,
   and those cases are excluded on purpose.
3. **Configured C is total, in expectation.** Each generation, the
   covariance $U$ of $(u_1, u_2)$ is the configured $C$ covariance matrix
   ($C_{1,g}$, $C_{2,g}$, $r_C\sqrt{C_{1,g}C_{2,g}}$) minus the realized
   covariance $S$ of the transmitted parts over that generation's offspring.
   Realized rather than analytic, because under assortative mating with
   $b_f > 0$ the mates' $C$ covariance is induced and has no simple closed
   form. The contract is stochastic, like the fresh draw it replaces: the
   realized $\mathrm{Var}(C)$ varies by sampling and carries a
   finite-household bias of $-\sum_h w_h^2 \cdot U$ ($w_h$ the household's
   share of offspring, about $10^{-4}$ relative at $N = 10^5$), left
   uncorrected so that the zero-coefficient path means the same thing.
   Feasibility is checked twice. **Structurally**, at config-load: the
   random-mating population residual
   $U = \Sigma - B_m \Sigma B_m - B_f \Sigma B_f$
   ($\Sigma$ the configured $C$ covariance matrix, $B$ the diagonal
   coefficient matrices) must be positive semidefinite; for example
   $r_C = 1$ with unequal trait coefficients is rejected.
   **By sampling**, at run time: $S$ splits exactly into the parents' own
   terms $B_m V_m B_m + B_f V_f B_f$ and the mates' cross term
   $B_m V_{mf} B_f + B_f V_{fm} B_m$. Along an eigenvector of $U$ with a
   negative eigenvalue, a cross term beyond 4 of its standard errors is
   assortative mating pushing $S$ past the target, and stops the replicate.
   Otherwise an excess within a multiple of the standard error of the own
   terms (effective size $1/\sum_h w_h^2$) is clipped to 0, logged as a
   warning, and recorded; a larger one stops the replicate with the
   generation and the offending values. The clip records travel in the
   pedigree's Parquet metadata to `report.yaml`
   (`truth.recorded_pedigree.c_transmission_clips`, plus a line in
   `limitations`), so a run whose realized $C$ exceeds the configured total
   says so. At config-load, a warning flags a trait whose first-generation
   transmitted share $b_m^2 + b_f^2 + 2 b_m b_f r_{CC}$ exceeds 1, with
   $r_{CC} = r\,C/V_P$ the mates' $C$ correlation induced by assortment at
   liability correlation $r$. This is the transmitted part of the sibling
   environment correlation, $\psi = f_F^2 + f_M^2 + 2 f_F f_M (u + p\varphi^2)$,
   in Rao, Morton and Cloninger (1979), eq. 2, which must not exceed 1 for
   standardized $C$; with no social homogamy ($u = 0$) their $p\varphi^2$ is
   our $r_{CC}$, and $\varphi^2 = C/V_P$ while $A$ and $C$ are uncorrelated,
   so the equilibrium share is larger. Cloninger, Rice and Reich (1979) give
   the same quantity as the denominator of their eq. 7 (with $r_{BB}$, the
   mates' cultural correlation) and, in eq. 3.1, a sufficient bound that
   replaces $r_{BB}$ by the larger phenotypic mate correlation $m$:
   $\beta_M^2 + \beta_F^2 + 2 m \beta_M \beta_F \le 1$. Herzig et al. (2026) treat the same
   co-occurrence of assortment and environment-to-environment transmission;
   their supplementary eq. 28 (section 3.4),
   $\nu \le \sqrt{(1 + r_{E_1E_2})/2}$ for the correlation $\nu$ of the
   child's environment with each parent's, is the symmetric case
   $b_m = b_f = b$: there $\nu = b(1 + r_{CC})$ and the bound is
   $2b^2(1 + r_{CC}) \le 1$. With every
   coefficient at 0 the draw is today's, so the same seed gives the same
   pedigree. The fresh draw always tops up what transmission does not
   supply: $b_m = 1$ copies the mother's $C$ exactly only when $C$ is
   constant across generations.
4. **Truth.** Analyze records realized per-relationship
   $\mathrm{Cov}(A)$, $\mathrm{Cov}(C)$ and $\mathrm{Cov}(A, C)$ from
   relationship moments ([0022](0022-relationship-moments.md)) of the
   **recorded pedigree, before ascertainment**. This scope was confirmed in
   the 2026-10-14 plan review. Analysis-sample moments, if reported, remain
   separate from this truth because phenotype-based selection changes
   component covariances even without transmission. Recorded-pedigree
   moments cover any mating model and lineage-mixed types (GP, Av, 1C and
   beyond, which pedigree-graph does not split by lineage).
   `simace.core.relationships` keeps an analytic random-mating shared-C
   fraction for single-path types: MZ and FS 1, MO $b_m$, FO $b_f$, MHS
   $1 - b_f^2$, PHS $b_f^2$. It returns no value for lineage-mixed types.
   The half-sib fractions sum to the full-sib one, so
   $r_{MHS} + r_{PHS} = r_{FS}$ under random mating with any coefficients.
   Full, maternal half- and paternal half-sib correlations therefore give
   two independent equations in $A$, $C$ and $b_f$, and cannot separate
   $C$ from $b_f$. An estimator that pools relationship types needs the
   lineage-split parent–offspring pairs (MO $= \tfrac12 A + b_m C$, FO
   $= \tfrac12 A + b_f C$, over $V_P$) to identify the coefficients.
5. **Assortative mating** may be combined with transmission. Mates are then
   correlated in $C$, and offspring acquire an $A$–$C$ covariance that the
   Bulmer recursion (`simace/simulation/am_equilibrium.py`) and the
   `rho_w` cross-trait term (`simace/simulation/assortment.py`) leave out.
   After one generation of assortment with mate liability correlation $r$
   that covariance is $\tfrac12 (b_m + b_f) r A C / V_P$ (a toy simulation
   with $b_f = 0$ gave 0.00756 against 0.0075). It is a first-generation
   value: transmission carries the covariance forward, so it grows over
   generations to a larger equilibrium. In correlation form the equilibrium
   satisfies $a = (a + r\varphi\gamma)(b_f + b_m)/2$, that is
   $a = \lambda r \varphi \gamma$ with $\lambda = (b_f + b_m)/(2 - b_f - b_m)$,
   where $\gamma$ and $\varphi$ are the correlations of liability with $A$
   and with $C$; both depend on $a$, so $a$ solves a quadratic (Rao, Morton
   and Cloninger 1979, p. 178 and eq. 1, with their $f_F, f_M$ as our
   $b_f, b_m$ and no social homogamy; Cloninger, Rice and Reich 1979,
   eqs. 18.1 and 19, write it $w = \lambda m \gamma \varphi$). Nothing in simACE uses the
   equilibrium value, because the truth block records the realized
   $\mathrm{Cov}(A, C)$ terms.
   When both are active the AM-equilibrium validation is informational, not
   scored. Realized liability variance is $\mathrm{Var}(A + C + E)$,
   computed directly, not the sum of component variances, and is the
   denominator of the reported liability-scale $h^2$.
6. **Old replicates.** A `run.yaml` recorded before a configuration key
   existed compares as that key's declared legacy value, so the new keys
   (legacy value 0) do not make finished replicates stale.

## Considered options

- **Co-residence environment $H$** shared by everyone living together,
  parents included. Rejected for now: it needs an adult environment term,
  and spouses would share it, which collides with assortative mating. It may
  return as a misspecification test.
- **Cultural transmission from parental phenotype.** Rejected: the
  offspring's environment would depend on the parent's $A$, and the
  resulting gene–environment covariance makes the true $h^2$ and $c^2$
  ambiguous. Cloninger, Rice and Reich (1979, p. 196) reject the same path
  (parent's phenotype to child's phenotype or environment) for their
  cultural model: it implies that foster parents' genotypes shape the
  children they rear and that non-transmissible environments of parents
  and offspring correlate, and it tends to inflate $h^2$ and $e^2$ at the
  expense of the cultural variance. It is nonetheless the usual choice in
  extended twin-family models: Maes et al. (2006, 2007) fit "phenotypic
  cultural transmission from the parent's phenotype to the offspring's
  shared environment" with phenotypic assortment, so their estimates are
  not a direct check of this model. Our model is Cloninger's $B \to B$
  transmission and Rao, Morton and Cloninger's transmitted indexed
  environment ($C_F, C_M \to C$).
- **Additive variance** (fresh draw keeps the configured variance and
  transmission adds to it). Rejected: $\mathrm{Var}(C)$ and liability
  variance drift upward across generations, and every consumer that assumes
  $A + C + E$ (phenotype calibration, `rho_w`, the AM equilibrium) would
  need the realized variance.
- **Analytic rescale** ($\mathrm{Var}(u) = (1 - b_m^2 - b_f^2) C$).
  Rejected: wrong under assortative mating with $b_f > 0$. This is
  Cloninger, Rice and Reich's (1979) eq. 5, under which the cultural
  variance at equilibrium grows to
  $V_{B0}(1 - \beta_M^2 - \beta_F^2)/(1 - (\beta_M^2 + \beta_F^2 + 2\beta_M\beta_F r_{BB}))$
  (eq. 7) instead of staying at the configured total.
- **Per-generation coefficients.** Deferred: they make the truth
  generation-specific, the same problem as generation-dependent $C$ and $E$
  variance.

## Consequences

- The household-by-mother rule still assigns the fresh draw. With
  $b_f > 0$, householdmates with different fathers no longer have identical
  $C$.
- Expected-correlation lines in plots: with every coefficient 0, the
  analytic line for any mating. Otherwise the analytic line where it applies
  (random mating, scalar $C$ and $E$, single-path relationship type) and the
  recorded-pedigree truth otherwise, averaged over replicates on
  scenario-level plots. The analytic value is divided by
  $V_P = A + C + E$ in every case, so it is continuous as a coefficient
  leaves 0; before this ADR the line was the undivided
  $2\phi A + C_{shared} C$, which differs wherever $A + C + E \ne 1$.
  Per-generation $C$ or $E$ has no single $V_P$ and keeps the undivided
  line. The truth pass runs only when transmission is on,
  so scenarios without it pay nothing.
- Liability-scale estimators that assume relatives in different households
  share no $C$ (Falconer from parent–offspring, PA-FGRS, household-kernel
  REML/PCGC/tetraher) become misspecified under transmission. That is the
  intended stressor. fitACE_epimight's parent-conditional CIF truth assumes
  no transmission and is withheld when any coefficient is nonzero.
- fitACE builds its $C$ matrix from `household_id` alone, so its ACE
  estimates on transmission scenarios are biased by design and are not a
  validation target. Its observed-binary plot draws the analytic line only
  where it is exact and otherwise none, because fitACE has no recorded truth.
- The direct liability variance of decision 5 applies to every scenario, not
  only transmission ones: the reported `h2_liability` and the
  `total_variance_trait{t}` observed value move on the next Analyze of any
  replicate. Without transmission the shift is the sample covariance of the
  components: on one $N = 3000$ replicate, with and without assortative
  mating, the total variance moved by 0.005 to 0.022 and `h2_liability` by
  0.003 to 0.008, and no check changed outcome against the previous code.
- The sampling tolerance is 12 standard errors (`CLIP_SE_MULTIPLIER` in
  `simace/simulation/simulate.py`). With constant $C$ and $b_m = 1$ there is
  no structural slack, every clip leaves the realized $\mathrm{Var}(C)$ a
  little above target, and the next generation copies it, so the excess
  accumulates over generations: the largest ratio at $N = 1000$ was 5.96
  over 1000 seeds at 8 generations, 7.23 at 20 and 8.71 at 50. The upward
  drift in $\mathrm{Var}(C)$ is of the order of a standard error times the
  square root of the generation count. The tolerance is wide (about 20% of
  the transmitted variance at $N = 20\,000$, 90% at $N = 1000$), so a
  structural excess must not rely on it: the separate cross-term test
  catches assortative mating once the mates' $C$ covariance is past 4
  standard errors, which raised in none of 300 random-mating replicates at
  $b_m = b_f = \sqrt{0.5}$. Below that the excess is indistinguishable from
  sampling; it is clipped and recorded in the report.
- `run_simulation` accepts scalar $C$ only, so the structural check is on
  one constant $\Sigma$. Per-generation $C$ would need it per generation,
  with each generation's parents at the previous generation's $\Sigma$.
- The truth pass runs once per replicate in Analyze phase 1, and only when a
  coefficient is nonzero. It reuses the pedigree graph validation builds.

## References

- Cloninger, C. R., Rice, J., and Reich, T. (1979). Multifactorial
  inheritance with cultural transmission and assortative mating. II. A
  general model of combined polygenic and cultural inheritance. *American
  Journal of Human Genetics* 31(2), 176–198. PMID 453202, PMC1685756

- Rao, D. C., Morton, N. E., and Cloninger, C. R. (1979). Path analysis
  under generalized assortative mating. I. Theory. *Genetical Research*
  33(2), 175–188. doi:10.1017/S0016672300018310
- Herzig, A. F., Noûs, C., Saint Pierre, A., and Perdry, H. (2026). Impact
  of co-occurrent assortative mating and vertical cultural transmission on
  measures of genetic associations. *Theoretical Population Biology* 170,
  26–35. doi:10.1016/j.tpb.2026.06.003 Supplementary material
  (mmc1), eq. 28.
- Maes, H. H., Neale, M. C., Kendler, K. S., Martin, N. G., Heath, A. C.,
  and Eaves, L. J. (2006). Genetic and cultural transmission of smoking
  initiation: an extended twin kinship model. *Behavior Genetics* 36(6),
  795–808. doi:10.1007/s10519-006-9085-4
- Maes, H. H., Silberg, J. L., Neale, M. C., and Eaves, L. J. (2007).
  Genetic and cultural transmission of antisocial behavior: an extended
  twin parent model. *Twin Research and Human Genetics* 10(1), 136–150.
  doi:10.1375/twin.10.1.136
