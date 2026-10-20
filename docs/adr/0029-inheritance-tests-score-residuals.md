# ADR 0029: The sibling inheritance tests score residuals, not relative correlations

## Status

Proposed 2026-10-20 for issue #67, decided in a design interview and revised
the same day after a review. Amends [ADR 0028](0028-analyze-validate-boundary.md)
§9 for the sibling and half-sib checks: they become calibrated tests, but of
the quantities below, not of the relative correlations they reported, and
adds a cell status to ADR 0028 §2's coverage requirement. Acceptance is
pending (Consequences).

## Context

ADR 0028 moves `heritability.dz_sibling_A{t}_correlation`,
`half_sibs.half_sib_A{t}_correlation` and `half_sibs.half_sib_shared_C{t}`
out of Analyze's quality checks into `simace validate`, "each with a
cluster-robust SE". The candidate SE is pg-phenotype's Relative-Pair
Correlation sandwich (its ADR 0013), which clusters two pairs when they
share a member or their most recent common ancestor (MRCA).

That SE was measured on the 600 `se_calibration` reps (6 scenarios × 100,
N = 20,000; `mating_lambda` 0.5 and 3; C2 0.2–0.3; no C transmission or
assortative mating), with z per cell over both traits:

- **r(A) against the nominal 0.5 (FS) and 0.25 (MHS ∪ PHS).** SD(r)/SE is
  1.08–1.36 for FS and 0.79–1.22 for half-sibs. The parents' realized A
  variance differs from the configured one in each generation, and that
  shift moves every sibling pair of the generation together; no family
  cluster sees it. At N = 1,000 it is also a bias: in `small_test` the
  realized-midparent expectation of the trait-1 FS correlation is
  0.471–0.479 in all three reps checked, so a 0.5 target fails mean z
  across reps of a correct simulation.
- **r(A) against r\*, its expectation given the parents' A.** r\* =
  cov(mid₁, mid₂) / √((V_mid₁ + A/2)(V_mid₂ + A/2)) over the pairs absorbs
  the shift and assortative mating, but SD(r − r\*)/SE is 0.58–0.81: the
  MRCA clusters count variation between families that r\* conditions away.
  A matching SE would cluster by shared member alone, which pg-phenotype
  does not offer and simACE would have to rebuild from FS/MHS/PHS
  membership.
- **r(C) over PHS against 0** (b_f = 0): SD(r)/SE 0.98–1.08.

The simulator's inheritance has residuals with known targets. A child's A
is the midparent plus a Mendelian draw of variance A/2, the configured
founder variance, whatever the parents' realized variance
(`simulate.py:170-191`, `:1146-1152`); MZ co-twins copy one draw
(`:1173-1176`). A child's C is b_m·C_mother + b_f·C_father plus a
household draw, one per household and independent between households
(`simulate.py:1024`, `:1155-1164`).

## Decision

1. **Two calibrated tests on the recorded pedigree**, registered in
   `ON_DEMAND_CHECKS` with `ZCalibration(rep_rate=0.05)` and 20 minimum reps:
   - `mendelian_residual`: per trait with A_t > 0, the Pearson correlation
     of the **Mendelian residual** A − midparent over FS, MHS and PHS pairs
     (cells `trait{t}_FS`, `trait{t}_MHS`, `trait{t}_PHS`, expected 0), and
     its variance over the non-founders with one member of each MZ pair
     (cell `trait{t}_variance`, expected A_t/2).
   - `household_draw`: per trait with C_t > 0, the Pearson correlation of
     the **household draw** C − b_m·C_mother − b_f·C_father over PHS pairs
     (cell `trait{t}_PHS`, expected 0).

   A residual is defined where both parents are pedigree rows, found by id
   lookup. The targets hold for every mating model, assortment and
   transmission, so no cell is skipped as "not asserted". They are exact
   for the simulator's float64 draws; residuals rebuilt from the float32
   components in `pedigree.parquet` carry rounding about 10⁻⁷ of an SD.
2. **SEs.** The correlation cells take pg-phenotype's MRCA-or-shared-member
   sandwich (`relative_pairs.correlation(..., se=True)`, pg-phenotype ≥
   0.5). Mendelian residuals are independent between individuals other
   than MZ co-twins, who share a mother and so a cluster; a coarser
   cluster than needed stays valid. The variance cell takes the asymptotic
   iid SE √((m₄ − s⁴)/n).
3. **One dependence the C SE does not see.** Two PHS pairs whose
   households are the same two, linked through two different fathers
   (two mothers who each have children by the same two men), repeat one
   product u_H1·u_H2 but share no member and no MRCA. Each rep records, in
   the cell, the share of PHS pairs in such household pairs
   (`linked_household_share`). It was at most 0.06% over the 770 distinct
   pedigrees below; the test does not correct for it.
4. **Cell status.** A trait whose A (or C) is configured 0 has no cell in
   that test. A cell is `not_applicable`, with the reason, when its residual
   has zero variance to within rounding, as when the household draw is
   clipped away under full transmission; it is reported but left out of
   the coverage count. A cell with fewer than 100 pairs (residuals for the
   variance), or with no SE from pg-phenotype (its `se_reason`), is
   information with the reason and counts against coverage.
5. **Inputs.** The per-rep step reads the scenario parameters from the
   rep's `report.yaml` (`inputs.parameters`), as ADR 0028 §7 allows:
   `RepInputs` gains the report path and the per-rep primitive a
   `--report` flag.
6. **Analyze** drops the three checks' pass/fail in the same change. Their
   values stay where `report.yaml` puts them now: the half-sib A and
   shared-C correlations under `truth`, the full-sib A correlation under
   `estimators`. `quality_checks` no longer lists them, since it lists only
   checks with a pass.

## Evidence

The final cells on the 1,100 reps at N = 20,000 of `se_calibration` (600)
and `onset_tau` (500, one scenario with assort1 0.4), z = (observed −
expected) / SE, from `scripts/inheritance_residual_study.py se_calibration
onset_tau`. `se_calibration` spaces its 100-rep scenarios 10 seeds apart, and
a rep's seed is the scenario seed plus the rep number minus 1
(`simace/cli/stages.py:53`), so scenarios with the same `mating_lambda`
reuse pedigrees and, with random mating and rA = 0, the same standardized
draws. The script counts each pedigree once: 770 distinct pedigrees.

| cell | z per cell | mean z | SD z | \|z\| > 1.96 | SD z per scenario × trait |
|---|---|---|---|---|---|
| FS | 1,540 | 0.002 | 1.010 | 5.1% | 0.85–1.12 |
| MHS | 1,540 | 0.046 | 1.007 | 5.5% | 0.88–1.09 |
| PHS | 1,540 | −0.011 | 1.039 | 5.9% | 0.86–1.17 |
| variance | 1,540 | −0.074 | 0.978 | 4.7% | 0.89–1.05 |
| household draw, PHS | 770 | −0.050 | 0.969 | 4.3% | 0.86–1.04 |

The per-scenario range covers scenario × trait cells with at least 50
distinct reps; 100 reps give about ±0.14 around 1.

The variance cell's mean z, −0.074 (−0.070 under the exact χ² z), is
about 2.9 SE below 0. It is chance in these reps' random streams, not a
bias. Every stored pedigree of the 1,100 reps is bit-identical to an
in-process rerun of the simulator with its parameters. The residuals rebuilt
from the stored float32 values equal the simulator's raw Mendelian draws to
3.1 × 10⁻⁷, so nothing after the draw changes them. 10,000 fresh seeds with
`onset_tau`'s baseline parameters (20,000 rep × trait cells) give mean z
+0.008 ± 0.007.

## Rejected alternatives

- **Nominal targets with the MRCA SE** (the issue as filed): miscalibrated
  SE and a small-N bias, above.
- **r\* with a shared-member SE built in simACE**: the issue's quantities
  exactly, but a second classification of sibling pairs beside
  pedigree-graph's, and its calibration unmeasured.
- **r(C) over PHS against b_f²** where ADR 0025's closed form holds: the
  target assumes the fathers' realized C variance, the same shift as for
  A, is unscored outside random mating with scalar C and E, and was
  measured only at b_f = 0.
- **One test for every cell**: an A failure and a C failure would share
  one reps-over-bound count.
- **MHS ∪ PHS pooled**, as the old check did: a link error on one side is
  diluted by the other.
- **Parameters as stage flags**: a second copy of `inputs.parameters`,
  and a longer flag list for each later test.
- **Clustering PHS pairs by household pair as well**: no measured need
  (decision 3).

## Consequences

- Acceptance: the folder verdict passes on a new
  `config/inheritance_calibration.yaml` (4 scenarios × 100 reps at
  N = 1,000 with disjoint seed ranges: a base, C transmission with
  b_m = b_f = 0.3, assortative mating on both traits, `mating_lambda` 3)
  and on `onset_tau`. `se_calibration` cannot give a folder verdict (its
  scenarios' seed ranges overlap) and stays study evidence. Issues #68 and
  #69 reuse the new folder.
- The relative correlations themselves are reported and never tested.
  Two bugs the old checks could catch are now caught by nothing: pairs
  misclassified as siblings (a classification error keeps the residuals'
  correlation at 0 but changes r(A); pedigree-graph's differential oracle
  covers classification), and a wrong household-draw variance (it keeps
  the draws uncorrelated but changes the transmitted PHS correlation;
  total C variance is checked only on the founders).
- Under C transmission the PHS cell no longer checks ADR 0025's b_f²
  closed form; `truth` still reports r(C).
- ADR 0028 §2's coverage requirement counts `not_applicable` cells out.
