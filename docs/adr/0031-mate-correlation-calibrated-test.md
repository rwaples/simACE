# ADR 0031: Per-generation mate correlations as a calibrated test, with an SE that follows the pairing mode

## Status

Proposed 2026-10-21 for issue #69, decided in a design interview and revised
the same day after a review; plan in
`plans/issue-69-mate-correlation-test.md`. Acceptance pending (Consequences).
Amends [ADR 0026](0026-mate-correlation-checks-by-generation.md) §1 and §4:
induced cells are no longer scored, and the per-generation checks leave
Analyze for `simace validate`. Amends
[ADR 0028](0028-analyze-validate-boundary.md) §9 for
`assortative_mating.mate_corr_*`, and brings the multiplicity rule that ADR
0026 deferred.

## Context

Analyze compares each mate correlation with its target within
max(0.1, 3·SE), SE = (1 − ρ²)/√(n − 1), pooled or split per parents'
generation as `{name}_gen{g}` (ADR 0026). Split, a stratum under about 750
Mating Pairs makes 3·SE bind, so each check fails about 0.27% of correct reps,
multiplied by generations × cells with no multiplicity rule.

The simulator pairs parents in one of three ways per mating iteration
(`_mating_standard`, `simace/simulation/simulate.py`):

- **Both traits assort**: a 4-variate Gaussian copula, a k-d match, then
  Metropolis swaps until all four slot-level Pearson correlations are within
  5e-4 of the target matrix (`_assortative_pair_partners`). The realized
  correlation over all couples is the target, up to that tolerance and the
  dedup swaps that follow.
- **One trait assorts**: a bivariate Gaussian copula on ranks, without
  refinement; the realized correlation is random around the target.
- **Neither**: a random shuffle (`pair_partners`).

`allocate_offspring` then spreads the N children multinomially over the M
couples. Couples with no child never reach the recorded pedigree, about 20%
of them at `mating_lambda` 0.5 and 52% at 3.

Under single-trait assortment the other trait's mate correlation, and both
cross-trait ones, are **induced** through the within-person correlation ρ_w.
`expected_mate_corr_matrix` gives them from the configured A, C and E, which
assortment moves (burn-in included).

## Decision

1. **Test.** `mate_correlation` in `simace/analysis/validate/mate_correlation.py`,
   registered in `ON_DEMAND_CHECKS` with `ZCalibration(rep_rate=0.05)` and 20
   minimum reps. Per parents' generation g of each recorded mating iteration
   (`recorded_matings`), it scores the Pearson correlation over the
   generation's Mating Pairs against that iteration's nominal target. Cells:
   `g{g}_liability1`, `g{g}_liability2`, and `g{g}_cross_12`,
   `g{g}_cross_21` on iterations where both traits assort.
2. **SE.** The iid delta-method sandwich over the pairs' standardized
   liabilities u (mother) and v (father): ψ = u·v − r/2·(u² + v²),
   SE = √(Σψ²)/n. On an iteration where both traits assort it is multiplied
   by √(1 − p̂), p̂ = n/M̂, where M̂ solves n = M̂·(1 − (1 − 1/M̂)^{N_c}) and N_c
   is the number of children (generation g+1 rows). Given n couples with a
   child, multinomial allocation makes every n-subset of the M couples
   equally likely, so this is a without-replacement finite-population
   correction around a population correlation pinned to the target. At
   n = N_c there is no finite root and the factor is 1.
3. **Not applicable.** The section, when no recorded iteration of the
   standard mating model assorts on either trait (Wright-Fisher has zero
   targets). A cell, when it is induced; when its trait's A, C and E are all
   configured 0; when it has 1 to 249 Mating Pairs (`MIN_PAIRS = 250`); or,
   on an iteration where both traits assort, when the Metropolis tolerance
   exceeds a quarter of the SE (about N > 300,000), since the stopping error
   does not shrink with N. The tolerance is one named constant that the
   simulator and the test share.
4. **Information** (counts against coverage): zero Mating Pairs where the
   simulator recorded matings, a constant side the configuration does not
   explain, a non-finite r, or no positive SE.
5. **Multiplicity.** The per-rep Bonferroni bound over the scored cells
   (`summarize`, ADR 0028) is the rule ADR 0026 deferred.
6. **Cells and pages.** `cell_names` lists generations 0 to 11; later
   generations are scored and judged but get no gather column. The test has
   its own atlas page, drawing every generation the reps report.
7. **Analyze** reports every mate correlation as information, with no
   per-generation entries. `report.yaml` keeps
   `truth.recorded_pedigree.assortative_mating`'s four pooled values; the
   per-generation and cross-trait values are in each rep's
   `validation.yaml`. `mate_corr_A{t}` stays information.
8. **Target is nominal.** The single-trait copula's realized correlation
   sits below its target (Evidence). The test reports it and does not
   correct it; a separate simulator issue tracks it.

## Evidence

`run_simulation` with G_ped 6, G_sim 8, `mating_lambda` 0.5, p_mztwin 0.02,
A 0.5/0.4, C 0.2/0.2, rA 0.3 unless stated (`plans/se69.py`). SD of z per
design:

| design | cells | naive | iid | two-way cluster | iid·√(1−p̂) |
|---|---|---|---|---|---|
| both 0.3/0.3, N 1,000 | 8,000 | 0.45–0.46 | 0.45–0.47 | 0.44–0.46 | **1.02–1.05** |
| both, λ 3 | 8,000 | 0.70–0.73 | 0.71–0.73 | 0.65–0.71 | **0.99–1.02** |
| both, N 5,000 | 8,000 | 0.45–0.46 | 0.45–0.46 | 0.43–0.46 | **0.99–1.03** |
| `assort_matrix` | 8,000 | 0.44–0.46 | 0.44–0.46 | 0.44–0.45 | **0.99–1.03** |
| single 0.4 | 4,000 | 0.98–1.00 | **1.00–1.01** | 0.99–1.00 | 2.2 |
| single 0.4, λ 3 | 4,000 | 0.99–1.02 | **1.00–1.02** | 0.95–1.01 | 1.4 |
| single, assort 0.2–0.8 × λ 0.5/3, N 2,000 | 8,000 | | **0.95–1.03** | 0.80–1.02 | |
| random, λ 0.5 / 3 | 8,000 | | **0.99–1.03** | 1.01–1.03 | |

With the bold rule every design has SD z 1.00–1.03 and P(|z| > 3) 0.15–0.41%,
including a schedule that switches mode between generations (1.009 / 1.015)
and C transmission (b_m 0.5, b_f 0.3; 0.997–1.032). Two-way clustering on
mother and father under-disperses at λ 3; parents with several partners do
not need it.

Stratum size, same rule:

| Mating Pairs per stratum | SD z | P(\|z\| > 3) |
|---|---|---|
| ~97 (N 200) | 1.055–1.070 | 0.62–0.92% |
| ~246 (N 500) | 1.026–1.033 | 0.45–0.52% |
| ~495 (N 1,000) | 1.009–1.028 | 0.35% |

A Fisher-z scale gave 1.050–1.053 at 97 pairs.

**Single-trait shortfall.** At N 2,000 the realized r is 0.0017–0.0029 below
`assort` at λ 0.5 and 0.0017–0.0136 below at λ 3 (assort 0.2–0.8; mean z
−0.04 to −0.19 and −0.06 to −1.17). On the pairing function alone
(`plans/copula69.py`, `plans/copula69_dedup.py`) the copula gives −0.0030 at
M ≈ 640 couples and −0.00014 at M ≈ 12,700, and dedup adds about −0.004 at
λ 3. On `onset_tau_pergen_am` (N 20,000, 60 reps): −0.0007, mean z −0.081.

**Induced cells.** Single-trait assort1 0.6, rA 0.9, A 0.8, E 0.2, N 5,000:
trait 2's mate correlation exceeds its nominal target 0.311 by 0.07 to 0.10
in every generation (mean z +5.4 over 2,000 cells).

## Rejected alternatives

- **The naive or the iid SE everywhere**: SD z 0.45 on both-trait
  iterations.
- **Two-way clustering on mother and father**: SD z 0.80–0.90 at λ 3.
- **A 100-pair floor, or 100 with a fitted small-n factor**: SD z 1.06 at
  97 pairs; the factor has no derivation.
- **Floors as information**: a correct small-N scenario, or a trait
  configured 0, would fail coverage.
- **A realized induced target** r₁·ρ̂_f·ρ̂_m: needs a joint SE over three
  correlations and assumes linear propagation; left for later.
- **Scoring induced cells against the nominal target**: z near +5 in the
  stress design.
- **Fixing the copula first**: changes every single-trait rep's output;
  tracked separately.
- **Adding tol²/3 to the variance on Metropolis iterations**: the stopping
  error's distribution is not uniform on ±tol, and the dedup error is not
  bounded.
- **Keeping Analyze's pooled checks as gross checks**: at about 200 pairs
  3·SE binds and they fail 0.27% of correct reps; the induced pooled cell is
  biased toward the 0.1 floor.
- **Scoring random-mating scenarios at 0**: about 30 more tests per scenario
  family, against a shuffle.
- **One cell per type, Stouffer over generations**: dilutes a bias confined
  to one iteration by √G.
- **Dynamic `cell_names`**: changes the registry, gather and the shared
  page; the cap covers G_ped ≤ 13, and later generations are still judged
  and drawn.

## Consequences

- Duplicate couples left by dedup (0 of 611 slots at λ 0.5, 5 of 1,498 at
  λ 3, 489 of 14,342 at λ 30 in a review probe) make the occupancy equation
  approximate at high λ. `config/` uses λ ≤ 3; a λ 30 probe still had SD z
  near 1.
- Single-trait scenarios with strong assortment or λ 3 are expected to fail
  the mean-z test until the copula is fixed. That failure is a real defect,
  not a false positive.
- Induced cells are not tested anywhere.
- Large-N scenarios (about N > 300,000) lose their both-trait cells.
- `report.yaml` no longer holds per-generation mate correlations, and
  `report_summary.tsv` gains 51 columns.
- Acceptance: `inh_am_mixed` (seed 68000; assort1 0.3, assort2 {0: 0, 5: 0.3},
  rA 0.3: copula in parents' generations 0–1, Metropolis in 2–4) and
  `inh_am_matrix` (seed 68200; `assort_matrix` [[0.3, 0.05], [0.05, 0.15]])
  join `config/inheritance_calibration.yaml` at 100 reps, N 1,000. Results
  pending.
