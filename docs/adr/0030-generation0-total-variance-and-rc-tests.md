# ADR 0030: Total variance and cross-trait rC score recorded generation 0 with a household-cluster SE

## Status

Accepted 2026-10-20 for issue #68, decided in a design interview and revised
the same day after two reviews; plan in `plans/issue-68-generation0-tests.md`.
Both acceptance folders pass (Consequences).
Amends [ADR 0028](0028-analyze-validate-boundary.md) §9 for
`statistical.total_variance_trait{t}` and `statistical.cross_trait_rC`: they
become calibrated tests on recorded generation 0, scored as below.

## Context

Analyze checks `total_variance_trait{t}` (|Var(A+C+E) − 1| < 0.15) and
`cross_trait_rC` (|r(C1, C2) − rC| < 0.15; NaN passes when rC = 0) on the
rows it calls founders (`mother == −1`). At N = 1,000 the tolerance is about
3 effective SE (ADR 0028 Context), so correct reps fail.

Those rows are **recorded generation 0**, not founders in the glossary's
sense: the offspring of iteration `burnin = G_sim − G_ped`
(`_fill_pedigree_slice`, `simace/simulation/simulate.py:1586`). Founders are
iteration 0's draws, upstream of the recorded pedigree when burn-in > 0.
Recorded generation 0 keeps its households (one per mother,
`simulate.py:875`): householdmates share C and maternal sibs share their
mother's A contribution. Its parents are unrecorded, so its `mother` and
`father` are −1 (`simulate.py:1270-1272`). Iteration `i` draws C and E with
`sd_*_per_gen[i]` (`simulate.py:1566`), so recorded generation 0 has the C
and E of index `burnin`.

Without C transmission each household gets one (C1, C2) draw at
correlation rC (`simulate.py:1159`). With it, the fresh draw's covariance is
the target minus the realized transmitted part (`_transmitted_c`,
`simulate.py:1080-1082`), unless an eigenvalue is clipped; clips are
recorded in `report.yaml` as `c_transmission_clips`.

A study on 1,500 existing reps (`inheritance_calibration` at N = 1,000;
`onset_tau` and `se_calibration` at N = 20,000; 100 reps per scenario;
`plans/founder_study.py`) gave the SD of z per scenario:

| quantity | iid SE | household-cluster SE |
|---|---|---|
| Var(A+C+E), no assortative mating | 0.97–1.36 | 0.91–1.16 |
| Var(A) alone | 1.15–1.40 | 0.96–1.18 |
| Var(C) alone | 1.62–1.98 | 0.86–1.10 |
| r(C1, C2), rC = 0 | one row per household: 0.89–1.00 | all rows: 0.96–1.08 |

Under assortative mating the total variance's mean z was +1.10 and +0.89
(`inh_am_both`) and +8.52 (`onset_tau_pergen_am` trait 1).

## Decision

1. **Total variance.** Per trait, the ddof-0 variance m₂ of x = A_t + C_t +
   E_t over recorded generation 0, against V₀ = A_t + C_t + E_t[burnin] (A
   and C are scalar, E may be per generation). The SE is a sandwich clustered
   on `household_id`: ψ_i = (x_i − x̄)² − m₂, SE_obs = √(Σ_h (Σ_{i∈h} ψ_i)²) / n.
   The test scores the relative error, z = (m₂/V₀ − 1) / (SE_obs/m₂), i.e.
   the cell's SE is SE_obs·V₀/m₂.
2. **Cross-trait rC.** The Pearson r(C1, C2) over every row of recorded
   generation 0, against rC, with the household-cluster delta-method SE:
   ψ_i = z1_i·z2_i − r/2·(z1_i² + z2_i²) on the standardized components,
   SE = √(Σ_h (Σ_{i∈h} ψ_i)²) / n.
3. **Two tests** registered in `ON_DEMAND_CHECKS` with
   `ZCalibration(rep_rate=0.05)` and 20 minimum reps, on the shared
   calibrated-tests atlas page: `total_variance` (cells `trait1_total`,
   `trait2_total`) and `cross_trait_rC` (cell `rC`), in
   `simace/analysis/validate/generation0.py`.
4. **Applicability.** A test's section is `not_applicable`, leaving the
   scenario's verdict, when:
   - for `total_variance` only: assortative mating is active on either trait
     in any mating iteration up to `burnin` (`assort1`, `assort2`, or an
     `assort_matrix` diagonal, resolved per iteration). It inflates Var(A),
     and rA carries one trait's assortment into the other's. Assortment
     starting after `burnin` does not touch recorded generation 0. Under
     `mating_model: wright_fisher` the assortment parameters are ignored at
     runtime (`simulate.py:1401-1406`), so the test applies. rC keeps
     scoring: without transmission each household's (C1, C2) is a fresh
     draw at rC that does not depend on the parents (`simulate.py:1155-1161`),
     so assortment cannot shift it (a review finding, after the interview
     had dropped both tests).
   - C transmission is configured on either trait (any `Ctm_*` ≠ 0). The
     fresh draw is sized against the realized transmitted part, which
     under-disperses rC's z, and generation-0 clips are common at strong
     transmission. Total variance is not under-dispersed there, but the
     test leaves too: one rule for both, no rule reads
     `c_transmission_clips`, and only `inh_ctm` among the studied scenarios
     is lost.
   - for `cross_trait_rC` only: C1 or C2 configured 0, or |rC| = 1.
   A cell is `not_applicable` when V₀ = 0. A cell is information, and counts
   against coverage, with fewer than 100 households, a non-finite estimate,
   or an SE that is not positive; an observed zero variance with V₀ > 0 is
   information, not `not_applicable`.
5. **Analyze** reports both checks as information, with their values. Its
   `c{t}_inheritance` now groups recorded generation 0 by `household_id` when
   the trait's paternal coefficient b_f is 0, so a broken household share of
   C in generation 0 still fails a quality check. With b_f ≠ 0 generation 0
   stays out: its fathers are unrecorded, so the mating-pair grouping cannot
   apply.

## Evidence

Reruns with `run_simulation` (`plans/verify68.py`; G_ped 6, G_sim 8,
`mating_lambda` 0.5, `p_mztwin` 0.02 unless stated):

- Assortment on trait 1 only (assort1 0.6, assort2 0, rA 0.9, N 20,000,
  30 reps): trait 2's Var is 1.376, z 19.9 to 22.9.
- b_m 0.9, rC 0.5, N 4,000, 300 reps, none clipped: rC's SD z is 0.634 (all
  rows) and 0.783 (one row per household); total variance's SD z is 1.087 /
  0.958. At b_m 1 and N 1,000, 91 of 100 reps clip in generation 0 at
  rC 0.5 and 97 of 100 at rC 0.
- Permuting C2 over generation 0's non-first householdmates (`inh_rc`, 400
  reps) gives r 0.205 and z −9.6 over all rows; one row per household is
  blind to it (mean z −0.001).
- `inh_rc` (rC 0.5, 400 reps): rC's mean z 0.034, SD z 1.020; at
  `mating_lambda` 3 (200 reps) 0.115 and 1.043.

The total variance's SE was chosen on the non-assortative, no-transmission
reps of `inheritance_calibration`, `onset_tau` and `se_calibration` plus the
simulated `inh_rc` reps (`plans/se68.py`):

| N | SE | cells | mean z | SD z |
|---|---|---|---|---|
| 1,000 | SE_obs | 1,600 | −0.118 | 1.064 |
| 1,000 | ψ centred at V₀ | 1,600 | −0.116 | 1.056 |
| 1,000 | SE_obs·V₀/m₂ | 1,600 | −0.058 | 1.055 |
| 20,000 | SE_obs | 2,000 | −0.064 | 1.002 |
| 20,000 | SE_obs·V₀/m₂ | 2,000 | −0.053 | 1.000 |

SE_obs grows with m₂, so the studentized z skews negative at N = 1,000; the
relative scale halves that shift without raising SD z. It costs per-rep
power at N = 1,000 (a total wrong by +0.2 crosses |z| 3 in 56% of reps
against 73%), while every mean-z test still catches it (mean z −3.15). A
residual mean z near −0.06 remains under every SE at both N; its cause is
not established.

## Rejected alternatives

- **SE_obs or an SE centred at V₀**: the studentization skew above.
- **One row per household for rC**: blind to a broken within-household
  share of C.
- **An iid SE**: SD z up to 1.98 for Var(C).
- **Conditioning on `c_transmission_clips`**: easy to misread, and the
  unclipped transmission case is still under-dispersed.
- **Skipping only the assorting trait**: rA carries the assortment into the
  other trait.

## Consequences

- Known blind spot: paternal half-sibs in different households share a
  father who is an unrecorded burn-in row, a dependence the household
  cluster does not see. Total-variance SD z reaches 1.162 in the study
  (`onset_tau_gompertz_none` trait 2) and 1.147 at `inh_rc` with
  `mating_lambda` 3.
- The target ignores the finite-sample bias of s² from within-household
  covariance, O(C·Σ_h w_h²), about 0.0006 at N = 1,000 against an SE near
  0.05; an exact correction needs generation 0's unrecorded parents.
- Scenarios with any C transmission no longer test total variance or rC,
  and scenarios with assortative mating up to burn-in no longer test total
  variance; `truth` still reports both.
- Acceptance: `inh_rc` (seed 67800, rC 0.5, C 0.3/0.3, no transmission) joins
  `config/inheritance_calibration.yaml`. Both folders passed on 2026-10-20
  with coverage 1.00 in every cell of the two tests:
  - `inheritance_calibration`: 163 tests at α 3.07e-04 each over 500 reps.
    Total variance applies in `inh_base`, `inh_lambda3` and `inh_rc` and
    leaves `inh_ctm` (transmission) and `inh_am_both` (assortment); rC
    applies everywhere but `inh_ctm`, at rC 0 except in `inh_rc` (0.5). Per
    scenario, total variance mean z −0.112 to +0.101 and SD z 0.826 to
    1.040, rC mean z −0.187 to +0.122 and SD z 0.960 to 1.043; reps over the
    per-rep bound 3–8 of 14 allowed. Pooled: total variance over 300 reps,
    mean z −0.053 / +0.009, SD z 0.969 / 1.001; rC over 400 reps, −0.021,
    1.001.
  - `onset_tau`: 303 tests at α 1.65e-04 each over 500 reps. Total variance
    applies in four scenarios and leaves `onset_tau_pergen_am` (assortment);
    rC leaves everywhere (C1 or C2 configured 0). Per scenario, mean z
    −0.192 to +0.166 and SD z 0.918 to 1.156; reps over the bound 4–8 of 14.
    Pooled over 400 reps: mean z −0.049 / +0.003, SD z 1.002 / 1.043.
