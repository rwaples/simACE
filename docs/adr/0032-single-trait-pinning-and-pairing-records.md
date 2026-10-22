# ADR 0032: Single-trait assortment is pinned by Metropolis, and each mating iteration's pairing is recorded

## Status

Proposed 2026-10-22 for issues #73 and #74; plan in
`plans/issue-73-74-assortative-pinning.md`. Amends
[ADR 0031](0031-mate-correlation-calibrated-test.md) §2, §3 and §8, its
"Fixing the copula first" rejection, and its Consequences line on
single-trait scenarios. Acceptance evidence (the before/after grid and the
acceptance folders) is pending.

## Context

- **#74.** The single-trait path of `_assortative_pair_partners` pairs
  parents by a rank-based bivariate Gaussian copula with no refinement. Its
  slot-level Pearson correlation is random around a value below the target.
  The dedup swaps that follow use a nearby female by slot order, which the
  copula has reordered, so they move it further. A probe at assort 0.4
  (300 draws of the pairing function) gave mean r − target −0.0026 (SD 0.037)
  at N 1,000, λ 0.5; −0.0035 (SD 0.024) at N 1,000, λ 3; and −0.0182
  (SD 0.055) at N 200, λ 3. ADR 0031 measured the shortfall in the test:
  mean z down to −1.17 at λ 3.
- **#73.** On the both-trait path, Metropolis non-convergence was only a log
  warning, so nothing a rep keeps recorded it. Yet `mate_correlation`
  multiplies the SE by √(1 − p̂) on the assumption that the correlation over
  all M couples is pinned. The golden configuration `standard_both_am`
  (`tests/simulation/test_simulate_golden.py`) misses the tolerance in 3 of
  its iterations today.

## Decision

1. **Single-trait refinement.** After the copula and dedup, a 1-D Metropolis
   (`_metropolis_single`) swaps fathers between random pairs of couples. It
   keeps a swap only when the swap brings the active trait's slot-level
   Pearson correlation closer to `assort_t`, and stops within
   `METROPOLIS_TOL` or after 8·M proposals, as the both-trait path does. It
   refuses a swap that would create a duplicate couple. Correlations are
   computed on the couples left after dedup, so the recorded error is the
   final one. Random mating and Wright-Fisher are unchanged: their
   pedigrees are byte-identical for a given seed.
2. **Pairing records.** `run_simulation(mating_log=...)` records one entry
   per mating iteration, burn-in included: `iteration`, `generation`
   (recorded coordinates) and `mode` (`random`, `single`, `both` or
   `wright_fisher`). An assortative entry also records `M`, `proposals`,
   `max_proposals`, the final post-dedup slot-level `errors` (correlation
   minus target, one per pinned correlation), `duplicates_left` and
   `converged` (every error within `METROPOLIS_TOL`). The simulate stage
   writes them to the rep's `mating.yaml`, a declared stage output.
3. **The SE follows pinning.** `mate_correlation` reads `mating.yaml`
   (`simace validate --mating`). On a `single` or `both` iteration, every
   scored cell takes the finite-population factor √(1 − p̂) and the
   `METROPOLIS_TOL` gate. This amends ADR 0031 §2–§3, where these applied
   only where both traits assort. Induced cells stay not applicable: only the
   active trait is pinned.
4. **A missed target is information.** On an iteration whose record says
   `converged: false`, a cell that passes the not-applicable gates (induced,
   configured 0, too few Mating Pairs, the tolerance gate) is information.
   Its reason lists the recorded errors, which are measured after dedup, and
   it counts against coverage. A record with nothing pinned (M < 2, or a
   constant active trait) is not converged.
5. **Consistency.** A record whose mode disagrees with the iteration's
   assortment raises. So does a missing `mating.yaml` in an assortative
   scenario, with a "recompute the rep" message.
6. **Duplicates left stay an approximation.** Allocation samples slots, and
   the test collapses a duplicate couple to one Mating Pair, so the factor
   is approximate when duplicates remain (ADR 0031 Consequences).
   `converged` ignores them. Each scored cell reports `duplicates_left`.
7. **One proposal draw for both kernels.** `_metropolis_full` and
   `_metropolis_single` take their pairs from `_draw_pairs`, a partial
   Fisher-Yates over all M rows. `_metropolis_full` used to shuffle only
   the first `2·(M // 2)` positions among themselves, so with M odd one
   couple never moved (a review finding). This changes both-trait output
   for a given seed too.

## Evidence

`plans/issue-74-grid.py` runs `run_simulation` and `run_mate_correlation`
with the same seeds on `dev` 949203f (before) and on this change (after).
Settings: 400 reps per design, G_ped 6, G_sim 8, A 0.5/0.4, C 0.2/0.2, rA 0.3,
p_mztwin 0.02. The scored cells are the assorting liability cells, 5 parents'
generations per rep.

| design | before mean z | before SD z | after mean z | after SD z | before r − target | after r − target |
|---|---|---|---|---|---|---|
| single 0.2, λ 0.5, N 1,000 | −0.024 | 1.015 | −0.017 | 1.016 | −0.0018 | −0.0005 |
| single 0.2, λ 3, N 1,000 | −0.092 | 1.033 | +0.017 | 1.005 | −0.0038 | +0.0002 |
| single 0.4, λ 0.5, N 1,000 | −0.081 | 0.995 | +0.005 | 1.025 | −0.0044 | −0.0002 |
| single 0.4, λ 3, N 1,000 | −0.177 | 0.995 | −0.004 | 1.019 | −0.0064 | −0.0006 |
| single 0.4, λ 3, N 2,000 | −0.144 | 1.006 | +0.015 | 1.000 | −0.0036 | −0.0000 |
| single 0.8, λ 0.5, N 1,000 | −0.326 | 0.985 | +0.024 | 1.033 | −0.0068 | −0.0001 |
| single 0.8, λ 0.5, N 2,000 | −0.222 | 1.036 | −0.007 | 1.000 | −0.0033 | −0.0002 |
| single 0.8, λ 3, N 1,000 | −1.401 | 0.917 | +0.045 | 1.040 | −0.0253 | −0.0003 |
| single 0.8, λ 3, N 2,000 | −1.226 | 0.964 | +0.039 | 1.025 | −0.0143 | −0.0001 |
| both 0.3/0.3, λ 0.5, N 1,000 | −0.005 | 0.998 | +0.028 | 1.005 | −0.0003 | +0.0003 |
| both 0.3/0.3, λ 3, N 1,000 | −0.007 | 1.017 | −0.004 | 1.018 | −0.0006 | −0.0005 |
| `assort_matrix`, N 1,000 | +0.018 | 1.024 | +0.020 | 1.014 | +0.0001 | +0.0002 |

- **Single-trait, after:** across all 12 single-trait designs (the table
  shows 9), mean z is −0.017 to +0.045 and SD z 0.997 to 1.040, with the
  finite-population factor applied. All 2,000 recorded iterations of each
  design converged (400 reps × 5 parents' generations).
- **Both-trait and `assort_matrix`:** calibrated before and after the
  shared proposal draw (decision 7). Mean z is −0.004 to +0.028 and SD z
  1.005 to 1.018, and none of the 2,000 iterations per design missed. The
  odd-M flaw was too small to see at this size.
- **Residual at assort 0.8, λ 3:** the after mean z is +0.039 to +0.045,
  about 2 SE over 2,000 cells, and r − target is −0.0003 to −0.0001. A
  first version of the kernel never proposed the last row when M was odd.
  At assort 0.8 it gave +0.040 and +0.089 at λ 0.5 (now +0.024 and
  −0.007) and +0.048 and +0.056 at λ 3 (now +0.045 and +0.039). The
  residual is below what the verdict's mean-z test resolves at 100 reps
  (SE 0.1).
- **Duplicates left:** the both-trait λ 3 design has 3,974 cells from
  iterations with duplicates left, with SD z 1.02. The approximation of
  decision 6 holds there.

## Rejected alternatives

- **Dedup before Metropolis on both paths**: its gain, putting dedup's
  drift inside the swaps' reach, is not needed. Both-trait errors after
  dedup stay within tolerance (0 of 2,000 iterations per design missed),
  and the record catches a miss when one happens.
- **Rank-adjusting the copula without refinement**: the correlation stays
  random, the dedup shortfall remains, and the SE keeps no factor.
- **Aligning dedup's female liabilities with the copula's rows**: a
  duplicate's nearest row is then often the same woman's other slot, so the
  swap changes nothing. At N 1,000, λ 3, assort −0.8 it left 2,882
  duplicates over 300 draws, against 8 with slot order.
- **Counting duplicates left as non-convergence**: many correct λ 3
  iterations would become information cells.
- **Non-converged cells scored with the iid SE, or not applicable**: the
  first hides a simulator missing its own specification; the second drops
  it from the verdict.
- **Records in the pedigree's Parquet metadata or `report.yaml`**: the user
  chose a separate file.

## Consequences

- Assortative reps change for a given seed: the single-trait refinement
  and the shared proposal draw (golden digests `standard_single_am`,
  `standard_pergen` and `standard_both_am`). Random-mating and
  Wright-Fisher pedigrees do not change.
- Every rep gains `mating.yaml`, so `simace run` recomputes existing
  complete reps, whose new output is missing (`simace/cli/manifest.py`).
- Large-N single-trait scenarios lose their cells to the tolerance gate,
  as both-trait ones did under ADR 0031.
- Evidence to add before acceptance: per-design realized r − target, mean z,
  SD z, P(|z| > 3), non-convergence rate and duplicates left, before and
  after, over 400 reps per design. Plus `inheritance_calibration` and
  `onset_tau` rerun and validated.
