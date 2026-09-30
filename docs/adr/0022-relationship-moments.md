# ADR 0022: Analyze computes relationship statistics from relationship moments

## Status

Accepted. Decision approved 2026-09-30 in `plans/relationship-moments-v10.md`
(simACE #25, pedigree-graph #28); implemented against pedigree-graph 0.11.

Renumbered from 0020 on 2026-09-30, when it met the standalone-CLI branch,
whose ADR 0020 and 0021 were already numbered. Under results layout 2
(ADR 0021) `pedigree.parquet` is the whole recorded pedigree, so phase 3
reads only the report columns for the stats and reads the A/C/E components
for the plotting sample separately at the end, rather than holding the
whole pedigree's components through the stats as point 5 describes.

## Context

Every pair statistic in Analyze was computed from a pair list:
`PedigreeGraph.relationship_pairs` materialized the row indices of every
`MZ`/`FS`/`MO`/`FO`/`MHS`/`PHS`/`1C` pair of the analysis sample, and
`simace.analysis.stats.correlations` gathered liabilities and affection
flags through those indices for each stratum (overall, last three
generations, same-sex) before running Pearson, phi and the tetrachoric
MLE. At 20M rows the degree-5 pair list is 15.8 GiB (pedigree-graph ADR
0010), and the four tetrachoric functions ran on a `ThreadPoolExecutor` to
hide their per-stratum gathers.

Validate did the same on the recorded pedigree, and capped the full-sib and
half-sib pairs it correlated at 5,000 per category (`_MAX_CORR_PAIRS`) with
a seeded subsample, so `n_pairs` in the report was the cap rather than the
count, every sibling correlation carried sampling error, and the tolerance
`max(4·SE, 0.05)` was evaluated at the capped n.

pedigree-graph 0.11 adds `relationship_moments` (its ADR 0013): one engine
pass reduces every pair of the selected categories to exact integer
accumulators per cell of caller-named per-member factors, with pair counts,
sums, sums of squares and cross sums of caller-supplied value columns, and
derives every float from those integers on access. Grouping and selection
(`select`, `sum`, `merge`) are integer additions, so they are exact and
order-independent.

## Decision

1. **One moments table per graph replaces the pair list.**
   `simace.analysis.stats.moments.relationship_moments_for(df, source)`
   declares the factors and value columns the report reads and returns the
   pedigree-graph result: first-member factors `generation`, `sex`,
   `affected{t}`; second-member factors `sex`, `affected{t}`; value columns
   `liability{t}`; the `first × second` product of each. The dicts are
   built over the trait list, so a third trait is one more entry. simACE
   does no label arithmetic; pedigree-graph maps each factor's distinct
   values to axis levels, so every distinct generation in the frame,
   founders and older generations included, is its own level, and a frame
   without a `generation` column gets one level.
2. **`compute_*` take `moments=`.** `compute_liability_correlations`,
   `compute_affected_correlations`, `compute_tetrachoric`,
   `compute_tetrachoric_by_generation`, `compute_cross_trait_tetrachoric`
   and `compute_tetrachoric_by_sex` take `moments=` instead of `pairs=`;
   the unused `seed=` is gone. Each reduces the table with
   `Stratum`: a selection folded to one cell per category, from which the
   pair count, the 2×2 affection table (`tetrachoric_from_table`, phi) and
   the liability Pearson correlation follow. The report selections are
   unchanged: the last three generations, a generation stratum keyed by the
   first member (the offspring for `MO`/`FO`, the lower receiver row for
   the symmetric codes), same-sex strata reading sex from both members,
   `None` below ten pairs, `0.0` for a constant liability side (what the
   single-pass kernel returned on a zero denominator). There is no
   pairs-based adapter; the `ThreadPoolExecutor` in the stats runner and
   in fitACE's `fitace.ltm.stats` is gone, and fitACE's `--max-degree`
   flag with it (it fed only the pair extraction).
3. **Validate is exact over all pairs.** One moments call on the recorded
   pedigree over `FS`/`MHS`/`PHS` with columns `A{t}`, `C{t}` and
   `P{t} = A + C + E` per trait, one cell per category. The pooled
   `MHS ∪ PHS` cell is `select(...).sum("category")`, an exact integer
   fold. `_subsample_pairs`, `_MAX_CORR_PAIRS` and the validation RNGs are
   deleted; `n_pairs` is the true pair count and the tolerance formulas
   and floors are evaluated with it. `safe_corrcoef`'s rule is kept from
   the moments: NaN when either side's `sqrt(m2 / n)` is below the
   zero-variance threshold. MZ correlations still come from the `twin`
   column and the parent-offspring regressions from the parent columns.
4. **Sibling household counts are O(N).** `n_offspring_with_sibs` and
   `n_offspring_with_maternal_half_sib` come from household (mother) group
   sizes: an offspring with a known mother has a sibling when its household
   holds another offspring besides itself and its own co-twin, and a
   maternal half-sib when the household holds an offspring with a different
   father, where an unknown father differs from every father including
   another unknown one (the engine classifies two offspring of one mother
   with unknown fathers as `MHS`). Offspring with an unknown mother take
   part in neither count. With pedigree-graph #29 (co-twins join sibling
   groups) these equal the distinct members of the `FS ∪ MHS` and `MHS`
   pair sets.
5. **Two graphs stay.** Validate builds the recorded-pedigree graph and
   stats the analysis-pedigree graph, as ADR 0008's non-goal says: the two
   frames differ (6,000 recorded against 5,312 analysis rows on
   `small_test`), so one graph per Analyze job would need a view for one
   of them and buy nothing. Each build is timed in the log. Phase 3 reads
   `pedigree.parquet` once, projecting the report columns for the stats
   and the A/C/E components for the plotting sample from the same frame.

## Consequences

- Descriptive statistics keep parity with the pair-list implementation on
  the same engine: relationship counts, `n_pairs` and 2×2 tables are
  identical; Pearson, phi, tetrachoric r/SE and `liability_r` agree within
  1e-10 (measured maximum 8.8e-12 on `bench1M`, ~8e-14 on the rest). The
  plot payload is byte-identical.
- Validate statistics change wherever the 5,000-pair cap used to bind: on
  `bench1M` (625,802 full-sib pairs) and `ascertainment_case5x_50k`
  (704,428) the sibling correlations move by up to 0.03 to their all-pairs
  values; on `small_test` (about 4,000 pairs, below the cap) they change
  only in the last bits. No scored outcome changed on the five acceptance
  runs.
- `bench1M` Analyze: the analysis-sample moments pass takes 1.5 s and the
  tetrachoric block 0.1 s, against 5.8 s for the pair extraction and 1.5 s
  for the tetrachoric block on five threads in the step-1 pair-list run
  (same code as commit P, pedigree-graph 0.10). Each graph build is 0.2 s.
- Tests that compared the stats Pearson path with `np.corrcoef` bit for bit
  now allow 1e-12 (43-bit quantization); pair-swap symmetry and
  row-permutation invariance stay exact. Tests that plant explicit pair
  sets build the table with `tests/analysis/moments_oracle.py`.
- The report schema is unchanged. `docs/user-guide/output-structure.md`
  notes that validate correlations are exact over all pairs.

## Rejected

- **One graph per Analyze job.** See decision 5.
- **A pairs-based adapter for `compute_*`.** It would keep the pair list
  alive for callers that no longer need it; fitACE migrated in the same
  wave (plan D4.3).
- **Affection flags as value columns for phi.** The 2×2 table the moments
  already hold gives phi exactly with no extra accumulators.
