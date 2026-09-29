# ADR 0021: Two canonical Parquet files per replicate

## Status

Accepted. Decided 2026-09-28. Implemented in simACE 2026-09-29 (results
layout 2); see Amendment (2026-09-29). Amends ADR 0008, 0011, and 0020.

## Context

The current run writes a full recorded pedigree, an ascertained pedigree, and
separate raw, pre-ascertainment, and sampled trait Parquets. The selected
pedigree repeats most rows and columns of the full pedigree. The trait files
repeat outcomes across stages. The CLI's stage subprocesses serialize and
reload data at these file boundaries during a replicate.

The four report scopes remain distinct: recorded pedigree, phenotyped
population, analysis pedigree, and analysis sample. Ascertainment can remove
ancestors and sever parent or twin links. The finished outputs need the final
analysis pedigree and sample; pre-ascertainment individual outcomes can be
regenerated when needed.

## Decision

The canonical finished Parquet outputs for a replicate are:

- `pedigree.parquet`: the full recorded pedigree, with the original links,
  demographics, A/C/E components, and liabilities. This takes the role of the
  current `pedigree.full.parquet`; the current selected `pedigree.parquet`
  meaning is retired.
- `cohort.parquet`: one row for each member of the final analysis pedigree.
  It carries observable trait outcomes only for the selected analysis sample;
  outcome columns are null for the other retained ancestors, even when those
  ancestors were phenotyped before ascertainment. The schema requires a
  non-null `affected1` for every selected person, so that column distinguishes
  the analysis sample from the retained ancestors. The file has no separate
  membership indicators. The selected pedigree's parent and twin links are
  derived by filtering `pedigree.parquet` to the IDs in `cohort.parquet` and
  replacing links to IDs outside that set with `-1`.

The two files must reproduce the current analysis sample trait and selected
analysis pedigree views without changing their statistical meaning. Analyze
must compute pre-ascertainment summaries while the phenotyped population is
available during the run. Rebuilding those individual outcomes later requires
the recorded pedigree, resolved scenario parameters, seed, and compatible
simulation code and environment. Standalone stage commands may still accept
explicit files, but `simace run` should not serialize each intermediate trait
or selected pedigree as a canonical output. Keep the Analyze stage's distinct
memory phases when changing the execution path.

`plotting_sample.parquet` is outside this decision. The scientific report,
plot payload, parameters, timing, and run manifest remain separate artifacts.

## Consequences

The new meaning of `pedigree.parquet` and the removal of the separate
`trait.parquet` change fitACE's input contract. The implementation must migrate
fitACE's readers and rules or provide an explicit projection of the old pair;
the rollout strategy is not decided here. Existing results cannot be
silently interpreted under the new names.

This supersedes ADR 0011's separate outcomes-only trait-file decision for
canonical finished outputs. Its definition of outcome fields and strict
identity joins remains relevant to the views derived from `cohort.parquet`.

## Amendment (2026-09-29): implementation

Decided while implementing the Decision above
(`plans/adr-0021-implications-v2.md`).

- **One `cohort` stage.** `simace run` computes a rep in three stages:
  `simulate`, `cohort`, `analyze`. The `cohort` stage
  (`simace/cli/cohort_stage.py`) runs phenotype, censoring, and
  ascertainment in one process on in-memory frames and writes only
  `cohort.parquet` and `phenotyped_population.yaml`. No trait file reaches
  disk during a rep, so peak disk during a rep equals the finished output.
  Each phase goes through `normalize_for_parquet`, so it sees the values it
  would have read back from a file. The stage log records each phase's wall
  time and peak RSS; `timing.tsv` has one `cohort` row. The standalone
  `simace phenotype`, `censor`, and `ascertain` commands keep their
  explicit-file inputs and outputs.
- **Durable phenotyped-population summary.** This amends the sentence
  "Analyze must compute pre-ascertainment summaries while the phenotyped
  population is available during the run". The `cohort` stage computes them
  while it holds the censored phenotyped rows and writes them to
  `phenotyped_population.yaml`: `n_individuals`, `n_generations`, and the
  `compute_prevalence` result. Analyze reads that file and assembles the
  report from it. Analyze takes four durable inputs, `--pedigree`,
  `--params`, `--cohort`, and `--phenotyped-population`, so a standalone
  `simace analyze` works on any finished rep, and every report scope and
  `report_summary.tsv` column is filled. `n_generations` of the phenotyped
  population is now counted on the hydrated rows; before, it was counted on
  the outcomes-only frame and reported 1.
- **One view builder.** `simace/core/cohort.py` builds the cohort from
  `run_ascertainment`'s two outputs (`build_cohort`), checks it
  (`check_cohort`), and rebuilds the analysis pedigree and analysis sample
  (`selected_views`). `check_cohort` enforces four invariants: ids unique
  and in the pedigree; every outcome column except `t1` and `t2` non-null
  where `affected1` is not null (the raw onsets stay nullable, ADR 0019);
  every outcome column null where `affected1` is null; rows in pedigree
  order. `tests/core/test_cohort.py` checks that `selected_views` rebuilds
  `run_ascertainment`'s two outputs frame for frame, including row order,
  which the seeded plotting sample and stats draws depend on.
- **Layout marker.** The two-file layout is results layout 2, recorded in
  two places. `run.yaml` carries `layout: 2`, compared like `stages`: a rep
  from before this change reads stale (`layout: (absent) -> 2`), so
  `simace run` refuses it without `--force`, `simace ls` and `gather` report
  it as stale, and `simace show` leaves it out of its timing. A recompute
  removes the layout 1 files (`pedigree.full.parquet`, `trait.parquet`,
  `trait.raw.parquet`, `trait.full.parquet`). `pedigree.parquet` and
  `cohort.parquet` carry the Parquet key-value metadata `simace_layout=2`,
  written by `write_pedigree` and `write_cohort`. `read_pedigree` and `read_cohort` refuse a file
  without it, which stops a layout 1 selected `pedigree.parquet` from being
  read as the recorded pedigree. `simace cohort`, `simace analyze`, and
  `simace effective-size` read through them, and every `selected_views` call checks the cohort
  structurally.
- **Route C is the retained information loss.** A person enters
  `cohort.parquet` as drawn (route A, `affected1` not null), as an ancestor
  of a drawn person older than the phenotyped window (route B), or as an
  ancestor inside the phenotyped window who was phenotyped but not drawn
  (route C). Routes B and C both carry null outcomes, although route C's
  outcomes were computed. The file alone cannot separate them; a reader
  needs `generation` from `pedigree.parquet` and `G_pheno`. Route C arises
  only when the draw is not pass-through and `G_pheno >= 2`. The layout 1
  pair of selected `pedigree.parquet` and `trait.parquet` lost the same
  information, so nothing is lost relative to it, but the null now sits in
  the same file as the outcomes. [Ascertainment, Who is in
  `cohort.parquet`](../user-guide/ascertainment.md#who-is-in-cohortparquet)
  has the full table.
- **Measured cost of the merged stage (accepted).** Interleaved runs against
  the five-stage layout, medians of three per arm on one clock-limited host:
  whole-rep wall -25% at baseline100K and -11% at bench5M; peak disk during
  the rep -41% and -51%; Analyze peak RSS unchanged. The `cohort` stage's
  peak RSS is +5% over the largest of the three stages it replaces at
  bench5M and +37% at baseline100K (445 MB against 325 MB). The 100K rise is
  freed memory the allocators keep between phases, not live data: jemalloc
  dirty-page decay 0 removes it but costs about 16% of the stage's wall at
  bench5M. The plan made a +10% bound on this stage the condition for
  merging. It was relaxed on 2026-09-29 because Analyze sets each rep's peak
  memory at every measured scale (about 870 MB at 100K), so the merged
  stage does not raise it. Allocators stay at their defaults.
- **fitACE rollout: hard cut.** This settles the rollout the Consequences
  left open. fitACE reads only cohort-shaped files, through one loader
  that calls `selected_views`; fitACE_epimight uses the same loader. No
  command writes the old pair, and neither layout is dual-written. As with
  ADR 0008 and 0011, simACE merges first, fitACE and fitACE_epimight
  repoint, and a brief CI gap between the merges is expected.
