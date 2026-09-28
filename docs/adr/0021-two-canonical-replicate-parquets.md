# ADR 0021: Two canonical Parquet files per replicate

## Status

Accepted design; implementation pending. Decided 2026-09-28.

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
