# ADR 0006: Combined Analyze Stage (Validate + Stats)

## Status

Accepted; the combined-stage decision stands, but its written artifacts are
superseded. The Decision below writes `validation.yaml` + `stats_report.yaml` +
`plotting_sample.parquet`; ADR 0007 merged the first two into `report.yaml`, and
ADR 0008 added `plot_payload.yaml` and recurated the shape. The "No schema or
artifact-path change" non-goal is superseded by ADR 0007. The single-Analyze-job
structure (Validate then Stats in one process, with phased memory frees) is
current — see ADR 0007/0008 for the artifact shape.

## Context

`CONTEXT.md` names **Analyze** as a single pipeline stage: the combined
production of ground-truth sanity checks (the **Validate output**) and
descriptive statistics (the **Stats output**), while keeping both as distinct
artifacts. The implementation, however, was two separate Snakemake jobs with
**disjoint inputs**:

| Stage | Reads | Pedigree scope | Producer rule |
|---|---|---|---|
| Validate | `pedigree.full.parquet` + `params.yaml` | full, pre-ascertainment | `validate_pedigree_liability` (`validate.smk`) |
| Stats | `trait.parquet` + `pedigree.parquet` | post-ascertainment subsample | `build_stats_report` (`stats.smk`) |

The two halves share **no input files** and operate over different node sets:
Validate's graph is built on the full pedigree, Stats' graph on the
post-ascertainment subsample. Genuine cross-stage graph reuse exists only in the
no-ascertainment (pass-through) case, which Stats already fast-paths internally
(`_same_ordered_ids` in `stats/runner.py`).

This change is a *first step* toward managing Validate and Stats as a single
unit. It is scaffolding, not a performance win: the actual
computational-efficiency work (cross-stage graph/pair sharing) is **deferred**.

## Decision

Introduce one **Analyze** stage:

- `simace/analysis/analyze.py::run_analysis()` runs Validate first, then Stats,
  in a single process, and writes all three frozen artifacts:
  `validation.yaml`, `stats_report.yaml`, `plotting_sample.parquet`.
- `validate`'s `run_validation` is split into a pure builder
  (`build_validation_report(df, params, *, df_indexed=None, sibling_pairs=None)`)
  and a thin disk-loading wrapper. `run_analysis` calls the builder directly.
  (`validate` is now the `simace/analysis/validate/` package; both functions live
  in `validate/runner.py`.)
- Stats reuses `build_stats_report` unchanged (Validate on the full pedigree,
  Stats on the post-ascertainment subsample — different scopes, no shared graph).
- One merged Snakemake rule (`analyze`, `analyze.smk`) replaces
  `validate_pedigree_liability` and `build_stats_report`. It uses `threads: 5`
  and reuses the existing `_scale_mem(G_ped)` / `_scale_runtime(G_ped)` helpers.
- `run_analysis` runs Validate first and **explicitly frees** its full-pedigree
  frame + graph (`del` + `gc.collect()`) before loading Stats inputs, so peak
  memory is `max(validate, stats)`, not their sum.

We deliberately do **not** thread a `RelationshipContext` into
`build_stats_report` — it would only pay off in the pass-through case, which is
already fast-pathed.

## Consequences

- `validation.yaml` is now produced **post-ascertainment** by the Analyze rule
  (previously it was available as soon as the full pedigree existed). The
  standalone `simace-validate` CLI still runs early on `pedigree.full.parquet`
  for ascertainment-independent debugging.
- The previous Validate ∥ Stats job parallelism is lost — they now run
  sequentially in one job.
- `validate.done` and `stats.done` both resolve to the Analyze rule (each pulls
  the full Analyze compute), because their targets reference output **files**,
  which Analyze still produces at the same paths.
- The standalone CLIs (`simace-validate`, `simace-phenotype-stats`) and a new
  `simace-analyze` CLI are retained for debug parity.

## Non-goals

- **No schema or artifact-path change.** Per-rep `validation.yaml`,
  `stats_report.yaml` (six groups), and `plotting_sample.parquet` keep the same
  paths and contents (see ADR 0003). fitACE consumes `stats_report.yaml` at its
  existing path and needs no change.
- **No cross-stage graph/pair sharing yet.** The two halves still build their
  own graphs over their own scopes.
- **`effective_size.yaml` is excluded.** It is a separate opt-in rule
  (`effective_size.smk`), not part of the stats runner, and is left untouched.

## Future

A later rethink can revisit single-unit management for genuine efficiency
(shared relationship extraction across stages), once the pass-through-only
benefit is worth the added coupling.
