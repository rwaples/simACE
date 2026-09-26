# Architecture Decision Records

One file per decision, numbered in the order they were made. Numbers are never
reused; a retired ADR leaves a gap. When a later ADR amends an earlier one, the
earlier one's Status section says so and points forward.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-unified-ascertainment-stage.md) | Unify dropout and subsampling into a single ascertainment stage | accepted |
| [0002](0002-wright-fisher-mating-model.md) | Sex-structured Wright-Fisher mating model alongside the standard model | accepted |
| [0005](0005-prevalence-target-semantics-and-model-assignment.md) | Prevalence-target semantics and phenotype-model assignment for the calibrated scenario folders | accepted |
| [0008](0008-curated-analyze-report.md) | Curated Analyze report (`report.yaml` v2) + `plot_payload.yaml` | accepted; carries the retired 0003/0006/0007 lineage |
| [0009](0009-relationship-semantics-home.md) | Home for relationship semantics | accepted |
| [0010](0010-html-primary-atlas-rendering.md) | HTML as the primary atlas rendering | accepted |
| [0011](0011-outcomes-only-trait-files.md) | Outcomes-only trait parquet files | accepted, amended 2026-06-10 |
| [0012](0012-lockstep-family-versioning.md) | Lockstep CalVer versioning across the simACE/fitACE family | accepted, membership amended by 0017 |
| [0015](0015-polars-primary-dataframe-library.md) | Polars is the primary DataFrame library | accepted; carries the retired 0014 measurements |
| [0016](0016-pixi-canonical-simace-environment.md) | pixi is simACE's canonical environment | accepted, conda half retired by 0018, Snakemake removed by 0020 |
| [0017](0017-family-monorepo-epimight-cut.md) | Family monorepo with the epimight cut (13 repos → 5) | accepted, implemented 2026-08-21 |
| [0018](0018-retire-conda-family-environment.md) | Retire the conda family environment (Linux-only pipeline) | accepted |
| [0019](0019-null-raw-onset-censoring-semantics.md) | Null raw onsets mean never-onset at the censor boundary | accepted |
| [0020](0020-standalone-simace-cli.md) | A standalone `simace` CLI replaces Snakemake | accepted |

Retired numbers: 0003, 0006, 0007 (report lineage folded into 0008); 0004
(Claude Code subagent housekeeping, not an architecture decision); 0013 (ty
budget ratchet, documented in `tools/typecheck_family.py`); 0014 (superseded
by 0015, measurements folded in).
