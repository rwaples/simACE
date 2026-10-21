# Architecture Decision Records

One file per decision, numbered in the order they were made. Numbers are never
reused; a retired ADR leaves a gap. When a later ADR amends an earlier one, the
earlier one's Status section says so and points forward.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-unified-ascertainment-stage.md) | Unify dropout and subsampling into a single ascertainment stage | accepted |
| [0002](0002-wright-fisher-mating-model.md) | Sex-structured Wright-Fisher mating model alongside the standard model | accepted |
| [0005](0005-prevalence-target-semantics-and-model-assignment.md) | Prevalence-target semantics and phenotype-model assignment for the calibrated scenario folders | accepted |
| [0008](0008-curated-analyze-report.md) | Curated Analyze report (`report.yaml` v2) + `plot_payload.yaml` | accepted; carries the retired 0003/0006/0007 lineage; Analyze inputs amended by 0021 |
| [0009](0009-relationship-semantics-home.md) | Home for relationship semantics | accepted |
| [0010](0010-html-primary-atlas-rendering.md) | HTML as the primary atlas rendering | accepted |
| [0011](0011-outcomes-only-trait-files.md) | Outcomes-only trait parquet files | accepted; superseded for finished outputs by 0021 |
| [0012](0012-lockstep-family-versioning.md) | Lockstep CalVer versioning across the simACE/fitACE family | accepted, membership amended by 0017, scheme and floor amended by 0023 |
| [0015](0015-polars-primary-dataframe-library.md) | Polars is the primary DataFrame library | accepted; carries the retired 0014 measurements |
| [0016](0016-pixi-canonical-simace-environment.md) | pixi is simACE's canonical environment | accepted, conda half retired by 0018, Snakemake removed by 0020 |
| [0017](0017-family-monorepo-epimight-cut.md) | Family monorepo with the epimight cut (13 repos → 5) | accepted, implemented 2026-08-21 |
| [0018](0018-retire-conda-family-environment.md) | Retire the conda family environment (Linux-only pipeline) | accepted |
| [0019](0019-null-raw-onset-censoring-semantics.md) | Null raw onsets mean never-onset at the censor boundary | accepted |
| [0020](0020-standalone-simace-cli.md) | A standalone `simace` CLI replaces Snakemake | accepted; output contract and rep stages replaced by 0021 |
| [0021](0021-two-canonical-replicate-parquets.md) | Two canonical Parquet files per replicate | accepted, implemented 2026-09-29; amends 0008, 0011, 0020 |
| [0022](0022-relationship-moments.md) | Analyze computes relationship statistics from relationship moments | accepted |
| [0023](0023-lockstep-semver.md) | Lockstep SemVer for the family version | accepted; amends 0012 |
| [0024](0024-realised-assortment-feasibility.md) | Reject infeasible realised two-trait assortment | accepted, implemented 2026-10-08 |
| [0025](0025-vertical-c-transmission.md) | Vertical transmission of the common-environment component | accepted, implemented 2026-10-14; amends 0009 |
| [0026](0026-mate-correlation-checks-by-generation.md) | Split mate-correlation checks by generation when pooling is invalid | accepted, implemented 2026-10-16; §1 and §4 amended by 0031 |
| [0027](0027-onset-tau-check.md) | Score onset τ against its realized expectation | accepted, implemented 2026-10-17; placement and per-rep pass amended by 0028 |
| [0028](0028-analyze-validate-boundary.md) | Quality checks in Analyze, calibrated tests in `simace validate` | accepted, implemented 2026-10-17; amends 0008, 0026, 0027; §9 amended by 0029, 0030, 0031 |
| [0029](0029-inheritance-tests-score-residuals.md) | The sibling inheritance tests score residuals, not relative correlations | accepted, implemented 2026-10-20; amends 0028 |
| [0030](0030-generation0-total-variance-and-rc-tests.md) | Total variance and cross-trait rC score recorded generation 0 with a household-cluster SE | accepted, implemented 2026-10-20; amends 0028 |
| [0031](0031-mate-correlation-calibrated-test.md) | Per-generation mate correlations as a calibrated test, with an SE that follows the pairing mode | proposed; amends 0026, 0028 |

Retired numbers: 0003, 0006, 0007 (report lineage folded into 0008); 0004
(Claude Code subagent housekeeping, not an architecture decision); 0013 (ty
budget ratchet, documented in `tools/typecheck_family.py`); 0014 (superseded
by 0015, measurements folded in).
