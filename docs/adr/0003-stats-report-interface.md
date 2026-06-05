# ADR 0003: Grouped Stats Report Interface

## Status

Accepted, but superseded. ADR 0007 replaced the `stats_report.yaml` artifact
with a single `report.yaml`, and ADR 0008 then replaced the six-group shape with
the curated `quality_checks` / `truth` / `observed` / `estimators` groups. The
"No Validate-stage change" non-goal is likewise superseded by ADR 0007. This ADR
remains the historical origin of the grouped report; see ADR 0007/0008 for the
current shape. (`plotting_sample.parquet`, introduced here, is unchanged by those
ADRs.)

## Context

The Stats stage historically wrote a flat `phenotype_stats.yaml` file and a plotting-only `phenotype_samples.parquet` file. The flat report grew as new summaries were added, which made the Interface hard to scan and easy to couple accidentally to implementation order.

fitACE also declared `phenotype_stats.yaml` as an expected simACE output path. That made the artifact name part of the cross-repo Interface even though fitACE does not read the report contents.

## Decision

The Stats stage now writes:

- `stats_report.yaml`
- `plotting_sample.parquet`

`stats_report.yaml` uses six top-level groups: `metadata`, `incidence`, `censoring`, `pedigree`, `correlations`, and `heritability`.

This is a breaking Interface change. simACE emits only the new artifacts and fitACE updates its expected simACE target from `phenotype_stats.yaml` to `stats_report.yaml`. There is no one-cycle dual emit and no old-schema reader.

`joint_affection` lives under `correlations` in the new report because it is a bivariate Trait 1 x Trait 2 affection summary. Its current implementation lives in `incidence.py`, but the report grouping follows domain meaning rather than Module location.

## Consequences

Existing result directories with `phenotype_stats.yaml` and `phenotype_samples.parquet` must be regenerated before plotting with the current code.

The simACE change should merge first, followed immediately by the fitACE path update. fitACE CI failures in the gap are expected and signal that the coordinated migration is incomplete.

## Non-goals

- No Validate-stage change.
- No fitACE behavior change beyond the simACE target-path migration.
- No compatibility shim for old flat stats reports.
