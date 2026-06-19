# ADR 0011: Outcomes-Only Trait Parquet Files

## Status

Accepted. Amended 2026-06-10 (see Amendment below).

## Context

simACE historically wrote self-contained trait-family parquet files. For
example, `trait.parquet` repeated pedigree links, demography, ACE components,
and liabilities already present in `pedigree.parquet`, then added event times,
censoring flags, and affected status. This made downstream code convenient but
made large runs duplicate the highest-volume columns across multiple files.

The duplication is especially costly because the same latent columns appear in
`pedigree.full.parquet`, `pedigree.parquet`, `trait.full.parquet`, and
`trait.parquet`. The project now treats the
file boundary as part of the domain model: pedigree files own pedigree,
demography, ACE components, and liabilities; trait files own only trait
outcomes.

## Decision

Trait-family parquet files are outcomes-only:

- `trait.raw.parquet`: `id`, `t1`, `t2`
- `trait.full.parquet` and `trait.parquet`: `id`, `t1`, `t2`, `death_age`,
  `age_censored1`, `t_observed1`, `death_censored1`, `affected1`,
  `age_censored2`, `t_observed2`, `death_censored2`, `affected2`

Consumers that need pedigree/demography/liability columns hydrate explicitly by
joining on `id` with `simace.core.trait_schema.hydrate_trait(...)`. Hydration is
strict: trait IDs and pedigree IDs must be unique, every trait ID must exist in
the pedigree, and requested pedigree columns must not already be present in the
trait frame. The hydrated in-memory shape puts pedigree columns first and trait
outcome columns second, matching the old self-contained column order as closely
as possible.

Pre-ascertainment trait files hydrate against the actual phenotype input
pedigree (`pedigree.full.parquet`, or `pedigree.full.tstrait.parquet` when a
scenario uses gene-drop-derived values). Post-ascertainment trait files hydrate
against `pedigree.parquet`, the ancestor-closure pedigree supporting the final
analysis sample.

This is a hard output-contract cut: the pipeline writes only outcomes-only trait
files. Old self-contained trait files are not silently accepted by the strict
hydration helper; existing results must be regenerated or handled by explicit
one-off migration tooling.

## Consequences

- Large runs write substantially less duplicated parquet data.
- `trait.parquet` is no longer sufficient on
  its own for relationship, liability, or generation-stratified analyses;
  consumers must read the appropriate pedigree file and hydrate explicitly.
- The Analyze and Stats stages hydrate before computing summaries that need
  generation, sex, pedigree links, or liabilities.
- fitACE consumers must be updated in lockstep because they read simACE trait
  outputs for liabilities, household IDs, and pedigree-aware fits.
- The file contract is clearer: `pedigree*.parquet` owns latent/pedigree state;
  `trait*.parquet` owns outcomes.

## Non-goals

- No `trait_schema` config flag and no dual-writing hydrated trait files.
- No automatic migration of old result directories.
- No change to relationship semantics, ascertainment semantics, or report field
  meanings.

## Amendment (2026-06-10): `trait.simple_ltm.*` retired

The original decision listed a `trait.simple_ltm.full.parquet` /
`trait.simple_ltm.parquet` pair (`id`, `affected1`, `affected2`) produced by a
**parallel, censoring-free liability-threshold phenotyping path**. That parallel
path has been removed:

- `simple_ltm` is now a regular phenotype **model** in the registry (probit
  threshold for case status + a `fixed`/`normal` age-of-onset), so it flows
  through the standard `phenotype → censor → ascertainment` pipeline like every
  other model and writes only `trait.raw.parquet` → `trait.full.parquet` →
  `trait.parquet`.
- There is no longer a separate clean-binary trait file. The descriptive binary
  statistics that consumed it (fitACE's `observed_binary_*`, Falconer h²) now
  read `affected1`/`affected2` from the censored `trait.parquet` for **every**
  scenario — describing whatever phenotype model the scenario configured, not a
  censoring-free benchmark.

The outcomes-only contract for `trait.raw.parquet`, `trait.full.parquet`, and
`trait.parquet` is unchanged.
