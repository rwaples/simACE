# Pipeline schema

The pipeline is a chain of stages: simulate, phenotype, censor, ascertainment, then analysis. Each stage hands the next a `polars.DataFrame` or a parquet file. `simace run` runs phenotype, censoring, and ascertainment in one `cohort` process that passes frames in memory, so a replicate's files are `pedigree.parquet` from simulate and `cohort.parquet` from the `cohort` stage (ADR 0021). Stage boundaries are Polars-only (ADR 0015). The columns at each handoff are a contract. Without an explicit contract, a stage that renames a column breaks a stage far downstream, and the failure appears nowhere near the rename.

Two modules define the contract:

- `simace.core.schema` defines the pedigree schema, `PEDIGREE`, and the hydrated in-memory schemas `PHENOTYPE` and `CENSORED` that tests and analysis helpers use.
- `simace.core.trait_schema` defines the outcomes-only trait schemas and `hydrate_trait`, which joins trait outcomes to pedigree columns by `id` (ADR 0011).
- `simace.core.cohort` defines the cohort file, its invariants, and `selected_views`, which rebuilds the analysis pedigree and analysis sample from `pedigree.parquet` and `cohort.parquet` (ADR 0021).

The schema checker permits extra columns. The one exception is a hydration call that asks the pedigree for a column the trait frame already has. In that case `hydrate_trait` raises, so an old self-contained trait file cannot pass as an outcomes-only file.

## Pedigree schema

### `PEDIGREE`: output of `run_simulation`

| Column | Kind |
|---|---|
| `id`, `generation`, `sex`, `mother`, `father`, `twin`, `household_id` | `iu` (integer) |
| `A1`, `C1`, `E1`, `liability1` | `f` (float) |
| `A2`, `C2`, `E2`, `liability2` | `f` |

`pedigree.parquet` is the recorded pedigree, written once by simulate. The analysis pedigree, the sampled IDs plus every ancestor reachable from them, is not a file: `selected_views` rebuilds it from `pedigree.parquet` and `cohort.parquet`.

## Outcomes-only trait schemas

Trait frames hold only trait outcomes. Pedigree links, demography, ACE components, household IDs, and liabilities live in the pedigree. A consumer that needs them joins explicitly. Under `simace run` these frames stay in memory. The standalone `simace phenotype`, `censor`, and `ascertain` commands write them to the explicit paths they are given.

### `RAW_TRAIT`: output of `run_phenotype`

| Column | Kind |
|---|---|
| `id` | `iu` |
| `t1`, `t2` | `f` |

### `CENSORED_TRAIT`: output of `run_censor` and `run_ascertainment`

| Column | Kind |
|---|---|
| `id` | `iu` |
| `t1`, `t2`, `death_age`, `t_observed1`, `t_observed2` | `f` |
| `age_censored1`, `death_censored1`, `affected1` | `b` (bool) |
| `age_censored2`, `death_censored2`, `affected2` | `b` |

## Cohort file

`cohort.parquet` has the `CENSORED_TRAIT` columns (`COHORT_COLUMNS` in `simace.core.cohort`), one row per member of the analysis pedigree. `check_cohort` enforces four invariants whenever a cohort is built or read through `selected_views`:

1. `id` is unique and every `id` is in the pedigree.
2. On rows with `affected1` not null, every column except `id`, `t1`, and `t2` is non-null. These rows are the analysis sample.
3. On rows with `affected1` null, every column except `id` is null.
4. Rows are in pedigree row order.

[Ascertainment, Who is in `cohort.parquet`](../user-guide/ascertainment.md#who-is-in-cohortparquet) explains which individuals are in the file and why.

```python
from simace.core.cohort import read_cohort, read_pedigree, selected_views

views = selected_views(read_pedigree("pedigree.parquet"), read_cohort("cohort.parquet"))
views.pedigree  # analysis pedigree: links outside the cohort set to -1
views.trait  # analysis sample: CENSORED_TRAIT rows with affected1 not null
```

`pedigree.parquet` and `cohort.parquet` carry the Parquet key-value metadata `simace_layout=2`. `read_pedigree` and `read_cohort` raise when it is missing.

## Hydration

A consumer that needs a self-contained frame calls:

```python
from simace.core.trait_schema import hydrate_trait

hydrated = hydrate_trait(trait_df, pedigree_df, kind="censored")
```

Hydration keeps the trait row order and returns pedigree columns first, then trait outcome columns. It raises unless all four conditions hold:

- trait IDs are unique
- pedigree IDs are unique
- every trait ID exists in the pedigree
- no requested pedigree column is already in the trait frame

Pre-ascertainment trait frames hydrate against the pedigree the phenotype stage read. That is `pedigree.parquet`, or `pedigree.full.tstrait.parquet` for gene-drop scenarios. The analysis sample, `selected_views(...).trait`, hydrates against the analysis pedigree, `selected_views(...).pedigree`.

## Why the checker compares dtype kinds

The checker compares dtypes at the kind level, `i` or `u` for integer, `f` for float, and `b` for bool, rather than exact widths. [`save_parquet`][simace.core.parquet.save_parquet] narrows ID columns to `int32`, sex to `int8`, and ACE components to `float32` at save time, and an exact-width check would reject every file it wrote. A kind-level check still catches the regressions that matter: a boolean column written as `int8`, a string in an integer ID column, or a float in `generation`.

## Where the checker runs

The `@stage(reads=..., writes=...)` decorator in `simace.core.stage` wraps each DataFrame stage. It asserts the input schema on the first argument and the output schema on the return value. It also exposes both schemas as `fn.reads` and `fn.writes`. Every phenotype model, including `simple_ltm`, is a `PhenotypeModel` that runs through `run_phenotype` and `run_censor`. There is no separate threshold stage.

```mermaid
flowchart LR
    sim[run_simulation] -- PEDIGREE --> phen[run_phenotype]
    phen -- RAW_TRAIT --> cen[run_censor]
    cen -- CENSORED_TRAIT --> asc[run_ascertainment]
    asc -- CENSORED_TRAIT + PEDIGREE --> coh[build_cohort]
    coh -- cohort.parquet --> sv[selected_views]
    sim -- pedigree.parquet --> sv
    sv -- analysis sample + analysis pedigree --> ana[analyze / stats]
    ana -- hydrate_trait --> hyd[hydrated in-memory frames]
```

| Stage | Input asserted | Output asserted |
|---|---|---|
| `run_simulation` | none (no input frame) | `PEDIGREE` |
| `run_phenotype` | `PEDIGREE` | `RAW_TRAIT` |
| `run_censor` | `RAW_TRAIT`. The pedigree argument, from which `run_censor` hydrates `generation`, gets an explicit `PEDIGREE` check | `CENSORED_TRAIT` |
| `run_ascertainment` | outcomes-only trait frame plus pedigree. ID-level checks, not `@stage` | outcomes-only trait frame plus analysis pedigree |
| `build_cohort` | `run_ascertainment`'s two outputs | cohort, checked by `check_cohort` |
| `selected_views` | pedigree plus cohort, checked by `check_cohort` | analysis pedigree plus analysis sample |
| Analyze and stats | analysis sample plus analysis pedigree | hydrated in-memory frames |

A failed check raises `ValueError` naming the boundary and the offending column:

```
censor input: missing required columns ['t1']
trait columns collide with requested pedigree columns; hydrate outcomes-only trait files or drop duplicate columns first: ['generation']
```

The error points at the boundary that broke, not at the analysis code that read the column later.

## Schemas in tests

When a unit test builds a `DataFrame` by hand, use the schema constants in `simace.core.trait_schema` for outcomes-only trait frames. Call `hydrate_trait(...)` before passing a frame to a stats helper that needs pedigree columns. `tests/conftest.py` exposes `schema_pad(df, schema)` for the older hydrated-schema fixtures.

## API reference

See [`simace.core.schema`](../api/core.md#schema), [`simace.core.trait_schema`](../api/core.md#trait_schema), and [`simace.core.cohort`](../api/core.md#cohort).
