# Migrating to the polars simACE API

As of the polars migration ([ADR 0015](adr/0015-polars-primary-dataframe-library.md),
Wave 2), simACE's public DataFrame surface is **polars-only**. Pandas frames
are rejected at every changed boundary with an actionable `TypeError`; there
is no auto-conversion and no long-lived shim. This guide covers every changed
public function, the index→column replacements, null semantics, and the one
reproducibility change.

## TL;DR for callers

```python
import polars as pl

# was: pd.read_parquet(path)
df = load_parquet(path)  # simace.core.parquet.load_parquet

# holding a pandas frame you can't avoid? convert at the call site:
hydrate_trait(pl.from_pandas(trait_pd), pl.from_pandas(ped_pd), kind="censored")
```

## Changed public functions

| Function | Old accepted / returned | New accepted / returned |
|---|---|---|
| `simace.simulation.simulate.run_simulation` | returned `pd.DataFrame` | returns `pl.DataFrame` |
| `simace.simulation.simulate.add_to_pedigree` | `pd.DataFrame` in/out | `pl.DataFrame` in/out |
| `simace.core.parquet.save_parquet` | `pd.DataFrame` (later either) | `pl.DataFrame` only; `TypeError` otherwise |
| `simace.core.parquet.load_parquet` | — (new in this migration) | returns eager `pl.DataFrame` |
| `simace.core.schema.assert_schema` | `pd.DataFrame` (later either) | `pl.DataFrame` only; rejects `LazyFrame` too |
| `simace.core.trait_schema.hydrate_trait` | `pd.DataFrame` in/out | `pl.DataFrame` in/out; `TypeError` otherwise |
| `simace.core.trait_schema.strip_trait_to_outcomes` | `pd.DataFrame` in/out | `pl.DataFrame` in/out |
| `simace.core.pedigree_filter.filter_pedigree_to_observed` | `pd.DataFrame` in/out | `pl.DataFrame` in/out |
| `simace.core.snakemake_adapter.write_parquet_plain` | `pd.DataFrame` | `pl.DataFrame` |
| `simace.phenotype.run_phenotype` | `pd.DataFrame` in/out | `pl.DataFrame` in/out |
| `simace.phenotype.blended_post.blended_diagnosis` | `pd.DataFrame` in/out | `pl.DataFrame` in/out; `TypeError` otherwise |
| `simace.censoring.censor.run_censor` | `pd.DataFrame` in/out | `pl.DataFrame` in/out |
| `simace.ascertainment.run_ascertainment` | `pd.DataFrame` in/out | `pl.DataFrame` in/out; `TypeError` otherwise |
| `simace.analysis.stats.runner.create_sample` | `pd.DataFrame` in/out | `pl.DataFrame` in/out; `TypeError` otherwise |
| `simace.analysis.stats.build_stats_report` and the `compute_*` family | `pd.DataFrame` | any frame whose columns expose `.to_numpy()` (library-agnostic; polars is canonical) |
| `simace.analysis.validate.build_validation_report` | `pd.DataFrame` | same library-agnostic contract |
| `simace.core.pedigree_arrays.PedigreeArrays.from_frame` | `pd.DataFrame` | structural: any frame with `.columns` / `__getitem__` / column `.to_numpy()` |

`PedigreeGraph` (external `pedigree_graph` package) accepts polars, pandas,
and `dict[str, np.ndarray]` through its structural `FrameLike` protocol —
that compatibility promise is separate from simACE's and unchanged.

## Index → column replacements

The DataFrame index carries no identity or order anywhere in simACE anymore.

- `run_simulation` output was default-indexed; unchanged semantics — row
  order and the `id` column are the identity, as before.
- `create_sample` previously **leaked its input's pandas row labels** in the
  returned index. The polars frame has no index; use the `id` column. If you
  relied on those row labels to map back to input positions, join on `id`.
- `hydrate_trait` always reset its index; no change beyond the frame type.
- Everything else already communicated through explicit columns.

## Null semantics (read this if you touch missing values)

- **On disk, missing = parquet null** — as it was in the pandas era
  (`pd.to_parquet` always wrote NaN as null). `save_parquet` self-enforces by
  normalizing float NaN → null on every write.
- **In memory, missing = polars null, never NaN.** `load_parquet` returns
  null-carrying frames. `Series.to_numpy()` materializes nulls as NaN for
  float compute — normalize back (e.g. `Series.fill_nan(None)`) if the array
  re-enters a frame.
- Do not assume pandas-skipna parity for aggregations: polars `mean`/`sum`
  skip **null** but propagate **NaN**. Keep missing as null and the behavior
  matches pandas' skipna defaults.
- Reading R-written TSVs? pandas parsed the string `NA` as missing
  implicitly; polars does not — pass `null_values=["NA", ""]` to
  `pl.read_csv` (all in-tree readers already do).

## Reproducibility

- **Scientific sampling is bit-identical**: ascertainment/dropout and
  `create_sample` retain their NumPy row-position RNG, so fixed-seed selected
  IDs are unchanged (test-pinned).
- **One change**: plotting-only downsampling in `plot_phenotype` (the 200K
  scatter cap) now draws with `pl.sample(seed=42)` instead of pandas
  `random_state=42`. Deterministic, same cap, different row IDs. Plot inputs
  only; no scientific artifact is affected.

## Dependency changes

- `polars>=1.43.2,<2` is a base dependency of every frame-owning family repo.
- `pandas` is no longer a **direct** base dependency anywhere in the family;
  it is declared only in extras that use it (simACE: `plot`, `workflow`,
  `test`). Transitive installation may still occur where a third party
  requires it.
