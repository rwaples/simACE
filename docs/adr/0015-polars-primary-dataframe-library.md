# ADR 0015: Polars is the primary DataFrame library

## Status

Accepted. Grill-with-docs session 2026-08-13. Supersedes the earlier
three-layer boundary (polars writes, pandas transports, numpy computes) and its
deferral of the full migration; that decision's measurements are retained
below as evidence and the decision itself is retired. The implementation plan (three waves, Wave 0 contract and plumbing, Wave 1 simACE
internals stage by stage, Wave 2 the coordinated boundary break across repos)
was completed in v2026.08 and retired; the surviving artifact is the user-facing
guide `docs/polars-migration-guide.md`.

## Context

An earlier investigation measured a pandas→polars migration and deferred it:
polars kept only the parquet write path, pandas remained the transport type at
every module and repo boundary, and numpy owned compute. The deferral produced
a deliberate but awkward shape — two DataFrame libraries in the dependency set
with one function using polars — and left the family's frame conventions
inconsistent across repos.

### Measurements from the deferred migration

These numbers justified the earlier boundary and now serve as regression
guardrails. Re-measure rather than assume they still hold.

- **Parquet writes.** `pl.from_pandas(..., nan_to_null=False).write_parquet`
  versus `pd.to_parquet` at 6M rows: 3.6s → 0.55s and 273 MB → 248 MB (5.8x
  faster, 9% smaller); 7.5x and 9% at 18M rows. The conversion is zero-copy
  for the numeric dtypes this pipeline writes.
- **Parquet reads.** `pl.read_parquet` is faster in isolation, but with every
  caller needing a pandas frame the `to_pandas()` copy more than cancelled the
  gain: 410ms vs pandas' 297ms at 6M rows. The read win is only collectable by
  a polars-native consumer.
- **Pandas compute was already thin.** Counting `groupby`/`merge`/
  `sort_values`/`value_counts`/`agg`/`concat`/`pivot`/`apply`/
  `drop_duplicates`: `phenotype`, `censoring`, `ascertainment`, `core` had 0;
  `simulation` 2; `analysis/stats` 6; `plotting` 10 (3 files seaborn-bound).
  Pandas in most packages was a carrier for reads, writes, and column access,
  not an engine, so converting a transport layer while pandas held both ends
  would have produced a convert-in/convert-out sandwich.
- **Memory is not an argument either way.** In-memory footprint was identical
  at 0.390 GB under both libraries (both Arrow-backed for these dtypes). The
  RSS spikes in this pipeline come from `np.unique` temporaries in
  `pedigree-graph` (+1.22 GB) and simulation internals (+1.58 GB), neither of
  which either library helps.
- Validation, the largest DataFrame consumer at 37.45s, was moved onto numpy
  arrays (`simace.core.pedigree_arrays.PedigreeArrays`), removing the biggest
  frame workload from the question entirely.

The revisit came not from performance but from maintainability: the family now
spans ten frame-owning Python repos plus two frame-neutral externals, and every
cross-repo surface (fitACE's `hydrate_trait` consumption, EPIMIGHT's emitter,
plotting preprocessing) pays the cost of a mixed pandas/polars idiom.
Standardizing on one primary frame API is the point; the benchmarks above
become regression guardrails rather than the justification.

An investigation on 2026-08-13 (pandas 3.0.5, polars 1.43.2: `pd.to_parquet`
writes a float NaN as parquet null, null_count=1, while
`pl.from_pandas(..., nan_to_null=False).write_parquet` writes a literal NaN,
null_count=0) also found that `nan_to_null=False` — added to `save_parquet`
by the earlier decision as "load-bearing", on the reasoning that polars
distinguishes NaN from null where pandas conflates them and a pandas
round-trip could not catch the regression — had actually *changed* the
historical on-disk contract rather than preserved it: pandas' own
`to_parquet` writes float NaN as parquet **null**, so every pandas-era artifact
already carried nulls. The only NaN-encoded files were locally regenerated,
unpushed test results.

## Decision

1. **Polars-primary, scope B.** Polars is the default DataFrame type wherever
   frame-owning family code is written and maintained: simACE, fitACE core, the
   seven `fitACE_*` method sisters, and pedsum. `pedigree-graph` stays
   frame-library-neutral over its native (Rust) kernels (structural frame protocol:
   `.columns`, `__getitem__`, column `.to_numpy()`; accepts dict/polars/pandas;
   neither library is a runtime dependency).
2. **Pandas survives only at forced third-party boundaries**: the seaborn plot
   modules (three files family-wide, converting the minimal plot-input frame at
   the call edge), the pandas-native tstrait/tskit gene-drop scripts, and
   immediate third-party edge shims (e.g. cmdstanpy results). Pandas is never a
   direct base dependency after migration; it moves to the extras that actually
   import it (simACE: `plot`, `workflow`, `test`).
3. **Null contract (restoration).** Missing = parquet null on disk and null in
   polars frames. `save_parquet` drops `nan_to_null=False` and self-enforces
   with `fill_nan(None)` at the write edge. NumPy/SciPy compute may transiently
   materialize nulls as NaN; arrays re-entering a frame are normalized.
   Regression tests pin the on-disk null mask; non-nullable integer schemas
   (pedigree ids, EPIMIGHT int8/int16) are frozen.
4. **Staged waves over parquet seams.** Pipeline stages communicate through
   parquet on disk, so stages flip independently: Wave 0 = contract + core
   plumbing; Wave 1 = simACE internals stage by stage; Wave 2 = the coordinated
   cross-repo boundary break (simulation + fitACE family, removing transitional
   pandas acceptance); independent early track = pedigree-graph, then pedsum.
5. **Eager-only.** Public and stage APIs accept/return `pl.DataFrame`, never
   `LazyFrame`; `load_parquet` is eager. Lazy/streaming is a separate
   benchmark-driven follow-up.
6. **DataFrame index carries no meaning.** Durable identity/order lives in
   explicit columns or row position; public return indexes that carried
   information become named columns (breaking change, documented in the Wave 2
   migration guide).
7. **Version floor** `polars>=1.43.2,<2` across the ten frame-owning repos.
   `FAMILY_FLOOR`, pins, tags, and pushes are untouched during implementation;
   the next coordinated family release performs the floor bump (ADR 0012
   machinery).
8. **Seeded reproducibility.** Ascertainment/dropout and `create_sample` keep
   their NumPy RNG/row-position logic and exact fixed-seed selected IDs. Only
   plotting-only sampling may change IDs, and only deterministically.

## Consequences

- The Wave 2 boundary break removes lockstep-family pandas compatibility with
  no long-lived shim; changed public APIs reject pandas with an actionable
  `TypeError`, and a migration guide ships with the release.
- The earlier read-path argument (the `to_pandas()` copy cancels the read win)
  dissolves: consumers are polars-native, so the read win is collectable.
  Its write-path win is retained unchanged.
- The one intentional scientific-artifact change is the NaN→null restoration;
  artifact equivalence checks treat it as the single allowed exception.
- Pandas remains installed transitively where a required third party owns it;
  a structural source check keeps pandas imports confined to a documented
  allowlist (seaborn, tstrait/tskit, compatibility tests, edge shims,
  comparative benchmarks, archival scratch).
- EPIMIGHT's R→Python TSV handoff parses `NA` as missing implicitly under
  pandas; every migrated TSV read must pass `null_values=["NA", ""]`
  explicitly — this is the one silent-data-loss trap the migration introduces
  and it is pinned by tests.
