# DataFrame library boundaries: polars writes, pandas transports, numpy computes

An investigation into replacing pandas with polars measured three candidate
changes. Only the narrowest of them earned its place, which leaves the codebase
in a state that looks accidental unless the reasoning is written down: **polars
is a declared dependency used by exactly one function.** This ADR records the
boundary that produced that shape, and what would have to change for the wider
migration to become worthwhile.

## Decision

Three layers, each owning what it measurably does best:

- **polars owns parquet writes.** `simace.core.parquet.save_parquet` converts the
  dtype-narrowed frame with `pl.from_pandas(..., nan_to_null=False)` and writes
  through `write_parquet`. Measured at 6M rows: 3.6s → 0.55s and 273 MB → 248 MB
  (5.8x faster, 9% smaller); 7.5x and 9% at 18M rows. The conversion is zero-copy
  for the numeric dtypes this pipeline writes, so it costs nothing.
- **pandas remains the transport type** — at reads, and at every function fitACE
  imports. It is what crosses module and repo boundaries.
- **numpy owns compute.** Analysis reaches pedigree columns through
  `simace.core.pedigree_arrays.PedigreeArrays`, not through a DataFrame index.

## Why reads stayed on pandas

`pl.read_parquet` is faster in isolation, but every caller needs a
DataFrame-returning API, and the `to_pandas()` copy that requires more than
cancels the gain: **410ms vs pandas' 297ms at 6M rows.** The read win is only
available to a caller that consumes polars natively, and there is none — which
is precisely what the deferred migration would have to change.

## Why the full migration is deferred

Outside the validation package, simACE has almost no pandas *compute* left to
migrate. Counting `groupby`/`merge`/`sort_values`/`value_counts`/`agg`/`concat`/
`pivot`/`apply`/`drop_duplicates`:

| Package | heavy pandas ops |
|---|---|
| `phenotype`, `censoring`, `ascertainment`, `core` | **0** |
| `simulation` | 2 |
| `analysis/stats` | 6 |
| `plotting` | 10 (3 files seaborn-bound) |

Four packages do none at all. Pandas in them is a carrier for reads, writes, and
column access — not an engine. Converting a transport layer to polars while
pandas holds both ends produces a convert-in/convert-out sandwich, which is the
same shape the read benchmark already rejected.

The other decisions compounded this. The writer took the one large win; the read
measurement pinned reads to pandas; keeping the fitACE-facing signatures on
pandas avoided a breaking cross-repo change for a boundary whose load-bearing
member is a single function (`hydrate_trait`, via `fitace/trait_input.py`); and
moving validation — the largest consumer, at 37.45s — onto numpy arrays removed
the biggest DataFrame workload from the question entirely.

**Memory is not an argument either way.** In-memory footprint is identical at
0.390 GB under both libraries — both are Arrow-backed for these dtypes. The RSS
spikes in this pipeline come from `np.unique` temporaries in `pedigree-graph`
(+1.22 GB) and simulation internals (+1.58 GB). Neither is pandas, and neither
would be helped by polars.

Deferred, not rejected: nothing here says polars is the wrong choice, only that
today's code gives it too little to do.

## Revisit triggers

Reopen this when any of the following becomes true. Re-measure rather than
assuming the numbers above still hold.

1. **Pandas compute grows.** If heavy-op counts in `simulation` or
   `analysis/stats` rise materially above the table above — particularly
   `groupby` over multi-million-row frames — the sandwich argument weakens,
   because there is then real work to amortise a conversion against.
2. **A pipeline stage can go polars-native end to end**, read through write, with
   no pandas in the middle. That is the unit at which the read win becomes
   collectable, since the `to_pandas()` copy disappears. Stages are the natural
   granularity: each is read → transform → write.
3. **fitACE wants polars at the boundary.** That reopens C1. The surface is
   small — three frame-returning functions, only `hydrate_trait` on a production
   path — and ADR 0012 lockstep versioning makes the coordinated release routine.
4. **The hard boundaries move.** Seven of the twelve
   `workflow/scripts/simace/tskit/` gene-drop scripts use pandas and six write
   parquet directly; none imports `simace`, so they sit outside this boundary
   entirely. Seaborn binds three plotting modules to pandas. If either
   dependency gains polars support or is dropped, the region polars could own
   grows.
5. **A new large-frame consumer is added** that would be written against a
   DataFrame API from scratch. Greenfield code pays no migration cost, so the
   calculus differs from converting existing code.

## Consequences

A reader encountering a one-function polars dependency, or a `save_parquet` that
writes with one library and reads with another, will find the asymmetry
deliberate and measured rather than half-finished. The cost is that simACE
carries two DataFrame libraries in its dependency set for a single function —
accepted because that function is on the hot path and the win there is 5.8–7.5x.

`nan_to_null=False` in `save_parquet` is load-bearing and easy to lose. Polars
distinguishes NaN from null where pandas conflates them, so the default would
rewrite float NaN as parquet null. A pandas round-trip still looks correct
either way, so **no test catches a regression here** — but the on-disk null mask
changes, which matters for the non-pandas readers of these files (LDAK,
EPIMIGHT's R driver).

`CONTEXT.md` is deliberately untouched. This is implementation, not domain
vocabulary, and that glossary stays a strict domain glossary.
