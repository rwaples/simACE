# ADR 0019: Null raw onsets mean never-onset at the censor boundary

## Status

Accepted. Decision approved 2026-08-25 during the cross-stage pipeline
property-testing initiative.

## Context

ADR 0015 defines how missing values are represented: parquet null on disk and
null in polars frames, with NumPy NaN permitted only transiently during
computation. It does not define what a missing raw onset means when the censor
stage derives age-censoring, death-censoring, and affected status.

The raw-trait schema permits null `t1` / `t2`, and the phenotype stage
normalizes model-produced NaN to null. Before this decision, a null became NaN
inside `run_censor`; comparisons against age-window and death boundaries were
all false. The row therefore emerged as affected with a null observed time,
violating the censor identity's intended semantics.

Current in-tree models normally represent no onset with the finite sentinel
`1e6`, while hazard inversion also uses `1e6` as a saturation ceiling. The
sentinel is therefore an implementation convention, not a universal semantic
synonym for never-onset.

## Decision

At the censor boundary, a null raw onset means **never-onset**. `run_censor`
materializes null `t1` / `t2` values as positive infinity only in the arrays
used to derive censoring columns. The original raw onset columns remain null in
the outcomes-only result.

Under any finite observation window, the existing censor arithmetic then marks
the row as age-censored, not affected, and gives it a finite observed time at
the applicable right boundary or earlier death age.

Null and a finite sentinel have equivalent derived censoring columns only when
the sentinel is strictly greater than the applicable right boundary for every
replaced row. In particular, `1e6` is not equivalent to null under a window
whose right boundary is `1e6`.

## Considered alternatives

- **Reject null raw onsets.** Rejected because the raw-trait and upstream
  phenotype contracts permit them; the censor stage must define an
  interpretation for valid input.
- **Preserve the previous comparison behavior and document it.** Rejected
  because it marks a missing onset as affected while producing no finite
  observed onset.
- **Rewrite nulls to the finite `1e6` sentinel.** Rejected because `1e6` can be
  inside an inclusive observation window and is also a valid saturation value
  from hazard models.

## Consequences

- Null raw onsets are always unaffected after censoring under finite windows.
- Their derived observed times remain finite, while their raw `t1` / `t2`
  values remain null for auditability and ADR 0015 compliance.
- Tests may compare null handling with `1e6` only after proving every relevant
  right boundary is strictly below `1e6`.
- The outcomes-only schemas and parquet representation do not change.
