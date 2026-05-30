# Non-Plotting Code Quality Issues

Findings from the code-quality review of `simace/` (non-plotting). **Triaged
2026-05-30** via a grill-with-docs session against the current code; verdicts
and locked scope are recorded per issue. Execution order is revised at the
bottom.

## Scope

- `simace/simulation/simulate.py`
- `simace/analysis/validate.py`
- `simace/core/relationships.py`
- `simace/core/parquet.py`
- `simace/ascertainment/__init__.py`, `simace/phenotype/__init__.py`

## Issue 3 (KEEP) — Centralize relationship semantics in `core.relationships`

**Verdict: keep, scoped down.** See ADR 0009.

Expected liability correlations and the maternal-vs-paternal-half-sib C-sharing
rule are currently inline literals in `validate.py` and prose in plotting
docstrings. This is scientifically sensitive and cross-repo coupled.

**Locked scope:**
- Add `shared_environment_coefficient(relationship_type) -> float` and
  `expected_liability_corr(relationship_type, A, C) -> float` to
  `simace/core/relationships.py`.
- `expected_liability_corr = 2 * PAIR_KINSHIP[rt] * A + shared_environment_coefficient(rt) * C`.
  Kinship **always** from `pedigree_graph.PAIR_KINSHIP` — never a literal.
- `shared_environment_coefficient`: same-mother types (`MZ`, `FS`, `MHS`) → 1.0;
  `PHS`, `MO`, `FO`, `1C` → 0.0.
- Raise `ValueError` on a type outside `RELATIONSHIP_TYPES`.
- **No** `pooled_relationship_classes()` in core — pooling stays at call sites.

**Acceptance:** validation/plotting stop hard-coding `0.25`/`0.5`/`1.0`; MHS vs
PHS shared-C behavior is tested; coefficients trace to `PAIR_KINSHIP`; helper
names checked against fitACE before finalizing.

## Issue 2 (KEEP) — Split `validate.py` into a validation package

**Verdict: keep.** Lowest-risk, highest-readability win; direct precedent in
`stats/`.

`validate.py` (~1385 lines, ~30 functions) is already cleanly grouped by
domain. Convert to a package mirroring `simace/analysis/stats/`:

```text
simace/analysis/validate/
  __init__.py            # re-exports only
  runner.py              # build_validation_report, run_validation, cli
  structural.py twins.py half_sibs.py consanguinity.py
  statistical.py heritability.py population.py
  assortative_mating.py effective_size.py
  _common.py             # cross-cutting validation helpers
```

**Locked scope:**
- Cross-cutting helpers (`_result`, `_corr_se`, `_corr_tolerance`,
  `_subsample_pairs`, `_extract_comp_vals`, `_MIN_PAIRS_FOR_CORR`,
  `_DEFAULT_RNG_SEED`) → `validate/_common.py`. Subdomain modules depend only on
  `_common`, never on each other (avoids cycles).
- Generic numerics (`safe_corrcoef`, `fast_linregress`) stay in
  `core/numerics.py` — same as `stats/`.

**Acceptance:** public imports still work; existing validation tests pass with
at most import-only edits.

## Issue 6a (KEEP) — `save_parquet` no longer silently mutates callers

**Verdict: keep the mutation fix; atomic writes killed (see below).**

`simace/core/parquet.py::save_parquet()` calls `_optimize_dtypes(df)` which
narrows dtypes **in place**, mutating the caller's DataFrame. Latent today
(every call site is a one-shot tail-end write), but a footgun.

**Locked scope:** copy before narrowing, or rename the helper to make mutation
explicit and document it. No behavior change to written output.

## Issue 5 (KEEP, LOW PRIORITY) — Move implementation out of `__init__.py`

**Verdict: keep as a consistency cleanup only.** Not a bug fix — no concrete
import side effect exists. Justified solely by matching `stats/`, which already
uses a `runner.py` + re-export `__init__` shape.

**Locked scope:**
- `simace/ascertainment/__init__.py` (340L) → `ascertainment/runner.py`.
- `simace/phenotype/__init__.py` (231L) → `phenotype/runner.py`.
- Module name is **`runner.py`** (matching `stats/`), not `run.py`.
- `__init__.py` becomes re-exports + metadata; CLI entry points and script
  wrappers continue to resolve.

## Issue 1 (KEEP, REFRAMED) — Split `simulate.py` around params + assortment

**Verdict: keep, reframed and gated.** `run_simulation()` (~360 lines) is long,
but most of it is already model-agnostic and factored into helpers
(`reproduce`, `_fill_pedigree_slice`, `_init_pedigree_arrays`, `mating`).
The actual mess is two `mating_model == "standard"` blocks: the AM/`rho_w`/PSD
setup (~55 lines) and the per-generation `R_mf` computation inside the loop
(~30 lines). The WF path is two lines.

`simulate.py` is the scientific core — nearly every CLAUDE.md gotcha is a
pedigree/simulation correctness bug — so this trades silent-bug risk for
readability and must be done behind a safety net.

**Locked scope (reduced from the original 4-concept proposal):**
- Extract `SimulationParams` — the validation block (it currently duplicates
  config-load validation because `run_simulation` is also a public test API).
- Extract `AssortmentPlan` — owns *all* standard-only AM logic (per-gen assort
  resolution, `rho_w`, the 4×4 PSD check, per-generation `R_mf`), so both
  `if standard` blocks empty out of `run_simulation`.
- **Drop** the `MatingStrategy` protocol and the two strategy classes — the
  standard/WF asymmetry makes a symmetric interface ceremony; a plain `if`
  dispatch over `_mating_standard` / `_mating_wf` is clearer.
- **Drop** `SimulationState` — passing a mutable state object into `reproduce()`
  introduces aliasing risk in exactly the code that breaks silently.

**Precondition:** a golden-output characterization test (hash a fixed-seed
pedigree for both `standard` and `wright_fisher`) lands **first**, so the
refactor is provably behavior-preserving.

**Acceptance:** `run_simulation()` is substantially shorter with both standard
blocks moved into `AssortmentPlan`; golden test passes unchanged; public API
preserved.

## Issue 4 (KILLED) — Type internal report/stats boundaries

**Verdict: killed.** The report *output* shape is already pinned by
`assert_report_contract()` + `REPORT_SUMMARY_REGISTRY` + `MetricSpec` in
`analysis/report_schema.py`. The remaining looseness is in `report.py`, a
*curation adapter* that hand-maps heterogeneous per-check diagnostics into the
curated report — and that input resists a single type: `report.py` reaches into
named per-check long-tail keys (`n_half_sib_matings`, `total_missing_gp_links`,
`observed_rate`, …) that no one `ValidationCheck` TypedDict can hold. Typing the
bare envelope buys ~30-site churn for negligible benefit. Not worth doing.

## Issue 6b (KILLED) — Atomic YAML/parquet writes

**Verdict: killed as speculative.** The pipeline runs under Snakemake, which
already removes a failed job's outputs on rerun, and several outputs use
`temp()`. Atomic temp-file+rename would only guard against hard kills mid-write
or direct out-of-Snakemake CLI use — a narrow, mostly-theoretical gap with no
recorded incident. Revisit only if a corrupt/partial output actually bites.

## Revised execution order

1. **Issue 3** — relationship semantics (independent, low-risk; see ADR 0009).
2. **Issue 2** — validate package split (independent, low-risk).
3. **Issue 6a** — `save_parquet` mutation fix (tiny).
4. **Issue 5** — `runner.py` extraction (low priority, consistency only).
5. **Issue 1** — `simulate.py` split (last; golden test first, then extract
   `SimulationParams` + `AssortmentPlan`).

## Side-fixes surfaced during triage (independent of the above)

- CLAUDE.md gotcha #4 cites `simace.core.pedigree_graph` for `PAIR_KINSHIP`;
  the real import is the top-level `pedigree_graph` package. Stale path — fix.
- 16 validation result dicts bypass `_result()` (no `passed` key) — a
  deliberate "informational" variant. Noted for whoever next touches them.
