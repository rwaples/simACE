# Polars migration — family-wide plan

**Status:** draft, awaiting go-ahead. Supersedes the deferral in ADR 0014.
**Scope decision:** polars-primary in the 10 frame-owning Python repos;
`pedigree-graph` stays frame-library-neutral; pandas survives only at forced
third-party boundaries. Locked in a grill-with-docs session
2026-08-13; decision log below.

## Decision log (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Scope | **B — polars-primary.** Polars is the default frame type everywhere frame-owning code is written and maintained; `pedigree-graph` remains frame-neutral over NumPy kernels. Pandas survives only where a third party forces it: seaborn plot modules (3 files family-wide), the pandas-native tstrait/tskit scripts, and edge shims. The fitACE frame boundary **breaks to polars** under lockstep (ADR 0012). |
| 2 | Public non-lockstep repos | **2b — frame-neutral pedigree-graph; polars-primary pedsum.** `pedigree-graph` is already NumPy/scipy internally and keeps both pandas and Polars out of runtime dependencies. Its small structural frame protocol requires `.columns`, `__getitem__`, and column `.to_numpy()`; `dict[str, np.ndarray]`, Polars, and pandas inputs remain accepted by all three constructors. Both frame libraries live only in its test extra. Existing kernels/outputs stay unchanged. `pedsum` migrates internally behind its unchanged CLI/generated-file contract. Implementation adds `Unreleased` changelog entries; independent version bumps/tags occur at the external release gate. |
| 3 | Strategy | **3b — staged waves, parquet files as seams.** simACE pipeline stages communicate through parquet on disk, so stages flip independently without inter-stage frame-conversion shims. Wave 1 = simACE internals stage-by-stage; Wave 2 = the cross-repo boundary break; independent track = pedigree-graph + pedsum (pedigree-graph early). |
| 4 | Missingness | **4c — missing = parquet null on disk and null in Polars DataFrames.** NumPy/SciPy compute may transiently materialize null floats as NaN; normalize them when results re-enter a frame. See "Null contract" below. Confirmed riders: write-edge `fill_nan(None)` self-enforcement; null-mask regression test; non-nullable-int schema freeze; stale local results regenerated; EPIMIGHT TSV reads use `null_values=["NA", ""]`. |
| 5 | Tests & scripts | **5b — tests and workflow scripts migrate in the same commit as the module they cover.** Frame assertions via `polars.testing.assert_frame_equal`. Pandas remains only in: pedigree-graph's focused compatibility tests, seaborn/tstrait boundary tests, and one core round-trip compat test pinning the null contract across both libraries. |
| 6 | Mechanics | Per-wave verification bar (below); **no `FAMILY_FLOOR` bump, tag, or push during implementation**; ADR 0015 supersedes 0014; dormant repos (fitACE_stan, fitACE_frailty) last, suites-green bar only. The next coordinated family release performs the floor bump immediately before local lockstep tagging and reinstall/verification. |
| 7 | Transitional frame contract | **Temporary dual-frame core support through Wave 2.** `save_parquet` and `assert_schema` accept pandas and Polars while stages migrate independently. Migrated wrappers use `load_parquet`; unmigrated simulation retains pandas. `hydrate_trait`/`create_sample` return the same frame type as their input. Because `run_simulation` has no input frame and fitACE uses it in fixtures, simulation migrates only in the coordinated Wave 2 break. Remove pandas support afterward. Parquet remains the inter-stage seam. |
| 8 | Primary justification | **Maintainability and API standardization, not performance.** Establish one primary DataFrame API across maintained family code and remove cross-repo frame-boundary inconsistency. ADR 0014's performance findings remain valid evidence; benchmarks are regression guardrails and a hold point, not a promised speedup. Structural success means pandas imports and conversions are confined to an explicit allowlist. |
| 9 | Dependency policy | Polars is a base dependency in the 10 frame-owning Python repos; `pedigree-graph` remains runtime frame-library-neutral. Pandas is never a **direct** base dependency after migration. Unavoidable transitive installation is permitted when a required third party (for example cmdstanpy) owns it; third-party pandas frames convert immediately via `pl.DataFrame(...)` without a pandas import. simACE declares pandas explicitly in `plot`, `workflow`, and `test`; fitACE core in `plot` and `test`; method sisters only in extras that directly import/test pandas. `pedigree-graph` keeps pandas in `test`; pedsum removes it. Direct-dependency and source checks prevent pandas returning to internals; transitive owners are inventoried/reported. |
| 10 | Shared core frame helpers | `hydrate_trait` and other frame-returning core helpers become temporary same-type dual-frame APIs early in Wave 1: Polars input returns Polars, pandas input returns pandas, and mixed-library inputs fail clearly. Migrated simACE callers therefore become Polars-native while fitACE remains compatible. Wave 2 removes pandas acceptance atomically with downstream migration. |
| 11 | Execution model | **Eager-only migration.** Public and stage APIs accept/return `pl.DataFrame`, never `pl.LazyFrame`; `load_parquet` uses `pl.read_parquet`. The stage decorator enforces stage input/output types; changed public cross-repo APIs check their boundaries. Private helpers rely on annotations/tests. Lazy scanning/streaming is a separate benchmark-driven follow-up; no general frame-validation framework. |
| 12 | Auxiliary tracked code | Inventory source and third-party frame boundaries per repo immediately before that repo's migration, not family-wide in Wave 0. Maintained examples/operational scripts migrate with owners; production-representative harnesses migrate; comparative benchmarks may retain pandas. Retire obsolete tools and record rationale for archival/scratch exclusions. Run one global source-policy check at completion. |
| 13 | Version floor | The 10 frame-owning repos standardize on the tested floor `polars>=1.43.2,<2`. Run normal per-repo verification in the active environment and one clean cross-repo install at completion; no per-repo minimum-version matrix. Lowering the floor later requires evidence. Retained pandas boundary versions are recorded separately. |
| 14 | Seeded reproducibility | Preserve scientific sampling exactly: ascertainment/dropout and `create_sample` keep their existing NumPy RNG/row-position logic and selected IDs for a fixed seed. Only plotting-only pandas `.sample(...)` sites may adopt Polars sampling and change IDs; if they do, retain fixed-seed determinism and document the narrow plotting-sample change. Scientific artifacts/reports remain exactly comparable. |
| 15 | Index semantics | DataFrame index is not public or semantic after migration; durable identity/order lives in explicit columns/row positions. Audit index idioms per module immediately before conversion, adding tests where identity/order/labels are observable. Wave 0 inspects only shared/public APIs for user-visible index behavior; a final structural search catches unexplained leftovers. |
| 16 | Plot verification | Preprocessing stays Polars-native; only minimal frames passed directly to seaborn convert to pandas. Snapshot final plot-input tables, force-regenerate every affected atlas, verify manifest order/text/axes/legends, and inspect HTML/representative images for clipping and category-order changes. Pixel identity is not required, but semantic visual changes need explanation. |
| 17 | Null operation semantics | No blanket pandas-skipna equivalence claim. For each converted operation, identify nullable columns it actually touches and compare one representative missing-data fixture. Add targeted edge tests only for reachable/statistically important library differences; pin explicit statistical parameters. Shared helpers own reusable null/reduction tests. Non-null join keys still fail before joining. |
| 18 | Working-tree isolation | Never develop a wave around unrelated dirty files. Wait for the owner or use dedicated worktrees from recorded commits; never stash/reset/overwrite unrelated work. Each migration worktree contains only intended diffs. “Commit-set” is a logical cross-repo unit; commits happen only when explicitly requested. Clean-install/cross-repo tests target the exact worktrees, not incidental editable installs. |
| 19 | Thread policy | **Measure before changing.** Wave 0 compares representative single/concurrent Snakemake jobs at matched allocations. Only if material Polars oversubscription/runtime/RSS regression is observed, add the smallest startup-level `POLARS_MAX_THREADS={threads}` fix and a focused subprocess test. Otherwise workflow plumbing stays unchanged. Standalone CLIs retain defaults. |
| 20 | Text outputs | Check only text writers changed by migration; do not build formal text-contract objects. Preserve filenames, delimiter/header, column order, parsed values/nulls, and explicit interface tokens. Byte equivalence is reserved for external-tool representations that require it (notably LDAK). Compare gzip after decompression unless compressed bytes are explicitly contractual. Shared writer helpers carry focused edge-case tests. |
| 21 | User-facing API break | Wave 2 removes lockstep-family pandas compatibility immediately—no long-lived shim. Publish a migration guide covering changed functions/types, index→column replacements, null semantics, and any plotting-sample seed change. Update docs/examples/type hints. Changed public cross-repo APIs reject pandas early with an actionable `TypeError`; private helpers rely on annotations/tests. One API-surface test per public function family is sufficient. Pedigree-graph's structural frame compatibility remains separate. |

## Grounding facts (verified this session)

- **Census (git-tracked files importing pandas / heavy ops / polars):**
  simACE 84/68/1 · fitACE 36/11/0 · epimight 33/41/0 · pcgc 21/12/0 ·
  iter_reml 10/1/0 · tetraher 7/3/0 · pafgrs 13/0/0 · stan 5/4/0 ·
  frailty 3/0/0 · pedigree-graph 9/4/0 · pedsum 28/17/0.
- **The null-contract inversion.** Measured (pandas 3.0.5, polars 1.43.2):
  `pd.to_parquet` writes float NaN as parquet **null** (null_count=1);
  `pl.from_pandas(..., nan_to_null=False).write_parquet` writes literal NaN
  (null_count=0). Every pandas-era file already carries nulls. ADR 0014's
  `nan_to_null=False` (added yesterday) *changed* the historical contract
  rather than preserving it. Decision 4c is therefore a **restoration**, and
  the only NaN-encoded files are locally regenerated, unpushed test results.
- **EPIMIGHT exposure is a no-op at the parquet edge.** The R driver reads
  `pipeline_input.parquet` via `arrow::open_dataset`
  (`fitACE_epimight/R/run_pipeline.R:67`), but the emitter
  (`fitace_epimight/create_input.py:150-165`) writes only non-nullable
  int8/int16/bool/str columns — no missing values exist in that file.
- **EPIMIGHT's real trap is the TSV read-back.** The R→Python results handoff
  is TSV (`atlas_io.py`, `_draws_common.py`, `bias_analysis.py`, `cli.py`,
  `plot_onset_censoring.py`); R writes missing as the string `NA`. pandas
  `read_csv` parses that as missing by default; **polars does not** — every
  migrated TSV read needs explicit `null_values=["NA", ""]`, verified against
  what the R driver actually emits.
- **LDAK never reads parquet** — fitACE exports text
  (`fitace/kinship/export.py:76,130`). Out of scope per user instruction;
  the text-export edge just keeps rendering missing identically.
- **Seaborn surface:** `simace/plotting/plot_liability.py`,
  `simace/plotting/plot_validation.py`, and
  `fitace/plotting/plot_observed_binary.py`. Each gets `.to_pandas()` at the
  plot-call edge. `fitACE_stan/fitace_stan/fit_ace.py` has only a commented-out
  seaborn import; it is not a forced pandas boundary and migrates with the
  dormant repo if retained. Retiring/redesigning that scratch module is a
  separate cleanup decision.
- **fitACE frame boundary is three functions:** `hydrate_trait` (production,
  via `fitACE/fitace/trait_input.py:323`), `create_sample` (one call site),
  `run_simulation` (test fixtures only). Six other entry points already
  return `dict`/`None`.
- **tskit scripts** (12; 7 pandas, 6 write parquet, 0 import simace) stay
  pandas per B — and since pandas already writes nulls, their outputs conform
  to the 4c contract as-is. No shim needed.
- **`PedigreeGraph`** already accepts `dict[str, np.ndarray] | pd.DataFrame`
  (`external/pedigree-graph/pedigree_graph/_core.py:197`), but its production
  computation is already NumPy/scipy. The package has no runtime pandas import:
  pandas appears under `TYPE_CHECKING`, frame extraction uses `.values`, and
  pandas is already confined to the `test` extra. Most reported pandas imports
  are tests/benchmarks. This track therefore generalizes the frame boundary and
  migrates fixtures/benchmarks rather than rewriting computational internals.

## Null contract (decision 4c, spelled out)

1. On disk: missing = parquet null. `save_parquet` drops `nan_to_null=False`.
2. In Polars DataFrames: missing = null, not literal NaN. Equivalence with
   pandas missing-value behavior is operation-specific and must be established
   explicitly; do not assume generic skipna parity.
3. At explicit Polars→NumPy/SciPy compute boundaries, null floats may
   transiently materialize as NaN (for example, `Series.to_numpy()`). Normalize
   NaN back to null whenever arrays re-enter a Polars DataFrame.
4. Write edge self-enforces with a lean policy: adapt the existing name-based
   `_optimized_dtypes` mapping to Polars, range-check only columns it narrows,
   run `fill_nan(None)` before writing, and never mutate the caller. There is no
   production post-write readback or generalized artifact-contract framework.
5. Focused regression tests read representative canonical artifacts through
   pyarrow and assert physical dtypes, required structural columns have no
   nulls, missing floats are parquet null, and zero literal NaN exists on disk.
   Cover pedigree, trait, EPIMIGHT's existing int8/int16 schema, and one
   representative tskit-script output. Test overflow and expected float32
   rounding. A broader formal artifact-schema system is a separate future
   feature, not part of this migration.
7. YAML/report scalar missingness is outside this parquet/DataFrame contract
   and retains its existing behavior unless separately specified.
8. Local results regenerated after the writer change (snakemake-regenerable;
   nothing NaN-encoded was pushed).

## Waves

### Wave 0 — contract + plumbing (simACE core)

1. Before edits, record exact source commits for all 13 repos and establish
   clean migration worktrees for every touched repo (or wait for unrelated
   dirty work to be resolved). Never stash, reset, overwrite, or include work
   that is not ours. Then capture representative stage runtime and peak-RSS
   baselines and representative Arrow schemas/null counts for canonical
   pedigree, trait, EPIMIGHT, and tskit outputs (including
   censoring/missingness). Baseline simACE tests before Wave 0; baseline each
   external repo immediately before its track; baseline the nine Python
   lockstep repos immediately before Wave 2. Store commands/results under
   `plans/polars-investigation/`. Then re-run `bench_polars.py` and
   `bench_verify.py`.
   Re-confirm ordering assumptions per ADR 0014's own instruction. Measure
   baseline variance and agree the regression tolerance before Wave 1 rather
   than inventing a threshold in advance.
2. **Hold point:** material runtime/RSS regression triggers redesign or explicit
   acceptance before migration proceeds. Performance is a guardrail, not the
   migration's justification or a promised speedup.
3. Inventory simACE tracked Python by package, tests, workflow, docs example,
   operational script/tool, benchmark/profiler, or scratch/archive, plus its
   third-party frame boundaries. Assign maintained files to waves, migrate
   production-representative harnesses, preserve explicitly comparative
   pandas-vs-polars benchmarks, retire obsolete tools such as
   `tools/pandas3_rehearsal.py`, and record rationale for archival/scratch
   exclusions. Repeat this inventory locally for each external/fitACE repo
   immediately before its own migration; record transitive pandas owners and
   convert third-party frames immediately without importing pandas.
4. Inspect shared/public APIs for user-visible pandas index behavior and define
   explicit replacement columns where needed. Defer internal `.index`/`.loc`/
   `.iloc`/`set_index`/`reset_index` audits to the owning module's migration;
   do not build a permanent all-site classification inventory.
5. Record the frame-owning-repo target `polars>=1.43.2,<2` and current per-repo
   bounds; update simACE only in Wave 0. Each other frame-owning repo adds the
   target bound in the
   same logical change set as its code migration, and removes pandas from base
   dependencies only when its production migration is complete. Add retained
   pandas extras before their tests/boundaries need them. Record exact active
   Polars and retained-boundary pandas versions in baseline artifacts.
6. Write **ADR 0015** (`docs/adr/0015-polars-primary-dataframe-library.md`)
   superseding 0014 (0014 gains a `Superseded by ADR 0015` status line;
   its measurements referenced, not repeated). Record the maintainability/API
   standardization rationale, B-scope, stage-seam waves, the 4c restoration +
   `nan_to_null=False` finding and surviving pandas boundaries. If plotting-only
   sampling IDs change, record that narrow reproducibility note in ADR/release
   notes; scientific ascertainment remains unchanged.
7. As each repo migrates, inventory only TSV/CSV writers actually changed.
   Preserve filenames, delimiter/header, column order, parsed values/nulls, and
   explicit interface tokens such as `NA`. Require byte equivalence only where
   an external parser/tool needs exact representation (notably LDAK); compare
   gzip after decompression unless compressed bytes are explicitly contractual.
   Internal diagnostics/benchmarks compare semantically. Put formatting/null/
   quoting edge cases in shared writer-helper tests rather than every call site.
   Do not build text- or parquet-contract objects.
8. Benchmark representative single-job and concurrent-job Snakemake execution
   at matched allocations, recording CPU utilization, runtime, and peak RSS.
   Only if material Polars oversubscription/regression is observed, add the
   smallest startup-level `POLARS_MAX_THREADS={threads}` fix plus a focused
   subprocess test; otherwise leave workflow plumbing unchanged.
9. `simace/core/parquet.py`: writer flips to the null contract (drop
   `nan_to_null=False`, add `fill_nan(None)`), adapts the existing name-based
   dtype mapping to Polars with range checks for narrowed integers, and adds a
   polars-returning loader (`load_parquet` → `pl.DataFrame`). There is no
   production post-write readback. During Wave 1, `save_parquet` accepts both
   pandas and polars frames; migrated wrappers use `load_parquet`, while
   unmigrated wrappers retain `pd.read_parquet`. Migrated stage code must use
   the core loader/writer — no raw `pl.read_parquet`/`write_parquet` outside
   core. `load_parquet` uses eager `pl.read_parquet`; it does not expose
   `LazyFrame`/`scan_parquet`.
10. Make `assert_schema` understand both pandas and polars logical dtypes for
   the Wave 1 transition; do not overload it with parquet physical-schema or
   nullability checks.
11. Focused null-mask + pandas↔polars round-trip tests covering canonical
   pedigree/trait outputs, EPIMIGHT widths, overflow/underflow, required-column
   nulls, NaN normalization, float32 rounding, and caller-frame non-mutation.
12. Regenerate `results/test/small_test/`; full simACE bar (below). If the
   Wave 0 concurrency benchmark triggered a thread-limit change, include its
   focused concurrent-job regression check.

### Wave 1 — simACE internals, stage by stage (one commit-set per stage)

Order: `core` remainder → `phenotype` (transport-only, establish the idiom) →
`censoring` → `ascertainment` → `analysis/stats` → `plotting` (seaborn shims
last). Simulation stays pandas until Wave 2: `run_simulation`
has no input frame for same-type dispatch, and fitACE consumes it in fixtures. Early in `core`, inventory every
frame-returning helper. `hydrate_trait`, `strip_trait_to_outcomes`, and any
similar helper shared with unmigrated or downstream code become temporary
same-type dual-frame APIs: Polars input returns Polars, pandas input returns
pandas, and mixed-library inputs fail clearly rather than converting
implicitly. `analysis/validate` is already library-agnostic (PedigreeArrays) —
only its 8 groupbys in `statistical.py` / `population.py` convert.

Each stage commit-set includes: module + its tests + its workflow scripts
(5b); a local audit of its pandas index idioms; verification bar; direct
before/after artifact equivalence; and, for
compute-touching stages, a check of nullable columns actually touched, targeted
edge tests where behavior can differ, and report-level equivalence as an
additional safeguard.
The plotting commit-set additionally snapshots each final plot-input table,
converts only the minimal seaborn-call frame to pandas, force-regenerates the
scenario and validation atlases, and inspects HTML/representative images for
manifest order, titles/captions, axes/legends, clipping, and category order.
Pixel identity is not required, but semantic visual changes need explanation.
Keep pandas acceptance in `save_parquet`, `assert_schema`, and same-type shared
helpers through Wave 2 because simulation remains pandas. Remove it only after
the coordinated simulation/downstream boundary break.

### Wave 2 — the boundary break (coordinated, one commit-set across repos)

1. Migrate the simulation stage and `run_simulation` to Polars together with
   its fitACE fixture consumers. Remove temporary pandas acceptance from
   `hydrate_trait`, `strip_trait_to_outcomes`, `create_sample`, `save_parquet`,
   `assert_schema`, and other shared frame helpers; make signatures Polars-only.
   Add early actionable `TypeError`s at changed public boundaries; do not
   auto-convert. Remove simACE's base pandas dependency after this coordinated
   change and add focused API tests.
2. fitACE core + all seven sisters convert their consumption in the same
   coordinated commit set. Publish a migration guide listing each changed
   public function's old/new accepted and returned types, index→column changes,
   null/NaN semantics, and any narrow plotting-sample reproducibility change;
   update docstrings,
   annotations, examples, and generated API docs. Keep pedigree-graph's
   separate structural frame-compatibility promise explicit. EPIMIGHT track:
   emitter goes polars (schema
   assertion per null contract §5); all TSV reads gain
   `null_values=["NA", ""]`.
3. Leave `FAMILY_FLOOR` and family dependency pins unchanged during
   implementation: no unreleased version can satisfy a raised floor under
   setuptools-scm. Record the boundary break as requiring the next coordinated
   family release. That release separately bumps the floor and pins, commits
   them, creates local lockstep tags, reinstalls, and verifies via the
   coordinated-release procedure. No tag or push in this migration.
4. Dormant repos (fitACE_stan, fitACE_frailty) convert last, suites-green
   bar only.
5. Update family pedigree-graph Git pins only after the migrated external tag is
   published at the release gate; then run clean-install verification.
6. For every changed plotting module, snapshot the final plot-input table and
   force-regenerate every affected fitACE/method-sister atlas (observed-binary,
   EPIMIGHT/bias/onset-censoring, or PA-FGRS as applicable). Verify manifest
   order/text/axes/legends and inspect HTML/representative images. Generate PDF
   only when affected render/export code requires it.
7. `repo-status --tests` final 13-repo health sweep. Wave 2 is not complete
   until the external release gate has been performed.

### Independent track — pedigree-graph, then pedsum (start early)

1. `pedigree-graph`: interface/test migration only; computational kernels are
   already NumPy/scipy and stay untouched, with no pandas or Polars runtime
   dependency. Define one local structural frame `Protocol` requiring
   `.columns`, `__getitem__`, and returned columns with `.to_numpy()`; continue
   accepting `dict[str, np.ndarray]`. Preserve `PedigreeGraph.__init__`,
   `from_dataframe`, and `from_subsample`; keep `from_dataframe` as a
   compatibility name and update annotations/docs. Use Polars for ordinary
   fixtures/benchmarks, retain focused pandas compatibility coverage (including
   nullable integers), and test all three entry points plus dictionary inputs.
   Existing NumPy/scipy outputs
   stay unchanged. Add an `Unreleased` changelog entry; no version bump/tag in
   the implementation commit.
2. `pedsum`: migrate against the exact local pedigree-graph migration worktree,
   behind its documented CLI/generated-file contract, with direct equivalence
   for its CLI artifacts. Its empty `pedsum.__init__` exports no frame-typed
   Python API. Add an `Unreleased` changelog entry; defer static version/pin
   bumps and tags to the final external release gate.
3. Develop simACE/fitACE callers against the same recorded pedigree-graph
   commit, passing Polars or the NumPy-dict escape hatch so the family does not
   exercise pandas compatibility.
4. **Final external release gate:** once all implementation/tests are ready, the
   maintainer releases pedigree-graph first; updates pedsum/family pins to that
   tag; bumps/releases pedsum; and runs clean-install verification. Tags/pushes
   are not migration actions, but Wave 2 remains incomplete until released pins
   verify.

## Per-wave verification bar

- Each module/commit-set: targeted pytest; `ruff check` (no extra `--select`);
  `ruff format --check`; `ty check --ignore all --error unresolved-import`;
  focused artifact/behavior comparison; and an import smoke after bulk edits
  (naive replacements can match a prefix of a longer import line — ruff won't
  catch it). Benchmark before/after at identical allocations. Add thread-pool/
  concurrency checks only if Wave 0 measurement justified a
  `POLARS_MAX_THREADS` change.
- Each completed pipeline-stage conversion: relevant integration tests and a
  run against frozen upstream artifacts. End of Wave 0 and Wave 1: full simACE
  pytest plus the forced small-test Snakemake scenario. Re-run a full suite
  mid-wave only when impact analysis or failures justify it.
- Every migrated stage: **direct artifact equivalence** on before/after
  parquet and covered text outputs. For parquet, compare column names/order,
  Arrow physical dtypes, row count and
  emitted row order, exact discrete values, and null masks. Transport-only
  floats compare exactly; genuinely recomputed floats use tolerances agreed
  before that stage begins. Wave 0's intended NaN→null restoration is the one
  scientific-artifact exception. Ascertainment/dropout and `create_sample`
  retain their NumPy RNG logic and exact fixed-seed selected IDs. Plotting-only
  sampling may change IDs; compare its deterministic caps/semantics rather than
  row identity. For changed TSV/CSV,
  preserve filenames, delimiter/header, column order, parsed values/nulls, and
  explicit tokens. Compare bytes only for exact external-tool representations;
  otherwise compare parsed semantics, decompressing gzip first. Shared helper
  tests cover relevant formatting/null/quoting edge cases.
- Plotting changes additionally snapshot final plot-input tables, force-build
  every affected atlas, verify manifest order/text/axes/legends and artifacts,
  and inspect HTML/representative images for clipping and category-order
  changes. Deterministic inputs compare semantically exactly; plotting-only
  sampled inputs may differ in row identity but retain deterministic caps and
  semantics. Pixel identity is not required.
- Compute-touching stages additionally get **report-level equivalence** on
  identical frozen inputs. Generate baselines from the recorded clean baseline
  worktree/commit and migrated outputs from the dedicated migration worktree;
  never use `git stash`. Scientific ascertainment keeps exact fixed-seed IDs, so
  report diffs remain exact subject to explicitly agreed numeric tolerances.
  Reports are higher-level safeguards, not substitutes for row-level checks. Store comparison commands,
  versions, commits, manifests, and results under
  `plans/polars-investigation/`; reruns must not alter either worktree.
- End of Wave 0 and Wave 1:
  `snakemake --cores 4 -F results/test/small_test/scenario.done`. Treat any
  DAG/job count as a baseline only after verifying it at the start of that
  wave. Wave 2 runs each migrated repo's full suite once after conversion plus
  focused cross-repo integration tests.
- Tests use track-local baselines: simACE before Wave 0, each external repo
  before its track, and the nine Python lockstep repos immediately before Wave
  2. Require zero failures and explain intentional skip/count changes without
  treating exact pass totals as a durable contract. Refresh only a track whose
  source commit changed.
- At final completion, run the 13-repo
  `.agents/skills/repo-status/scripts/repo-status.sh --tests` sweep once.
- Structural migration check covers all tracked Python, not only packages. It
  fails on pandas imports/conversions outside documented seaborn, tstrait/tskit,
  compatibility-test, third-party edge shim (for example immediate cmdstanpy
  frame conversion), explicitly comparative benchmark, and named
  archival/scratch boundaries. It also reports residual pandas-index idioms;
  every retained occurrence must be allowlisted rather than accidentally
  emulated.
- Verify each migrated repo in the active environment. At final Wave 2/
  external-track completion, run one clean cross-repo install and require every
  frame-owning repo to declare the same `polars>=1.43.2,<2` base bound;
  pedigree-graph
  remains free of runtime frame-library dependencies. Polars is a base
  dependency in those 10 repos;
  pandas is absent from every **direct** base dependency set. Unavoidable
  transitive installation is allowed where a required third party owns it and
  is inventoried/reported separately. Repos declare pandas explicitly only in
  extras that directly import/test it: simACE's `plot`, `workflow`, and `test`;
  fitACE core's `plot` and `test`; a method sister's specific forced-boundary
  extra where needed; and `pedigree-graph`'s `test`. pedsum has no direct pandas
  dependency. Conda/environment files encode the same direct policy. Retained
  pandas boundary versions are recorded separately rather than treated as a
  family-wide runtime contract. Pedigree-graph declares both Polars and pandas
  only in its test extra.

## Conversion gotchas (checklist for every stage)

- Do not add Polars thread plumbing without measured oversubscription. If Wave
  0 justifies it, set `POLARS_MAX_THREADS` before import (process-startup state)
  and verify with one focused subprocess/concurrency test; never resize after
  import. Standalone CLIs retain user/default behavior.
- Eager `pl.DataFrame` is the public/stage frame type. Enforce it centrally in
  the stage decorator and at changed public cross-repo APIs; private helpers
  rely on annotations/tests. Internal lazy work, if any, collects before return.
- Define ordering per operation; never use blanket post-join key sorting. For
  left joins whose contract preserves input order (notably `hydrate_trait`),
  carry a temporary row ordinal through the join and restore/remove it, or use
  verified `maintain_order="left"` behavior pinned by a regression test. Sort
  by keys only where the existing API/artifact already requires canonical key
  order. For `group_by`, preserve first-observed group order only where pandas
  currently does; otherwise sort by documented output keys. Add shuffled-input
  tests for `hydrate_trait`, ascertainment, sampling, and other order-sensitive
  helpers. Preserve exact fixed-seed IDs for ascertainment/dropout and
  `create_sample` by retaining their NumPy position-selection logic.
  Plotting-only samples may change IDs if they remain deterministic and preserve
  caps/semantics. **Artifact equivalence still depends on ordering.**
- Missing-value behavior is operation-specific. For each changed operation,
  test a representative reachable missing-data case; add focused edge coverage
  when pandas/Polars defaults differ or the statistic is important. Pin
  variance/standard-deviation and other statistical parameters explicitly.
  Shared helpers carry reusable null/reduction tests. Required join keys fail
  on null before joining.
- Changed text writers preserve focused interface expectations. Where exact
  external-tool formatting requires pandas-style significant digits (for
  example `%.6g`/`%.10g`), pre-format those columns or use a small
  standard-library writer; do not retain pandas for convenience. LDAK export
  bytes remain unchanged; other outputs generally compare as parsed data.
- `null_values=["NA", ""]` on every CSV/TSV read of R output (EPIMIGHT).
- `.apply` → expressions, never `map_elements`, on hot paths.
- `value_counts` → `group_by(...).len()`; no index — joins on columns.
- Polars errors on integer overflow where numpy may wrap. Introduce no new
  compute/in-memory narrowing below int32/float32. Existing narrower storage or
  external-contract widths (for example pedigree `sex` int8 and EPIMIGHT
  int8/int16 fields) are preserved by focused write-edge mappings/assertions
  with range tests. Any new narrow
  artifact field needs explicit justification and overflow coverage.
- DataFrame index carries no implicit identity/order after migration. During
  each module conversion, map ID lookup, row position, boolean selection, and
  output labels to explicit columns/arrays; test shuffled/non-contiguous inputs
  where behavior is observable. Public return indexes that carried information become
  named columns and are documented as breaking changes.
- pandas fixtures in tests: migrate with the module (5b), assert via
  `polars.testing.assert_frame_equal`.
- `plans/` is gitignored — promote this file to `docs/plans/` when finalized.
- fitACE + fitACE_epimight working trees hold uncommitted work that is not
  ours — leave it untouched. Wave 2 waits for resolution or uses dedicated
  worktrees from explicitly recorded commits; never stash/reset or "work
  around" unrelated edits in place. “Commit-set” means a logical coordinated
  unit across repos; actual commits occur only when explicitly requested.
  Clean-install and cross-repo verification target those exact worktrees, not
  incidental editable installs.

## Out of scope

- LDAK / text-export edge behavior changes (user instruction, 2026-08-13).
- tskit/tstrait script conversion (pandas-native dependency; conforms to the
  null contract as-is).
- Dropping seaborn (would be the ADR 0014 trigger-4 follow-up, not this plan).
- Any `git push` or release tagging. The maintainer-run external release gate
  is a prerequisite for completing Wave 2 but is not executed as part of the
  migration.
