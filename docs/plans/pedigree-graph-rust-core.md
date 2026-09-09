# pedigree-graph Rust core and public API redesign

Status: APPROVED — recorded as pedigree-graph ADR 0006 (public API and coordinate
semantics, `external/pedigree-graph/docs/adr/0006-public-api-and-coordinate-semantics.md`)
and ADR 0007 (Rust core, host boundary, and release,
`external/pedigree-graph/docs/adr/0007-rust-core-host-boundary-and-release.md`).
Implementation awaits the prerequisite issues listed below. Supersedes
`plans/pedigree-graph-rust-core.md`.

The existing Rust pair-engine spike remains useful evidence, not the implementation
plan. It is committed on branch `rust-spike` in
`external/pedigree-graph-rust-spike` at `659aa0c`. It matches the current Python
pair sets on fixtures, inbred random pedigrees, and simulated pedigrees through
300,000 rows, but does not cover the redesigned API, graph views, semantic pair
orientation, arbitrary input order, bindings, kinship, or R.

## Goal

Create one host-neutral Rust implementation of pedigree structure, relationship
classification, kinship, inbreeding, lineage primitives, and effective-size
prerequisites. Python remains the primary host and gains a deliberately redesigned
public API before native migration begins. A small R package follows after the
Python/Rust boundary stabilizes.

The migration optimizes for:

1. scientific correctness and explicit semantics;
2. one canonical production implementation per migrated operation;
3. safe Python/R embedding;
4. no material wall-time or memory regression; and
5. maintainable host-specific representations without host dependencies in the core.

## Non-goals

- Preserving the current public API through deprecation shims.
- Publishing the Rust core on crates.io before 1.0.
- Porting the experimental BFS engine to Rust.
- Full Python/R capability parity in the first R release.
- Changing effective-size estimator formulas as part of the language migration.
- Assuming that a Rust port alone improves the output-dominated kinship DP.

## Grounded reasons for the redesign

- `from_subsample` stores a full graph while returning relationship pairs in caller
  coordinates (`pedigree_graph/_core.py:880-943`); kinship matrices remain in graph
  coordinates. This hidden split caused PGQ-001.
- Public `mother`/`father` arrays are remapped row coordinates despite names that
  look like original IDs (`pedigree_graph/_core.py:244-248`).
- Supplied `generation` is currently used as DP traversal depth
  (`pedigree_graph/_kinship_dp.py:104-123`), even though structural depth is derived
  separately elsewhere.
- Pair extraction currently canonicalizes orientation during view/subsample remap
  (`pedigree_graph/_pair_utils.py:124-153`), so lineal orientation depends on the
  construction path.
- `min_kinship` on the DP is propagation pruning, not an exact final-value filter;
  ADR 0005 and `fitace/kinship/kinship.py:15-21` document a concrete inbred
  counterexample.
- The streaming counter mixes exact and approximate categories under an
  implementation-oriented name; its handling lives in `REL_PLAN`.
- Descendant counts are path counts while ancestor counts are distinct counts
  (`tests/test_n_descendants.py`, `tests/test_n_ancestors.py`).
- fitACE reaches into `_Am`/`_Af` for connected components
  (`fitace/kinship/grm_io.py:354-369`).
- Python effective-size orchestration creates a separate thread pool
  (`pedigree_graph/_effective_size.py:132-206`).

## Canonical vocabulary

The updated glossary is `external/pedigree-graph/CONTEXT.md`.

- **Graph-space**: input-row coordinates of the full pedigree.
- **View-space**: row coordinates of an explicitly ordered `PedigreeView`.
- **Structural depth**: derived solely from represented parent edges.
- **Generation label**: optional cohort metadata that never affects relationships or
  kinship.
- **Nominal kinship**: the category constant.
- **Pedigree-specific kinship**: the coefficient obtained from all relevant pedigree
  paths, including inbreeding and MZ identity, before documented float32 matrix
  rounding.
- **Relationship pair**: role-ordered for asymmetric categories and biologically
  symmetric for symmetric categories.

## Python API established in 0.8.0

### Construction

```python
from pedigree_graph import PedigreeGraph

full = PedigreeGraph.from_frame(frame)
full = PedigreeGraph.from_arrays(
    ids=ids,
    mother_ids=mother_ids,
    father_ids=father_ids,
    twin_ids=twin_ids,  # optional
    sex=sex,  # optional
    generation=generation,  # optional, may be partial
    birth_year=birth_year,  # optional, may be partial
)
```

The loose `PedigreeGraph(frame_or_dict)` constructor and `from_subsample` are removed.

Required fields are only `id`, `mother`, and `father`. Frame construction accepts
host-native nulls. Array construction also accepts `-1` sentinels. Individual IDs are
nonnegative signed 64-bit integers; row coordinates are signed 32-bit integers.

Construction accepts any acyclic input row order. Rust/Python creates a stable private
topological order, while every public graph-space row remains aligned with the input.
Cycles produce a structured validation error.

Unresolved parent/twin IDs are valid partial-pedigree references and remain distinct
from missing references. Original unresolved parent IDs continue to support sibling
classification. A represented child cannot name the same known individual in both
parent roles.

MZ references whose targets are represented must be non-self, reciprocal, exactly
two-member, parent-identical, and sex-identical when both sexes are known. An external
co-twin reference does not establish an internal MZ pair.

Sex is internally `Female`, `Male`, or `Unknown`, transported as `0`, `1`, and `-1`
in Python. Other codes are rejected. Parent-role/sex conflicts do not block graph
construction, allowing validation tools to represent and report imperfect data.

### Graph and view coordinates

```python
view = full.view(ids=sample_ids)
view = full.view(rows=sample_rows)
```

Exactly one keyword is required. Selection order is preserved; duplicates, missing
IDs, and out-of-range rows are structured errors.

`PedigreeGraph` operations use graph-space. `PedigreeView` operations use view-space.
`PedigreeView` initially exposes only:

- `relationship_pairs(...)`;
- `relationship_counts(...)`;
- `pair_kinship(...)`;
- read-only `ids` and `graph_rows`; and
- `len(view)` / `view.n_individuals`.

Matrix, inbreeding, lineage, connectivity, and effective-size operations remain on the
full graph until a scientifically clear view contract is needed.

### Read-only properties

The full graph exposes lazy, memoized, read-only host copies:

- `ids` (`int64`);
- `mother_ids`, `father_ids`, `twin_ids` (`int64`, `-1` missing);
- `mother_rows`, `father_rows`, `twin_rows` (`int32`, `-1` absent/external);
- `sex` (`int8`, `-1` unknown) or `None` when wholly absent;
- `depth` (`int32`, always present);
- `generation_labels` (`int32`, `-1` missing) or `None` when wholly absent;
- `birth_year` (`int32`, `-1` missing) or `None`; and
- `n_individuals`, mirrored by `len(graph)`.

There is no ambiguous public `generation`, `mother`, `father`, `twin`, or `n`.
Rust remains authoritative; mutating an exposed array fails instead of appearing to
change the graph.

### Relationship registry

One immutable public registry replaces `REL_REGISTRY` plus `PAIR_KINSHIP`:

```python
RELATIONSHIPS["FS"].code
RELATIONSHIPS["FS"].label
RELATIONSHIPS["FS"].degree
RELATIONSHIPS["FS"].nominal_kinship
RELATIONSHIPS["FS"].up
RELATIONSHIPS["FS"].down
RELATIONSHIPS["FS"].ancestor_count
RELATIONSHIPS["MO"].first_role
RELATIONSHIPS["MO"].second_role
```

The ordered Rust 23-variant relationship enum is the eventual source. Registry order
also provides the documented same-degree precedence used for closest-category
classification.

### Relationship-pair queries

```python
pairs = graph.relationship_pairs(max_degree=3)
pairs = graph.relationship_pairs(categories={"FS", "MHS", "PHS"})
```

Exactly one of `max_degree` or `categories` is required. The same selector contract
applies to exact relationship counts and relationship kinship matrices. Selection is
an output filter, not a reclassification rule: all closer-category dependencies are
still resolved internally.

`RelationshipPairs` is an immutable mapping containing all 23 keys. Each
`RelationshipPairBlock` contains:

```python
block.first_rows
block.second_rows
block.first_role
block.second_role
block.requested
len(block)
```

Blocks remain unpackable as `(first_rows, second_rows)`. Unrequested blocks are empty
but have `requested=False`; count results report them as unrequested rather than zero.

Pair contracts:

1. Asymmetric categories have fixed semantic roles: offspring→mother,
   offspring→father, descendant→ancestor, and corresponding collateral roles.
2. Symmetric categories have no biological role distinction.
3. Internal subtraction keys canonicalize independently of public role order.
4. Every unordered individual pair appears in at most one category: lowest degree
   first, then registry precedence for same-degree conflicts.
5. If both orientations of an asymmetric category are valid through different paths,
   return the individual pair once and choose deterministically between the valid
   orientations by input-row order.
6. Within each block, records are sorted by canonical unordered row key so Rayon
   scheduling cannot affect output order.
7. Row arrays are `int32` in Python and 1-based R integers in R.
8. Relationship results own their arrays and do not retain the graph/view or its
   caches. An opaque coordinate-space token prevents use with the wrong receiver.

Specialized `sibling_pairs()` is removed; callers request the sibling categories.
Per-category exclusion lists remain next to each relationship implementation rather
than moving into one generic table. For example, the current `1C1R` candidate product
subtracts closer categories at `pedigree_graph/_pair_extractor.py:388-421`.

### Counts and estimates

```python
counts = graph.relationship_counts(max_degree=3)
estimates = graph.estimate_relationship_counts(max_degree=5)
```

`relationship_counts` is exact and follows the same closest-category semantics as
`relationship_pairs`.

`estimate_relationship_counts` replaces `count_pairs_streaming`. Its typed result
contains values, requested status, exact categories, approximate categories, and
clamped categories. A call with clamping emits one Python `RuntimeWarning` summarizing
all affected requested categories. Cached retrieval does not warn again.

The streaming estimator is full-graph-only initially; it is not exposed on
`PedigreeView`.

### Kinship and inbreeding

```python
values = graph.pair_kinship(first_rows, second_rows)
values = graph.pair_kinship(pairs["FS"])
values = graph.pair_kinship(pairs)

K_full = graph.kinship_matrix()
K_rel = graph.relationship_kinship_matrix(max_degree=3)
K_rel = graph.relationship_kinship_matrix(categories={"FS", "MHS", "PHS"})
F = graph.inbreeding()
```

`pair_kinship` returns float64 values and accepts arbitrary/self pairs. A collection
query flattens requested blocks into one core call so one recurrence memo serves every
category.

`kinship_matrix()` is complete: it contains every nonzero pedigree kinship.
`relationship_kinship_matrix(...)` is structurally limited to selected closest
relationship categories. Every retained coefficient is computed from the full
pedigree rather than a propagation-pruned DP. Both include the diagonal.

CSC contracts remain:

- SciPy `csc_matrix`;
- float32 data;
- int32 indices and indptr;
- sorted row indices within each column; and
- read-only cached arrays at the Python boundary.

R promotes the same float32 values to the double `x` slot of `Matrix::dgCMatrix`.
“Pedigree-specific” permits only this documented final float32 rounding.

Canonical `inbreeding()` is MZ-aware and satisfies
`F_i = 2 * phi(i, i) - 1`. Issue #8 determines whether the existing MZ-naive
Meuwissen–Luo implementation is deleted or retained under an explicit noncanonical
name.

The choice between always running the pairwise recurrence and sampling an already
cached complete matrix remains deferred to issue #6. Either path must return the same
values; the plan does not silently choose one.

### Generation summaries, lineage, and connectivity

```python
summary = graph.mean_kinship_by_generation()
summary.generations
summary.mean_kinship
summary.pair_counts
summary.unlabelled_individual_count

graph.distinct_ancestor_counts()
graph.descendant_path_counts()
graph.connected_component_ids()
```

If generation metadata is wholly absent, generation summaries use structural depth.
If it is partial, unlabelled individuals are excluded and reported; they are never
silently assigned depth. Generation labels may be sparse and the result includes only
observed labels rather than allocating through `max(label)`.

Effective-size estimators that require generations reject partial labels until their
missing-label semantics receive a separate statistical decision. Sex-dependent
estimators similarly reject unknown/missing sex; sex-independent estimators continue.

`connected_component_ids()` returns a read-only int64 vector aligned to input rows.
Each value is the smallest original individual ID in that represented parent-edge
component, preserving fitACE's current deterministic FID contract without importing
fitACE's “founder family” policy into the graph API.

### Package namespace

Package-root exports are limited to:

- `PedigreeGraph`, `PedigreeView`;
- `RelationshipCategory`, `RelationshipPairs`, `RelationshipPairBlock`;
- `RELATIONSHIPS`;
- `PedigreeValidationError`, `MissingMetadataError`, `ResourceError`; and
- `configure_threads`.

`FrameLike` moves to `pedigree_graph.typing`.

Effective-size functions, cohort utilities, generation-interval types, and result
classes move to public `pedigree_graph.effective_size`. Existing scientifically named
`ne_*` functions remain there. `compute_all_ne` becomes
`estimate_effective_sizes`; estimator formulas do not change in this migration.

## Rust architecture

### Workspace

```text
Cargo.toml
crates/
  core/       # pedigree-graph-core; no Python/R imports; publish = false
  python/     # PyO3 module pedigree_graph._native
pedigree_graph/
  ...         # typed Python facade and host representations
r/
  ...         # initial extendr package, added in 0.9.0
```

The Rust type is also `PedigreeGraph`, imported as
`pedigree_graph_core::PedigreeGraph`. The crate remains unpublished until 1.0; Python
uses a workspace path and R vendors the source.

Conceptual core state:

```text
PedigreeGraph
  original IDs and typed missing/external/internal parent references
  input-row ↔ private-topological-row maps
  sex / optional generation labels / optional birth years
  derived structural depth
  shared execution context
  bounded one-shot memoization for modest invariant-derived results

PedigreeView
  Arc<PedigreeGraph>
  ordered view-row ↔ graph-row maps
  independent coordinate-space token
```

Host `-1` sentinels do not enter the domain model. Internally use typed
`IndividualId`, `RowId`, and explicit missing/external/internal reference states,
implemented in compact structure-of-arrays storage.

### Relationship engine invariants

- `Ak(0)` is an `Identity` variant, never an accidental parent hop.
- Pair subtraction uses one canonical 64-bit key built from two 32-bit rows.
- Public semantic orientation is restored only after internal key operations.
- Full/half classification preserves path multiplicity through the `>= 2` decision.
- `CousinSplit { full, half }` removes degree-gated side-channel caches.
- `[PairBlock; 23]`, indexed by the relationship enum, makes registry coverage
  structural.
- Exclusion lists remain explicit within each category implementation.
- A global uniqueness/precedence check catches missing exclusions.
- Query-local intermediates are freed at call completion.

Issue #9 must resolve the spike's unchecked `i32` CSR multiplicity arithmetic before
the pair engine is promoted. Production arithmetic may not silently wrap.

### Kinship DP storage

No row-storage representation is preselected. `Vec<Vec<_>>` is the first simplicity
prototype, not an accepted tradeoff. It must pass the wall/RSS gate against the warmed
current implementation. If it fails, port a slab/arena design with retirement.

### Memoization and ownership

The public graph is immutable, but safe interior memoization is allowed.

- Rust `OnceLock`-style caches are preferred for modest, single-valued host-neutral
  results such as inbreeding, lineage vectors, and generation summaries.
- Arbitrary pairwise requests and arbitrary category/view query combinations do not
  receive unbounded caches.
- Large relationship and CSC buffers require an ownership benchmark: compare
  Rust-owned caching plus host conversion against transferring ownership once and
  caching only the read-only host representation.
- No compatibility cache field names remain public or test-observable.
- Relationship result objects never retain the graph.

### Threads and determinism

One package-wide Rayon pool is configured before first parallel work:

```text
explicit configure_threads(n)
    > PEDIGREE_GRAPH_THREADS
    > default 1
```

Reconfiguration after initialization is an error unless it repeats the existing value.
There are no per-call thread counts. fitACE's thread-cap helper must set
`PEDIGREE_GRAPH_THREADS`. `estimate_effective_sizes` prepares Rust prerequisites on
this pool and applies Python formulas serially; it creates no Python worker pool.

Recommended acceptance criterion, pending explicit final sign-off: integer outputs are
bit-identical across thread counts; floating reductions use fixed partitions and
ordered combination where practical. Any tolerance must be declared per kernel and
justified by benchmarked cost.

### Safety and allocation

`pedigree-graph-core` uses `#![forbid(unsafe_code)]`. CI asserts core types are
`Send + Sync`, rejects PyO3/extendr dependencies in core, runs Clippy with warnings as
errors, and forbids user-reachable panics.

Potentially large buffers use fallible reservation and propagate structured
`ResourceError`s through Rayon and host bindings. Subprocess tests verify adversarial
allocation failures raise rather than abort. No arbitrary default memory budget is
invented before profiling.

## Host bindings and packaging

### Python

Maturin is the preferred build backend. The scaffold must prove:

- mixed Python/Rust packaging with `pedigree_graph._native`;
- `abi3-py313` wheels;
- editable installation through the pedigree-graph pixi manifest;
- wheel installation tests independent of the source tree;
- clean sdist → wheel builds; and
- inclusion of type stubs and `py.typed`.

If a concrete scaffold gate fails, document it before falling back to setuptools-rust.
There is no production Python fallback after an operation migrates to Rust.

Initial native-wheel support:

- CPython 3.13+ via ABI3;
- manylinux x86-64 and AArch64;
- macOS x86-64 and Apple Silicon;
- Windows x86-64; and
- an sdist requiring Rust for other platforms.

No initial PyPy, musllinux, Windows ARM, or 32-bit guarantee.

### Versioning

The pure-Python 0.8.0 baseline may be the final setuptools-scm release. Once the Cargo
workspace lands, `[workspace.package].version` becomes authoritative. Maturin reads the
Python version from Cargo. The release tool updates Cargo and `r/DESCRIPTION` together;
CI asserts wheel, sdist, Cargo metadata, DESCRIPTION, and Git tag agree.

pedigree-graph keeps independent SemVer and is not part of the simACE/fitACE CalVer
family.

### R 0.9.0 scope

The first R package remains intentionally small:

```r
pg <- pedigree_graph(df)
pairs <- relationship_pairs(pg, max_degree = 3)
phi <- pair_kinship(pg, pairs$FS$first, pairs$FS$second)
K <- kinship_matrix(pg)
F <- inbreeding(pg)
```

`relationship_pairs` returns a named list of all 23 categories. Each element is a
zero-or-more-row data frame with integer `first`/`second` columns, role attributes,
requested status, deterministic ordering, and 1-based rows.

Views, count estimation, lineage, connectivity, effective-size estimators, and full
Rust-core parity are deferred.

The R source tarball must compile offline. Packaging stages the core under `r/src/rust`,
runs `cargo vendor` for the complete locked dependency graph (including extendr and
Rayon transitives), writes the source replacement config, and proves
`cargo --offline` plus `R CMD check` against the final tarball in network-disabled CI.
The layout is CRAN-compatible from the first release, though actual CRAN submission is
deferred.

## Structured errors

Core errors are enums carrying relevant IDs, rows, capacities, and operation context.
Python maps construction failures to `PedigreeValidationError` (`ValueError` subclass)
with a stable `.code`; missing analysis metadata maps to `MissingMetadataError`;
allocation/capacity failures map to `ResourceError`. R uses classed conditions.

Tests assert codes and fields, not verbatim prose. Existing regex-matched error strings
are not compatibility requirements.

## Migration and release sequence

Every published slice is green, removes the production implementation it replaces,
and passes the cross-repository release gate. Exact 0.8.x patch numbers are assigned at
release time rather than baked into this plan.

### 0.8.0 — pure-Python API redesign

1. Implement the entire canonical Python API above using current kernels.
2. Separate structural depth from generation labels.
3. Add arbitrary-row-order support and strict represented-MZ validation.
4. Add graph/view objects and typed role-ordered relationship results.
5. Enforce global closest-category exclusivity and deterministic block ordering.
6. Replace registry, counting, kinship-matrix, lineage, connectivity, constructor, and
   effective-size names/contracts.
7. Make canonical inbreeding MZ-aware; resolve issue #8's old-algorithm disposition.
8. Migrate simACE, fitACE, fitACE_epimight, PA-FGRS, and pedsum in the same coordinated
   change. Replace fitACE's private connected-components reach and PA-FGRS's private
   `_compute_depth` import.
9. Keep the experimental BFS engine temporarily, adapting only what the pure-Python API
   break requires.
10. Publish 0.8.0 as the frozen differential baseline for Rust migration.

### 0.8.x — native construction and build scaffold

1. Add the Cargo workspace, host-neutral input model, stable topological reorder,
   structured errors, and depth calculation.
2. Add the PyO3 module and typed Python facade.
3. Switch to Maturin and Cargo-authoritative versioning after all scaffold gates pass.
4. Delete replaced Python validation, ID remapping, and depth-construction code.

### 0.8.x — remove BFS before relationship migration

Resolve issue #7. Delete the experimental BFS engine, kernel, tests, documentation, and
BFS-only relationship metadata. Do not adapt it to Rust-owned adjacency state. Numba
remains until the other production kernels migrate.

### 0.8.x — streaming relationship-count estimator

Port the estimator and its structured precision/clamping result before deleting shared
Python/SciPy adjacency powers. Preserve exact/approximate category behavior and add one
host warning per clamped call.

### 0.8.x — relationship-pair engine

1. Resolve issue #9's multiplicity representation.
2. Promote the spike into focused CSR, key, relationship, view-remap, and role-ordering
   modules.
3. Add selected-category dependency resolution, global exclusivity, and deterministic
   output sorting.
4. Compare against the independent oracle and released 0.8.0 baseline at fixture,
   inbred, 30k, and 300k scales.
5. Delete the Python matrix extractor, pair utilities, SciPy adjacency powers, sibling
   matrices, release hooks, and old coordinate fields once no remaining engine uses
   them.

### 0.8.x — pairwise kinship

Port the direct recurrence with arbitrary/self-pair, graph/view, MZ, inbreeding, and
multi-path tests. Resolve issue #6 before choosing recurrence-only versus cached complete
matrix sampling. Delete the production Python/Numba implementation; retain an
independent test oracle.

### 0.8.x — complete and relationship kinship matrices

1. Implement complete CSC construction and exact-coefficient relationship-limited CSC
   construction.
2. Preserve float32/int32/sorted-column host contracts.
3. Benchmark and choose DP row storage; do not pre-accept `Vec<Vec<_>>`.
4. Add graceful capacity/NNZ failures.
5. Delete replaced DP, allocator, and CSC production modules after parity.

### 0.8.x — inbreeding, generation summary, lineage, and effective-size prerequisites

Port MZ-aware inbreeding, generation mean kinship, distinct ancestor counts, descendant
path counts, equivalent generations, founder contribution sums, and Caballero–Toro
accumulators. Move modest invariant caches into Rust where beneficial. Keep high-level
estimator formulas in `pedigree_graph.effective_size`. Remove Numba when no production
or retained experimental code uses it.

### 0.9.0 — initial R package

Add the small extendr surface, role-aware R pair frames, Matrix CSC conversion, offline
vendoring, testthat parity, and source-tarball `R CMD check`.

### 1.0.0 — stabilization

Close or explicitly disposition all deferred issues, remove temporary migration
allowlists, complete architecture guardrails, decide whether to publish the Rust core,
and verify supported Python/R artifacts from clean installations.

## Correctness gates

1. All 23 relationship categories exist in every typed result, with request status.
2. Semantic endpoint roles, global exclusivity, and deterministic ordering are tested.
3. Graph-space/view-space conversion is tested under reordered and empty views.
4. Arbitrary input order produces the same row-aligned results as topological input.
5. Multiplicity is preserved through full/half classification; no unchecked overflow.
6. `Ak(0)` remains identity.
7. Pairwise kinship covers arbitrary keys/blocks, self-pairs, MZ ancestry, inbreeding,
   and duplicate paths.
8. Canonical inbreeding equals `2 * self_kinship - 1`.
9. Complete and relationship-limited matrix entries match float32-rounded pairwise
   values.
10. CSC data/indices/indptr dtypes and per-column sorting are pinned.
11. Partial generation and sex metadata follow the explicit analysis rules.
12. Structured validation/resource errors are tested at both host boundaries.

## Differential and property-test gates

- A readable independent Python oracle handles small acyclic, inbred, partial, MZ,
  reordered, and overlapping-generation pedigrees.
- Property tests compare Rust with the oracle.
- Large differential tests compare each migrated slice with the released pure-Python
  0.8.0 baseline.
- Rust-native tests independently assert type-level and arithmetic invariants.
- Replaced Python production code is not retained as a fallback.

## Performance gate

Apply the 5% blocker only to behavior-equivalent comparisons.

- Warm Python/Numba and compile Rust in release mode.
- Use the same configured thread budget.
- Run each measurement in a fresh process and interleave baseline/candidate runs.
- Report median wall time, peak RSS, and uncertainty intervals.
- Block only when the regression is confidently greater than 5%.
- Compare redesigned relationship matrices with fitACE's current exact construction,
  not with propagation-pruned output.
- Any exception requires explicit maintainer sign-off and documentation.

## Cross-repository release gate

Before 0.8.0 and every published migration patch:

- pedigree-graph: full pytest, Ruff, type checking, and—once present—Cargo tests,
  rustfmt, and Clippy;
- simACE: relevant full test groups plus workflow smoke test;
- fitACE monorepo: core and every consuming method package;
- fitACE_epimight: relevant integration tests;
- pedsum: full tests plus representative CLI smoke tests; and
- from 0.9.0 onward: testthat and offline source-tarball `R CMD check`.

Intermediate commits may run scoped tests; release gates may not.

## Deferred issues and blockers

- [#6](https://github.com/rwaples/pedigree-graph/issues/6): recurrence-only versus
  reuse of a cached complete matrix in `pair_kinship`.
- [#7](https://github.com/rwaples/pedigree-graph/issues/7): remove BFS after 0.8.0 and
  before Rust relationship migration.
- [#8](https://github.com/rwaples/pedigree-graph/issues/8): compare MZ-aware and
  Meuwissen–Luo inbreeding and decide whether the old algorithm remains explicitly.
- [#9](https://github.com/rwaples/pedigree-graph/issues/9): audit and fix Rust CSR
  multiplicity overflow before promoting the spike.
- DP row storage: choose only after benchmark.
- Maturin: preferred, conditional on the scaffold gates.
- Cross-thread floating determinism: recommended above but not explicitly confirmed in
  the review; confirm before implementation.

## Documentation before implementation

Two ADRs record the decisions because both are hard to reverse, surprising without
context, and genuine tradeoffs:

1. ADR 0006 — public API and coordinate/relationship semantics for 0.8.0.
2. ADR 0007 — Rust core, host ownership/memoization, threading, build, and release
   architecture.

Both live in `external/pedigree-graph/docs/adr/` (pedigree-graph commit `4fd07ba`,
which also updates `CONTEXT.md`).

Update README, architecture documentation, limitations, changelog, typing docs, and all
consumer examples in the same 0.8.0 migration. Do not present the old plan as active
after this version is approved.
