# CLAUDE.md

simACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (common environment), **E** (unique environment) variance components. Provides simulate → phenotype → censor → ascertainment → validate → stats → plot. Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan) lives in the sister repo [fitACE](../fitACE), which depends on simace.


## Project Layout

- `simace/` — simulation package (`pip install -e .`), organized into sub-packages:
  - `core/` — shared infrastructure: `pedigree_graph`, `compute_hazard_terms`, `cli_base`, `numerics`, `parquet`, `pedigree_filter`, `relationships`, `schema`, `yaml_io`
  - `simulation/` — pedigree simulation
  - `phenotype/` — `runner.py` (run_phenotype dispatcher, re-exported from `__init__.py`), `hazards.py`, `blended_post.py`, plus a `models/` sub-package of model classes inheriting from a `PhenotypeModel` (the liability-threshold idiom lives in `models/_prevalence.py`)
  - `censoring/` — age-window and death censoring
  - `ascertainment/` — unified dropout + case-weighted N_sample selection (per ADR 0001)
  - `analysis/` — `stats/` (package: censoring, correlations, incidence, pedigree, sampling, tetrachoric, runner), `validate.py`, `gather.py`
  - `plotting/` — all plot modules and plot utilities
- `workflow/rules/simace/*.smk` — Snakemake rules; `workflow/scripts/simace/` — thin script wrappers
- `config/_default.yaml` — default parameters; `config/{folder}.yaml` — per-folder scenario files (auto-discovered; files starting with `_` are skipped)
- `results/{folder}/{scenario}/` — output per scenario

Each nested repo has its own `origin` wired to the matching GitHub repo — `git push` from inside each directory goes to the right place. 

## Snakemake

- Root `Snakefile` is the entry point — not `-s workflow/Snakefile`
- Use `--cores 4` running one scenario, `--cores 8` for multiple scenarios, `--cores 1` for debugging. 
- Always dry-run (`-n`) before long runs.
- Targets are per-scenario: `results/{folder}/{scenario}/{scenario,simulate,phenotype,validate,stats}.done`
- Force-rebuild plot atlas (HTML is the default artifact; PDF is on-demand): `snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.html` (or `.../atlas.pdf` for the PDF export)

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `simace/plotting/atlas_manifest.py`

## Key Rules

- The `simACE` conda env is always active. Do NOT use `conda run -n simACE` — run commands directly.

## Code review gotchas (statistical correctness)

Bugs that have occurred in pedigree/variance/phenotyping code. Check these
patterns whenever changing the relevant module.

1. **Booleanise sparse matrices only AFTER using multiplicity.** Sparse
   matrix products counting shared ancestors must preserve edge weights
   before thresholding. Booleanising too early collapses full vs half
   distinctions. Has occurred ≥3 times: `_cousin_pairs` (1C/H1C),
   `_second_cousin_matrix` (2C/H2C), and the up=1 avuncular variant.
   Correct pattern: `data[data < 2] = 0; eliminate_zeros()` before
   booleanising.
2. **Full vs half classification needs ≥2 shared ancestors through a
   mated pair.** Code that checks `> 0` instead of `>= 2` silently
   misclassifies.
3. **`_get_Ak(0)` must return identity**, not chain through the parent
   adjacency matrix. Chaining adds a spurious parent hop to up=1
   relationships.
4. **Cross-package coupling**: both `fit_ace` and simace import
   `PAIR_KINSHIP` and pair extraction (`PedigreeGraph.extract_pairs`) from the
   external top-level `pedigree_graph` package (not from simace). Changes to
   pair extraction or kinship values in `pedigree_graph` silently bias
   `fit_ace` heritability and PA-FGRS. Additionally, `fitace.relationships`
   maintains fitACE relationship-type kinship at EPIMIGHT-compatible granularity
   and must stay in sync with `PAIR_KINSHIP`.
5. **Generation-dependent C/E variance can bias `rho_w`** (assortative
   mating correlation) calculations.
6. **`affected = NOT (age_censored OR death_censored)`** — preserve this
   identity through any censoring change.
7. **Degree-gating side effects.** Pair extraction proceeds degree by
   degree; lower-degree methods populate caches consumed at higher
   degrees (e.g., `_cousin_pairs` → `_h1c_pairs_cache`). `max_degree >= N`
   and `_needed()` guards can skip producing methods; downstream
   `getattr(self, "_cache", fallback)` reads then silently return empty
   results.
8. **Pair key encoding `lo * max_id + hi`** (int64) requires canonical
   `lo < hi` ordering and overflows beyond ~3B individuals.

### Liability correlation expected values

Formula: `r = 2 * kinship * A + C_shared` where C_shared = C if the pair
shares a household, 0 otherwise. **Household is assigned by mother**
(`simulate.py`: `np.unique(parent_idxs[:, 0])`) — so maternal half-sibs
share C but paternal half-sibs do not.

Reference: MZ = A+C, FS = 0.5A+C, MHS = 0.25A+C, PHS = 0.25A, PO = 0.5A.
Source of truth for kinship: `PAIR_KINSHIP` in the external `pedigree_graph`
package (`pedigree_graph/_registry.py`). With inbreeding,
`PedigreeGraph.compute_pair_kinship()` returns per-pair values that may differ
from `PAIR_KINSHIP`.

## Repo Map

Thirteen related repos, all under `rwaples/` on GitHub. simACE is the umbrella working directory; the others are nested checkouts (gitignored from simACE — no submodules). The seven `fitACE_*` method repos were peeled out of fitACE in the method-split (each depends on `fitace`/`simace`, never on another method repo).

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → ascertainment → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model-fitting **core + Snakemake orchestrator**: kinship, config, LTM/Falconer, GRM export, phenotyping, shared result writers + cross-method plots. The fitting methods live in the `fitACE_*` sister repos below. Consumes simACE outputs. |
| [`fitACE_epimight`](https://github.com/rwaples/fitACE_epimight) | private | `./fitACE/fitACE_epimight/` | EPIMIGHT v2.0 integration for fitACE: long-form input emitter, Snakemake rules, atlas/bias plotting. Included by `fitACE/Snakefile` via cross-repo `include:` directives. |
| [`fitACE_pcgc`](https://github.com/rwaples/fitACE_pcgc) | private | `./fitACE/fitACE_pcgc/` | PCGC / Haseman–Elston moment estimator (reference/numba/cpp backends + `ace_pcgc` C++ binary). |
| [`fitACE_iter_reml`](https://github.com/rwaples/fitACE_iter_reml) | private | `./fitACE/fitACE_iter_reml/` | Iterative PCG-AI-REML (`ace_iter_reml`) + sparse REML (`ace_sreml`, env-only). Holds packages `fitace_iter_reml` + `fitace_sreml`. |
| [`fitACE_tetraher`](https://github.com/rwaples/fitACE_tetraher) | private | `./fitACE/fitACE_tetraher/` | LDAK TetraHer (`--tetra-her`) liability-scale A+C+E. Consumes the `tetraher_simace` LDAK fork. |
| [`fitACE_pafgrs`](https://github.com/rwaples/fitACE_pafgrs) | private | `./fitACE/fitACE_pafgrs/` | PA-FGRS (Pearson–Aitken Family Genetic Risk Scores) + diagnostic atlas. |
| [`fitACE_stan`](https://github.com/rwaples/fitACE_stan) | private | `./fitACE/fitACE_stan/` | Stan/cmdstanpy Bayesian ACE fitting. Dormant (no Snakemake rule). |
| [`fitACE_frailty`](https://github.com/rwaples/fitACE_frailty) | private | `./fitACE/fitACE_frailty/` | Weibull-frailty pairwise correlation stats. Dormant (no Snakemake rule). |
| [`ace_iter_reml`](https://github.com/rwaples/ace_iter_reml) | private | `./fitACE/fitACE_iter_reml/ace_iter_reml/` | C++ PCG-AI-REML binary. Driven by `fitACE_iter_reml`. |
| [`tetraher_simace`](https://github.com/rwaples/tetraher_simace) | private | `./external/tetraher_simace/` | Fork of LDAK 6.2 (grouping + warm-start + OMP opt-in). Binary consumed by `fitACE/fitace/tetraher/`. |
| [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) | public | `./external/pedigree-graph/` | Sparse-matrix pedigree relationship extraction and kinship computation. |
| [`pedsum`](https://github.com/rwaples/pedsum) | public | `./external/pedsum/` | Pedigree summary CLI: structure, relatedness, inbreeding, Ne estimators. Built on `pedigree-graph`. |

## Cross-repo edits (simACE + fitACE)

- Treat simACE and fitACE as a coordinated pair when work spans both. Verify edits land in each repo's working tree (`git status` in both), run tests in both, and make parallel commits — do not assume changes propagate.

## Git usage
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Commit only when explicitly asked
- Prefer batching commits — changed files grouped by purpose

## Versioning
- **Lockstep family CalVer.** simACE, fitACE core, the seven `fitACE_*` method
  sisters, and the `ace_iter_reml` binary share **one** CalVer
  (`vYYYY.MM[.patch]`), tagged together each release (ADR 0012). External deps
  (`pedigree-graph`, `pedsum`, `tetraher_simace`) keep their own versions.
- **CalVer** (`YYYY.MM`) via `setuptools-scm`, derived from git tags; the
  `ace_iter_reml` binary embeds `git describe` via CMake.
- Tag format: `v2026.06`, `v2026.06.1` (second release same month). First unified
  lockstep release: `v2026.06`. Between tags: `2026.6.dev4+g<hash>`.
- Compatibility is one `FAMILY_FLOOR` in `fitace._deps` (`>=` semantics),
  enforced across every family `pyproject.toml` by `test_dependency_floors`.
- To cut a release: `python tools/release.py vYYYY.MM` tags all ten repos
  locally and prints the per-repo `git push` commands (it never pushes).

## Testing

- Full suite: `pytest tests/ -v`
- Single module: `pytest tests/simulation/test_simulate.py -v`
- Run relevant tests before commit
- Smoke test: `snakemake --cores 4 results/test/small_test/scenario.done`

## Linting

- Check: `ruff check`
- Auto-fix: `ruff check --fix`
- Format Python: `ruff format`
- Format Snakemake: `snakefmt workflow/rules/**/*.smk Snakefile`
- Run `ruff check` with **no extra `--select`**. The configured rules (incl. `D`/pydocstyle) plus the `ignore` and `per-file-ignores` in `pyproject.toml` are authoritative. Passing any `--select` (e.g. `--select D`) discards those ignores and surfaces false positives.

## Documentation & Citations

- Never generate citations, DOIs, author lists, journal names, years, page numbers, or any bibliographic field from memory. Memory recall of bib metadata is treated as fabrication.
- Verify every entry against a live source before writing it: resolve `https://doi.org/<doi>` via WebFetch, or pull from Crossref/PubMed/publisher page. Confirm the returned title/authors/year match the citation you are about to write.
- One verification per field set — do not extrapolate. 
- If verification fails (DOI 404s, source unreachable, ambiguous match), do NOT write the entry. Insert a `% TODO: verify <what>` placeholder and tell the user which entries could not be verified.

## Planning and Implementation

- For non-trivial implementation tasks, propose 2-3 approaches with tradeoffs before writing code. Wait for approval.
- For non-trivial plans/refactors (multi-file, cross-repo, or with unresolved design decisions), default to invoking the `grill-with-docs` skill before proposing implementation. Lock each design decision explicitly before exiting plan mode. Skip grilling for bugfixes, doc tweaks, and renames.
- When starting a design interview or /grill-me session, if there is no existing plan, first explore the relevant codebase 
and read key files and related modules before asking questions. Ground the interview in what the code actually does.
- When a plan relies on formulas, thresholds, complexity claims, or memory/allocation models, enumerate each such assumption explicitly and verify it against the primary source and the actual codebase before locking the decision. Treat quantitative claims recalled from memory as unverified; flag any you cannot confirm rather than proceeding on them.

## Performance Optimization

- Always profile/benchmark first to identify the actual bottleneck before implementing changes
- Do not assume which component is slow — show profiling data before proposing a solution
- When narrowing numeric dtypes for memory optimization, never narrow below int32 or float32

## Session Management

- Prefer focused sessions (one feature per session)
- Run pipeline commands in background when >30 seconds
- For pipelines with verbose output, redirect to a log file and grep/tail a summary rather than streaming everything — streaming full output floods context and can hit the output-token ceiling. The `run_in_background` + Monitor tools do this natively.
- Use targeted line ranges instead of reading entire large files

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

This project has a knowledge graph (embeddings on, auto-updated on session
start). Reach for the graph when a question is *structural* — relationships,
impact, coverage, flows — not when you already know the path you want to read.

### Cheap entry point

Before any non-trivial graph exploration, call `get_minimal_context` (~100
tokens). It returns risk score, top communities/flows, and suggests which
tool to call next. Cheaper than guessing wrong.

### Use the graph for

| Task | Tool | Notes |
|------|------|-------|
| Review a diff / PR | `detect_changes` | Primary review tool; risk-scored + prioritized. Supersedes `get_review_context` for change-aware work. |
| Blast radius of a change | `get_impact_radius` | Beats manually tracing imports. |
| Which entry points are affected | `get_affected_flows` | Identifies user-facing/critical paths touched by a diff. |
| Trace a single relationship | `query_graph` | Patterns: `callers_of`, `callees_of`, `imports_of`, `importers_of`, `tests_for`, `children_of`, `inheritors_of`, `file_summary`. |
| Find a symbol by name/concept | `semantic_search_nodes` | FTS5 + embeddings; faster than `grep` for fuzzy lookups. |
| High-level architecture | `get_architecture_overview` + `list_communities` | Use when onboarding to an unfamiliar area. |
| Trace an execution path | `list_flows` → `get_flow` | Each flow = call chain from entry point (CLI, test, etc.). |
| Where are the risks | `get_suggested_questions`, `get_knowledge_gaps`, `get_surprising_connections` | Use at start of review to surface untested hubs, thin communities, cross-community coupling. |
| Decomposition audit | `find_large_functions` | Line-count threshold; filter by file path. |
| Rename / dead code | `refactor_tool` (modes: `rename`, `dead_code`, `suggest`) | Rename preview returns an edit list; apply via `apply_refactor_tool`. |

### Cross-repo searches

`cross_repo_search` covers the repos registered under
`~/.code-review-graph/registry.json` (below). The seven `fitACE_*` method repos
from the method-split are pending registration (see **Adding a new repo**);
register them once they're cloned into the layout.

| Repo | Languages indexed | Embeddings |
|------|-------------------|------------|
| simACE | python, bash | yes |
| fitACE | python | yes |
| fitACE_epimight | python, r | yes |
| pedigree-graph | python | yes |
| pedsum | python | yes |
| ace_iter_reml | cpp | yes |
| tetraher_simace | c, python, bash | yes |

Use `cross_repo_search` when:

- A symbol/concept might live in *any* of the related repos (e.g. "kinship",
  "reml", "tetrachoric").
- You're coordinating cross-repo edits (per the simACE↔fitACE note above).
- You're tracing how a simACE concept is consumed downstream.

Stick with single-repo tools (`semantic_search_nodes`, `query_graph`) when
you know which repo to look in — same hits, less noise.

**Adding a new repo:**

```
code-review-graph build --repo <path>       # if no graph DB exists yet
code-review-graph register <path> --alias <name>
# then via MCP: embed_graph_tool(repo_root="<path>")
```

### Skip the graph when

- You already know the exact file path → just `Read` it.
- You need a string literal that won't be a graph node (config values,
  error messages, log strings) → `grep`.
- You're editing a file you just read in this session.

### Maintenance

Graph auto-updates on file changes via the hook in `.claude/settings.json`.
If stats look stale, run `code-review-graph status`.

## Agent skills

### Issue tracker

GitHub Issues at github.com/rwaples/simACE/issues, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
