# CLAUDE.md

simACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (common environment), **E** (unique environment) variance components. Provides simulate → phenotype → censor → ascertainment → validate → stats → plot. Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan) lives in the sister repo [fitACE](../fitACE), which depends on simace.


## Project Layout

- `simace/` — simulation package (`pip install -e .`), organized into sub-packages:
  - `core/` — shared infrastructure: `pedigree_graph`, `compute_hazard_terms`, `cli_base`, `numerics`, `parquet`, `pedigree_filter`, `relationships`, `schema`, `yaml_io`
  - `simulation/` — pedigree simulation
  - `phenotype/` — `__init__.py` (run_phenotype dispatcher), `threshold.py`, `hazards.py`, plus a `models/` sub-package of model classes inheriting from a `PhenotypeModel`
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
- Force-rebuild plot atlas: `snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.pdf`

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `simace/plotting/atlas_manifest.py`

## Key Rules

- The ACE conda env is always active. Do NOT use `conda run -n ACE` — run commands directly.

## Repo Map

Five related repos, all under `rwaples/` on GitHub. simACE is the umbrella working directory; the others are nested checkouts (gitignored from simACE — no submodules).

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → ascertainment → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan, PCGC). Consumes simACE outputs. |
| [`ace_iter_reml`](https://github.com/rwaples/ace_iter_reml) | private | `./fitACE/fitace/ace_iter_reml/` | C++ PCG-AI-REML binary. Driven by `fitACE/fitace/iter_reml/`. |
| [`tetraher_simace`](https://github.com/rwaples/tetraher_simace) | private | `./external/tetraher_simace/` | Fork of LDAK 6.2 (grouping + warm-start + OMP opt-in). Binary consumed by `fitACE/fitace/tetraher/`. |
| [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) | public | `./external/pedigree-graph/` | Sparse-matrix pedigree relationship extraction and kinship computation. |

## Git usage
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Commit only when explicitly asked
- Prefer batching commits — changed files grouped by purpose

## Versioning
- **CalVer** (`YYYY.MM`) via `setuptools-scm`, derived from git tags
- Tag format: `v2026.03`, `v2026.04`, `v2026.04.1` (second release same month)
- Between tags: `2026.4.dev4+g<hash>`
- To cut a release: `git tag -a v2026.MM -m "description"`

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

## Documentation & Citations

- Never generate citations, DOIs, author lists, journal names, years, page numbers, or any bibliographic field from memory. Memory recall of bib metadata is treated as fabrication.
- Verify every entry against a live source before writing it: resolve `https://doi.org/<doi>` via WebFetch, or pull from Crossref/PubMed/publisher page. Confirm the returned title/authors/year match the citation you are about to write.
- One verification per field set — do not extrapolate. 
- If verification fails (DOI 404s, source unreachable, ambiguous match), do NOT write the entry. Insert a `% TODO: verify <what>` placeholder and tell the user which entries could not be verified.

## Planning and Implementation

- For non-trivial implementation tasks, propose 2-3 approaches with tradeoffs before writing code. Wait for approval.
- When starting a design interview or /grill-me session, if there is no existing plan, first explore the relevant codebase 
and read key files and related modules before asking questions. Ground the interview in what the code actually does.

## Performance Optimization

- Always profile/benchmark first to identify the actual bottleneck before implementing changes
- Do not assume which component is slow — show profiling data before proposing a solution
- When narrowing numeric dtypes for memory optimization, never narrow below int32 or float32

## Session Management

- Prefer focused sessions (one feature per session)
- Run pipeline commands in background when >30 seconds
- Use targeted line ranges instead of reading entire large files

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, and gives
you the callers, dependents, and test coverage directlty.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## Agent skills

### Issue tracker

GitHub Issues at github.com/rwaples/simACE/issues, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
