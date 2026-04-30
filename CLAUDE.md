# CLAUDE.md

simACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (shared environment), **E** (unique environment) variance components. Provides simulate → phenotype → censor → sample → validate → stats → plot. Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan) lives in the sister repo [fitACE](../fitACE), which depends on simace.

## Repo Map

Four related repos, all under `rwaples/` on GitHub. simACE is the umbrella working directory; the others are nested checkouts (gitignored from simACE — no submodules).

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → sample → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan, PCGC). Consumes simACE outputs. |
| [`ace_iter_reml`](https://github.com/rwaples/ace_iter_reml) | private | `./fitACE/fitace/ace_iter_reml/` | C++ PCG-AI-REML binary. Driven by `fitACE/fitace/iter_reml/`. |
| [`tetraher_simace`](https://github.com/rwaples/tetraher_simace) | private | `./external/tetraher_simace/` | Fork of LDAK 6.2 (grouping + warm-start + OMP opt-in). Binary consumed by `fitACE/fitace/tetraher/`. |

Each nested repo has its own `origin` wired to the matching GitHub repo — `git push` from inside each directory goes to the right place. Build artifacts (`build-fp*/`, `ldak6.2.simace`, Stan binaries) are gitignored — rebuild from source.

## Key Rules

- The ACE conda env is always active. Do NOT use `conda run -n ACE` — run commands directly.
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Commit when explicitly asked
- Prefer batching commits at the end of a session — show changed files grouped by purpose, then commit per logical change

## Snakemake

- Root `Snakefile` is the entry point — not `-s workflow/Snakefile`
- Use `--cores 4` (testing) or `--cores 1` (debugging). Always dry-run (`-n`) before long runs.
- Targets are per-scenario: `results/{folder}/{scenario}/{scenario,simulate,phenotype,validate,stats}.done`
- Force-rebuild plot atlas: `snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.pdf`

## Testing

- Full suite: `pytest tests/ -v`
- Single module: `pytest tests/simulation/test_simulate.py -v`
- Run relevant tests before committing
- Smoke test: `snakemake --cores 4 results/test/small_test/scenario.done`

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `simace/plotting/plot_atlas.py`

## Project Layout

- `simace/` — simulation package (`pip install -e .`), organized into sub-packages:
  - `core/` — shared infrastructure: `pedigree_graph`, `compute_hazard_terms`, `cli_base`, `numerics`, `parquet`, `relationships`, `schema`, `yaml_io` (utils.py was split into intent-named modules)
  - `simulation/` — pedigree simulation
  - `phenotyping/` — phenotype models: `phenotype.py` (frailty), `threshold.py`, `hazards.py`, plus a `models/` sub-package of model classes inheriting from a `PhenotypeModel` ABC
  - `censoring/` — age-window and death censoring
  - `sampling/` — dropout and subsampling
  - `analysis/` — `stats/` (package: censoring, correlations, incidence, pedigree, sampling, tetrachoric, runner), `validate.py`, `gather.py`
  - `plotting/` — all plot modules and plot utilities
- `workflow/rules/simace/*.smk` — Snakemake rules; `workflow/scripts/simace/` — thin script wrappers
- `config/_default.yaml` — default parameters; `config/{folder}.yaml` — per-folder scenario files (auto-discovered; files starting with `_` are skipped)
- `results/{folder}/{scenario}/` — output per scenario

## Versioning

- **CalVer** (`YYYY.MM`) via `setuptools-scm`, derived from git tags
- Version is the single source of truth from git tags — never hardcode it
- Tag format: `v2026.03`, `v2026.04`, `v2026.04.1` (second release same month)
- Between tags, versions look like `2026.4.dev4+g<hash>` (dev build, 4 commits after tag)
- To cut a release: `git tag -a v2026.MM -m "description"`
- Check current version: `python -c "import simace; print(simace.__version__)"`

## Linting

- Check: `ruff check`
- Auto-fix: `ruff check --fix`
- Format Python: `ruff format`
- Format Snakemake: `snakefmt workflow/rules/**/*.smk Snakefile`

## Documentation & Citations

- Never generate citations, DOIs, author lists, journal names, years, page numbers, or any bibliographic field from memory. Memory recall of bib metadata is treated as fabrication.
- Verify every entry against a live source before writing it: resolve `https://doi.org/<doi>` via WebFetch, or pull from Crossref/PubMed/publisher page. Confirm the DOI resolves AND that the returned title/authors/year match the citation you are about to write.
- One verification per field set — do not extrapolate. If you verified the DOI but not the author list, the author list is still unverified.
- If verification fails (DOI 404s, source unreachable, ambiguous match), do NOT write the entry. Insert a `% TODO: verify <what>` placeholder and tell the user which entries could not be verified.
- Apply this to BibTeX, inline references in docs, citation comments in code, and prose mentions in notes. Same rule everywhere.

## Implementation Approach

- For non-trivial implementation tasks, propose 2-3 approaches with tradeoffs before writing code. Wait for approval.

## Design Interviews

- When starting a design interview or /grill-me session, if there is no existing plan, first explore the relevant codebase (grep for related modules, read key files) before asking questions. Ground the interview in what the code actually does.

## Session Management

- Prefer focused sessions (one feature per session)
- Run pipeline commands in background when >30 seconds
- Use targeted line ranges instead of reading entire large files

## Performance Optimization

- Always profile/benchmark first to identify the actual bottleneck before implementing changes
- Do not assume which component is slow — show profiling data before proposing a solution
- When narrowing numeric dtypes for memory optimization, never narrow below int32 for generation columns and always verify float precision doesn't break existing test tolerances before committing
