# CLAUDE.md

ACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (shared environment), **E** (unique environment) variance components. The repo contains two packages: `sim_ace` (simulation, analysis, plotting) and `fit_ace` (statistical model fitting: EPIMIGHT, PA-FGRS, Weibull correlation, Stan models).

## Key Rules

- The ACE conda env is always active. Do NOT use `conda run -n ACE` — run commands directly.
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Only commit when explicitly asked

## Snakemake

- Root `Snakefile` is the entry point — never use `-s workflow/Snakefile`
- Use `--cores 4` (testing) or `--cores 1` (debugging). Always dry-run (`-n`) before long runs.
- Targets: `simulate_all`, `phenotype_all`, `validate_all`, `stats_all`, `epimight_all`
- Single scenario: `snakemake --cores 4 results/{folder}/{scenario}/scenario.done`
- Force-rebuild plots: `snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.pdf`
- Single EPIMIGHT kind: `snakemake --cores 4 results/{folder}/{scenario}/rep1/epimight/tsv/h2_d1_FS.tsv`

## Testing

- Full suite: `pytest tests/ -v`
- Single module: `pytest tests/simulation/test_simulate.py -v`
- Run relevant tests before committing
- Smoke test: `snakemake --cores 4 results/test/small_test/scenario.done`

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `sim_ace/plotting/plot_atlas.py`

## Project Layout

- `sim_ace/` — simulation package (`pip install -e .`), organized into sub-packages:
  - `core/` — shared infrastructure (utils, pedigree_graph, compute_hazard_terms, weibull_mle, cli_base)
  - `simulation/` — pedigree simulation
  - `phenotyping/` — frailty/threshold phenotype models
  - `censoring/` — age-window and death censoring
  - `sampling/` — dropout and subsampling
  - `analysis/` — stats, validation, Falconer h², survival correlation wrappers, gather
  - `plotting/` — all plot modules and plot utilities
- `fit_ace/` — model fitting package (`pip install -e fit_ace/`), organized into sub-packages:
  - `pafgrs/` — PA-FGRS genetic risk scores
  - `epimight/` — EPIMIGHT heritability pipeline (separate `epimight` conda env; uses `conda run -n epimight`)
  - `stan/` — Stan-based model fitting
  - `plotting/` — PA-FGRS and EPIMIGHT bias plots
- `workflow/rules/*.smk` — Snakemake rules; `workflow/scripts/` — thin script wrappers
- `config/config.yaml` — default parameters; `config/{folder}.yaml` — per-folder scenario files (auto-discovered)
- `results/{folder}/{scenario}/` — output per scenario

## Versioning

- Both packages use **CalVer** (`YYYY.MM`) via `setuptools-scm`, derived from git tags
- Version is the single source of truth from git tags — never hardcode it
- Tag format: `v2026.03`, `v2026.04`, `v2026.04.1` (second release same month)
- Between tags, versions look like `2026.4.dev4+g<hash>` (dev build, 4 commits after tag)
- To cut a release: `git tag -a v2026.MM -m "description"`
- Check current version: `python -c "import sim_ace; print(sim_ace.__version__)"`

## Linting

- Check: `ruff check`
- Auto-fix: `ruff check --fix`
- Format Python: `ruff format`
- Format Snakemake: `snakefmt workflow/rules/*.smk Snakefile`

## Session Management

- Prefer focused sessions (one feature per session)
- Run pipeline commands in background when >30 seconds
- Use targeted line ranges instead of reading entire large files

## Performance Optimization

- Always profile/benchmark first to identify the actual bottleneck before implementing changes
- Do not assume which component is slow — show profiling data before proposing a solution
