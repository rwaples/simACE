# CLAUDE.md

ACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (shared environment), **E** (unique environment) variance components. Weibull frailty model fitting is in a separate project at `~/Documents/fitACE/`.

## Key Rules

- The ACE conda env is always active. Do NOT use `conda run -n ACE` — run commands directly.
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Only commit when explicitly asked

## Snakemake

- Root `Snakefile` is the entry point — never use `-s workflow/Snakefile`
- Use `--cores 4` (testing) or `--cores 1` (debugging). Always dry-run (`-n`) before long runs.
- Targets: `simulate_all`, `phenotype_all`, `validate_all`, `stats_all`
- Single scenario: `snakemake --cores 4 results/{folder}/{scenario}/scenario.done`
- Force-rebuild plots: `snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.pdf`

## Testing

- Full suite: `pytest tests/ -v`
- Single module: `pytest tests/test_simulate.py -v`
- Run relevant tests before committing
- Smoke test: `snakemake --cores 4 results/test/small_test/scenario.done`

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `plot_atlas.py`

## Project Layout

- `sim_ace/` — installable package (`pip install -e .`) with simulation, phenotyping, validation, stats, and plotting modules
- `workflow/rules/*.smk` — Snakemake rules; `workflow/scripts/` — thin script wrappers
- `config/config.yaml` — named scenarios (seed, folder, A, C, N, G_ped, G_sim, fam_size, p_mztwin, p_nonsocial_father)
- `results/{folder}/{scenario}/` — output per scenario

## Session Management

- Prefer focused sessions (one feature per session)
- Run pipeline commands in background when >30 seconds
- Use targeted line ranges instead of reading entire large files
