# CLAUDE.md

simACE simulates multi-generational pedigrees with **A** (additive genetic), **C** (common environment), **E** (unique environment) variance components: simulate → phenotype → censor → ascertain → analyze (validate + stats) → plot. Model fitting lives in the sister repo [fitACE](./fitACE), which depends on simace.

## Project Layout

- `simace/` — the package, one sub-package per pipeline stage:
  - `cli/` — the `simace` command: `run.py` (`simace run`), `layout.py` (the `results/` path convention), `stages.py` (stage argv), `manifest.py` (per-rep `run.yaml`)
  - `core/` — shared infrastructure (schemas, parquet/yaml I/O, CLI helpers, atomic `publish`, numerics, hazard terms, pedigree filtering)
  - `simulation/` — pedigree simulation (household assignment in `simulate.py`)
  - `phenotype/` — `runner.py` (`run_phenotype` dispatcher), `hazards.py`, `blended_post.py`, and `models/` (subclasses of `PhenotypeModel`; the liability-threshold idiom is in `models/_prevalence.py`)
  - `censoring/` — age-window and death censoring
  - `ascertainment/` — unified dropout + case-weighted N_sample selection (ADR 0001)
  - `analysis/` — `stats/`, `validate/` (one module per check family), `analyze.py`, `report.py`, `gather.py`
  - `plotting/` — plot modules; atlas page order is in `atlas_manifest.py`
- `scripts/examples/`, `scripts/gene_drop/` — standalone example-comparison and gene-drop scripts (not run by `simace run`)
- `config/_default.yaml` — defaults; `config/{folder}.yaml` — scenario files (auto-discovered; `_`-prefixed files skipped)
- `results/{folder}/{scenario}/` — output per scenario

## Environment

Everything runs through pixi; there is no ambient env (ADR 0016, 0018).

- simACE: `pixi run <cmd>` at the repo root. `pixi install --locked` materializes `.pixi/`.
- fitACE family work (incl. `tools/typecheck_family.py`): `pixi run --manifest-path fitACE/pixi.toml <cmd>`.
- pedigree-graph: its own manifest in `external/pedigree-graph/`.
- Dedicated conda envs (`epimight-2.1`, `ace_iter_reml*`, `ace_sreml`) are still invoked by name.
- `pixi.toml` is the pin source. Never rewrite `pixi.lock` as a side effect; upgrades are deliberate (`pixi lock` after a manifest edit, then review the diff).

## Running the pipeline

- `pixi run simace run <scenario>` computes every rep, then the scenario plots and atlas (ADR 0020). `--dry-run` prints the exact stage commands. `--jobs N` runs N reps at once (each child pinned to one thread); `--rep 2 3` limits the reps. `--max-memory 8G` kills any stage process whose RSS goes over 8 GiB (polled from `/proc`, per stage, not per run).
- A rep is all or nothing. Its `run.yaml` is written only after every stage exits 0. A rerun skips reps whose `run.yaml` matches the current config, refuses reps whose config changed (`--force` recomputes), and always rebuilds plots and the atlas. Code changes never make a rep stale; `simace ls` flags reps built by another simace version. One run per scenario at a time (`flock` on `results/{folder}/{scenario}/.run.lock`).
- `pixi run simace gather <folder>` writes `report_summary.tsv` and the validation atlas for a folder. `--format pdf` gives the PDF atlas on either command.
- Stage subcommands (`simace simulate|phenotype|censor|ascertain|analyze|plot|atlas|...`) take explicit paths and flags and never read config. `simace show <scenario>` and `simace ls [folder]` inspect config and rep state.
- Per-rep stage wall time and peak RSS land in `results/{folder}/{scenario}/rep{rep}/timing.tsv`; stage logs in `logs/{folder}/{scenario}/rep{rep}/{stage}.log`.
- After changing a `plot_*.py`, rerun `pixi run simace run <scenario>` (complete reps are skipped; plots and atlas always rebuild) and check labels/titles fit.
- `use_gene_drop` / `drop_from` scenarios are not run by `simace run`; see `scripts/gene_drop/`.

## Testing and Linting

- Full suite: `pixi run test` (6 xdist workers, `--dist worksteal`, one thread per worker). Extra args pass through: `pixi run test tests/simulation -x`. The rationale for these settings is in the comment above `[tasks.test]` in `pixi.toml`; re-measure with `tools/bench_pytest_workers.sh` before changing them.
- Serial/debug (`-v`, `-s`, `--pdb`, single modules): `pixi run pytest tests/ -v`.
- Smoke test: `pixi run simace run small_test && pixi run simace gather test`
- Run relevant tests before committing.
- `ruff check` with **no extra `--select`** — it discards the `ignore`/`per-file-ignores` in `pyproject.toml` and surfaces false positives.

## Statistical-correctness gotchas

Bugs that have actually occurred. Check these whenever touching the relevant code.

1. **Relationship classification.** Full vs half needs two shared ancestors through a mated pair (`>= 2`, not `> 0`), so ancestor multiplicity must survive until that test; zero generations up is the individual, not a parent hop; a lower degree that a higher one depends on must be computed even when the caller's cutoff stops below it. Production classification is the Rust engine in `external/pedigree-graph/crates/core/src/relationships/` (invariants in pedigree-graph ADR 0010); the old SciPy extractor is its differential oracle at `external/pedigree-graph/tests/oracle/relationship_pairs.py`.
2. **Cross-package coupling.** simace and `fitace` both import `RELATIONSHIPS` and `PedigreeGraph.relationship_pairs` from the external `pedigree_graph` package. Changes there silently bias fitACE heritability and PA-FGRS. `fitace/relationships.py` must stay in sync with `RELATIONSHIPS[code].nominal_kinship`.
3. **Generation-dependent C/E variance** can bias `rho_w` (assortative mating correlation).
4. **`affected = NOT (age_censored OR death_censored)`** must hold through any censoring change.
5. **Pair key `lo * max_id + hi`** (int64) needs canonical `lo < hi` and overflows beyond ~3B individuals.

### Expected liability correlations

`r = 2 * kinship * A + C_shared`, where `C_shared = C` only if the pair shares a household. **Household is assigned by mother** (`simulate.py`: `np.unique(parent_idxs[:, 0])`), so maternal half-sibs share C and paternal half-sibs do not.

MZ = A+C, FS = 0.5A+C, MHS = 0.25A+C, PHS = 0.25A, PO = 0.5A.

Kinship source of truth: `RELATIONSHIPS[code].nominal_kinship` in `pedigree_graph/_registry.py`. Under inbreeding, `PedigreeGraph.pair_kinship()` returns per-pair float32 values; comparisons against a float64 recurrence need a tolerance.

## Repo Map

Five repos under `rwaples/` on GitHub. simACE is the umbrella working directory; the others are checkouts inside it, gitignored from simACE (no submodules). Method packages depend on `fitace`/`simace`, never on each other (fitACE ADR 0001).

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| `simACE` | public | `.` | Simulation pipeline |
| `fitACE` | private | `./fitACE/` | Model-fitting monorepo: core, its own workflow orchestrator (reads simACE `results/`), and `fitACE_<x>/` method packages (PCGC, iter/sparse REML + `ace_iter_reml` C++, TetraHer + `tetraher_simace` LDAK fork, PA-FGRS, Stan, frailty). See `fitACE/CLAUDE.md`. |
| `fitACE_epimight` | private | `./fitACE/fitACE_epimight/` | EPIMIGHT integration; own repo tracking the BioPsyk/epimight upstream, included by `fitACE/Snakefile`. |
| `pedigree-graph` | public | `./external/pedigree-graph/` | Rust relationship extraction and kinship (Python + R bindings) |
| `pedsum` | public | `./external/pedsum/` | Pedigree summary CLI built on pedigree-graph |

When work spans simACE, fitACE, and fitACE_epimight: check `git status` in each, run tests in each, and make parallel commits. Changes do not propagate between checkouts.

## Git

- Push only through the user-invoked `/push` skill. Never run `git push` directly; the permission rules deny it.
- No `Co-Authored-By` lines in commit messages.
- Commit only when asked; batch changed files into commits by purpose.

## Versioning

simACE, fitACE, and fitACE_epimight share one lockstep CalVer (`vYYYY.MM[.patch]`, setuptools-scm from git tags; ADR 0012, 0017). Everything inside fitACE, including the `ace_iter_reml` binary, reads fitACE's tag. Compatibility is one `FAMILY_FLOOR` in `fitace._deps`, enforced by `test_dependency_floors`. pedigree-graph and pedsum version independently. To cut a release, invoke the `coordinated-release` skill; don't work from memory.

## Planning

- For non-trivial work (multi-file, cross-repo, or open design questions), explore the code first, then propose 2-3 approaches with tradeoffs and wait for approval. For plans and refactors, default to the `grill-with-docs` skill and lock each decision explicitly. Skip this for bugfixes, doc tweaks, and renames.
- Never exit plan mode without an explicit go-ahead.
- Drafts go in `plans/<slug>.md` (gitignored); state the absolute path in chat. Never overwrite an existing plan — add `-v2` or ask. Promote finished plans to `docs/plans/`, locked architectural decisions to `docs/adr/`. See `plans/README.md`.
- Treat every coupling/structural claim as a hypothesis until backed by `file:line`; list what you couldn't confirm. Hold subagents to the same standard.
- Enumerate and verify any formula, threshold, complexity, or memory-model assumption against the primary source and the code before relying on it.

## Performance

- Profile before optimizing.
- Never narrow numeric dtypes below int32/float32.

## Citations

- Never write any bibliographic field (DOI, authors, journal, year, pages) from memory.
- Verify each entry against a live source (`https://doi.org/<doi>`, Crossref, PubMed, publisher) and confirm title/authors/year match.
- If verification fails, write `% TODO: verify <what>` instead and tell the user.

## Agent docs

Skills are authored in `.agents/skills/` (Claude Code, Codex, Pi). Issues: GitHub Issues via `gh`, with canonical triage labels. Domain docs: `CONTEXT.md` + `docs/adr/`. Details in `docs/agents/`.
