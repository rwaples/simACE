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

- Runs in the pixi env: `pixi run snakemake …` (ADR 0016)
- Root `Snakefile` is the entry point — not `-s workflow/Snakefile`
- Use `--cores 4` running one scenario, `--cores 8` for multiple scenarios, `--cores 1` for debugging. 
- Always dry-run (`-n`) before long runs.
- Targets are per-scenario: `results/{folder}/{scenario}/{scenario,simulate,phenotype,validate,stats}.done`
- Force-rebuild plot atlas (HTML is the default artifact; PDF is on-demand): `pixi run snakemake --cores 4 -f results/{folder}/{scenario}/plots/atlas.html` (or `.../atlas.pdf` for the PDF export)

## Plotting

- After modifying `plot_*.py`, force-regenerate the atlas to verify
- Check that labels/titles fit within figure bounds
- Page order is controlled in `simace/plotting/atlas_manifest.py`

## Key Rules

- **Environment routing (ADR 0016 + 0018): everything is pixi; no ambient env.** simACE-scoped commands run `pixi run <cmd>` at the umbrella root against the committed `pixi.lock` (`pixi install --locked` to materialize `.pixi/`). Family-scoped work (fitACE + method packages + epimight, `tools/typecheck_family.py`) runs `pixi run --manifest-path fitACE/pixi.toml <cmd>`. Editable pedigree-graph work uses its own manifest in `external/pedigree-graph/`. The old always-active `simACE` conda env is retired (ADR 0018); the dedicated conda envs (`epimight-master`, `ace_iter_reml*`, `ace_sreml`) remain, invoked by name.
- `pixi.toml` is the authoritative simACE pin source. Normal pixi commands must not rewrite `pixi.lock`; upgrades are deliberate (`pixi lock` after a manifest edit, review the diff).

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

Five repos, all under `rwaples/` on GitHub (ADR 0017 collapsed the former 13: fitACE became a private monorepo absorbing the six Python method sisters, the `ace_iter_reml` C++ source, and the `tetraher_simace` LDAK fork — the retired standalone repos are archived on GitHub with their pre-merge history). simACE is the umbrella working directory; fitACE and its nested fitACE_epimight are checkouts inside it (gitignored from simACE — no submodules). fitACE ADR 0001's package invariants are unchanged: method packages depend on `fitace`/`simace`, never on each other.

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → ascertainment → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model-fitting **monorepo**: core + Snakemake orchestrator + method packages in `fitACE_<x>/` subdirs (PCGC, iter/sparse REML + `ace_iter_reml` C++ source, TetraHer + the `tetraher_simace` LDAK fork, PA-FGRS, Stan, frailty). Seven distributions, one repo — see `fitACE/CLAUDE.md` for the in-repo layout. Consumes simACE outputs. |
| [`fitACE_epimight`](https://github.com/rwaples/fitACE_epimight) | private | `./fitACE/fitACE_epimight/` | EPIMIGHT integration: long-form input emitter, R driver, Snakemake rules, atlas/bias plotting. The one method outside the monorepo — its own repo, tracking the BioPsyk/epimight R upstream; included by `fitACE/Snakefile` via a cross-repo `include:`. |
| [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) | public | `./external/pedigree-graph/` | Sparse-matrix pedigree relationship extraction and kinship computation. |
| [`pedsum`](https://github.com/rwaples/pedsum) | public | `./external/pedsum/` | Pedigree summary CLI: structure, relatedness, inbreeding, Ne estimators. Built on `pedigree-graph`. |

## Cross-repo edits (simACE + fitACE + fitACE_epimight)

- Treat simACE, fitACE, and fitACE_epimight as a coordinated set when work spans them. Verify edits land in each repo's working tree (`git status` in each), run tests in each, and make parallel commits — do not assume changes propagate. Edits across fitACE's method packages are ordinary single-repo commits since the monorepo (ADR 0017).

## Git usage
- Do NOT run `git push` under any circumstances
- Do NOT include Co-Authored-By in commit messages
- Commit only when explicitly asked
- Prefer batching commits — changed files grouped by purpose

## Versioning
- **Lockstep family CalVer.** simACE, fitACE core, the seven `fitACE_*` method
  distributions, the `ace_iter_reml` binary, and the `tetraher_simace` LDAK fork
  share **one** CalVer (`vYYYY.MM[.patch]`), tagged together each release
  (ADR 0012, membership amended by ADR 0017). Since the monorepo that is
  **three tagged checkouts** — simACE, fitACE, fitACE_epimight — because
  everything inside fitACE reads fitACE's one tag. External deps
  (`pedigree-graph`, `pedsum`) keep their own versions.
- **CalVer** (`YYYY.MM`) via `setuptools-scm`, derived from git tags; the
  `ace_iter_reml` binary embeds `git describe` via CMake.
- Compatibility is one `FAMILY_FLOOR` in `fitace._deps` (`>=` semantics),
  enforced across every family `pyproject.toml` by `test_dependency_floors`.
- Cutting a release — tag formats, `tools/release.py`, the push handoff — is the
  `coordinated-release` skill's job. Invoke it rather than working from memory.

## Testing

- Full suite: `pixi run pytest tests/ -v`
- Single module: `pixi run pytest tests/simulation/test_simulate.py -v`
- Run relevant tests before commit
- Smoke test: `pixi run snakemake --cores 4 results/test/small_test/scenario.done`

## Linting

- Check: `pixi run ruff check`
- Auto-fix: `pixi run ruff check --fix`
- Format Python: `pixi run ruff format`
- Format Snakemake: `pixi run snakefmt workflow/rules/**/*.smk Snakefile`
- Run `ruff check` with **no extra `--select`**. The configured rules (incl. `D`/pydocstyle) plus the `ignore` and `per-file-ignores` in `pyproject.toml` are authoritative. Passing any `--select` (e.g. `--select D`) discards those ignores and surfaces false positives.

## Documentation & Citations

- Never generate citations, DOIs, author lists, journal names, years, page numbers, or any bibliographic field from memory. Memory recall of bib metadata is treated as fabrication.
- Verify every entry against a live source before writing it: resolve `https://doi.org/<doi>` via WebFetch, or pull from Crossref/PubMed/publisher page. Confirm the returned title/authors/year match the citation you are about to write.
- One verification per field set — do not extrapolate. 
- If verification fails (DOI 404s, source unreachable, ambiguous match), do NOT write the entry. Insert a `% TODO: verify <what>` placeholder and tell the user which entries could not be verified.

## Planning and Implementation

- For non-trivial implementation tasks, propose 2-3 approaches with tradeoffs before writing code. Wait for approval.
- For non-trivial plans/refactors (multi-file, cross-repo, or with unresolved design decisions), default to invoking the `grill-with-docs` skill before proposing implementation. Lock each design decision explicitly, then wait for an explicit go-ahead before calling `ExitPlanMode` — never auto-exit plan mode to present a plan as final. Skip grilling for bugfixes, doc tweaks, and renames.
- Write working plan drafts to `plans/<slug>.md` (kebab-case, gitignored) and state the **absolute path** in chat when you do. Never overwrite an existing plan file — add a `-v2` suffix (etc.) or ask first. Promote a finalized plan to `docs/plans/` (tracked) when it should join the record; a locked architectural decision belongs in `docs/adr/`. See `plans/README.md`.
- When starting a design interview or /grill-me session, if there is no existing plan, first explore the relevant codebase 
and read key files and related modules before asking questions. Ground the interview in what the code actually does.
- During code exploration, treat every dependency/coupling/structural claim as a hypothesis until cited: back it with the exact `file:line` that proves it, and list anything you couldn't confirm as unverified rather than asserting it. Hold `Task`/`Explore` subagents to the same standard in their instructions.
- When a plan relies on formulas, thresholds, complexity claims, or memory/allocation models, enumerate each such assumption explicitly and verify it against the primary source and the actual codebase before locking the decision. Treat quantitative claims recalled from memory as unverified; flag any you cannot confirm rather than proceeding on them.

## Performance Optimization

- Always profile/benchmark first to identify the actual bottleneck before implementing changes
- When narrowing numeric dtypes for memory optimization, never narrow below int32 or float32

## Agent skills

### Issue tracker

GitHub Issues at github.com/rwaples/simACE/issues, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
