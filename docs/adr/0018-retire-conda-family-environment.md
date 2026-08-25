# ADR 0018: Retire the conda family environment (Linux-only pipeline)

## Status

Accepted. Design interview (grill) 2026-08-21; implementation same day.
Closes the environment split ADR 0016 left open (amends ADR 0016).

Follow-through 2026-08-25: pedsum adopted decision 2's pattern (its own
committed manifest + lock), so the family now carries **four** pixi manifests,
not three. No decision changed — see the Consequences note below.

## Context

ADR 0016 made pixi canonical for simACE work but kept the always-active
`simACE` conda env as the *family environment*: fitACE + sisters as editables,
editable pedigree-graph (compat mode), and the release reinstall choreography.
Two later results removed the reasons for that split. ADR 0017's monorepo gave
the family a committed, validated pixi environment (`fitACE/pixi.toml`), and
issue #12 established that ty/pytest results are identical under conda and
pixi resolution — the conda env no longer did anything the pixi environments
could not.

What still depended on it, enumerated before this decision: the
`code-review-graph` hooks and MCP server (bare PATH-resolved commands), the
ruff PostToolUse hook, editable pedigree-graph development, RELEASE.md's
family-reinstall step, `envs/environment.yml`'s role as pin mirror (with ADR
0016's same-commit sync rule) and as the documented macOS path, and the
launch habit of activating the env before sessions.

A measured constraint: extending the pixi lock to `osx-arm64` fails to solve
(the manifest pins `mkl`, which has no Apple Silicon build); pixi-native macOS
support would need platform-conditional dependencies on a platform with no
local hardware to test.

Out of scope, unaffected: the dedicated conda envs invoked by name —
`epimight-master` (R) and the C++ build envs (`ace_iter_reml`,
`ace_iter_reml_fp32`, `ace_sreml`). Conda-the-tool survives; the *family env*
retires.

## Decision

1. **Delete `envs/` entirely; the simACE pipeline is Linux-only, explicitly.**
   All three recipes go (`environment.yml`, `environment-fitace.yml`,
   `environment-unpinned.yml`), and with them the same-commit pin-sync rule.
   macOS users keep the documented plain-pip *library* path; the pipeline is
   supported on Linux (and WSL2), full stop.
2. **pedigree-graph development becomes self-contained**: its own committed
   pixi manifest + lock in its (public) repo, installing itself editable with
   its test extra. ty sees first-party source, structurally ending the
   compat-mode-editable / extra-paths workarounds. The umbrella environments
   keep consuming the released PyPI wheel.
3. **code-review-graph moves to pipx** (pinned 2.3.3 with the torch-CPU /
   sentence-transformers recipe), so the hooks and `.mcp.json` keep their
   bare commands via `~/.local/bin` and the tool no longer lives, undeclared,
   in an env that rebuilds can silently drop. The ruff PostToolUse hook runs
   the repo-pinned ruff via pixi instead of whatever is ambient.
4. **No ambient environment.** Sessions launch from a bare shell; simACE work
   is `pixi run` at the umbrella root, family work is
   `pixi run --manifest-path fitACE/pixi.toml …`, and `pixi shell` covers
   interactive use. The conda env is deleted after a comfort window
   (`conda env remove -n simACE`), as a user step from a fresh session.

## Considered Options

- **Keep `environment.yml` as a best-effort macOS fallback.** Rejected: with
  no local conda use left, the recipe would drift unverified — a documented
  path nobody can vouch for is worse than an honest platform statement.
- **Pixi absorbs macOS** (platform-conditional mkl/tbb). Rejected: real
  maintenance for an untestable platform.
- **Umbrella `pg-dev` feature env** for editable pedigree-graph. Rejected: it
  commits a path fresh public clones lack, and re-imports the wheel-shadowing
  dance at every release.
- **A minimal "tools" conda env** for code-review-graph. Rejected: keeps the
  failure mode this retirement exists to end.
- **direnv auto-activation** to preserve the ambient-env feel. Rejected: adds
  a tool to replace a habit `pixi run` already made unnecessary.

## Consequences

- One environment story: four pixi manifests (simACE, fitACE,
  pedigree-graph, pedsum), each with a committed lock; no sync rule, no
  undeclared residents, no activation step. pedsum joined on 2026-08-25, after
  its named conda env was found stale enough that it could not collect the
  suite; the checks had been silently borrowing the umbrella env. Its
  `environment.yml` was **kept**, unlike simACE's `envs/` — the drift argument
  above turns on nobody being left to vouch for the recipe, and pedsum is a
  public repo whose outside users install from it. The stale `pedsum` conda env
  was removed.
- The macOS pipeline path is **dropped** — the sharpest cut here, and the
  reason this is an ADR. Revisit via pixi platform-conditional deps if a Mac
  ever matters.
- RELEASE.md's family reinstall step targets the pixi environments.
- cmdstanpy (runtime dep of the dormant `fitace_stan`) existed only in the
  conda env; it joins the fitACE manifest if Stan reactivates.
