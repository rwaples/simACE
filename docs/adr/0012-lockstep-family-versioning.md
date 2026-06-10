# ADR 0012: Lockstep CalVer versioning across the simACE/fitACE family

## Status

Accepted. Design interview 2026-06-10.

## Context

Ten git repos are developed as one coordinated set inside the simACE umbrella
working directory, all `pip install -e` (editable) and **none published to
PyPI**: simACE, fitACE core, the seven `fitACE_*` method sisters
(`epimight`, `pcgc`, `iter_reml`, `tetraher`, `pafgrs`, `stan`, `frailty`), and
the nested `ace_iter_reml` C++ binary.

Each was versioned **independently** — CalVer via setuptools-scm for the Python
repos, CMake for the binary. In practice that produced drift and confusion:
simACE was at `v2026.05.3`, fitACE at `v2026.04`, the **seven sisters were
untagged** (so they emitted `0.1.devN` / setuptools-scm "next release" guesses),
and the binary at `v2026.04`. Installed `fitace` resolved to `2026.5.dev…`,
*colliding* with simACE's real `2026.5.x` line. Cross-repo compatibility was
held together by a two-number floor system in `fitace._deps`
(`MIN_SIMACE="2026.05"`, `MIN_FITACE="2026.04"`) whose own docstring conceded
the two numbers differed *only* because the repos released on separate cadences.

The external dependencies (`pedigree-graph`, `pedsum`, `tetraher_simace`) are
public and/or independently consumed and keep their own (SemVer) versions.

## Decision

1. **Lockstep family.** The ten repos above share **one** version. The external
   dependencies are excluded and keep independent versions.

2. **True lockstep (one identical version).** A release tags every member at the
   same `vYYYY.MM`, including untouched members. Compatibility becomes a string
   equality, not a floor comparison.

3. **Scheme stays CalVer** (`vYYYY.MM[.patch]`), not SemVer. Lockstep already
   cures the drift regardless of scheme, and a new `vYYYY.MM` tag sorts above all
   prior tags, so there is no PEP 440 downgrade to engineer around. The first
   unified release tags all members at `v2026.06`.

4. **Mechanism: coordinated setuptools-scm tags** (git-describe via a CMake
   `configure_file` for the binary). A release helper creates the tags locally;
   the maintainer pushes (per the repo-wide no-`git push` rule). Members are
   byte-identical only *at* a tagged release; between releases each repo's dev
   build diverges only by its setuptools-scm commit-distance suffix — accepted
   as cosmetic.

5. **One Family floor.** `fitace._deps` collapses `MIN_SIMACE` / `MIN_FITACE`
   into a single `FAMILY_FLOOR`. The `pyproject` pins (`simace>=`, `fitace>=`),
   the `test_dependency_floors` consistency test, and the `fitace.config`
   runtime guard all reference it with `>=` semantics (a dev build of a later
   release still passes). simACE is upstream of the floor and does not import it.

6. **Runtime version strings.** Every family Python package exposes
   `__version__` (the `importlib.metadata.version(...)` one-liner simACE already
   uses). A shared `cli_base.add_version_arg(parser, dist)` wires `--version`
   into the nine existing console entry points; the C++ binary gains
   `project(VERSION)` + a `--version` flag.

7. **Provenance: stamp the real per-layer versions** (not a single
   `family_version`, which would be fiction on divergent dev builds). simACE's
   `params.yaml` records `simace_version`; the Fit-result metadata sidecar
   records `simace_version` + `fitace_version` (added in the core-owned
   `FitRunContext.base_meta`); each method-sister adapter stamps its own
   `<package>_version`; the iter_reml adapter additionally stamps
   `ace_iter_reml_version`.

## Considered Options

- **Independent + floor pins (status quo).** Rejected: the drift and the stale
  `fitace 2026.5.dev` collision are precisely the problem.
- **Decoupled "family version" above per-repo versions (Model 2).** Rejected:
  two numbers per package; does not deliver one shared version.
- **Switch to SemVer, reset to `v1.0.0`.** Rejected: CalVer→SemVer is a PEP 440
  numeric *downgrade* (`2026.5.3 > 1.0.0`) that needs an epoch (`1!1.0.0`) for a
  monotonicity guarantee that is moot off-PyPI; CalVer is the established
  convention and lockstep alone fixes the drift that motivated the change.
- **Static shared version string read by all repos (D2b).** Rejected:
  reintroduces manual version editing across ten separate origins — the chore
  setuptools-scm exists to remove.
- **Exclude the `ace_iter_reml` binary.** Rejected: iter_reml is the one method
  whose computation *is* a compiled binary, so end-to-end provenance would have a
  conspicuous hole. The cost is the family's only C++/CMake version machinery.
- **A single `family_version` provenance field.** Rejected: no canonical single
  number exists between releases; stamp the real per-layer versions instead.

## Consequences

- A release is one coordinated step: the helper tags ten repos; the maintainer
  pushes. Untouched members get a tag too — accepted.
- simACE's **public** version now bumps on family releases, including
  fit-method-only releases — the accepted coupling cost of folding simACE into
  the family.
- `fitace._deps` simplifies from two floors to one; bumping the floor is one edit
  per release, enforced across every family `pyproject.toml` by
  `test_dependency_floors`.
- Every on-disk result records the real version chain that produced it — honest
  even on divergent dev builds.
- `ace_iter_reml` gains `project(VERSION)` + a git-describe embed + `--version`.
- The "**six** `fitACE_*` method repos" prose in `CLAUDE.md` is corrected to
  **seven** (the repo-map table and disk already show seven).

## Non-goals

- Not lockstep-versioning the external dependencies (`pedigree-graph`, `pedsum`,
  `tetraher_simace`).
- Not publishing to PyPI — the off-PyPI assumption is load-bearing for skipping
  the PEP 440 epoch.
- Not changing any fitting behavior or any output schema beyond adding the
  provenance keys above.
