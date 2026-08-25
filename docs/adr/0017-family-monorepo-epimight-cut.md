# ADR 0017: Family monorepo with the epimight cut (13 repos → 5)

## Status

Accepted and **implemented** 2026-08-21 (design interview / grill the same day).
The staged scratch build validated and the live checkouts were swapped: the six
Python method sisters, the `ace_iter_reml` C++ source, and the `tetraher_simace`
LDAK fork now live in the fitACE monorepo at their original directory names,
`fitACE_epimight` remains its own nested repo, and the eight retired GitHub
repos are archived read-only. Lockstep tagging is down to the three checkouts
(`tools/release.py`, `tools/family_repos.py::lockstep_repos`). Amends ADR 0012
(lockstep membership: ten repos → three) and unblocks
[ADR 0018](0018-retire-conda-family-environment.md), which retired the conda
family environment the same day. Decision log: `plans/repo-organization.md`
(session-local draft).

## Context

The family is developed as one coordinated set inside the simACE umbrella:
thirteen repos, nine of them private nested checkouts (fitACE core + seven
`fitACE_*` method sisters + the `ace_iter_reml` C++ binary), gitignored from
the umbrella — no submodules. Two prior decisions already fixed the sharpest
pains of that shape: ADR 0012 cured version drift with lockstep CalVer over
the ten family repos, and fitACE ADR 0001 locked a clean core↔sister *code*
boundary (core never imports a sister; sisters never import each other).
Neither addressed the repo count itself, and two frictions remained, both
evidenced concretely:

1. **The family environment definition has no honest owner.** The pixi
   family-layer spike produced a working manifest (nine editable path deps),
   but its paths encode the umbrella nesting layout that no single repo
   describes — it could not be committed anywhere truthfully. This blocks the
   ADR 0016 endgame (conda retirement). Today the definition is smeared
   across `envs/environment-fitace.yml` plus a manual nine-editable install
   choreography in RELEASE.md.
2. **N-way commit/push/tag choreography.** One logical change fans out into
   up to seven commits in seven repos with seven manual pushes (the polars
   ty-triage, issue #13, was exactly this), and every release tags ten repos.

Notably, fitACE ADR 0001 recorded the *package* boundary but no rationale for
the sisters being separate *repos* — the boundary invariants are
package-level and survive any repo arrangement. There is almost no CI coupled
to the current structure (one docs workflow, one publish workflow, both on
repos this decision does not touch).

Hard constraints: simACE is public and the family is private, so at least two
repos must exist; `pedigree-graph` (PyPI) and `pedsum` are public with
independent lives.

## Decision

1. **Consolidate to five repos — but not to one.** fitACE becomes a private
   monorepo absorbing the six Python method sisters (`pcgc`, `iter_reml`,
   `tetraher`, `pafgrs`, `stan`, `frailty`), the `ace_iter_reml` C++ source,
   and the `tetraher_simace` LDAK fork. **`fitACE_epimight` stays its own
   repo**, nested inside fitACE as today: it is the one member with an
   independent external identity (tracks the BioPsyk/epimight R upstream,
   dedicated R environments, its own investigation cadence and upstream bug
   reports). End state: simACE, fitACE, fitACE_epimight, pedigree-graph,
   pedsum.

2. **Repo move ≠ packaging change.** The seven Python distributions keep
   their own `pyproject.toml` files in their monorepo subdirectories and
   install as seven editables from one repo. No import churn; `FAMILY_FLOOR`,
   `test_dependency_floors`, and every fitACE ADR 0001 invariant survive
   verbatim (as package rules, which is what they always were). Collapsing to
   one distribution remains a separate, later decision.

3. **Squash imports at today's paths.** Each absorbed repo lands as one
   import commit at its current directory name, so the working tree is
   byte-identical to today's layout (Snakemake cross-repo `include:` chains
   and script paths unchanged) — except `tetraher_simace`, which moves from
   the umbrella's `external/` into the monorepo. The sisters' standalone
   histories are only ~3 months old (method-split 2026-05/06); their pre-split
   history is already fitACE history.

4. **Retired repos are archived, never deleted.** The eight retired GitHub
   repos each get a final pointer-README commit ("merged into rwaples/fitACE
   <date> — history preserved here") and are then archived read-only. With
   squash imports they are the permanent history record.

5. **The family manifest lands at the monorepo root**: seven internal
   editable path deps plus exactly one documented external reference
   (`fitACE_epimight`, nested gitignored checkout). It supersedes
   `envs/environment-fitace.yml` and RELEASE.md's manual editable step, and
   unblocks the conda-retirement decision.

6. **Staged migration.** All unpushed work is pushed first; the merged repo
   is built and fully validated in a scratch clone (all suites,
   `tools/typecheck_family.py`, orchestrator dry-run, locked pixi solve)
   before the live checkout is swapped; pushes and archiving are the
   maintainer's final step.

## Considered Options

- **Single full monorepo (epimight folded in too).** Rejected: buries the one
  member with a genuine independent external identity and cadence.
- **Core/methods two-repo cut** (lean fitACE + one `fitACE_methods`).
  Rejected: re-splits the ADR 0001 §2 config-registry seam across a repo
  boundary — every sister-parameter rename becomes a two-repo change again.
- **Status quo + batch tooling.** Rejected: scripts soften the choreography
  but the family manifest still has no honest owner, which is driver #1.
- **Submodules / meta-manifest workspace repo.** Rejected: makes the layout
  reproducible but collapses nothing — commit fan-out remains, pin-bump churn
  is added.
- **Keeping the natives out** (`ace_iter_reml` as nested repo,
  `tetraher_simace` external). Rejected after inspection: the binary is
  single-consumer, 38 files, already lockstep; the LDAK fork has no upstream
  git remote — it is a source-drop import plus four local patches, a story
  `git log -- <subdir>` preserves, and a future LDAK upgrade is the same
  import-and-reapply in a subdirectory.
- **Full-history path-rewrite merge** instead of squash. Rejected as not
  worth the ceremony given the ~3-month standalone histories and the archived
  originals.

## Consequences

- Lockstep tags drop from ten repos to three (simACE, fitACE,
  fitACE_epimight), and within the monorepo drift becomes structurally
  impossible — all seven distributions read the same setuptools-scm tags;
  the `ace_iter_reml` CMake `git describe` reads monorepo tags.
  `tools/release.py` shrinks accordingly.
- `tetraher_simace` gives up its independent version (its ADR 0012 exclusion
  ends) and rides family tags.
- One logical family change is one commit and one push (plus epimight when
  touched).
- Tooling updated in the migration: `tools/typecheck_family.py` checkout
  list, the code-review-graph registry, CLAUDE.md repo maps (simACE +
  fitACE), RELEASE.md, `tools/release.py`.
- This deliberately reverses the *repo-per-sister form* of the method-split
  while preserving its *boundary* (fitACE ADR 0001 stands unchanged) — the
  split's value was the placement rule and import invariants, which are
  package-level, not the repo count.
