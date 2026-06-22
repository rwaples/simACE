# ty Type Checking Rollout (implemented)

## Goal

Run the [`ty`](https://docs.astral.sh/ty/) type checker across the simACE/fitACE
family **locally, as part of the commit gate — not in CI**. The high-value
target is *silent cross-repo / cross-module signature drift* (a symbol renamed,
moved, or removed in a sibling package), which fails with no exception and is
exactly what tests tend to miss. ty does **not** catch the statistical-
correctness bugs (array-heavy code uses bare `np.ndarray`); those remain the job
of the test suites.

## Decisions (locked via design interview)

- **Hook point:** the `/commit` skill gate (`.agents/skills/commit/SKILL.md`),
  alongside `ruff`/`pytest`. Not CI, not a `pre-commit` framework, not the
  `code-review-graph`-owned `.git/hooks/pre-commit`.
- **No CI / no GitHub Actions** — deliberate (ty is 0.0.x with churning
  diagnostics; nested repos have no CI). Revisit only if ty reaches a stable API.
- **Drift-only gate:** the **only** blocking diagnostic is `unresolved-import`.
  Every other ty finding is advisory (printed, non-blocking). ty has ~110 rules
  and no wildcard severity, so "block only on unresolved-import" is enforced at
  the gate/runner (block iff `error[unresolved-import]` appears), not via a
  110-rule downgrade.
- **Pin `ty==0.0.51`** via a `typecheck` extra, one shared version family-wide
  (folded into `dev` where a `dev` extra exists). Bumping the pin is its own
  change.
- **`python-version = "3.13"`** family-wide (matches every `requires-python`).
- **Editable resolution via `extra-paths`:** every family package is installed
  with the import-hook editable mode (`__editable__*.pth` + `_finder.py`), which
  ty cannot follow. Cross-importing repos point `[tool.ty.environment]
  extra-paths` at the sibling **source** roots (relies on the fixed checkout
  layout in CLAUDE.md). No env reinstall.
- **Scope of the commit gate:** simACE always (when Python changed), plus
  `fitACE`, `external/pedigree-graph`, and the `external/tetraher_simace` Python
  helper when they have staged changes. The 7 `fitACE_*` sisters + pedsum are
  swept by the manual runner, not the commit gate.

## Per-repo config

`[tool.ty]` in `pyproject.toml` (or a standalone `ty.toml`, same structure
without the `[tool.ty]` prefix, for repos without a pyproject — `tetraher_simace`
helper, pedsum). Example (fitACE):

```toml
[project.optional-dependencies]
typecheck = ["ty==0.0.51"]

[tool.ty.environment]
python-version = "3.13"
extra-paths = ["..", "../external/pedigree-graph"]   # sibling source roots

[tool.ty.src]
include = ["fitace"]
exclude = ["tests", "workflow/scripts"]

[tool.ty.terminal]
error-on-warning = false
```

`extra-paths` by repo: simACE → `["external/pedigree-graph"]`; fitACE →
`["..", "../external/pedigree-graph"]`; each `fitACE_*` sister →
`["..", "../..", "../../external/pedigree-graph"]`; pedsum →
`["../pedigree-graph"]`. pedigree-graph and the tetraher helper need none.

## Manual family runner

`tools/typecheck_family.py` — runs `ty check` from each present Python repo's
root (so each repo's config + `extra-paths` apply), blocks only on
`error[unresolved-import]`, prints a per-repo `blocking / advisory` summary, and
exits non-zero if any repo has a blocker. Covers the whole family incl. the
sisters + pedsum. `--verbose` also prints advisory findings.

## What landed

- ty config + `typecheck` extra (or `ty.toml`) in: simACE, fitACE,
  pedigree-graph, pedsum, tetraher_simace helper, and all 7 `fitACE_*` sisters.
- `tools/typecheck_family.py`.
- `ty check` drift-only gate added to `/commit` SKILL.md steps 3 & 4.
- **Real bug fixed** (found by ty): `fitACE/fitace/plotting/plot_observed_binary.py`
  imported `_expected_liability_corr` from `simace.plotting.plot_correlations`
  (nonexistent) with the wrong argument order — corrected to
  `expected_liability_corr` from `simace.core.relationships` with `(rel, A, C)`.
- One narrow suppression: `from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]`
  (real class; scipy stubs omit it).

## Verification

- `python tools/typecheck_family.py` → all repos pass the drift gate
  (0 blocking; advisory counts vary).
- Per repo: `ty check --python "$CONDA_PREFIX"`; `ruff check`; tests unaffected.
- Gate behavior: a staged cross-repo import of a nonexistent symbol →
  `/commit` stops on `error[unresolved-import]`; scipy/numba/None-union findings
  print but do not block.

## Future ratchet (not done)

Tighten advisory→blocking per-rule as desired (e.g. promote a specific
high-value rule via `[tool.ty.rules] <rule> = "error"`); add `tests/` to
`src.include` once package code is clean; bump the `ty` pin deliberately.
