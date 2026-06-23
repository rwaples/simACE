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
- **Drift-only gate:** the **only** blocking diagnostic is `unresolved-import`
  (the cross-repo / cross-module signature-drift class — a sibling renaming,
  moving, or removing a symbol, including a missing member of an existing
  module). The gate runs `ty check --ignore all --error unresolved-import` and
  blocks on a **non-zero exit**: ty 0.0.51 supports wildcard severity
  (`--ignore all` + `--error <rule>`), so every other finding is suppressed and
  there is nothing to parse. Plain `ty check` still surfaces the advisory findings.
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

`tools/typecheck_family.py` — runs `ty` from each present Python repo's root (so
each repo's config + `extra-paths` apply). The repo list comes from the shared
`tools/family_repos.py` manifest (single source of truth, also consumed by
`tools/release.py` and `repo-status.sh`). **One hard-zero check per repo:**

- `ty check --output-format concise --color never --error-on-warning`; the
  **exit code is authoritative** (no stdout parsing for the pass/fail decision).
  `--error-on-warning` makes *any* finding — error or warning — fail the repo, so
  the family is held at **zero** ty findings. A failure is labelled `DRIFT` (an
  `unresolved-import`, the high-value cross-repo signature-drift class, always
  printed) vs `ADVISORY` (a library-stub false positive) by parsing the output —
  for the human only; the label never gates. `--verbose` prints advisory
  findings; `--repo <label> ...` limits the sweep.

Covers the whole family incl. the sisters + pedsum. Exits non-zero on any finding
or ty failure. Three invariants are enforced by the test suite alongside the
sweep: the `ty` pin (`tests/test_ty_pin_consistency.py`), the shared
`python-version` (`tests/test_ty_python_version_consistency.py`), and the
rule-code discipline on every suppression (`tests/test_ty_suppressions_coded.py`).

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

## Hard-zero & future work

- **Hard-zero (current):** the family sits at **zero** ty findings, so the sweep
  enforces exactly that — one `ty check --error-on-warning` per repo, exit-code
  authoritative (ADR 0013). The earlier per-repo advisory-budget ratchet
  (`tools/ty_budget.json`) was retired once every repo reached zero: an all-zero
  budget is just "advisory must be 0", which the exit code already gives. The
  escape valve for a genuinely unavoidable false positive is a *specific*
  `# ty: ignore[rule]` suppression — kept rule-coded by
  `tests/test_ty_suppressions_coded.py`, so hard-zero stays safe.
- **Shared manifest:** `tools/family_repos.py` holds the one repo list plus
  `TY_PIN` and `TY_PYTHON_VERSION`, removing the duplicated lists/pins across the
  tooling (enforced by `tests/test_ty_pin_consistency.py` +
  `tests/test_ty_python_version_consistency.py`).
- **Future:** promote a specific high-value rule advisory→blocking via
  `[tool.ty.rules] <rule> = "error"` — note the highest-signal signature-drift
  rules (`invalid-argument-type`, `no-matching-overload`) are also the noisiest
  with library false positives, so this is gated on controlling that surface;
  add `tests/` to `src.include` once package code is clean; bump the `ty` pin
  deliberately (re-triaged, in lockstep across all pyprojects).
