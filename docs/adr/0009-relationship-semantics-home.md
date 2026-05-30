# ADR 0009: Home for Relationship Semantics

## Status

Accepted.

## Context

simACE's relationship-type vocabulary is split awkwardly. `RELATIONSHIP_TYPES`
and `SEX_LEVELS` already live in `simace/core/relationships.py`, but the
*quantitative* relationship semantics are scattered:

- Expected liability correlations are inline literals at their point of use in
  `simace/analysis/validate.py` (`expected_a = 0.25`, `expected_dz = 0.5`, the
  MZ `> 0.99` check, the PHS shared-C `0` check).
- The maternal- vs paternal-half-sib C-sharing rule (household is assigned by
  mother, so MHS share C but PHS do not) is re-encoded inline in validation and
  restated as prose in plotting docstrings.

Kinship coefficients are *not* simACE's to own: the source of truth is the
external `pedigree_graph` package (`PAIR_KINSHIP`, `REL_REGISTRY`), which
`validate.py`, `stats`, and `fit_ace` all import. CLAUDE.md gotcha #4 records
that `fit_ace` couples to `PAIR_KINSHIP` and that `ltm_falconer.py` keeps a
parallel `KINSHIP` dict that must stay in sync. Any place simACE re-declares a
kinship literal is a latent drift bug that can silently bias downstream fitACE
estimates.

Separately, some "relationship" logic is not a property of the relationship
type at all but an *analysis-design choice* — e.g. validation pools MHS ∪ PHS
for the additive-genetic correlation (both have kinship 0.25) but uses PHS-only
for the liability and shared-C correlations. That pooling is a decision about
how to estimate, not a fact about the pair.

## Decision

`simace.core.relationships` is the canonical home for relationship-type
*properties*. Two helpers are added:

- `shared_environment_coefficient(relationship_type: str) -> float` — encodes
  the household-by-mother rule. Same-mother types (`MZ`, `FS`, `MHS`) → `1.0`;
  `PHS`, `MO`, `FO`, `1C` → `0.0`.
- `expected_liability_corr(relationship_type: str, A: float, C: float) ->
  float` — derived, never stored:
  `2 * PAIR_KINSHIP[relationship_type] * A + shared_environment_coefficient(relationship_type) * C`.

Kinship is **always** read from `pedigree_graph.PAIR_KINSHIP`. simACE never
writes a kinship literal. Both helpers operate over the canonical 7-type
`RELATIONSHIP_TYPES` subset and **raise `ValueError` on an unknown type** (an
unknown key is a caller bug, not a "no expectation" signal).

Pooling and presentation groupings stay at their call sites (validation,
plotting). No `pooled_relationship_classes()` is added to `core`.

## Consequences

- `validate.py` and plotting stop hard-coding `0.25` / `0.5` / `1.0` and the
  PHS-shared-C-is-zero rule; they call the helpers instead.
- The maternal- vs paternal-half-sib C behavior gets one tested home, with an
  explicit test that MHS and PHS differ in shared C.
- Coefficients cannot drift from the registry, because they trace back to
  `PAIR_KINSHIP` rather than to copied numbers.
- Helper *names and signatures* become a cross-repo concern: fitACE may choose
  to consume them, so renames require the simACE↔fitACE coordination noted in
  CLAUDE.md. The numeric values do not, since they are derived.

## Non-goals

- Not centralizing pooling/presentation logic — that stays at call sites.
- Not re-declaring or renaming `PAIR_KINSHIP` or `RELATIONSHIP_TYPES`.
- Not changing any kinship coefficient or any fitACE behavior.
- Not typing the validation result dicts (separately killed during triage —
  the report output contract already exists via `assert_report_contract`).
