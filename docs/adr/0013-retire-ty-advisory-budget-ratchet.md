# Retire the ty advisory-budget ratchet for hard-zero

The ty rollout (`docs/plans/ty-typecheck-rollout.md`) built a per-repo
advisory-budget ratchet (`tools/ty_budget.json`) so each family repo could carry
a non-zero baseline of library-stub false positives while still blocking *new*
ones. Once every repo was driven to zero advisory findings, every budget entry
was `0` — the ratchet had degenerated into a flat "advisory must be 0", which a
single `ty check --error-on-warning` (exit-code authoritative) already enforces.

So we retired it: `tools/typecheck_family.py` now runs **one hard-zero check per
repo** and `tools/ty_budget.json` is deleted — ~40 lines of budget machinery
gone, and one `ty` invocation per repo instead of two.

## Consequences

The trade-off is the lost escape valve: a new unavoidable false positive (e.g.
after a `ty` pin bump or a library-stub change) now fails the family sweep until
it is cleared with a *specific* `# ty: ignore[rule]` suppression. That is the
normal response anyway, and `tests/test_ty_suppressions_coded.py` enforces that
every suppression names its rule code — so hard-zero stays safe and suppressions
stay narrow and self-documenting. The `/commit` drift gate
(`ty check --ignore all --error unresolved-import`) is unchanged; only the
family sweep went hard-zero.
