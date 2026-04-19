# Phase 5 M2 — Regression reference values

**Status:** ship note, 2026-04-19 (v2026.05 release).  Source:
`iter_reml_phase5_m2_regression.plan.md`.

The Continuous and MeanOnce iter_reml paths have been stable since
Phase 1/2.  Phase 3 (Laplace) and Phase 4 (MCEM) drew active
development; these two paths had no regression harness.  `v2026.05`
adds `tests/iter_reml/test_continuous_regression.py` and
`tests/iter_reml/test_meanonce_regression.py` to pin σ² against a
locked snapshot on the session-scoped `tiny_ace_inputs` fixture.

## What is pinned

On `tiny_ace_inputs` (n=900, seed=2024) with `fast_kwargs`:

| Path | V(A) | V(C) | V(E) | Vp |
|---|---:|---:|---:|---:|
| Continuous | 0.400301 | 0.268169 | 0.327825 | 0.996294 |
| MeanOnce, K=0.15 | 0.136199 | 0.149466 | 0.682463 | 0.968127 |

Tolerance: ±0.03 absolute on each component, ±0.05 on Vp
(absorbs three component variances).  This is roughly 3·AI-SE +
cross-platform float-rounding headroom; past fp32↔fp64 parity runs
sit within 1e-4 of each other on this fixture so 0.03 is generous.

## What these values mean

* They are **v2026.04 snapshots** — not ground truth.  Truth on this
  fixture is `(A=0.5, C=0.2, E=0.3)` on the liability scale; the
  tiny-fixture MLE is a noisy draw around it.
* They validate **stability** of the code path: the binary, CLI,
  parser, and AI-REML numerics.  They do not validate scientific
  accuracy — the dev-grid does that on larger n.
* Earlier drafts of the plan quoted `dev_cont_n10k` values as
  a ballpark; those were `V(A)≈0.489, V(C)≈0.201, V(E)≈0.298`.
  The numbers in the tests are the actual tiny-fixture outputs,
  measured at v2026.04.

## When to update the pinned values

Only update deliberately, together with:

1. A findings note explaining which algorithmic change moved the
   numbers (and by how much on the dev-grid, not just this fixture).
2. A bump to the tolerance if the change itself is noisier (e.g.,
   an adaptive SLQ probe schedule that trades determinism for
   speed).
3. A re-run of the full Phase 3 / Phase 4 ship-gate (Laplace SE
   coverage, MCEM dev-grid) to confirm no collateral regression.

A silent drift past the ±0.03 tolerance is a bug, not a refactor.

## Re-capture procedure

```python
from fitace.iter_reml.fit import fit_iter_reml
# tiny_ace_inputs fixture (fitACE/tests/iter_reml/conftest.py)
# + fast_kwargs; run both Continuous and MeanOnce K=0.15.
```

See `/tmp/capture_m2_refvals.py` (session-local, not committed) or
regenerate from the fixture in `tests/iter_reml/conftest.py`.
