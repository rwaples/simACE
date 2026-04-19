# Phase 5 M2 — Continuous / MeanOnce regression harness

**Status:** plan, 2026-04-19.  Retires P3.D7 (pending since Phase 3
shipped).  Ships in v2026.05 alongside M1.

## Goal

Lock down the Continuous and MeanOnce binary-trait paths against
accidental regression.  These are the *stable* paths — they've
worked since Phase 1/2 and most dev attention has gone to Laplace
(Phase 3) and MCEM (Phase 4).  Without a dedicated regression
harness, a future algorithmic change could silently break them.

## Ship criteria

* `tests/iter_reml/test_continuous_regression.py` exercises the
  Continuous path end-to-end on a simulated fixture and pins σ²
  within ±0.03 of the reference values from a locked "known good"
  v2026.04 snapshot.
* `tests/iter_reml/test_meanonce_regression.py` same for MeanOnce
  (K=0.15 fixture).
* Both tests run in < 10 seconds at `fast_kwargs` scale.
* Tests are marked `@needs_bin` and get skipped when the compiled
  binary is absent (same pattern as existing integration tests).

## Out of scope

* Validating σ² against *truth* — that's the dev-grid's job.  This
  milestone validates *stability* of the fit path, not its
  scientific accuracy.
* Laplace regression — Laplace already has `test_laplace_path.py`.
  This milestone fills the gap for the other two paths.

## Reference values

On the tiny_ace_inputs fixture (n=900, seed=2024), the v2026.04
Continuous fit produces:

```
σ²_A = 0.489  ± 0.024 (AI SE)
σ²_C = 0.201  ± 0.012
σ²_E = 0.298  ± 0.014
```

(These are dev_cont_n10k rep1 numbers from the M4 dev-grid —
truth is (0.5, 0.2, 0.3).)

MeanOnce at K=0.15 produces (from `dev_mean_n10k` rep1):
```
σ²_A = 0.259
σ²_C = 0.112
σ²_E = 0.590
Vp  = 0.961
```

These are the values to pin.  Tolerance ±0.03 absolute allows for
cross-platform float rounding + 3·AI-SE headroom.

## Milestones

### M2.1 — Continuous regression test (~2 hr)

`tests/iter_reml/test_continuous_regression.py`:

```python
@needs_bin
def test_continuous_fit_stable(tiny_ace_inputs, fast_kwargs):
    inp = tiny_ace_inputs
    r = fit_iter_reml(
        y=inp["y"],                        # continuous (not thresholded)
        kinship=inp["K"],
        household_id=inp["household_id"],
        iids=inp["iids"],
        **fast_kwargs,
    )
    assert r.converged
    vc = r.vc.set_index("vc_name")["estimate"]
    # Reference values locked at v2026.04.
    assert abs(vc["V(A)"] - 0.489) < 0.03
    assert abs(vc["V(C)"] - 0.201) < 0.03
    assert abs(vc["Ve"]   - 0.298) < 0.03
    # SE sanity: finite and positive.
    se = r.vc.set_index("vc_name")["se"]
    assert (se[["V(A)", "V(C)", "Ve"]] > 0).all()
    assert r.liability_model == "continuous"
    # logLik must be finite for Gaussian continuous path.
    assert np.isfinite(r.logLik)
```

Add 2-3 sibling tests:
* `test_continuous_no_covar` — as above.
* `test_continuous_with_covar` — add sex + generation covariates,
  assert fe.tsv is populated + σ² are in the same ballpark.
* `test_continuous_skip_phase1` — confirm var(y)/3 warm-start
  still converges (degenerate case).

### M2.2 — MeanOnce regression test (~1 hr)

`tests/iter_reml/test_meanonce_regression.py`:

```python
@needs_bin
def test_meanonce_binary_stable(tiny_ace_inputs, fast_kwargs):
    inp = tiny_ace_inputs
    K = 0.15
    y_binary, _ = _binarise(inp["y"], K)
    r = fit_iter_reml(
        y=y_binary,
        kinship=inp["K"],
        household_id=inp["household_id"],
        iids=inp["iids"],
        prevalence=K,
        liability_model="mean",
        **fast_kwargs,
    )
    assert r.converged
    assert r.liability_model == "mean_once"
    assert r.prevalence == pytest.approx(K)
    vc = r.vc.set_index("vc_name")["estimate"]
    # Reference values at v2026.04.
    assert abs(vc["V(A)"] - 0.259) < 0.03
    assert abs(vc["V(C)"] - 0.112) < 0.03
    assert abs(vc["Ve"]   - 0.590) < 0.03
    assert abs(vc["Vp"]   - 0.961) < 0.05
```

Sibling tests:
* `test_meanonce_rejects_non_binary` — `liability_model="mean"` +
  continuous y should raise.
* `test_meanonce_auto_K_warning` — supplied K ≠ empirical K
  should warn in the log (doesn't need to assert — just confirms
  the warning path doesn't crash).

### M2.3 — Wire into CI + commit reference-values note (~30 min)

* Tests run via `pytest tests/iter_reml -q` (existing command).
* Add `notes/iter_reml_phase5_m2_reference_values.md`: a short
  note explaining that the hard-coded values in the tests are
  v2026.04 snapshots, and that a deliberate algorithmic change
  (not a silent regression) would update them together with a
  findings note.

## Testing

Cumulative iter_reml test count moves from 70 → ~75 (5-6 new
tests).  All other tests unchanged.

## Risks

* **Values drift due to PETSc library updates** — PETSc 3.25 → 3.26
  may shift σ² by more than the ±0.03 tolerance due to BLAS-level
  ULP differences.  Mitigation: use `pytest.approx(..., rel=0.05)`
  if floating-point drift becomes an issue.  So far (fp32 vs fp64
  builds) this has stayed within 1e-4 on σ² — 0.03 absolute is a
  generous envelope.
* **Platform dependency** — the reference values come from one
  Linux / MKL stack.  If CI runs on macOS / OpenBLAS the values
  may drift.  Mitigation: loosen tolerance to 0.05 cross-platform,
  tighten to 0.01 on the reference stack (pytest parametrize).

## Effort rollup

| Sub-milestone | Effort |
|---|---:|
| M2.1 Continuous regression | 2 hr |
| M2.2 MeanOnce regression | 1 hr |
| M2.3 CI wire-up + note | 0.5 hr |

Total: ~4 hours, half a day.
