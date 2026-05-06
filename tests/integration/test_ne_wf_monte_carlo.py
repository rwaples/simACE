"""Wright–Fisher Monte Carlo benchmark for the eight Ne estimators.

Generates Wright–Fisher pedigrees (random-mating, 50/50 sex, multinomial
parent picks) and asserts that the mean of each estimator across reps
lands within ±10 % of the true population size N=200.

Coverage:

* Seven estimators (Ne_I, Ne_C, Ne_V, Ne_sr, Ne_iΔF, Ne_H, Ne_CT) are
  checked against ``N``.
* :func:`ne_long_term_contributions` is excluded — under finite WF noise
  the per-generation contribution vector fluctuates by ``O(1/√N)`` per
  step, far above the default ``tol = 1e-6``, so the asymptote check
  rarely passes within 10 generations.  The closed-line toy test in
  ``external/pedigree-graph/tests/test_effective_size.py`` exercises the
  formula on a deterministic pedigree.

Marked ``slow`` — full run is ~30 s in the ACE conda env.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pedigree_graph import PedigreeGraph, compute_all_ne

N = 200
N_GENS = 10
N_REPS = 30
TOL_FRAC = 0.10  # ±10 %


def _build_wf_pedigree(rng: np.random.Generator, n: int = N, n_gens: int = N_GENS) -> pd.DataFrame:
    """Wright–Fisher pedigree builder.

    Sex is fixed-alternating (M/F/M/F/…) to lock ``Nm = Nf = N/2``;
    each offspring picks one father uniformly at random from the
    previous generation's males and one mother uniformly from females.
    Offspring sex is also alternating.  No twins.
    """
    rows: list[dict] = [
        {
            "id": i,
            "sex": 1 if i % 2 == 0 else 0,
            "generation": 0,
            "mother": -1,
            "father": -1,
            "twin": -1,
        }
        for i in range(n)
    ]

    next_id = n
    for g in range(1, n_gens + 1):
        prev_start = (g - 1) * n
        # Even-indexed (within a cohort) → male; odd → female.
        male_ids = np.arange(prev_start, prev_start + n, 2)
        female_ids = np.arange(prev_start + 1, prev_start + n, 2)
        f_pick = rng.choice(male_ids, size=n)
        m_pick = rng.choice(female_ids, size=n)
        rows.extend(
            {
                "id": next_id + i,
                "sex": 1 if i % 2 == 0 else 0,
                "generation": g,
                "mother": int(m_pick[i]),
                "father": int(f_pick[i]),
                "twin": -1,
            }
            for i in range(n)
        )
        next_id += n
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_wf_monte_carlo_recovers_N():
    """Mean Ne across 30 WF reps lies within ±10 % of the analytic value."""
    rng = np.random.default_rng(2026)
    means: dict[str, list[float]] = {}

    for _ in range(N_REPS):
        df = _build_wf_pedigree(rng)
        pg = PedigreeGraph(df)
        results = compute_all_ne(pg)
        for name, r in results.items():
            ne = r.ne
            if ne is None or not np.isfinite(ne):
                continue
            means.setdefault(name, []).append(float(ne))

    expected: dict[str, float] = dict.fromkeys(
        (
            "ne_inbreeding",
            "ne_coancestry",
            "ne_variance_family_size",
            "ne_sex_ratio",
            "ne_individual_delta_f",
            "ne_hill_overlapping",
            "ne_caballero_toro",
        ),
        float(N),
    )

    failures: list[str] = []
    for name, target in expected.items():
        samples = means.get(name, [])
        if len(samples) < N_REPS:
            failures.append(f"{name}: only {len(samples)}/{N_REPS} reps produced a finite Ne")
            continue
        mean_ne = float(np.mean(samples))
        rel_err = abs(mean_ne / target - 1.0)
        if rel_err >= TOL_FRAC:
            failures.append(f"{name}: mean {mean_ne:.2f} vs target {target:.2f} (rel err {rel_err:.3f})")

    assert not failures, "WF Monte Carlo failures:\n  " + "\n  ".join(failures)
