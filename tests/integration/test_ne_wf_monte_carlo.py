"""Wright–Fisher Monte Carlo benchmark for the eight Ne estimators.

Generates Wright–Fisher pedigrees (random-mating, 50/50 sex, multinomial
parent picks) and asserts that the mean of each estimator across reps
lands within ±10 % of its expectation under N=200.

Coverage:

* Six estimators (Ne_I, Ne_C, Ne_V, Ne_sr, Ne_H, Ne_GC) are checked
  against ``N``.
* Ne_iΔF is checked against ``N·t/(t−1)`` with ``t = N_GENS``.  Gutiérrez
  Eq. 2 recovers ``N`` only after ``t`` generations of drift, and a pedigree
  whose founders are unrelated by construction runs one generation behind,
  so the estimator is biased high by ``t/(t−1)`` (pedigree-graph ADR 0012,
  "Consequences").  Every last-cohort row here has ``t = N_GENS``.
* ``ne_unrelated_founders``, the package's founder-lag-corrected companion
  on the Ne_iΔF record, is checked against ``N``: the founders of a
  simulated pedigree are unrelated, which is the assumption it needs.
* :func:`ne_long_term_contributions` is excluded.  Its expectation is the
  harmonic mean ``2/Ne_LTC = 1/N + 1/Ne_V`` (pedigree-graph ADR 0012), which
  ``tests/analysis/test_effective_size.py::test_ne_ltc_expectation_matches_simulator_mc``
  checks against simACE's own simulator.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import UnavailableEffectiveSize, estimate_effective_sizes

N = 200
N_GENS = 10
N_REPS = 30
TOL_FRAC = 0.10  # ±10 %


def _build_wf_pedigree(rng: np.random.Generator, n: int = N, n_gens: int = N_GENS) -> pl.DataFrame:
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
    return pl.DataFrame(rows)


def test_wf_monte_carlo_recovers_N():
    """Mean Ne across 30 WF reps lies within ±10 % of each estimator's expectation."""
    rng = np.random.default_rng(2026)
    means: dict[str, list[float]] = {}

    for _ in range(N_REPS):
        df = _build_wf_pedigree(rng)
        pg = PedigreeGraph.from_frame(df)
        results = estimate_effective_sizes(pg)
        unavailable = [name for name, r in results.items() if isinstance(r, UnavailableEffectiveSize)]
        assert not unavailable, f"WF pedigree carries every prerequisite, but {unavailable} refused it"
        for name, r in results.items():
            ne = r.ne
            if ne is None or not np.isfinite(ne):
                continue
            means.setdefault(name, []).append(float(ne))
        companion = results["ne_individual_delta_f"].ne_unrelated_founders
        if companion is not None and np.isfinite(companion):
            means.setdefault("ne_unrelated_founders", []).append(float(companion))

    expected: dict[str, float] = dict.fromkeys(
        (
            "ne_inbreeding",
            "ne_coancestry",
            "ne_variance_family_size",
            "ne_sex_ratio",
            "ne_hill_overlapping",
            "ne_group_coancestry",
            "ne_unrelated_founders",
        ),
        float(N),
    )
    expected["ne_individual_delta_f"] = N * N_GENS / (N_GENS - 1)

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
