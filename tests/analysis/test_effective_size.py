"""Per-rep Ne wrapper: compute_effective_size + main() integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from simace.analysis.stats.effective_size import (
    compute_effective_size,
    family_size_variance_expected_ztp,
    ne_drift_expected_ztp,
    regression_estimator_regime_ok,
    theoretical_expectations,
)
from simace.analysis.stats.effective_size import (
    main as run_effective_size,
)
from simace.core.cohort import COHORT_COLUMNS, build_cohort, write_cohort, write_pedigree
from simace.core.pedigree_filter import filter_pedigree_to_observed

EXPECTED_KEYS = set(ALL_EFFECTIVE_SIZE_ESTIMATORS)


def _is_unavailable(entry: dict) -> bool:
    """Whether an estimator key carries an UnavailableEffectiveSize payload."""
    return "reason" in entry


@pytest.fixture(scope="module")
def tiny_pedigree() -> pl.DataFrame:
    """Reuse the 200-individual / G_ped=2 fixture without phenotype/censor cost."""
    from simace.simulation.simulate import run_simulation

    return run_simulation(
        seed=7,
        N=200,
        G_ped=2,
        G_sim=3,
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    ).pedigree


# ---------------------------------------------------------------------------
# theoretical_expectations
# ---------------------------------------------------------------------------


class TestTheoreticalExpectations:
    def test_none_config(self):
        exp = theoretical_expectations(None)
        assert set(exp.keys()) == EXPECTED_KEYS
        assert all(v is None for v in exp.values())

    def test_assortative_mating_disables_expectations(self):
        cfg = {"N": 200, "assort1": 0.3, "assort2": 0.0}
        exp = theoretical_expectations(cfg)
        assert all(v is None for v in exp.values())

    def test_baseline100K_default_drops_regression_estimators(self):
        # baseline100K config: N=100K, G_ped=6 ⇒ N·G² = 3.6M < 120·Ne (~10.2M),
        # regression-based Ne_I/Ne_C/Ne_GC must report None.
        cfg = {"N": 100000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 6}
        exp = theoretical_expectations(cfg)
        ne = ne_drift_expected_ztp(100000.0, 0.5)
        assert exp["ne_inbreeding"] is None
        assert exp["ne_coancestry"] is None
        assert exp["ne_group_coancestry"] is None
        # G_ped=6 ⇒ the last cohort has t=5 equivalent complete generations but
        # only 4 of drift, so Ne_iΔF is expected 5/4 high (pedigree-graph #15).
        assert exp["ne_individual_delta_f"] == pytest.approx(ne * 5.0 / 4.0)
        assert exp["ne_sex_ratio"] == pytest.approx(100000.0)
        assert exp["ne_long_term_contributions"] == pytest.approx(ne)

    def test_standard_family_size_estimators_expect_the_drift_ne(self):
        # pedigree-graph 0.15 computes Ne_V from Hill 1979 eq. 9
        # (pedigree-graph#36); Ne_H passes it through without birth_year.
        cfg = {"N": 2000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 12}
        exp = theoretical_expectations(cfg)
        ne = ne_drift_expected_ztp(2000.0, 0.5)
        assert exp["ne_variance_family_size"] == pytest.approx(ne)
        assert exp["ne_hill_overlapping"] == pytest.approx(ne)

    def test_drift_estimators_share_the_hill_expectation(self):
        # Issue #72: every drift-rate estimator targets 4N/(2+V(k)).
        cfg = {"N": 2000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 2.0, "G_ped": 12}
        exp = theoretical_expectations(cfg)
        v_k = 2.0 + 4.0 * family_size_variance_expected_ztp(2.0)["cov"]
        hill = 4.0 * 2000 / (2.0 + v_k)
        for name in ("ne_inbreeding", "ne_coancestry", "ne_group_coancestry", "ne_long_term_contributions"):
            assert exp[name] == pytest.approx(hill), name
        assert exp["ne_individual_delta_f"] == pytest.approx(hill * 11.0 / 10.0)

    def test_missing_g_ped_drops_regression_estimators(self):
        cfg = {"N": 100000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5}
        exp = theoretical_expectations(cfg)
        assert exp["ne_inbreeding"] is None
        assert exp["ne_coancestry"] is None
        assert exp["ne_group_coancestry"] is None
        # Ne_iΔF needs G_ped too: without it the last cohort's pedigree depth,
        # and so the size of its founder-boundary bias, is unknown.
        assert exp["ne_individual_delta_f"] is None
        # Other estimators still populated.
        assert exp["ne_long_term_contributions"] is not None

    def test_per_generation_assortment_disables_expectations(self):
        # A per-generation dict used to crash float(); AM in any generation
        # nullifies the random-mating expectations.
        cfg = {"N": 2000, "G_ped": 12, "mating_lambda": 0.5, "assort1": {0: 0.0, 5: 0.3}, "assort2": 0.0}
        assert all(v is None for v in theoretical_expectations(cfg).values())

    def test_assort_matrix_alone_disables_expectations(self):
        cfg = {
            "N": 2000,
            "G_ped": 12,
            "mating_lambda": 0.5,
            "assort1": 0.0,
            "assort2": 0.0,
            "assort_matrix": [[0.3, 0.1], [0.1, 0.3]],
        }
        assert all(v is None for v in theoretical_expectations(cfg).values())

    def test_missing_mating_lambda_disables_expectations(self):
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0}
        exp = theoretical_expectations(cfg)
        assert all(v is None for v in exp.values())

    # ── Wright-Fisher branch ─────────────────────────────────────────────

    def test_wf_short_g_ped_drops_regression(self):
        # WF: regression gate reduces to G_ped² ≥ 120 ⇒ G_ped ≥ 11.  At G_ped=4
        # the regression-based estimators must be None; variance-family ones
        # equal N exactly.  N is independent of the gate under WF.
        cfg = {"mating_model": "wright_fisher", "N": 2000, "G_ped": 4}
        exp = theoretical_expectations(cfg)
        assert exp["ne_inbreeding"] is None
        assert exp["ne_coancestry"] is None
        assert exp["ne_group_coancestry"] is None
        assert exp["ne_variance_family_size"] == pytest.approx(2000.0)
        assert exp["ne_sex_ratio"] == pytest.approx(2000.0)
        # G_ped=4 ⇒ t=3, so Ne_iΔF is expected 3/2 high.
        assert exp["ne_individual_delta_f"] == pytest.approx(3000.0)
        assert exp["ne_hill_overlapping"] == pytest.approx(2000.0)
        assert exp["ne_long_term_contributions"] == pytest.approx(2000.0)

    def test_wf_long_g_ped_populates_regression(self):
        # G_ped=12 ⇒ G_ped² = 144 ≥ 120 ⇒ regression-based estimators populated.
        cfg = {"mating_model": "wright_fisher", "N": 2000, "G_ped": 12}
        exp = theoretical_expectations(cfg)
        assert exp["ne_inbreeding"] == pytest.approx(2000.0)
        assert exp["ne_coancestry"] == pytest.approx(2000.0)
        assert exp["ne_group_coancestry"] == pytest.approx(2000.0)

    def test_wf_ignores_inherited_assort_and_lambda(self):
        # Inherited standard-only knobs must not gate WF expectations.
        cfg = {
            "mating_model": "wright_fisher",
            "N": 2000,
            "G_ped": 4,
            "assort1": 0.3,  # would zero out standard exps
            "assort2": 0.2,
            "mating_lambda": None,  # would None-out standard exps
        }
        exp = theoretical_expectations(cfg)
        assert exp["ne_variance_family_size"] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# ne_drift_expected_ztp closed-form limits
# ---------------------------------------------------------------------------


class TestNeDriftExpectedZtp:
    def test_lambda_zero_limit_is_N(self):
        # ZTP degenerates to m=1 ⇒ no extra mating-count variance ⇒ Ne = N.
        assert ne_drift_expected_ztp(1000, 0.0) == pytest.approx(1000.0)
        assert ne_drift_expected_ztp(1000, 1e-9) == pytest.approx(1000.0, rel=1e-6)

    def test_large_lambda_limit_approaches_N(self):
        # Poisson, no truncation ⇒ r = Var[m]/E[m]² → 1/λ ⇒ slow approach to N.
        # At λ=100, r = 1/100, Ne/N = 1/1.01.
        assert ne_drift_expected_ztp(1000, 100.0) == pytest.approx(1000.0 / 1.01, rel=1e-3)
        # By λ=10000 we're within 0.01 % of N.
        assert ne_drift_expected_ztp(1000, 10000.0) == pytest.approx(1000.0, rel=1e-3)

    def test_default_lambda_05_value(self):
        # Hill 1979 eq. 10: 4N/(2 + V(k)) with V(k) = 2.7216 at λ=0.5.
        assert ne_drift_expected_ztp(100000.0, 0.5) == pytest.approx(84716.5, rel=1e-4)

    def test_is_hill_eq_10(self):
        for lam in (0.3, 0.5, 2.0):
            v_k = 2.0 + 4.0 * family_size_variance_expected_ztp(lam)["cov"]
            assert ne_drift_expected_ztp(2000, lam) == pytest.approx(4.0 * 2000 / (2.0 + v_k)), lam

    def test_monotone_in_lambda_below_unity(self):
        # ZTP overdispersion grows with λ for λ < 1 ⇒ Ne/N decreases.
        from itertools import pairwise

        n = 1000
        ratios = [ne_drift_expected_ztp(n, lam) / n for lam in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for a, b in pairwise(ratios):
            assert a >= b, f"Not monotone: {ratios}"


class TestRegressionEstimatorRegimeOk:
    def test_returns_false_for_g_ped_below_2(self):
        # No slope possible with fewer than 2 transitions.
        assert not regression_estimator_regime_ok(1e9, 1, 1.0)

    def test_baseline100K_default_is_not_ok(self):
        # N=100K, G=6, Ne=84,717 ⇒ N·G² = 3.6M < 120·84,717 = 10.2M
        assert not regression_estimator_regime_ok(100000.0, 6, 84717.0)

    def test_high_g_ped_brings_regime_into_range(self):
        # N=100K, G=15 ⇒ N·G² = 22.5M > 10.2M
        assert regression_estimator_regime_ok(100000.0, 15, 84717.0)


# ---------------------------------------------------------------------------
# compute_effective_size
# ---------------------------------------------------------------------------


class TestComputeEffectiveSize:
    def test_returns_eight_keys_with_to_dict_payload(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree)
        assert set(result.keys()) == EXPECTED_KEYS
        for k, entry in result.items():
            assert isinstance(entry, dict), k
            if _is_unavailable(entry):
                assert set(entry) == {"reason", "code", "fields"}, k
                continue
            assert "ne" in entry
            assert "expected" in entry
            assert entry["expected"] is None  # no config provided

    def test_skip_ne_coancestry_reports_the_refusal_not_a_null_ne(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree, skip_ne_coancestry=True)
        assert set(result.keys()) == EXPECTED_KEYS
        assert result["ne_coancestry"] == {"reason": "not_requested", "code": None, "fields": {}}
        assert not _is_unavailable(result["ne_inbreeding"])

    def test_missing_sex_marks_only_the_sex_dependent_estimators_unavailable(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree.drop("sex"))
        for name in ("ne_variance_family_size", "ne_sex_ratio", "ne_hill_overlapping"):
            assert result[name]["reason"] == "missing_metadata", name
            assert result[name]["code"] == "missing_sex", name
        assert not _is_unavailable(result["ne_inbreeding"])

    def test_expected_attached_when_config_standard(self, tiny_pedigree):
        # G_ped=20 puts the regime check well into "ok" for N=200, so the
        # regression estimators receive the drift expectation too.
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 20}
        result = compute_effective_size(tiny_pedigree, config=cfg)
        ne = ne_drift_expected_ztp(200, 0.5)
        for k in ("ne_inbreeding", "ne_coancestry", "ne_group_coancestry", "ne_long_term_contributions"):
            assert result[k]["expected"] == pytest.approx(ne), k
        assert result["ne_individual_delta_f"]["expected"] == pytest.approx(ne * 19.0 / 18.0)
        assert result["ne_sex_ratio"]["expected"] == pytest.approx(200.0)
        assert result["ne_variance_family_size"]["expected"] == pytest.approx(ne)
        assert result["ne_hill_overlapping"]["expected"] == pytest.approx(ne)

    def test_cohort_arrays_are_sized_by_the_observed_labels_they_carry(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree)
        observed = sorted(set(tiny_pedigree["generation"].to_list()))
        for name, cohort_field in (
            ("ne_inbreeding", "mean_f_per_gen"),
            ("ne_coancestry", "mean_theta_per_gen"),
            ("ne_sex_ratio", "n_male_per_gen"),
            ("ne_individual_delta_f", "mean_eqg_per_gen"),
            ("ne_group_coancestry", "mean_group_coancestry_per_gen"),
        ):
            entry = result[name]
            assert entry["generations"] == observed, name
            assert len(entry[cohort_field]) == len(observed), name

    def test_rate_estimators_report_ne_per_transition_not_per_cohort(self, tiny_pedigree):
        # Ne_I, Ne_C and Ne_GC measure a rate between adjacent observed
        # cohorts, so their Ne array is one shorter than their labels.
        result = compute_effective_size(tiny_pedigree)
        for name in ("ne_inbreeding", "ne_coancestry", "ne_group_coancestry"):
            entry = result[name]
            generations = entry["generations"]
            assert len(entry["ne_per_gen"]) == len(generations) - 1, name
            assert entry["transition_from"] == generations[:-1], name
            assert entry["transition_to"] == generations[1:], name

    def test_family_size_variance_is_indexed_by_parent_generation(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree)
        entry = result["ne_variance_family_size"]
        parents = entry["parent_generations"]
        assert parents == sorted(set(tiny_pedigree["generation"].to_list()))
        for col in ("ne_per_transition", "v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f"):
            assert len(entry[col]) == len(parents), col

    def test_ne_hill_birth_year_branch_round_trips(self, tiny_pedigree):
        # Add a synthetic birth_year column so the wrapper engages the
        # Hill 1979 eq. (10) branch.  In simACE, generations are
        # strictly discrete so this is artificial — but the wrapper
        # contract is what we're testing.
        df = tiny_pedigree.with_columns((pl.col("generation").cast(pl.Int64) * 5 + 2000).alias("birth_year"))
        result = compute_effective_size(df)
        h = result["ne_hill_overlapping"]
        assert h["collapses_to_ne_v"] is False
        assert h["cohort_window"] is not None
        assert h["T_m"] is not None
        # When birth_year is set, the analytic Ne_V expectation no longer
        # applies → expected must be overridden to None.
        assert h["expected"] is None

    def test_ne_hill_expected_overridden_to_none_when_birth_year_set(self, tiny_pedigree):
        # Even when a config that would normally set an expectation is
        # passed, the birth-year branch overrides it to None because no
        # closed-form expectation exists for Hill 1979 eq. (10).
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 20}
        df = tiny_pedigree.with_columns((pl.col("generation").cast(pl.Int64) * 5 + 2000).alias("birth_year"))
        result = compute_effective_size(df, config=cfg)
        assert result["ne_hill_overlapping"]["expected"] is None
        # Other estimators still receive their expected values.
        assert result["ne_inbreeding"]["expected"] is not None


# ---------------------------------------------------------------------------
# Cross-estimator consistency under random mating (excludes Ne_LTC)
# ---------------------------------------------------------------------------


def _drift_only_pedigree(
    *, seed: int, n: int, g_ped: int, mating_lambda: float, mating_model: str = "standard"
) -> pl.DataFrame:
    """One simulated pedigree with the phenotype knobs held at drift-neutral values.

    Every Ne estimator reads pedigree structure alone, so the A/C/E variances,
    trait correlations, twinning rate and assortment are pinned here and only
    the mating parameters vary between callers.
    """
    from simace.simulation.simulate import run_simulation

    return run_simulation(
        seed=seed,
        N=n,
        G_ped=g_ped,
        G_sim=g_ped + 1,
        mating_lambda=mating_lambda,
        mating_model=mating_model,
        p_mztwin=0.0,
        A1=0.5,
        C1=0.0,
        E1=0.5,
        A2=0.5,
        C2=0.0,
        E2=0.5,
        rA=0.0,
        rC=0.0,
        assort1=0.0,
        assort2=0.0,
    ).pedigree


def _build_wf_pedigree(rng: np.random.Generator, n: int = 50, n_gens: int = 8) -> pd.DataFrame:
    """Symmetric Wright–Fisher pedigree (alternating M/F sex, multinomial parents).

    Stays pandas to keep a pandas frame in the ``from_frame`` coverage; the
    structural ``FrameLike`` protocol accepts either library.
    """
    rows: list[dict] = [
        {"id": i, "sex": 1 if i % 2 == 0 else 0, "generation": 0, "mother": -1, "father": -1, "twin": -1}
        for i in range(n)
    ]
    next_id = n
    for g in range(1, n_gens + 1):
        prev = (g - 1) * n
        males = np.arange(prev, prev + n, 2)
        females = np.arange(prev + 1, prev + n, 2)
        f_pick = rng.choice(males, size=n)
        m_pick = rng.choice(females, size=n)
        for i in range(n):
            rows.append(
                {
                    "id": next_id,
                    "sex": 1 if i % 2 == 0 else 0,
                    "generation": g,
                    "mother": int(m_pick[i]),
                    "father": int(f_pick[i]),
                    "twin": -1,
                }
            )
            next_id += 1
    return pd.DataFrame(rows)


def test_pg_ne_v_tracks_the_drift_ne():
    """pedigree-graph's Ne_V and Ne_H target the drift Ne (pedigree-graph#36).

    From pedigree-graph 0.15, ``ne_variance_family_size`` computes Hill 1979
    eq. 9, ``4N/(2 + V(k))`` at equal sexes, and without ``birth_year``
    ``ne_hill_overlapping`` passes it through, so
    :func:`theoretical_expectations` gives both the drift Ne.  Before 0.15 Ne_V
    was ``2N/V(k)``, 13 % below it at λ = 0.5.

    12 reps at N=2000, G_ped=4, 36 transitions per λ.  Measured on
    pedigree-graph 0.15.0: Ne_V −1.5 % (λ = 0.5) and −0.6 % (λ = 2) of the
    drift Ne, Ne_H −1.6 % and −0.7 %.
    """
    from pedigree_graph.effective_size import ne_hill_overlapping, ne_variance_family_size

    n, n_reps = 2000, 12
    for mating_lambda in (0.5, 2.0):
        drift_ne = ne_drift_expected_ztp(n, mating_lambda)
        per_transition: list[float] = []
        hill: list[float] = []
        for rep in range(n_reps):
            ped = _drift_only_pedigree(seed=1000 + rep, n=n, g_ped=4, mating_lambda=mating_lambda)
            pg = PedigreeGraph.from_frame(ped)
            result = ne_variance_family_size(pg)
            per_transition.extend(result.ne_per_transition[np.isfinite(result.ne_per_transition)].tolist())
            hill.append(ne_hill_overlapping(pg).ne)
        for label, values in (("Ne_V", per_transition), ("Ne_H", hill)):
            mean_ne = float(np.mean(values))
            assert abs(mean_ne / drift_ne - 1.0) < 0.05, (
                f"λ={mating_lambda}: {label} mean over {len(values)} = {mean_ne:.1f}, drift Ne = {drift_ne:.1f}"
            )


@pytest.mark.parametrize(
    ("mating_model", "g_ped"),
    [("standard", 8), ("standard", 12), ("wright_fisher", 8), ("wright_fisher", 12)],
)
def test_drift_expectation_matches_simulator_mc(mating_model: str, g_ped: int):
    """The drift Ne of ``theoretical_expectations`` matches the simulator.

    This is the committed method behind :func:`theoretical_expectations`'
    drift target, Hill 1979 eq. 10 ``4N/(2 + V(k))`` (issue #72).  It checks
    Ne_LTC at every depth and Ne_C wherever the regression gate passes
    (``G_ped = 12`` here).

    Ne_LTC is read as ``2/Σc²`` out of ``sum_c_squared`` rather than the
    record's ``ne`` so the check does not depend on how pedigree-graph
    scales or withholds the record's ``ne`` (ADR 0012).

    Measured at N=1000 over these 12 seeds, against 847.2 (ZTP) and 1000
    (WF): Ne_LTC −0.17 % (standard, G_ped=8), −0.49 % (standard, 12), +0.15 %
    (WF, 12); Ne_C −2.12 % (standard, 12; sem 0.62 %) and +0.01 % (WF, 12).
    Ne_C's ZTP shortfall is plausibly the extra family-size variance that
    slot balancing adds at finite N (:func:`ne_drift_expected_ztp`), so Ne_C
    gets ±5 % against Ne_LTC's ±3 %.  Both bands reject ``2N/V(k)`` = 734.9,
    which sits 13 % below the drift Ne.
    """
    from pedigree_graph.effective_size import ne_coancestry, ne_long_term_contributions

    n, mating_lambda, n_reps = 1000, 0.5, 12
    cfg = {
        "N": n,
        "assort1": 0.0,
        "assort2": 0.0,
        "mating_lambda": mating_lambda,
        "G_ped": g_ped,
        "mating_model": mating_model,
    }
    exp = theoretical_expectations(cfg)

    ltc, coancestry = [], []
    for rep in range(n_reps):
        ped = _drift_only_pedigree(
            seed=1000 + rep, n=n, g_ped=g_ped, mating_lambda=mating_lambda, mating_model=mating_model
        )
        pg = PedigreeGraph.from_frame(ped)
        ltc.append(2.0 / ne_long_term_contributions(pg).sum_c_squared)
        if exp["ne_coancestry"] is not None:
            coancestry.append(ne_coancestry(pg).ne)

    checks = [("Ne_LTC 2/Sum(c^2)", ltc, exp["ne_long_term_contributions"], 0.03)]
    if coancestry:
        checks.append(("Ne_C", coancestry, exp["ne_coancestry"], 0.05))
    for label, observed, expected, tol in checks:
        mean = float(np.mean(observed))
        rel_err = abs(mean / expected - 1.0)
        assert rel_err < tol, (
            f"{label} mean across {n_reps} reps = {mean:.1f}, expected {expected:.1f} (rel err {rel_err:.3f})"
        )


def test_cross_estimator_consistency_under_wf():
    """Under Wright–Fisher, all 7 estimators (excl. Ne_LTC) agree within ±15 % over 30 reps.

    Compares each estimator's mean across reps against ``ne_sex_ratio``'s
    mean, which is the cleanest deterministic baseline (Ne_sr_t = N
    exactly when Nm = Nf = N/2 every generation).  Ne_LTC is excluded —
    its asymptote tolerance rarely passes within 8 generations of WF
    drift (see ``tests/integration/test_ne_wf_monte_carlo.py`` for the
    rationale).
    """
    rng = np.random.default_rng(2026)
    n_reps = 30
    keys = (
        "ne_inbreeding",
        "ne_coancestry",
        "ne_variance_family_size",
        "ne_sex_ratio",
        "ne_individual_delta_f",
        "ne_hill_overlapping",
        "ne_group_coancestry",
    )
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_reps):
        df = _build_wf_pedigree(rng)
        pg = PedigreeGraph.from_frame(df)
        results = compute_effective_size(pg)
        for k in keys:
            ne = results[k]["ne"]
            if ne is None:
                continue
            samples[k].append(float(ne))

    means = {k: float(np.mean(v)) for k, v in samples.items() if v}
    baseline = means["ne_sex_ratio"]
    failures: list[str] = []
    for k, mean_ne in means.items():
        if k == "ne_sex_ratio":
            continue
        rel_err = abs(mean_ne / baseline - 1.0)
        if rel_err >= 0.15:
            failures.append(f"{k}: {mean_ne:.2f} vs Ne_sr {baseline:.2f} (rel err {rel_err:.3f})")
    assert not failures, "Cross-estimator mismatches:\n  " + "\n  ".join(failures)


# ---------------------------------------------------------------------------
# Runner integration: stats yaml has an effective_size block
# ---------------------------------------------------------------------------


def test_effective_size_main_writes_yaml(tmp_path, tiny_pedigree):
    """`effective_size.main` should write a yaml with all 8 estimator keys.

    The cohort's analysis sample is the last generation only, so the
    analysis pedigree rebuilt from it carries the ancestor closure
    (founders/intermediate ancestors pulled back through parent pointers).
    Per-rep ``params.yaml`` is supplied inline so the validator-facing
    ``expected`` field is populated.
    """
    ped_path = tmp_path / "pedigree.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    params_path = tmp_path / "params.yaml"
    out_path = tmp_path / "effective_size.yaml"

    write_pedigree(tiny_pedigree, ped_path)
    last_gen = int(tiny_pedigree["generation"].max())
    sample_ids = tiny_pedigree.filter(pl.col("generation") == last_gen)["id"]
    sample = pl.DataFrame({"id": sample_ids}).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("t1"),
        pl.lit(None, dtype=pl.Float64).alias("t2"),
        pl.lit(80.0).alias("death_age"),
        *(pl.lit(1.0 if c.startswith("t_") else False).alias(c) for c in COHORT_COLUMNS[4:]),
    )
    analysis_pedigree = filter_pedigree_to_observed(tiny_pedigree, sample_ids.to_numpy())
    write_cohort(build_cohort(analysis_pedigree, sample), cohort_path)
    assert len(analysis_pedigree) > len(sample_ids)
    with open(params_path, "w") as f:
        yaml.safe_dump({"N": 200, "mating_lambda": 0.5}, f)

    run_effective_size(
        pedigree_path=str(ped_path),
        cohort_path=str(cohort_path),
        params_path=str(params_path),
        output_path=str(out_path),
    )

    assert out_path.exists()
    with open(out_path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    assert set(loaded.keys()) == EXPECTED_KEYS
    for name, entry in loaded.items():
        if _is_unavailable(entry):
            assert entry["reason"] in ("not_requested", "missing_metadata"), name
            assert "ne" not in entry, name
            continue
        assert "ne" in entry
        assert "expected" in entry


@pytest.mark.parametrize(
    ("argv_extra", "expected_skip"),
    [
        pytest.param([], True, id="default-skips-ne-c"),
        pytest.param(["--ne-coancestry"], False, id="opt-in-runs-ne-c"),
    ],
)
def test_cli_ne_coancestry_flag_routes_to_main(monkeypatch, tmp_path, argv_extra, expected_skip):
    """``--ne-coancestry`` is a positive opt-in, negated on the way to main().

    Off by default, matching the ``analysis.skip_ne_coancestry`` pipeline
    default and pedsum's flag of the same name.
    """
    from simace.analysis.stats import effective_size as mod

    captured: dict[str, object] = {}

    def fake_main(pedigree_path, cohort_path, params_path, output_path, *, skip_ne_coancestry=False):
        captured["skip_ne_coancestry"] = skip_ne_coancestry
        Path(output_path).write_text("{}\n")

    monkeypatch.setattr(mod, "main", fake_main)
    out = tmp_path / "effective_size.yaml"
    mod.cli(
        [
            "--pedigree",
            "/dev/null/ped",
            "--cohort",
            "/dev/null/cohort",
            "--params",
            "/dev/null/params",
            "--output",
            str(out),
            *argv_extra,
        ]
    )
    assert captured == {"skip_ne_coancestry": expected_skip}
    assert out.exists()
