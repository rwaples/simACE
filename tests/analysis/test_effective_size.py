"""Per-rep Ne wrapper: compute_effective_size + validator + main() integration."""

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
    ne_v_expected_ztp,
    regression_estimator_regime_ok,
    theoretical_expectations,
)
from simace.analysis.stats.effective_size import (
    main as run_effective_size,
)
from simace.analysis.validate import validate_effective_size
from simace.core.parquet import save_parquet

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
    )


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

    def test_standard_random_mating_regression_regime_ok(self):
        # Small N=200 + reasonable G=20 ⇒ N·G² = 80,000 > 120·Ne_V, regime ok.
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 20}
        exp = theoretical_expectations(cfg)
        ne_v = ne_v_expected_ztp(200, 0.5)
        # Five drift/variance estimators expected = ne_v; Ne_sr = N;
        # Ne_LTC = the harmonic mean of N and Ne_V.
        for k in EXPECTED_KEYS - {"ne_sex_ratio", "ne_long_term_contributions", "ne_individual_delta_f"}:
            assert exp[k] == pytest.approx(ne_v)
        # G_ped=20 ⇒ t=19, so Ne_iΔF is expected 19/18 high (pedigree-graph #15).
        assert exp["ne_individual_delta_f"] == pytest.approx(ne_v * 19.0 / 18.0)
        assert exp["ne_sex_ratio"] == pytest.approx(200.0)
        assert 2.0 / exp["ne_long_term_contributions"] == pytest.approx(1.0 / 200.0 + 1.0 / ne_v)
        # Sanity: ZTP(0.5) gives ~0.7349·N.
        assert ne_v == pytest.approx(0.7349 * 200, abs=0.5)

    def test_baseline100K_default_drops_regression_estimators(self):
        # baseline100K config: N=100K, G_ped=6 ⇒ N·G² = 3.6M < 120·Ne_V (~8.8M),
        # regression-based Ne_I/Ne_C/Ne_CT must report None.
        cfg = {"N": 100000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 6}
        exp = theoretical_expectations(cfg)
        ne_v = ne_v_expected_ztp(100000.0, 0.5)
        assert exp["ne_inbreeding"] is None
        assert exp["ne_coancestry"] is None
        assert exp["ne_group_coancestry"] is None
        # Variance/cohort-mean estimators stay populated.
        assert exp["ne_variance_family_size"] == pytest.approx(ne_v)
        # G_ped=6 ⇒ the last cohort has t=5 equivalent complete generations but
        # only 4 of drift, so Ne_iΔF is expected 5/4 high (pedigree-graph #15).
        assert exp["ne_individual_delta_f"] == pytest.approx(ne_v * 5.0 / 4.0)
        assert exp["ne_hill_overlapping"] == pytest.approx(ne_v)
        assert exp["ne_sex_ratio"] == pytest.approx(100000.0)
        assert 2.0 / exp["ne_long_term_contributions"] == pytest.approx(1.0 / 100000.0 + 1.0 / ne_v)

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
        assert exp["ne_variance_family_size"] is not None

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
        # WF puts Ne_V at N, where the harmonic mean of N and Ne_V is N itself.
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
# ne_v_expected_ztp closed-form limits
# ---------------------------------------------------------------------------


class TestNeVExpectedZtp:
    def test_lambda_zero_limit_is_N(self):
        # ZTP degenerates to m=1 ⇒ no extra mating-count variance ⇒ Ne_V = N.
        assert ne_v_expected_ztp(1000, 0.0) == pytest.approx(1000.0)
        assert ne_v_expected_ztp(1000, 1e-9) == pytest.approx(1000.0, rel=1e-6)

    def test_large_lambda_limit_approaches_N(self):
        # Poisson, no truncation ⇒ Var[m]/E[m]² → 1/λ ⇒ slow approach to N.
        # At λ=100, Var[m]/E[m]² = 1/100, Ne_V/N = 1/1.02 ≈ 0.9804.
        assert ne_v_expected_ztp(1000, 100.0) == pytest.approx(1000.0 / 1.02, rel=1e-3)
        # By λ=10000 we're within 0.02 % of N.
        assert ne_v_expected_ztp(1000, 10000.0) == pytest.approx(1000.0, rel=1e-3)

    def test_default_lambda_05_value(self):
        # Numerically verified: 0.7349·N at λ=0.5 (matches baseline100K observation).
        assert ne_v_expected_ztp(100000.0, 0.5) == pytest.approx(73489.5, rel=1e-4)

    def test_monotone_in_lambda_below_unity(self):
        # ZTP overdispersion grows with λ for λ < 1 ⇒ Ne_V/N decreases.
        from itertools import pairwise

        n = 1000
        ratios = [ne_v_expected_ztp(n, lam) / n for lam in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for a, b in pairwise(ratios):
            assert a >= b, f"Not monotone: {ratios}"


class TestRegressionEstimatorRegimeOk:
    def test_returns_false_for_g_ped_below_2(self):
        # No slope possible with fewer than 2 transitions.
        assert not regression_estimator_regime_ok(1e9, 1, 1.0)

    def test_baseline100K_default_is_not_ok(self):
        # N=100K, G=6, Ne_V=73,485 ⇒ N·G² = 3.6M < 120·73,485 = 8.82M
        assert not regression_estimator_regime_ok(100000.0, 6, 73485.0)

    def test_high_g_ped_brings_regime_into_range(self):
        # N=100K, G=15 ⇒ N·G² = 22.5M > 8.82M
        assert regression_estimator_regime_ok(100000.0, 15, 73485.0)

    def test_small_N_with_modest_G(self):
        # N=200, G=20, Ne_V=147 ⇒ N·G² = 80K vs 120·147 = 17.6K, ok.
        assert regression_estimator_regime_ok(200.0, 20, 147.0)


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
        # G_ped=20 puts the regime check well into "ok" for N=200, so all
        # six drift/variance estimators receive expectations.
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 20}
        result = compute_effective_size(tiny_pedigree, config=cfg)
        ne_v = ne_v_expected_ztp(200, 0.5)
        for k in EXPECTED_KEYS - {"ne_sex_ratio", "ne_long_term_contributions", "ne_individual_delta_f"}:
            assert result[k]["expected"] == pytest.approx(ne_v)
        delta_f = result["ne_individual_delta_f"]
        assert delta_f["expected"] == pytest.approx(ne_v * 19.0 / 18.0)
        assert result["ne_sex_ratio"]["expected"] == pytest.approx(200.0)
        ltc = result["ne_long_term_contributions"]
        assert 2.0 / ltc["expected"] == pytest.approx(1.0 / 200.0 + 1.0 / ne_v)

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
        assert result["ne_variance_family_size"]["expected"] is not None


# ---------------------------------------------------------------------------
# validate_effective_size
# ---------------------------------------------------------------------------


class TestValidateEffectiveSize:
    def test_passes_when_observed_within_tolerance(self):
        ne_stats = {
            "ne_inbreeding": {"ne": 195.0, "expected": 200.0},
            "ne_sex_ratio": {"ne": 200.0, "expected": 200.0},
        }
        out = validate_effective_size(ne_stats, params={})
        assert out["ne_inbreeding"]["passed"] is True
        assert out["ne_sex_ratio"]["passed"] is True

    def test_fails_when_observed_off_by_more_than_20pct(self):
        ne_stats = {
            "ne_inbreeding": {"ne": 100.0, "expected": 200.0},  # 50% off
        }
        out = validate_effective_size(ne_stats, params={})
        assert out["ne_inbreeding"]["passed"] is False
        assert out["ne_inbreeding"]["relative_error"] == pytest.approx(0.5)

    def test_passes_vacuously_when_expected_none(self):
        ne_stats = {
            "ne_inbreeding": {"ne": 200.0, "expected": None},
        }
        out = validate_effective_size(ne_stats, params={})
        assert out["ne_inbreeding"]["passed"] is True
        assert out["ne_inbreeding"]["expected"] is None

    def test_not_requested_passes_vacuously(self):
        ne_stats = {"ne_coancestry": {"reason": "not_requested", "code": None, "fields": {}}}
        out = validate_effective_size(ne_stats, params={})
        assert out["ne_coancestry"]["passed"] is True
        assert out["ne_coancestry"]["observed"] is None

    def test_missing_metadata_fails_and_names_the_refusal_code(self):
        ne_stats = {
            "ne_sex_ratio": {"reason": "missing_metadata", "code": "missing_sex", "fields": {"status": "absent"}}
        }
        out = validate_effective_size(ne_stats, params={})
        assert out["ne_sex_ratio"]["passed"] is False
        assert out["ne_sex_ratio"]["code"] == "missing_sex"
        assert "missing_sex" in out["ne_sex_ratio"]["details"]

    def test_returns_empty_when_ne_stats_empty(self):
        assert validate_effective_size({}, params={}) == {}
        assert validate_effective_size(None, params={}) == {}


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
    )


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


@pytest.mark.slow
def test_ne_v_formula_matches_simulator_mc():
    """Closed-form ``ne_v_expected_ztp`` matches simACE simulator within ±5 %.

    Runs 12 reps at N=2000, G_ped=4 with default ``mating_lambda=0.5``,
    extracts the per-transition Ne_V from each rep, and asserts the
    grand mean agrees with ``ne_v_expected_ztp(2000, 0.5)`` to within
    ±5 %.  Tighter tolerance than the validator's ±20 % because we are
    averaging over ~3·12 = 36 transitions which suppresses the
    multinomial-allocation noise.
    """
    from pedigree_graph.effective_size import ne_variance_family_size

    n = 2000
    n_reps = 12
    mating_lambda = 0.5
    expected = ne_v_expected_ztp(n, mating_lambda)

    per_transition: list[float] = []
    for rep in range(n_reps):
        ped = _drift_only_pedigree(seed=1000 + rep, n=n, g_ped=4, mating_lambda=mating_lambda)
        pg = PedigreeGraph.from_frame(ped)
        result = ne_variance_family_size(pg)
        finite = result.ne_per_transition[np.isfinite(result.ne_per_transition)]
        per_transition.extend(finite.tolist())

    mean_ne = float(np.mean(per_transition))
    rel_err = abs(mean_ne / expected - 1.0)
    assert rel_err < 0.05, (
        f"Ne_V mean across {len(per_transition)} transitions = {mean_ne:.1f}, "
        f"expected {expected:.1f} (rel err {rel_err:.3f}); formula needs reviewing."
    )


@pytest.mark.parametrize(
    ("mating_model", "g_ped"),
    [("standard", 8), ("standard", 12), ("wright_fisher", 8)],
)
def test_ne_ltc_expectation_matches_simulator_mc(mating_model: str, g_ped: int):
    """``theoretical_expectations``' Ne_LTC matches the simulator within ±3 %.

    This is the committed method behind the numbers cited in
    :func:`theoretical_expectations`' docstring.  Wray & Thompson 1990 eq. 31
    with their p. 51 relation ``σ_r² = V(k)/2`` is ``4N/(2 + V(k))`` while
    ``ne_v_expected_ztp`` is ``2N/V(k)``, so the prediction is the harmonic mean
    of ``N`` and ``Ne_V`` and ``V(k)`` never has to be estimated.

    Reads ``2/Σc²`` out of ``sum_c_squared`` rather than the record's ``ne``
    because that field means the same thing under both pinned versions:
    pedigree-graph 0.8 reports ``1/(2·Σc²)``, four times lower, and withholds it
    entirely unless its asymptote gate fires, which on a stochastic pedigree it
    does not (ADR 0012).

    Measured at N=1000 over 12 reps: +0.75 % (standard, G_ped=8), +0.73 %
    (standard, G_ped=12), +0.58 % (Wright-Fisher, G_ped=8), each under 0.8
    standard errors of the replicate mean.  The ±3 % band is about 3.5 sem and
    still rejects every formula this replaced — the old ``Ne_V/2`` sits 57 %
    low under ZTP(0.5) and a bare ``Ne_V`` 13 % low.
    """
    from pedigree_graph.effective_size import ne_long_term_contributions

    n, mating_lambda, n_reps = 1000, 0.5, 12
    cfg = {
        "N": n,
        "assort1": 0.0,
        "assort2": 0.0,
        "mating_lambda": mating_lambda,
        "G_ped": g_ped,
        "mating_model": mating_model,
    }
    expected = theoretical_expectations(cfg)["ne_long_term_contributions"]

    observed = []
    for rep in range(n_reps):
        ped = _drift_only_pedigree(
            seed=1000 + rep, n=n, g_ped=g_ped, mating_lambda=mating_lambda, mating_model=mating_model
        )
        res = ne_long_term_contributions(PedigreeGraph.from_frame(ped))
        observed.append(2.0 / res.sum_c_squared)

    mean_ltc = float(np.mean(observed))
    rel_err = abs(mean_ltc / expected - 1.0)
    assert rel_err < 0.03, (
        f"2/Sum(c^2) mean across {n_reps} reps = {mean_ltc:.1f}, expected {expected:.1f} "
        f"(rel err {rel_err:.3f}); the W&T eq. 31 harmonic-mean relation needs reviewing."
    )


@pytest.mark.slow
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

    Phenotype input is restricted to the late generation so the closure logic
    is exercised (founders/intermediate ancestors are pulled back through
    parent-pointer walking).  Per-rep ``params.yaml`` is supplied inline so
    the validator-facing ``expected`` field is populated.
    """
    ped_path = tmp_path / "pedigree.parquet"
    phe_path = tmp_path / "trait.parquet"
    params_path = tmp_path / "params.yaml"
    out_path = tmp_path / "effective_size.yaml"

    save_parquet(tiny_pedigree, ped_path)
    # Observed = last generation only — closure must recover all ancestors.
    last_gen = int(tiny_pedigree["generation"].max())
    df_phe = tiny_pedigree.filter(pl.col("generation") == last_gen).select("id")
    save_parquet(df_phe, phe_path)
    with open(params_path, "w") as f:
        yaml.safe_dump({"N": 200, "mating_lambda": 0.5}, f)

    run_effective_size(
        pedigree_path=str(ped_path),
        phenotype_path=str(phe_path),
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

    def fake_main(pedigree_path, phenotype_path, params_path, output_path, *, skip_ne_coancestry=False):
        captured["skip_ne_coancestry"] = skip_ne_coancestry
        Path(output_path).write_text("{}\n")

    monkeypatch.setattr(mod, "main", fake_main)
    out = tmp_path / "effective_size.yaml"
    mod.cli(
        [
            "--pedigree",
            "/dev/null/ped",
            "--phenotype",
            "/dev/null/phe",
            "--params",
            "/dev/null/params",
            "--output",
            str(out),
            *argv_extra,
        ]
    )
    assert captured == {"skip_ne_coancestry": expected_skip}
    assert out.exists()
