"""Per-rep Ne wrapper: compute_effective_size + validator + runner integration."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pedigree_graph import PedigreeGraph

from simace.analysis.stats.effective_size import (
    compute_effective_size,
    ne_v_expected_ztp,
    regression_estimator_regime_ok,
    theoretical_expectations,
)
from simace.analysis.stats.runner import main as run_stats
from simace.analysis.validate import validate_effective_size

EXPECTED_KEYS = {
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_hill_overlapping",
    "ne_caballero_toro",
}


@pytest.fixture(scope="module")
def tiny_pedigree() -> pd.DataFrame:
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
        # All six drift/variance estimators expected = ne_v; Ne_sr = N; Ne_LTC = ne_v/2.
        for k in EXPECTED_KEYS - {"ne_sex_ratio", "ne_long_term_contributions"}:
            assert exp[k] == pytest.approx(ne_v)
        assert exp["ne_sex_ratio"] == pytest.approx(200.0)
        assert exp["ne_long_term_contributions"] == pytest.approx(ne_v / 2.0)
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
        assert exp["ne_caballero_toro"] is None
        # Variance/cohort-mean estimators stay populated.
        assert exp["ne_variance_family_size"] == pytest.approx(ne_v)
        assert exp["ne_individual_delta_f"] == pytest.approx(ne_v)
        assert exp["ne_hill_overlapping"] == pytest.approx(ne_v)
        assert exp["ne_sex_ratio"] == pytest.approx(100000.0)
        assert exp["ne_long_term_contributions"] == pytest.approx(ne_v / 2.0)

    def test_missing_g_ped_drops_regression_estimators(self):
        cfg = {"N": 100000, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5}
        exp = theoretical_expectations(cfg)
        assert exp["ne_inbreeding"] is None
        assert exp["ne_coancestry"] is None
        assert exp["ne_caballero_toro"] is None
        # Other estimators still populated.
        assert exp["ne_variance_family_size"] is not None

    def test_missing_mating_lambda_disables_expectations(self):
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0}
        exp = theoretical_expectations(cfg)
        assert all(v is None for v in exp.values())


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
            assert "ne" in entry
            assert "expected" in entry
            assert entry["expected"] is None  # no config provided

    def test_expected_attached_when_config_standard(self, tiny_pedigree):
        # G_ped=20 puts the regime check well into "ok" for N=200, so all
        # six drift/variance estimators receive expectations.
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5, "G_ped": 20}
        result = compute_effective_size(tiny_pedigree, config=cfg)
        ne_v = ne_v_expected_ztp(200, 0.5)
        for k in EXPECTED_KEYS - {"ne_sex_ratio", "ne_long_term_contributions"}:
            assert result[k]["expected"] == pytest.approx(ne_v)
        assert result["ne_sex_ratio"]["expected"] == pytest.approx(200.0)
        assert result["ne_long_term_contributions"]["expected"] == pytest.approx(ne_v / 2.0)

    def test_per_gen_series_length_matches_n_generations(self, tiny_pedigree):
        result = compute_effective_size(tiny_pedigree)
        n_gens = int(tiny_pedigree["generation"].max()) + 1
        # ne_inbreeding and ne_coancestry expose ne_per_gen aligned to cohort 0..G.
        assert len(result["ne_inbreeding"]["ne_per_gen"]) == n_gens
        assert len(result["ne_coancestry"]["mean_theta_per_gen"]) == n_gens
        assert len(result["ne_sex_ratio"]["n_male_per_gen"]) == n_gens


# ---------------------------------------------------------------------------
# validate_effective_size
# ---------------------------------------------------------------------------


class TestValidateEffectiveSize:
    def test_passes_when_observed_within_tolerance(self):
        stats = {
            "effective_size": {
                "ne_inbreeding": {"ne": 195.0, "expected": 200.0},
                "ne_sex_ratio": {"ne": 200.0, "expected": 200.0},
            },
        }
        out = validate_effective_size(stats, params={})
        assert out["ne_inbreeding"]["passed"] is True
        assert out["ne_sex_ratio"]["passed"] is True

    def test_fails_when_observed_off_by_more_than_20pct(self):
        stats = {
            "effective_size": {
                "ne_inbreeding": {"ne": 100.0, "expected": 200.0},  # 50% off
            },
        }
        out = validate_effective_size(stats, params={})
        assert out["ne_inbreeding"]["passed"] is False
        assert out["ne_inbreeding"]["relative_error"] == pytest.approx(0.5)

    def test_passes_vacuously_when_expected_none(self):
        stats = {
            "effective_size": {
                "ne_inbreeding": {"ne": 200.0, "expected": None},
            },
        }
        out = validate_effective_size(stats, params={})
        assert out["ne_inbreeding"]["passed"] is True
        assert out["ne_inbreeding"]["expected"] is None

    def test_returns_empty_when_no_effective_size_block(self):
        out = validate_effective_size({}, params={})
        assert out == {}


# ---------------------------------------------------------------------------
# Cross-estimator consistency under random mating (excludes Ne_LTC)
# ---------------------------------------------------------------------------


def _build_wf_pedigree(rng: np.random.Generator, n: int = 50, n_gens: int = 8) -> pd.DataFrame:
    """Symmetric Wright–Fisher pedigree (alternating M/F sex, multinomial parents)."""
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
    from pedigree_graph import ne_variance_family_size

    from simace.simulation.simulate import run_simulation

    n = 2000
    n_reps = 12
    mating_lambda = 0.5
    expected = ne_v_expected_ztp(n, mating_lambda)

    per_transition: list[float] = []
    for rep in range(n_reps):
        ped = run_simulation(
            seed=1000 + rep,
            N=n,
            G_ped=4,
            G_sim=5,
            mating_lambda=mating_lambda,
            p_mztwin=0.0,
            A1=0.5, C1=0.0, E1=0.5,
            A2=0.5, C2=0.0, E2=0.5,
            rA=0.0, rC=0.0,
            assort1=0.0, assort2=0.0,
        )
        pg = PedigreeGraph(ped)
        result = ne_variance_family_size(pg)
        finite = result.ne_per_transition[np.isfinite(result.ne_per_transition)]
        per_transition.extend(finite.tolist())

    mean_ne = float(np.mean(per_transition))
    rel_err = abs(mean_ne / expected - 1.0)
    assert rel_err < 0.05, (
        f"Ne_V mean across {len(per_transition)} transitions = {mean_ne:.1f}, "
        f"expected {expected:.1f} (rel err {rel_err:.3f}); formula needs reviewing."
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
        "ne_caballero_toro",
    )
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_reps):
        df = _build_wf_pedigree(rng)
        pg = PedigreeGraph(df)
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


def test_runner_attaches_effective_size_block(tmp_path, tiny_pedigree):
    """run_stats should include an `effective_size` block with all 8 keys.

    Uses the pedigree as both the phenotype input (no censoring needed) and the
    pedigree input — the runner only requires the phenotype DataFrame to expose
    enough columns for downstream stats.  Per-rep `params` is passed inline so
    the validator-facing `expected` field is populated.
    """
    from simace.censoring.censor import run_censor
    from simace.phenotyping.phenotype import run_phenotype

    phenotype = run_phenotype(
        tiny_pedigree,
        G_pheno=2,
        seed=7,
        standardize=True,
        phenotype_model1="frailty",
        phenotype_params1={"distribution": "weibull", "scale": 2160, "rho": 0.8},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 333, "rho": 1.2},
        beta2=1.0,
        beta_sex2=0.0,
    )
    censored = run_censor(
        phenotype,
        censor_age=80,
        seed=7,
        gen_censoring={},
        death_scale=164,
        death_rho=2.73,
    )

    ped_path = tmp_path / "pedigree.parquet"
    phe_path = tmp_path / "phenotype.parquet"
    tiny_pedigree.to_parquet(ped_path)
    censored.to_parquet(phe_path)

    stats_yaml = tmp_path / "phenotype_stats.yaml"
    samples_pq = tmp_path / "phenotype_samples.parquet"

    run_stats(
        phenotype_path=str(phe_path),
        censor_age=80.0,
        stats_output=str(stats_yaml),
        samples_output=str(samples_pq),
        seed=42,
        pedigree_path=str(ped_path),
        max_degree=2,
        params={"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5},
    )

    with open(stats_yaml, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    assert "effective_size" in loaded
    assert set(loaded["effective_size"].keys()) == EXPECTED_KEYS
    for entry in loaded["effective_size"].values():
        assert "ne" in entry
        assert "expected" in entry
