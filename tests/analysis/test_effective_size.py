"""Per-rep Ne wrapper: compute_effective_size + validator + runner integration."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pedigree_graph import PedigreeGraph

from simace.analysis.stats.effective_size import (
    compute_effective_size,
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

    def test_standard_random_mating(self):
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5}
        exp = theoretical_expectations(cfg)
        # Seven estimators expect Ne ≈ N; LTC under our formula gives N/2.
        for k in EXPECTED_KEYS - {"ne_long_term_contributions"}:
            assert exp[k] == pytest.approx(200.0)
        assert exp["ne_long_term_contributions"] == pytest.approx(100.0)


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
        cfg = {"N": 200, "assort1": 0.0, "assort2": 0.0, "mating_lambda": 0.5}
        result = compute_effective_size(tiny_pedigree, config=cfg)
        for k in EXPECTED_KEYS - {"ne_long_term_contributions"}:
            assert result[k]["expected"] == pytest.approx(200.0)
        assert result["ne_long_term_contributions"]["expected"] == pytest.approx(100.0)

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
