"""End-to-end integration test: simulate → phenotype → censor → stats.

Runs on a tiny scenario (N=100, G_ped=2) to verify interface compatibility
between modules. Asserts output structure, not numerical correctness.
"""

import numpy as np
import pandas as pd
import pytest

from simace.censoring.censor import run_censor
from simace.phenotyping.phenotype import run_phenotype
from simace.simulation.simulate import run_simulation


@pytest.fixture(scope="module")
def integration_params():
    """Tiny scenario parameters for integration testing."""
    return {
        "seed": 99,
        "N": 100,
        "G_ped": 2,
        "G_sim": 3,
        "G_pheno": 2,
        "mating_lambda": 0.5,
        "p_mztwin": 0.02,
        "A1": 0.5,
        "C1": 0.2,
        "A2": 0.5,
        "C2": 0.2,
        "rA": 0.3,
        "rC": 0.5,
        "assort1": 0.0,
        "assort2": 0.0,
        "standardize": True,
        "beta1": 1.0,
        "beta2": 1.5,
        "beta_sex1": 0.0,
        "beta_sex2": 0.0,
        "phenotype_model1": "frailty",
        "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
        "phenotype_model2": "frailty",
        "phenotype_params2": {"distribution": "weibull", "scale": 333, "rho": 1.2},
        "censor_age": 80,
        "gen_censoring": {0: [80, 80], 1: [0, 80]},
        "death_scale": 164,
        "death_rho": 2.73,
    }


@pytest.fixture(scope="module")
def pedigree(integration_params):
    """Run simulation step."""
    p = integration_params
    return run_simulation(
        seed=p["seed"],
        N=p["N"],
        G_ped=p["G_ped"],
        G_sim=p["G_sim"],
        mating_lambda=p["mating_lambda"],
        p_mztwin=p["p_mztwin"],
        A1=p["A1"],
        C1=p["C1"],
        A2=p["A2"],
        C2=p["C2"],
        rA=p["rA"],
        rC=p["rC"],
        assort1=p["assort1"],
        assort2=p["assort2"],
    )


@pytest.fixture(scope="module")
def phenotype(pedigree, integration_params):
    """Run phenotype step."""
    p = integration_params
    return run_phenotype(
        pedigree,
        G_pheno=p["G_pheno"],
        seed=p["seed"],
        standardize=p["standardize"],
        phenotype_model1=p["phenotype_model1"],
        phenotype_model2=p["phenotype_model2"],
        beta1=p["beta1"],
        beta_sex1=p["beta_sex1"],
        phenotype_params1=p["phenotype_params1"],
        beta2=p["beta2"],
        beta_sex2=p["beta_sex2"],
        phenotype_params2=p["phenotype_params2"],
    )


@pytest.fixture(scope="module")
def censored(phenotype, integration_params):
    """Run censor step."""
    p = integration_params
    return run_censor(
        phenotype,
        censor_age=p["censor_age"],
        seed=p["seed"],
        gen_censoring=p["gen_censoring"],
        death_scale=p["death_scale"],
        death_rho=p["death_rho"],
    )


class TestSimulateStep:
    def test_returns_dataframe(self, pedigree):
        assert isinstance(pedigree, pd.DataFrame)

    def test_has_required_columns(self, pedigree):
        required = {
            "id",
            "generation",
            "sex",
            "mother",
            "father",
            "twin",
            "household_id",
            "A1",
            "C1",
            "E1",
            "liability1",
            "A2",
            "C2",
            "E2",
            "liability2",
        }
        assert required.issubset(set(pedigree.columns))

    def test_has_expected_generations(self, pedigree, integration_params):
        n_gens = pedigree["generation"].nunique()
        assert n_gens == integration_params["G_ped"]

    def test_nonempty(self, pedigree):
        assert len(pedigree) > 0


class TestPhenotypeStep:
    def test_returns_dataframe(self, phenotype):
        assert isinstance(phenotype, pd.DataFrame)

    def test_has_event_time_columns(self, phenotype):
        assert "t1" in phenotype.columns
        assert "t2" in phenotype.columns

    def test_event_times_positive(self, phenotype):
        assert (phenotype["t1"] > 0).all()
        assert (phenotype["t2"] > 0).all()

    def test_event_times_finite(self, phenotype):
        assert np.isfinite(phenotype["t1"]).all()
        assert np.isfinite(phenotype["t2"]).all()

    def test_preserves_pedigree_columns(self, phenotype):
        assert "id" in phenotype.columns
        assert "generation" in phenotype.columns
        assert "sex" in phenotype.columns
        assert "liability1" in phenotype.columns


class TestCensorStep:
    def test_returns_dataframe(self, censored):
        assert isinstance(censored, pd.DataFrame)

    def test_has_censoring_columns(self, censored):
        required = {
            "death_age",
            "age_censored1",
            "t_observed1",
            "death_censored1",
            "affected1",
            "age_censored2",
            "t_observed2",
            "death_censored2",
            "affected2",
        }
        assert required.issubset(set(censored.columns))

    def test_affected_is_boolean(self, censored):
        assert censored["affected1"].dtype == bool
        assert censored["affected2"].dtype == bool

    def test_observed_times_positive(self, censored):
        assert (censored["t_observed1"] > 0).all()
        assert (censored["t_observed2"] > 0).all()

    def test_observed_bounded_by_window(self, censored, integration_params):
        # Observed times should not exceed the max censoring window
        censor_age = integration_params["censor_age"]
        assert (censored["t_observed1"] <= censor_age + 1e-10).all()
        assert (censored["t_observed2"] <= censor_age + 1e-10).all()

    def test_preserves_row_count(self, phenotype, censored):
        assert len(censored) == len(phenotype)


class TestStatsStep:
    def test_compute_person_years_on_censored(self, censored, integration_params):
        from simace.analysis.stats import compute_person_years

        result = compute_person_years(
            censored,
            integration_params["censor_age"],
            integration_params["gen_censoring"],
        )
        assert "total" in result
        assert "deaths" in result
        assert "trait1" in result
        assert "trait2" in result
        assert result["total"] > 0

    def test_compute_mean_family_size_on_censored(self, censored):
        from simace.analysis.stats import compute_mean_family_size

        result = compute_mean_family_size(censored)
        if result:  # may be empty if only one generation phenotyped
            assert "mean" in result
            assert "n_families" in result
            assert result["mean"] > 0

    def test_compute_censoring_confusion(self, censored, integration_params):
        from simace.analysis.stats import compute_censoring_confusion

        result = compute_censoring_confusion(
            censored,
            integration_params["censor_age"],
            integration_params["gen_censoring"],
        )
        for trait in ["trait1", "trait2"]:
            if trait in result:
                cm = result[trait]
                assert cm["tp"] + cm["fn"] + cm["fp"] + cm["tn"] == cm["n"]
