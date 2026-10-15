"""Recorded-pedigree truth under C transmission (ADR 0025): the moments, the report block, the view, gather."""

import numpy as np
import pytest
import yaml

from simace.analysis.analyze import run_analysis
from simace.analysis.gather import extract_metrics
from simace.analysis.truth import AM_LIMITATION, recorded_pedigree_truth
from simace.core.cohort import read_c_transmission_clips, write_cohort, write_pedigree
from simace.core.parquet import normalize_for_parquet
from simace.core.relationships import RELATIONSHIP_TYPES, shared_environment_coefficient
from simace.core.yaml_io import dump_yaml, load_yaml
from simace.plotting.stats_report import plotting_report_view
from simace.simulation.simulate import run_simulation

_SIM = dict(
    N=60_000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.4,
    C1=0.3,
    E1=0.3,
    A2=0.3,
    C2=0.2,
    E2=0.5,
    rA=0.2,
    rC=0.4,
    rE=0.0,
)
_CT = {"Ctm_maternal1": 0.5, "Ctm_paternal1": 0.3}
_RM = {"assort1": 0.0, "assort2": 0.0, "mating_model": "standard"}


@pytest.fixture(scope="module")
def rm_truth():
    ped = run_simulation(seed=4, **_SIM, **_CT)
    return ped, recorded_pedigree_truth(ped, {**_SIM, **_RM, **_CT}, max_degree=2)


def test_no_truth_without_transmission():
    ped = run_simulation(seed=1, **{**_SIM, "N": 500})
    assert recorded_pedigree_truth(ped, {**_SIM, **_RM}, max_degree=2) is None


@pytest.mark.parametrize("code", ["FS", "MO", "FO", "MHS", "PHS"])
def test_shared_c_matches_the_random_mating_formula(rm_truth, code):
    _, truth = rm_truth
    entry = truth["traits"]["trait1"]["relationships"][code]
    gamma = shared_environment_coefficient(code, _CT["Ctm_maternal1"], _CT["Ctm_paternal1"])
    assert entry["cov_C"] / _SIM["C1"] == pytest.approx(gamma, abs=0.04)


def test_liability_covariance_decomposes_into_components(rm_truth):
    _, truth = rm_truth
    for code, entry in truth["traits"]["trait1"]["relationships"].items():
        parts = entry["cov_A"] + entry["cov_C"] + entry["cov_AC"] + entry["cov_CA"]
        assert parts == pytest.approx(entry["cov_L"], abs=0.02), code  # the rest is E, zero in expectation


def test_categories_window_and_midparent_slope(rm_truth):
    ped, truth = rm_truth
    relationships = truth["traits"]["trait1"]["relationships"]
    assert set(RELATIONSHIP_TYPES) | {"GP", "Av"} <= set(relationships)
    assert list(relationships["MO"]["by_generation"]) == [1, 2]  # last three of 0..2, founders excluded
    assert relationships["MO"]["n_pairs"] == sum(g["n_pairs"] for g in relationships["MO"]["by_generation"].values())
    slope = truth["traits"]["trait1"]["midparent_slope"]
    expected = (_SIM["A1"] + 0.8 * _SIM["C1"]) / (_SIM["A1"] + _SIM["C1"] + _SIM["E1"])
    assert slope["slope"] == pytest.approx(expected, abs=0.02)
    assert slope["n"] == int((ped["generation"] >= 1).sum())
    assert truth["limitations"] == []


def test_clips_round_trip_through_the_pedigree_and_reach_the_truth(tmp_path):
    ped = run_simulation(seed=0, **{**_SIM, "N": 1000}, Ctm_maternal1=1.0, Ctm_maternal2=1.0)
    clips = [{"generation": 1, "excess": 0.003, "tolerance": 0.02, "direction": [1.0, 0.0]}]
    write_pedigree(normalize_for_parquet(ped), tmp_path / "p.parquet", c_transmission_clips=clips)
    assert read_c_transmission_clips(tmp_path / "p.parquet") == clips
    write_pedigree(normalize_for_parquet(ped), tmp_path / "q.parquet")
    assert read_c_transmission_clips(tmp_path / "q.parquet") is None

    params = {**_SIM, **_RM, "Ctm_maternal1": 1.0, "Ctm_maternal2": 1.0}
    truth = recorded_pedigree_truth(ped, params, max_degree=1, clips=clips)
    assert truth["c_transmission_clips"] == clips
    assert "exceeded the configured C in 1 generation(s) (1;" in truth["limitations"][0]
    clean = recorded_pedigree_truth(ped, params, max_degree=1, clips=[])
    assert clean["c_transmission_clips"] == []
    assert clean["limitations"] == []
    assert "c_transmission_clips" not in recorded_pedigree_truth(ped, params, max_degree=1)


def test_assortative_mating_records_the_limitation():
    params = {**_SIM, **_RM, "assort1": 0.3, **_CT}
    ped = run_simulation(seed=2, **{**_SIM, "N": 2000}, assort1=0.3, **_CT)
    assert recorded_pedigree_truth(ped, params, max_degree=1)["limitations"] == [AM_LIMITATION]


def _analyze(tmp_path, pedigree_kwargs, params):
    from simace.cli.cohort_stage import run_cohort

    pedigree = normalize_for_parquet(run_simulation(seed=42, **pedigree_kwargs))
    cohort, population = run_cohort(
        pedigree,
        seed=42,
        phenotype={
            "G_pheno": 3,
            "standardize": True,
            "phenotype_model1": "frailty",
            "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
            "beta1": 1.0,
            "beta_sex1": 0.0,
            "phenotype_model2": "frailty",
            "phenotype_params2": {"distribution": "weibull", "scale": 333, "rho": 1.2},
            "beta2": 1.0,
            "beta_sex2": 0.0,
        },
        censor={"censor_age": 80, "gen_censoring": {}, "death_scale": 164, "death_rho": 2.73},
        ascertain={},
    )
    rep = tmp_path / "test" / "transmission" / "rep1"
    rep.mkdir(parents=True)
    write_pedigree(pedigree, rep / "pedigree.parquet")
    write_cohort(cohort, rep / "cohort.parquet")
    dump_yaml(population, rep / "phenotyped_population.yaml")
    (rep / "params.yaml").write_text(yaml.safe_dump(params))
    run_analysis(
        pedigree_path=str(rep / "pedigree.parquet"),
        params_path=str(rep / "params.yaml"),
        cohort_path=str(rep / "cohort.parquet"),
        phenotyped_population_path=str(rep / "phenotyped_population.yaml"),
        report_output=str(rep / "report.yaml"),
        plot_payload_output=str(rep / "plot_payload.yaml"),
        samples_output=str(rep / "plotting_sample.parquet"),
        folder="test",
        scenario="transmission",
        censor_age=80.0,
        max_degree=2,
    )
    return rep / "report.yaml"


@pytest.mark.slow
def test_report_view_and_gather_carry_the_truth(tmp_path):
    sim = {**_SIM, "N": 1000}
    report_path = _analyze(tmp_path, {**sim, **_CT}, {**sim, **_RM, **_CT, "seed": 42, "rep": 1})
    report = load_yaml(report_path)
    recorded = report["truth"]["recorded_pedigree"]
    assert recorded["limitations"] == []
    mo = recorded["traits"]["trait1"]["relationships"]["MO"]
    assert set(mo) == {"n_pairs", "r", "cov_L", "cov_A", "cov_C", "cov_AC", "cov_CA", "by_generation"}
    assert recorded["traits"]["trait1"]["midparent_slope"]["slope"] is not None

    view = plotting_report_view(report)
    assert view["truth_relationships"]["trait1"]["MO"]["r"] == mo["r"]
    assert view["truth_midparent_slope"]["trait2"] == recorded["traits"]["trait2"]["midparent_slope"]

    row = extract_metrics(str(report_path))
    assert row["truth_r_MO_trait1"] == mo["r"]
    assert row["Ctm_maternal1"] == 0.5
    assert np.isfinite(row["truth_midparent_slope_trait1"])


@pytest.mark.slow
def test_no_truth_block_without_transmission(tmp_path):
    sim = {**_SIM, "N": 1000}
    report_path = _analyze(tmp_path, sim, {**sim, **_RM, "seed": 42, "rep": 1})
    report = load_yaml(report_path)
    recorded = report["truth"]["recorded_pedigree"]
    assert "limitations" not in recorded
    assert all("relationships" not in t for t in recorded["traits"].values())
    assert plotting_report_view(report)["truth_relationships"] == {}
    row = extract_metrics(str(report_path))
    assert row["truth_r_MO_trait1"] is None
    assert row["Ctm_paternal2"] == 0.0
