"""``simace cohort``: the merged phenotype, censor, ascertain stage and its two durable outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.testing
import pytest

import simace.cli.cohort_stage as cohort_stage
from simace.ascertainment.runner import cli as ascertain_cli
from simace.censoring.censor import cli as censor_cli
from simace.censoring.censor import run_censor
from simace.core.cohort import read_cohort, read_pedigree, selected_views, write_pedigree
from simace.core.parquet import load_parquet, normalize_for_parquet, save_parquet
from simace.core.yaml_io import load_yaml
from simace.phenotype.runner import cli as phenotype_cli
from simace.simulation.simulate import run_simulation

if TYPE_CHECKING:
    from pathlib import Path

SEED = 5
PHENOTYPE_FLAGS = [
    "--G-pheno", "2",
    "--phenotype-model1", "frailty",
    "--phenotype-params1", "{distribution: weibull, scale: 2160, rho: 0.8}",
    "--phenotype-model2", "frailty",
    "--phenotype-params2", "{distribution: weibull, scale: 333, rho: 1.2}",
    "--beta2", "1.5",
]  # fmt: skip
CENSOR_FLAGS = [
    "--censor-age",
    "80",
    "--death-scale",
    "164",
    "--death-rho",
    "2.73",
    "--gen-censoring",
    '{"3": [0, 45]}',
]
ASCERTAIN_FLAGS = ["--dropout-rate", "0.1", "--N-sample", "120", "--case-ascertainment-ratio", "2"]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("recorded") / "pedigree.parquet"
    pedigree = run_simulation(
        seed=SEED,
        N=200,
        G_ped=4,
        G_sim=4,
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
    write_pedigree(pedigree, path)
    return path


def _cohort(recorded: Path, out: Path, *extra: str) -> None:
    out.mkdir()
    cohort_stage.cli(
        [
            "--pedigree", str(recorded),
            "--output-cohort", str(out / "cohort.parquet"),
            "--output-phenotyped-population", str(out / "phenotyped_population.yaml"),
            "--seed", str(SEED),
            *PHENOTYPE_FLAGS, *CENSOR_FLAGS, *ASCERTAIN_FLAGS, *extra,
        ]
    )  # fmt: skip


def test_writes_only_the_marked_cohort_and_the_population_summary(recorded, tmp_path) -> None:
    out = tmp_path / "rep"
    _cohort(recorded, out)

    assert sorted(p.name for p in out.iterdir()) == ["cohort.parquet", "phenotyped_population.yaml"]
    assert pl.read_parquet_metadata(out / "cohort.parquet")["simace_layout"] == "2"
    selected_views(read_pedigree(recorded), read_cohort(out / "cohort.parquet"))

    summary = load_yaml(out / "phenotyped_population.yaml")
    generations = load_parquet(recorded, columns=["generation"])["generation"]
    phenotyped = generations.filter(generations >= generations.max() - 1)
    assert list(summary) == ["n_individuals", "n_generations", "prevalence"]
    assert (summary["n_individuals"], summary["n_generations"]) == (len(phenotyped), 2)
    assert list(summary["prevalence"]) == ["trait1", "trait2", "by_generation"]
    assert sorted(summary["prevalence"]["by_generation"]) == sorted(phenotyped.unique().to_list())


def test_matches_the_three_standalone_stages_through_files(recorded, tmp_path) -> None:
    _cohort(recorded, tmp_path / "merged")

    staged = tmp_path / "staged"
    staged.mkdir()
    common = ["--pedigree", str(recorded), "--seed", str(SEED)]
    phenotype_cli([*common, "--output", str(staged / "raw.parquet"), *PHENOTYPE_FLAGS])
    censor_cli(
        [*common, "--phenotype", str(staged / "raw.parquet"), "--output", str(staged / "full.parquet"), *CENSOR_FLAGS]
    )
    ascertain_cli(
        [
            *common,
            "--trait", str(staged / "full.parquet"),
            "--out-pedigree", str(staged / "pedigree.parquet"),
            "--out-trait", str(staged / "trait.parquet"),
            *ASCERTAIN_FLAGS,
        ]
    )  # fmt: skip

    views = selected_views(read_pedigree(recorded), read_cohort(tmp_path / "merged" / "cohort.parquet"))
    for got, name in ((views.pedigree, "pedigree.parquet"), (views.trait, "trait.parquet")):
        polars.testing.assert_frame_equal(got, load_parquet(staged / name), check_exact=True)


def test_refuses_a_pedigree_without_the_layout_marker(recorded, tmp_path) -> None:
    unmarked = tmp_path / "unmarked.parquet"
    save_parquet(load_parquet(recorded), unmarked)
    with pytest.raises(ValueError, match=r"predates results layout 2"):
        _cohort(unmarked, tmp_path / "rep")
    assert list((tmp_path / "rep").iterdir()) == []


def test_each_phase_sees_the_float32_values_a_file_would_hold(recorded, monkeypatch) -> None:
    """Onsets just above the death age in float64 but not in float32 must censor as the staged files did."""
    pedigree = read_pedigree(recorded)
    phenotyped = pedigree.filter(pl.col("generation") == pedigree["generation"].max())
    censor = {"censor_age": 1000.0, "gen_censoring": {}, "death_scale": 164.0, "death_rho": 2.73}
    u = 1.0 - np.random.default_rng(SEED + 1000).uniform(size=len(phenotyped))
    death_age = censor["death_scale"] * (-np.log(u)) ** (1 / censor["death_rho"])
    onset = death_age * (1 + 1e-12)
    raw = pl.DataFrame({"id": phenotyped["id"], "t1": onset, "t2": onset})
    monkeypatch.setattr(cohort_stage, "run_phenotype", lambda *_a, **_k: raw)

    cohort, _ = cohort_stage.run_cohort(pedigree, seed=SEED, phenotype={}, censor=censor, ascertain={})

    staged = run_censor(normalize_for_parquet(raw), pedigree, seed=SEED, **censor)
    unnormalized = run_censor(raw, pedigree, seed=SEED, **censor)
    assert not staged["affected1"].equals(unnormalized["affected1"])
    written = normalize_for_parquet(cohort.filter(pl.col("affected1").is_not_null()))
    polars.testing.assert_frame_equal(written, normalize_for_parquet(staged), check_exact=True)
