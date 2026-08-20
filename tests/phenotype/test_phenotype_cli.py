"""End-to-end CLI tests for ``simace.phenotype.cli``.

Exercises argparse → ``from_cli`` → ``to_params_dict`` → ``from_config``
round-trip plus the eager-registration foreign-flag rejection.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing
import pytest

from simace.phenotype import cli as phenotype_cli


def _write_pedigree(tmp_path: Path, n: int = 100, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    A1 = rng.standard_normal(n)
    C1 = rng.standard_normal(n)
    E1 = rng.standard_normal(n)
    L1 = A1 + C1 + E1
    A2 = rng.standard_normal(n)
    C2 = rng.standard_normal(n)
    E2 = rng.standard_normal(n)
    L2 = A2 + C2 + E2
    df = pl.DataFrame(
        {
            "id": np.arange(n),
            "generation": np.zeros(n, dtype=int),
            "sex": rng.integers(0, 2, n),
            "household_id": np.arange(n),
            "mother": np.full(n, -1, dtype=int),
            "father": np.full(n, -1, dtype=int),
            "twin": np.zeros(n, dtype=int),
            "A1": A1,
            "C1": C1,
            "E1": E1,
            "liability1": L1,
            "A2": A2,
            "C2": C2,
            "E2": E2,
            "liability2": L2,
        }
    )
    path = tmp_path / "pedigree.parquet"
    df.write_parquet(path)
    return path


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["phenotype", *argv])
    phenotype_cli()


def test_cli_frailty_round_trip(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "trait.parquet"
    _run_cli(
        monkeypatch,
        [
            "--pedigree",
            str(pedigree),
            "--output",
            str(output),
            "--seed",
            "42",
            "--G-pheno",
            "1",
            "--phenotype-model1",
            "frailty",
            "--frailty-distribution1",
            "weibull",
            "--frailty-scale1",
            "316.228",
            "--frailty-rho1",
            "2.0",
            "--phenotype-model2",
            "frailty",
            "--frailty-distribution2",
            "weibull",
            "--frailty-scale2",
            "316.228",
            "--frailty-rho2",
            "2.0",
        ],
    )
    out = pl.read_parquet(output)
    assert "t1" in out.columns
    assert "t2" in out.columns
    assert np.all(np.isfinite(out["t1"].to_numpy()))
    assert np.all(out["t1"].to_numpy() > 0)


def test_cli_adult_round_trip(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "trait.parquet"
    _run_cli(
        monkeypatch,
        [
            "--pedigree",
            str(pedigree),
            "--output",
            str(output),
            "--G-pheno",
            "1",
            "--phenotype-model1",
            "adult",
            "--adult-method1",
            "ltm",
            "--adult-prevalence1",
            "0.10",
            "--phenotype-model2",
            "adult",
            "--adult-method2",
            "cox",
            "--adult-prevalence2",
            "0.20",
        ],
    )
    out = pl.read_parquet(output)
    case_rate1 = (out["t1"] < 1e6).mean()
    assert 0.05 < case_rate1 < 0.20  # n=100 noisy; expect ~10%


def test_cli_foreign_flag_rejected(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "trait.parquet"
    with pytest.raises(ValueError, match=r"--frailty-rho1"):
        _run_cli(
            monkeypatch,
            [
                "--pedigree",
                str(pedigree),
                "--output",
                str(output),
                "--G-pheno",
                "1",
                "--phenotype-model1",
                "adult",
                "--adult-method1",
                "ltm",
                "--adult-prevalence1",
                "0.10",
                "--frailty-rho1",  # foreign
                "2.0",
                "--phenotype-model2",
                "adult",
                "--adult-method2",
                "ltm",
                "--adult-prevalence2",
                "0.10",
            ],
        )


def test_run_phenotype_polars_stage_contract(tmp_path):
    """Polars-only stage (ADR 0015): eager polars out, deterministic, no in-frame NaN."""
    from simace.core.parquet import load_parquet
    from simace.phenotype import run_phenotype

    ped_path = _write_pedigree(tmp_path, n=500, seed=3)
    ped = load_parquet(ped_path)
    kwargs = dict(
        G_pheno=1,
        seed=42,
        standardize="global",
        phenotype_model1="adult",
        phenotype_params1={"method": "ltm", "prevalence": 0.10, "cip_x0": 50.0, "cip_k": 0.1},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        beta2=1.0,
        beta_sex2=0.0,
    )

    out = run_phenotype(ped, **kwargs)

    assert isinstance(out, pl.DataFrame)
    assert {"t1", "t2"} <= set(out.columns)
    # missingness is null, never NaN, in stage frames (ADR 0015 null contract)
    assert out["t1"].is_nan().fill_null(False).sum() == 0
    # same seed → identical output
    polars.testing.assert_frame_equal(run_phenotype(ped, **kwargs), out)
