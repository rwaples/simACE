"""End-to-end CLI tests for ``simace.phenotyping.phenotype.cli``.

Exercises argparse → ``from_cli`` → ``to_params_dict`` → ``from_config``
round-trip plus the eager-registration foreign-flag rejection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simace.phenotyping.phenotype import cli as phenotype_cli


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
    df = pd.DataFrame(
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
    df.to_parquet(path)
    return path


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["phenotype", *argv])
    phenotype_cli()


def test_cli_frailty_round_trip(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "phenotype.parquet"
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
    out = pd.read_parquet(output)
    assert "t1" in out.columns
    assert "t2" in out.columns
    assert np.all(np.isfinite(out["t1"]))
    assert np.all(out["t1"] > 0)


def test_cli_adult_round_trip(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "phenotype.parquet"
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
    out = pd.read_parquet(output)
    case_rate1 = (out["t1"] < 1e6).mean()
    assert 0.05 < case_rate1 < 0.20  # n=100 noisy; expect ~10%


def test_cli_foreign_flag_rejected(tmp_path, monkeypatch):
    pedigree = _write_pedigree(tmp_path)
    output = tmp_path / "phenotype.parquet"
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
