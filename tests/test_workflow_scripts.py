"""Smoke tests for the Tier 1 Snakemake wrappers.

Each wrapper is exec'd as a script with a hand-rolled ``snakemake`` namespace
injected into the module globals — the same shape Snakemake itself uses.  The
test asserts the output file exists; behavioral correctness lives in the
underlying domain-function tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "workflow" / "scripts" / "simace"


def _exec_wrapper(script: Path, snakemake_obj: SimpleNamespace) -> dict:
    """Exec a wrapper script with ``snakemake`` injected into module globals."""
    src = script.read_text()
    namespace: dict = {
        "__name__": "wrapper_smoke_test",
        "__file__": str(script),
        "snakemake": snakemake_obj,
    }
    exec(compile(src, str(script), "exec"), namespace)
    return namespace


def _make_snakemake(
    *,
    inputs: dict,
    outputs: dict,
    params: dict,
    log_path: Path,
    wildcards: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input=SimpleNamespace(**inputs),
        output=SimpleNamespace(**outputs),
        params=SimpleNamespace(**params),
        wildcards=SimpleNamespace(**(wildcards or {"folder": "test", "scenario": "smoke", "rep": "1"})),
        log=[str(log_path)],
    )


@pytest.fixture
def pedigree_parquet(tmp_path):
    """Tiny pedigree parquet file with the columns downstream functions expect."""
    rng = np.random.default_rng(0)
    n = 60
    path = tmp_path / "pedigree.parquet"
    pd.DataFrame(
        {
            "id": np.arange(n),
            "generation": np.repeat([0, 1, 2], n // 3),
            "sex": rng.integers(0, 2, n),
            "household_id": np.arange(n),
            "mother": np.full(n, -1, dtype=np.int64),
            "father": np.full(n, -1, dtype=np.int64),
            "twin": np.full(n, -1, dtype=np.int64),
            "A1": rng.standard_normal(n),
            "C1": rng.standard_normal(n),
            "E1": rng.standard_normal(n),
            "liability1": rng.standard_normal(n),
            "A2": rng.standard_normal(n),
            "C2": rng.standard_normal(n),
            "E2": rng.standard_normal(n),
            "liability2": rng.standard_normal(n),
        }
    ).to_parquet(path, index=False)
    return path


@pytest.fixture
def phenotype_parquet(tmp_path, pedigree_parquet):
    """Raw phenotype parquet (pedigree columns + t1/t2)."""
    df = pd.read_parquet(pedigree_parquet)
    rng = np.random.default_rng(1)
    df["t1"] = rng.uniform(10, 200, len(df))
    df["t2"] = rng.uniform(10, 200, len(df))
    path = tmp_path / "phenotype.raw.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def censored_phenotype_parquet(tmp_path, phenotype_parquet):
    """Censored phenotype parquet — provides the full CENSORED schema sample reads."""
    df = pd.read_parquet(phenotype_parquet)
    rng = np.random.default_rng(2)
    n = len(df)
    df["death_age"] = rng.uniform(50, 90, n).astype(float)
    df["age_censored1"] = rng.choice([True, False], n)
    df["t_observed1"] = rng.uniform(10, 200, n).astype(float)
    df["death_censored1"] = rng.choice([True, False], n)
    df["affected1"] = rng.choice([True, False], n)
    df["age_censored2"] = rng.choice([True, False], n)
    df["t_observed2"] = rng.uniform(10, 200, n).astype(float)
    df["death_censored2"] = rng.choice([True, False], n)
    df["affected2"] = rng.choice([True, False], n)
    path = tmp_path / "phenotype.parquet"
    df.to_parquet(path, index=False)
    return path


def test_sample_wrapper(tmp_path, censored_phenotype_parquet):
    out = tmp_path / "phenotype.sampled.parquet"
    sm = _make_snakemake(
        inputs={"phenotype": str(censored_phenotype_parquet)},
        outputs={"phenotype": str(out)},
        params={"N_sample": 30, "case_ascertainment_ratio": 1.0, "seed": 42},
        log_path=tmp_path / "sample.log",
    )
    _exec_wrapper(SCRIPT_DIR / "sample.py", sm)
    assert out.exists()
    assert len(pd.read_parquet(out)) == 30


def test_dropout_wrapper(tmp_path, pedigree_parquet):
    out = tmp_path / "pedigree.dropped.parquet"
    sm = _make_snakemake(
        inputs={"pedigree": str(pedigree_parquet)},
        outputs={"pedigree": str(out)},
        params={"pedigree_dropout_rate": 0.2, "seed": 42},
        log_path=tmp_path / "dropout.log",
    )
    _exec_wrapper(SCRIPT_DIR / "dropout.py", sm)
    assert out.exists()
    n_in = len(pd.read_parquet(pedigree_parquet))
    n_out = len(pd.read_parquet(out))
    assert n_out == n_in - round(n_in * 0.2)


def test_censor_wrapper(tmp_path, phenotype_parquet):
    out = tmp_path / "phenotype.censored.parquet"
    sm = _make_snakemake(
        inputs={"phenotype": str(phenotype_parquet)},
        outputs={"phenotype": str(out)},
        params={
            "censor_age": 80,
            "seed": 42,
            "gen_censoring": {0: [80, 80], 1: [0, 80], 2: [0, 80]},
            "death_scale": 164,
            "death_rho": 2.73,
        },
        log_path=tmp_path / "censor.log",
    )
    _exec_wrapper(SCRIPT_DIR / "censor.py", sm)
    df = pd.read_parquet(out)
    assert "affected1" in df.columns
    assert "t_observed1" in df.columns


def test_phenotype_wrapper(tmp_path, pedigree_parquet):
    out = tmp_path / "phenotype.raw.parquet"
    sm = _make_snakemake(
        inputs={"pedigree": str(pedigree_parquet)},
        outputs={"phenotype": str(out)},
        params={
            "G_pheno": 2,
            "seed": 42,
            "standardize": True,
            "phenotype_model1": "frailty",
            "phenotype_model2": "frailty",
            "beta1": 1.0,
            "beta_sex1": 0.0,
            "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
            "beta2": 1.0,
            "beta_sex2": 0.0,
            "phenotype_params2": {"distribution": "weibull", "scale": 333, "rho": 1.2},
        },
        log_path=tmp_path / "phenotype.log",
    )
    _exec_wrapper(SCRIPT_DIR / "phenotype.py", sm)
    df = pd.read_parquet(out)
    assert "t1" in df.columns
    assert "t2" in df.columns


def test_phenotype_threshold_wrapper(tmp_path, pedigree_parquet):
    out = tmp_path / "phenotype.threshold.parquet"
    sm = _make_snakemake(
        inputs={"pedigree": str(pedigree_parquet)},
        outputs={"phenotype": str(out)},
        params={
            "phenotype_params1": {"prevalence": 0.10},
            "phenotype_params2": {"prevalence": 0.20},
            "G_pheno": 2,
            "standardize": True,
        },
        log_path=tmp_path / "phenotype_threshold.log",
    )
    _exec_wrapper(SCRIPT_DIR / "phenotype_threshold.py", sm)
    df = pd.read_parquet(out)
    assert "affected1" in df.columns
    assert df["affected1"].dtype == bool


def test_parquet_to_tsv_wrapper(tmp_path, pedigree_parquet):
    out = tmp_path / "pedigree.tsv.gz"
    sm = SimpleNamespace(
        input=[str(pedigree_parquet)],
        output=[str(out)],
        params=SimpleNamespace(float_precision=4, gzip=True),
        wildcards=SimpleNamespace(),
        log=[str(tmp_path / "tsv.log")],
    )
    # snakemake.params on the rule supports .get(); SimpleNamespace doesn't.
    sm.params.get = lambda key, default=None: getattr(sm.params, key, default)
    _exec_wrapper(SCRIPT_DIR / "parquet_to_tsv.py", sm)
    assert out.exists()
    df = pd.read_csv(out, sep="\t")
    assert "id" in df.columns
