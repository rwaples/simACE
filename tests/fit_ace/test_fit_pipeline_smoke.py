"""End-to-end smoke test: simulate → threshold phenotype → Falconer h².

Runs a tiny pedigree through the sim_ace pipeline and fits Falconer h² via
fit_ace.ltm.falconer on the binary phenotype. Asserts only that the pipeline
wires up and produces a finite estimate — not a precise recovery test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import fit_ace.ltm.falconer as falconer_mod
from sim_ace.phenotyping.threshold import run_threshold
from sim_ace.simulation.simulate import run_simulation


@pytest.fixture(scope="module")
def smoke_pedigree_and_phenotype():
    pedigree = run_simulation(
        seed=7,
        N=1500,
        G_ped=3,
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.6,
        C1=0.0,
        A2=0.5,
        C2=0.0,
        rA=0.0,
        rC=0.0,
        G_sim=4,
    )
    phenotype = run_threshold(
        pedigree,
        {"G_pheno": 3, "prevalence1": 0.20, "prevalence2": 0.20},
    )
    return pedigree, phenotype


def test_falconer_runs_against_sim_output(smoke_pedigree_and_phenotype, tmp_path):
    pedigree, phenotype = smoke_pedigree_and_phenotype

    simple_ltm_path = tmp_path / "phenotype.simple_ltm.parquet"
    pedigree_path = tmp_path / "pedigree.parquet"
    phenotype.to_parquet(simple_ltm_path)
    pedigree.to_parquet(pedigree_path)

    output_path = tmp_path / "falconer.json"
    falconer_mod.main(
        simple_ltm_path=str(simple_ltm_path),
        pedigree_path=str(pedigree_path),
        output_path=str(output_path),
        kinds=["FS"],
        seed=0,
    )

    assert output_path.exists()
    result = json.loads(output_path.read_text())
    assert "FS" in result
    fs = result["FS"]
    assert "h2_falconer" in fs
    assert np.isfinite(fs["h2_falconer"])
    assert fs["n_pairs"] > 0
