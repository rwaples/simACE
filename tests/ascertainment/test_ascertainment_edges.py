"""Edge-case + CLI tests for ``simace.ascertainment``."""

import sys

import numpy as np
import pandas as pd
import pytest

from simace.ascertainment import _sever_dangling_links, copy_passthrough_if_possible, run_ascertainment
from simace.ascertainment import cli as ascertain_cli
from simace.core.schema import CENSORED
from simace.simulation.simulate import run_simulation
from tests.conftest import schema_pad


@pytest.fixture
def small_pedigree():
    return run_simulation(
        seed=42,
        N=200,
        G_ped=3,
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


def _build_trait(pedigree: pd.DataFrame, *, g_pheno: int, n_cases: int, seed: int) -> pd.DataFrame:
    """Build a trait DataFrame with exactly ``n_cases`` affected1 individuals."""
    rng = np.random.default_rng(seed)
    max_gen = int(pedigree["generation"].max())
    min_gen = max_gen - g_pheno + 1
    phenotyped = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)
    n = len(phenotyped)
    affected1 = np.zeros(n, dtype=bool)
    affected1[rng.choice(n, min(n_cases, n), replace=False)] = True
    df = pd.DataFrame(
        {
            "id": phenotyped["id"].to_numpy(),
            "generation": phenotyped["generation"].to_numpy(),
            "sex": phenotyped["sex"].to_numpy(),
            "liability1": rng.standard_normal(n),
            "liability2": rng.standard_normal(n),
            "t_observed1": rng.uniform(10, 80, n),
            "t_observed2": rng.uniform(10, 80, n),
            "affected1": affected1,
            "affected2": rng.random(n) < 0.10,
        }
    )
    return schema_pad(df, CENSORED)


class TestEmptyPool:
    """Empty post-dropout trait pool returns empty outputs without error."""

    def test_empty_trait_input(self, small_pedigree):
        empty_trait = _build_trait(small_pedigree, g_pheno=1, n_cases=0, seed=1).iloc[0:0]
        ped_out, trait_out, ltm_out = run_ascertainment(
            small_pedigree,
            empty_trait,
            empty_trait,
            dropout_rate=0.0,
            N_sample=10,
            seed=7,
        )
        assert len(trait_out) == 0
        assert len(ltm_out) == 0
        assert len(ped_out) == 0


class TestControlsOnly:
    """``case_ascertainment_ratio=0`` draws only controls and clamps to n_controls."""

    def test_zero_ratio_draws_only_controls(self, small_pedigree):
        trait = _build_trait(small_pedigree, g_pheno=2, n_cases=50, seed=11)
        n_controls = int((~trait["affected1"]).sum())
        n_sample = min(20, n_controls)
        _, trait_out, _ = run_ascertainment(
            small_pedigree,
            trait,
            trait,
            dropout_rate=0.0,
            case_ascertainment_ratio=0.0,
            N_sample=n_sample,
            seed=3,
        )
        assert len(trait_out) == n_sample
        assert not trait_out["affected1"].any()

    def test_zero_ratio_clamps_when_nsample_exceeds_controls(self, small_pedigree):
        trait = _build_trait(small_pedigree, g_pheno=2, n_cases=100, seed=11)
        n_controls = int((~trait["affected1"]).sum())
        oversized = n_controls + 50
        _, trait_out, _ = run_ascertainment(
            small_pedigree,
            trait,
            trait,
            dropout_rate=0.0,
            case_ascertainment_ratio=0.0,
            N_sample=oversized,
            seed=4,
        )
        assert len(trait_out) == n_controls
        assert not trait_out["affected1"].any()


class TestDegeneratePool:
    """All-case or all-control pools fall through the degenerate branch."""

    def test_all_case_pool_with_nonunit_ratio(self, small_pedigree):
        trait = _build_trait(small_pedigree, g_pheno=1, n_cases=0, seed=5)
        trait["affected1"] = True
        _, trait_out, _ = run_ascertainment(
            small_pedigree,
            trait,
            trait,
            dropout_rate=0.0,
            case_ascertainment_ratio=5.0,  # ignored when degenerate
            N_sample=20,
            seed=6,
        )
        assert len(trait_out) == 20
        assert trait_out["affected1"].all()

    def test_all_control_pool_with_nonunit_ratio(self, small_pedigree):
        trait = _build_trait(small_pedigree, g_pheno=1, n_cases=0, seed=5)
        _, trait_out, _ = run_ascertainment(
            small_pedigree,
            trait,
            trait,
            dropout_rate=0.0,
            case_ascertainment_ratio=5.0,
            N_sample=20,
            seed=7,
        )
        assert len(trait_out) == 20
        assert not trait_out["affected1"].any()


class TestNsamplePassThroughLogging:
    """``N_sample >= n_pool`` triggers the info-log pass-through branch."""

    def test_nsample_exceeds_pool(self, small_pedigree, caplog):
        import logging

        trait = _build_trait(small_pedigree, g_pheno=1, n_cases=10, seed=2)
        with caplog.at_level(logging.INFO, logger="simace.ascertainment"):
            _, trait_out, _ = run_ascertainment(
                small_pedigree,
                trait,
                trait,
                dropout_rate=0.0,
                N_sample=len(trait) + 50,
                seed=8,
            )
        assert len(trait_out) == len(trait)
        assert any("passing all through" in rec.message for rec in caplog.records)


class TestSeverDanglingTwinLinks:
    """``_sever_dangling_links`` rewrites twin pointers outside the valid set to -1."""

    def test_twin_link_to_outside_id_severed(self):
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "mother": [-1, -1, -1],
                "father": [-1, -1, -1],
                # id 0 points at twin 999 (outside valid set); id 1 ↔ id 2 (both valid).
                "twin": [999, 2, 1],
            }
        )
        out = _sever_dangling_links(df, valid_ids=df["id"].to_numpy())
        assert out.loc[0, "twin"] == -1  # dangling severed
        assert out.loc[1, "twin"] == 2  # in-set survives
        assert out.loc[2, "twin"] == 1


class TestPassThroughCopyFastPath:
    """File-level no-op ascertainment can copy exact-ID inputs unchanged."""

    def test_copies_when_all_inputs_have_same_ordered_ids(self, tmp_path, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        ltm_path = tmp_path / "trait.simple_ltm.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"
        out_ltm = tmp_path / "out_ltm.parquet"

        small_pedigree.to_parquet(ped_path)
        small_pedigree.to_parquet(trait_path)
        small_pedigree.to_parquet(ltm_path)

        copied = copy_passthrough_if_possible(
            ped_path,
            trait_path,
            ltm_path,
            out_ped,
            out_trait,
            out_ltm,
            dropout_rate=0.0,
            N_sample=0,
        )

        assert copied is True
        pd.testing.assert_frame_equal(pd.read_parquet(out_ped), small_pedigree)
        pd.testing.assert_frame_equal(pd.read_parquet(out_trait), small_pedigree)
        pd.testing.assert_frame_equal(pd.read_parquet(out_ltm), small_pedigree)

    def test_declines_when_trait_is_only_a_phenotyped_subset(self, tmp_path, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        ltm_path = tmp_path / "trait.simple_ltm.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"
        out_ltm = tmp_path / "out_ltm.parquet"

        small_pedigree.to_parquet(ped_path)
        trait = _build_trait(small_pedigree, g_pheno=1, n_cases=10, seed=1)
        trait.to_parquet(trait_path)
        trait.to_parquet(ltm_path)

        copied = copy_passthrough_if_possible(
            ped_path,
            trait_path,
            ltm_path,
            out_ped,
            out_trait,
            out_ltm,
            dropout_rate=0.0,
            N_sample=0,
        )

        assert copied is False
        assert not out_ped.exists()
        assert not out_trait.exists()
        assert not out_ltm.exists()


class TestAscertainmentCLI:
    """End-to-end CLI: pedigree + 2 trait parquets in, 3 parquets out."""

    def test_cli_writes_three_outputs(self, tmp_path, monkeypatch, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        ltm_path = tmp_path / "trait.simple_ltm.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"
        out_ltm = tmp_path / "out_ltm.parquet"

        small_pedigree.to_parquet(ped_path)
        trait = _build_trait(small_pedigree, g_pheno=2, n_cases=30, seed=1)
        trait.to_parquet(trait_path)
        trait.to_parquet(ltm_path)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ascertain",
                "--pedigree",
                str(ped_path),
                "--trait",
                str(trait_path),
                "--trait-simple-ltm",
                str(ltm_path),
                "--out-pedigree",
                str(out_ped),
                "--out-trait",
                str(out_trait),
                "--out-trait-simple-ltm",
                str(out_ltm),
                "--dropout-rate",
                "0.2",
                "--N-sample",
                "20",
                "--seed",
                "9",
            ],
        )
        ascertain_cli()

        assert out_ped.exists()
        assert out_trait.exists()
        assert out_ltm.exists()
        result_trait = pd.read_parquet(out_trait)
        result_ltm = pd.read_parquet(out_ltm)
        assert len(result_trait) == 20
        # Branch consistency: same IDs in both trait outputs.
        np.testing.assert_array_equal(
            result_trait["id"].to_numpy(),
            result_ltm["id"].to_numpy(),
        )
