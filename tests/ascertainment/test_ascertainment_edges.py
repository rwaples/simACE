"""Edge-case + CLI tests for ``simace.ascertainment``."""

import numpy as np
import polars as pl
import polars.testing
import pytest

from simace.ascertainment import cli as ascertain_cli
from simace.ascertainment import copy_passthrough_if_possible, run_ascertainment
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


def _build_trait(pedigree: pl.DataFrame, *, g_pheno: int, n_cases: int, seed: int) -> pl.DataFrame:
    """Build a trait DataFrame with exactly ``n_cases`` affected1 individuals."""
    rng = np.random.default_rng(seed)
    max_gen = int(pedigree["generation"].max())
    min_gen = max_gen - g_pheno + 1
    phenotyped = pedigree.filter(pl.col("generation") >= min_gen)
    n = len(phenotyped)
    affected1 = np.zeros(n, dtype=bool)
    affected1[rng.choice(n, min(n_cases, n), replace=False)] = True
    df = pl.DataFrame(
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
        empty_trait = _build_trait(small_pedigree, g_pheno=1, n_cases=0, seed=1).head(0)
        ped_out, trait_out = run_ascertainment(
            small_pedigree,
            empty_trait,
            dropout_rate=0.0,
            N_sample=10,
            seed=7,
        )
        assert len(trait_out) == 0
        assert len(ped_out) == 0


class TestPassThroughCopyFastPath:
    """File-level no-op ascertainment can copy exact-ID inputs unchanged."""

    def test_copies_when_all_inputs_have_same_ordered_ids(self, tmp_path, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"

        small_pedigree.write_parquet(ped_path)
        small_pedigree.write_parquet(trait_path)

        copied = copy_passthrough_if_possible(
            ped_path,
            trait_path,
            out_ped,
            out_trait,
            dropout_rate=0.0,
            N_sample=0,
        )

        assert copied is True
        polars.testing.assert_frame_equal(pl.read_parquet(out_ped), small_pedigree)
        polars.testing.assert_frame_equal(pl.read_parquet(out_trait), small_pedigree)

    def test_declines_when_trait_is_only_a_phenotyped_subset(self, tmp_path, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"

        small_pedigree.write_parquet(ped_path)
        trait = _build_trait(small_pedigree, g_pheno=1, n_cases=10, seed=1)
        trait.write_parquet(trait_path)

        copied = copy_passthrough_if_possible(
            ped_path,
            trait_path,
            out_ped,
            out_trait,
            dropout_rate=0.0,
            N_sample=0,
        )

        assert copied is False
        assert not out_ped.exists()
        assert not out_trait.exists()


class TestAscertainmentCLI:
    """End-to-end CLI: pedigree + trait parquet in, 2 parquets out."""

    def test_cli_writes_outputs(self, tmp_path, small_pedigree):
        ped_path = tmp_path / "pedigree.parquet"
        trait_path = tmp_path / "trait.parquet"
        out_ped = tmp_path / "out_ped.parquet"
        out_trait = tmp_path / "out_trait.parquet"

        small_pedigree.write_parquet(ped_path)
        trait = _build_trait(small_pedigree, g_pheno=2, n_cases=30, seed=1)
        trait.write_parquet(trait_path)

        ascertain_cli(
            [
                "--pedigree",
                str(ped_path),
                "--trait",
                str(trait_path),
                "--out-pedigree",
                str(out_ped),
                "--out-trait",
                str(out_trait),
                "--dropout-rate",
                "0.2",
                "--N-sample",
                "20",
                "--seed",
                "9",
            ],
        )

        assert out_ped.exists()
        assert out_trait.exists()
        result_trait = pl.read_parquet(out_trait)
        assert len(result_trait) == 20
