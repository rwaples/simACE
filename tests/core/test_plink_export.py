"""PLINK export: recoding, the analysis pedigree with outcomes, and a round trip through the files."""

from __future__ import annotations

import polars as pl
import pytest

from simace.ascertainment.runner import run_ascertainment
from simace.censoring.censor import run_censor
from simace.core.cohort import build_cohort, selected_views, write_cohort, write_pedigree
from simace.core.parquet import normalize_for_parquet
from simace.core.plink_export import cli, write_plink
from simace.phenotype import run_phenotype
from simace.simulation.simulate import run_simulation

_LINK = pl.Int32


def _read_fam(path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        null_values=["NA", ""],
        has_header=False,
        new_columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
        schema_overrides={"FID": pl.String, "IID": pl.String, "PAT": pl.String, "MAT": pl.String},
    )


def _simace_id(col: str) -> pl.Expr:
    """Invert ``plink_iid``: ``id_<x>`` -> x, ``0`` -> -1."""
    return pl.when(pl.col(col) == "0").then(-1).otherwise(pl.col(col).str.strip_prefix("id_").cast(_LINK))


def test_recorded_pedigree_recoding(tmp_path) -> None:
    """Founders 0 (female) and 1 (male), MZ twins 2 and 3, and 4 whose father is unknown and sex is not 0/1."""
    pedigree = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "sex": [0, 1, 0, 0, 9],
            "mother": [-1, -1, 0, 0, 2],
            "father": [-1, -1, 1, 1, -1],
            "twin": [-1, -1, 3, 2, -1],
        },
        schema={"id": _LINK, "sex": pl.Int8, "mother": _LINK, "father": _LINK, "twin": _LINK},
    )
    written = write_plink(pedigree, tmp_path / "ped")

    assert sorted(p.name for p in written) == ["ped.fam", "ped.pheno", "ped.twins"]
    assert (tmp_path / "ped.fam").read_text().splitlines() == [
        "1\tid_0\t0\t0\t2\t-9",
        "1\tid_1\t0\t0\t1\t-9",
        "1\tid_2\tid_1\tid_0\t2\t-9",
        "1\tid_3\tid_1\tid_0\t2\t-9",
        "1\tid_4\t0\tid_2\t0\t-9",
    ]
    pheno = (tmp_path / "ped.pheno").read_text().splitlines()
    assert pheno[0] == "FID\tIID\taffected1\taffected2"
    assert pheno[1:] == [f"1\tid_{i}\t-9\t-9" for i in range(5)]
    assert (tmp_path / "ped.twins").read_text().splitlines() == ["IID1\tIID2", "id_2\tid_3"]


@pytest.fixture(scope="module")
def rep() -> tuple[pl.DataFrame, pl.DataFrame]:
    """A recorded pedigree and its cohort, with dropout so some links in the analysis pedigree are severed."""
    pedigree = normalize_for_parquet(
        run_simulation(
            seed=3,
            N=200,
            G_ped=3,
            G_sim=3,
            mating_lambda=0.5,
            p_mztwin=0.05,
            A1=0.5,
            C1=0.2,
            E1=0.3,
            A2=0.5,
            C2=0.2,
            E2=0.3,
            rA=0.0,
            rC=0.0,
            assort1=0.0,
            assort2=0.0,
        )
    )
    raw = normalize_for_parquet(
        run_phenotype(
            pedigree,
            G_pheno=2,
            seed=3,
            standardize="global",
            phenotype_model1="frailty",
            phenotype_params1={"distribution": "weibull", "scale": 2160, "rho": 0.8},
            beta1=1.0,
            beta_sex1=0.0,
            phenotype_model2="frailty",
            phenotype_params2={"distribution": "weibull", "scale": 333, "rho": 1.2},
            beta2=1.5,
            beta_sex2=0.0,
        )
    )
    censored = normalize_for_parquet(
        run_censor(raw, pedigree, censor_age=80, seed=3, gen_censoring={}, death_scale=164, death_rho=2.73)
    )
    analysis, sample = run_ascertainment(
        pedigree, censored, seed=3, dropout_rate=0.2, case_ascertainment_ratio=1.0, N_sample=0
    )
    return pedigree, build_cohort(analysis, sample)


def test_cohort_export_round_trips_the_analysis_pedigree(tmp_path, rep) -> None:
    pedigree, cohort = rep
    write_plink(pedigree, tmp_path / "rep", cohort)
    analysis = selected_views(pedigree, cohort).pedigree.sort("id")

    fam = _read_fam(tmp_path / "rep.fam")
    assert set(fam["FID"]) == {"1"}
    back = fam.select(
        _simace_id("IID").alias("id"), _simace_id("MAT").alias("mother"), _simace_id("PAT").alias("father")
    )
    assert back.equals(analysis.select("id", "mother", "father"))
    recorded = analysis.select("id", "mother").join(pedigree.select("id", "mother"), on="id", suffix="_recorded")
    assert recorded.filter((pl.col("mother") == -1) & (pl.col("mother_recorded") >= 0)).height > 0, "no severed link"

    pheno = pl.read_csv(tmp_path / "rep.pheno", separator="\t", null_values=["NA", ""])
    assert pheno.columns == ["FID", "IID", "affected1", "affected2"]
    outcomes = analysis.select("id").join(cohort.select("id", "affected1", "affected2"), on="id", how="left")
    for trait in ("affected1", "affected2"):
        want = outcomes[trait].map_elements(lambda a: 2 if a else 1, return_dtype=pl.Int64).fill_null(-9)
        assert pheno[trait].to_list() == want.to_list()
    assert fam["PHENO"].to_list() == pheno["affected1"].to_list()
    assert set(fam["PHENO"]) == {-9, 1, 2}

    twins = pl.read_csv(tmp_path / "rep.twins", separator="\t", null_values=["NA", ""])
    got = set(zip(twins["IID1"], twins["IID2"], strict=True))
    want_pairs = analysis.filter(pl.col("twin") > pl.col("id"))
    assert got == {(f"id_{a}", f"id_{b}") for a, b in zip(want_pairs["id"], want_pairs["twin"], strict=True)}
    assert got


def test_cli_writes_all_three_files(tmp_path, rep) -> None:
    pedigree, cohort = rep
    write_pedigree(pedigree, tmp_path / "pedigree.parquet")
    write_cohort(cohort, tmp_path / "cohort.parquet")

    cli(
        [
            "--pedigree",
            str(tmp_path / "pedigree.parquet"),
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "-o",
            str(tmp_path / "out" / "rep1"),
        ]
    )

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["rep1.fam", "rep1.pheno", "rep1.twins"]
    assert len(_read_fam(tmp_path / "out" / "rep1.fam")) == len(cohort)


def test_cli_refuses_an_output_with_an_extension(tmp_path) -> None:
    with pytest.raises(SystemExit):
        cli(["--pedigree", str(tmp_path / "pedigree.parquet"), "-o", str(tmp_path / "rep1.fam")])
    assert not list(tmp_path.iterdir())
