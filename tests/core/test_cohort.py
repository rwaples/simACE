"""The cohort contract: invariants, routes in and out, the rebuilt views, and the layout marker."""

from __future__ import annotations

import numpy as np
import polars as pl
import polars.testing
import pytest

from simace.ascertainment.runner import _apply_dropout, run_ascertainment
from simace.censoring.censor import run_censor
from simace.core.cohort import (
    COHORT_COLUMNS,
    build_cohort,
    check_cohort,
    read_cohort,
    read_pedigree,
    selected_views,
    write_cohort,
    write_pedigree,
)
from simace.core.parquet import normalize_for_parquet, save_parquet
from simace.phenotype import run_phenotype
from simace.simulation.simulate import run_simulation

G_PHENO = 2
ASCERTAIN = {"dropout_rate": 0.2, "case_ascertainment_ratio": 3.0, "N_sample": 150}
ASCERTAIN_SEED = 11


@pytest.fixture(scope="module")
def chain() -> tuple[pl.DataFrame, pl.DataFrame]:
    """A recorded pedigree and its censored phenotyped rows, as the stages read them from disk.

    Every 7th ``t1`` and 11th ``t2`` is nulled before censoring (ADR 0019).
    """
    pedigree = normalize_for_parquet(
        run_simulation(
            seed=7,
            N=300,
            G_ped=4,
            G_sim=4,
            mating_lambda=0.5,
            p_mztwin=0.05,
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
    )
    raw = normalize_for_parquet(
        run_phenotype(
            pedigree,
            G_pheno=G_PHENO,
            seed=7,
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
    row = np.arange(len(raw))
    t1, t2 = raw["t1"].to_numpy().copy(), raw["t2"].to_numpy().copy()
    t1[row % 7 == 0] = np.nan
    t2[row % 11 == 0] = np.nan
    raw = raw.with_columns(pl.Series("t1", t1).fill_nan(None), pl.Series("t2", t2).fill_nan(None))
    censored = normalize_for_parquet(
        run_censor(raw, pedigree, censor_age=80, seed=7, gen_censoring={}, death_scale=164, death_rho=2.73)
    )
    return pedigree, censored


@pytest.fixture(scope="module")
def ascertained(chain) -> tuple[pl.DataFrame, pl.DataFrame]:
    pedigree, censored = chain
    return run_ascertainment(pedigree, censored, seed=ASCERTAIN_SEED, **ASCERTAIN)


@pytest.fixture(scope="module")
def cohort(ascertained) -> pl.DataFrame:
    return build_cohort(*ascertained)


def _assert_same(got: pl.DataFrame, want: pl.DataFrame) -> None:
    polars.testing.assert_frame_equal(got, want, check_row_order=True, check_dtypes=True, check_exact=True)


def test_selected_views_rebuild_the_ascertainment_outputs(chain, ascertained, cohort) -> None:
    pedigree, _ = chain
    ped_out, trait_out = ascertained
    assert trait_out["t1"].null_count() > 0
    assert trait_out["t2"].null_count() > 0
    views = selected_views(pedigree, cohort)
    _assert_same(views.pedigree, ped_out)
    _assert_same(views.trait, trait_out)


def test_selected_views_survive_a_parquet_round_trip(tmp_path, chain, ascertained, cohort) -> None:
    pedigree, _ = chain
    ped_out, trait_out = ascertained
    write_pedigree(pedigree, tmp_path / "pedigree.parquet")
    write_cohort(cohort, tmp_path / "cohort.parquet")
    views = selected_views(read_pedigree(tmp_path / "pedigree.parquet"), read_cohort(tmp_path / "cohort.parquet"))
    _assert_same(views.pedigree, ped_out)
    _assert_same(views.trait, trait_out)


def test_selected_views_take_any_pedigree_column_subset_with_id(chain, ascertained, cohort) -> None:
    pedigree, _ = chain
    ped_out, _ = ascertained
    columns = ["id", "mother", "father", "twin", "sex", "generation", "liability1", "liability2"]
    _assert_same(selected_views(pedigree.select(columns), cohort).pedigree, ped_out.select(columns))
    _assert_same(selected_views(pedigree.select("id", "A1"), cohort).pedigree, ped_out.select("id", "A1"))


def test_the_cohort_holds_the_analysis_pedigree_in_its_order(ascertained, cohort) -> None:
    ped_out, _ = ascertained
    assert tuple(cohort.columns) == COHORT_COLUMNS
    assert cohort["id"].equals(ped_out["id"])


def test_null_raw_onsets_on_sample_rows_are_accepted(chain, cohort) -> None:
    pedigree, _ = chain
    sample = cohort.filter(pl.col("affected1").is_not_null())
    assert sample["t1"].null_count() > 0
    assert sample["t2"].null_count() > 0
    check_cohort(cohort, pedigree["id"].to_numpy())


def _set_on_first(cohort: pl.DataFrame, where: pl.Expr, column: str, value) -> pl.DataFrame:
    first = cohort.with_row_index("_row").filter(where)["_row"][0]
    return cohort.with_columns(
        pl.when(pl.int_range(pl.len()) == first).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )


_SAMPLE = pl.col("affected1").is_not_null()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.drop("t2"), r"cohort columns are \["),
        (lambda c: pl.concat([c, c.tail(1)]), r"cohort ids are not unique"),
        (lambda c: c.with_columns(pl.col("id") + 10_000_000), r"cohort ids missing from the pedigree"),
        (lambda c: _set_on_first(c, _SAMPLE, "death_age", None), r"sample rows .* have nulls in \['death_age'\]"),
        (lambda c: _set_on_first(c, _SAMPLE, "affected2", None), r"have nulls in \['affected2'\]"),
        (
            lambda c: _set_on_first(c, ~_SAMPLE, "t_observed1", 50.0),
            r"outside the sample .* values in \['t_observed1'\]",
        ),
        (lambda c: _set_on_first(c, ~_SAMPLE, "t1", 50.0), r"values in \['t1'\]"),
        (lambda c: pl.concat([c.slice(1, 1), c.slice(0, 1), c.slice(2)]), r"not in pedigree order"),
    ],
    ids=[
        "columns",
        "duplicate",
        "unknown-id",
        "sample-null",
        "sample-null-trait2",
        "non-sample-set",
        "non-sample-onset",
        "order",
    ],
)
def test_check_cohort_rejects_each_broken_invariant(chain, cohort, mutate, message) -> None:
    pedigree, _ = chain
    with pytest.raises(ValueError, match=message):
        check_cohort(mutate(cohort), pedigree["id"].to_numpy())


def test_build_cohort_rejects_a_sample_out_of_pedigree_order(ascertained) -> None:
    ped_out, trait_out = ascertained
    with pytest.raises(ValueError, match="sample trait ids are not the analysis pedigree's ids in its order"):
        build_cohort(ped_out, trait_out.reverse())


def test_build_cohort_rejects_a_sample_id_outside_the_analysis_pedigree(ascertained) -> None:
    ped_out, trait_out = ascertained
    with pytest.raises(ValueError, match="sample trait ids are not the analysis pedigree's ids in its order"):
        build_cohort(ped_out.filter(pl.col("id") != trait_out["id"][0]), trait_out)


def test_routes_in_and_out_of_the_cohort(chain, cohort) -> None:
    pedigree, _ = chain
    min_gen = int(pedigree["generation"].max()) - G_PHENO + 1

    rows = cohort.join(pedigree.select("id", "generation"), on="id", how="left", maintain_order="left")
    drawn = rows["affected1"].is_not_null()
    in_window = rows["generation"] >= min_gen
    routes = {
        "A drawn": int(drawn.sum()),
        "B ancestor, never phenotyped": int((~drawn & ~in_window).sum()),
        "C ancestor, phenotyped, not drawn": int((~drawn & in_window).sum()),
    }
    assert routes["A drawn"] == ASCERTAIN["N_sample"]
    assert bool(in_window.filter(drawn).all())
    assert min(routes.values()) >= 1, routes

    kept = _apply_dropout(pedigree, ASCERTAIN["dropout_rate"], np.random.default_rng(ASCERTAIN_SEED))["id"]
    out = pedigree.filter(~pl.col("id").is_in(cohort["id"].implode()))
    dropped = out.filter(~pl.col("id").is_in(kept.implode()))
    survivors = out.filter(pl.col("id").is_in(kept.implode()))
    exits = {
        "dropped": len(dropped),
        "phenotyped, neither drawn nor an ancestor": len(survivors.filter(pl.col("generation") >= min_gen)),
        "unphenotyped, not an ancestor": len(survivors.filter(pl.col("generation") < min_gen)),
    }
    assert exits["dropped"] == len(pedigree) - len(kept)
    assert min(exits.values()) >= 1, exits

    recorded = pedigree.filter(pl.col("id").is_in(cohort["id"].implode()))
    analysis = selected_views(pedigree, cohort).pedigree
    for parent in ("mother", "father"):
        lost = recorded[parent].is_in(dropped["id"].implode())
        assert bool(lost.any()), f"no cohort member lost a dropped {parent}"
        assert bool((analysis[parent].filter(lost) == -1).all())
        assert analysis[parent].filter(~lost).equals(recorded[parent].filter(~lost))


def test_writers_mark_the_layout_and_readers_check_it(tmp_path, chain, cohort) -> None:
    pedigree, _ = chain
    write_pedigree(pedigree, tmp_path / "pedigree.parquet")
    write_cohort(cohort, tmp_path / "cohort.parquet")
    for name in ("pedigree.parquet", "cohort.parquet"):
        assert pl.read_parquet_metadata(tmp_path / name)["simace_layout"] == "2"
    _assert_same(read_pedigree(tmp_path / "pedigree.parquet"), pedigree)
    _assert_same(read_pedigree(tmp_path / "pedigree.parquet", columns=["id", "sex"]), pedigree.select("id", "sex"))
    _assert_same(read_cohort(tmp_path / "cohort.parquet"), cohort)

    save_parquet(pedigree, tmp_path / "old_pedigree.parquet")
    save_parquet(cohort, tmp_path / "old_cohort.parquet")
    with pytest.raises(ValueError, match=r"old_pedigree.parquet lacks .* predates results layout 2 \(ADR 0021\)"):
        read_pedigree(tmp_path / "old_pedigree.parquet")
    with pytest.raises(ValueError, match=r"old_cohort.parquet lacks .* predates results layout 2 \(ADR 0021\)"):
        read_cohort(tmp_path / "old_cohort.parquet")
