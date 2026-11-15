"""Tests for simace.analysis.stats.runner.build_stats_report and the SampleStats it returns."""

import numpy as np
import pytest
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

from simace.analysis.report_schema import partition_dense
from simace.analysis.sample_stats import SampleStats
from simace.analysis.stats.moments import Stratum
from simace.analysis.stats.runner import PEDIGREE_REPORT_COLUMNS, _build_relationship_context, build_stats_report
from simace.core.relationships import RELATIONSHIP_TYPES
from simace.core.trait_schema import hydrate_trait


@pytest.fixture(scope="module")
def tiny_phenotype():
    """Build a tiny censored phenotype DataFrame via simulate → phenotype → censor."""
    from simace.censoring.censor import run_censor
    from simace.phenotype import run_phenotype
    from simace.simulation.simulate import run_simulation

    sim_params = dict(
        seed=7,
        N=200,
        G_ped=2,
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
    pedigree = run_simulation(**sim_params).pedigree
    phenotype = run_phenotype(
        pedigree,
        G_pheno=2,
        seed=7,
        standardize=True,
        phenotype_model1="frailty",
        phenotype_params1={"distribution": "weibull", "scale": 2160, "rho": 0.8},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 333, "rho": 1.2},
        beta2=1.0,
        beta_sex2=0.0,
    )
    censored = run_censor(
        phenotype,
        pedigree,
        censor_age=80,
        seed=7,
        gen_censoring={},
        death_scale=164,
        death_rho=2.73,
    )
    return pedigree, censored


@pytest.fixture(scope="module")
def sample(tiny_phenotype):
    pedigree, phenotype = tiny_phenotype
    return hydrate_trait(phenotype, pedigree, kind="censored", columns=PEDIGREE_REPORT_COLUMNS), pedigree


@pytest.fixture(scope="module")
def stats(sample):
    df, pedigree = sample
    return build_stats_report(df, pedigree, 80.0, max_degree=2)


@pytest.fixture(scope="module")
def stats_with_windows(sample):
    df, pedigree = sample
    return build_stats_report(df, pedigree, 80.0, gen_censoring={0: [80, 80], 1: [0, 80]}, max_degree=2)


def _report(stats: SampleStats) -> dict:
    """The parts of a v2 report that SampleStats writes, as assemble_report writes them."""
    observed, dense = partition_dense(
        {"analysis_sample": stats.analysis_sample.to_report(), "analysis_pedigree": stats.analysis_pedigree.to_report()}
    )
    size = {"n_individuals": stats.sample_size.n_individuals, "n_generations": stats.sample_size.n_generations}
    ped = {"n_individuals": stats.pedigree_size.n_individuals, "n_generations": stats.pedigree_size.n_generations}
    report = {
        "scopes": {"analysis_sample": size, "analysis_pedigree": ped},
        "observed": observed,
        "estimators": {"heritability": {"observed_scale": stats.observed_h2_estimators}},
    }
    return {"report": report, "payload": dense}


class TestBuildStatsReport:
    def test_scope_sizes(self, stats, sample):
        df, pedigree = sample
        assert stats.sample_size.n_individuals == len(df)
        assert stats.pedigree_size.n_individuals == len(pedigree)
        assert stats.sample_size.n_generations == len(np.unique(df["generation"].to_numpy()))

    def test_pair_counts_carry_every_code_and_none_for_uncomputed(self, stats):
        # The fixture ran at max_degree=2, so every deeper code is reported as
        # "not computed" (None) rather than as a zero count.
        for counts in (
            stats.analysis_sample.relationship_pair_counts,
            stats.analysis_pedigree.relationship_pair_counts,
        ):
            assert set(counts) == set(RELATIONSHIPS)
            computed = {code for code, value in counts.items() if value is not None}
            assert computed == {code for code, cat in RELATIONSHIPS.items() if cat.degree <= 2}
            assert all(isinstance(counts[code], int) for code in computed)

    def test_censoring_tables_only_with_windows(self, stats, stats_with_windows):
        for name in ("censoring_windows", "censoring_confusion", "censoring_cascade"):
            assert getattr(stats.analysis_sample, name) is None
            assert name not in stats.analysis_sample.to_report()
            assert getattr(stats_with_windows.analysis_sample, name) is not None
            assert name in stats_with_windows.analysis_sample.to_report()

    def test_observed_scale_estimators(self, stats):
        assert set(stats.observed_h2_estimators) == {"trait1", "trait2"}

    @pytest.mark.parametrize("windows", [False, True], ids=["no_windows", "windows"])
    @pytest.mark.parametrize("with_payload", [False, True], ids=["scalars", "payload"])
    def test_report_round_trip(self, stats, stats_with_windows, windows, with_payload):
        """Reading a written report back writes the same report (the serialized form, ADR 0008)."""
        written = _report(stats_with_windows if windows else stats)
        payload = written["payload"] if with_payload else None
        again = _report(SampleStats.from_report(written["report"], payload))
        assert again["report"] == written["report"]
        assert again["payload"] == (written["payload"] if with_payload else {})


class TestRelationshipContext:
    """Counts and the correlation moments come from one moments pass."""

    @pytest.mark.parametrize("max_degree", [1, 3, 5])
    @pytest.mark.parametrize("subsample", [False, True], ids=["same_ids", "view"])
    def test_counts_equal_the_pair_block_lengths(self, tiny_phenotype, subsample, max_degree):
        pedigree, phenotype = tiny_phenotype
        df = hydrate_trait(phenotype, pedigree, kind="censored")
        if subsample:
            df = df[::3]
        context = _build_relationship_context(df, pedigree, max_degree=max_degree)
        graph = PedigreeGraph.from_frame(pedigree)
        source = graph.view(ids=df["id"].to_numpy()) if subsample else graph
        counted = source.relationship_pairs(max_degree=max_degree)
        assert context.counts == {code: len(b) if b.requested else None for code, b in counted.items()}
        assert context.full_counts == dict(graph.relationship_counts(max_degree=max_degree))
        reference = counted if max_degree >= 3 else source.relationship_pairs(max_degree=3)
        stratum = Stratum(context.moments)
        assert stratum.categories == tuple(code for code in RELATIONSHIPS if code in RELATIONSHIP_TYPES)
        for code in RELATIONSHIP_TYPES:
            assert int(stratum.n_pairs[stratum.index(code)]) == len(reference[code])
        liability = df["liability1"].to_numpy()
        for code in RELATIONSHIP_TYPES:
            block = reference[code]
            if len(block) < 2:
                continue
            # liability_r reports 0.0 where a side is constant (one person in every pair).
            with np.errstate(invalid="ignore", divide="ignore"):
                want = np.nan_to_num(np.corrcoef(liability[block.first_rows], liability[block.second_rows])[0, 1])
            got = stratum.liability_r(1)[stratum.index(code)]
            assert got == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize(
        ("max_degree", "error"), [(-1, ValueError), (6, ValueError), (True, TypeError), (3.0, TypeError)]
    )
    def test_rejects_max_degree_outside_the_registry(self, tiny_phenotype, max_degree, error):
        pedigree, phenotype = tiny_phenotype
        df = hydrate_trait(phenotype, pedigree, kind="censored")
        with pytest.raises(error):
            _build_relationship_context(df, pedigree, max_degree=max_degree)
