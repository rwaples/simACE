"""Tests for the PA-FGRS scoring pipeline: extract_relatives, prepare, score_probands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fit_ace.pafgrs.pafgrs import (
    build_sparse_kinship,
    extract_relatives,
    prepare_univariate_scoring,
    score_probands,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scoring_data():
    """Small 3-generation pedigree with phenotype columns for scoring tests.

    N=50 per generation (150 total).  Runs real simulation + threshold to
    get realistic pedigree structure, then synthesises simple CIP curve.
    """
    from sim_ace.phenotyping.threshold import apply_threshold
    from sim_ace.simulation.simulate import run_simulation

    ped = run_simulation(
        seed=99,
        N=50,
        G_ped=3,
        G_sim=3,
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.5,
        C1=0.2,
        A2=0.5,
        C2=0.2,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )
    gen = ped["generation"].values
    rng = np.random.default_rng(99)
    for t in [1, 2]:
        liab = ped[f"liability{t}"].values
        ped[f"affected{t}"] = apply_threshold(liab, gen, 0.10)
        aff = ped[f"affected{t}"].values
        ped[f"t_observed{t}"] = np.where(aff, rng.uniform(20, 60, len(ped)), 80.0)

    cip_ages = np.array([0.0, 80.0])
    cip_values = np.array([0.0, 0.10])

    return ped, cip_ages, cip_values


@pytest.fixture(scope="module")
def kmat(scoring_data):
    """Pre-built kinship matrix."""
    ped, _, _ = scoring_data
    return build_sparse_kinship(
        ped["id"].values,
        ped["mother"].values,
        ped["father"].values,
        ped["twin"].values,
    )


@pytest.fixture(scope="module")
def rel_arrays(scoring_data, kmat):
    """Pre-extracted relative arrays at default threshold (0.0625)."""
    ped, _, _ = scoring_data
    n = len(ped)
    pheno_ped_idx = np.arange(n, dtype=np.int32)
    pheno_lookup_valid = np.ones(n, dtype=bool)
    return extract_relatives(pheno_ped_idx, kmat, 0.0625, pheno_lookup_valid, n)


@pytest.fixture(scope="module")
def scored_trait1(scoring_data, kmat):
    """Cached score_probands result for trait 1."""
    ped, cip_ages, cip_values = scoring_data
    return score_probands(
        ped,
        ped,
        h2=0.5,
        cip_ages=cip_ages,
        cip_values=cip_values,
        lifetime_prevalence=0.10,
        trait_num=1,
        kmat=kmat,
    )


# ---------------------------------------------------------------------------
# TestExtractRelatives
# ---------------------------------------------------------------------------


class TestExtractRelatives:
    def test_correct_shapes(self, scoring_data, rel_arrays):
        ped, _, _ = scoring_data
        n = len(ped)
        rel_starts, rel_flat_idx, rel_flat_kin, rel_counts = rel_arrays
        assert rel_starts.shape == (n + 1,)
        assert rel_counts.shape == (n,)
        assert len(rel_flat_idx) == len(rel_flat_kin)
        assert rel_starts[-1] == len(rel_flat_idx)

    def test_no_self_relatives(self, scoring_data, rel_arrays):
        ped, _, _ = scoring_data
        rel_starts, rel_flat_idx, _, _ = rel_arrays
        for i in range(len(ped)):
            start, end = rel_starts[i], rel_starts[i + 1]
            relatives = rel_flat_idx[start:end]
            assert i not in relatives, f"Individual {i} found in own relatives"

    def test_high_threshold_fewer_relatives(self, scoring_data, kmat, rel_arrays):
        ped, _, _ = scoring_data
        n = len(ped)
        pheno_ped_idx = np.arange(n, dtype=np.int32)
        pheno_lookup_valid = np.ones(n, dtype=bool)

        _, _, _, counts_low = rel_arrays
        _, _, _, counts_high = extract_relatives(pheno_ped_idx, kmat, 0.25, pheno_lookup_valid, n)
        assert counts_high.sum() <= counts_low.sum()


# ---------------------------------------------------------------------------
# TestPrepareUnivariateScoring
# ---------------------------------------------------------------------------


class TestPrepareUnivariateScoring:
    def test_returns_prep_data(self, scoring_data, kmat):
        ped, _, _ = scoring_data
        prep = prepare_univariate_scoring(ped, ped, ndegree=2, kmat=kmat)
        assert hasattr(prep, "pheno_ids")
        assert hasattr(prep, "rel_starts")
        assert hasattr(prep, "rel_counts")
        assert len(prep.pheno_ids) == len(ped)

    def test_prep_without_kmat(self, scoring_data):
        """Without kmat, prepare_univariate_scoring builds it internally."""
        ped, _, _ = scoring_data
        prep = prepare_univariate_scoring(ped, ped, ndegree=2)
        assert len(prep.pheno_ids) == len(ped)


# ---------------------------------------------------------------------------
# TestScoreProbands
# ---------------------------------------------------------------------------


class TestScoreProbands:
    def test_end_to_end(self, scored_trait1):
        assert isinstance(scored_trait1, pd.DataFrame)
        assert "est" in scored_trait1.columns
        assert "var" in scored_trait1.columns
        assert np.all(np.isfinite(scored_trait1["est"].values))
        assert np.all(scored_trait1["var"].values > 0)

    def test_output_columns(self, scored_trait1):
        expected_cols = {"id", "est", "var", "true_A", "affected", "generation", "n_relatives"}
        assert expected_cols.issubset(set(scored_trait1.columns))

    def test_scores_correlate_with_true_A(self, scored_trait1):
        r = np.corrcoef(scored_trait1["est"].values, scored_trait1["true_A"].values)[0, 1]
        assert r > 0, f"Score-true_A correlation should be positive, got {r}"

    def test_trait2(self, scoring_data, kmat):
        ped, cip_ages, cip_values = scoring_data
        result = score_probands(
            ped,
            ped,
            h2=0.5,
            cip_ages=cip_ages,
            cip_values=cip_values,
            lifetime_prevalence=0.10,
            trait_num=2,
            kmat=kmat,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ped)
