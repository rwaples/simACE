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

    # Simple CIP: linear from 0 to 0.10 over [0, 80]
    cip_ages = np.array([0.0, 80.0])
    cip_values = np.array([0.0, 0.10])

    return ped, cip_ages, cip_values


@pytest.fixture(scope="module")
def kmat_and_lookups(scoring_data):
    """Pre-built kinship matrix."""
    ped, _, _ = scoring_data
    return build_sparse_kinship(
        ped["id"].values,
        ped["mother"].values,
        ped["father"].values,
        ped["twin"].values,
    )


# ---------------------------------------------------------------------------
# TestExtractRelatives
# ---------------------------------------------------------------------------


class TestExtractRelatives:
    def test_correct_shapes(self, scoring_data, kmat_and_lookups):
        ped, _, _ = scoring_data
        kmat = kmat_and_lookups
        n_ped = len(ped)
        n_pheno = n_ped
        pheno_ped_idx = np.arange(n_pheno, dtype=np.int32)
        pheno_lookup_valid = np.ones(n_ped, dtype=bool)

        rel_starts, rel_flat_idx, rel_flat_kin, rel_counts = extract_relatives(
            pheno_ped_idx,
            kmat,
            0.0625,
            pheno_lookup_valid,
            n_pheno,
        )
        assert rel_starts.shape == (n_pheno + 1,)
        assert rel_counts.shape == (n_pheno,)
        assert len(rel_flat_idx) == len(rel_flat_kin)
        assert rel_starts[-1] == len(rel_flat_idx)

    def test_no_self_relatives(self, scoring_data, kmat_and_lookups):
        ped, _, _ = scoring_data
        kmat = kmat_and_lookups
        n_ped = len(ped)
        pheno_ped_idx = np.arange(n_ped, dtype=np.int32)
        pheno_lookup_valid = np.ones(n_ped, dtype=bool)

        rel_starts, rel_flat_idx, _, _ = extract_relatives(
            pheno_ped_idx,
            kmat,
            0.0625,
            pheno_lookup_valid,
            n_ped,
        )
        for i in range(n_ped):
            start, end = rel_starts[i], rel_starts[i + 1]
            relatives = rel_flat_idx[start:end]
            assert i not in relatives, f"Individual {i} found in own relatives"

    def test_high_threshold_fewer_relatives(self, scoring_data, kmat_and_lookups):
        ped, _, _ = scoring_data
        kmat = kmat_and_lookups
        n_ped = len(ped)
        pheno_ped_idx = np.arange(n_ped, dtype=np.int32)
        pheno_lookup_valid = np.ones(n_ped, dtype=bool)

        _, _, _, counts_low = extract_relatives(pheno_ped_idx, kmat, 0.0625, pheno_lookup_valid, n_ped)
        _, _, _, counts_high = extract_relatives(pheno_ped_idx, kmat, 0.25, pheno_lookup_valid, n_ped)
        assert counts_high.sum() <= counts_low.sum()


# ---------------------------------------------------------------------------
# TestPrepareUnivariateScoring
# ---------------------------------------------------------------------------


class TestPrepareUnivariateScoring:
    def test_returns_prep_data(self, scoring_data, kmat_and_lookups):
        ped, _, _ = scoring_data
        kmat = kmat_and_lookups
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
    def test_end_to_end(self, scoring_data, kmat_and_lookups):
        ped, cip_ages, cip_values = scoring_data
        kmat = kmat_and_lookups
        result = score_probands(
            ped,
            ped,
            h2=0.5,
            cip_ages=cip_ages,
            cip_values=cip_values,
            lifetime_prevalence=0.10,
            trait_num=1,
            kmat=kmat,
        )
        assert isinstance(result, pd.DataFrame)
        assert "est" in result.columns
        assert "var" in result.columns
        assert np.all(np.isfinite(result["est"].values))
        assert np.all(result["var"].values > 0)

    def test_output_columns(self, scoring_data, kmat_and_lookups):
        ped, cip_ages, cip_values = scoring_data
        kmat = kmat_and_lookups
        result = score_probands(
            ped,
            ped,
            h2=0.5,
            cip_ages=cip_ages,
            cip_values=cip_values,
            lifetime_prevalence=0.10,
            trait_num=1,
            kmat=kmat,
        )
        expected_cols = {"id", "est", "var", "true_A", "affected", "generation", "n_relatives"}
        assert expected_cols.issubset(set(result.columns))

    def test_scores_correlate_with_true_A(self, scoring_data, kmat_and_lookups):
        ped, cip_ages, cip_values = scoring_data
        kmat = kmat_and_lookups
        result = score_probands(
            ped,
            ped,
            h2=0.5,
            cip_ages=cip_ages,
            cip_values=cip_values,
            lifetime_prevalence=0.10,
            trait_num=1,
            kmat=kmat,
        )
        r = np.corrcoef(result["est"].values, result["true_A"].values)[0, 1]
        assert r > 0, f"Score-true_A correlation should be positive, got {r}"

    def test_trait2(self, scoring_data, kmat_and_lookups):
        ped, cip_ages, cip_values = scoring_data
        kmat = kmat_and_lookups
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
