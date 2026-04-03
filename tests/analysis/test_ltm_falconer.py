"""Tests for sim_ace.analysis.ltm_falconer — Falconer h2 from binary phenotype."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim_ace.analysis.ltm_falconer import (
    KIND_TO_PAIRS,
    KINSHIP,
    _empty_result,
    compute_ltm_falconer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ltm_df():
    """Simulated pedigree with affected columns for Falconer h2 tests."""
    from sim_ace.phenotyping.threshold import apply_threshold
    from sim_ace.simulation.simulate import run_simulation

    ped = run_simulation(
        seed=42,
        N=1000,
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
    for t in [1, 2]:
        liab = ped[f"liability{t}"].values
        ped[f"affected{t}"] = apply_threshold(liab, gen, 0.10)
    return ped


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_kinds_have_kinship(self):
        """Every key in KIND_TO_PAIRS must also exist in KINSHIP."""
        for kind in KIND_TO_PAIRS:
            assert kind in KINSHIP, f"Missing kinship for {kind}"

    def test_kinship_values(self):
        assert KINSHIP["PO"] == pytest.approx(0.25)
        assert KINSHIP["FS"] == pytest.approx(0.25)
        assert KINSHIP["HS"] == pytest.approx(0.125)
        assert KINSHIP["mHS"] == pytest.approx(0.125)
        assert KINSHIP["pHS"] == pytest.approx(0.125)
        assert KINSHIP["1C"] == pytest.approx(0.0625)
        assert KINSHIP["Av"] == pytest.approx(0.125)
        assert KINSHIP["1G"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# TestComputeLtmFalconer
# ---------------------------------------------------------------------------


class TestComputeLtmFalconer:
    def test_fs_returns_result(self, ltm_df):
        result = compute_ltm_falconer(ltm_df, kinds=["FS"])
        assert "FS" in result
        entry = result["FS"]
        assert set(entry.keys()) == {"r_tetrachoric", "se_r", "h2_falconer", "se_h2", "n_pairs", "kinship"}
        assert entry["n_pairs"] > 0
        assert entry["kinship"] == pytest.approx(0.25)

    def test_po_returns_result(self, ltm_df):
        """PO merges MO+FO pairs."""
        result = compute_ltm_falconer(ltm_df, kinds=["PO"])
        assert "PO" in result
        assert result["PO"]["n_pairs"] > 0
        assert result["PO"]["kinship"] == pytest.approx(0.25)

    def test_hs_returns_result(self, ltm_df):
        """HS merges MHS+PHS pairs."""
        result = compute_ltm_falconer(ltm_df, kinds=["HS"])
        assert "HS" in result

    def test_multiple_kinds(self, ltm_df):
        result = compute_ltm_falconer(ltm_df, kinds=["PO", "FS", "HS"])
        assert set(result.keys()) == {"PO", "FS", "HS"}

    def test_unknown_kind_returns_empty(self, ltm_df):
        result = compute_ltm_falconer(ltm_df, kinds=["UNKNOWN"])
        assert "UNKNOWN" in result
        assert result["UNKNOWN"]["r_tetrachoric"] is None
        assert result["UNKNOWN"]["n_pairs"] == 0

    def test_too_few_pairs(self):
        """Tiny pedigree where all kinds have < MIN_PAIRS → None values."""
        df = pd.DataFrame(
            {
                "id": np.arange(10),
                "mother": np.concatenate([np.full(5, -1), np.array([0, 0, 2, 2, 0])]),
                "father": np.concatenate([np.full(5, -1), np.array([1, 1, 3, 3, 3])]),
                "twin": np.full(10, -1),
                "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                "generation": np.concatenate([np.zeros(5), np.ones(5)]).astype(int),
                "affected1": np.array([True, False, True, False, True, False, True, False, True, False]),
            }
        )
        result = compute_ltm_falconer(df, kinds=["FS"], trait_num=1)
        assert result["FS"]["r_tetrachoric"] is None

    def test_h2_is_positive(self, ltm_df):
        """With A1=0.5, FS h2 should be positive (if enough pairs)."""
        result = compute_ltm_falconer(ltm_df, kinds=["FS"])
        h2 = result["FS"]["h2_falconer"]
        if h2 is not None:
            assert h2 > 0

    def test_trait2(self, ltm_df):
        """trait_num=2 should use affected2."""
        result = compute_ltm_falconer(ltm_df, kinds=["FS"], trait_num=2)
        assert "FS" in result
        assert result["FS"]["n_pairs"] > 0


# ---------------------------------------------------------------------------
# TestEmptyResult
# ---------------------------------------------------------------------------


class TestEmptyResult:
    def test_structure(self):
        entry = _empty_result("FS")
        assert entry["r_tetrachoric"] is None
        assert entry["se_r"] is None
        assert entry["h2_falconer"] is None
        assert entry["se_h2"] is None
        assert entry["n_pairs"] == 0
        assert entry["kinship"] == pytest.approx(0.25)

    def test_unknown_kind_kinship_zero(self):
        entry = _empty_result("NONEXISTENT")
        assert entry["kinship"] == pytest.approx(0.0)
