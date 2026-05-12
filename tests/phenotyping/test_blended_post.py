"""Smoke tests for simace.phenotyping.blended_post.blended_diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simace.phenotyping.blended_post import (
    DEFAULT_CIP_K,
    DEFAULT_CIP_X0,
    MAX_AGE,
    blended_diagnosis,
)


def _make_phenotype(rng: np.random.Generator, n_per_gen: int = 5000, gens: tuple[int, ...] = (2, 3, 4)) -> pd.DataFrame:
    """Synthesize a minimal phenotype DataFrame with A1/C1/E1 + L1/L2 columns."""
    n_total = n_per_gen * len(gens)
    rows = []
    for g in gens:
        A1 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        E1 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        A2 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        E2 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        L1 = A1 + E1
        L2 = A2 + E2
        for i in range(n_per_gen):
            rows.append({
                "id": len(rows),
                "generation": g,
                "sex": rng.integers(0, 2),
                "A1": A1[i], "C1": 0.0, "E1": E1[i], "liability1": L1[i],
                "A2": A2[i], "C2": 0.0, "E2": E2[i], "liability2": L2[i],
                "death_age": 1e6,
                "affected1": False,
                "t_observed1": MAX_AGE,
                "age_censored1": True,
                "death_censored1": False,
            })
    return pd.DataFrame(rows)


class TestBlendedDiagnosis:
    def test_alpha_zero_recovers_pure_trait1(self):
        """With α=0 everywhere, case fraction should track K_by_gen on L1 alone."""
        rng = np.random.default_rng(0)
        pheno = _make_phenotype(rng)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.0, 4: 0.0},
            K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
        )
        # Per-gen K should be ~5% within sampling tolerance.
        rates = out.groupby("generation")["affected1"].mean()
        for g, rate in rates.items():
            assert 0.035 < rate < 0.065, f"gen {g} case rate {rate:.3f} too far from 5%"

    def test_per_gen_K_targets(self):
        """Per-gen case fractions should match per-gen K_by_gen."""
        rng = np.random.default_rng(1)
        pheno = _make_phenotype(rng, n_per_gen=10000)
        alpha = {2: 0.0, 3: 0.3, 4: 0.55}
        K = {2: 0.03, 3: 0.05, 4: 0.10}
        out = blended_diagnosis(pheno, alpha_by_gen=alpha, K_by_gen=K)
        rates = out.groupby("generation")["affected1"].mean()
        for g in (2, 3, 4):
            target = K[g]
            assert abs(rates[g] - target) < 0.01, (
                f"gen {g}: realized K={rates[g]:.3f} vs target {target}"
            )

    def test_late_onset_cases_right_censored(self):
        """Onset > MAX_AGE should be right-censored: affected1=False, t_observed1=MAX_AGE."""
        rng = np.random.default_rng(2)
        pheno = _make_phenotype(rng, n_per_gen=2000)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.0, 4: 0.0},
            K_by_gen={2: 0.5, 3: 0.5, 4: 0.5},  # Very high K → some onsets exceed MAX_AGE
        )
        assert (out["t_observed1"] <= MAX_AGE).all(), "no t_observed1 > MAX_AGE expected"
        # No NaNs or impossibly young onsets
        assert (out["t_observed1"] > 0).all()
        # age_censored1 is a boolean flag
        assert out["age_censored1"].dtype == bool

    def test_audit_columns_present(self):
        """Output should expose A_blend, C_blend, E_blend, liability_blend."""
        rng = np.random.default_rng(3)
        pheno = _make_phenotype(rng, n_per_gen=200)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.5, 4: 0.5},
            K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
        )
        for col in ("A_blend", "C_blend", "E_blend", "liability_blend"):
            assert col in out.columns

    def test_blend_components_match_formula(self):
        """A_blend = (1-α)·A1 + α·A2 per row, similar for others."""
        rng = np.random.default_rng(4)
        pheno = _make_phenotype(rng, n_per_gen=100)
        alpha = {2: 0.2, 3: 0.5, 4: 0.8}
        K = {2: 0.05, 3: 0.05, 4: 0.05}
        out = blended_diagnosis(pheno, alpha_by_gen=alpha, K_by_gen=K)
        for g, a in alpha.items():
            sub = out[out["generation"] == g]
            expected_A = (1 - a) * sub["A1"] + a * sub["A2"]
            np.testing.assert_allclose(sub["A_blend"], expected_A, atol=1e-5)

    def test_missing_gen_in_alpha_raises(self):
        """alpha_by_gen lacking an observed generation should raise ValueError."""
        rng = np.random.default_rng(5)
        pheno = _make_phenotype(rng, n_per_gen=100, gens=(2, 3, 4))
        with pytest.raises(ValueError, match="alpha_by_gen.*missing generation"):
            blended_diagnosis(
                pheno,
                alpha_by_gen={2: 0.0, 3: 0.0},  # gen 4 missing
                K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
            )

    def test_missing_required_columns_raises(self):
        """phenotype lacking A1 or liability1 should raise."""
        df = pd.DataFrame({"generation": [2, 2], "sex": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            blended_diagnosis(df, alpha_by_gen={2: 0.0}, K_by_gen={2: 0.05})
