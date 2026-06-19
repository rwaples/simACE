"""Unit tests for the unified ascertainment stage (run_ascertainment)."""

import numpy as np
import pandas as pd
import pytest

from simace.ascertainment import run_ascertainment
from simace.core.schema import CENSORED
from simace.simulation.simulate import run_simulation
from tests.conftest import schema_pad


@pytest.fixture
def small_sim_pedigree():
    """A small pedigree (N=200, G_ped=4) suitable for ancestor-closure tests."""
    return run_simulation(
        seed=42,
        N=200,
        G_ped=4,
        G_sim=4,
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


def _make_trait(pedigree: pd.DataFrame, g_pheno: int, case_rate: float, seed: int) -> pd.DataFrame:
    """Build a synthetic trait DataFrame for the trailing g_pheno generations."""
    rng = np.random.default_rng(seed)
    max_gen = int(pedigree["generation"].max())
    min_gen = max_gen - g_pheno + 1
    phenotyped = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)
    n = len(phenotyped)
    df = pd.DataFrame(
        {
            "id": phenotyped["id"].to_numpy(),
            "generation": phenotyped["generation"].to_numpy(),
            "sex": phenotyped["sex"].to_numpy(),
            "liability1": rng.standard_normal(n),
            "liability2": rng.standard_normal(n),
            "t_observed1": rng.uniform(10, 80, n),
            "t_observed2": rng.uniform(10, 80, n),
            "affected1": rng.random(n) < case_rate,
            "affected2": rng.random(n) < case_rate,
        }
    )
    return schema_pad(df, CENSORED)


@pytest.fixture
def trait_data(small_sim_pedigree):
    """Synthetic per-individual trait for G_pheno=2 trailing generations."""
    return _make_trait(small_sim_pedigree, g_pheno=2, case_rate=0.10, seed=11)


# ---------------------------------------------------------------------------
# Pass-through tests
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_no_dropout_no_sampling(self, small_sim_pedigree, trait_data):
        """dropout_rate=0, N_sample=0 → trait unchanged; pedigree narrows to ancestor closure."""
        ped_out, trait_out = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.0,
            N_sample=0,
            seed=42,
        )
        # All trait rows preserved.
        assert len(trait_out) == len(trait_data)
        # Pedigree is the ancestor closure — should include every phenotyped id plus their ancestors.
        assert set(trait_data["id"]).issubset(set(ped_out["id"]))


# ---------------------------------------------------------------------------
# Dropout semantics (catches v1 weight-multiplier bug)
# ---------------------------------------------------------------------------


class TestDropout:
    def test_dropout_reduces_trait_when_nsample_zero(self, small_sim_pedigree, trait_data):
        """dropout_rate > 0, N_sample = 0 → trait+pedigree row counts strictly drop."""
        _, trait_out = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.5,
            N_sample=0,
            seed=42,
        )
        # Trait drops because half the pedigree (incl. phenotyped IDs) is removed.
        assert len(trait_out) < len(trait_data)

    def test_dropout_affects_pool_with_nsample(self, small_sim_pedigree, trait_data):
        """Deterministic clamp test: with extreme dropout, output is clamped to trait survivors.

        Picks dropout_rate high enough that *trait-level* survivors < N_sample.
        Asserts (a) output length equals trait-level survivor count (not N_sample),
        and (b) no output id is in the dropped set.
        Catches the v1 weight-multiplier bug deterministically (without that bug,
        dropout would silently cancel under a fixed-size weighted draw).
        """
        # 95% dropout, large N_sample → ascertainment must clamp to ~5% trait pool.
        rate = 0.95
        n_sample = len(trait_data)  # more than trait survivors with dropout=0.95

        _, trait_out = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=rate,
            N_sample=n_sample,
            seed=42,
        )

        # Recompute expected trait survivors deterministically.
        rng = np.random.default_rng(42)
        n_total = len(small_sim_pedigree)
        n_drop = round(n_total * rate)
        drop_idx = rng.choice(n_total, n_drop, replace=False)
        keep_mask = np.ones(n_total, dtype=bool)
        keep_mask[drop_idx] = False
        survivor_ids = small_sim_pedigree.loc[keep_mask, "id"].to_numpy()
        dropped_ids = small_sim_pedigree.loc[~keep_mask, "id"].to_numpy()
        expected_trait_survivors = int(trait_data["id"].isin(survivor_ids).sum())

        # (a) clamp to trait pool, not N_sample.
        assert len(trait_out) == expected_trait_survivors, (
            f"expected clamp to {expected_trait_survivors} trait survivors, got {len(trait_out)}"
        )
        # (b) no output id is in the dropped set.
        assert set(trait_out["id"]).isdisjoint(set(dropped_ids))


# ---------------------------------------------------------------------------
# Ancestor-closure invariant
# ---------------------------------------------------------------------------


class TestAncestorClosure:
    def test_every_sampled_id_has_ancestors_in_pedigree(self, small_sim_pedigree, trait_data):
        """Walking parent edges from any sampled id never hits a missing id."""
        ped_out, trait_out = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.2,
            N_sample=20,
            seed=42,
        )
        ped_ids = set(ped_out["id"].to_numpy().tolist())
        parents = dict(
            zip(
                ped_out["id"].to_numpy(),
                zip(ped_out["mother"].to_numpy(), ped_out["father"].to_numpy(), strict=True),
                strict=True,
            )
        )
        for sid in trait_out["id"]:
            cur = [int(sid)]
            while cur:
                nxt = []
                for ind in cur:
                    if ind not in parents:
                        continue
                    for p in parents[ind]:
                        if p < 0:
                            continue
                        assert p in ped_ids, f"ancestor {p} of sampled id {sid} missing from pedigree"
                        nxt.append(int(p))
                cur = nxt

    def test_no_dangling_links(self, small_sim_pedigree, trait_data):
        """All mother/father/twin references in output point to ids that exist in output or are -1."""
        ped_out, _ = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.3,
            N_sample=30,
            seed=42,
        )
        ped_ids = set(ped_out["id"].to_numpy().tolist())
        for col in ("mother", "father", "twin"):
            vals = ped_out[col].to_numpy()
            for v in vals:
                if v >= 0:
                    assert int(v) in ped_ids, f"dangling {col} ref to id {v} not in pedigree"


# ---------------------------------------------------------------------------
# Case-ascertainment enrichment
# ---------------------------------------------------------------------------


class TestCaseAscertainment:
    def test_enrichment_increases_case_fraction(self, small_sim_pedigree, trait_data):
        """case_ascertainment_ratio > 1 raises observed case fraction vs uniform."""
        n_sample = 40
        _, trait_uniform = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.0,
            case_ascertainment_ratio=1.0,
            N_sample=n_sample,
            seed=42,
        )
        _, trait_enriched = run_ascertainment(
            small_sim_pedigree,
            trait_data,
            dropout_rate=0.0,
            case_ascertainment_ratio=10.0,
            N_sample=n_sample,
            seed=42,
        )
        # With 10x case weighting and a low base case rate, enrichment should clearly raise the case fraction.
        case_frac_uniform = trait_uniform["affected1"].mean()
        case_frac_enriched = trait_enriched["affected1"].mean()
        assert case_frac_enriched > case_frac_uniform


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_negative_dropout_rate(self, small_sim_pedigree, trait_data):
        with pytest.raises(ValueError, match="dropout_rate"):
            run_ascertainment(
                small_sim_pedigree,
                trait_data,
                dropout_rate=-0.1,
                N_sample=0,
                seed=42,
            )

    def test_rejects_full_dropout(self, small_sim_pedigree, trait_data):
        with pytest.raises(ValueError, match="dropout_rate"):
            run_ascertainment(
                small_sim_pedigree,
                trait_data,
                dropout_rate=1.0,
                N_sample=0,
                seed=42,
            )

    def test_rejects_negative_ratio(self, small_sim_pedigree, trait_data):
        with pytest.raises(ValueError, match="case_ascertainment_ratio"):
            run_ascertainment(
                small_sim_pedigree,
                trait_data,
                dropout_rate=0.0,
                case_ascertainment_ratio=-0.5,
                N_sample=10,
                seed=42,
            )
