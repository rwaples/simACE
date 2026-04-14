"""Unit tests for sim_ace.phenotyping.gaussian_ace."""

import json

import numpy as np
import pandas as pd
import pytest

from sim_ace.phenotyping.gaussian_ace import simulate_gaussian_ace, write_truth_json


def _fake_pedigree(n: int = 1000, n_households: int = 250, seed: int = 0) -> pd.DataFrame:
    """Synthetic pedigree-like DataFrame with A1/C1/E1 matching simulator semantics."""
    rng = np.random.default_rng(seed)
    household_id = rng.integers(0, n_households, size=n, dtype=np.int32)
    a1 = rng.normal(0, np.sqrt(0.5), size=n).astype(np.float32)
    hh_c = rng.normal(0, np.sqrt(0.2), size=n_households).astype(np.float32)
    c1 = hh_c[household_id]
    e1 = rng.normal(0, np.sqrt(0.3), size=n).astype(np.float32)
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int32),
            "household_id": household_id,
            "A1": a1,
            "C1": c1,
            "E1": e1,
        }
    )


class TestSimulateGaussianAce:
    def test_output_columns(self):
        ped = _fake_pedigree(n=100)
        out = simulate_gaussian_ace(ped, var_a=0.5, var_c=0.2, var_e=0.3)
        assert set(out.columns) == {"id", "fid", "iid", "A", "C", "E", "y"}

    def test_length_matches_input(self):
        ped = _fake_pedigree(n=777)
        out = simulate_gaussian_ace(ped)
        assert len(out) == 777

    def test_variance_hits_targets(self):
        ped = _fake_pedigree(n=20_000, n_households=5_000, seed=42)
        out = simulate_gaussian_ace(ped, var_a=0.4, var_c=0.25, var_e=0.35, seed=1)
        # Rescaling is deterministic: empirical var should hit target exactly
        # (up to floating-point noise) because we divide by the sample std.
        assert out["A"].var(ddof=1) == pytest.approx(0.4, rel=1e-6)
        assert out["C"].var(ddof=1) == pytest.approx(0.25, rel=1e-6)
        assert out["E"].var(ddof=1) == pytest.approx(0.35, rel=1e-6)

    def test_y_equals_sum_of_components(self):
        ped = _fake_pedigree(n=500)
        out = simulate_gaussian_ace(ped, var_a=0.6, var_c=0.1, var_e=0.3)
        np.testing.assert_allclose(out["y"].to_numpy(), (out["A"] + out["C"] + out["E"]).to_numpy())

    def test_truth_attrs_attached(self):
        ped = _fake_pedigree(n=100)
        out = simulate_gaussian_ace(ped, var_a=0.6, var_c=0.2, var_e=0.2)
        truth = out.attrs["truth"]
        assert truth["var_a"] == 0.6
        assert truth["var_c"] == 0.2
        assert truth["var_e"] == 0.2
        assert truth["h2"] == pytest.approx(0.6)
        assert truth["c2"] == pytest.approx(0.2)

    def test_zero_variance_component_gives_zeros(self):
        ped = _fake_pedigree(n=300)
        out = simulate_gaussian_ace(ped, var_a=0.7, var_c=0.0, var_e=0.3)
        assert np.all(out["C"].to_numpy() == 0.0)
        assert out["y"].var(ddof=1) == pytest.approx(1.0, rel=5e-2)

    def test_missing_column_raises(self):
        ped = _fake_pedigree(n=100).drop(columns=["C1"])
        with pytest.raises(ValueError, match="C1"):
            simulate_gaussian_ace(ped)

    def test_negative_variance_raises(self):
        ped = _fake_pedigree(n=100)
        with pytest.raises(ValueError, match="non-negative"):
            simulate_gaussian_ace(ped, var_a=-0.1)

    def test_zero_column_variance_raises(self):
        ped = _fake_pedigree(n=100)
        ped["A1"] = 0.0
        with pytest.raises(ValueError, match="zero variance"):
            simulate_gaussian_ace(ped)

    def test_mz_twin_a_identity_preserved(self):
        """Rescaling A1 by a scalar preserves identical A between twins."""
        ped = _fake_pedigree(n=100)
        ped.loc[0, "A1"] = ped.loc[1, "A1"] = 0.42
        out = simulate_gaussian_ace(ped, var_a=0.5, var_c=0.2, var_e=0.3)
        assert out.loc[0, "A"] == out.loc[1, "A"]

    def test_fresh_e_is_seed_deterministic(self):
        ped = _fake_pedigree(n=200)
        out1 = simulate_gaussian_ace(ped, seed=7, fresh_e=True)
        out2 = simulate_gaussian_ace(ped, seed=7, fresh_e=True)
        np.testing.assert_array_equal(out1["E"].to_numpy(), out2["E"].to_numpy())

    def test_fresh_e_differs_from_e1_column(self):
        ped = _fake_pedigree(n=200)
        base = simulate_gaussian_ace(ped, seed=7, fresh_e=False)
        fresh = simulate_gaussian_ace(ped, seed=7, fresh_e=True)
        assert not np.allclose(base["E"].to_numpy(), fresh["E"].to_numpy())

    def test_household_identity_preserved_in_c(self):
        """Individuals in the same household must have identical C."""
        ped = _fake_pedigree(n=500, n_households=100, seed=3)
        out = simulate_gaussian_ace(ped, var_a=0.3, var_c=0.4, var_e=0.3)
        df = out.copy()
        df["household_id"] = ped["household_id"].to_numpy()
        per_hh_std = df.groupby("household_id")["C"].std(ddof=0)
        assert per_hh_std.max() < 1e-10


class TestWriteTruthJson:
    def test_roundtrip(self, tmp_path):
        truth = {"var_a": 0.5, "var_c": 0.2, "var_e": 0.3, "h2": 0.5}
        path = write_truth_json(truth, tmp_path / "truth.json")
        loaded = json.loads(path.read_text())
        assert loaded == truth

    def test_creates_parent_dir(self, tmp_path):
        truth = {"x": 1}
        path = write_truth_json(truth, tmp_path / "nested" / "dir" / "t.json")
        assert path.exists()
